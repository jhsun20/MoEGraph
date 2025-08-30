import torch
import torch.nn as nn
import torch.nn.functional as F

from entmax import entmax_bisect
from torch_geometric.nn import global_mean_pool

from models.expert import Experts
from models.gnn_models import GINEncoderWithEdgeWeight


class MoE(nn.Module):
    """
    Mixture-of-Experts wrapper:
      - owns a shared Experts module (K experts inside);
      - owns a gate over the RAW input graph;
      - aggregates per-expert losses using (avg) gate weights.
    """
    def __init__(self, config, dataset_info):
        super().__init__()
        self.verbose         = config['experiment']['debug']['verbose']
        self.num_experts     = config['model']['num_experts']
        self.aggregation     = config['model']['aggregation']     # expect 'weighted_mean'
        self.train_after     = config['gate']['train_after']
        self.weight_ce       = config['model']['weight_ce']
        self.weight_reg      = config['model']['weight_reg']
        self.weight_la      = config['model']['weight_la']
        self.weight_ea      = config['model']['weight_ea']
        self.weight_str      = config['model']['weight_str']
        self.weight_div      = config['model']['weight_div']
        self.weight_load     = config['model']['weight_load']
        self.topk_train_k    = int(config['gate'].get('topk_train_k', 2))    # 0 => disabled
        self.topk_after      = int(config['gate'].get('topk_after', self.train_after + 1))

        # Shared expert block (contains all experts)
        self.shared = Experts(config, dataset_info)

        # Gating network on raw graph
        self.num_features   = dataset_info['num_features']
        self.dataset_name   = dataset_info['dataset_name']
        self.num_classes    = dataset_info['num_classes']
        if self.num_classes == 1 and dataset_info['metric'] == "Accuracy":
            self.num_classes = 2
        self.metric         = dataset_info['metric']

        gate_hidden    = config['gate']['hidden_dim']
        gate_depth     = config['gate']['depth']
        dropout        = config['model']['dropout']
        self.gate_enc  = GINEncoderWithEdgeWeight(self.num_features, gate_hidden, gate_depth, dropout, train_eps=True, global_pooling='mean')
        self.gate_mlp  = nn.Sequential(nn.Linear(gate_hidden, gate_hidden),
                                       nn.ReLU(),
                                       nn.Linear(gate_hidden, self.num_experts))
        self.entmax_alpha = float(config['gate']['entmax_alpha'])
        gcfg = config.get('gate', {})
        self.gate_tau_oracle = float(gcfg.get('tau_oracle', 0.75))     # softness for teacher
        self.gate_temperature = float(gcfg.get('temperature', 1.0))    # scales gate_scores
        self.weight_gate      = float(gcfg.get('weight_gate', 1.0))    # total gate loss weight
        self.weight_gate_oracle = float(gcfg.get('weight_oracle', 1.0))# teacher KL weight
        self.gate_teacher_w_ea = float(gcfg.get('teacher_w_ea', 1.0))  # + means "more invariant" is better
        self.gate_teacher_w_la = float(gcfg.get('teacher_w_la', 1.0))  # + means "less leakage" is better
        self.current_epoch = 0

        if self.verbose:
            print(f"[MoE] K={self.num_experts}, aggregation={self.aggregation}, gate=raw-graph")

    # ------------- Public API -------------
    def set_epoch(self, epoch: int):
        """Forward epoch to experts and update gate schedule (warm-up)."""
        self.current_epoch = int(epoch)
        self.shared.set_epoch(epoch)

    def forward(self, data):
        # Run experts
        shared_out = self.shared(data, data.y)  # (B,K,C) etc.

        # Gate scores from RAW graph
        # gate_scores = self._gate_logits_raw(data)  # (B,K)
        gate_scores = self._gate_logits_expert_features(data, shared_out)  # (B,K)

        # --- NEW: train the gate (independent of routing choice) ---
        gate_loss, p_gate_train = self._compute_gate_loss(shared_out, data.y, gate_scores)

        # Routing policy
        if self.current_epoch < self.train_after:
            # Warm-up: keep routing uniform, but gate still learns via gate_loss above
            gate_probs = torch.full_like(gate_scores, 1.0 / self.num_experts)
        else:
            # Post warm-up: route using the gate's own distribution
            gate_probs = p_gate_train

        if self.verbose:
            with torch.no_grad():
                print("[MoE] gate (first row):", gate_probs[0].tolist())

        # Aggregate logits and losses (task + expert-reg losses)
        out = self._aggregate(shared_out, gate_probs, targets=data.y)

        # --- NEW: add gate training loss to total (weighted) ---
        out['loss_gate']  = self.weight_gate * gate_loss
        out['loss_total'] = out['loss_total'] + out['loss_gate']
        out['p_gate_train'] = p_gate_train  # handy for logging

        return out

    # ------------- Internals -------------
    def _gate_logits_raw(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = self.gate_enc(x, edge_index, batch=batch)
        return self.gate_mlp(h)  # (B, K)

    def _aggregate(self, shared_out, gate_probs, targets):
        """
        Gate-weighted aggregation:
          - logits: per-sample gate-weighted sum of expert logits
          - CE:    per-expert, per-sample CE weighted by that sample's gate prob for that expert
          - other losses: gate-averaged across batch
        """
        expert_logits = shared_out['expert_logits']       # (B, K, C)
        B, K, C = expert_logits.shape

        # logits aggregation
        weighted = expert_logits * gate_probs.unsqueeze(-1)           # (B, K, C)
        agg_logits = weighted.sum(dim=1)                              # (B, C)

        # CE aggregated per expert, per sample
        ce = self._gate_weighted_ce(expert_logits.transpose(0, 1), targets, gate_probs.transpose(0, 1))

        # ---- NEW: gate-weight per-sample LA/EA/STR if provided ----
        ps = shared_out.get('per_sample', None)  # dict of (B,K): 'ce','la','ea','str'
        if ps is not None:
            # gate-weighted mean over (B,K) -> scalar
            def gate_weighted_mean(mat_bk):             # mat_bk: (B,K)
                return (gate_probs * mat_bk).sum(dim=1).mean()

            # note: your per-sample vectors are already scaled by the respective weights (λ_E, weight_str, etc.)
            la  = gate_weighted_mean(ps['la'])
            ea  = gate_weighted_mean(ps['ea'])
            strl= gate_weighted_mean(ps['str'])

            # Regularization remains expert-scalar × avg gate (batch)
            avg_gate = gate_probs.mean(dim=0).detach()  # (K,)
            reg = (avg_gate * shared_out['loss_reg_list']).sum()
        else:
            # Fallback: old behavior (batch-mean gate weighting of expert scalars)
            avg_gate = gate_probs.mean(dim=0).detach()  # (K,)
            reg = (avg_gate * shared_out['loss_reg_list']).sum()
            la  = (avg_gate * shared_out['loss_la_list']).sum()
            ea  = (avg_gate * shared_out['loss_ea_list']).sum()
            strl= (avg_gate * shared_out['loss_str_list']).sum()

        # Diversity (already scalar); Load-balance on gate (optional)
        div = shared_out['loss_div']

        total = (ce + reg + la + ea + strl + div)

        return {
            'logits': agg_logits,
            'loss_total': total,
            'loss_ce': ce,
            'loss_reg': reg,
            'loss_la': la,
            'loss_ea': ea,
            'loss_str': strl,
            'loss_div': div,
            'gate_weights': gate_probs,              # (B, K)
            'rho': shared_out['rho'],
            'expert_logits': expert_logits,         # (B, K, C)
            'node_masks': shared_out['node_masks'],
            'edge_masks': shared_out['edge_masks']
        }

    def _compute_gate_loss(self, shared_out, targets, gate_scores, eps: float = 1e-12):
        """
        Train the gate to pick competent experts, even during warm-up:
        - Teacher q from per-expert CE on expert logits (stop-grad).
        - Student p from gate_scores (entmax/softmax on scores).
        - Loss = KL(q || p) + weight_lb * load_balance(p).
        Does NOT backprop through experts (teacher is detached).
        """
        expert_logits = shared_out['expert_logits']            # (B,K,C)
        B, K, C = expert_logits.shape

        # ----- Teacher: oracle from per-expert CE (stop-grad) -----
        with torch.no_grad():
            # Per-sample, per-expert task CE (lower is better)

            # uniform weights: each expert equally weighted (K,B)
            uni = torch.full((K, B), 1.0 / K, device=expert_logits.device)

            # reuse same CE construction (incl. GOOD-HIV variants) to get (B,K)
            ce_per = self._gate_weighted_ce(
                expert_logits.transpose(0,1), targets, uni, return_matrix=True
            )  # (B,K)

            # Start from -CE
            r = -ce_per

            # --- NEW: use per-sample LA/EA/STR if available ---
            ps = shared_out.get('per_sample', None)
            if ps is not None:
                # Lower LA/EA/STR should increase reward, so subtract them
                if self.gate_teacher_w_ea != 0.0 and 'ea' in ps:
                    r = r + self.gate_teacher_w_ea * (-ps['ea'].detach())
                if self.gate_teacher_w_la != 0.0 and 'la' in ps:
                    r = r + self.gate_teacher_w_la * (-ps['la'].detach())
                # (Optional) include STR as well if you want the gate to avoid spurious experts:
                # r = r + w_str_oracle * (-ps['str'].detach())
            else:
                # Fallback: batch-mean scalars (previous behavior)
                if 'loss_ea_list' in shared_out and self.gate_teacher_w_ea != 0.0:
                    ea_k = shared_out['loss_ea_list'].detach()        # (K,)
                    r = r + self.gate_teacher_w_ea * ea_k.view(1, K).expand_as(r)
                if 'loss_la_list' in shared_out and self.gate_teacher_w_la != 0.0:
                    la_k = shared_out['loss_la_list'].detach()        # (K,)
                    r = r + self.gate_teacher_w_la * la_k.view(1, K).expand_as(r)

            q = F.softmax(r / self.gate_tau_oracle, dim=1)        # (B,K)

        # ----- Student: gate probabilities from gate_scores (independent of warm-up routing) -----
        scores = gate_scores / max(self.gate_temperature, 1e-6)
        if self.entmax_alpha > 1:
            p = entmax_bisect(scores, alpha=self.entmax_alpha, dim=1)  # (B,K)
        else:
            p = F.softmax(scores, dim=1)

        # ----- KL(q || p) -----
        kl = F.kl_div((q.clamp_min(eps)).log(), p.clamp_min(eps), reduction='batchmean')

        # ----- Load balance on the gate's own distribution p -----
        lb = self._load_balance(p)

        # Weighted sum; return scalar and p if the caller wants to log/inspect it
        gate_loss = self.weight_gate_oracle * kl + self.weight_load * lb
        return gate_loss, p
    
    def _gate_logits_expert_features(self, data, shared_out, eps: float = 1e-12):
        """
        Replacement for _gate_logits_raw: builds label-free per-expert features Phi (B,K,d)
        and maps them to gate scores (B,K) with a tiny row-wise MLP.

        Expects in shared_out:
        - 'expert_logits': (B,K,C)  [required]
        - optional: 'weak_kl_list': (B,K)           # KL(p, p_weak_aug)
        - optional: 'node_masks':   (B,K,N) or (K,N_total)
        - optional: 'edge_masks':   (B,K,E) or (K,E_total)
        - optional: 'env_post_causal': (B,K,|E|)
        - optional: 'label_post_spur': (B,K,C)
        """
        import torch
        import torch.nn.functional as F

        logits = shared_out['expert_logits'].detach()     # (B,K,C)
        probs  = logits.softmax(dim=-1)
        B, K, C = probs.shape
        dev, dtype = probs.device, probs.dtype

        # ---------- A) Confidence proxies ----------
        maxp    = probs.max(dim=-1).values.unsqueeze(-1)                    # (B,K,1)
        top2    = probs.topk(k=min(2, C), dim=-1).values
        margin  = (top2[..., 0] - (top2[..., 1] if C > 1 else 0.)).unsqueeze(-1)
        entropy = (-(probs.clamp_min(eps) * probs.clamp_min(eps).log())
                .sum(-1, keepdim=True))                                  # (B,K,1)
        energy  = (-torch.logsumexp(logits, dim=-1, keepdim=True))          # (B,K,1)

        # ---------- B) Stability (if provided) ----------
        weak_kl = shared_out.get('weak_kl_list', None)  # (B,K)
        stab    = (-weak_kl.detach()).unsqueeze(-1) if weak_kl is not None else torch.zeros(B,K,1, device=dev, dtype=dtype)

        # ---------- C) Disagreement (avg sym-KL to others) ----------
        with torch.no_grad():
            if K > 1:
                logp   = probs.clamp_min(eps).log()
                kl_kj  = (probs.unsqueeze(2) * (logp.unsqueeze(2) - logp.unsqueeze(1))).sum(-1)  # (B,K,K)
                sym_kl = 0.5*(kl_kj + kl_kj.transpose(1,2))
                disagree = -(sym_kl.sum(dim=2) / (K - 1)).unsqueeze(-1)      # higher = less disagreement
            else:
                disagree = torch.zeros(B,K,1, device=dev, dtype=dtype)

        # ---------- D) Invariance/leakage entropies (label-free) ----------
        pe_c = shared_out.get('env_post_causal', None)   # (B,K,|E|)
        env_ent = (-(pe_c.clamp_min(eps) * pe_c.clamp_min(eps).log()).sum(-1, keepdim=True)
                if pe_c is not None else torch.zeros(B,K,1, device=dev, dtype=dtype))
        py_s = shared_out.get('label_post_spur', None)   # (B,K,C)
        leak_ent = (-(py_s.clamp_min(eps) * py_s.clamp_min(eps).log()).sum(-1, keepdim=True)
                    if py_s is not None else torch.zeros(B,K,1, device=dev, dtype=dtype))

        # ---------- F) Simple graph stats ----------
        n = torch.tensor([data.num_nodes], device=dev, dtype=dtype).view(1,1,1).expand(B,K,1)
        m = torch.tensor([data.num_edges], device=dev, dtype=dtype).view(1,1,1).expand(B,K,1)

        # Concatenate features and standardize
        Phi = torch.cat([
            maxp, margin, entropy, energy,           # A
            stab, disagree,                          # B,C
            env_ent, leak_ent,                       # D  # E
            n, m                                     # F
        ], dim=-1)                                   # (B,K,d_phi)

        mu = Phi.mean(dim=(0,1), keepdim=True)
        sd = Phi.std(dim=(0,1), keepdim=True).clamp_min(1e-3)
        Phi = (Phi - mu) / sd

        # Row-wise MLP (lazy init) -> scores (B,K)
        d_phi = Phi.size(-1)
        if not hasattr(self, "_gate_mlp") or self._gate_mlp is None:
            self._gate_mlp = nn.Sequential(
                nn.Linear(d_phi, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ).to(dev)

        scores = self._gate_mlp(Phi).squeeze(-1)  # (B,K)
        return scores
    
    def _gate_weighted_ce(self, stacked_logits, targets, gate_weights, return_matrix: bool = False):
        """
        stacked_logits: (K, B, C)
        gate_weights:   (K, B)
        return_matrix:  if True -> return (B,K) CE for each (sample, expert)
        """

        K, B, C = stacked_logits.shape
        device = stacked_logits.device
        y = targets.view(-1).long().to(device)
        ce_bk = stacked_logits[0, :, 0].new_zeros(B, K)  # will hold (B,K)

        if self.dataset_name != "GOODHIV":
            for k in range(K):
                ce_bk[:, k] = F.cross_entropy(stacked_logits[k], y, reduction='none')
            return ce_bk if return_matrix else (gate_weights * ce_bk.T).sum(dim=0).mean()

        # ---- GOOD-HIV variants (as you already added) ----
        tau_logitadj, beta_cb, gamma_focal, eps_smooth = 1.0, 0.999, 2.0, 0.05
        eps = 1e-8

        counts = torch.bincount(y, minlength=C).float().to(device)
        counts[counts == 0] = 1.0
        prior = (counts / counts.sum()).clamp_min(1e-8)

        eff_num = (1.0 - counts.pow(beta_cb)) / (1.0 - beta_cb)
        w_cb = (1.0 / eff_num).clamp_min(1e-8)
        w_cb = (w_cb / w_cb.sum()).detach()

        for k in range(K):
            logits = stacked_logits[k]

            logits_la = logits + tau_logitadj * prior.log().view(1, -1)
            ce_la = F.cross_entropy(logits_la, y, reduction='none')

            ce_cb = F.cross_entropy(logits, y, reduction='none', weight=w_cb)

            logp = F.log_softmax(logits, dim=1)
            pt   = logp.exp().gather(1, y.view(-1,1)).squeeze(1).clamp_min(eps)
            ce_flat  = F.nll_loss(logp, y, reduction='none')
            ce_focal = (1.0 - pt).pow(gamma_focal) * ce_flat

            with torch.no_grad():
                q = torch.full_like(logp, eps_smooth / (C - 1))
                q.scatter_(1, y.view(-1,1), 1.0 - eps_smooth)
            ce_ls = -(q * logp).sum(dim=1)

            # combine (sum or average — up to you)
            # ce_all = ce_la + ce_cb + ce_focal + ce_ls
            ce_all = ce_focal
            ce_bk[:, k] = ce_all

        return ce_bk if return_matrix else (gate_weights * ce_bk.T).sum(dim=0).mean()


    @staticmethod
    def _load_balance(gate_probs, lam=0.2, T=0.2, eps=1e-12):
        """
        gate_probs: (B, K) from entmax/softmax; rows sum to 1, entries >= 0
        lam: weight on per-sample entropy penalty (higher lam -> sparser routing)
        T:   sharpening temperature for "winner" histogram (smaller -> sharper)
        """
        B, K = gate_probs.shape
        uniform = gate_probs.new_full((K,), 1.0 / K)

        # 1) Balanced usage across the batch: KL(uniform || p_bar)
        p_bar = gate_probs.mean(dim=0).clamp_min(eps)     # (K,)
        log_pbar = p_bar.log()
        log_u = uniform.log()                             # constant
        L_bal = torch.sum(uniform * (log_u - log_pbar))   # scalar

        # 2) Per-sample sharpness: penalize high entropy (encourage sparsity)
        #    H(p_i) = -sum_j p_ij log p_ij
        H_rows = -(gate_probs.clamp_min(eps) * gate_probs.clamp_min(eps).log()).sum(dim=1).mean()

        # 3) Spread "winners" across experts (soft top-1 histogram)
        #    Sharpen with power-normalization: q_ij ∝ p_ij^(1/T)
        if T <= 0:
            # avoid division by zero; default to argmax one-hots
            q = torch.zeros_like(gate_probs)
            q.scatter_(1, gate_probs.argmax(dim=1, keepdim=True), 1.0)
        else:
            q_pow = gate_probs.clamp_min(eps).pow(1.0 / T)
            q = q_pow / q_pow.sum(dim=1, keepdim=True)    # (B,K)
        counts = q.mean(dim=0).clamp_min(eps)             # (K,)
        L_top1 = torch.sum(uniform * (log_u - counts.log()))

        # Final: reverse-KL balance + entropy penalty + (small) winner-spread
        return L_bal + lam * H_rows + 0.2 * L_top1
    
    
