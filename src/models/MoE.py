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
        gate_scores = self._gate_logits_raw(data)  # (B,K)

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

        # Other per-expert losses (scalars) — combine via avg gate weights over batch
        avg_gate = gate_probs.mean(dim=0).detach()  # (K,)
        reg = (avg_gate * shared_out['loss_reg_list']).sum()
        la = (avg_gate * shared_out['loss_la_list']).sum()
        ea = (avg_gate * shared_out['loss_ea_list']).sum()
        strl = (avg_gate * shared_out['loss_str_list']).sum()

        # Diversity (already scalar); Load-balance on gate (optional)
        div = shared_out['loss_div']

        total = (ce + reg + la + ea + strl + div)

        return {
            'logits': agg_logits,
            'loss_total': total,
            'loss_ce': ce,
            'loss_reg': reg,
            'loss_la': la * self.weight_la,
            'loss_ea': ea * self.weight_ea,
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
            ce_per = F.cross_entropy(
                expert_logits.reshape(B * K, C),
                targets.repeat_interleave(K),
                reduction='none'
            ).reshape(B, K)  # (B,K)
            q = F.softmax(-ce_per / self.gate_tau_oracle, dim=1)  # (B,K), teacher

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

    @staticmethod
    def _gate_weighted_ce(stacked_logits, targets, gate_weights):
        """
        stacked_logits: (K, B, C)
        gate_weights:   (K, B)
        """
        K, B, _ = stacked_logits.shape
        # print(f"metric: {metric}, num_classes: {num_classes}")
        loss = stacked_logits[0, :, 0].new_tensor(0.0)
        for k in range(K):
            ce_per_sample = F.cross_entropy(stacked_logits[k], targets, reduction='none')  # (B,)
            loss += (gate_weights[k] * ce_per_sample).mean()
        return loss
    
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
    
    
