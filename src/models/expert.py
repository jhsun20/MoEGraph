import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import BatchNorm as PYG_BatchNorm  # for isinstance checks
from models.gnn_models import GINEncoderWithEdgeWeight
from torch_geometric.utils import to_undirected


# -----------------------
# Gradient Reversal Layer (for LA)
# -----------------------
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd: float):
    return _GradReverse.apply(x, lambd)


class Experts(nn.Module):
    """
    Multi-expert module that:
      - learns hard-concrete node/edge/feature masks per expert;
      - computes per-expert logits;
      - applies (a) gate-agnostic regularization and diversity losses,
               (b) LA (label-adversarial) on semantic residual,
               (c) VIB on h_C (causal semantic) instead of EA,
               (d) (scheduled) structural invariance weight (_weight_str_live), even if str_loss is currently 0.0.
    """
    def __init__(self, config, dataset_info):
        super().__init__()

        num_features = dataset_info['num_features']
        num_classes  = dataset_info['num_classes']
        if num_classes == 1:
            if config['experiment']['debug']['verbose']:
                print("[warn] num_classes=1; forcing to 2 for CE over {0,1}")
            num_classes = 2

        mcfg         = config.get('model', {})
        hidden_dim   = mcfg['hidden_dim']
        num_layers   = mcfg['num_layers']
        dropout      = mcfg['dropout']

        # --- BN freeze controls (optional; set via config['model']) ---
        mcfg_bn = config.get('model', {})
        # Freeze BN running stats (and optionally affine params) at/after this epoch. Use -1 to disable.
        self.bn_freeze_epoch  = int(mcfg_bn.get('bn_freeze_epoch', -1))
        self.bn_freeze_affine = bool(mcfg_bn.get('bn_freeze_affine', False))
        self._bn_frozen = False

        self.num_experts = mcfg['num_experts']
        self.verbose    = config['experiment']['debug']['verbose']

        # ---------- Static/base loss weights from config ----------
        self.weight_ce  = float(mcfg['weight_ce'])
        self.weight_reg = float(mcfg['weight_reg'])
        self.weight_la = float(mcfg['weight_la'])
        self.weight_ea = float(mcfg['weight_ea'])
        self.weight_div = float(mcfg['weight_div'])
        self.weight_str = float(mcfg['weight_str'])  # baseline/static value; live value is scheduled below

        # ---------- Schedulers (mask temp, LA ramp, IB ramp, STR ramp) ----------
        # LA (label adversary) target
        self.lambda_L_end = float(mcfg.get('lambda_L_end', mcfg.get('mi_lambda_l_sem', 0.1)))
        self.adv_warmup_epochs = int(mcfg.get('adv_warmup_epochs', 5))
        self.adv_ramp_epochs   = int(mcfg.get('adv_ramp_epochs', 20))
        self._lambda_L = 0.0  # live value

        # Environment inference + EA scheduling (mirrors your LA/IB ramps)
        self.env_inference = bool(mcfg.get('env_inference', True))
        self.lambda_E_end = float(mcfg.get('lambda_E_end', 0.01))
        self.ea_warmup_epochs = int(mcfg.get('ea_warmup_epochs', self.adv_warmup_epochs))
        self.ea_ramp_epochs   = int(mcfg.get('ea_ramp_epochs',   self.adv_ramp_epochs))
        self._lambda_E = 0.0  # live value updated in set_epoch

        # VIB (IB) target
        self.beta_ib_end  = float(mcfg.get('beta_ib_end', 1e-3))
        self.ib_warmup_epochs  = int(mcfg.get('ib_warmup_epochs', 5))
        self.ib_ramp_epochs    = int(mcfg.get('ib_ramp_epochs', 20))
        self.ib_free_bits      = float(mcfg.get('ib_free_bits', 0.5))
        self._beta_ib = 0.0  # live value

        # Structural invariance (live) weight schedule (even while str_loss=0.0 for now)
        self.weight_str_end       = float(mcfg.get('weight_str_end', self.weight_str))
        self.strinv_warmup_epochs = int(mcfg.get('strinv_warmup_epochs', 5))
        self.strinv_ramp_epochs   = int(mcfg.get('strinv_ramp_epochs', 20))
        self._weight_str_live     = 0.0

        # Mask temperature schedule (hard-concrete)
        self.mask_temp_start          = float(mcfg.get('mask_temp_start', 5.0))
        self.mask_temp_end            = float(mcfg.get('mask_temp_end', 0.1))
        self.mask_temp_anneal_epochs  = int(mcfg.get('mask_temp_anneal_epochs', 20))
        self.mask_temp_schedule       = str(mcfg.get('mask_temp_schedule', 'exp'))
        self._mask_temp               = self.mask_temp_start
        self.eval_mask_temp_floor     = float(mcfg.get('eval_mask_temp_floor', 0.0))

        # ---------- Encoders / Heads ----------
        # Causal/selector encoder to get node embeddings Z for masks
        self.causal_encoder = GINEncoderWithEdgeWeight(
            num_features, hidden_dim, num_layers, dropout, train_eps=True
        )

        # Classifier encoder (masked pass)
        self.classifier_encoder = GINEncoderWithEdgeWeight(
            num_features, hidden_dim, num_layers, dropout, train_eps=True
        )

        # --- LECI-style tiny GINs for invariant (lc) and environment (ea) heads ---
        tiny_lc_hidden = int(mcfg.get('leci_lc_hidden', hidden_dim))
        tiny_ea_hidden = int(mcfg.get('leci_ea_hidden', hidden_dim))
        tiny_la_hidden = int(mcfg.get('leci_la_hidden', hidden_dim))
        tiny_depth     = int(mcfg.get('leci_depth', 2))
        tiny_drop      = float(mcfg.get('leci_dropout', dropout))

        # One tiny lc_gnn and ea_gnn per expert (keeps behavior closest to LECI while preserving MoE)
        self.lc_encoders = nn.ModuleList([
            GINEncoderWithEdgeWeight(num_features, tiny_lc_hidden, tiny_depth, tiny_drop, train_eps=True)
            for _ in range(self.num_experts)
        ])

        self.ea_encoders = nn.ModuleList([
            GINEncoderWithEdgeWeight(num_features, tiny_ea_hidden, tiny_depth, tiny_drop, train_eps=True)
            for _ in range(self.num_experts)
        ])

        # Per-expert EA classifier (predicts environment id)
        self.num_envs = dataset_info['num_envs']
        self.ea_classifiers = nn.ModuleList([
            nn.Linear(tiny_ea_hidden, self.num_envs) for _ in range(self.num_experts)
        ])

        # --- LECI-style tiny GINs for the LA (spur/complement) branch ---
        self.la_encoders = nn.ModuleList([
            GINEncoderWithEdgeWeight(num_features, tiny_la_hidden, tiny_depth, tiny_drop, train_eps=True)
            for _ in range(self.num_experts)
        ])


        # Per-expert maskers
        self.expert_node_masks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
            for _ in range(self.num_experts)
        ])
        self.expert_edge_masks = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
            for _ in range(self.num_experts)
        ])
        self.expert_feat_masks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_features))
            for _ in range(self.num_experts)
        ])

        # Per-expert classifiers
        self.expert_classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(self.num_experts)
        ])

        # Label-adversarial head on semantic residual h_S
        self.lbl_head_spur_sem = nn.Linear(hidden_dim, num_classes)

        # VIB heads on h_C (created lazily to bind H on first forward)
        self.ib_mu_head_sem      = None
        self.ib_logvar_head_sem  = None

        # Keep-rate priors (trainable) per expert for regularization
        self.rho_node = nn.Parameter(torch.empty(self.num_experts).uniform_(0.2, 0.8))
        self.rho_edge = nn.Parameter(torch.empty(self.num_experts).uniform_(0.2, 0.8))
        self.rho_feat = nn.Parameter(torch.empty(self.num_experts).uniform_(0.2, 0.8))

        # print(f"dataset_info['num_envs']: {dataset_info['num_envs']}")

    # ----------------- Public API -----------------
    def set_epoch(self, epoch: int):
        """Update schedulers for mask temperature, LA lambda, VIB beta, and structural live weight."""
        # Mask temp schedule
        t = max(epoch, 0)
        T = max(self.mask_temp_anneal_epochs, 1)
        if self.mask_temp_schedule.lower() == 'exp':
            r = min(t / T, 1.0)
            self._mask_temp = float(self.mask_temp_start * (self.mask_temp_end / self.mask_temp_start) ** r)
        else:  # linear fallback
            r = min(t / T, 1.0)
            self._mask_temp = float(self.mask_temp_start + r * (self.mask_temp_end - self.mask_temp_start))

        # LA ramp (after warmup)
        if epoch < self.adv_warmup_epochs:
            self._lambda_L = 0.0
        else:
            r_adv = min((epoch - self.adv_warmup_epochs) / max(self.adv_ramp_epochs, 1), 1.0)
            self._lambda_L = float(r_adv * self.lambda_L_end)

        # EA ramp (after warmup) -- mirrors LA ramp
        if epoch < self.ea_warmup_epochs:
            self._lambda_E = 0.0
        else:
            r_ea = min((epoch - self.ea_warmup_epochs) / max(self.ea_ramp_epochs, 1), 1.0)
            self._lambda_E = float(r_ea * self.lambda_E_end)

        # IB ramp (after warmup)
        if epoch < self.ib_warmup_epochs:
            self._beta_ib = 0.0
        else:
            r_ib = min((epoch - self.ib_warmup_epochs) / max(self.ib_ramp_epochs, 1), 1.0)
            self._beta_ib = float(r_ib * self.beta_ib_end)

        # Structural invariance (live) weight ramp (after warmup)
        if epoch < self.strinv_warmup_epochs:
            self._weight_str_live = 0.0
        else:
            p = min((epoch - self.strinv_warmup_epochs) / max(self.strinv_ramp_epochs, 1), 1.0)
            self._weight_str_live = float(self.weight_str_end * p)

        if self.verbose and epoch % 10 == 0:
            print(f"[Experts.set_epoch] epoch={epoch} | temp={self._mask_temp:.3f} "
                  f"| lambda_L={self._lambda_L:.4f} | beta_ib={self._beta_ib:.6f} "
                  f"| w_str_live={self._weight_str_live:.4f}")
            
        # Optionally freeze BN at/after a chosen epoch (once).
        if (not self._bn_frozen) and (self.bn_freeze_epoch >= 0) and (epoch >= self.bn_freeze_epoch):
            self._freeze_bn(freeze_affine=self.bn_freeze_affine, verbose=self.verbose)
            self._bn_frozen = True

    def forward(self, data, target=None):
        """
        Returns:
            dict with keys:
              h_stable, h_orig, node_masks, edge_masks, feat_masks, expert_logits,
              loss_total_list, loss_ce_list, loss_reg_list, loss_sem_list, loss_str_list, loss_div, rho
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Use ground-truth environment ids if available
        env_labels = getattr(data, "env_id", None)

        # Base embeddings used to produce masks
        Z = self.causal_encoder(x, edge_index, batch=batch)
        edge_feat = torch.cat([Z[edge_index[0]], Z[edge_index[1]]], dim=1)

        node_masks, edge_masks, feat_masks = [], [], []
        expert_logits, h_stable_list = [], []
        h_ea_list = []

        is_eval = not self.training

        for k in range(self.num_experts):
            node_mask_logits = self.expert_node_masks[k](Z)
            edge_mask_logits = self.expert_edge_masks[k](edge_feat)
            feat_mask_logits = self.expert_feat_masks[k](Z)

            
            node_mask = self._hard_concrete_mask(node_mask_logits, self._mask_temp, is_eval=is_eval)
            edge_mask = self._hard_concrete_mask(edge_mask_logits, self._mask_temp, is_eval=is_eval)
            # Enforce symmetry for edges that have reverse edges
            edge_mask = self.enforce_edge_mask_symmetry(edge_index, edge_mask, num_nodes=Z.size(0))
            feat_mask = self._hard_concrete_mask(feat_mask_logits, self._mask_temp, is_eval=is_eval)
            # if k == 0:  # or pick any expert index you want to monitor
            #     report = mask_symmetry_report(edge_index, edge_mask.view(-1))
            #     print(f"[Symmetry check, Expert {k}] {report}")

            node_masks.append(node_mask)
            edge_masks.append(edge_mask)
            feat_masks.append(feat_mask)

            # Apply masks
            masked_x = x * node_mask * feat_mask  # (N, D)
            edge_weight = edge_mask.view(-1)

            # LECI-style invariant head: tiny GIN -> mean pool
            masked_Z_lc = self.lc_encoders[k](masked_x, edge_index, batch=batch, edge_weight=edge_weight)
            h_stable = global_mean_pool(masked_Z_lc, batch)  # h_C

            # Classify with your existing per-expert classifier
            logit = self.expert_classifiers[k](h_stable)

            # Environment head: tiny GIN -> mean pool (for EA / environment inference)
            masked_Z_ea = self.ea_encoders[k](masked_x, edge_index, batch=batch, edge_weight=edge_weight)
            h_ea = global_mean_pool(masked_Z_ea, batch)

            h_stable_list.append(h_stable)
            expert_logits.append(logit)
            h_ea_list.append(h_ea)

        node_masks    = torch.stack(node_masks, dim=1)        # (N, K, 1)
        edge_masks    = torch.stack(edge_masks, dim=1)        # (E, K, 1)
        feat_masks    = torch.stack(feat_masks, dim=1)        # (N, K, D)
        expert_logits = torch.stack(expert_logits, dim=1)     # (B, K, C)
        h_stable_list = torch.stack(h_stable_list, dim=1)     # (B, K, H)
        h_ea_list     = torch.stack(h_ea_list, dim=1)         # (B, K, Hea)
        h_orig        = global_mean_pool(Z, batch)            # (B, H)

        out = {
            'h_stable': h_stable_list,
            'h_orig':   h_orig,
            'node_masks': node_masks,
            'edge_masks': edge_masks,
            'feat_masks': feat_masks,
            'expert_logits': expert_logits,
            'rho': [self.rho_node, self.rho_edge, self.rho_feat],
        }

        if target is not None:
            ce_list, reg_list, str_list, tot_list, la_list, ea_list = [], [], [], [], [], []

            for k in range(self.num_experts):
                logits_k = expert_logits[:, k, :]
                hC_k     = h_stable_list[:, k, :]

                # Classification CE (class-weighted)
                ce = self._ce(logits_k, target)
                ce_list.append(ce)

                # Mask regularization (per-graph keep-rate prior)
                reg = self._mask_reg(
                    node_masks[:, k, :], edge_masks[:, k, :], feat_masks[:, k, :],
                    node_batch=batch, edge_batch=batch[edge_index[0]], expert_idx=k
                )
                reg_list.append(reg)

                if not is_eval:
                    # Semantic invariance: VIB on h_C + LA on residual
                    h_spur = self._encode_complement_subgraph(
                        data=data,
                        node_mask=node_mask,           # (N,1) or (N,)
                        edge_mask=edge_mask.view(-1),  # (E,)
                        feat_mask=feat_mask,           # (N,F) or (F,)
                        encoder=self.la_encoders[k],
                        symmetrize=False,              # set True if you want undirected EW averaging
                    )

                    ea = self._causal_loss(
                        h_masked=hC_k,
                        h_ea=h_ea_list[:, k, :],
                        expert_idx=k,
                        env_labels=env_labels,
                    )

                    la = self._spur_loss(
                        h_masked=hC_k,
                        labels=target,
                        h_spur=h_spur
                    )

                else:
                    la = hC_k.new_tensor(0.0)
                    ea = hC_k.new_tensor(0.0)

                la_list.append(la)
                ea_list.append(ea)

                # Structural invariance currently OFF (keep explicit zero)
                str_loss = hC_k.new_tensor(0.0)
                str_list.append(str_loss)

                # Use live scheduled weight for structure (even though str_loss==0.0)
                total = (self.weight_ce * ce +
                         self.weight_reg * reg +
                         self._weight_str_live * str_loss +
                         self.weight_la * la +
                         self.weight_ea * ea)
                tot_list.append(total)

            # Diversity across experts' masks
            div_loss = self.weight_div * self._diversity_loss(node_masks, edge_masks, feat_masks,
                                            node_batch=batch, edge_batch=batch[edge_index[0]])

            out.update({
                'loss_total_list': torch.stack(tot_list),   # (K,)
                'loss_ce_list':    torch.stack(ce_list),
                'loss_reg_list':   torch.stack(reg_list),
                'loss_la_list':   torch.stack(la_list),
                'loss_ea_list':   torch.stack(ea_list),
                'loss_str_list':   torch.stack(str_list),
                'loss_div':        div_loss,
            })

        return out

    # ----------------- Internals -----------------
    def _hard_concrete_mask(self, logits, temperature=0.1, is_eval=False):
        # choose a temperature for eval that won't saturate
        T = max(temperature, self.eval_mask_temp_floor) if is_eval else max(temperature, 1e-6)

        if self.training:
            u = torch.rand_like(logits)
            g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            y_soft = torch.sigmoid((logits + g) / T)
        else:
            u = torch.rand_like(logits)
            g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            y_soft = torch.sigmoid((logits + g) / T)

        y_hard = (y_soft > 0.5).float()
        return y_hard + (y_soft - y_soft.detach())

    def _ce(self, pred, target, use_weights=True):
        if use_weights:
            C = pred.size(1)
            counts = torch.bincount(target, minlength=C).float()
            counts[counts == 0] = 1.0
            w = (1.0 / counts)
            w = w / w.sum()
            return F.cross_entropy(pred, target, weight=w.to(pred.device))
        return F.cross_entropy(pred, target)

    def _diversity_loss(
        self,
        node_masks: torch.Tensor,   # [N_nodes, K, 1] or [N_nodes, K]
        edge_masks: torch.Tensor,   # [N_edges, K, 1] or [N_edges, K]
        feat_masks: torch.Tensor,   # [N_nodes, K, F] (per-feature) or [N_nodes, K, 1]
        node_batch: torch.Tensor,   # [N_nodes] graph indices (0..B-1)
        edge_batch: torch.Tensor,   # [N_edges] graph indices (0..B-1)
    ):
        """
        Combines three complementary pieces:
        1) correlation hinge (shape-level de-correlation across experts)
        2) union/coverage (avoid "nobody selects anything")
        3) overlap/exclusivity (avoid "everybody selects the same thing")

        Hyperparameters are kept local with sensible defaults.
        """

        # ---- local hyperparameters (no config dependency) ----
        # weights of each component
        w_corr = 1.0
        w_uo   = 0.0     # multiplies (coverage + overlap) per modality before averaging

        # correlation hinge threshold (lower => stricter decorrelation)
        tau_corr = 0.10

        # union/overlap targets: encourage U_i >= tau_cov and S_i <= tau_over
        tau_cov  = 0.60   # require at least ~0.6 union coverage
        tau_over = 1.20   # allow ≈1 expert on average (softly), >1.2 gets penalized

        eps = 1e-8

        # -------- helpers --------
        def _maybe_squeeze(v: torch.Tensor) -> torch.Tensor:
            # squeeze last dim if it's singleton
            return v.squeeze(-1) if v.dim() >= 2 and v.size(-1) == 1 else v

        def _per_graph_abs_corr_hinge(V: torch.Tensor, bidx: torch.Tensor) -> torch.Tensor:
            """
            V: [items, K] mask probabilities for one modality
            For each graph, z-score over items, compute |corr| matrix across experts,
            take off-diagonal mean, hinge above tau_corr, then average over graphs.
            """
            if V.numel() == 0 or bidx.numel() == 0:
                return V.new_tensor(0.0)

            B = int(bidx.max().item()) + 1 if bidx.numel() > 0 else 0
            vals = []
            for g in range(B):
                sel = (bidx == g)
                if sel.sum() < 2:
                    continue
                X = V[sel]  # [n_g, K]
                # standardize over items to avoid trivial scale effects
                X = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, unbiased=False, keepdim=True) + eps)
                # correlation proxy via normalized inner product
                C = (X.t() @ X) / max(X.size(0), 1)  # [K,K]
                M = C.abs()
                # off-diagonal mean
                off_sum = M.sum() - M.diag().sum()
                denom   = max(M.size(0) * (M.size(0) - 1), 1)
                off_mean = off_sum / denom
                vals.append(F.relu(off_mean - tau_corr))
            return torch.stack(vals).mean() if len(vals) else V.new_tensor(0.0)

        def _union_overlap_terms(V: torch.Tensor, bidx: torch.Tensor) -> torch.Tensor:
            """
            V: [items, K] mask probabilities in [0,1]
            For each graph:
            U_i = 1 - Π_k (1 - m_{k,i})      (smooth union)
            S_i = Σ_k m_{k,i}                (aggregate overlap)
            Loss per graph = mean( ReLU(tau_cov - U_i) + ReLU(S_i - tau_over) )
            """
            if V.numel() == 0 or bidx.numel() == 0:
                return V.new_tensor(0.0)

            Vc = V.clamp(0.0, 1.0)
            B = int(bidx.max().item()) + 1 if bidx.numel() > 0 else 0
            vals = []
            for g in range(B):
                sel = (bidx == g)
                if sel.sum() == 0:
                    continue
                M = Vc[sel]                       # [n_g, K]
                # smooth union (better gradients than torch.max)
                U = 1.0 - torch.prod(1.0 - M + eps, dim=1)  # [n_g]
                # aggregate overlap
                S = M.sum(dim=1)                               # [n_g]
                cov  = F.relu(tau_cov  - U).mean()
                over = F.relu(S - tau_over).mean()
                vals.append(cov + over)
            return torch.stack(vals).mean() if len(vals) else V.new_tensor(0.0)

        # -------- prepare modality tensors to shape [items, K] --------
        N = _maybe_squeeze(node_masks)                   # [N_nodes, K]
        E = _maybe_squeeze(edge_masks)                   # [N_edges, K]
        # feat: if per-feature masks exist, average across features to a single prob per node
        if feat_masks.dim() == 3 and feat_masks.size(-1) > 1:
            Fm = feat_masks.mean(dim=-1)                 # [N_nodes, K]
        else:
            Fm = _maybe_squeeze(feat_masks)              # [N_nodes, K]

        # -------- compute components per modality --------
        n_corr = _per_graph_abs_corr_hinge(N, node_batch)
        e_corr = _per_graph_abs_corr_hinge(E, edge_batch)
        f_corr = _per_graph_abs_corr_hinge(Fm, node_batch)
        corr   = torch.stack([n_corr, e_corr, f_corr]).mean()

        n_uo = _union_overlap_terms(N, node_batch)
        e_uo = _union_overlap_terms(E, edge_batch)
        f_uo = _union_overlap_terms(Fm, node_batch)
        uo   = torch.stack([n_uo, e_uo, f_uo]).mean()

        # final combined diversity loss
        # print(f"corr: {corr}, uo: {uo}")
        # print(f"w_corr: {w_corr}, w_uo: {w_uo}")
        # print(f"added loss: {w_corr * corr + w_uo * uo}")
        loss = w_corr * corr + w_uo * uo
        return loss

    def _mask_reg(self, node_mask, edge_mask, feat_mask, node_batch, edge_batch, expert_idx: int,
                  use_fixed_rho: bool = False, fixed_rho_vals: tuple = (0.5, 0.5, 0.5)):
        if use_fixed_rho:
            rho_node, rho_edge, rho_feat = [float(min(max(v, 0.0), 1.0)) for v in fixed_rho_vals]
        else:
            rho_node = torch.clamp(self.rho_node[expert_idx], 0.4, 0.6)
            rho_edge = torch.clamp(self.rho_edge[expert_idx], 0.4, 0.6)
            rho_feat = torch.clamp(self.rho_feat[expert_idx], 0.4, 0.6)

        def per_graph_keep(mask_vals, batch_idx):
            G = batch_idx.max().item() + 1
            keep = torch.zeros(G, device=mask_vals.device)
            cnt  = torch.zeros(G, device=mask_vals.device)
            keep.scatter_add_(0, batch_idx, mask_vals.squeeze())
            cnt.scatter_add_(0, batch_idx, torch.ones_like(mask_vals.squeeze()))
            return keep / (cnt + 1e-8)

        node_keep_pg = per_graph_keep(node_mask, node_batch)
        edge_keep_pg = per_graph_keep(edge_mask, edge_batch)
        feat_keep_pg = per_graph_keep(feat_mask.mean(dim=1, keepdim=True), node_batch)

        return ((node_keep_pg - rho_node) ** 2).mean() + \
               ((edge_keep_pg - rho_edge) ** 2).mean() + \
               ((feat_keep_pg - rho_feat) ** 2).mean()

    def _causal_loss(
        self,
        h_masked: torch.Tensor,         # h_C (B, H)
        h_ea:   Optional[torch.Tensor] = None,    # (B, Hea)
        expert_idx: Optional[int] = None,
        env_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        VIB on h_C + LA on residual h_S = (h_orig - h_C) to discourage label signal in spur part.
        """
        device = h_masked.device
        B, H = h_masked.shape

        # ----- VIB (q(z|h_C)) -----
        # if self.ib_mu_head_sem is None:
        #     self.ib_mu_head_sem = nn.Linear(H, H).to(device)
        #     self.ib_logvar_head_sem = nn.Linear(H, H).to(device)
        #     nn.init.constant_(self.ib_logvar_head_sem.weight, 0.0)
        #     nn.init.constant_(self.ib_logvar_head_sem.bias, -5.0)  # small var init

        # mu     = self.ib_mu_head_sem(h_masked)
        # logvar = self.ib_logvar_head_sem(h_masked)
        # kl_ps  = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1)   # [B]
        # kl_ps  = torch.clamp(kl_ps - self.ib_free_bits, min=0.0)
        # vib = self._beta_ib * kl_ps.mean()

        # --- EA on h_ea (environment adversary on G_C) ---
        ea = h_masked.new_tensor(0.0)
        if self._lambda_E > 0.0 and h_ea is not None and env_labels is not None:
            # 1) sanitize env labels
            env_tgt = env_labels.view(-1).long().to(h_ea.device)   # (B,)
            # 2) sanity checks
            if env_tgt.numel() != h_ea.size(0):
                raise ValueError(f"env_labels shape {tuple(env_tgt.shape)} "
                                f"must match batch size {h_ea.size(0)}")
            if int(env_tgt.max()) >= self.num_envs or int(env_tgt.min()) < 0:
                raise ValueError(f"env_labels must be in [0, {self.num_envs-1}] "
                                f"(got min={int(env_tgt.min())}, max={int(env_tgt.max())})")
            # 3) adversarial logits (GRL applies the reversed grad only to features)
            h_ea_adv = grad_reverse(h_ea, self._lambda_E)          # scale via λ_E here
            logits_E = self.ea_classifiers[expert_idx](h_ea_adv)   # (B, num_envs)
            if logits_E.size(1) != self.num_envs:
                raise RuntimeError(f"EA head out={logits_E.size(1)} != num_envs={self.num_envs}")
            # 4) standard CE (do NOT multiply by λ_E again—GRL already scales feature grads)
            ea = F.cross_entropy(logits_E, env_tgt)

        return ea
    
    def _spur_loss(
        self,
        h_masked: torch.Tensor,         # h_C (B, H)
        labels: torch.Tensor,           # (B,)
        h_spur: Optional[torch.Tensor] = None,    # (B, H_spur)
    ) -> torch.Tensor:
        device = h_masked.device
        B, H = h_masked.shape

        # ----- VIB (q(z|h_C)) -----
        # if self.ib_mu_head_sem is None:
        #     self.ib_mu_head_sem = nn.Linear(H, H).to(device)
        #     self.ib_logvar_head_sem = nn.Linear(H, H).to(device)
        #     nn.init.constant_(self.ib_logvar_head_sem.weight, 0.0)
        #     nn.init.constant_(self.ib_logvar_head_sem.bias, -5.0)  # small var init

        # mu     = self.ib_mu_head_sem(h_masked)
        # logvar = self.ib_logvar_head_sem(h_masked)
        # kl_ps  = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1)   # [B]
        # kl_ps  = torch.clamp(kl_ps - self.ib_free_bits, min=0.0)
        # vib = self._beta_ib * kl_ps.mean()

        # --- LA on complement embedding ---
        if h_spur is None:
            raise ValueError("h_spur must be provided: encode complement subgraph with la_encoders[k]")
        la = h_masked.new_tensor(0.0)
        if self._lambda_L > 0.:
            logits_y_spur = self.lbl_head_spur_sem(grad_reverse(h_spur, self._lambda_L))
            la = F.cross_entropy(logits_y_spur, labels)

        return la
    
    def _kmeans_labels(self, X: torch.Tensor, K: int, iters: int = 10) -> torch.Tensor:
        # X: (B, D)
        B = X.size(0)
        # init with random samples
        idx = torch.randperm(B, device=X.device)[:K]
        C = X[idx].clone()  # (K, D)
        for _ in range(max(iters, 1)):
            # assign
            d2 = torch.cdist(X, C, p=2.0)  # (B, K)
            a = d2.argmin(dim=1)           # (B,)
            # update
            for k in range(K):
                sel = (a == k)
                if sel.any():
                    C[k] = X[sel].mean(dim=0)
        return a
    
    def _encode_complement_subgraph(
        self,
        data,
        node_mask: torch.Tensor,   # (N,) or (N,1)
        edge_mask: torch.Tensor,   # (E,) or (E,1)
        feat_mask: torch.Tensor,   # (F,) or (N,F)
        encoder: torch.nn.Module,  # e.g., self.classifier_encoder or a tiny spur encoder
        symmetrize: bool = False,  # set True if your graph is undirected and you want EW symmetry
    ):
        """
        Build spur graph G_S by complementing masks (1 - mask) and re-encode to get h_S.

        Returns:
        h_spur: (B, H_enc) pooled embedding of complement subgraph.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        N, F = x.size()
        device = x.device

        # --- clamp + shape ---
        node_m = node_mask
        if node_m.dim() == 1: node_m = node_m.view(-1, 1)
        node_m = node_m.clamp(0.0, 1.0)                     # (N,1)

        edge_m = edge_mask
        if edge_m.dim() == 1: edge_m = edge_m.view(-1, 1)
        edge_m = edge_m.clamp(0.0, 1.0).view(-1)            # (E,)

        feat_m = feat_mask.clamp(0.0, 1.0)
        if feat_m.dim() == 1:
            feat_m = feat_m.view(1, -1).expand(N, -1)       # (N,F)
        elif feat_m.size(0) == 1:
            feat_m = feat_m.expand(N, -1)                   # (N,F)
        elif feat_m.size(0) != N or feat_m.size(1) != F:
            raise ValueError(f"feat_mask must be (F,), (1,F), or (N,F); got {tuple(feat_mask.shape)}")

        # --- complement masks ---
        node_m_S = 1.0 - node_m                              # (N,1)
        edge_m_S = 1.0 - edge_m                              # (E,)
        feat_m_S = 1.0 - feat_m                              # (N,F)

        # --- apply to features (node-wise * feature-wise) ---
        x_spur = (x * node_m_S) * feat_m_S                   # (N,F)

        # --- optional symmetrization of edge weights for undirected graphs ---
        ei, ew_spur = edge_index, edge_m_S
        if symmetrize:
            ei, ew_spur = to_undirected(ei, ew_spur, reduce='mean')

        # --- encode + pool ---
        Z_spur = encoder(x_spur, ei, batch=batch, edge_weight=ew_spur)
        h_spur = global_mean_pool(Z_spur, batch)             # (B,H_enc)
        return h_spur
    
    def enforce_edge_mask_symmetry(self, edge_index: torch.Tensor,
                                edge_mask: torch.Tensor,
                                num_nodes: int) -> torch.Tensor:
        """
        Post-hoc symmetrize edge_mask across (i,j) and (j,i) only if both directions exist.
        Does NOT create new edges. Self-loops are unaffected.

        Args:
            edge_index : LongTensor [2, E] - directed edges
            edge_mask  : Tensor [E] or [E,1] - mask values
            num_nodes  : int - number of nodes (for pair indexing)

        Returns:
            sym_edge_mask: Tensor with the same shape as edge_mask, symmetric where possible.
        """
        # Flatten to [E]
        m = edge_mask.view(-1)
        row, col = edge_index[0], edge_index[1]

        # Unique undirected pair IDs
        u = torch.minimum(row, col)
        v = torch.maximum(row, col)
        pair_id = u * num_nodes + v

        uniq, inv, counts = torch.unique(pair_id, return_inverse=True, return_counts=True)
        has_both = (counts == 2)
        sel = has_both[inv]  # per-edge flag

        if sel.any():
            inv_sel = inv[sel]
            m_sel = m[sel]

            # Aggregate across both directions
            sum_per = torch.zeros_like(counts, dtype=m.dtype, device=m.device)
            cnt_per = torch.zeros_like(counts, dtype=m.dtype, device=m.device)
            sum_per.scatter_add_(0, inv_sel, m_sel)
            cnt_per.scatter_add_(0, inv_sel, torch.ones_like(m_sel))
            mean_per = sum_per / cnt_per.clamp_min(1)

            # Choose reducer:
            reducer_per = mean_per  # <-- MEAN
            # reducer_per = torch.zeros_like(mean_per).scatter_reduce_(0, inv_sel, m_sel, reduce='amax', include_self=False) or mean_per  # <-- MAX alternative

            # Apply tied value back
            m[sel] = reducer_per[inv_sel]

        return m.view_as(edge_mask)
    
    def _remap_env_labels(self, env_labels: torch.Tensor) -> torch.Tensor:
        """
        Map raw environment IDs (possibly {1,2,3} or {2,4,7}, etc.) to contiguous 0..K-1 ids,
        keeping the mapping consistent across batches.
        """
        env_labels = env_labels.view(-1).long()
        device = env_labels.device

        # Lazily build/update mapping dicts
        if not hasattr(self, "_env_label_map"):
            self._env_label_map = {}   # raw_id -> mapped_id
            self._env_label_inv = []   # mapped_id -> raw_id

        # Add unseen raw ids to mapping
        uniq = torch.unique(env_labels).tolist()
        print(f"uniq: {uniq}")
        for raw in sorted(uniq):
            if raw not in self._env_label_map:
                self._env_label_map[raw] = len(self._env_label_inv)
                self._env_label_inv.append(raw)

        # Safety: ensure classifier has enough outputs
        num_seen = len(self._env_label_inv)
        if num_seen > self.num_envs:
            raise ValueError(
                f"num_envs={self.num_envs} is smaller than the number of distinct env ids seen={num_seen}. "
                "Increase model.num_envs (and ensure ea_classifiers out_features match)."
            )

        # Vectorized remap
        mapped = env_labels.clone()
        # Build a tensor lookup table for speed: size = max_raw+1, fill with -1 then assign
        max_raw = int(max(self._env_label_inv))
        lut = torch.full((max_raw + 1,), -1, dtype=torch.long, device=device)
        for raw, mapped_id in self._env_label_map.items():
            lut[raw] = mapped_id
        mapped = lut[mapped]  # (B,)

        # Final sanity
        if (mapped < 0).any():
            bad = env_labels[(mapped < 0).nonzero(as_tuple=True)[0]][:5]
            raise RuntimeError(f"Found unmapped env ids in batch (examples: {bad.tolist()}).")

        return mapped


    # ----------------- BN control -----------------
    def _freeze_bn(self, freeze_affine: bool = False, verbose: bool = False):
        """
        Puts all BN layers contained in this Experts module (including sub-encoders)
        into eval() so they use running stats, and optionally freezes affine params.
        """
        bn_count = 0
        for m in self.modules():
            # Cover both torch.nn BN and PyG BN
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, PYG_BatchNorm)):
                m.eval()  # stop updating running_mean/var; use buffers
                # (Optional) also stop training gamma/beta
                if freeze_affine:
                    if hasattr(m, 'weight') and m.weight is not None:
                        m.weight.requires_grad_(False)
                    if hasattr(m, 'bias') and m.bias is not None:
                        m.bias.requires_grad_(False)
                bn_count += 1
        if verbose:
            aff = " + affine" if freeze_affine else ""
            print(f"[Experts] Frozen {bn_count} BatchNorm layers (running stats{aff}).")


def mask_symmetry_report(edge_index: torch.Tensor,
                         edge_mask: torch.Tensor,
                         num_nodes: int = None,
                         atol: float = 1e-6):
    """
    Checks if edge masks are symmetric for undirected pairs (i, j) vs (j, i).

    Args:
        edge_index: LongTensor [2, E] (directed list; may contain both directions)
        edge_mask:  Tensor   [E]     (mask for each directed edge)
        num_nodes:  int or None      (used to create unique pair ids; inferred if None)
        atol:       float            (tolerance for considering masks equal)

    Returns:
        dict with:
          - symmetric_fraction: fraction of undirected pairs with both directions present AND |m_ij - m_ji| <= atol
          - both_dirs_fraction: fraction of undirected pairs that have both directions present
          - max_abs_pair_diff:  max |m_ij - m_ji| observed across pairs with both directions
          - mean_abs_pair_diff: mean |m_ij - m_ji| across pairs with both directions
          - num_pairs:          number of undirected (i, j) pairs (i<j) encountered
          - num_pairs_both:     number of pairs that have both directions present
          - note:               reminder that self-loops (i==j) are ignored here
    """
    assert edge_index.dim() == 2 and edge_index.size(0) == 2, "edge_index must be [2, E]"
    E = edge_index.size(1)
    m = edge_mask.view(-1)
    assert m.numel() == E, "edge_mask must have length E"

    row, col = edge_index[0], edge_index[1]
    if num_nodes is None:
        num_nodes = int(max(row.max().item(), col.max().item())) + 1

    # Build unordered pair ids for i<j; ignore self-loops for the symmetry test
    u = torch.minimum(row, col)
    v = torch.maximum(row, col)
    is_self = (u == v)
    u_noself, v_noself = u[~is_self], v[~is_self]
    m_noself = m[~is_self]

    pair_id = u_noself * num_nodes + v_noself  # unique id per undirected pair (i<j)

    # Group edges by pair_id
    unique_pairs, inv, counts = torch.unique(pair_id, return_inverse=True, return_counts=True)
    num_pairs = unique_pairs.numel()

    # We only care about pairs where both directions appear
    both_dirs_mask = (counts == 2)
    num_pairs_both = int(both_dirs_mask.sum().item())

    # Fast path: if no pairs have both directions, we can exit early
    if num_pairs_both == 0:
        return {
            "symmetric_fraction": 0.0,
            "both_dirs_fraction": 0.0,
            "max_abs_pair_diff": float("nan"),
            "mean_abs_pair_diff": float("nan"),
            "num_pairs": int(num_pairs),
            "num_pairs_both": 0,
            "note": "Self-loops ignored; no pairs had both directions present."
        }

    # For each pair with both directions, compute |m_ij - m_ji|
    # We'll do a tiny loop only over those pairs (diagnostic speed is fine).
    # Build an index list of edges belonging to each 'both-dir' pair
    pair_to_edges = [[] for _ in range(num_pairs)]
    for e_idx, p_idx in enumerate(inv.tolist()):
        pair_to_edges[p_idx].append(e_idx)

    diffs = []
    good = 0
    for p_idx, has2 in enumerate(both_dirs_mask.tolist()):
        if not has2:
            continue
        e_list = pair_to_edges[p_idx]
        assert len(e_list) == 2, "count==2 but did not collect 2 edges — unexpected"
        m1, m2 = m_noself[e_list[0]].item(), m_noself[e_list[1]].item()
        d = abs(m1 - m2)
        diffs.append(d)
        if d <= atol:
            good += 1

    diffs_t = torch.tensor(diffs, dtype=torch.float32)
    symmetric_fraction = good / num_pairs_both
    both_dirs_fraction = num_pairs_both / max(1, num_pairs)
    max_abs_pair_diff = float(diffs_t.max().item())
    mean_abs_pair_diff = float(diffs_t.mean().item())

    return {
        "symmetric_fraction": symmetric_fraction,
        "both_dirs_fraction": both_dirs_fraction,
        "max_abs_pair_diff": max_abs_pair_diff,
        "mean_abs_pair_diff": mean_abs_pair_diff,
        "num_pairs": int(num_pairs),
        "num_pairs_both": int(num_pairs_both),
        "note": "Self-loops (i==j) ignored; symmetry is evaluated only where both directions exist."
    }

