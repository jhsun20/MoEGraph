import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from torch_geometric.nn import global_mean_pool
from models.gnn_models import GINEncoderWithEdgeWeight


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
        self.num_experts = mcfg['num_experts']
        self.verbose    = config['experiment']['debug']['verbose']

        # ---------- Static/base loss weights from config ----------
        self.weight_ce  = float(mcfg['weight_ce'])
        self.weight_reg = float(mcfg['weight_reg'])
        self.weight_sem = float(mcfg['weight_sem'])
        self.weight_str = float(mcfg['weight_str'])  # baseline/static value; live value is scheduled below

        # ---------- Schedulers (mask temp, LA ramp, IB ramp, STR ramp) ----------
        # LA (label adversary) target
        self.lambda_L_end = float(mcfg.get('lambda_L_end', mcfg.get('mi_lambda_l_sem', 0.3)))
        self.adv_warmup_epochs = int(mcfg.get('adv_warmup_epochs', 5))
        self.adv_ramp_epochs   = int(mcfg.get('adv_ramp_epochs', 20))
        self._lambda_L = 0.0  # live value

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
        self.mask_temp_anneal_epochs  = int(mcfg.get('mask_temp_anneal_epochs', 40))
        self.mask_temp_schedule       = str(mcfg.get('mask_temp_schedule', 'exp'))
        self._mask_temp               = self.mask_temp_start

        # ---------- Encoders / Heads ----------
        # Causal/selector encoder to get node embeddings Z for masks
        self.causal_encoder = GINEncoderWithEdgeWeight(
            num_features, hidden_dim, num_layers, dropout, train_eps=True
        )

        # Classifier encoder (masked pass)
        self.classifier_encoder = GINEncoderWithEdgeWeight(
            num_features, hidden_dim, num_layers, dropout, train_eps=True
        )

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

    def forward(self, data, target=None):
        """
        Returns:
            dict with keys:
              h_stable, h_orig, node_masks, edge_masks, feat_masks, expert_logits,
              loss_total_list, loss_ce_list, loss_reg_list, loss_sem_list, loss_str_list, loss_div, rho
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Base embeddings used to produce masks
        Z = self.causal_encoder(x, edge_index, batch=batch)
        edge_feat = torch.cat([Z[edge_index[0]], Z[edge_index[1]]], dim=1)

        node_masks, edge_masks, feat_masks = [], [], []
        expert_logits, h_stable_list = [], []

        for k in range(self.num_experts):
            node_mask_logits = self.expert_node_masks[k](Z)
            edge_mask_logits = self.expert_edge_masks[k](edge_feat)
            feat_mask_logits = self.expert_feat_masks[k](Z)

            node_mask = self._hard_concrete_mask(node_mask_logits, temperature=self._mask_temp)
            edge_mask = self._hard_concrete_mask(edge_mask_logits, temperature=self._mask_temp)
            feat_mask = self._hard_concrete_mask(feat_mask_logits, temperature=self._mask_temp)

            node_masks.append(node_mask)
            edge_masks.append(edge_mask)
            feat_masks.append(feat_mask)

            # Apply masks
            masked_x = x * node_mask * feat_mask  # (N, D)
            edge_weight = edge_mask.view(-1)

            masked_Z  = self.classifier_encoder(masked_x, edge_index, batch=batch, edge_weight=edge_weight)
            h_stable  = global_mean_pool(masked_Z, batch)
            logit     = self.expert_classifiers[k](h_stable)

            h_stable_list.append(h_stable)
            expert_logits.append(logit)

        node_masks    = torch.stack(node_masks, dim=1)        # (N, K, 1)
        edge_masks    = torch.stack(edge_masks, dim=1)        # (E, K, 1)
        feat_masks    = torch.stack(feat_masks, dim=1)        # (N, K, D)
        expert_logits = torch.stack(expert_logits, dim=1)     # (B, K, C)
        h_stable_list = torch.stack(h_stable_list, dim=1)     # (B, K, H)
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
            ce_list, reg_list, sem_list, str_list, tot_list = [], [], [], [], []

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

                # Semantic invariance: VIB on h_C + LA on residual
                sem = self._semantic_invariance_loss(h_masked=hC_k, labels=target, h_orig=h_orig)
                sem_list.append(sem)

                # Structural invariance currently OFF (keep explicit zero)
                str_loss = hC_k.new_tensor(0.0)
                str_list.append(str_loss)

                # Use live scheduled weight for structure (even though str_loss==0.0)
                total = (self.weight_ce * ce +
                         self.weight_reg * reg +
                         self._weight_str_live * str_loss +
                         self.weight_sem * sem)
                tot_list.append(total)

            # Diversity across experts' masks
            div_loss = self._diversity_loss(node_masks, edge_masks, feat_masks,
                                            node_batch=batch, edge_batch=batch[edge_index[0]])

            out.update({
                'loss_total_list': torch.stack(tot_list),   # (K,)
                'loss_ce_list':    torch.stack(ce_list),
                'loss_reg_list':   torch.stack(reg_list),
                'loss_sem_list':   torch.stack(sem_list),
                'loss_str_list':   torch.stack(str_list),
                'loss_div':        div_loss,
            })

        return out

    # ----------------- Internals -----------------
    def _hard_concrete_mask(self, logits, temperature=0.1):
        if self.training:
            u = torch.rand_like(logits)
            g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            y_soft = torch.sigmoid((logits + g) / max(temperature, 1e-6))
        else:
            y_soft = torch.sigmoid(logits / max(temperature, 1e-6))
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

    def _diversity_loss(self, node_masks, edge_masks, feat_masks, node_batch, edge_batch, tau=0.10, eps=1e-8):
        """
        Per-graph correlation hinge on experts' masks (lower correlation => higher diversity).
        """
        def _per_graph_abs_corr_hinge(V: torch.Tensor, bidx: torch.Tensor) -> torch.Tensor:
            B = int(bidx.max().item()) + 1
            vals = []
            for g in range(B):
                sel = (bidx == g)
                if sel.sum() < 2:
                    continue
                X = V[sel]
                X = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, unbiased=False, keepdim=True) + eps)
                C = (X.t() @ X) / X.size(0)
                K_ = C.size(0)
                M = C.abs()
                off_mean = (M.sum() - M.diag().sum()) / max(K_ * (K_ - 1), 1)
                vals.append(F.relu(off_mean - tau))
            return torch.stack(vals).mean() if vals else V.new_tensor(0.0)

        parts = []
        parts.append(_per_graph_abs_corr_hinge(node_masks.squeeze(-1), node_batch))
        parts.append(_per_graph_abs_corr_hinge(edge_masks.squeeze(-1), edge_batch))
        parts.append(_per_graph_abs_corr_hinge(feat_masks.mean(dim=2), node_batch))
        return torch.stack(parts).mean()

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

    def _semantic_invariance_loss(
        self,
        h_masked: torch.Tensor,   # h_C (B, H)
        labels: torch.Tensor,     # (B,)
        h_orig: Optional[torch.Tensor] = None,  # (B, H)
        normalize: bool = False
    ) -> torch.Tensor:
        """
        VIB on h_C + LA on residual h_S = (h_orig - h_C) to discourage label signal in spur part.
        """
        device = h_masked.device
        B, H = h_masked.shape

        # ----- VIB (q(z|h_C)) -----
        if self.ib_mu_head_sem is None:
            self.ib_mu_head_sem = nn.Linear(H, H).to(device)
            self.ib_logvar_head_sem = nn.Linear(H, H).to(device)
            nn.init.constant_(self.ib_logvar_head_sem.weight, 0.0)
            nn.init.constant_(self.ib_logvar_head_sem.bias, -5.0)  # small var init

        mu     = self.ib_mu_head_sem(h_masked)
        logvar = self.ib_logvar_head_sem(h_masked)
        kl_ps  = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1)   # [B]
        kl_ps  = torch.clamp(kl_ps - self.ib_free_bits, min=0.0)
        vib = self._beta_ib * kl_ps.mean()

        # ----- LA on residual h_S -----
        la = h_masked.new_tensor(0.0)
        if self._lambda_L > 0.0:
            assert h_orig is not None, "h_orig required for LA on residual"
            h_S = (h_orig.detach() - h_masked).detach()
            logits_y_spur = self.lbl_head_spur_sem(grad_reverse(h_S, self._lambda_L))
            la = F.cross_entropy(logits_y_spur, labels) * self._lambda_L

        return vib + la
