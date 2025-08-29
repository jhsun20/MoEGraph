import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn import BatchNorm as PYG_BatchNorm  # for isinstance checks
from models.gnn_models import GINEncoderWithEdgeWeight, ExpertClassifier
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from contextlib import contextmanager
import math


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

        self.num_features = dataset_info['num_features']
        self.num_classes  = dataset_info['num_classes']
        self.metric = dataset_info['metric']
        if self.metric == "Accuracy" and self.num_classes == 1:
            self.num_classes = 2

        mcfg         = config.get('model', {})
        hidden_dim   = mcfg['hidden_dim']
        num_layers   = mcfg['num_layers']
        dropout      = mcfg['dropout']

        # --- BN freeze controls (optional; set via config['model']) ---
        mcfg_bn = config.get('model', {})
        self.global_pooling = mcfg_bn.get('global_pooling', 'mean')

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
        self.weight_str = float(mcfg['weight_str'])

        # ---------- Schedulers (mask temp, LA ramp, IB ramp, STR ramp) ----------
        # LA (label adversary) target
        self.lambda_L_end = float(mcfg.get('lambda_L_end', self.weight_la))
        self.adv_warmup_epochs = int(mcfg.get('adv_warmup_epochs', 5))
        self.adv_ramp_epochs   = int(mcfg.get('adv_ramp_epochs', 5))
        self._lambda_L = 0.0  # live value

        # Environment inference + EA scheduling (mirrors your LA/IB ramps)
        self.env_inference = bool(mcfg.get('env_inference', True))
        self.lambda_E_end = float(mcfg.get('lambda_E_end', self.weight_ea))
        self.ea_warmup_epochs = int(mcfg.get('ea_warmup_epochs', self.adv_warmup_epochs))
        self.ea_ramp_epochs   = int(mcfg.get('ea_ramp_epochs',   self.adv_ramp_epochs))
        self._lambda_E = 0.0  # live value updated in set_epoch

        # Structural invariance (live) weight schedule (even while str_loss=0.0 for now)
        self.weight_str_end       = float(mcfg.get('weight_str_end', self.weight_str))
        self.strinv_warmup_epochs = int(mcfg.get('strinv_warmup_epochs', self.adv_warmup_epochs))
        self.strinv_ramp_epochs   = int(mcfg.get('strinv_ramp_epochs', self.adv_ramp_epochs))
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
            self.num_features, hidden_dim, num_layers, dropout, train_eps=True, global_pooling='none'
        )

        # Per-expert maskers
        # self.expert_node_masks = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(hidden_dim, hidden_dim),
        #         nn.ReLU(),
        #         nn.Linear(hidden_dim, 1)
        #     )
        #     for _ in range(self.num_experts)
        # ])

        self.expert_edge_masks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(self.num_experts)
        ])

        # Classifier encoder (masked pass)
        self.classifier_encoders = nn.ModuleList([
            GINEncoderWithEdgeWeight(
                self.num_features, hidden_dim, num_layers, dropout, train_eps=True, global_pooling=self.global_pooling)
        for _ in range(self.num_experts)
        ])

        # Per-expert classifiers
        self.expert_classifiers_causal = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_classes)
            )
            for _ in range(self.num_experts)
        ])

        self.expert_classifiers_spur = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_classes)
            )
            for _ in range(self.num_experts)
        ])

        self.num_envs = dataset_info['num_envs']
        self.env_classifier_encoders = nn.ModuleList([
            GINEncoderWithEdgeWeight(
                self.num_features, hidden_dim, num_layers, dropout, train_eps=True, global_pooling=self.global_pooling)
        for _ in range(self.num_experts)
        ])

        self.expert_env_classifiers_causal = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_envs)
            )
            for _ in range(self.num_experts)
        ])
        self.expert_env_classifiers_spur = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_envs)
            )
            for _ in range(self.num_experts)
        ])

        self.spur_classifier_encoders = nn.ModuleList([
            GINEncoderWithEdgeWeight(
                self.num_features, hidden_dim, num_layers, dropout, train_eps=True, global_pooling=self.global_pooling)
        for _ in range(self.num_experts)
        ])

        # self.expert_classifiers_spur = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(hidden_dim, hidden_dim),
        #         nn.ReLU(),
        #         nn.Linear(hidden_dim, self.num_classes)
        #     )
        #     for _ in range(self.num_experts)
        # ])

        # --- LECI-style tiny GINs for the LA (spur/complement) branch ---
        # self.label_classifier_encoder = GINEncoderWithEdgeWeight(
        #     self.num_features, hidden_dim, num_layers, dropout, train_eps=True
        # )

        # Label-adversarial head on semantic residual h_S
        # self.expert_label_classifiers = nn.ModuleList([
        #     nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, self.num_classes)) for _ in range(self.num_experts)
        # ])



        # Keep-rate priors (trainable) per expert for regularization
        self.rho_node = nn.Parameter(torch.empty(self.num_experts).uniform_(0.3, 0.5))
        self.rho_edge = nn.Parameter(torch.empty(self.num_experts).uniform_(0.4, 0.7))

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

        # Structural invariance (live) weight ramp (after warmup)
        if epoch < self.strinv_warmup_epochs:
            self._weight_str_live = 0.0
        else:
            p = min((epoch - self.strinv_warmup_epochs) / max(self.strinv_ramp_epochs, 1), 1.0)
            self._weight_str_live = float(self.weight_str_end * p)

        if self.verbose and epoch % 10 == 0:
            print(f"[Experts.set_epoch] epoch={epoch} | temp={self._mask_temp:.3f} "
                  f"| lambda_L={self._lambda_L:.4f} "
                  f"| w_str_live={self._weight_str_live:.4f}")

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

        node_masks, edge_masks = [], []
        expert_logits, h_stable_list = [], []

        is_eval = not self.training

        for k in range(self.num_experts):
            # node_mask_logits = self.expert_node_masks[k](Z)
            edge_mask_logits = self.expert_edge_masks[k](edge_feat)

            
            # node_mask = self._hard_concrete_mask(node_mask_logits, self._mask_temp, is_eval=is_eval)
            edge_mask = self._hard_concrete_mask(edge_mask_logits, self._mask_temp, is_eval=is_eval)

            # Keep edges as weighted by edge_mask (0 removes, 1 keeps; soft works too)
            src, dst = edge_index                    # (2, E) — must be Long
            e_on = edge_mask.view(-1).to(            # (E,)
                dtype=torch.float32, device=x.device
            )

            N = x.size(0)
            node_weight = e_on.new_zeros(N)          # (N,) float — matches e_on
            node_weight.index_add_(0, src, e_on)
            node_weight.index_add_(0, dst, e_on)

            # choose hard or soft
            node_weight = (node_weight > 0).float()  # or: node_weight = node_weight.clamp(max=1.0)

            # logs use what you actually applied
            node_masks.append(node_weight.view(-1, 1))
            edge_masks.append(e_on.view(-1, 1))                    

            # Apply masks to features/graph
            masked_x   = x * node_weight.view(-1, 1)           # zero-out nodes with no kept incident edges
            edge_weight = e_on                                  # (E,)

            h_stable = self.classifier_encoders[k](masked_x, edge_index, edge_weight=edge_weight, batch=batch)

            # Classify with your existing per-expert classifier
            logit = self.expert_classifiers_causal[k](h_stable)

            h_stable_list.append(h_stable)
            expert_logits.append(logit)

        node_masks    = torch.stack(node_masks, dim=1)        # (N, K, 1)
        edge_masks    = torch.stack(edge_masks, dim=1)        # (E, K, 1)
        expert_logits = torch.stack(expert_logits, dim=1)     # (B, K, C)
        h_stable_list = torch.stack(h_stable_list, dim=1)     # (B, K, H)

        if self.global_pooling == 'mean':
            h_orig = global_mean_pool(Z, batch)            # (B, H)
        elif self.global_pooling == 'sum':
            h_orig = global_add_pool(Z, batch)            # (B, H)  

        out = {
            'h_stable': h_stable_list,
            'h_orig':   h_orig,
            'node_masks': node_masks,
            'edge_masks': edge_masks,
            'expert_logits': expert_logits,
            'rho': [self.rho_node, self.rho_edge]
        }

        if target is not None:
            ce_list, reg_list, str_list, tot_list, la_list, ea_list = [], [], [], [], [], []

            # NEW: per-sample (B,K) collectors
            B = expert_logits.size(0)
            K = self.num_experts
            ce_ps  = []   # each (B,)
            la_ps  = []
            ea_ps  = []
            str_ps = []

            for k in range(self.num_experts):
                logits_k = expert_logits[:, k, :]
                hC_k     = h_stable_list[:, k, :]
                node_mask_k = node_masks[:, k, :]
                edge_mask_k = edge_masks[:, k, :]

                # ----- CE (per-sample + scalar) -----
                ce_vec = F.cross_entropy(logits_k, target.view(-1).long(), reduction='none') * self.weight_ce  # (B,)
                ce = ce_vec.mean()
                ce_ps.append(ce_vec)
                ce_list.append(ce)

                # Mask regularization (per-graph keep-rate prior)
                reg = self._mask_reg(
                    node_masks[:, k, :], edge_masks[:, k, :],
                    node_batch=batch, edge_batch=batch[edge_index[0]], expert_idx=k
                ) * self.weight_reg
                reg_list.append(reg)

                # Defaults for per-sample terms when the path is off
                la_vec = hC_k.new_zeros(B)   # (B,)
                ea_vec = hC_k.new_zeros(B)   # (B,)
                str_vec= hC_k.new_zeros(B)   # (B,)

                if not is_eval:
                    if self._lambda_L > 0:
                        h_spur, edge_weight_spur = self._encode_complement_subgraph(
                                data=data,
                                node_mask=node_mask_k,           # (N,1) or (N,)
                                edge_mask=edge_mask_k.view(-1),  # (E,)
                                encoder=self.classifier_encoders[k],
                                symmetrize=False
                        )
                                
                        la_vec = self._spur_loss(
                            h_masked=hC_k,
                            labels=target,
                            h_spur=h_spur,
                            expert_idx=k,
                            edge_index=edge_index,
                            edge_weight=edge_weight_spur,
                            batch=batch,
                            cf_mode="entropy",              # keep your mode
                            reduction="none",             # <-- per-sample
                        ) * self._lambda_L                                  # (B,)
                        la = la_vec.mean()                                  # (B,) 
                    else:
                        la = hC_k.new_tensor(0.0)
                        B = hC_k.size(0)
                        la_vec = hC_k.new_zeros(B)

                    # ----- EA / STR (env) with per-sample signals -----
                    if self._lambda_E > 0:
                        # 1) STR (spur env) — per-sample CE on env classifier fed the complement (spur) graph
                        h_spur_env, edge_weight_spur_env = self._encode_complement_subgraph(
                            data=data,
                            node_mask=node_mask_k,            # (N,1) or (N,)
                            edge_mask=edge_mask_k.view(-1),   # (E,)
                            encoder=self.env_classifier_encoders[k],
                            symmetrize=False
                        )
                        spur_env_logits = self.expert_env_classifiers_spur[k](h_spur_env)  # (B, |E|)

                        # Per-sample STR vector
                        str_vec = F.cross_entropy(
                            spur_env_logits, env_labels.view(-1).long(), reduction="none"
                        ) * self.weight_str                     # (B,)
                        # Scalar for your existing loss accounting
                        str_loss = str_vec.mean()

                        # 2) EA (stable env) — per-sample via _causal_loss(..., reduction="none")
                        masked_x_k    = x * node_mask_k
                        node_weight_k = node_mask_k.view(-1)
                        edge_weight_k = edge_mask_k.view(-1)

                        h_stable_env_k = self.env_classifier_encoders[k](
                            masked_x_k, edge_index, edge_weight=edge_weight_k, batch=batch, node_weight=node_weight_k
                        )                                         # (B, H_e)

                        ea_vec = self._causal_loss(
                            h_masked=hC_k,
                            h_ea=h_stable_env_k,
                            expert_idx=k,
                            env_labels=env_labels,
                            edge_index=edge_index,
                            edge_weight=edge_weight_k,
                            batch=batch,
                            cf_mode="entropy",
                            reduction="none",                     # <-- per-sample
                        ) * self._lambda_E                        # (B,)
                        ea = ea_vec.mean()      # scalar for logs

                    else:
                        # off-path defaults
                        ea       = hC_k.new_tensor(0.0)
                        str_loss = hC_k.new_tensor(0.0)
                        B        = hC_k.size(0)
                        ea_vec   = hC_k.new_zeros(B)              # (B,)
                        str_vec  = hC_k.new_zeros(B)              # (B,)

                else:
                    la = hC_k.new_tensor(0.0)
                    ea = hC_k.new_tensor(0.0)
                    str_loss = hC_k.new_tensor(0.0)


                # Append scalars (existing logs)
                la_list.append(la)
                ea_list.append(ea)
                str_list.append(str_loss)

                # Append per-sample vectors to collectors
                la_ps.append(la_vec)
                ea_ps.append(ea_vec)
                str_ps.append(str_vec)

                # Use live scheduled weight for structure (even though str_loss==0.0)
                total = (ce + reg + str_loss + la + ea)
                tot_list.append(total)


            # Stack per-sample into (B,K)
            ce_ps  = torch.stack(ce_ps,  dim=1)   # (B,K)
            la_ps  = torch.stack(la_ps,  dim=1)   # (B,K)
            ea_ps  = torch.stack(ea_ps,  dim=1)   # (B,K)
            str_ps = torch.stack(str_ps, dim=1)   # (B,K)

            # Diversity across experts' masks
            div_loss = self.weight_div * self._diversity_loss(node_masks, edge_masks,
                                            node_batch=batch, edge_batch=batch[edge_index[0]])

            out.update({
                'loss_total_list': torch.stack(tot_list),   # (K,)
                'loss_ce_list':    torch.stack(ce_list),    # (K,)
                'loss_reg_list':   torch.stack(reg_list),   # (K,)
                'loss_la_list':    torch.stack(la_list),    # (K,)
                'loss_ea_list':    torch.stack(ea_list),    # (K,)
                'loss_str_list':   torch.stack(str_list),   # (K,)
                'loss_div':        div_loss,
                # NEW: per-sample (B,K) matrices
                'per_sample': {
                    'ce':  ce_ps,
                    'la':  la_ps,
                    'ea':  ea_ps,
                    'str': str_ps,
                }
            })

        return out

    # ----------------- Internals -----------------
    def _hard_concrete_mask(self, logits, temperature=0.1, is_eval=False):
        # choose a temperature for eval that won't saturate
        T = max(temperature, self.eval_mask_temp_floor) if is_eval else max(temperature, 1e-6)

        if self.training:
            u = torch.rand_like(logits)
            g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            y_soft = torch.sigmoid((logits + g) / 0.1)
        else:
            u = torch.rand_like(logits)
            g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            y_soft = torch.sigmoid((logits + g) / 0.1)

        y_hard = (y_soft > 0.5).float()
        return y_hard + (y_soft - y_soft.detach())

    def _ce(self, pred, target, use_weights=False):
        #print(f"pred: {pred.shape}, target: {target.shape}")
        if use_weights:
            C = pred.size(1)
            counts = torch.bincount(target, minlength=C).float()
            counts[counts == 0] = 1.0
            w = (1.0 / counts)
            w = w / w.sum()
            return F.cross_entropy(pred, target, weight=w.to(pred.device))
        return F.cross_entropy(pred, target)
    
    def _reg(self, pred, target):
        return F.l1_loss(pred, target)
    
    def _diversity_loss(
        self,
        node_masks: torch.Tensor,   # [N_nodes, K, 1] or [N_nodes, K]
        edge_masks: torch.Tensor,   # [N_edges, K, 1] or [N_edges, K]
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
        tau_cov  = 0.40   # require at least ~0.6 union coverage
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

        # -------- compute components per modality --------
        n_corr = _per_graph_abs_corr_hinge(N, node_batch)
        e_corr = _per_graph_abs_corr_hinge(E, edge_batch)
        corr   = torch.stack([n_corr, e_corr]).mean()

        n_uo = _union_overlap_terms(N, node_batch)
        e_uo = _union_overlap_terms(E, edge_batch)
        uo   = torch.stack([n_uo, e_uo]).mean()

        # final combined diversity loss
        # print(f"corr: {corr}, uo: {uo}")
        # print(f"w_corr: {w_corr}, w_uo: {w_uo}")
        # print(f"added loss: {w_corr * corr + w_uo * uo}")
        loss = w_corr * corr + w_uo * uo
        return loss

    def _mask_reg(self, node_mask, edge_mask, node_batch, edge_batch, expert_idx: int,
                  use_fixed_rho: bool = False, fixed_rho_vals: tuple = (0.5, 0.5)):
        if use_fixed_rho:
            rho_node, rho_edge = [float(min(max(v, 0.0), 1.0)) for v in fixed_rho_vals]
        else:
            # rho_node = torch.clamp(self.rho_node[expert_idx], 0.3, 0.5)
            rho_edge = torch.clamp(self.rho_edge[expert_idx], 0.2, 1.0)

        def per_graph_keep(mask_vals, batch_idx):
            G = batch_idx.max().item() + 1
            keep = torch.zeros(G, device=mask_vals.device)
            cnt  = torch.zeros(G, device=mask_vals.device)
            keep.scatter_add_(0, batch_idx, mask_vals.squeeze())
            cnt.scatter_add_(0, batch_idx, torch.ones_like(mask_vals.squeeze()))
            return keep / (cnt + 1e-8)

        node_keep_pg = per_graph_keep(node_mask, node_batch)
        edge_keep_pg = per_graph_keep(edge_mask, edge_batch)

        #return ((node_keep_pg - rho_node) ** 2).mean() + ((edge_keep_pg - rho_edge) ** 2).mean()
        return ((edge_keep_pg - rho_edge) ** 2).mean()

    def _causal_loss(
        self,
        h_masked: torch.Tensor,
        h_ea:   Optional[torch.Tensor] = None,
        expert_idx: Optional[int] = None,
        env_labels: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        *,
        cf_mode: str = "revce",
        cf_weights: Optional[dict] = None,
        tau_prob: float = 0.3,
        tau_logit: float = 0.0,
        reduction: str = "mean",                # <-- NEW
    ) -> torch.Tensor:
        """
        Causal->env CF objective. Default is adversarial CE (reverse-CE behavior via GRL).
        For entropy/KL-style CF, use the same env head as the main EA path, frozen, no GRL.
        """
        if h_ea is None:
            raise ValueError("h_ea (env features from G_C) must be provided")

        if cf_mode == "revce":
            env_tgt = env_labels.view(-1).long().to(h_ea.device)
            h_ea_adv = grad_reverse(h_ea, self._lambda_E)
            logits_E = self.expert_env_classifiers_causal[expert_idx](h_ea_adv)  # (B, |E|)
            if reduction == "none":
                return F.cross_entropy(logits_E, env_tgt, reduction="none")      # (B,)
            return F.cross_entropy(logits_E, env_tgt)                             # scalar

        with self._frozen_params(self.expert_env_classifiers_spur[expert_idx], freeze_bn_running_stats=True):
            logits_env_cf = self.expert_env_classifiers_spur[expert_idx](h_ea)    # (B, |E|)

        # map cf_mode -> weights
        w = dict(w_entropy=0.0, w_kl_uniform=0.0, w_true_hinge_prob=0.0, w_true_hinge_logit=0.0, w_one_minus_true=0.0)
        if cf_mode == "entropy":
            w["w_entropy"] = 1.0
        elif cf_mode == "kl_uniform":
            w["w_kl_uniform"] = 1.0
        elif cf_mode == "hinge_prob":
            w["w_true_hinge_prob"] = 1.0
        elif cf_mode == "hinge_logit":
            w["w_true_hinge_logit"] = 1.0
        elif cf_mode == "one_minus_true":
            w["w_one_minus_true"] = 1.0
        elif cf_mode == "mix":
            if cf_weights:
                for k in w.keys():
                    if k in cf_weights:
                        w[k] = float(cf_weights[k])
            else:
                w.update(dict(w_entropy=0.8, w_true_hinge_prob=0.2))
        else:
            raise ValueError(f"Unknown cf_mode={cf_mode}")

        # we still need env labels for hinges on the "true" env class (when used)
        env_tgt = env_labels.view(-1).long().to(logits_env_cf.device)

        nec = self._cf_label_necessity_losses(
            logits_drop=logits_env_cf,
            target=env_tgt,
            num_classes=self.num_envs,
            **w,
            tau_prob=tau_prob,
            tau_logit=tau_logit,
            reduction=reduction,                 # <-- propagate
        )
        return nec["loss_nec"]


    def _spur_loss(
        self,
        h_masked: torch.Tensor,
        labels: torch.Tensor,
        h_spur: Optional[torch.Tensor] = None,
        expert_idx: Optional[int] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        *,
        cf_mode: str = "revce",
        cf_weights: Optional[dict] = None,
        tau_prob: float = 0.3,
        tau_logit: float = 0.0,
        reduction: str = "mean",               # <-- NEW
    ) -> torch.Tensor:
        """
        Spurious->label CF objective. Default is reverse-CE (as before).
        For entropy/KL-style CF, use the same label head as the main path, frozen.
        """
        device = h_masked.device
        if h_spur is None:
            raise ValueError("h_spur must be provided: encode complement subgraph with la_encoders[k]")

        # normalize label shape/type
        y = labels.view(-1).long()
        if cf_mode == "revce":
            h_spur_adv = grad_reverse(h_spur, self._lambda_L)
            logits_y_spur = self.expert_classifiers_spur[expert_idx](h_spur_adv)  # (B,C)
            if reduction == "none":
                return F.cross_entropy(logits_y_spur, y, reduction="none")        # (B,)
            return F.cross_entropy(logits_y_spur, y)                               # scalar

        with self._frozen_params(self.expert_classifiers_causal[expert_idx], freeze_bn_running_stats=True):
            logits_cf = self.expert_classifiers_causal[expert_idx](h_spur)         # (B,C) or (B,1)

        # map cf_mode -> weights for _cf_label_necessity_losses
        w = dict(w_entropy=0.0, w_kl_uniform=0.0, w_true_hinge_prob=0.0, w_true_hinge_logit=0.0, w_one_minus_true=0.0)
        if cf_mode == "entropy":
            w["w_entropy"] = 1.0
        elif cf_mode == "kl_uniform":
            w["w_kl_uniform"] = 1.0
        elif cf_mode == "hinge_prob":
            w["w_true_hinge_prob"] = 1.0
        elif cf_mode == "hinge_logit":
            w["w_true_hinge_logit"] = 1.0
        elif cf_mode == "one_minus_true":
            w["w_one_minus_true"] = 1.0
        elif cf_mode == "mix":
            if cf_weights:
                for k in w.keys():
                    if k in cf_weights:
                        w[k] = float(cf_weights[k])
            else:
                # sensible default mix
                w.update(dict(w_entropy=0.7, w_true_hinge_prob=0.3))
        else:
            raise ValueError(f"Unknown cf_mode={cf_mode}")

        nec = self._cf_label_necessity_losses(
            logits_drop=logits_cf,
            target=y,
            num_classes=self.num_classes,
            **w,
            tau_prob=tau_prob,
            tau_logit=tau_logit,
            reduction=reduction,                 # <-- propagate
        )
        return nec["loss_nec"]

    
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

    def _cf_label_necessity_losses(
        self,
        logits_drop: torch.Tensor,   # (B,C) or (B,1)
        target: torch.Tensor,        # (B,) or (B,1)
        num_classes: int,
        # toggles/weights (all >= 0)
        w_entropy: float = 0.0,
        w_kl_uniform: float = 0.0,
        w_true_hinge_prob: float = 0.0,
        w_true_hinge_logit: float = 0.0,
        w_one_minus_true: float = 0.0,
        # thresholds / margins
        tau_prob: float = 0.3,
        tau_logit: float = 0.0,
        reduction: str = "mean",     # <-- accept "mean" | "sum" | "none"
        eps: float = 1e-8,
    ):
        """
        Returns dict with individual necessity components (all >= 0) and 'loss_nec' (weighted sum).
        NOTE: 'w_entropy' now weights an entropy-deficit term (logC - H), which is >= 0.
            'w_entropy' and 'w_kl_uniform' are redundant surrogates; usually enable ONE.
        """
        # ---- normalize shapes ----
        if logits_drop.dim() == 1:
            logits_drop = logits_drop.unsqueeze(1)
        B, C_infer = logits_drop.shape
        C = num_classes if num_classes is not None else C_infer
        binary_single_logit = (C == 1 and logits_drop.size(1) == 1)

        # targets
        if target.dim() == 2 and target.size(1) == 1:
            target = target[:, 0]
        tgt_long = target.long().view(-1)

        # probabilities on drop (and a multiclass-style logits tensor for hinges)
        if binary_single_logit:
            p_pos = torch.sigmoid(logits_drop.squeeze(-1)).clamp(eps, 1 - eps)    # (B,)
            p_drop = torch.stack([1 - p_pos, p_pos], dim=-1)                      # (B,2)
            logits_drop_mc = torch.stack([torch.zeros_like(p_pos), logits_drop.squeeze(-1)], dim=-1)  # (B,2)
            C_eff = 2
            tgt_for_mc = tgt_long.clamp(0, 1)
        else:
            p_drop = F.softmax(logits_drop, dim=-1).clamp_min(eps)                # (B,C)
            logits_drop_mc = logits_drop
            C_eff = p_drop.size(1)
            tgt_for_mc = tgt_long
        # --- components (ALL NON-NEGATIVE) ---
        ent = -(p_drop * p_drop.log()).sum(dim=-1)      # (B,)
        logC = math.log(C_eff)
        loss_entropy = (logC - ent).clamp_min(0.0)      # (B,)

        u = p_drop.new_full((B, C_eff), 1.0 / C_eff)
        loss_kl_uniform = (p_drop * (p_drop.log() - u.log())).sum(dim=-1)  # (B,)

        idx = torch.arange(B, device=logits_drop_mc.device)
        py_drop = p_drop[idx, tgt_for_mc]               # (B,)
        loss_true_hinge_prob  = torch.relu(py_drop - tau_prob)             # (B,)

        ly_drop = logits_drop_mc[idx, tgt_for_mc]       # (B,)
        loss_true_hinge_logit = torch.relu(ly_drop - tau_logit)            # (B,)

        loss_one_minus_true = -(1 - py_drop).clamp_min(eps).log()          # (B,)

        # ---- apply reduction at the very end ----
        def _reduce(v):
            if reduction == "mean": return v.mean()
            if reduction == "sum":  return v.sum()
            if reduction == "none": return v
            raise ValueError(f"reduction must be 'mean'|'sum'|'none', got {reduction}")

        loss_entropy        = _reduce(loss_entropy)
        loss_kl_uniform     = _reduce(loss_kl_uniform)
        loss_true_hinge_prob= _reduce(loss_true_hinge_prob)
        loss_true_hinge_logit=_reduce(loss_true_hinge_logit)
        loss_one_minus_true = _reduce(loss_one_minus_true)

        loss_nec = (
            w_entropy         * loss_entropy +
            w_kl_uniform      * loss_kl_uniform +
            w_true_hinge_prob * loss_true_hinge_prob +
            w_true_hinge_logit* loss_true_hinge_logit +
            w_one_minus_true  * loss_one_minus_true
        )

        return {
            'loss_nec': loss_nec,
            'nec_entropy': loss_entropy,
            'nec_kl_uniform': loss_kl_uniform,
            'nec_true_hinge_prob': loss_true_hinge_prob,
            'nec_true_hinge_logit': loss_true_hinge_logit,
            'nec_one_minus_true': loss_one_minus_true,
            'binary_single_logit': binary_single_logit,
        }

    
    def _encode_complement_subgraph(
        self,
        data,
        node_mask: torch.Tensor,   # (N,) or (N,1)
        edge_mask: torch.Tensor,   # (E,) or (E,1)
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

        # --- complement masks ---
        node_m_S = 1.0 - node_m                              # (N,1)
        edge_m_S = 1.0 - edge_m                              # (E,)
        node_w_S = node_m_S.view(-1)  

        # --- apply to features (node-wise * feature-wise) ---
        x_spur = (x * node_m_S) # * feat_m_S                   # (N,F)

        # --- optional symmetrization of edge weights for undirected graphs ---
        ei, ew_spur = edge_index, edge_m_S
        if symmetrize:
            ei, ew_spur = to_undirected(ei, ew_spur, reduce='mean')

        # --- encode + pool ---
        Z_spur = encoder(x_spur, ei, batch=batch, edge_weight=ew_spur, node_weight=node_w_S)
        return Z_spur, ew_spur
    
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

    @contextmanager
    def _frozen_params(self, module: nn.Module, freeze_bn_running_stats: bool = True):
        """
        Temporarily sets requires_grad=False for all params in `module` so its
        weights won't update, while still allowing gradients to flow to the inputs.
        If `freeze_bn_running_stats=True`, switch the module to eval() for the block
        so BN running stats (and dropout) are not affected by the spur view.
        """
        was_training = module.training
        if freeze_bn_running_stats:
            module.eval()
        saved = [p.requires_grad for p in module.parameters()]
        for p in module.parameters():
            p.requires_grad_(False)
        try:
            yield
        finally:
            for p, rg in zip(module.parameters(), saved):
                p.requires_grad_(rg)
            module.train(was_training)


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


def _check_graph(x, edge_index, edge_weight=None, batch=None, where=""):
    assert x.dtype in (torch.float32, torch.float64), f"{where}: x dtype {x.dtype}"
    assert edge_index.dtype == torch.long, f"{where}: edge_index dtype {edge_index.dtype}"
    N = x.size(0)
    E = edge_index.size(1)
    assert edge_index.dim() == 2 and edge_index.size(0) == 2, f"{where}: edge_index shape {edge_index.shape}"
    max_id = int(edge_index.max().item()) if E > 0 else -1
    min_id = int(edge_index.min().item()) if E > 0 else 0
    assert 0 <= min_id, f"{where}: negative node id in edge_index"
    assert max_id < N, f"{where}: node id {max_id} >= N={N}"
    if edge_weight is not None:
        assert edge_weight.dim() == 1 or (edge_weight.dim()==2 and edge_weight.size(-1)==1), f"{where}: edge_weight shape {edge_weight.shape}"
        assert edge_weight.size(0) == E, f"{where}: |edge_weight|={edge_weight.size(0)} != E={E}"
    if batch is not None:
        assert batch.size(0) == N, f"{where}: |batch|={batch.size(0)} != N={N}"
