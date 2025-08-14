import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gnn_models import GCN, GIN, GraphSAGE, GINEncoderWithEdgeWeight

from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data
import math
from typing import Optional, Tuple

# -----------------------
# Gradient Reversal Layer
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


# -----------------------
# Simple Torch K-Means
# -----------------------
def kmeans_torch(x: torch.Tensor, K: int, iters: int = 10, eps: float = 1e-6, seed: Optional[int] = None) -> torch.Tensor:
    """
    x: (B, d) batch of vectors
    returns labels: (B,) in {0..K-1}
    """
    B, d = x.shape
    if B < K:
        # fall back: unique labels up to B
        return torch.arange(B, device=x.device) % max(1, K)

    if seed is not None:
        g = torch.Generator(device=x.device)
        g.manual_seed(seed)
        sel = torch.randperm(B, generator=g, device=x.device)[:K]
    else:
        sel = torch.randperm(B, device=x.device)[:K]

    centers = x[sel].clone()  # (K, d)
    for _ in range(max(1, iters)):
        # assign
        # (B, K) distances
        dist = (x.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(dim=2)
        labels = dist.argmin(dim=1)  # (B,)

        # update
        new_centers = []
        moved = 0.0
        for k in range(K):
            mask = (labels == k)
            if mask.any():
                c = x[mask].mean(dim=0)
            else:
                # reinit a dead cluster
                c = x[torch.randint(B, (1,), device=x.device)].squeeze(0)
            moved += (centers[k] - c).pow(2).sum().item()
            new_centers.append(c)
        centers = torch.stack(new_centers, dim=0)

        if moved / (K * d + 1e-9) < eps:
            break
    return labels


class Experts(nn.Module):
    def __init__(self, config, dataset_info, rho=0.5):
        super().__init__()
        num_features = dataset_info['num_features']
        num_classes = dataset_info['num_classes']
        if num_classes == 1:
            if config['experiment']['debug']['verbose']:
                print("[warn] dataset_info['num_classes']=1, forcing to 2 for CE with labels {0,1}")
            num_classes = 2  # force 2 for binary CE
        hidden_dim = config['model']['hidden_dim']
        num_layers = config['model']['num_layers']
        dropout = config['model']['dropout']
        num_experts = config['model']['num_experts']
        
        self.weight_str = config['model']['weight_str']
        self.weight_sem = config['model']['weight_sem']
        self.weight_reg = config['model']['weight_reg']
        self.weight_ce = config['model']['weight_ce']
        self.verbose = config['experiment']['debug']['verbose']
        self.num_experts = num_experts

        # -------- MI hyperparams (defaults if absent) --------
        mcfg = config.get('model', {})
        self.num_envs = int(mcfg.get('num_envs', 3))
        # semantic EA/LA
        self.mi_lambda_e_sem = float(mcfg.get('mi_lambda_e_sem', 0.1))
        self.mi_lambda_l_sem = float(mcfg.get('mi_lambda_l_sem', 0.1))
        # structural EA/LA
        self.mi_mu_e_str = float(mcfg.get('mi_mu_e_str', 0.1))
        self.mi_mu_l_str = float(mcfg.get('mi_mu_l_str', 0.1))

        self.causal_encoder = GINEncoderWithEdgeWeight(num_features, hidden_dim, num_layers, dropout, train_eps=True)
        
        self.expert_node_masks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
            for _ in range(num_experts)
        ])
        
        self.expert_edge_masks = nn.ModuleList([
            nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
            for _ in range(num_experts)
        ])
        
        self.expert_feat_masks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_features))
            for _ in range(num_experts)
        ])
        
        # MAYBE ADD A DIFFERENT GNN ENCODER HERE
        self.classifier_encoder = GINEncoderWithEdgeWeight(num_features, hidden_dim, num_layers, dropout, train_eps=True)
        
        self.expert_classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_experts)
        ])

        # ---- NEW: small adversarial heads (shared across experts) ----
        # Semantic heads (on h_C and residual h_S)
        self.env_head_sem = nn.Linear(hidden_dim, self.num_envs)
        self.lbl_head_spur_sem = nn.Linear(hidden_dim, num_classes)
        # Structural heads (on s_C and residual s_S). We'll bind output dim at runtime.
        # We create placeholders and reset in first call if dims differ.
        self.env_head_str: Optional[nn.Linear] = None
        self.lbl_head_spur_str: Optional[nn.Linear] = None

        # Keep-rate priors per expert
        self.rho_node = nn.Parameter(torch.empty(num_experts).uniform_(0.2, 0.8))
        self.rho_edge = nn.Parameter(torch.empty(num_experts).uniform_(0.2, 0.8))
        self.rho_feat = nn.Parameter(torch.empty(num_experts).uniform_(0.2, 0.8))

    def forward(self, data, target=None, embeddings_by_env=None, labels_by_env=None):
        """
        Forward pass with integrated loss computation for multiple experts.

        Returns:
            output: dict with keys:
                'logits', 'h_stable', 'h_orig',
                'node_masks', 'edge_masks', 'feat_masks', 'expert_logits',
                'loss_total', 'loss_ce', 'loss_reg', 'loss_sem', 'loss_str'
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        Z = self.causal_encoder(x, edge_index, batch=batch)

        # build original graph for structure comparison (no features needed)
        orig_edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)
        orig_node_weight = torch.ones(x.size(0), device=x.device)

        original_data = Data(
            edge_index=edge_index,
            batch=batch,
            edge_attr=orig_edge_attr,   # unmasked edges
            node_weight=orig_node_weight # unmasked nodes
        )
        
        node_masks = []
        edge_masks = []
        feat_masks = []
        expert_logits = []
        h_stable_list = []
        
        edge_feat = torch.cat([Z[edge_index[0]], Z[edge_index[1]]], dim=1)

        for expert_idx in range(self.num_experts):
            node_mask_logits = self.expert_node_masks[expert_idx](Z)
            edge_mask_logits = self.expert_edge_masks[expert_idx](edge_feat)
            feat_mask_logits = self.expert_feat_masks[expert_idx](Z)

            node_mask = self._hard_concrete_mask(node_mask_logits, temperature=0.1)
            edge_mask = self._hard_concrete_mask(edge_mask_logits, temperature=0.1)
            feat_mask = self._hard_concrete_mask(feat_mask_logits, temperature=0.1)
            
            node_masks.append(node_mask)
            edge_masks.append(edge_mask)
            feat_masks.append(feat_mask)
            
            masked_x = x * node_mask * feat_mask  # (N, D), broadcasted elementwise

            edge_weight = edge_mask.view(-1)
            masked_Z = self.classifier_encoder(masked_x, edge_index, batch=batch, edge_weight=edge_weight)
            h_stable = global_mean_pool(masked_Z, batch)
            h_stable_list.append(h_stable)

            expert_logit = self.expert_classifiers[expert_idx](h_stable)
            expert_logits.append(expert_logit)
        
        node_masks = torch.stack(node_masks, dim=1)
        edge_masks = torch.stack(edge_masks, dim=1)
        feat_masks = torch.stack(feat_masks, dim=1)
        expert_logits = torch.stack(expert_logits, dim=1)
        h_stable_list = torch.stack(h_stable_list, dim=1)
        h_orig = global_mean_pool(Z, batch)

        self.cached_masks = {
            'node_masks': node_masks,
            'edge_masks': edge_masks,
            'feat_masks': feat_masks,
            'Z': Z,
            'h_stable_list': h_stable_list,
            'expert_logits': expert_logits,
            'edge_index': edge_index,
            'batch': batch,
            'edge_batch': batch[edge_index[0]]
        }

        output = {
            'h_stable': h_stable_list,
            'h_orig': h_orig,
            'node_masks': node_masks,
            'edge_masks': edge_masks,
            'feat_masks': feat_masks,
            'expert_logits': expert_logits,
            'loss_total_list': None,
            'loss_ce_list': None,
            'loss_reg_list': None,
            'loss_sem_list': None,
            'loss_str_list': None,
            'rho': [self.rho_node, self.rho_edge, self.rho_feat],
            'cached_masks': self.cached_masks
        }

        if target is not None:
            # Precompute once per batch: original structural vector
            # s_orig = self._rw_struct_vec_batched(
            #     Data(edge_index=original_data.edge_index,
            #          batch=original_data.batch,
            #          edge_attr=original_data.edge_attr,
            #          node_weight=original_data.node_weight),
            #     rw_max_steps=4
            # )  # (B, T)

            # Initialize structural heads lazily now that T is known
            # T = s_orig.size(1)
            # if (self.env_head_str is None) or (self.env_head_str.in_features != T):
            #     self.env_head_str = nn.Linear(T, self.num_envs).to(s_orig.device)
            #     self.lbl_head_spur_str = nn.Linear(T, self.expert_classifiers[0].out_features).to(s_orig.device)

            ce_loss_list, reg_loss_list, sem_loss_list, str_loss_list, total_loss_list = [], [], [], [], []
            
            for expert_idx in range(self.num_experts):
                expert_logit = expert_logits[:, expert_idx, :]
                expert_h_stable = h_stable_list[:, expert_idx, :]
                assert target.dtype == torch.long, f"target dtype {target.dtype}"
                num_classes = expert_logit.size(1)
                tmin, tmax = int(target.min().item()), int(target.max().item())
                assert 0 <= tmin and tmax < num_classes, f"labels out of range [0,{num_classes-1}]: min={tmin}, max={tmax}"

                ce_loss = self.compute_classification_loss(expert_logit, target)
                ce_loss_list.append(ce_loss)

                expert_node_mask = node_masks[:, expert_idx, :]
                expert_edge_mask = edge_masks[:, expert_idx, :]
                expert_feat_mask = feat_masks[:, expert_idx, :]
                reg_loss = self.compute_mask_regularization_loss(expert_node_mask, expert_edge_mask, expert_feat_mask, expert_idx, node_batch=batch, edge_batch=batch[edge_index[0]], use_fixed_rho=False)
                reg_loss_list.append(reg_loss)

                # ---- MI-augmented SEMANTIC invariance ----
                sem_loss = self.compute_semantic_invariance_loss(
                    h_masked=expert_h_stable,
                    labels=target,
                    h_orig=h_orig,
                    lambda_e=self.mi_lambda_e_sem,
                    lambda_l=self.mi_lambda_l_sem
                )
                sem_loss_list.append(sem_loss)

                # structural masked graph for THIS expert (structure-only)
                # masked_data = Data(
                #     edge_index=edge_index,
                #     batch=batch,
                #     edge_attr=expert_edge_mask.view(-1),   # (E, 1) -> (E,)
                #     node_weight=expert_node_mask.view(-1)  # (N, 1) -> (N,)
                # )

                # ---- MI-augmented STRUCTURAL invariance ----
                # str_loss = self.compute_structural_invariance_loss(
                #     data_masked=masked_data,
                #     labels=target,
                #     s_orig=s_orig,
                #     mode="randomwalk",
                #     rw_max_steps=4,
                #     graphon_bins=4,
                #     mu_e=self.mi_mu_e_str,
                #     mu_l=self.mi_mu_l_str
                # )
                str_loss = torch.tensor(0.0)
                str_loss_list.append(str_loss)

                total_loss = (self.weight_ce * ce_loss + 
                              self.weight_reg * reg_loss + 
                              self.weight_str * str_loss + 
                              self.weight_sem * sem_loss)
                total_loss_list.append(total_loss)
            
            diversity_loss = self.compute_diversity_loss(node_masks, edge_masks, feat_masks, node_batch=batch, edge_batch=batch[edge_index[0]])

            output.update({
                'loss_total_list': torch.stack(total_loss_list),
                'loss_ce_list': torch.stack(ce_loss_list),
                'loss_reg_list': torch.stack(reg_loss_list),
                'loss_sem_list': torch.stack(sem_loss_list),
                'loss_str_list': torch.stack(str_loss_list),
                'loss_div': diversity_loss
            })

        return output
    
    def _hard_concrete_mask(self, logits, temperature=0.1):
        """
        Sample a hard 0/1 mask from logits with straight-through gradient.
        """
        uniform_noise = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-20) + 1e-20)
        y_soft = torch.sigmoid((logits + gumbel_noise) / temperature)
        y_hard = (y_soft > 0.5).float()
        return y_hard + (y_soft - y_soft.detach())

    def compute_classification_loss(self, pred, target, use_weights=False):
        if use_weights:
            num_classes = pred.size(1)
            class_counts = torch.bincount(target, minlength=num_classes).float()
            class_counts[class_counts == 0] = 1.0
            weight = 1.0 / class_counts
            weight = weight / weight.sum()
            return F.cross_entropy(pred, target, weight=weight.to(pred.device))
        else:
            return F.cross_entropy(pred, target)
        
    def compute_diversity_loss(
            self,
            node_masks: torch.Tensor,   # (N, K, 1)
            edge_masks: torch.Tensor,   # (E, K, 1)
            feat_masks: torch.Tensor,   # (N, K, D)
            node_batch: torch.Tensor,   # (N,)
            edge_batch: torch.Tensor,   # (E,),
            tau: float = 0.10,
            eps: float = 1e-8,
        ) -> torch.Tensor:
            """
            Per-graph diversity across experts' masks (nodes/edges/features).
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
                    off_mean = (M.sum() - M.diag().sum()) / (K_ * (K_ - 1))
                    vals.append(F.relu(off_mean - tau))
                if not vals:
                    return V.new_tensor(0.0)
                return torch.stack(vals).mean()

            parts = []
            if node_masks is not None:
                parts.append(_per_graph_abs_corr_hinge(node_masks.squeeze(-1), node_batch))
            if edge_masks is not None:
                parts.append(_per_graph_abs_corr_hinge(edge_masks.squeeze(-1), edge_batch))
            if feat_masks is not None:
                parts.append(_per_graph_abs_corr_hinge(feat_masks.mean(dim=2), node_batch))

            if not parts:
                dev = node_batch.device if node_masks is None else node_masks.device
                return torch.tensor(0.0, device=dev)
            return torch.stack(parts).mean()

    def compute_mask_regularization_loss(self, node_mask, edge_mask, feat_mask, expert_idx: int, node_batch, edge_batch, use_fixed_rho: bool = False, fixed_rho_vals: tuple = (0.5, 0.5, 0.5),):
        if use_fixed_rho:
            rho_node, rho_edge, rho_feat = [
                float(min(max(v, 0.0), 1.0)) for v in fixed_rho_vals
            ]
        else:
            rho_node = torch.clamp(self.rho_node[expert_idx], 0.4, 0.6)
            rho_edge = torch.clamp(self.rho_edge[expert_idx], 0.4, 0.6)
            rho_feat = torch.clamp(self.rho_feat[expert_idx], 0.4, 0.6)

        def per_graph_keep(mask_vals, batch_idx):
            keep_per_graph = torch.zeros(batch_idx.max().item() + 1,
                                        device=mask_vals.device)
            count_per_graph = torch.zeros_like(keep_per_graph)
            keep_per_graph.scatter_add_(0, batch_idx, mask_vals.squeeze())
            count_per_graph.scatter_add_(0, batch_idx,
                                        torch.ones_like(mask_vals.squeeze()))
            return keep_per_graph / (count_per_graph + 1e-8)

        node_keep_pg = per_graph_keep(node_mask, node_batch)
        edge_keep_pg = per_graph_keep(edge_mask, edge_batch)
        feat_keep_pg = per_graph_keep(feat_mask.mean(dim=1, keepdim=True),
                                    node_batch)

        node_dev = ((node_keep_pg - rho_node) ** 2).mean()
        edge_dev = ((edge_keep_pg - rho_edge) ** 2).mean()
        feat_dev = ((feat_keep_pg - rho_feat) ** 2).mean()

        return node_dev + edge_dev + feat_dev

    # --------- MI-augmented Semantic Invariance ----------
    def compute_semantic_invariance_loss(
        self,
        h_masked: torch.Tensor,
        labels: torch.Tensor,
        h_orig: Optional[torch.Tensor] = None,
        beta: float = 0.99,
        normalize: bool = True,
        var_floor: float = 0.0,
        var_floor_weight: float = 0.0,
        lambda_e: float = 0.0,
        lambda_l: float = 0.0
    ) -> torch.Tensor:
        """
        Original: prototype pull.
        Added: EA on h_C (env ⟂ h_C) and LA on residual h_S (Y ⟂ h_S) via GRL and pseudo-envs.
        """
        # ---- original prototype compactness ----
        h = F.normalize(h_masked, p=2, dim=1) if normalize else h_masked

        unique = labels.unique()
        loss = h.new_tensor(0.0)
        total_w = h.new_tensor(0.0)

        # for c in unique:
        #     idx = (labels == c).nonzero(as_tuple=False).view(-1)
        #     n_c = int(idx.numel())
        #     if n_c < 2:
        #         continue
        #     h_c = h[idx]
        #     mu_c = F.normalize(h_c.mean(0, keepdim=True), p=2, dim=1) if normalize else h_c.mean(0, keepdim=True)
        #     sim = F.cosine_similarity(h_c, mu_c.expand_as(h_c), dim=1)
        #     class_loss = 1.0 - sim.mean()
        #     w_c = (1.0 - beta) / (1.0 - (beta ** n_c)) if beta is not None else 1.0
        #     loss = loss + w_c * class_loss
        #     total_w = total_w + w_c

        # loss = loss / torch.clamp(total_w, min=1.0)

        # if var_floor_weight > 0.0 and h.size(0) > 1:
        #     std_per_dim = h.std(dim=0, unbiased=False)
        #     vf_penalty = F.relu(var_floor - std_per_dim).mean()
        #     loss = loss + var_floor_weight * vf_penalty

        # ---- MI bits (opt-in) ----
        if (lambda_e > 0.0) or (lambda_l > 0.0):
            assert h_orig is not None, "h_orig required for MI-augmented semantic invariance"
            h_C = h_masked
            with torch.no_grad():
                h_S = (h_orig.detach() - h_C).detach()

            # Pseudo-envs from nuisance residuals (cheap & effective)
            with torch.no_grad():
                # Normalize for clustering stability
                r = F.normalize(h_S, p=2, dim=1)
                pseudo_env = kmeans_torch(r, K=self.num_envs, iters=10)  # (B,)

            # EA on h_C
            if lambda_e > 0.0:
                logits_e = self.env_head_sem(grad_reverse(h_C, lambda_e))
                loss = loss + lambda_e * F.cross_entropy(logits_e, pseudo_env)

            # LA on h_S
            if lambda_l > 0.0:
                logits_y_spur = self.lbl_head_spur_sem(grad_reverse(h_S, lambda_l))
                loss = loss + lambda_l * F.cross_entropy(logits_y_spur, labels)

        return loss

    # --------- MI-augmented Structural Invariance ----------
    def compute_structural_invariance_loss(
        self,
        data_masked: Data,
        labels: torch.Tensor,
        s_orig: Optional[torch.Tensor] = None,
        mode: str = "randomwalk",
        rw_max_steps: int = 8,
        graphon_bins: int = 8,
        mu_e: float = 0.0,
        mu_l: float = 0.0
    ):
        """
        Original: intra-class variance on structural vectors of masked graphs.
        Added: EA on s_C and LA on residual s_S with pseudo-envs from s_S (no labels).
        """
        # Step 1: Get structural vectors for masked graphs
        if mode == "randomwalk":
            s_mask = self._rw_struct_vec_batched(data_masked, rw_max_steps)  # (B, d)
        elif mode == "graphon":
            s_mask = self._graphon_vec_batched(data_masked, graphon_bins)    # (B, d)
        else:
            raise ValueError("Unsupported mode: choose 'randomwalk' or 'graphon'.")

        # Step 2: Weighted intra-class variance (original)
        unique_labels = labels.unique()
        total_weight = 0.0
        loss = s_mask.new_tensor(0.0)

        # for lbl in unique_labels:
        #     idx = (labels == lbl).nonzero(as_tuple=False).view(-1)
        #     if idx.numel() <= 1:
        #         continue
        #     vecs = s_mask[idx]
        #     class_mean = vecs.mean(dim=0, keepdim=True)
        #     class_loss = ((vecs - class_mean) ** 2).mean()
        #     weight = 1.0
        #     loss = loss + weight * class_loss
        #     total_weight += weight

        # if total_weight > 0:
        #     loss = loss / total_weight

        # Step 3: MI bits (opt-in)
        if (mu_e > 0.0) or (mu_l > 0.0):
            assert s_orig is not None, "s_orig required for MI-augmented structural invariance"

            s_C = s_mask
            with torch.no_grad():
                s_S = (s_orig.detach() - s_C).detach()

            # pseudo-envs from structure residuals
            with torch.no_grad():
                r = F.normalize(s_S, p=2, dim=1)
                pseudo_env = kmeans_torch(r, K=self.num_envs, iters=10)

            # lazy-build heads (match dim)
            if (self.env_head_str is None) or (self.env_head_str.in_features != s_C.size(1)):
                self.env_head_str = nn.Linear(s_C.size(1), self.num_envs).to(s_C.device)
                self.lbl_head_spur_str = nn.Linear(s_C.size(1), self.expert_classifiers[0].out_features).to(s_C.device)

            # EA on s_C
            if mu_e > 0.0:
                logits_e = self.env_head_str(grad_reverse(s_C, mu_e))
                loss = loss + mu_e * F.cross_entropy(logits_e, pseudo_env)

            # LA on s_S
            if mu_l > 0.0:
                logits_y_spur = self.lbl_head_spur_str(grad_reverse(s_S, mu_l))
                loss = loss + mu_l * F.cross_entropy(logits_y_spur, labels)

        return loss


    def _rw_struct_vec_batched(self, data: Data, rw_max_steps: int):
        """
        Returns (B, T) vector of average return probabilities across nodes,
        for T walk lengths (1..rw_max_steps). Uses edge weights and node weights if present.
        """
        A_list = self._dense_adj_per_graph(data)
        out = []
        for A in A_list:
            n = A.size(0)
            if n == 0:
                out.append(torch.zeros(rw_max_steps, device=A.device))
                continue
            deg = A.sum(dim=1, keepdim=True)  # (n,1)
            deg[deg == 0] = 1
            P = A / deg

            P_power = torch.eye(n, device=A.device)
            feats = []
            for _t in range(rw_max_steps):
                P_power = P_power @ P  # P^{t+1}
                feats.append(torch.diag(P_power))  # (n,)

            feats = torch.stack(feats, dim=1).mean(dim=0)
            out.append(feats)

        return torch.stack(out, dim=0)  # (B, T)
    
    def _dense_adj_per_graph(self, data: Data):
        """
        Returns a list of dense adjacency matrices [A_0, ..., A_{B-1}] (each (n_i, n_i)),
        where A_i applies edge weights (edge_attr) and soft node masking (node_weight) if provided.
        """
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            adj = to_dense_adj(data.edge_index, batch=data.batch, edge_attr=data.edge_attr).squeeze(0)
        else:
            adj = to_dense_adj(data.edge_index, batch=data.batch).squeeze(0)  # (B, N_max, N_max)

        B = int(data.batch.max().item()) + 1 if data.batch is not None else 1
        A_list = []
        for b in range(B):
            node_idx = (data.batch == b).nonzero(as_tuple=False).view(-1)
            n = node_idx.numel()
            A = adj[b, :n, :n]

            if hasattr(data, "node_weight") and data.node_weight is not None:
                w = data.node_weight[node_idx].view(n, 1)  # (n,1)
                A = (A * w) * w.T

            A_list.append(A)
        return A_list  # list of (n_i, n_i)

    def _graphon_vec_batched(self, data: Data, graphon_bins: int):
        """
        Returns (B, bins*bins) graphon-like vector per graph via degree-sorted block averages.
        Respects edge weights and soft node weights if provided.
        """
        A_list = self._dense_adj_per_graph(data)
        vecs = []
        for A in A_list:
            n = A.size(0)
            if n == 0:
                vecs.append(torch.zeros(graphon_bins * graphon_bins, device=A.device))
                continue

            degrees = A.sum(dim=1)
            idx = torch.argsort(degrees, descending=True)
            A_sorted = A[idx][:, idx]

            bin_size = max(1, math.ceil(n / graphon_bins))
            G = torch.zeros((graphon_bins, graphon_bins), device=A.device)
            for i in range(graphon_bins):
                i0, i1 = i * bin_size, min((i + 1) * bin_size, n)
                if i0 >= n:
                    break
                for j in range(graphon_bins):
                    j0, j1 = j * bin_size, min((j + 1) * bin_size, n)
                    if j0 >= n:
                        break
                    block = A_sorted[i0:i1, j0:j1]
                    if block.numel() > 0:
                        G[i, j] = block.mean()
            vecs.append(G.flatten())

        return torch.stack(vecs, dim=0)  # (B, bins*bins)
