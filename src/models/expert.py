import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gnn_models import GCN, GIN, GraphSAGE, GINEncoderWithEdgeWeight

from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data
import math


class Experts(nn.Module):
    def __init__(self, config, dataset_info, rho=0.5):
        super().__init__()
        num_features = dataset_info['num_features']
        num_classes = dataset_info['num_classes']
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
        
        # self.rho_node = nn.Parameter(torch.tensor(rho))
        # self.rho_edge = nn.Parameter(torch.tensor(rho))
        # self.rho_feat = nn.Parameter(torch.tensor(rho))
        self.rho_node = nn.Parameter(torch.full((num_experts,), float(rho)))
        self.rho_edge = nn.Parameter(torch.full((num_experts,), float(rho)))
        self.rho_feat = nn.Parameter(torch.full((num_experts,), float(rho)))

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
            'batch': batch
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
            ce_loss_list, reg_loss_list, sem_loss_list, str_loss_list, total_loss_list = [], [], [], [], []
            
            for expert_idx in range(self.num_experts):
                expert_logit = expert_logits[:, expert_idx, :]
                expert_h_stable = h_stable_list[:, expert_idx, :]
                ce_loss = self.compute_classification_loss(expert_logit, target)
                ce_loss_list.append(ce_loss)

                expert_node_mask = node_masks[:, expert_idx, :]
                expert_edge_mask = edge_masks[:, expert_idx, :]
                expert_feat_mask = feat_masks[:, expert_idx, :]
                reg_loss = self.compute_mask_regularization_loss(expert_node_mask, expert_edge_mask, expert_feat_mask, expert_idx)
                reg_loss_list.append(reg_loss)

                sem_loss = self.compute_semantic_invariance_loss(expert_h_stable, h_orig, target)
                #sem_loss = torch.tensor(0.0, device=expert_logit.device)
                sem_loss_list.append(sem_loss)

                #str_loss = self.compute_structural_invariance_loss(expert_h_stable, target, edge_index, batch, expert_node_mask, expert_edge_mask)
                # structural masked graph for THIS expert (structure-only)
                masked_data = Data(
                    edge_index=edge_index,
                    batch=batch,
                    edge_attr=expert_edge_mask.view(-1),   # (E, 1) -> (E,)
                    node_weight=expert_node_mask.view(-1)  # (N, 1) -> (N,)
                )

                # === structural loss here (choose 'randomwalk' or 'graphon') ===
                str_loss = self.compute_structural_invariance_loss(
                    original_data,
                    masked_data,
                    mode="randomwalk",     # or "graphon"
                    rw_max_steps=4,        # tweak if needed
                    graphon_bins=4
                )
                str_loss_list.append(str_loss)

                total_loss = (self.weight_ce * ce_loss + 
                              self.weight_reg * reg_loss + 
                              self.weight_str * str_loss + 
                              self.weight_sem * sem_loss)
                total_loss_list.append(total_loss)
            
            output.update({
                'loss_total_list': torch.stack(total_loss_list),
                'loss_ce_list': torch.stack(ce_loss_list),
                'loss_reg_list': torch.stack(reg_loss_list),
                'loss_sem_list': torch.stack(sem_loss_list),
                'loss_str_list': torch.stack(str_loss_list)
            })

        return output
    
    def _hard_concrete_mask(self, logits, temperature=0.1):
        """
        Sample a hard 0/1 mask from logits with straight-through gradient.
        Args:
            logits: (N, 1) or (N, D) pre-activation mask values
            temperature: controls smoothness; lower = harder
        Returns:
            mask: hard {0,1} mask (same shape as logits) with gradient
        """
        # Add Gumbel noise
        uniform_noise = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-20) + 1e-20)

        # Continuous sample in (0,1)
        y_soft = torch.sigmoid((logits + gumbel_noise) / temperature)

        # Hard binary sample
        y_hard = (y_soft > 0.5).float()

        # Straight-through: gradients flow through y_soft
        return y_hard + (y_soft - y_soft.detach())

    def compute_classification_loss(self, pred, target, use_weights=False):
        """
        Computes cross-entropy loss with optional class imbalance correction.
        
        Args:
            pred (Tensor): Logits of shape (B, C)
            target (Tensor): Ground truth labels of shape (B,)
            use_weights (bool): Whether to use class weights for imbalance correction. Default: False
        
        Returns:
            Tensor: Cross-entropy loss
        """
        if use_weights:
            # Compute class counts from current batch
            num_classes = pred.size(1)
            class_counts = torch.bincount(target, minlength=num_classes).float()
            
            # Avoid division by zero
            class_counts[class_counts == 0] = 1.0

            # Inverse frequency weighting normalized to sum to 1
            weight = 1.0 / class_counts
            weight = weight / weight.sum()
            
            return F.cross_entropy(pred, target, weight=weight.to(pred.device))
        else:
            return F.cross_entropy(pred, target)

    def compute_mask_regularization_loss(self, node_mask, edge_mask, feat_mask, expert_idx: int):
        # clamp each expert's rho into [0.2, 0.8] as you wanted
        rho_node = torch.clamp(self.rho_node[expert_idx], 0.2, 0.8)
        rho_edge = torch.clamp(self.rho_edge[expert_idx], 0.2, 0.8)
        rho_feat = torch.clamp(self.rho_feat[expert_idx], 0.2, 0.8)

        # Mean deviation from target rho
        node_dev = (node_mask.mean() - rho_node).pow(2)
        edge_dev = (edge_mask.mean() - rho_edge).pow(2)
        feat_dev = (feat_mask.mean() - rho_feat).pow(2)

        # Sparsity surrogate: use mean(mask) instead of (mask > 0)
        node_l0 = node_mask.mean()
        edge_l0 = edge_mask.mean()
        feat_l0 = feat_mask.mean()
        l0_dev = (node_l0 - rho_node).pow(2) + (edge_l0 - rho_edge).pow(2) + (feat_l0 - rho_feat).pow(2)

        return node_dev + edge_dev + feat_dev + l0_dev


    def compute_semantic_invariance_loss(self, h_stable, h_orig, target):
        """
        Match h_stable to its class prototype from h_orig.
        Prototypes are batch-wise (not EMA) for simplicity.
        """
        device = h_stable.device
        loss = torch.tensor(0.0, device=device)
        unique_labels = target.unique()

        for lbl in unique_labels:
            idx = (target == lbl).nonzero(as_tuple=False).squeeze()
            if idx.numel() < 2:
                continue
            proto = h_orig[idx].mean(dim=0)  # class prototype
            loss += F.mse_loss(h_stable[idx], proto)

        return loss / max(1, len(unique_labels))

    def compute_structural_invariance_loss(
        self,
        data_orig: Data,
        data_masked: Data,
        mode: str = "randomwalk",
        rw_max_steps: int = 8,
        graphon_bins: int = 8,
    ):
        """
        Structural invariance between original and masked graphs.
        Returns MSE between per-graph structural vectors (B, d).
        Supports modes: 'randomwalk' or 'graphon'.
        Expects:
        - data.edge_index (batched)
        - data.batch (node->graph mapping)
        - Optional: data.edge_attr as edge weights
        - Optional: data.node_weight as node mask (soft)
        """
        if mode == "randomwalk":
            s_orig = self._rw_struct_vec_batched(data_orig, rw_max_steps)   # (B, T_agg)
            s_mask = self._rw_struct_vec_batched(data_masked, rw_max_steps) # (B, T_agg)
        elif mode == "graphon":
            s_orig = self._graphon_vec_batched(data_orig, graphon_bins)     # (B, bins*bins)
            s_mask = self._graphon_vec_batched(data_masked, graphon_bins)   # (B, bins*bins)
        else:
            raise ValueError("Unsupported mode: choose 'randomwalk' or 'graphon'.")

        return F.mse_loss(s_orig, s_mask)  # averages over batch & dims

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
                # return probability diag(P^t)
                feats.append(torch.diag(P_power))  # (n,)

            # (n,T) -> mean over nodes -> (T,)
            feats = torch.stack(feats, dim=1).mean(dim=0)
            out.append(feats)

        return torch.stack(out, dim=0)  # (B, T)
    
    # --- SHARED: build dense adj per graph (respects edge_attr & node_weight) ---
    def _dense_adj_per_graph(self, data: Data):
        """
        Returns a list of dense adjacency matrices [A_0, ..., A_{B-1}] (each (n_i, n_i)),
        where A_i applies edge weights (edge_attr) and soft node masking (node_weight) if provided.
        """
        # Build weighted adjacency for the whole batch; shape (B, N_max, N_max)
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            adj = to_dense_adj(data.edge_index, batch=data.batch, edge_attr=data.edge_attr).squeeze(0)
        else:
            adj = to_dense_adj(data.edge_index, batch=data.batch).squeeze(0)  # (B, N_max, N_max)

        B = int(data.batch.max().item()) + 1 if data.batch is not None else 1
        A_list = []
        for b in range(B):
            # mask per-graph valid nodes
            node_idx = (data.batch == b).nonzero(as_tuple=False).view(-1)
            n = node_idx.numel()
            A = adj[b, :n, :n]  # crop to actual size

            # apply soft node mask if provided
            if hasattr(data, "node_weight") and data.node_weight is not None:
                w = data.node_weight[node_idx].view(n, 1)  # (n,1)
                # soft delete: A' = diag(w) * A * diag(w)
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

            degrees = A.sum(dim=1)  # weighted degree after node masking
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

    
    def compute_semantic_invariance_loss2(self, h_stable, h_orig):
        return F.mse_loss(h_stable, h_orig)
    

    def compute_structural_invariance_loss2(self, h_stable, labels, edge_index, batch, node_mask, edge_mask, mode="embedding", topk=10):
        """
        Computes structural invariance loss.
        
        Args:
            h_stable: Tensor (B, D), graph-level embeddings from causal subgraphs
            labels: Tensor (B,), integer class labels
            edge_index: Tensor (2, E), edge list of the input graph
            batch: Tensor (N,), mapping nodes to graphs
            node_mask: Tensor (N, 1), soft node mask ∈ [0,1]
            edge_mask: Tensor (E, 1), soft edge mask ∈ [0,1]
            mode: "laplacian" or "embedding"
            topk: number of eigenvalues to use for Laplacian comparison

        Returns:
            Scalar loss (tensor)
        """
        device = h_stable.device
        loss = torch.tensor(0.0, device=device)
        unique_labels = labels.unique()

        if mode == "embedding":
            # === Embedding-based structural invariance (variance version) ===
            count = 0
            for lbl in unique_labels:
                indices = (labels == lbl).nonzero(as_tuple=False).squeeze()
                if indices.numel() < 2:
                    continue
                h_group = h_stable[indices]  # (B_lbl, D)
                mean = h_group.mean(dim=0)   # (D,)
                var = ((h_group - mean) ** 2).mean()  # scalar
                loss += var
                count += 1

            return loss / count if count > 0 else torch.tensor(0.0, device=device)

        elif mode == "laplacian":
            # === Laplacian spectrum-based structural invariance ===
            # print(f"Computing Laplacian spectrum-based structural invariance.")
            adj_dense = to_dense_adj(edge_index, batch=batch, edge_attr=edge_mask.view(-1)).squeeze(0)  # (B, N, N)
            if adj_dense.dim() == 2:
                adj_dense = adj_dense.unsqueeze(0)  # (1, N, N)

            spectra_by_label = {}

            for i in range(h_stable.size(0)):
                A = adj_dense[i]
                if A.size(0) != A.size(1):
                    continue

                deg = A.sum(dim=-1)
                D = torch.diag(deg)
                L = D - A

                try:
                    eigvals = torch.linalg.eigvalsh(L)
                    topk_eigs = eigvals[:topk]
                except RuntimeError as e:
                    if "linalg.eigh" in str(e):
                        # Skip this ill-conditioned graph
                        continue
                    else:
                        raise e

                lbl = labels[i].item()
                if lbl not in spectra_by_label:
                    spectra_by_label[lbl] = []
                spectra_by_label[lbl].append(topk_eigs)
                # print(f"Graph {i} with label {lbl}: top-{topk} eigenvalues = {topk_eigs}")

            for lbl, spectra in spectra_by_label.items():
                if len(spectra) < 2:
                    continue
                spectra_stack = torch.stack(spectra)  # (B_lbl, k)
                mean_spectrum = spectra_stack.mean(dim=0)  # (k,)
                var = ((spectra_stack - mean_spectrum)**2).mean()
                loss += var

            return loss / len(spectra_by_label) if spectra_by_label else torch.tensor(0.0, device=device)

        else:
            raise ValueError(f"Unsupported mode: {mode}. Choose 'laplacian' or 'embedding'.")
        