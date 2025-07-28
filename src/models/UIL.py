import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gnn_models import GCN, GIN, GraphSAGE, GINEncoderWithEdgeWeight

from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_adj

class UILModel(nn.Module):
    def __init__(self, config, dataset_info, rho=0.5):
        super().__init__()
        num_features = dataset_info['num_features']
        num_classes = dataset_info['num_classes']
        hidden_dim = config['model']['hidden_dim']
        num_layers = config['model']['num_layers']
        dropout = config['model']['dropout']
        self.weight_str = config['model']['weight_str']
        self.weight_sem = config['model']['weight_sem']
        self.weight_reg = config['model']['weight_reg']
        self.weight_ce = config['model']['weight_ce']
        self.verbose = config['experiment']['debug']['verbose']  # Check if debug mode is enabled
        self.encoder = GINEncoderWithEdgeWeight(num_features, hidden_dim, num_layers, dropout, train_eps=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)  # MLP for classification
        self.mask_mlp_node = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))  # MLP for node masks
        self.mask_mlp_edge = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))  # MLP for edge masks
        self.rho = nn.Parameter(torch.tensor(rho))  # Learnable stable ratio

    def forward(self, data, target=None, embeddings_by_env=None, labels_by_env=None):
        """
        Forward pass with integrated loss computation.

        Returns:
            output: dict with keys:
                'logits', 'h_stable', 'h_orig',
                'node_mask', 'edge_mask',
                'loss_total', 'loss_ce', 'loss_reg', 'loss_sem', 'loss_str'
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First pass: full graph
        Z = self.encoder(x, edge_index, batch=batch)  # node embeddings (N, d)
        node_mask = torch.sigmoid(self.mask_mlp_node(Z.detach()))  # (N, 1)
        edge_feat = torch.cat([Z[edge_index[0]], Z[edge_index[1]]], dim=1)
        edge_mask = torch.sigmoid(self.mask_mlp_edge(edge_feat.detach()))  # (E, 1)

        # Apply masks and re-encode
        masked_x = x * node_mask
        edge_weight = edge_mask.view(-1)
        masked_Z = self.encoder(masked_x, edge_index, batch=batch, edge_weight=edge_weight)
        h_stable = global_mean_pool(masked_Z, batch)
        h_orig = global_mean_pool(Z, batch)
        logits = self.classifier(h_stable)

        # Cache everything
        self.cached_masks = {
            'node_mask': node_mask,
            'edge_mask': edge_mask,
            'Z': Z,
            'h_stable': h_stable,
            'h_orig': h_orig,
            'edge_index': edge_index,
            'batch': batch
        }

        # Prepare output dictionary
        output = {
            'logits': logits,
            'h_stable': h_stable,
            'h_orig': h_orig,
            'node_mask': node_mask,
            'edge_mask': edge_mask,
            'loss_total': None,
            'loss_ce': None,
            'loss_reg': None,
            'loss_sem': None,
            'loss_str': None,
            'cached_masks': self.cached_masks
        }

        # Loss computation
        if target is not None:
            ce_loss = self.compute_classification_loss(logits, target)
            reg_loss = self.compute_mask_regularization()
            sem_loss = self.compute_semantic_invariance_loss()
            str_loss = torch.tensor(0.0, device=logits.device)
            if self.weight_str > 0 and embeddings_by_env is not None and labels_by_env is not None:
                str_loss = self.compute_structural_invariance_loss(embeddings_by_env, labels_by_env)

            total_loss = self.weight_ce * ce_loss + self.weight_reg * reg_loss + self.weight_str * str_loss + self.weight_sem * sem_loss

            output.update({
                'loss_total': total_loss,
                'loss_ce': ce_loss,
                'loss_reg': reg_loss,
                'loss_sem': sem_loss,
                'loss_str': str_loss
            })

        return output
    
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

    def compute_mask_regularization(self):
        node_mask = self.cached_masks['node_mask']
        edge_mask = self.cached_masks['edge_mask']
        rho = torch.clamp(self.rho, 0.0, 1.0)

        node_ratio = node_mask.mean()
        edge_ratio = edge_mask.mean()
        node_dev = (node_ratio - rho).pow(2)
        edge_dev = (edge_ratio - rho).pow(2)

        node_l0 = (node_mask > 0).float().mean()
        edge_l0 = (edge_mask > 0).float().mean()
        l0_dev = (node_l0 - rho).pow(2) + (edge_l0 - rho).pow(2)

        return node_dev + edge_dev + l0_dev

    def compute_semantic_invariance_loss(self):
        h_stable = self.cached_masks['h_stable']
        h_orig = self.cached_masks['h_orig']
        return F.mse_loss(h_stable, h_orig)

    def compute_structural_invariance_loss(self, embeddings_by_env, labels_by_env):
        loss = 0.0
        all_labels = set()
        for env_labels in labels_by_env.values():
            all_labels.update(env_labels)
        all_labels = list(all_labels)

        for label in all_labels:
            h_by_env = []
            for env_id in embeddings_by_env:
                env_h = [h for h, y in zip(embeddings_by_env[env_id], labels_by_env[env_id]) if y == label]
                if env_h:
                    h_by_env.append(torch.stack(env_h).mean(dim=0))

            if len(h_by_env) > 1:
                for i in range(len(h_by_env)):
                    for j in range(i + 1, len(h_by_env)):
                        loss += F.mse_loss(h_by_env[i], h_by_env[j])

        return loss
    

    
class UILModelSharedEncoder(nn.Module):
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
        self.verbose = config['experiment']['debug']['verbose']  # Check if debug mode is enabled
        self.num_experts = num_experts
        
        # Shared encoder
        self.encoder = GINEncoderWithEdgeWeight(num_features, hidden_dim, num_layers, dropout, train_eps=True)
        
        # Multiple expert heads - each expert has its own node and edge mask MLPs
        self.expert_node_masks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
            for _ in range(num_experts)
        ])
        
        self.expert_edge_masks = nn.ModuleList([
            nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
            for _ in range(num_experts)
        ])
        
        # Multiple expert classifiers
        self.expert_classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_experts)
        ])
        
        self.rho = nn.Parameter(torch.tensor(rho))  # Learnable stable ratio

    def forward(self, data, target=None, embeddings_by_env=None, labels_by_env=None):
        """
        Forward pass with integrated loss computation for multiple experts.

        Returns:
            output: dict with keys:
                'logits', 'h_stable', 'h_orig',
                'node_masks', 'edge_masks', 'expert_logits',
                'loss_total', 'loss_ce', 'loss_reg', 'loss_sem', 'loss_str'
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First pass: full graph - shared encoder
        Z = self.encoder(x, edge_index, batch=batch)  # node embeddings (N, d)
        
        # Generate masks for each expert
        node_masks = []
        edge_masks = []
        expert_logits = []
        h_stable_list = []
        
        edge_feat = torch.cat([Z[edge_index[0]], Z[edge_index[1]]], dim=1)
        
        for expert_idx in range(self.num_experts):
            # Generate expert-specific masks
            node_mask = torch.sigmoid(self.expert_node_masks[expert_idx](Z.detach()))  # (N, 1)
            edge_mask = torch.sigmoid(self.expert_edge_masks[expert_idx](edge_feat.detach()))  # (E, 1)
            
            node_masks.append(node_mask)
            edge_masks.append(edge_mask)
            
            # Apply masks and re-encode for this expert
            masked_x = x * node_mask
            edge_weight = edge_mask.view(-1)
            masked_Z = self.encoder(masked_x, edge_index, batch=batch, edge_weight=edge_weight)
            h_stable = global_mean_pool(masked_Z, batch)
            h_stable_list.append(h_stable)
            
            # Expert-specific classification
            expert_logit = self.expert_classifiers[expert_idx](h_stable)
            expert_logits.append(expert_logit)
        
        # Stack all expert outputs
        node_masks = torch.stack(node_masks, dim=1)  # (N, num_experts, 1)
        edge_masks = torch.stack(edge_masks, dim=1)  # (E, num_experts, 1)
        expert_logits = torch.stack(expert_logits, dim=1)  # (batch_size, num_experts, num_classes)
        h_stable_list = torch.stack(h_stable_list, dim=1)  # (batch_size, num_experts, hidden_dim)
        h_orig = global_mean_pool(Z, batch)

        # Cache everything
        self.cached_masks = {
            'node_masks': node_masks,
            'edge_masks': edge_masks,
            'Z': Z,
            'h_stable_list': h_stable_list,
            'expert_logits': expert_logits,
            'edge_index': edge_index,
            'batch': batch
        }

        # Prepare output dictionary
        output = {
            'h_stable': h_stable_list,
            'h_orig': h_orig,
            'node_masks': node_masks,
            'edge_masks': edge_masks,
            'expert_logits': expert_logits,
            'loss_total_list': None,
            'loss_ce_list': None,
            'loss_reg_list': None,
            'loss_sem_list': None,
            'loss_str_list': None,
            'cached_masks': self.cached_masks
        }

        # Loss computation for each expert
        if target is not None:
            ce_loss_list = []
            reg_loss_list = []
            sem_loss_list = []
            str_loss_list = []
            total_loss_list = []
            
            for expert_idx in range(self.num_experts):
                # Get expert-specific logits and embeddings
                expert_logit = expert_logits[:, expert_idx, :]  # (batch_size, num_classes)
                expert_h_stable = h_stable_list[:, expert_idx, :]  # (batch_size, hidden_dim)
                
                # Compute classification loss for this expert
                ce_loss = self.compute_classification_loss(expert_logit, target)
                ce_loss_list.append(ce_loss)
                
                # Compute mask regularization for this expert
                expert_node_mask = node_masks[:, expert_idx, :]  # (N, 1)
                expert_edge_mask = edge_masks[:, expert_idx, :]  # (E, 1)
                reg_loss = self.compute_mask_regularization_single(expert_node_mask, expert_edge_mask)
                reg_loss_list.append(reg_loss)
                
                # Compute semantic invariance loss for this expert
                sem_loss = self.compute_semantic_invariance_loss_single(expert_h_stable, h_orig)
                sem_loss_list.append(sem_loss)
                
                # Compute structural invariance loss (set to zero for now)
                str_loss = torch.tensor(0.0, device=expert_logit.device)
                if self.weight_str > 0 and embeddings_by_env is not None and labels_by_env is not None:
                    str_loss = self.compute_structural_invariance_loss(embeddings_by_env, labels_by_env)
                str_loss_list.append(str_loss)
                
                # Compute total loss for this expert
                total_loss = (self.weight_ce * ce_loss + 
                            self.weight_reg * reg_loss + 
                            self.weight_str * str_loss + 
                            self.weight_sem * sem_loss)
                total_loss_list.append(total_loss)
            
            # Stack all losses
            ce_loss_list = torch.stack(ce_loss_list)  # (num_experts,)
            reg_loss_list = torch.stack(reg_loss_list)  # (num_experts,)
            sem_loss_list = torch.stack(sem_loss_list)  # (num_experts,)
            str_loss_list = torch.stack(str_loss_list)  # (num_experts,)
            total_loss_list = torch.stack(total_loss_list)  # (num_experts,)

            output.update({
                'loss_total_list': total_loss_list,
                'loss_ce_list': ce_loss_list,
                'loss_reg_list': reg_loss_list,
                'loss_sem_list': sem_loss_list,
                'loss_str_list': str_loss_list
            })

        return output
    
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

    def compute_mask_regularization_single(self, node_mask, edge_mask):
        """Compute mask regularization for a single expert"""
        rho = torch.clamp(self.rho, 0.0, 1.0)

        node_ratio = node_mask.mean()
        edge_ratio = edge_mask.mean()
        node_dev = (node_ratio - rho).pow(2)
        edge_dev = (edge_ratio - rho).pow(2)

        node_l0 = (node_mask > 0).float().mean()
        edge_l0 = (edge_mask > 0).float().mean()
        l0_dev = (node_l0 - rho).pow(2) + (edge_l0 - rho).pow(2)

        return node_dev + edge_dev + l0_dev

    def compute_semantic_invariance_loss_single(self, h_stable, h_orig):
        """Compute semantic invariance loss for a single expert"""
        return F.mse_loss(h_stable, h_orig)

    def compute_structural_invariance_loss(self, embeddings_by_env, labels_by_env):
        loss = 0.0
        all_labels = set()
        for env_labels in labels_by_env.values():
            all_labels.update(env_labels)
        all_labels = list(all_labels)

        for label in all_labels:
            h_by_env = []
            for env_id in embeddings_by_env:
                env_h = [h for h, y in zip(embeddings_by_env[env_id], labels_by_env[env_id]) if y == label]
                if env_h:
                    h_by_env.append(torch.stack(env_h).mean(dim=0))

            if len(h_by_env) > 1:
                for i in range(len(h_by_env)):
                    for j in range(i + 1, len(h_by_env)):
                        loss += F.mse_loss(h_by_env[i], h_by_env[j])

        return loss
    

class Experts(nn.Module):
    def __init__(self, config, dataset_info, rho=0.9):
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
        
        self.rho_node = nn.Parameter(torch.tensor(rho))
        self.rho_edge = nn.Parameter(torch.tensor(rho))
        self.rho_feat = nn.Parameter(torch.tensor(rho))

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
        
        node_masks = []
        edge_masks = []
        feat_masks = []
        expert_logits = []
        h_stable_list = []
        
        edge_feat = torch.cat([Z[edge_index[0]], Z[edge_index[1]]], dim=1)

        for expert_idx in range(self.num_experts):
            node_mask = torch.sigmoid(self.expert_node_masks[expert_idx](Z))
            edge_mask = torch.sigmoid(self.expert_edge_masks[expert_idx](edge_feat))
            feat_mask = torch.sigmoid(self.expert_feat_masks[expert_idx](Z))  # (N, D)
            

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
                reg_loss = self.compute_mask_regularization_loss(expert_node_mask, expert_edge_mask, expert_feat_mask)
                reg_loss_list.append(reg_loss)

                sem_loss = self.compute_semantic_invariance_loss(expert_h_stable, h_orig)
                #sem_loss = torch.tensor(0.0, device=expert_logit.device)
                sem_loss_list.append(sem_loss)

                str_loss = self.compute_structural_invariance_loss(expert_h_stable, target, edge_index, batch, expert_node_mask, expert_edge_mask)
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

    def compute_mask_regularization_loss(self, node_mask, edge_mask, feat_mask):
        rho_node = torch.clamp(self.rho_node, 0.0, 1.0)
        rho_edge = torch.clamp(self.rho_edge, 0.0, 1.0)
        rho_feat = torch.clamp(self.rho_feat, 0.0, 1.0)

        node_ratio = node_mask.mean()
        edge_ratio = edge_mask.mean()
        feat_ratio = feat_mask.mean()
        node_dev = (node_ratio - rho_node).pow(2)
        edge_dev = (edge_ratio - rho_edge).pow(2)
        feat_dev = (feat_ratio - rho_feat).pow(2)

        node_l0 = (node_mask > 0).float().mean()
        edge_l0 = (edge_mask > 0).float().mean()
        feat_l0 = (feat_mask > 0).float().mean()
        l0_dev = (node_l0 - rho_node).pow(2) + (edge_l0 - rho_edge).pow(2) + (feat_l0 - rho_feat).pow(2)

        return node_dev + edge_dev + feat_dev + l0_dev
    
    # def compute_mask_regularization_loss(self, node_mask, edge_mask, feat_mask):
    #     rho = torch.clamp(self.rho, 0.0, 1.0)
    #     eps = 1e-6

    #     # Sparsity (L2 deviation from target rho)
    #     node_sparsity = (node_mask.mean() - rho).pow(2)
    #     edge_sparsity = (edge_mask.mean() - rho).pow(2)
    #     feat_sparsity = ((feat_mask.mean(dim=0)) - rho).pow(2).mean()

    #     return node_sparsity + edge_sparsity + feat_sparsity

    def compute_semantic_invariance_loss(self, h_stable, h_orig):
        return F.mse_loss(h_stable, h_orig)


    def compute_structural_invariance_loss(self, h_stable, labels, edge_index, batch, node_mask, edge_mask, mode="embedding", topk=10):
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