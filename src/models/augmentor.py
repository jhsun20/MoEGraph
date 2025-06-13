import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import random

from models.gnn_models import GCN, GIN, GraphSAGE

class GNNEncoder(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layers, dropout, model_type='GIN', pooling='mean'):
        super(GNNEncoder, self).__init__()
        if model_type == 'GCN':
            self.model = GCN(num_features, hidden_dim, hidden_dim, num_layers, dropout, pooling)
        elif model_type == 'GIN':
            self.model = GIN(num_features, hidden_dim, hidden_dim, num_layers, dropout, pooling)
        elif model_type == 'GraphSAGE':
            self.model = GraphSAGE(num_features, hidden_dim, hidden_dim, num_layers, dropout, pooling)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def forward(self, data):
        return self.model(data)
    

class MetaLearner(nn.Module):
    def __init__(self, config, dataset_info):
        super().__init__()
        in_dim = dataset_info['num_features']
        hidden_dim = config['augmentation']['hidden_dim']
        num_layers = config['augmentation']['num_layers']
        dropout = config['augmentation']['dropout']
        model_type = config['augmentation']['model_type']
        pooling = config['augmentation']['pooling']
        num_experts = config['moe']['num_experts']

        self.encoder = GNNEncoder(in_dim, hidden_dim, num_layers, dropout, model_type, pooling)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 4),  # 4 params: p_node_drop, p_edge_drop, p_edge_add, noise_std
                nn.Sigmoid()  # raw params ∈ (0, 1)
            )
            for _ in range(num_experts)
        ])

    def clip_params(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Post-process the raw meta-learner output to stay in plausible bounds.
        This version avoids in-place ops to maintain gradient flow.
        """
        # theta: [4,] or [batch_size, 4]
        bounds = torch.tensor([0.3, 0.3, 0.3, 0.1], device=theta.device)
        return torch.min(theta, bounds)


    def forward(self, data):
        h = self.encoder(data)
        h_pool = h.mean(dim=0, keepdim=True)  # graph-level embedding
        raw_params = [head(h_pool).squeeze(0) for head in self.heads]  # sigmoid applied
        param_sets = [self.clip_params(params) for params in raw_params]
        return param_sets


class Augmentor:
    def __init__(self):
        pass

    def apply(self, data: Data, params: torch.Tensor) -> Data:
        # Keep params as differentiable
        p_node_drop, p_edge_drop, p_edge_add, noise_std = params.unbind(0)

        x = self.node_mask(data.x, p_node_drop)
        edge_index, edge_weight = self.edge_perturb(data.edge_index, data.num_nodes, p_edge_drop)
        x = self.feature_perturb(x, noise_std)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=data.y, batch=data.batch)

    def node_mask(self, x, p_node_drop):
        mask_logits = torch.randn(x.size(0), device=x.device)
        soft_mask = torch.sigmoid(mask_logits - p_node_drop)
        return x * soft_mask.unsqueeze(1)

    def edge_perturb(self, edge_index, num_nodes, p_drop):
        # Convert edge_index to dense adjacency matrix
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]  # (N, N)
        adj = adj.to(edge_index.device)

        # Symmetrize and zero diagonal
        adj = torch.triu(adj, diagonal=1)
        adj = adj + adj.t()
        adj.fill_diagonal_(0)

        # Create a differentiable mask using sigmoid noise
        noise = torch.randn_like(adj)
        keep_prob = 1 - p_drop
        edge_mask = torch.sigmoid((noise - p_drop))  # higher noise = higher retention

        soft_adj = adj * edge_mask

        # Convert back to edge_index, edge_attr
        edge_index, edge_weight = dense_to_sparse(soft_adj)
        return edge_index, edge_weight

    def feature_perturb(self, x, noise_std):
        return x + noise_std * torch.randn_like(x)