import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool, MessagePassing, BatchNorm
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_geometric.nn.norm import LayerNorm  # <-- use PyG LayerNorm (node-wise)


class GCN(nn.Module):
    """Graph Convolutional Network (GCN) model."""
    
    def __init__(self, num_features, num_classes, hidden_dim, num_layers, dropout, pooling='mean'):
        super(GCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Set pooling function
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'sum':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unsupported pooling type: {pooling}")
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply GCN layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply global pooling
        x = self.pool(x, batch)
        
        # Apply classifier
        x = self.classifier(x)
        
        return x


class GIN(nn.Module):
    """Graph Isomorphism Network (GIN) model."""
    
    def __init__(self, num_features, num_classes, hidden_dim, num_layers, dropout, pooling='mean'):
        super(GIN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.convs = nn.ModuleList()
        nn1 = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(nn1))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            nn_i = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(nn_i))
        
        # Output layer
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Set pooling function
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'sum':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unsupported pooling type: {pooling}")
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply GIN layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply global pooling
        x = self.pool(x, batch)
        
        # Apply classifier
        x = self.classifier(x)
        
        return x


class GraphSAGE(nn.Module):
    """GraphSAGE model."""
    
    def __init__(self, num_features, num_classes, hidden_dim, num_layers, dropout, pooling='mean'):
        super(GraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(num_features, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Set pooling function
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'sum':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unsupported pooling type: {pooling}")
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply GraphSAGE layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply global pooling
        x = self.pool(x, batch)
        
        # Apply classifier
        x = self.classifier(x)
        
        return x 
    

class GINConvWithEdgeWeight(MessagePassing):
    def __init__(self, nn_module, eps=0.0, train_eps=False):
        super().__init__(aggr='add')  # GIN uses sum aggregation
        self.nn = nn_module
        self.initial_eps = eps
        self.eps = torch.nn.Parameter(torch.Tensor([eps])) if train_eps else eps
        self.train_eps = train_eps

    def forward(self, x, edge_index, edge_weight=None):
        # Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=x.dtype, device=x.device)
        else:
            edge_weight = torch.cat([
                edge_weight,
                torch.ones(x.size(0), dtype=x.dtype, device=x.device)  # self-loop weight = 1
            ], dim=0)

        return self.nn((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_weight=edge_weight))

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return f'{self.__class__.__name__}(nn={self.nn})'
    
class GINEncoderWithEdgeWeight(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout=0.5, train_eps=False, global_pooling='mean'):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()   # keep attribute name to avoid touching caller code
        self.acts  = nn.ModuleList()
        self.global_pooling = global_pooling

        for i in range(num_layers):
            input_dim = in_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            conv = GINConvWithEdgeWeight(mlp, train_eps=train_eps)
            self.convs.append(conv)

            # >>> swapped BatchNorm for LayerNorm (node-wise) <<<
            # PyG's LayerNorm normalizes per-node feature vector: shape (N, C)
            # self.bns.append(LayerNorm(hidden_dim, affine=True, mode='node'))
            self.bns.append(BatchNorm(hidden_dim))

            self.acts.append(nn.ReLU())

        self.dropout = float(dropout)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        for conv, bn, act in zip(self.convs, self.bns, self.acts):
            x = conv(x, edge_index, edge_weight)
            x = bn(x)
            x = act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.global_pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.global_pooling == 'sum':
            x = global_add_pool(x, batch)
        elif self.global_pooling == 'none':
            pass
        else:
            raise ValueError(f"Unsupported pooling type: {self.global_pooling}")
        return x  # (N, hidden_dim)
    
    
class ExpertClassifier(nn.Module):
    def __init__(self, hidden_dim, num_classes, dropout, global_pooling):
        super().__init__()
        self.encoder = GINEncoderWithEdgeWeight(
            hidden_dim, hidden_dim, 1, dropout,
            train_eps=True, global_pooling=global_pooling
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        h = self.encoder(x, edge_index, edge_weight, batch)  # → [B, hidden_dim]
        out = self.mlp(h)  # → [B, num_classes]
        return out
