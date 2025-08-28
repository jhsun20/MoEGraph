import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool, MessagePassing, BatchNorm
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax, coalesce, add_remaining_self_loops
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
        num_nodes = x.size(0)
        device, dtype = x.device, x.dtype
        pre = check_double_self_loops(edge_index, edge_weight, num_nodes=x.size(0))

        # If no weights provided, start with ones for existing edges
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), dtype=dtype, device=device)

        # 1) Merge duplicate edges (including any duplicate self-loops)
        edge_index, edge_weight = coalesce(edge_index, edge_weight, num_nodes=num_nodes)

        # 2) Add self-loops ONLY where missing; fill those with weight=1
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value=1.0, num_nodes=num_nodes
        )

        post = check_double_self_loops(edge_index, edge_weight, num_nodes=x.size(0))

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

    def forward(self, x, edge_index, edge_weight=None, batch=None, node_weight=None):
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
    


def check_double_self_loops(edge_index,
                            edge_weight,
                            num_nodes,
                            raise_on_duplicates = True):
    """
    Returns a dict summary and (optionally) raises if any node has >1 self-loop.
    edge_index: (2, E)
    edge_weight: (E,) or None
    num_nodes: optional but recommended (for complete bincount)
    """
    assert edge_index.dim() == 2 and edge_index.size(0) == 2, "edge_index must be (2, E)"
    E = edge_index.size(1)

    if edge_weight is not None:
        assert edge_weight.dim() == 1 and edge_weight.numel() == E, \
            f"edge_weight must be shape (E,), got {edge_weight.shape} vs E={E}"

    # self-loop mask
    self_mask = edge_index[0].eq(edge_index[1])  # (E,)
    self_nodes = edge_index[0, self_mask]        # nodes with self-loops (with multiplicity)
    n_self_edges = int(self_mask.sum())

    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1 if E > 0 else 0

    # count how many self-loops each node has
    per_node_counts = torch.bincount(self_nodes, minlength=num_nodes) if n_self_edges > 0 \
                      else torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)

    dup_nodes = (per_node_counts > 1).nonzero(as_tuple=False).view(-1)
    has_duplicates = dup_nodes.numel() > 0
    max_self_loops_on_a_node = int(per_node_counts.max().item()) if n_self_edges > 0 else 0

    summary = {
        "num_edges": E,
        "num_nodes": num_nodes,
        "num_self_loop_edges": n_self_edges,
        "num_nodes_with_self_loop": int((per_node_counts > 0).sum().item()),
        "max_self_loops_on_a_node": max_self_loops_on_a_node,
        "has_duplicate_self_loops": bool(has_duplicates),
        "duplicate_nodes": dup_nodes.detach().cpu().tolist(),  # nodes with >1 self-loop
    }

    if has_duplicates and raise_on_duplicates:
        raise ValueError(f"Duplicate self-loops detected on nodes {summary['duplicate_nodes']}. "
                         f"Max per node = {summary['max_self_loops_on_a_node']}.")

    return summary
