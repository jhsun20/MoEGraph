import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool

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