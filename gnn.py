import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, SAGEConv, global_mean_pool, global_add_pool

class GNN(nn.Module):
    """
    Base GNN class that can be configured with different layer types
    """
    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 out_channels, 
                 num_layers=3, 
                 dropout=0.5, 
                 gnn_type='gin',
                 pool='mean'):
        super(GNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Select pooling function
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'sum':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unsupported pooling type: {pool}")
        
        # Initialize layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(self._create_conv_layer(gnn_type, in_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(self._create_conv_layer(gnn_type, hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Last layer
        self.convs.append(self._create_conv_layer(gnn_type, hidden_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        self.lin = nn.Linear(hidden_channels, out_channels)
    
    def _create_conv_layer(self, gnn_type, in_channels, out_channels):
        if gnn_type.lower() == 'gin':
            mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            )
            return GINConv(mlp)
        elif gnn_type.lower() == 'gcn':
            return GCNConv(in_channels, out_channels)
        elif gnn_type.lower() == 'sage':
            return SAGEConv(in_channels, out_channels)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
    
    def forward(self, x, edge_index, batch):
        # Node embedding
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph-level readout
        x = self.pool(x, batch)
        
        # Prediction
        x = self.lin(x)
        
        return x 