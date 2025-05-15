import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_mean_pool

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, gnn_type='gin'):
        super(GNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        if gnn_type == 'gin':
            self.convs.append(GINConv(nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )))
        elif gnn_type == 'gcn':
            self.convs.append(GCNConv(in_channels, hidden_channels))
        elif gnn_type == 'sage':
            self.convs.append(SAGEConv(in_channels, hidden_channels))
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
            
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            if gnn_type == 'gin':
                self.convs.append(GINConv(nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels)
                )))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            elif gnn_type == 'sage':
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        if gnn_type == 'gin':
            self.convs.append(GINConv(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )))
        elif gnn_type == 'gcn':
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        elif gnn_type == 'sage':
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Final classifier
        self.classifier = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, batch):
        # Node embedding
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph embedding
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        
        return x 