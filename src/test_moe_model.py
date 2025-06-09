import torch
from torch_geometric.data import Data
from models.experts import MoEModel

# Sample configuration and dataset information
config = {
    'moe': {
        'num_experts': 2,
        'aggregation': 'mean'
    },
    'model': {
        'hidden_dim': 64,
        'num_layers': 3,
        'dropout': 0.5,
        'pooling': 'mean'
    }
}

dataset_info = {
    'num_features': 16,
    'num_classes': 3
}

# Create dummy data for two graphs
num_nodes_graph1 = 10
num_nodes_graph2 = 8

x = torch.rand((num_nodes_graph1 + num_nodes_graph2, dataset_info['num_features']))  # Random node features for both graphs
edge_index = torch.randint(0, num_nodes_graph1 + num_nodes_graph2, (2, 30))  # Random edges
batch = torch.cat((torch.zeros(num_nodes_graph1, dtype=torch.long), torch.ones(num_nodes_graph2, dtype=torch.long)))  # Two graphs in batch

data = Data(x=x, edge_index=edge_index, batch=batch)

# Initialize and test the MoEModel
model = MoEModel(config, dataset_info)
model.test_moe_model(data) 