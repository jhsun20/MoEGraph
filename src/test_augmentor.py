import torch
from torch_geometric.data import Data, Batch
from models.augmentor import MetaLearner, Augmentor
import random

# Configuration and dataset info for testing
config = {
    'augmentation': {
        'hidden_dim': 64,
        'num_layers': 3,
        'dropout': 0.5,
        'model_type': 'GIN',
        'pooling': 'mean'
    },
    'moe': {
        'num_experts': 4
    }
}
dataset_info = {
    'num_features': 16
}

# Generate multiple random graphs and batch them into mini-batches
num_graphs = 5
num_nodes = 10
num_edges = 20
batched_data_list = []
for _ in range(num_graphs):
    x = torch.randn((num_nodes, dataset_info['num_features']))
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    batch = torch.tensor([0] * (num_nodes // 2) + [1] * (num_nodes - num_nodes // 2))
    batched_data_list.append(Data(x=x, edge_index=edge_index, batch=batch))

batched_data = Batch.from_data_list(batched_data_list)

# Initialize MetaLearner and Augmentor
meta_learner = MetaLearner(config, dataset_info)
augmentor = Augmentor()

# Run MetaLearner to get augmentation parameters for each graph in the batch
param_sets = [meta_learner(data) for data in batched_data_list]

# Convert each parameter set to a torch.Tensor
param_sets = [[torch.tensor(params) for params in param_set] for param_set in param_sets]

# Print the original graph with batch information
print("Original Graph:")
print("Node features:", batched_data.x)
print("Edge index:", batched_data.edge_index)
print("Batch:", batched_data.batch)
print()

# Print the augmentation parameters with descriptions
for i, param_set in enumerate(param_sets):
    print(f"Augmentation Parameters for Graph {i+1}:")
    for j, params in enumerate(param_set):
        print(f"p_node_drop: {params[0]}, p_edge_drop: {params[1]}, p_edge_add: {params[2]}, noise_std: {params[3]}")
    print()

# Apply Augmentor to generate augmented graphs for each graph in the batch
augmented_graphs = [augmentor.apply(data, params) for data, param_set in zip(batched_data_list, param_sets) for params in param_set]

# Print the augmented graphs with batch information
for i, aug_data in enumerate(augmented_graphs):
    print(f"Augmented Graph {i+1}:")
    print("Node features:", aug_data.x)
    print("Edge index:", aug_data.edge_index)
    print("Batch:", aug_data.batch)
    print()