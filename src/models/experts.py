import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gnn_models import GCN, GIN, GraphSAGE

class MoEModel(nn.Module):
    def __init__(self, config, dataset_info):
        super(MoEModel, self).__init__()
        
        self.num_experts = config['moe']['num_experts']
        self.aggregation_method = config['moe']['aggregation']
        self.experts = nn.ModuleList()
        
        # Common parameters
        num_features = dataset_info['num_features']
        num_classes = dataset_info['num_classes']
        hidden_dim = config['model']['hidden_dim']
        num_layers = config['model']['num_layers']
        dropout = config['model']['dropout']
        pooling = config['model']['pooling']
        
        # Instantiate experts
        for _ in range(self.num_experts):
            expert = GIN(num_features, num_classes, hidden_dim, num_layers, dropout, pooling)
            self.experts.append(expert)

    def forward(self, data):
        # Forward pass through all experts
        expert_outputs = [expert(data) for expert in self.experts]
        
        if self.training:
            # During training, return individual expert outputs
            return expert_outputs
        else:
            # print('all_outputs', expert_outputs)
            # Stack the expert outputs along a new dimension
            stacked_outputs = torch.stack(expert_outputs, dim=0)
            # print('Stacked Outputs:', stacked_outputs)

            if self.aggregation_method == 'mean':
                aggregated_outputs = torch.mean(stacked_outputs, dim=0)
                # print('Aggregation Method: Mean')
                # print('Aggregated Outputs:', aggregated_outputs)
            elif self.aggregation_method == 'majority_vote':
                # Implement majority vote logic
                aggregated_outputs = torch.mode(stacked_outputs, dim=0)[0]
                # print('Aggregation Method: Majority Vote')
                # print('Aggregated Outputs:', aggregated_outputs)
            else:
                raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")
            
            return aggregated_outputs
            
    def test_moe_model(self, data):
        # Simulate training mode
        self.train()
        training_outputs = self.forward(data)
        assert len(training_outputs) == self.num_experts, "Training outputs should match the number of experts."
        
        # Print individual training outputs
        for i, output in enumerate(training_outputs):
            print(f"Training output from expert {i}: {output}")
        
        # Simulate evaluation mode
        self.eval()
        evaluation_output = self.forward(data)
        assert evaluation_output.shape == training_outputs[0].shape, "Evaluation output shape should match individual expert output shape."
        
        # Print evaluation output
        print(f"Evaluation output: {evaluation_output}")
        
        print("MoEModel test passed.")