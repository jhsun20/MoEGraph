import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from models.gnn import GNN

class MoEGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, 
                 gnn_type, num_experts, gating='uniform', diversity_loss=False, 
                 expert_specific_aug=False):
        super(MoEGNN, self).__init__()
        
        self.num_experts = num_experts
        self.gating = gating
        self.diversity_loss = diversity_loss
        self.expert_specific_aug = expert_specific_aug
        
        # Create multiple expert GNNs
        self.experts = nn.ModuleList([
            GNN(in_channels, hidden_channels, hidden_channels, num_layers, dropout, gnn_type)
            for _ in range(num_experts)
        ])
        
        # Create gating network if needed
        if gating != 'uniform':
            self.gate_network = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, num_experts)
            )
        
        # Final classifier
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, batch):
        # Get expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x, edge_index, batch)
            expert_outputs.append(expert_out)
        
        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, hidden_dim]
        
        # Compute gating weights
        if self.gating == 'uniform':
            # Uniform weights
            gate_weights = torch.ones(expert_outputs.size(0), self.num_experts, device=expert_outputs.device)
            gate_weights = gate_weights / self.num_experts
        else:
            # Compute input graph features for gating
            graph_features = global_mean_pool(x, batch)
            gate_logits = self.gate_network(graph_features)
            gate_weights = F.softmax(gate_logits, dim=1)
        
        # Apply gating weights
        gate_weights = gate_weights.unsqueeze(-1)  # [batch_size, num_experts, 1]
        weighted_outputs = expert_outputs * gate_weights
        
        # Aggregate expert outputs
        combined_output = weighted_outputs.sum(dim=1)  # [batch_size, hidden_dim]
        
        # Classification
        output = self.classifier(combined_output)
        
        # Store intermediate results for loss computation
        self.expert_outputs = expert_outputs
        self.gate_weights = gate_weights
        
        return output
    
    def get_diversity_loss(self):
        """Compute diversity loss to encourage expert specialization"""
        if not self.diversity_loss or not hasattr(self, 'expert_outputs'):
            return 0.0
        
        # Compute pairwise cosine similarity between expert outputs
        expert_outputs = self.expert_outputs  # [batch_size, num_experts, hidden_dim]
        
        # Normalize expert outputs
        normalized_outputs = F.normalize(expert_outputs, p=2, dim=2)
        
        # Compute pairwise cosine similarity
        similarity_matrix = torch.bmm(
            normalized_outputs, 
            normalized_outputs.transpose(1, 2)
        )  # [batch_size, num_experts, num_experts]
        
        # Zero out diagonal elements (self-similarity)
        mask = torch.eye(self.num_experts, device=similarity_matrix.device).unsqueeze(0)
        similarity_matrix = similarity_matrix * (1 - mask)
        
        # Compute mean similarity
        diversity_loss = similarity_matrix.sum() / (similarity_matrix.size(0) * self.num_experts * (self.num_experts - 1))
        
        return diversity_loss 