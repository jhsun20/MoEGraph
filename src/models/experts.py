import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gnn_models import GCN, GIN, GraphSAGE
from entmax import entmax15, entmax_bisect

class MoEModel(nn.Module):
    def __init__(self, config, dataset_info):
        super(MoEModel, self).__init__()
        
        self.verbose = config['experiment']['debug']['verbose']  # Check if debug mode is enabled
        
        self.num_experts = config['moe']['num_experts']
        self.aggregation_method = config['moe']['aggregation']
        self.augmentation_enabled = config['augmentation']['enable']
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

        # Instantiate gating mechanism
        gate_hidden_dim = config['gate']['hidden_dim']
        gate_depth = config['gate']['depth']
        self.gate = GIN(num_features, self.num_experts, gate_hidden_dim, gate_depth, dropout, pooling)
        self.entmax_alpha = config['gate']['entmax_alpha']

        if self.verbose:
            print(f"Initialized MoEModel with {self.num_experts} experts and a gating mechanism.")

    def forward(self, data):
        # Always give the gate the unaugmented input
        gate_input = data[0] if isinstance(data, list) else data
        gate_weights = self.gate(gate_input)
        if self.verbose:
            print("Gate weights before entmax:", gate_weights)
        gate_weights = entmax_bisect(gate_weights, alpha=self.entmax_alpha, dim=-1)

        if self.verbose:
            print("Gating weights:", gate_weights)

        # If using augmented data, expect a list of Data objects (one per expert)
        if self.augmentation_enabled and isinstance(data, list):
            assert len(data) == self.num_experts + 1, "Number of augmented graphs must match number of experts."
            expert_outputs = [self.experts[i](data[i]) for i in range(self.num_experts)]
            if self.verbose:
                for i, output in enumerate(expert_outputs):
                    print(f"Expert {i} received augmented view. Output: {output}")
        else:
            # If augmentation is off, all experts get the same graph
            expert_outputs = [expert(data) for expert in self.experts]
            if self.verbose:
                print("Augmentation disabled. All experts received the same input.")
                for i, output in enumerate(expert_outputs):
                    print(f"Expert {i} output: {output}")

        return expert_outputs, gate_weights

    def aggregate(self, expert_outputs, gate_weights):
        # Stack the expert outputs along a new dimension
        stacked_outputs = torch.stack(expert_outputs, dim=0)

        if self.verbose:
            print("Stacked expert outputs:", stacked_outputs)
            print("Gate weights:", gate_weights)

        # Ensure gate_weights is of shape (num_experts, batch_size, 1) for broadcasting
        gate_weights = gate_weights.transpose(0, 1).unsqueeze(-1)

        if self.aggregation_method == 'mean':
            aggregated_outputs = torch.mean(stacked_outputs, dim=0)
        elif self.aggregation_method == 'weighted_mean':
            weighted_outputs = stacked_outputs * gate_weights
            aggregated_outputs = torch.sum(weighted_outputs, dim=0)
        elif self.aggregation_method == 'majority_vote':
            aggregated_outputs = torch.mode(stacked_outputs, dim=0)[0]
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")

        if self.verbose:
            print(f"Aggregated outputs using {self.aggregation_method} method:", aggregated_outputs)

        return aggregated_outputs

    def test_moe_model(self, data):
        self.train()
        training_outputs, gate_weights = self.forward(data)
        assert len(training_outputs) == self.num_experts, "Training outputs should match the number of experts."

        for i, output in enumerate(training_outputs):
            print(f"Training output from expert {i}: {output}")

        self.eval()
        evaluation_output = self.aggregate(training_outputs, gate_weights)
        assert evaluation_output.shape == training_outputs[0].shape, "Evaluation output shape should match individual expert output shape."

        print(f"Evaluation output: {evaluation_output}")
        print("MoEModel test passed.")
