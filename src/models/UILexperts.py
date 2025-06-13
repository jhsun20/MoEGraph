import torch
import torch.nn as nn
import torch.nn.functional as F
from models.UIL import UILModel
from models.gnn_models import GIN
from entmax import entmax15, entmax_bisect

class MoEUILModel(nn.Module):
    def __init__(self, config, dataset_info):
        super(MoEUILModel, self).__init__()
        
        self.verbose = config['experiment']['debug']['verbose']  # Check if debug mode is enabled
        
        self.num_experts = config['model']['num_experts']
        self.aggregation_method = config['model']['aggregation']
        self.experts = nn.ModuleList()
        self.train_after = config['gate']['train_after']
        self.weight_ce = config['model']['weight_ce']
        self.weight_reg = config['model']['weight_reg']
        self.weight_sem = config['model']['weight_sem']
        self.weight_str = config['model']['weight_str']
        self.weight_div = config['model']['weight_div']
        self.weight_load = config['model']['weight_load']

        # Common parameters
        num_features = dataset_info['num_features']
        dropout = config['model']['dropout']
        pooling = config['model']['pooling']

        # Instantiate experts
        for _ in range(self.num_experts):
            expert = UILModel(config, dataset_info)
            self.experts.append(expert)

        # Instantiate gating mechanism
        gate_hidden_dim = config['gate']['hidden_dim']
        gate_depth = config['gate']['depth']
        self.gate = GIN(num_features, self.num_experts, gate_hidden_dim, gate_depth, dropout, pooling)
        self.entmax_alpha = config['gate']['entmax_alpha']

        if self.verbose:
            print(f"Initialized MoEUILModel with {self.num_experts} experts and a gating mechanism.")

    def forward(self, data, epoch):
        # Always give the gate the unaugmented input
        gate_weights = self.get_gate_weights(data, epoch)
        
        if self.verbose:
            print("Gate weights before entmax:", gate_weights)
        gate_weights = entmax_bisect(gate_weights, alpha=self.entmax_alpha, dim=-1)

        if self.verbose:
            print("Gating weights:", gate_weights)

        # Each expert output should be a dictionary with keys:
        # output = {logits, h_stable, h_orig, node_mask, edge_mask, loss_total, loss_ce, loss_reg, loss_sem, loss_str, cached_masks}
        expert_outputs = [expert(data, data.y) for expert in self.experts]
        if self.verbose:
            for i, output in enumerate(expert_outputs):
                print(f"Expert {i} output: {output['logits']}")

        # Get aggregated output should be a dictionary with keys:
        # output = {logits, loss_total, loss_ce, loss_reg, loss_sem, loss_str, gate_weights}
        aggregated_outputs = self.aggregate(expert_outputs, gate_weights, data)

        return aggregated_outputs

    def aggregate(self, expert_outputs, gate_weights, data):
        # Stack the expert outputs along a new dimension (using 'logits' from each expert)
        stacked_logits = torch.stack([eo['logits'] for eo in expert_outputs], dim=0)

        if self.verbose:
            print("Stacked expert outputs:", stacked_logits)
            print("Gate weights:", gate_weights)

        # Ensure gate_weights is of shape (num_experts, batch_size, 1) for broadcasting
        gate_weights = gate_weights.T.unsqueeze(-1).to(stacked_logits.device)  # (K, B, 1)
        if self.verbose:
            print("Gate weights shape after reshape:", gate_weights.shape)
            print("Gate weights (first 5):", gate_weights[:, :5, 0])  # print for first few examples

        aggregated_ce_loss = 0
        aggregated_reg_loss = 0
        aggregated_sem_loss = 0
        aggregated_str_loss = 0
        if self.aggregation_method == 'weighted_mean':
            # print('gate_weights', gate_weights.shape) # returns ([num_experts, batch_size, 1])
            # print('stacked_logits', stacked_logits.shape) # returns ([num_experts, batch_size, num_classes])
            weighted_logits = stacked_logits * gate_weights
            aggregated_logits = torch.sum(weighted_logits, dim=0)
            aggregated_ce_loss += self.compute_classification_loss(aggregated_logits, data.y)
            for i, output in enumerate(expert_outputs):
                weight = gate_weights[i, 0, 0]  # Extract scalar weight for this expert
                aggregated_reg_loss += weight * output['loss_reg']
                aggregated_sem_loss += weight * output['loss_sem']
                aggregated_str_loss += weight * output['loss_str']

            diversity_loss = self.compute_diversity_loss(expert_outputs)

            load_balance_loss = self.compute_load_balance_loss(gate_weights)

            aggregated_loss = self.weight_ce * aggregated_ce_loss + self.weight_reg * aggregated_reg_loss + self.weight_sem * aggregated_sem_loss + \
                self.weight_str * aggregated_str_loss + self.weight_div * diversity_loss + self.weight_load * load_balance_loss
            aggregated_outputs = {
                'logits': aggregated_logits,
                'loss_total': aggregated_loss,
                'loss_ce': aggregated_ce_loss,
                'loss_reg': aggregated_reg_loss,
                'loss_sem': aggregated_sem_loss,
                'loss_str': aggregated_str_loss,
                'loss_div': diversity_loss,
                'loss_load': load_balance_loss,
                'gate_weights': gate_weights
            }

        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")
        
        # TODO: ADD DIVERISTY AND LOAD BALANCE LOSS HERE

        if self.verbose:
            print(f"Aggregated outputs using {self.aggregation_method} method:", aggregated_outputs['logits'])

        return aggregated_outputs

    def get_gate_weights(self, data, epoch):
        batch_size = data.y.size(0)  # number of graphs in the batch

        if epoch < self.train_after:
            # Uniform weights for each expert and each example
            return torch.full((batch_size, self.num_experts),
                            1.0 / self.num_experts,
                            device=data.x.device)
        else:
            return self.gate(data)  # (batch_size, num_experts)

    def compute_diversity_loss(self, expert_outputs, div_loss_type='cosine_mask'):
        """
        Compute diversity loss across experts.

        Args:
            expert_outputs: list of expert output dicts, each with keys including 'node_mask' and 'edge_mask'
            div_loss_type (str): one of ['cosine_mask', 'cosine_embedding']

        Returns:
            diversity_loss: scalar torch tensor
        """
        def cosine_matrix(X):
            X = F.normalize(X, dim=1)
            return X @ X.T  # (K, K)

        def off_diagonal_mean(sim):
            K = sim.size(0)
            return (sim.sum() - torch.diagonal(sim).sum()) / (K * (K - 1))

        K = len(expert_outputs)

        if div_loss_type == 'cosine_mask':
            node_masks = torch.stack([eo['node_mask'].squeeze(-1).flatten() for eo in expert_outputs], dim=0)  # (K, N)
            edge_masks = torch.stack([eo['edge_mask'].squeeze(-1).flatten() for eo in expert_outputs], dim=0)  # (K, E)

            node_sim = cosine_matrix(node_masks)
            edge_sim = cosine_matrix(edge_masks)

            node_div = off_diagonal_mean(node_sim)
            edge_div = off_diagonal_mean(edge_sim)

            diversity_loss = (node_div + edge_div) / 2  # negative to encourage diversity
        else:
            raise ValueError(f"Unsupported diversity loss type: {div_loss_type}")

        return diversity_loss
    
    def compute_load_balance_loss(self, gate_weights, loss_type='kl'):
        """
        Compute load balancing loss based on the average gate distribution.

        Args:
            gate_weights (Tensor): shape (num_experts, batch_size, 1)
            loss_type (str): one of ['entropy', 'kl', 'l2']

        Returns:
            load_balance_loss (Tensor): scalar torch loss
        """
        # Reshape to (batch_size, num_experts)
        gate_weights = gate_weights.squeeze(-1).T  # (B, K)
        avg_weights = gate_weights.mean(dim=0)     # (K,)
        if self.verbose:
            print("Gate weights shape after reshape:", gate_weights)
            print("Average weights:", avg_weights)
        eps = 1e-8

        if loss_type == 'kl':
            uniform = torch.full_like(avg_weights, 1.0 / avg_weights.size(0))
            load_balance_loss = F.kl_div(
                input=(avg_weights + eps).log(),
                target=uniform,
                reduction='batchmean'
            )

        elif loss_type == 'l2':
            uniform = torch.full_like(avg_weights, 1.0 / avg_weights.size(0))
            load_balance_loss = F.mse_loss(avg_weights, uniform)

        else:
            raise ValueError(f"Unsupported load balance loss type: {loss_type}")

        return load_balance_loss
    
    def compute_classification_loss(self, pred, target):
        """
        Computes weighted cross-entropy loss with automatic class imbalance correction.
        
        Args:
            pred (Tensor): Logits of shape (B, C)
            target (Tensor): Ground truth labels of shape (B,)
        
        Returns:
            Tensor: Weighted cross-entropy loss
        """
        # Compute class counts from current batch
        num_classes = pred.size(1)
        class_counts = torch.bincount(target, minlength=num_classes).float()
        
        # Avoid division by zero
        class_counts[class_counts == 0] = 1.0

        # Inverse frequency weighting normalized to sum to 1
        weight = 1.0 / class_counts
        weight = weight / weight.sum()
        
        return F.cross_entropy(pred, target, weight=weight.to(pred.device))

