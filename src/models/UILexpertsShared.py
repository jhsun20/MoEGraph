import torch
import torch.nn as nn
import torch.nn.functional as F
from models.UIL import UILModelSharedEncoder, Experts
from models.gnn_models import GIN
from entmax import entmax15, entmax_bisect
from torch_geometric.data import Batch

class MoEUILModelSharedEncoder(nn.Module):
    def __init__(self, config, dataset_info):
        super(MoEUILModelSharedEncoder, self).__init__()
        
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
        
        # Initialize current epoch for gate training
        self.current_epoch = 0

        # Common parameters
        num_features = dataset_info['num_features']
        dropout = config['model']['dropout']
        pooling = config['model']['pooling']

        # Instantiate single shared encoder model (contains multiple experts internally)
        #self.shared_model = UILModelSharedEncoder(config, dataset_info)
        self.shared_model = Experts(config, dataset_info)

        # Instantiate gating mechanism
        gate_hidden_dim = config['gate']['hidden_dim']
        gate_depth = config['gate']['depth']
        self.gate = GIN(num_features, self.num_experts, gate_hidden_dim, gate_depth, dropout, pooling)
        self.entmax_alpha = config['gate']['entmax_alpha']

        if self.verbose:
            print(f"Initialized MoEUILModel with {self.num_experts} experts and a gating mechanism.")

    def forward(self, data):
        # Always give the gate the unaugmented input
        gate_weights = self.get_gate_weights(data)
        
        if self.verbose:
            print("Gate weights before entmax:", gate_weights)
        
        if self.entmax_alpha > 1:
            gate_weights = entmax_bisect(gate_weights, alpha=self.entmax_alpha, dim=-1)
        else:
            gate_weights = F.softmax(gate_weights, dim=-1)

        if self.verbose:
            print("Gating weights:", gate_weights)

        # Get output from shared model (contains all experts)
        # output = {h_stable, h_orig, node_masks, edge_masks, expert_logits, loss_total_list, loss_ce_list, loss_reg_list, loss_sem_list, loss_str_list, cached_masks}
        shared_output = self.shared_model(data, data.y)
    
        # Get aggregated output should be a dictionary with keys:
        # output = {logits, loss_total, loss_ce, loss_reg, loss_sem, loss_str, gate_weights}
        aggregated_outputs = self.aggregate(shared_output, gate_weights, data)

        return aggregated_outputs

    def aggregate(self, shared_output, gate_weights, data):
        # Get expert logits from shared model output
        expert_logits = shared_output['expert_logits']  # (batch_size, num_experts, num_classes)
        
        # Transpose to match expected shape (num_experts, batch_size, num_classes)
        stacked_logits = expert_logits.transpose(0, 1)

        if self.verbose:
            print("Stacked expert outputs:", stacked_logits.shape)
            print("Stacked expert outputs:", stacked_logits)

        # Ensure gate_weights is of shape (num_experts, batch_size, 1) for broadcasting
        gate_weights = gate_weights.T.unsqueeze(-1).to(stacked_logits.device)  # (K, B, 1)
        
        # Assert that gate weights still sum to 1 after reshaping
        gate_weights_sum = gate_weights.squeeze(-1).sum(dim=0)  # Sum across experts for each batch item
        assert torch.allclose(gate_weights_sum, torch.ones_like(gate_weights_sum), atol=1e-6), \
            f"Gate weights after reshape do not sum to 1: {gate_weights_sum}"
        
        if self.verbose:
            print("Gate weights shape after reshape:", gate_weights.shape)
            print("Gate weights (first 5):", gate_weights[:, :5, 0])  # print for first few examples

        if self.aggregation_method == 'weighted_mean':
            # Weighted aggregation of expert logits
            weighted_logits = stacked_logits * gate_weights
            aggregated_logits = torch.sum(weighted_logits, dim=0)
            
            # Compute classification loss on aggregated logits
            aggregated_ce_loss = self.compute_classification_loss(aggregated_logits, data.y)
            
            # Aggregate individual expert losses using gate weights
            aggregated_reg_loss = torch.tensor(0.0, device=stacked_logits.device)
            aggregated_sem_loss = torch.tensor(0.0, device=stacked_logits.device)
            aggregated_str_loss = torch.tensor(0.0, device=stacked_logits.device)
            
            if shared_output['loss_ce_list'] is not None:
                # Get expert-specific losses from shared model
                ce_loss_list = shared_output['loss_ce_list']  # (num_experts,)
                reg_loss_list = shared_output['loss_reg_list']  # (num_experts,)
                sem_loss_list = shared_output['loss_sem_list']  # (num_experts,)
                str_loss_list = shared_output['loss_str_list']  # (num_experts,)
                
                # Average gate weights across batch for loss aggregation
                avg_gate_weights = gate_weights.squeeze(-1).mean(dim=1)  # (num_experts,)
                
                # Weighted aggregation of expert losses
                for i in range(self.num_experts):
                    weight = avg_gate_weights[i]
                    aggregated_reg_loss += weight * reg_loss_list[i]
                    aggregated_sem_loss += weight * sem_loss_list[i]
                    aggregated_str_loss += weight * str_loss_list[i]

            # Compute diversity and load balance losses
            diversity_loss = self.compute_diversity_loss(shared_output)
            load_balance_loss = self.compute_load_balance_loss(gate_weights)

            # Combine all losses
            aggregated_loss = (self.weight_ce * aggregated_ce_loss + 
                             self.weight_reg * aggregated_reg_loss + 
                             self.weight_sem * aggregated_sem_loss + 
                             self.weight_str * aggregated_str_loss + 
                             self.weight_div * diversity_loss + 
                             self.weight_load * load_balance_loss)
            
            aggregated_outputs = {
                'logits': aggregated_logits,
                'loss_total': aggregated_loss,
                'loss_ce': aggregated_ce_loss,
                'loss_reg': aggregated_reg_loss,
                'loss_sem': aggregated_sem_loss,
                'loss_str': aggregated_str_loss,
                'loss_div': diversity_loss,
                'loss_load': load_balance_loss,
                'gate_weights': gate_weights,
                'rho': shared_output['rho'],
                'expert_logits': expert_logits  # Keep expert logits for analysis
            }

        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")

        if self.verbose:
            print(f"Aggregated outputs using {self.aggregation_method} method:", aggregated_outputs['logits'])

        return aggregated_outputs

    def get_gate_weights(self, data):
        batch_size = data.y.size(0)  # number of graphs in the batch

        if self.current_epoch < self.train_after:
            # Uniform weights for each expert and each example
            return torch.full((batch_size, self.num_experts),
                            1.0 / self.num_experts,
                            device=data.x.device)
        else:
            return self.gate(data)  # (batch_size, num_experts)

    def compute_diversity_loss(self, shared_output, div_loss_type='cosine_mask'):
        """
        Compute diversity loss across experts.

        Args:
            shared_output: dict with keys including 'node_masks' and 'edge_masks' (already stacked)
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

        if div_loss_type == 'cosine_mask':
            # Get masks from shared output - shape: (N, num_experts, 1) and (E, num_experts, 1)
            node_masks = shared_output['node_masks']  # (N, num_experts, 1)
            edge_masks = shared_output['edge_masks']  # (E, num_experts, 1)
            
            # Reshape to (num_experts, N) and (num_experts, E)
            node_masks = node_masks.squeeze(-1).transpose(0, 1)  # (num_experts, N)
            edge_masks = edge_masks.squeeze(-1).transpose(0, 1)  # (num_experts, E)

            # Compute cosine similarity matrices
            node_sim = cosine_matrix(node_masks)
            edge_sim = cosine_matrix(edge_masks)

            # Compute diversity (lower similarity = higher diversity)
            node_div = off_diagonal_mean(node_sim)
            edge_div = off_diagonal_mean(edge_sim)
            
            # Average diversity across node and edge masks
            diversity_loss = (node_div + edge_div) / 2
            
            if self.verbose:
                print(f"Diversity loss - node: {node_div:.4f}, edge: {edge_div:.4f}, avg: {diversity_loss:.4f}")
                
        elif div_loss_type == 'cosine_embedding':
            # Alternative: compute diversity based on expert embeddings
            h_stable_list = shared_output['h_stable_list']  # (batch_size, num_experts, hidden_dim)
            
            # Average embeddings across batch
            avg_embeddings = h_stable_list.mean(dim=0)  # (num_experts, hidden_dim)
            
            # Compute cosine similarity matrix
            sim_matrix = cosine_matrix(avg_embeddings)
            
            # Compute diversity
            diversity_loss = off_diagonal_mean(sim_matrix)
            
            if self.verbose:
                print(f"Embedding diversity loss: {diversity_loss:.4f}")
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
    
    def compute_classification_loss(self, pred, target, use_weights=False):
        """
        Computes cross-entropy loss with optional class imbalance correction.
        
        Args:
            pred (Tensor): Logits of shape (B, C)
            target (Tensor): Ground truth labels of shape (B,)
            use_weights (bool): Whether to use class weights for imbalance correction. Default: False
        
        Returns:
            Tensor: Cross-entropy loss
        """
        if use_weights:
            # Compute class counts from current batch
            num_classes = pred.size(1)
            class_counts = torch.bincount(target, minlength=num_classes).float()
            
            # Avoid division by zero
            class_counts[class_counts == 0] = 1.0

            # Inverse frequency weighting normalized to sum to 1
            weight = 1.0 / class_counts
            weight = weight / weight.sum()
            
            return F.cross_entropy(pred, target, weight=weight.to(pred.device))
        else:
            return F.cross_entropy(pred, target)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def get_expert_outputs(self, data):
        """
        Get individual expert outputs without aggregation for analysis.
        
        Args:
            data: Input data
            
        Returns:
            dict: Contains individual expert outputs and gate weights
        """
        # Get gate weights
        gate_weights = self.get_gate_weights(data)
        gate_weights = entmax_bisect(gate_weights, alpha=self.entmax_alpha, dim=-1)
        
        # Get shared model outputs
        shared_output = self.shared_model(data, data.y)
        
        return {
            'expert_logits': shared_output['expert_logits'],
            'expert_embeddings': shared_output['h_stable_list'],
            'node_masks': shared_output['node_masks'],
            'edge_masks': shared_output['edge_masks'],
            'gate_weights': gate_weights,
            'individual_losses': {
                'ce_losses': shared_output['loss_ce_list'],
                'reg_losses': shared_output['loss_reg_list'],
                'sem_losses': shared_output['loss_sem_list'],
                'str_losses': shared_output['loss_str_list']
            }
        }
