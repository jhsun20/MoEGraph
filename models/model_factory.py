import torch
import torch.nn as nn
from models.gnn import GNN
from models.moe import MoEGNN

def get_model(config, in_channels, out_channels):
    """Factory function to create models based on config"""
    
    # Get model parameters from config
    model_config = config['model']
    gnn_type = model_config['type']
    hidden_channels = model_config['hidden_channels']
    num_layers = model_config['num_layers']
    dropout = model_config['dropout']
    num_experts = model_config['num_experts']
    
    # If MoE is enabled and num_experts > 1, create MoE model
    if config['moe']['enabled'] and num_experts > 1:
        return MoEGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=gnn_type,
            num_experts=num_experts,
            gating=config['moe']['gating'],
            diversity_loss=config['moe']['diversity_loss'],
            expert_specific_aug=config['augmentation']['expert_specific']
        )
    else:
        # Create standard GNN model
        return GNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=gnn_type
        ) 