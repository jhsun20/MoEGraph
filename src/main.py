import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import datetime
import gc
from torch_geometric.nn import DataParallel

from data.dataset_loader import load_dataset
from models.model_factory import get_model
from utils.logger import Logger
from utils.train import train
from utils.tuning import run_optuna_tuning


def run(config):
    if config.get('experiment', {}).get('hyper_search', {}).get('enable', False):
        now = datetime.datetime.now()
        today_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H-%M-%S")

        output_dir = os.path.join(
            "results",
            f"{config['experiment']['name']}_{config['dataset']['dataset_name']}_tuning_{today_date}_{current_time}"
        )

        os.makedirs(output_dir, exist_ok=True)
        best_config = run_optuna_tuning(config=config, phase=1, output_dir=output_dir)
        # best_config = run_optuna_tuning(config=best_config, phase=2, output_dir=output_dir)
        if config.get('experiment', {}).get('hyper_search', {}).get('test_after', False):
            best_config['experiment']['hyper_search']['enable'] = False
            train(best_config)
    else:
        train(config)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a GNN model with configurable parameters')
    parser.add_argument('--config', type=str, default='config/config.yaml', 
                        help='Path to the config file')
    
    # Experiment arguments
    parser.add_argument('--experiment_name', type=str, help='Experiment name')
    parser.add_argument('--device', type=str, help='Device to use (cuda, cpu)')
    parser.add_argument('--debug_enable', type=str, help='Enable debug mode (true/false)')
    parser.add_argument('--debug_num_samples', type=int, help='Number of samples to use in debug mode')
    parser.add_argument('--debug_epochs', type=int, help='Number of epochs to run in debug mode')
    parser.add_argument('--verbose', type=str, help='Enable verbose mode (true/false)')
    parser.add_argument('--seeds', type=int, nargs='+', help='List of seeds for experiments')
    parser.add_argument('--hyper_search_enable', type=str, help='Enable hyperparameter search (true/false)')
    parser.add_argument('--hyper_search_n_trials', type=int, help='Number of trials for hyperparameter search')

    # Dataset arguments
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--task_type', type=str, help='Task type (e.g., graph_classification)')
    parser.add_argument('--dataset_path', type=str, help='Path to datasets')
    parser.add_argument('--shift_type', type=str, help='Shift type (covariate, concept, no_shift)')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--num_workers', type=int, help='Number of workers')
    parser.add_argument('--domain', type=str, help='Domain (scaffold, basis, size)')

    # Model arguments
    parser.add_argument('--model_type', type=str, help='Model type (GIN, GCN, GraphSAGE, etc.)')
    parser.add_argument('--parallel', type=str, help='Enable model parallelism (true/false)')
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, help='Number of layers')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    parser.add_argument('--pooling', type=str, help='Pooling method (mean, sum, max)')
    parser.add_argument('--global_pooling', type=str, help='Global pooling method (mean, sum, max)')
    parser.add_argument('--weight_sem', type=float, help='Weight for semantic loss')
    parser.add_argument('--weight_reg', type=float, help='Weight for regularization loss')
    parser.add_argument('--weight_ce', type=float, help='Weight for cross-entropy loss')
    parser.add_argument('--weight_div', type=float, help='Weight for diversity loss')
    parser.add_argument('--weight_load', type=float, help='Weight for load balancing loss')
    parser.add_argument('--num_experts', type=int, help='Number of experts for MoE models')
    parser.add_argument('--aggregation', type=str, help='Aggregation method (mean, majority_vote, weighted_mean)')
    parser.add_argument('--weight_la', type=float, help='Weight for local accuracy loss')
    parser.add_argument('--weight_ea', type=float, help='Weight for expert accuracy loss')
    parser.add_argument('--weight_str', type=float, help='Weight for structural loss')
    parser.add_argument('--adv_warmup_epochs', type=int, help='Number of epochs for warmup')
    parser.add_argument('--adv_ramp_epochs', type=int, help='Number of epochs for ramp')
    parser.add_argument('--rho_edge', type=float, help='Rho edge for expert')

    # Training arguments
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, help='Early stopping patience')
    parser.add_argument('--early_stopping_min_delta', type=float, help='Minimum delta for early stopping')

    # Logging arguments for early stopping
    parser.add_argument('--wandb_enable', type=str, help='Enable Weights & Biases logging (true/false)')
    parser.add_argument('--wandb_project', type=str, help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, help='Weights & Biases entity name')
    parser.add_argument('--log_interval', type=int, help='Logging interval')
    parser.add_argument('--save_model', type=str, help='Save model (true/false)')

    # Gating module arguments
    parser.add_argument('--gate_activation', type=str, help='Gating activation function')
    parser.add_argument('--entmax_alpha', type=float, help='Entmax alpha value')
    parser.add_argument('--gate_model', type=str, help='Gating model type')
    parser.add_argument('--gate_depth', type=int, help='Gating model depth')
    parser.add_argument('--gate_hidden_dim', type=int, help='Gating model hidden dimension')
    parser.add_argument('--train_after', type=int, help='Train gating model after specified epochs')
    parser.add_argument('--finetune_epochs', type=int, help='Finetune gating model for specified epochs')

    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments if provided
    if args.experiment_name:
        config['experiment']['name'] = args.experiment_name
    if args.device:
        config['experiment']['device'] = args.device
    if args.debug_enable:
        config['experiment']['debug']['enable'] = args.debug_enable.lower() == 'true'
    if args.debug_num_samples:
        config['experiment']['debug']['num_samples'] = args.debug_num_samples
    if args.debug_epochs:
        config['experiment']['debug']['epochs'] = args.debug_epochs
    if args.verbose:
        config['experiment']['debug']['verbose'] = args.verbose.lower() == 'true'
    if args.seeds:
        config['experiment']['seeds'] = args.seeds
    if args.hyper_search_enable:
        config['experiment']['hyper_search']['enable'] = args.hyper_search_enable.lower() == 'true'
    if args.hyper_search_n_trials:
        config['experiment']['hyper_search']['n_trials'] = args.hyper_search_n_trials

    if args.dataset_name:
        config['dataset']['dataset_name'] = args.dataset_name
    if args.task_type:
        config['dataset']['task_type'] = args.task_type
    if args.dataset_path:
        config['dataset']['path'] = args.dataset_path
    if args.shift_type:
        config['dataset']['shift_type'] = args.shift_type
    if args.batch_size:
        config['dataset']['batch_size'] = args.batch_size
    if args.num_workers:
        config['dataset']['num_workers'] = args.num_workers
    if args.domain:
        config['dataset']['domain'] = args.domain

    if args.model_type:
        config['model']['type'] = args.model_type
    if args.parallel:
        config['model']['parallel'] = args.parallel.lower() == 'true'
    if args.hidden_dim:
        config['model']['hidden_dim'] = args.hidden_dim
    if args.num_layers:
        config['model']['num_layers'] = args.num_layers
    if args.dropout:
        config['model']['dropout'] = args.dropout
    if args.pooling:
        config['model']['pooling'] = args.pooling
    if args.global_pooling:
        config['model']['global_pooling'] = args.global_pooling
    if args.weight_str:
        config['model']['weight_str'] = args.weight_str
    if args.weight_sem:
        config['model']['weight_sem'] = args.weight_sem
    if args.weight_reg:
        config['model']['weight_reg'] = args.weight_reg
    if args.weight_ce:
        config['model']['weight_ce'] = args.weight_ce
    if args.weight_div:
        config['model']['weight_div'] = args.weight_div
    if args.weight_load:
        config['model']['weight_load'] = args.weight_load
    if args.num_experts:
        config['model']['num_experts'] = args.num_experts
    if args.aggregation:
        config['model']['aggregation'] = args.aggregation
    if args.weight_la:
        config['model']['weight_la'] = args.weight_la
    if args.weight_ea:
        config['model']['weight_ea'] = args.weight_ea
    if args.weight_str:
        config['model']['weight_str'] = args.weight_str
    if args.adv_warmup_epochs:
        config['model']['adv_warmup_epochs'] = args.adv_warmup_epochs
    if args.adv_ramp_epochs:
        config['model']['adv_ramp_epochs'] = args.adv_ramp_epochs
    if args.rho_edge:
        config['model']['rho_edge'] = args.rho_edge

    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.lr:
        config['training']['lr'] = args.lr
    if args.weight_decay:
        config['training']['weight_decay'] = args.weight_decay
    if args.patience:
        config['training']['early_stopping']['patience'] = args.patience
    if args.early_stopping_min_delta:
        config['training']['early_stopping']['min_delta'] = args.early_stopping_min_delta

    if args.wandb_enable:
        config['logging']['wandb']['enable'] = args.wandb_enable.lower() == 'true'
    if args.wandb_project:
        config['logging']['wandb']['project'] = args.wandb_project
    if args.wandb_entity:
        config['logging']['wandb']['entity'] = args.wandb_entity
    if args.log_interval:
        config['logging']['log_interval'] = args.log_interval
    if args.save_model:
        config['logging']['save_model'] = args.save_model.lower() == 'true'

    if args.gate_activation:
        config['gate']['activation'] = args.gate_activation
    if args.entmax_alpha:
        config['gate']['entmax_alpha'] = args.entmax_alpha
    if args.gate_model:
        config['gate']['model'] = args.gate_model
    if args.gate_depth:
        config['gate']['depth'] = args.gate_depth
    if args.gate_hidden_dim:
        config['gate']['hidden_dim'] = args.gate_hidden_dim
    if args.train_after:
        config['gate']['train_after'] = args.train_after
    if args.finetune_epochs:
        config['gate']['finetune_epochs'] = args.finetune_epochs

    # Run training
    run(config) 