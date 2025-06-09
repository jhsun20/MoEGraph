import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse

from data.dataset_loader import load_dataset
from models.model_factory import get_model
from utils.logger import Logger
from utils.metrics import compute_metrics

def train_epoch(model, loader, optimizer, dataset_info, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    all_outputs = []
    all_targets = []
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = F.cross_entropy(output, data.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * data.num_graphs
        all_outputs.append(output.detach())
        all_targets.append(data.y.detach())
    
    # Compute epoch metrics
    avg_loss = total_loss / len(loader.dataset)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_outputs, all_targets, dataset_info['metric'])
    metrics['loss'] = avg_loss
    
    return metrics

def evaluate(model, loader, device, metric_type):
    """Evaluate model on validation or test set."""
    model.eval()
    total_loss = 0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # Forward pass
            output = model(data)
            loss = F.cross_entropy(output, data.y)
            
            # Track metrics
            total_loss += loss.item() * data.num_graphs
            all_outputs.append(output)
            all_targets.append(data.y)
    
    # Compute metrics
    avg_loss = total_loss / len(loader.dataset)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_outputs, all_targets, metric_type)
    metrics['loss'] = avg_loss
    
    return metrics

def train_epoch_moe(model, loader, optimizer, dataset_info, device):
    """Train MoE model for one epoch, allowing each expert to be trained independently."""
    model.train()
    total_loss = 0
    all_outputs = [[] for _ in range(model.num_experts)]
    all_targets = []
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass through each expert
        expert_outputs = model(data)
        losses = []
        for i, output in enumerate(expert_outputs):
            loss = F.cross_entropy(output, data.y)
            losses.append(loss)
            
            # Zero gradients for all experts
            for expert in model.experts:
                for param in expert.parameters():
                    param.grad = None
            
            # Backward pass for the current expert
            loss.backward()
            optimizer.step()
            
            # Check and print gradients to verify only the current expert is updated
            # for j, expert in enumerate(model.experts):
            #     has_gradients = any(param.grad is not None for param in expert.parameters())
            #     print(f"Expert {j} {'has' if has_gradients else 'does not have'} gradients.")
            
            # Track metrics
            total_loss += loss.item() * data.num_graphs
            all_outputs[i].append(output.detach())
            
            # Print statement to confirm individual expert training
            # print(f"Training Expert {i}: Loss = {loss.item():.4f}")
        
        all_targets.append(data.y.detach())
        
        # Print progress for each expert
        # for i, loss in enumerate(losses):
        #     print(f"Expert {i} - Loss: {loss.item():.4f}")
    
    # Compute epoch metrics
    metrics = evaluate_moe(model, loader, device, dataset_info['metric'])
    model.train()

    return metrics

def evaluate_moe(model, loader, device, metric_type):
    """Evaluate MoE model on validation or test set."""
    model.eval()
    total_loss = 0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # Forward pass
            output = model(data)  # Aggregated output from all experts
            loss = F.cross_entropy(output, data.y)
            
            # Track metrics
            total_loss += loss.item() * data.num_graphs
            all_outputs.append(output)
            all_targets.append(data.y)
    
    # Compute metrics
    avg_loss = total_loss / len(loader.dataset)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_outputs, all_targets, metric_type)
    metrics['loss'] = avg_loss
    
    return metrics

def train(config):
    """Main training function."""
    # Set random seeds for reproducibility
    seed = config['experiment']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Set device
    device = torch.device(config['experiment']['device'] if torch.cuda.is_available() else "cpu")
    
    # Initialize logger
    logger = Logger(config)
    logger.logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.logger.info("Loading dataset...")
    data_loaders = load_dataset(config)
    train_loader = data_loaders['train_loader']
    val_loader = data_loaders['val_loader']  # OOD validation
    test_loader = data_loaders['test_loader']  # OOD test
    id_val_loader = data_loaders['id_val_loader']  # In-distribution validation
    id_test_loader = data_loaders['id_test_loader']  # In-distribution test
    dataset_info = data_loaders['dataset_info']
    
    logger.logger.info(f"Dataset info: {dataset_info}")
    metric_type = dataset_info['metric']
    
    # Initialize model
    logger.logger.info(f"Initializing {config['model']['type']} model...")
    model = get_model(config, dataset_info)
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Training loop
    logger.logger.info("Starting training...")
    best_val_acc = 0
    patience_counter = 0
    
    # Adjust epochs if in debug mode
    num_epochs = config['experiment']['debug']['epochs'] if config['experiment']['debug']['enable'] else config['training']['epochs']
    
    for epoch in range(1, num_epochs + 1):
        # Train
        if config['model']['type'] == 'moe':
            train_metrics = train_epoch_moe(model, train_loader, optimizer, dataset_info, device)
            # Validate on OOD validation set
            val_metrics = evaluate_moe(model, val_loader, device, metric_type)
            # Validate on in-distribution validation set
            id_val_metrics = evaluate_moe(model, id_val_loader, device, metric_type)
        else:
            train_metrics = train_epoch(model, train_loader, optimizer, dataset_info, device)
            # Validate on OOD validation set
            val_metrics = evaluate(model, val_loader, device, metric_type)
            # Validate on in-distribution validation set
            id_val_metrics = evaluate(model, id_val_loader, device, metric_type)
        
        # Log metrics
        logger.log_metrics(train_metrics, epoch, phase="train")
        logger.log_metrics(val_metrics, epoch, phase="val_ood")
        logger.log_metrics(id_val_metrics, epoch, phase="val_id")
        
        # Early stopping based on OOD validation performance
        # Use the appropriate metric for early stopping
        primary_metric = 'accuracy' if metric_type == 'Accuracy' else metric_type.lower().replace('-', '_')
        if primary_metric not in val_metrics:
            primary_metric = list(val_metrics.keys())[0]  # Fallback to first metric
            
        current_metric = val_metrics[primary_metric]
        # For error metrics like RMSE and MAE, lower is better
        is_better = (current_metric < best_val_acc - config['training']['early_stopping']['min_delta']) if metric_type in ['RMSE', 'MAE'] else (current_metric > best_val_acc + config['training']['early_stopping']['min_delta'])
        
        if is_better:
            best_val_acc = current_metric
            patience_counter = 0
            logger.save_model(model, epoch, val_metrics)
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping']['patience']:
                logger.logger.info(f"Early stopping at epoch {epoch}")
                break
    
    # Final evaluation on test sets
    logger.logger.info("Evaluating on test sets...")
    test_ood_metrics = evaluate(model, test_loader, device, metric_type)
    test_id_metrics = evaluate(model, id_test_loader, device, metric_type)
    
    logger.log_metrics(test_ood_metrics, epoch, phase="test_ood")
    logger.log_metrics(test_id_metrics, epoch, phase="test_id")
    
    # Save results
    results_dir = os.path.join(
        "results", 
        f"{config['experiment']['name']}_{config['model']['type']}"
    )
    os.makedirs(results_dir, exist_ok=True)
    
    results_path = os.path.join(results_dir, "metrics.yaml")
    with open(results_path, 'w') as f:
        yaml.dump({
            'train': train_metrics,
            'val_ood': val_metrics,
            'val_id': id_val_metrics,
            'test_ood': test_ood_metrics,
            'test_id': test_id_metrics
        }, f)
    
    logger.logger.info(f"Results saved to {results_path}")
    logger.close()
    
    return {
        'test_ood': test_ood_metrics,
        'test_id': test_id_metrics
    }

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a GNN model with configurable parameters')
    parser.add_argument('--config', type=str, default='config/config.yaml', 
                        help='Path to the config file')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, help='Model type (GIN, GCN, GraphSAGE, etc.)')
    parser.add_argument('--num_experts', type=int, help='Number of experts for MoE models')
    
    # Training arguments
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--device', type=str, help='Device to use (cuda, cpu)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    # Augmentation arguments
    parser.add_argument('--augmentation', type=str, help='Enable or disable augmentation (true/false)')
    parser.add_argument('--aug_strategy', type=str, help='Augmentation strategy')
    parser.add_argument('--expert_specific_aug', type=str, help='Enable expert-specific augmentation (true/false)')
    
    # Gating arguments
    parser.add_argument('--gating', type=str, help='Gating mechanism (soft_attention, learned_voting, etc.)')
    
    # Debug mode arguments
    parser.add_argument('--debug', type=str, help='Enable debug mode (true/false)')
    parser.add_argument('--debug_samples', type=int, help='Number of samples to use in debug mode')
    parser.add_argument('--debug_epochs', type=int, help='Number of epochs to run in debug mode')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments if provided
    if args.model_type:
        config['model']['type'] = args.model_type
    if args.num_experts:
        config['model']['num_experts'] = args.num_experts
    if args.lr:
        config['training']['lr'] = args.lr
    if args.weight_decay:
        config['training']['weight_decay'] = args.weight_decay
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.device:
        config['experiment']['device'] = args.device
    if args.seed:
        config['experiment']['seed'] = args.seed
    if args.augmentation:
        config['augmentation']['enable'] = args.augmentation.lower() == 'true'
    if args.aug_strategy:
        config['augmentation']['strategy'] = args.aug_strategy
    if args.expert_specific_aug:
        config['augmentation']['expert_specific'] = args.expert_specific_aug.lower() == 'true'
    if args.gating:
        config['model']['gating'] = args.gating
    if args.debug:
        config['experiment']['debug']['enable'] = args.debug.lower() == 'true'
    if args.debug_samples:
        config['experiment']['debug']['num_samples'] = args.debug_samples
    if args.debug_epochs:
        config['experiment']['debug']['epochs'] = args.debug_epochs
    
    # Run training
    train(config) 