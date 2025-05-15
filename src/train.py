import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from data.dataset_loader import load_dataset
from models.model_factory import get_model
from utils.logger import Logger
from utils.metrics import compute_metrics

def train_epoch(model, loader, optimizer, device):
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
    metrics = compute_metrics(all_outputs, all_targets)
    metrics['loss'] = avg_loss
    
    return metrics

def evaluate(model, loader, device):
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
    metrics = compute_metrics(all_outputs, all_targets)
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
    val_loader = data_loaders['val_loader']
    test_loader = data_loaders['test_loader']
    dataset_info = data_loaders['dataset_info']
    
    logger.logger.info(f"Dataset info: {dataset_info}")
    
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
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        
        # Log metrics
        logger.log_metrics(train_metrics, epoch, phase="train")
        logger.log_metrics(val_metrics, epoch, phase="val")
        
        # Early stopping
        if val_metrics['accuracy'] > best_val_acc + config['training']['early_stopping']['min_delta']:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            logger.save_model(model, epoch, val_metrics)
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping']['patience']:
                logger.logger.info(f"Early stopping at epoch {epoch}")
                break
    
    # Final evaluation on test set
    logger.logger.info("Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, device)
    logger.log_metrics(test_metrics, epoch, phase="test")
    
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
            'val': val_metrics,
            'test': test_metrics
        }, f)
    
    logger.logger.info(f"Results saved to {results_path}")
    logger.close()
    
    return test_metrics

if __name__ == "__main__":
    # Load configuration
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Run training
    train(config) 