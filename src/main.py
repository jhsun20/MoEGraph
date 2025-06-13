import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import datetime

from data.dataset_loader import load_dataset
from models.model_factory import get_model
from utils.logger import Logger
from models.augmentor import Augmentor, MetaLearner
from utils.train import train_epoch, train_epoch_moe, evaluate, evaluate_moe, train_epoch_uil, evaluate_uil, train_epoch_moeuil, evaluate_moeuil
def train(config):
    """Main training function."""
    
    # Set device
    device = torch.device(config['experiment']['device'] if torch.cuda.is_available() else "cpu")
    verbose = config['experiment']['debug']['verbose']
    
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
 
    # Get today's date and current time
    now = datetime.datetime.now()
    today_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H-%M-%S")

    # Prepare results directory with today's date and current time
    results_dir = os.path.join(
        "results",
        f"{config['experiment']['name']}_{config['dataset']['dataset_name']}_{today_date}_{current_time}"
    )
    os.makedirs(results_dir, exist_ok=True)

    all_test_ood_metrics = []
    all_test_id_metrics = []
    all_train_metrics = []
    all_val_ood_metrics = []
    all_val_id_metrics = []
    
    # Iterate over each seed
    for seed in config['experiment']['seeds']:
        # Set new seed
        torch.cuda.empty_cache()
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        logger.logger.info(f"Running with seed: {seed}")

        # Initialize model
        logger.logger.info(f"Initializing {config['model']['type']} model...")
        model = get_model(config, dataset_info)
        model = torch.nn.DataParallel(model)
        model = model.to(device)
        
        if config['model']['type'] == 'uil' or config['model']['type'] == 'moe_uil':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config['training']['lr'],
                weight_decay=config['training']['weight_decay']
            )
        else:
            meta_learner = None
            augmentor = None
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
                train_metrics = train_epoch_moe(model, train_loader, optimizer, dataset_info, device, epoch, config, meta_learner, augmentor)
                # Validate on OOD validation set
                val_metrics = evaluate_moe(model, val_loader, device, metric_type, epoch)
                # Validate on in-distribution validation set
                id_val_metrics = evaluate_moe(model, id_val_loader, device, metric_type, epoch)
            elif config['model']['type'] == 'uil':
                train_metrics = train_epoch_uil(model, train_loader, optimizer, dataset_info, device, epoch, config)
                # Validate on OOD validation set
                val_metrics = evaluate_uil(model, val_loader, device, metric_type)
                # Validate on in-distribution validation set
                id_val_metrics = evaluate_uil(model, id_val_loader, device, metric_type)
            elif config['model']['type'] == 'moe_uil':
                train_metrics = train_epoch_moeuil(model, train_loader, optimizer, dataset_info, device, epoch, config)
                # Validate on OOD validation set
                val_metrics = evaluate_moeuil(model, val_loader, device, metric_type, epoch)
                # Validate on in-distribution validation set
                id_val_metrics = evaluate_moeuil(model, id_val_loader, device, metric_type, epoch)
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
            
            # Log individual losses if model is UIL
            if (config['model']['type'] == 'uil' or config['model']['type'] == 'moe_uil') and verbose:
                logger.log_metrics({
                    'loss_ce': round(train_metrics.get('loss_ce', 0), 2),
                    'loss_reg': round(train_metrics.get('loss_reg', 0), 2),
                    'loss_sem': round(train_metrics.get('loss_sem', 0), 2),
                    'loss_str': round(train_metrics.get('loss_str', 0), 2),
                    'loss_div': round(train_metrics.get('loss_div', 0), 2),
                    'loss_load': round(train_metrics.get('loss_load', 0), 2),
                }, epoch, phase="train")
                
                # Log individual losses for validation
                logger.log_metrics({
                    'loss_ce': round(val_metrics.get('loss_ce', 0), 2),
                    'loss_reg': round(val_metrics.get('loss_reg', 0), 2),
                    'loss_sem': round(val_metrics.get('loss_sem', 0), 2),
                    'loss_str': round(val_metrics.get('loss_str', 0), 2),
                    'loss_div': round(val_metrics.get('loss_div', 0), 2),
                    'loss_load': round(val_metrics.get('loss_load', 0), 2),
                    'avg_nodes_orig': round(val_metrics.get('avg_nodes_orig', 0), 2),
                    'avg_edges_orig': round(val_metrics.get('avg_edges_orig', 0), 2),
                    'avg_nodes_stable': round(val_metrics.get('avg_nodes_stable', 0), 2),
                    'avg_edges_stable': round(val_metrics.get('avg_edges_stable', 0), 2),
                }, epoch, phase="val_ood")

                # Log individual losses for in-distribution validation
                logger.log_metrics({
                    'loss_ce': round(id_val_metrics.get('loss_ce', 0), 2),
                    'loss_reg': round(id_val_metrics.get('loss_reg', 0), 2),
                    'loss_sem': round(id_val_metrics.get('loss_sem', 0), 2),
                    'loss_str': round(id_val_metrics.get('loss_str', 0), 2),
                    'loss_div': round(id_val_metrics.get('loss_div', 0), 2),
                    'loss_load': round(id_val_metrics.get('loss_load', 0), 2),
                    'avg_nodes_orig': round(id_val_metrics.get('avg_nodes_orig', 0), 2),
                    'avg_edges_orig': round(id_val_metrics.get('avg_edges_orig', 0), 2),
                    'avg_nodes_stable': round(id_val_metrics.get('avg_nodes_stable', 0), 2),
                    'avg_edges_stable': round(id_val_metrics.get('avg_edges_stable', 0), 2),
                }, epoch, phase="val_id")

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
        if config['model']['type'] == 'moe':
            test_ood_metrics = evaluate_moe(model, test_loader, device, metric_type)
            test_id_metrics = evaluate_moe(model, id_test_loader, device, metric_type)
        elif config['model']['type'] == 'uil':
            test_ood_metrics = evaluate_uil(model, test_loader, device, metric_type)
            test_id_metrics = evaluate_uil(model, id_test_loader, device, metric_type)
        elif config['model']['type'] == 'moe_uil':
            test_ood_metrics = evaluate_moeuil(model, test_loader, device, metric_type)
            test_id_metrics = evaluate_moeuil(model, id_test_loader, device, metric_type)
        else:
            test_ood_metrics = evaluate(model, test_loader, device, metric_type)
            test_id_metrics = evaluate(model, id_test_loader, device, metric_type)
        
        logger.log_metrics(test_ood_metrics, epoch, phase="test_ood")
        logger.log_metrics(test_id_metrics, epoch, phase="test_id")

        if (config['model']['type'] == 'uil' or config['model']['type'] == 'moe_uil') and verbose:
            logger.log_metrics({
                    'loss_ce': round(test_ood_metrics.get('loss_ce', 0), 2),
                    'loss_reg': round(test_ood_metrics.get('loss_reg', 0), 2),
                    'loss_sem': round(test_ood_metrics.get('loss_sem', 0), 2),
                    'loss_str': round(test_ood_metrics.get('loss_str', 0), 2),
                    'loss_div': round(test_ood_metrics.get('loss_div', 0), 2),
                    'loss_load': round(test_ood_metrics.get('loss_load', 0), 2),
                    'avg_nodes_orig': round(test_ood_metrics.get('avg_nodes_orig', 0), 2),
                    'avg_edges_orig': round(test_ood_metrics.get('avg_edges_orig', 0), 2),
                    'avg_nodes_stable': round(test_ood_metrics.get('avg_nodes_stable', 0), 2),
                    'avg_edges_stable': round(test_ood_metrics.get('avg_edges_stable', 0), 2),
                }, epoch, phase="test_ood")
        
            logger.log_metrics({
                    'loss_ce': round(test_id_metrics.get('loss_ce', 0), 2),
                    'loss_reg': round(test_id_metrics.get('loss_reg', 0), 2),
                    'loss_sem': round(test_id_metrics.get('loss_sem', 0), 2),
                    'loss_str': round(test_id_metrics.get('loss_str', 0), 2),
                    'loss_div': round(test_id_metrics.get('loss_div', 0), 2),
                    'loss_load': round(test_id_metrics.get('loss_load', 0), 2),
                    'avg_nodes_orig': round(test_id_metrics.get('avg_nodes_orig', 0), 2),
                    'avg_edges_orig': round(test_id_metrics.get('avg_edges_orig', 0), 2),
                    'avg_nodes_stable': round(test_id_metrics.get('avg_nodes_stable', 0), 2),
                    'avg_edges_stable': round(test_id_metrics.get('avg_edges_stable', 0), 2),
                }, epoch, phase="test_id")
            
        all_test_ood_metrics.append(test_ood_metrics)
        all_test_id_metrics.append(test_id_metrics)
        all_train_metrics.append(train_metrics)
        all_val_ood_metrics.append(val_metrics)
        all_val_id_metrics.append(id_val_metrics)
        # Save the final model checkpoint
        final_checkpoint_path = os.path.join(results_dir, "final_model_checkpoint_{seed}.pth")
        torch.save(model.state_dict(), final_checkpoint_path)
        logger.logger.info(f"Final model checkpoint saved to {final_checkpoint_path}")

    # Save results after all seeds have been processed
    if verbose:
        print(f"All test OOD metrics: {all_test_ood_metrics}")
        print(f"All test ID metrics: {all_test_id_metrics}")
    # Calculate average accuracies for test OOD and test ID metrics
    avg_test_ood_accuracy = sum(metrics['accuracy'] for metrics in all_test_ood_metrics) / len(all_test_ood_metrics)
    avg_test_id_accuracy = sum(metrics['accuracy'] for metrics in all_test_id_metrics) / len(all_test_id_metrics)
    results_path = os.path.join(results_dir, "metrics.yaml")
    with open(results_path, 'w') as f:
        yaml.dump({
            'train': all_train_metrics,
            'val_ood': all_val_ood_metrics,
            'val_id': all_val_id_metrics,
            'test_ood': all_test_ood_metrics,
            'test_id': all_test_id_metrics,
            'avg_test_ood_accuracy': avg_test_ood_accuracy,
            'avg_test_id_accuracy': avg_test_id_accuracy
        }, f)
    
    logger.logger.info(f"Results saved to {results_path}")
    
    # Save the configuration file used for the experiment
    config_path = os.path.join(results_dir, "config_used.yaml")
    with open(config_path, 'w') as config_file:
        yaml.dump(config, config_file)
    logger.logger.info(f"Configuration saved to {config_path}")

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