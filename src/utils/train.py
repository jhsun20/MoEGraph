import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import datetime
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch.cuda.amp import autocast, GradScaler
import gc
import time

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
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for data in pbar:
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
    
    pbar = tqdm(loader, desc='Evaluating', leave=False)
    with torch.no_grad():
        for data in pbar:
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

def train_epoch_moe(model, loader, optimizer, dataset_info, device, epoch, config):
    model.train()
    scaler = GradScaler()

    total_loss = 0
    total_ce_loss = 0
    total_reg_loss = 0
    total_sem_loss = 0
    total_str_loss = 0
    total_div_loss = 0
    total_load_loss = 0
    all_targets = []
    all_aggregated_outputs = []
    rho_node_sum = 0
    rho_edge_sum = 0
    rho_feat_sum = 0
    if config['model']['parallel']:
        verbose = model.module.verbose
        model.module.set_epoch(epoch)
    else:
        verbose = model.verbose
        model.set_epoch(epoch)

    # Track expert usage
    gate_weight_accumulator = []

    pbar = tqdm(loader, desc='Training MoEUIL', leave=False)
    for data in pbar:
        if config['model']['parallel']:
            data = data.to_data_list()
            data = [d.to(torch.device('cuda')) for d in data]  # or device_ids[0]
            print("batch is a list")
        else:
            data = data.to(device)

        optimizer.zero_grad()

        # Step 3: Forward through MoE
        with autocast():
            aggregated_outputs = model(data)
            loss = aggregated_outputs['loss_total']

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = data.y.size(0)

        # Step 4: Gate schedule
        gate_weights = aggregated_outputs['gate_weights']  # (K, B, 1)
        gate_weights = gate_weights.squeeze(-1).T          # → (B, K)
        gate_weight_accumulator.append(gate_weights.cpu()) # accumulate across batches

        # Check gradient flow to gate
        # for name, param in model.gate.named_parameters():
        #     if param.grad is not None:
        #         print(f"Param {name} grad norm: {param.grad.norm().item():.4f}")
        #     else:
        #         print(f"Param {name} has no gradient!")

        # Logging
        total_loss += loss.item() * batch_size
        total_ce_loss += aggregated_outputs['loss_ce'].item() * batch_size
        total_reg_loss += aggregated_outputs['loss_reg'].item() * batch_size
        total_sem_loss += aggregated_outputs['loss_sem'].item() * batch_size
        total_str_loss += aggregated_outputs['loss_str'].item() * batch_size
        total_div_loss += aggregated_outputs['loss_div'].item() * batch_size
        total_load_loss += aggregated_outputs['loss_load'].item() * batch_size
        #print(aggregated_outputs['rho'])
        rho_node_sum += aggregated_outputs['rho'][0].item()
        rho_edge_sum += aggregated_outputs['rho'][1].item()
        rho_feat_sum += aggregated_outputs['rho'][2].item()
        all_targets.append(data.y.detach())
        all_aggregated_outputs.append(aggregated_outputs['logits'].detach())

    gate_weights_all = torch.cat(gate_weight_accumulator, dim=0)  # (N, num_experts)
    load_balance = gate_weights_all.mean(dim=0)
    if not config.get("experiment", {}).get("hyper_search", {}).get("enable", False):
        print("\nGate Load per Expert:")
        for i, v in enumerate(load_balance):
            print(f"  Expert {i}: {v.item():.4f}")

    final_outputs = torch.cat(all_aggregated_outputs, dim=0)
    final_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(final_outputs, final_targets, dataset_info['metric'])
    metrics['loss'] = total_loss / len(loader.dataset)
    metrics['loss_ce'] = total_ce_loss / len(loader.dataset)
    metrics['loss_reg'] = total_reg_loss / len(loader.dataset)
    metrics['loss_sem'] = total_sem_loss / len(loader.dataset)
    metrics['loss_str'] = total_str_loss / len(loader.dataset)
    metrics['loss_div'] = total_div_loss / len(loader.dataset)
    metrics['loss_load'] = total_load_loss / len(loader.dataset)
    metrics['load_balance'] = load_balance.tolist()
    rho_node_average = rho_node_sum / len(loader)
    rho_edge_average = rho_edge_sum / len(loader)
    rho_feat_average = rho_feat_sum / len(loader)
    metrics['rho_node_average'] = rho_node_average
    metrics['rho_edge_average'] = rho_edge_average
    metrics['rho_feat_average'] = rho_feat_average
    print(f"Rho average (node, edge, feat): {rho_node_average}, {rho_edge_average}, {rho_feat_average}")

    gc.collect()
    torch.cuda.empty_cache()

    return metrics


def evaluate_moe(model, loader, device, metric_type, epoch, config):
    """Evaluate MoE model on validation or test set."""
    model.eval()
    total_loss = 0
    total_ce_loss = 0
    total_reg_loss = 0
    total_sem_loss = 0
    total_str_loss = 0
    total_div_loss = 0
    total_load_loss = 0
    all_targets = []
    all_aggregated_outputs = []
    # Track expert usage
    gate_weight_accumulator = []
    if config['model']['parallel']:
        verbose = model.module.verbose
        model.module.set_epoch(epoch)
    else:
        verbose = model.verbose
        model.set_epoch(epoch)

    pbar = tqdm(loader, desc='Evaluating MoEUIL', leave=False)
    for data in pbar:
        data = data.to(device)

        # Step 3: Forward through MoE
        with autocast():
            aggregated_outputs = model(data)
        batch_size = data.y.size(0)

        # Step 4: Gate schedule
        gate_weights = aggregated_outputs['gate_weights']
        gate_weights = gate_weights.squeeze(-1).T          # → (B, K)
        gate_weight_accumulator.append(gate_weights.cpu())
        
        # Logging
        total_loss += aggregated_outputs['loss_total'].item() * batch_size
        total_ce_loss += aggregated_outputs['loss_ce'].item() * batch_size
        total_reg_loss += aggregated_outputs['loss_reg'].item() * batch_size
        total_sem_loss += aggregated_outputs['loss_sem'].item() * batch_size
        total_str_loss += aggregated_outputs['loss_str'].item() * batch_size
        total_div_loss += aggregated_outputs['loss_div'].item() * batch_size
        total_load_loss += aggregated_outputs['loss_load'].item() * batch_size
        all_targets.append(data.y.detach())
        all_aggregated_outputs.append(aggregated_outputs['logits'].detach())

    gate_weights_all = torch.cat(gate_weight_accumulator, dim=0)  # (N, num_experts)
    load_balance = gate_weights_all.mean(dim=0)
    
    # Always print gate load information
    if not config.get("experiment", {}).get("hyper_search", {}).get("enable", False):
        print("\nGate Load per Expert:")
        for i, v in enumerate(load_balance):
            print(f"  Expert {i}: {v.item():.4f}")

    final_outputs = torch.cat(all_aggregated_outputs, dim=0)
    final_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(final_outputs, final_targets, metric_type)
    
    # Always include all losses in metrics
    metrics['loss'] = total_loss / len(loader.dataset)
    metrics['loss_ce'] = total_ce_loss / len(loader.dataset)
    metrics['loss_reg'] = total_reg_loss / len(loader.dataset)
    metrics['loss_sem'] = total_sem_loss / len(loader.dataset)
    metrics['loss_str'] = total_str_loss / len(loader.dataset)
    metrics['loss_div'] = total_div_loss / len(loader.dataset)
    metrics['loss_load'] = total_load_loss / len(loader.dataset)
    metrics['load_balance'] = load_balance.tolist()
    gc.collect()
    torch.cuda.empty_cache()

    return metrics


def train(config, trial=None):
    """Main training function."""
    
    # Set device
    device = torch.device(config['experiment']['device'] if torch.cuda.is_available() else "cpu")
    verbose = config['experiment']['debug']['verbose']
    
    # Initialize logger
    logger = Logger(config)
    is_tuning = config.get("experiment", {}).get("hyper_search", {}).get("enable", False)
    if not is_tuning:
        logger.logger.info(f"Using device: {device}")
    else:
        start_time = time.time()
    
    # Load dataset
    if not is_tuning:
        logger.logger.info("Loading dataset...")
    data_loaders = load_dataset(config)
    train_loader = data_loaders['train_loader']
    val_loader = data_loaders['val_loader']  # OOD validation
    test_loader = data_loaders['test_loader']  # OOD test
    id_val_loader = data_loaders['id_val_loader']  # In-distribution validation
    id_test_loader = data_loaders['id_test_loader']  # In-distribution test
    dataset_info = data_loaders['dataset_info']
    
    metric_type = dataset_info['metric']
    logger.set_metric_type(metric_type)  # Set the metric type in logger
 
    # Get today's date and current time
    now = datetime.datetime.now()
    today_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H-%M-%S")

    # Prepare results directory with today's date and current time
    if not is_tuning:
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
        
        if not is_tuning:
            logger.logger.info(f"Running with seed: {seed}")
        logger.reset()  # Reset best checkpoint tracking for new seed

        # Initialize model
        if not is_tuning:
            logger.logger.info(f"Initializing {config['model']['type']} model...")
        model = get_model(config, dataset_info)
        if config['model']['parallel']:
            model = DataParallel(model)
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
        if not is_tuning:
            logger.logger.info("Starting training...")
        best_val_acc = 0
        patience_counter = 0
        best_epoch = 0  # Track the best epoch
        
        # Adjust epochs if in debug mode
        num_epochs = config['experiment']['debug']['epochs'] if config['experiment']['debug']['enable'] else config['training']['epochs']
        
        for epoch in range(1, num_epochs + 1):
            # Train
            if config['model']['type'] == 'moe':
                train_metrics = train_epoch_moe(model, train_loader, optimizer, dataset_info, device, epoch, config)
                # Validate on OOD validation set
                val_metrics = evaluate_moe(model, val_loader, device, metric_type, epoch, config)
                # Validate on in-distribution validation set
                if not is_tuning:
                    id_val_metrics = evaluate_moe(model, id_val_loader, device, metric_type, epoch, config)
            else:
                train_metrics = train_epoch(model, train_loader, optimizer, dataset_info, device)
                # Validate on OOD validation set
                val_metrics = evaluate(model, val_loader, device, metric_type)
                # Validate on in-distribution validation set
                if not is_tuning:
                    id_val_metrics = evaluate(model, id_val_loader, device, metric_type)
            
            # Log metrics
            if not is_tuning:
                logger.log_metrics(train_metrics, epoch, phase="train")
                logger.log_metrics(val_metrics, epoch, phase="val_ood")
                logger.log_metrics(id_val_metrics, epoch, phase="val_id")
            
            # Log individual losses if model is UIL
            # if (config['model']['type'] == 'uil') and not is_tuning:
            #     logger.log_metrics({
            #         'loss_ce': round(train_metrics.get('loss_ce', 0), 2),
            #         'loss_reg': round(train_metrics.get('loss_reg', 0), 2),
            #         'loss_sem': round(train_metrics.get('loss_sem', 0), 2),
            #         'loss_str': round(train_metrics.get('loss_str', 0), 2),
            #         'loss_div': round(train_metrics.get('loss_div', 0), 2),
            #         'loss_load': round(train_metrics.get('loss_load', 0), 2),
            #     }, epoch, phase="train")
                
            #     # Log individual losses for validation
            #     logger.log_metrics({
            #         'loss_ce': round(val_metrics.get('loss_ce', 0), 2),
            #         'loss_reg': round(val_metrics.get('loss_reg', 0), 2),
            #         'loss_sem': round(val_metrics.get('loss_sem', 0), 2),
            #         'loss_str': round(val_metrics.get('loss_str', 0), 2),
            #         'loss_div': round(val_metrics.get('loss_div', 0), 2),
            #         'loss_load': round(val_metrics.get('loss_load', 0), 2),
            #         'avg_nodes_orig': round(val_metrics.get('avg_nodes_orig', 0), 2),
            #         'avg_edges_orig': round(val_metrics.get('avg_edges_orig', 0), 2),
            #         'avg_nodes_stable': round(val_metrics.get('avg_nodes_stable', 0), 2),
            #         'avg_edges_stable': round(val_metrics.get('avg_edges_stable', 0), 2),
            #     }, epoch, phase="val_ood")

            #     # Log individual losses for in-distribution validation
            #     logger.log_metrics({
            #         'loss_ce': round(id_val_metrics.get('loss_ce', 0), 2),
            #         'loss_reg': round(id_val_metrics.get('loss_reg', 0), 2),
            #         'loss_sem': round(id_val_metrics.get('loss_sem', 0), 2),
            #         'loss_str': round(id_val_metrics.get('loss_str', 0), 2),
            #         'loss_div': round(id_val_metrics.get('loss_div', 0), 2),
            #         'loss_load': round(id_val_metrics.get('loss_load', 0), 2),
            #         'avg_nodes_orig': round(id_val_metrics.get('avg_nodes_orig', 0), 2),
            #         'avg_edges_orig': round(id_val_metrics.get('avg_edges_orig', 0), 2),
            #         'avg_nodes_stable': round(id_val_metrics.get('avg_nodes_stable', 0), 2),
            #         'avg_edges_stable': round(id_val_metrics.get('avg_edges_stable', 0), 2),
            #     }, epoch, phase="val_id")

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
                best_epoch = epoch  # Update best epoch
                logger.save_model(model, epoch, val_metrics)
                if not is_tuning:
                    logger.logger.info(f"New best model saved with {primary_metric}: {best_val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= config['training']['early_stopping']['patience']:
                    if not is_tuning:
                        logger.logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        if is_tuning:
            del optimizer
            if 'scaler' in locals():
                del scaler  # if using AMP
            gc.collect()
            torch.cuda.empty_cache()
        
        else:
            # Final evaluation on test sets
            logger.logger.info("Evaluating on test sets...")
            print("Evaluating on test sets...")
            del optimizer
            if 'scaler' in locals():
                del scaler  # if using AMP
            gc.collect()
            torch.cuda.empty_cache()

            # Load the best model checkpoint before final evaluation
            logger.logger.info("Loading best model checkpoint for final evaluation...")
            logger.load_best_model(model)
            
            if config['model']['type'] == 'moe':
                test_ood_metrics = evaluate_moe(model, test_loader, device, metric_type, epoch, config)
                test_id_metrics = evaluate_moe(model, id_test_loader, device, metric_type, epoch, config)
            else:
                test_ood_metrics = evaluate(model, test_loader, device, metric_type)
                test_id_metrics = evaluate(model, id_test_loader, device, metric_type)
            
            # Log test metrics with the best epoch
            logger.log_metrics(test_id_metrics, best_epoch, phase="test_id")
            logger.log_metrics(test_ood_metrics, best_epoch, phase="test_ood")

            # if (config['model']['type'] == 'uil' or config['model']['type'] == 'moe_uil'):
            #     logger.log_metrics({
            #             'loss_ce': round(test_ood_metrics.get('loss_ce', 0), 2),
            #             'loss_reg': round(test_ood_metrics.get('loss_reg', 0), 2),
            #             'loss_sem': round(test_ood_metrics.get('loss_sem', 0), 2),
            #             'loss_str': round(test_ood_metrics.get('loss_str', 0), 2),
            #             'loss_div': round(test_ood_metrics.get('loss_div', 0), 2),
            #             'loss_load': round(test_ood_metrics.get('loss_load', 0), 2),
            #             'avg_nodes_orig': round(test_ood_metrics.get('avg_nodes_orig', 0), 2),
            #             'avg_edges_orig': round(test_ood_metrics.get('avg_edges_orig', 0), 2),
            #             'avg_nodes_stable': round(test_ood_metrics.get('avg_nodes_stable', 0), 2),
            #             'avg_edges_stable': round(test_ood_metrics.get('avg_edges_stable', 0), 2),
            #         }, best_epoch, phase="test_ood")
            
            #     logger.log_metrics({
            #             'loss_ce': round(test_id_metrics.get('loss_ce', 0), 2),
            #             'loss_reg': round(test_id_metrics.get('loss_reg', 0), 2),
            #             'loss_sem': round(test_id_metrics.get('loss_sem', 0), 2),
            #             'loss_str': round(test_id_metrics.get('loss_str', 0), 2),
            #             'loss_div': round(test_id_metrics.get('loss_div', 0), 2),
            #             'loss_load': round(test_id_metrics.get('loss_load', 0), 2),
            #             'avg_nodes_orig': round(test_id_metrics.get('avg_nodes_orig', 0), 2),
            #             'avg_edges_orig': round(test_id_metrics.get('avg_edges_orig', 0), 2),
            #             'avg_nodes_stable': round(test_id_metrics.get('avg_nodes_stable', 0), 2),
            #             'avg_edges_stable': round(test_id_metrics.get('avg_edges_stable', 0), 2),
            #         }, best_epoch, phase="test_id")
                
            all_test_ood_metrics.append(test_ood_metrics)
            all_test_id_metrics.append(test_id_metrics)
            all_train_metrics.append(train_metrics)
            all_val_ood_metrics.append(val_metrics)
            all_val_id_metrics.append(id_val_metrics)
            # Save the final model checkpoint
            if config['logging']['save_model']:
                final_checkpoint_path = os.path.join(results_dir, f"final_model_checkpoint_{seed}.pth")
                if config['model']['parallel']:
                    torch.save(model.module.state_dict(), final_checkpoint_path)
                else:
                    torch.save(model.state_dict(), final_checkpoint_path)
                logger.logger.info(f"Final model checkpoint saved to {final_checkpoint_path}")

    if is_tuning:
        logger.close()
        elapsed_time = time.time() - start_time
        print(f"Trial completed in {elapsed_time:.2f} seconds.")
        return best_val_acc

    # Calculate average accuracies for test OOD and test ID metrics
    avg_test_ood_primary_metric = sum(metrics[primary_metric] for metrics in all_test_ood_metrics) / len(all_test_ood_metrics)
    avg_test_id_primary_metric = sum(metrics[primary_metric] for metrics in all_test_id_metrics) / len(all_test_id_metrics)
    results_path = os.path.join(results_dir, "metrics.yaml")
    with open(results_path, 'w') as f:
        yaml.dump({
            'train': all_train_metrics,
            'val_ood': all_val_ood_metrics,
            'val_id': all_val_id_metrics,
            'test_ood': all_test_ood_metrics,
            'test_id': all_test_id_metrics,
            'avg_test_ood_primary_metric': avg_test_ood_primary_metric,
            'avg_test_id_primary_metric': avg_test_id_primary_metric
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