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

from data.dataset_loader import load_dataset
from models.model_factory import get_model
from utils.logger import Logger
from utils.metrics import compute_metrics
from models.augmentor import Augmentor, MetaLearner
from utils.losses import diversity_loss


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


def train_epoch_moe(model, loader, optimizer, dataset_info, device, epoch, config, meta_learner, augmentor):
    model.train()
    total_loss = 0
    all_targets = []
    all_aggregated_outputs = []
    verbose = model.verbose
    use_aug = config['augmentation']['enable']
    if use_aug:
        use_div = config['augmentation']['diversity']
        div_weight = config['augmentation']['diversity_weight']
    else:
        use_div = False
        div_weight = 0.0

    num_experts = model.num_experts

    # Track parameter outputs per expert
    param_accumulators = [[] for _ in range(num_experts)]
    # Track expert usage
    gate_weight_accumulator = []

    pbar = tqdm(loader, desc='Training MoE', leave=False)
    for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()

        if use_aug:
            # Step 1: Get K augmentation parameter sets from meta-learner
            param_sets = meta_learner(data)  # List[Tensor] of shape (4,) for each expert

            # Accumulate parameters
            for i in range(num_experts):
                param_accumulators[i].append(param_sets[i].cpu())
            
            # Step 2: Apply augmentation per expert
            augmented_data_list = [data]  # First entry is the original (unaugmented) graph
            for i in range(num_experts):
                augmented_view = augmentor.apply(data, param_sets[i])
                augmented_data_list.append(augmented_view)

            # Print augmentation parameters and their meanings
            if verbose:
                for i, params in enumerate(param_sets):
                    print(f"Expert {i} augmentation parameters:")
                    print(f"  p_node_drop: {params[0]:.4f}")
                    print(f"  p_edge_drop: {params[1]:.4f}")
                    print(f"  p_edge_add: {params[2]:.4f}")
                    print(f"  noise_std: {params[3]:.4f}")

        else:
            augmented_data_list = data
            if verbose:
                print("Augmentation disabled. Using identical input for all experts.")

        # Step 3: Forward through MoE
        expert_outputs, gate_weights = model(augmented_data_list)
        batch_size = data.y.size(0)

        # Step 4: Gate schedule
        if epoch < config['gate']['train_after']:
            gate_weights = torch.full_like(gate_weights, 1.0 / num_experts)
            for param in model.gate.parameters():
                param.requires_grad = False
            if verbose:
                print("Gating frozen — using uniform weights.")
        else:
            for param in model.gate.parameters():
                param.requires_grad = True
            if verbose:
                print("Gating unfreezed")
        gate_weight_accumulator.append(gate_weights.cpu())

        # Step 5: Compute gate-weighted classification loss
        losses = []
        for i in range(num_experts):
            ce = F.cross_entropy(expert_outputs[i], data.y, reduction='none')
            weighted_ce = ce * gate_weights[:, i]
            # if verbose:
            #     print(f"Expert {i} loss: {ce.item():.4f}, Gate weights avg: {gate_weights[:, i].mean().item():.4f}")
            losses.append(weighted_ce)

        total_ce_loss = torch.stack(losses, dim=1).sum(dim=1).mean() # add sparsity penalty here LATER

        # diversity loss
        if use_div:
            div_loss = diversity_loss(param_sets)
            if verbose:
                print(f"Diversity loss: {div_loss.item():.4f}")
            total_loss_value = total_ce_loss + div_weight * div_loss
        else:
            total_loss_value = total_ce_loss

        # load balance loss
        if config['gate']['load_balance_weight'] > 0:
            avg_usage = gate_weights.mean(dim=0)  # [num_experts]
            load_balance_loss = torch.var(avg_usage)  # scalar
            if verbose:
                print(f"Load balance loss: {load_balance_loss.item():.4f}")
            total_loss_value += config['gate']['load_balance_weight'] * load_balance_loss

        # Backward (STE): gradient flows to meta-learner through expert loss
        total_loss_value.backward()

        # Check gradient flow to meta-learner
        # if verbose and meta_learner is not None:
        #     for name, param in meta_learner.named_parameters():
        #         if param.grad is not None:
        #             print(f"Meta-learner parameter '{name}' gradient norm: {param.grad.norm().item():.4f}")
        #         else:
        #             print(f"Meta-learner parameter '{name}' has no gradient.")
        #     print("Meta-learner parameter norms (before step):")
        #     for name, param in meta_learner.named_parameters():
        #         print(f"{name}: {param.data.norm().item():.6f}")

        # Check gradient flow to gate
        # for name, param in model.gate.named_parameters():
        #     if param.grad is not None:
        #         print(f"Param {name} grad norm: {param.grad.norm().item():.4f}")
        #     else:
        #         print(f"Param {name} has no gradient!")

        optimizer.step()

        # if verbose:
        #     print("Meta-learner parameter norms (after step):")
        #     for name, param in meta_learner.named_parameters():
        #         print(f"{name}: {param.data.norm().item():.6f}")


        # Logging
        total_loss += total_loss_value.item() * batch_size
        all_targets.append(data.y.detach())
        aggregated = model.aggregate(expert_outputs, gate_weights)
        all_aggregated_outputs.append(aggregated.detach())

    avg_expert_params = {}
    if use_aug:
        for i in range(num_experts):
            avg_params = torch.stack(param_accumulators[i]).mean(dim=0)
            avg_expert_params[f'expert_{i}_p_node_drop'] = avg_params[0].item()
            avg_expert_params[f'expert_{i}_p_edge_drop'] = avg_params[1].item()
            avg_expert_params[f'expert_{i}_p_edge_add'] = avg_params[2].item()
            avg_expert_params[f'expert_{i}_noise_std'] = avg_params[3].item()
            print(f"Expert {i} avg params: p_node_drop={avg_params[0]:.4f}, "
                f"p_edge_drop={avg_params[1]:.4f}, p_edge_add={avg_params[2]:.4f}, noise_std={avg_params[3]:.4f}")

    gate_weights_all = torch.cat(gate_weight_accumulator, dim=0)  # (N, num_experts)
    load_balance = gate_weights_all.mean(dim=0)
    eff_expert_per_sample = 1.0 / (gate_weights_all ** 2).sum(dim=1)
    avg_effective_experts = eff_expert_per_sample.mean().item()
    print("\nGate Load per Expert:")
    for i, v in enumerate(load_balance):
        print(f"  Expert {i}: {v.item():.4f}")
    print(f"Average effective number of experts used per sample: {avg_effective_experts:.4f}")

    final_outputs = torch.cat(all_aggregated_outputs, dim=0)
    final_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(final_outputs, final_targets, dataset_info['metric'])
    metrics['loss'] = total_loss / len(loader.dataset)
    metrics.update(avg_expert_params)
    metrics['load_balance'] = load_balance.tolist()
    metrics['avg_effective_experts'] = avg_effective_experts
    
    return metrics


def evaluate_moe(model, loader, device, metric_type, epoch=1000):
    """Evaluate MoE model on validation or test set."""
    model.eval()
    total_loss = 0
    all_targets = []
    all_aggregated_outputs = []
    verbose = model.verbose

    if verbose:
        print("Evaluating on validation/test set now...")

    pbar = tqdm(loader, desc='Evaluating MoE', leave=False)
    with torch.no_grad():
        for data in pbar:
            data = data.to(device)

            # Forward pass
            assert not isinstance(data, list), "Data should not be augmented during evaluation."
            expert_outputs, gate_weights = model(data)

            # Apply uniform weights if in warm-up
            if epoch < config['gate']['train_after']:
                gate_weights = torch.full_like(gate_weights, 1.0 / model.num_experts)

            # Aggregate predictions using gate weights
            aggregated = model.aggregate(expert_outputs, gate_weights)
            all_aggregated_outputs.append(aggregated)
            all_targets.append(data.y)

            # Optionally compute CE loss per batch
            ce = F.cross_entropy(aggregated, data.y, reduction='mean')
            total_loss += ce.item() * data.num_graphs

    final_outputs = torch.cat(all_aggregated_outputs, dim=0)
    final_targets = torch.cat(all_targets, dim=0)

    metrics = compute_metrics(final_outputs, final_targets, metric_type)
    metrics['loss'] = total_loss / len(loader.dataset)

    return metrics


def train_epoch_uil(model, loader, optimizer, dataset_info, device, epoch, config):
    """Train UIL model for one epoch."""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_reg_loss = 0
    total_sem_loss = 0
    total_str_loss = 0
    all_outputs = []
    all_targets = []
    
    pbar = tqdm(loader, desc='Training UIL', leave=False)
    for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data, target=data.y)
        loss = output['loss_total']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * data.num_graphs
        all_outputs.append(output['logits'].detach())
        all_targets.append(data.y.detach())
        total_ce_loss += output['loss_ce'].item() * data.num_graphs
        total_reg_loss += output['loss_reg'].item() * data.num_graphs
        total_sem_loss += output['loss_sem'].item() * data.num_graphs
        total_str_loss += output['loss_str'].item() * data.num_graphs
    
    # Compute epoch metrics
    avg_loss = total_loss / len(loader.dataset)
    avg_ce_loss = total_ce_loss / len(loader.dataset)
    avg_reg_loss = total_reg_loss / len(loader.dataset)
    avg_sem_loss = total_sem_loss / len(loader.dataset)
    avg_str_loss = total_str_loss / len(loader.dataset)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_outputs, all_targets, dataset_info['metric'])
    metrics['loss'] = avg_loss
    metrics['loss_ce'] = avg_ce_loss
    metrics['loss_reg'] = avg_reg_loss
    metrics['loss_sem'] = avg_sem_loss
    metrics['loss_str'] = avg_str_loss
    return metrics


def evaluate_uil(model, loader, device, metric_type):
    """Evaluate UIL model on validation or test set."""
    model.eval()
    total_loss = 0
    total_ce_loss = 0
    total_reg_loss = 0
    total_sem_loss = 0
    total_str_loss = 0
    all_outputs = []
    all_targets = []
    orig_nodes_list = []
    orig_edges_list = []
    stable_nodes_list = []
    stable_edges_list = []
    verbose = model.verbose
    
    pbar = tqdm(loader, desc='Evaluating UIL', leave=False)
    with torch.no_grad():
        for data in pbar:
            data = data.to(device)
            
            # Forward pass
            output = model(data, target=data.y)
            loss = output['loss_total']
            
            # Track metrics
            total_loss += loss.item() * data.num_graphs
            total_ce_loss += output['loss_ce'].item() * data.num_graphs
            total_reg_loss += output['loss_reg'].item() * data.num_graphs
            total_sem_loss += output['loss_sem'].item() * data.num_graphs
            total_str_loss += output['loss_str'].item() * data.num_graphs
            all_outputs.append(output['logits'])
            all_targets.append(data.y)

            if verbose:
                node_mask = output['node_mask'].squeeze()  # (N,)
                edge_mask = output['edge_mask'].squeeze()  # (E,)
                edge_index = output['cached_masks']['edge_index']
                batch_vec = output['cached_masks']['batch']

                # Original node/edge counts per graph
                node_counts = torch.bincount(batch_vec)
                edge_graphs = batch_vec[edge_index[0]]
                edge_counts = torch.bincount(edge_graphs, minlength=node_counts.size(0))

                # Stable node/edge counts per graph
                stable_mask = (node_mask >= 0.5)
                stable_graphs = batch_vec[stable_mask]
                stable_node_counts = torch.bincount(stable_graphs, minlength=node_counts.size(0))

                src, dst = edge_index
                stable_edges_mask = (
                    (node_mask[src] >= 0.5) &
                    (node_mask[dst] >= 0.5) &
                    (edge_mask >= 0.5)
                )
                stable_edge_graphs = batch_vec[src][stable_edges_mask]
                stable_edge_counts = torch.bincount(stable_edge_graphs, minlength=node_counts.size(0))

                orig_nodes_list.extend(node_counts.tolist())
                orig_edges_list.extend(edge_counts.tolist())
                stable_nodes_list.extend(stable_node_counts.tolist())
                stable_edges_list.extend(stable_edge_counts.tolist())
    
    # Compute metrics
    avg_loss = total_loss / len(loader.dataset)
    avg_ce_loss = total_ce_loss / len(loader.dataset)
    avg_reg_loss = total_reg_loss / len(loader.dataset)
    avg_sem_loss = total_sem_loss / len(loader.dataset)
    avg_str_loss = total_str_loss / len(loader.dataset)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_outputs, all_targets, metric_type)
    metrics['loss'] = avg_loss
    metrics['loss_ce'] = avg_ce_loss
    metrics['loss_reg'] = avg_reg_loss
    metrics['loss_sem'] = avg_sem_loss
    metrics['loss_str'] = avg_str_loss
    if verbose:
        metrics['avg_nodes_orig'] = float(np.mean(orig_nodes_list)) if orig_nodes_list else 0.0
        metrics['avg_edges_orig'] = float(np.mean(orig_edges_list)) if orig_edges_list else 0.0
        metrics['avg_nodes_stable'] = float(np.mean(stable_nodes_list)) if stable_nodes_list else 0.0
        metrics['avg_edges_stable'] = float(np.mean(stable_edges_list)) if stable_edges_list else 0.0
    
    return metrics

def evaluate_uil_visually(model, loader, device, metric_type, visualize_every=10):
    """
    Evaluate UIL model and visually print original + stable subgraphs.

    Args:
        visualize_every: visualize 1 in every N graphs (to reduce spam).
    """
    model.eval()
    total_loss = 0
    total_ce_loss = 0
    total_reg_loss = 0
    total_sem_loss = 0
    total_str_loss = 0
    all_outputs = []
    all_targets = []

    orig_nodes_list = []
    orig_edges_list = []
    stable_nodes_list = []
    stable_edges_list = []

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            output = model(data, target=data.y)

            loss = output['loss_total']
            total_loss += loss.item() * data.num_graphs
            total_ce_loss += output['loss_ce'].item() * data.num_graphs
            total_reg_loss += output['loss_reg'].item() * data.num_graphs
            total_sem_loss += output['loss_sem'].item() * data.num_graphs
            total_str_loss += output['loss_str'].item() * data.num_graphs
            all_outputs.append(output['logits'])
            all_targets.append(data.y)

            node_mask = output['node_mask'].squeeze()  # (N,)
            edge_mask = output['edge_mask'].squeeze()  # (E,)
            edge_index = output['cached_masks']['edge_index']
            batch_vec = output['cached_masks']['batch']

            for graph_idx in range(data.num_graphs):
                if graph_idx % visualize_every != 0:
                    continue

                node_ids = (batch_vec == graph_idx).nonzero(as_tuple=True)[0]
                node_map = {i.item(): j for j, i in enumerate(node_ids)}

                # Convert to NetworkX (PyG handles batch slicing automatically)
                pyg_graph = data.__class__.from_data_list([data.get_example(graph_idx)])
                G = to_networkx(pyg_graph, to_undirected=True)

                # Record original graph stats
                orig_nodes_list.append(G.number_of_nodes())
                orig_edges_list.append(G.number_of_edges())

                # # Plot original
                # plt.figure(figsize=(10, 5))
                # plt.subplot(1, 2, 1)
                # nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray')
                # plt.title(f"Original Graph #{batch_idx * data.num_graphs + graph_idx}\n"
                #           f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

                # Extract stable subgraph
                stable_nodes = node_ids[node_mask[node_ids] >= 0.5].tolist()
                stable_edges = []
                for i, (src, dst) in enumerate(edge_index.t().tolist()):
                    if src in stable_nodes and dst in stable_nodes and edge_mask[i] >= 0.5:
                        stable_edges.append((src, dst))

                # Create stable subgraph
                SG = nx.Graph()
                SG.add_nodes_from(stable_nodes)
                SG.add_edges_from(stable_edges)

                # Record stable subgraph stats
                stable_nodes_list.append(SG.number_of_nodes())
                stable_edges_list.append(SG.number_of_edges())

                # # Plot stable subgraph
                # plt.subplot(1, 2, 2)
                # nx.draw(SG, with_labels=True, node_color='lightgreen', edge_color='black')
                # plt.title(f"Predicted Stable Subgraph\n"
                #           f"Nodes: {SG.number_of_nodes()}, Edges: {SG.number_of_edges()}")

                # plt.tight_layout()
                # plt.show()

    # Metrics
    avg_loss = total_loss / len(loader.dataset)
    avg_ce_loss = total_ce_loss / len(loader.dataset)
    avg_reg_loss = total_reg_loss / len(loader.dataset)
    avg_sem_loss = total_sem_loss / len(loader.dataset)
    avg_str_loss = total_str_loss / len(loader.dataset)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_outputs, all_targets, metric_type)
    metrics['loss'] = avg_loss
    metrics['loss_ce'] = avg_ce_loss
    metrics['loss_reg'] = avg_reg_loss
    metrics['loss_sem'] = avg_sem_loss
    metrics['loss_str'] = avg_str_loss

    # Add average node/edge stats
    metrics['avg_nodes_orig'] = float(np.mean(orig_nodes_list)) if orig_nodes_list else 0.0
    metrics['avg_edges_orig'] = float(np.mean(orig_edges_list)) if orig_edges_list else 0.0
    metrics['avg_nodes_stable'] = float(np.mean(stable_nodes_list)) if stable_nodes_list else 0.0
    metrics['avg_edges_stable'] = float(np.mean(stable_edges_list)) if stable_edges_list else 0.0

    print(f"Average nodes (original): {metrics['avg_nodes_orig']:.2f}")
    print(f"Average edges (original): {metrics['avg_edges_orig']:.2f}")
    print(f"Average nodes (stable): {metrics['avg_nodes_stable']:.2f}")
    print(f"Average edges (stable): {metrics['avg_edges_stable']:.2f}")

    return metrics


def train_epoch_moeuil(model, loader, optimizer, dataset_info, device, epoch, config):
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_reg_loss = 0
    total_sem_loss = 0
    total_str_loss = 0
    total_div_loss = 0
    total_load_loss = 0
    all_targets = []
    all_aggregated_outputs = []
    verbose = model.verbose

    # Track expert usage
    gate_weight_accumulator = []

    pbar = tqdm(loader, desc='Training MoEUIL', leave=False)
    for data in pbar:

        data = data.to(device)
        optimizer.zero_grad()

        # Step 3: Forward through MoE
        aggregated_outputs = model(data, epoch)
        batch_size = data.y.size(0)

        # Step 4: Gate schedule
        gate_weights = aggregated_outputs['gate_weights']  # (K, B, 1)
        gate_weights = gate_weights.squeeze(-1).T          # → (B, K)
        gate_weight_accumulator.append(gate_weights.cpu()) # accumulate across batches

        loss = aggregated_outputs['loss_total']

        loss.backward()

        # Check gradient flow to gate
        # for name, param in model.gate.named_parameters():
        #     if param.grad is not None:
        #         print(f"Param {name} grad norm: {param.grad.norm().item():.4f}")
        #     else:
        #         print(f"Param {name} has no gradient!")

        optimizer.step()

        # Logging
        total_loss += loss.item() * batch_size
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
    
    return metrics


def evaluate_moeuil(model, loader, device, metric_type, epoch):
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
    verbose = model.verbose

    # Track expert usage
    gate_weight_accumulator = []

    pbar = tqdm(loader, desc='Evaluating MoEUIL', leave=False)
    for data in pbar:
        data = data.to(device)

        # Step 3: Forward through MoE
        aggregated_outputs = model(data, epoch)
        batch_size = data.y.size(0)

        # Step 4: Gate schedule
        gate_weights = aggregated_outputs['gate_weights']
        gate_weights = gate_weights.squeeze(-1).T          # → (B, K)
        gate_weight_accumulator.append(gate_weights.cpu())
        
        # Logging
        total_loss += aggregated_outputs['loss_total'] * batch_size
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
   
    return metrics
