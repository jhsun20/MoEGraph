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

import warnings

# Ignore all FutureWarnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="An issue occurred while importing 'pyg-lib'")
warnings.filterwarnings("ignore", message="An issue occurred while importing 'torch-sparse'")
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")


def _named_children_safe(m):
    try:
        return dict(m.named_children())
    except Exception:
        return {}

def get_gate_modules(model):
    """
    Returns (gate_modules_dict, gate_param_list).
    Covers: gate_enc, gate_mlp, and the tiny _gate_mlp used in _gate_logits_expert_features.
    Works with DataParallel and plain modules.
    """
    moem = model.module if hasattr(model, "module") else model
    mods = _named_children_safe(moem)
    gate_mods = {}

    # Canonical gate parts
    if hasattr(moem, "gate_enc"):  gate_mods["gate_enc"]  = moem.gate_enc
    if hasattr(moem, "gate_mlp"):  gate_mods["gate_mlp"]  = moem.gate_mlp

    # The small per-expert-feature MLP is created lazily; include it if it exists
    if hasattr(moem, "_gate_mlp") and (moem._gate_mlp is not None):
        gate_mods["_gate_mlp"] = moem._gate_mlp

    # Collect parameters
    gate_params = []
    for m in gate_mods.values():
        gate_params += list(m.parameters())
    return gate_mods, gate_params

def freeze_all_but_gate(model):
    """
    Sets requires_grad=False for every parameter except those in the gate (enc+MLPs).
    Keeps BatchNorms etc. in eval() for frozen parts; gate stays train().
    """
    moem = model.module if hasattr(model, "module") else model
    # 1) Find gate params
    _, gate_params = get_gate_modules(model)
    gate_param_ids = {id(p) for p in gate_params}

    # 2) Freeze everything not in gate
    for p in moem.parameters():
        p.requires_grad = (id(p) in gate_param_ids)

    # 3) Put frozen submodules to eval() for stability; gate to train()
    #    (This avoids BN running-stat updates in frozen experts.)
    if hasattr(moem, "shared"):
        moem.shared.eval()
    if hasattr(moem, "gate_enc"):
        moem.gate_enc.eval()
    if hasattr(moem, "gate_mlp"):
        moem.gate_mlp.eval()
    if hasattr(moem, "_gate_mlp") and moem._gate_mlp is not None:
        moem._gate_mlp.train()


@torch.no_grad()
def _bump_epoch_counter(model, steps: int = 1):
    moem = model.module if hasattr(model, "module") else model
    moem.set_epoch(moem.current_epoch + int(steps))

def finetune_gate_only(model, train_loader, val_loader, id_val_loader, dataset_info, device, config, logger, best):
    """
    After loading the best checkpoint: freeze experts, optimize only the gate for a few epochs.
    Uses your existing train_epoch_moe/evaluate_moe so gradients flow to the gate via:
      - gate KL/load-balance loss, and
      - task losses through gate-weighted aggregation (experts are frozen).
    """
    gate_epochs = int(config.get('gate', {}).get('finetune_epochs', 5))
    if gate_epochs <= 0:
        return  # nothing to do

    # Build gate-only optimizer
    _, gate_params = get_gate_modules(model)
    ft_lr = float(config.get('gate', {}).get('finetune_lr', config['training']['lr']))
    ft_wd = float(config.get('gate', {}).get('finetune_weight_decay', config['training']['weight_decay']))
    gate_opt = torch.optim.Adam(gate_params, lr=ft_lr, weight_decay=ft_wd)

    # Freeze everything else
    freeze_all_but_gate(model)

    metric_type = dataset_info['metric']
    primary_metric = 'accuracy' if metric_type == 'Accuracy' else metric_type.lower().replace('-', '_')
    best_val_metric = best
    for e in range(1, gate_epochs + 1):
        # keep epoch counter advancing so routing uses learned gate (post-warmup)
        _bump_epoch_counter(model, steps=1)

        # Train one epoch (only gate params require grad)
        train_metrics = train_epoch_moe(model, train_loader, gate_opt, dataset_info, device, epoch=100, config=config)

        # (Optional) quick val to watch overfitting
        val_metrics = evaluate_moe(model, val_loader, device, metric_type, epoch=100, config=config)
        val_id_metrics = evaluate_moe(model, id_val_loader, device, metric_type, epoch=100, config=config)

        current_metric = val_metrics[primary_metric]
        is_better = (current_metric < best_val_metric - config['training']['early_stopping']['min_delta']) if primary_metric in ['RMSE', 'MAE', 'loss'] else (current_metric > best_val_metric + config['training']['early_stopping']['min_delta'])
        if is_better:
            best_val_metric = current_metric
            patience_counter = 0
            best_epoch = config['training']['epochs'] + e  # Update best epoch
            logger.save_model(model, best_epoch, val_metrics)
            logger.logger.info(f"New best model saved with {primary_metric}: {best_val_metric:.4f}")
    # After finetune, keep experts frozen or unfreeze as you wish (we keep frozen).


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

    total_loss = total_ce_loss = total_reg_loss = 0
    total_la_loss = total_ea_loss = total_str_loss = total_div_loss = total_gate_loss = 0
    all_targets = []
    all_aggregated_outputs = []

    if config['model']['parallel']:
        verbose = model.module.verbose
        model.module.set_epoch(epoch)
    else:
        verbose = model.verbose
        model.set_epoch(epoch)

    gate_weight_accumulator = []
    
    # ------ epoch-level accumulators for mask keep/drop ------
    acc_node_keep_sum = None; acc_node_cnt = 0
    acc_edge_keep_sum = None; acc_edge_cnt = 0
    acc_node_keep_pg_sum = None; acc_node_pg_count = 0
    acc_edge_keep_pg_sum = None; acc_edge_pg_count = 0

    pbar = tqdm(loader, desc='Training MoEUIL', leave=False)
    for data in pbar:
        if config['model']['parallel']:
            data = data.to_data_list()
            data = [d.to(torch.device('cuda')) for d in data]
            print("batch is a list")
        else:
            data = data.to(device)

        optimizer.zero_grad()

        with autocast():
            aggregated_outputs = model(data)
            loss = aggregated_outputs['loss_total']

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = data.y.size(0)

        # --- FIX: gate_weights is already (B, K); do NOT squeeze/transpose ---
        gate_weights = aggregated_outputs['gate_weights']  # (B, K)
        gate_weight_accumulator.append(gate_weights.cpu())

        total_loss += loss.item() * batch_size
        total_ce_loss += aggregated_outputs['loss_ce'].item() * batch_size
        total_reg_loss += aggregated_outputs['loss_reg'].item() * batch_size
        total_la_loss += aggregated_outputs['loss_la'].item() * batch_size    
        total_ea_loss += aggregated_outputs['loss_ea'].item() * batch_size    
        total_str_loss += aggregated_outputs['loss_str'].item() * batch_size
        total_div_loss += aggregated_outputs['loss_div'].item() * batch_size
        total_gate_loss += aggregated_outputs['loss_gate'].item() * batch_size
        all_targets.append(data.y.detach())
        all_aggregated_outputs.append(aggregated_outputs['logits'].detach())

        # ------- accumulate mask stats from THIS batch -------
        nm = aggregated_outputs.get('node_masks', None)    # (N, K, 1)
        em = aggregated_outputs.get('edge_masks', None)    # (E, K, 1)
        if (nm is not None) and (em is not None):
            node_batch = data.batch
            edge_batch = data.batch[data.edge_index[0]]   # map edges to graphs via source node
            # micro (size-weighted across items)
            Xn = nm.detach().float().squeeze(-1)   # (N, K)
            Xe = em.detach().float().squeeze(-1)   # (E, K)
            kn = Xn.sum(dim=0).cpu()               # (K,)
            ke = Xe.sum(dim=0).cpu()               # (K,)
            acc_node_keep_sum = kn if acc_node_keep_sum is None else acc_node_keep_sum + kn
            acc_edge_keep_sum = ke if acc_edge_keep_sum is None else acc_edge_keep_sum + ke
            acc_node_cnt += Xn.size(0)
            acc_edge_cnt += Xe.size(0)

            # macro (per-graph mean of per-graph keep rates)
            G_n = int(node_batch.max().item()) + 1
            G_e = int(edge_batch.max().item()) + 1

            keep_sum_pg_n = torch.zeros(G_n, Xn.size(1), device=Xn.device)
            cnt_sum_pg_n  = torch.zeros(G_n, Xn.size(1), device=Xn.device)
            keep_sum_pg_n.index_add_(0, node_batch, Xn)
            cnt_sum_pg_n.index_add_(0, node_batch, torch.ones_like(Xn))
            keep_pg_n = (keep_sum_pg_n / (cnt_sum_pg_n + 1e-8)).mean(dim=0).cpu()  # (K,)

            keep_sum_pg_e = torch.zeros(G_e, Xe.size(1), device=Xe.device)
            cnt_sum_pg_e  = torch.zeros(G_e, Xe.size(1), device=Xe.device)
            keep_sum_pg_e.index_add_(0, edge_batch, Xe)
            cnt_sum_pg_e.index_add_(0, edge_batch, torch.ones_like(Xe))
            keep_pg_e = (keep_sum_pg_e / (cnt_sum_pg_e + 1e-8)).mean(dim=0).cpu()  # (K,)

            acc_node_keep_pg_sum = keep_pg_n if acc_node_keep_pg_sum is None else acc_node_keep_pg_sum + keep_pg_n
            acc_edge_keep_pg_sum = keep_pg_e if acc_edge_keep_pg_sum is None else acc_edge_keep_pg_sum + keep_pg_e
            acc_node_pg_count += 1
            acc_edge_pg_count += 1
        # ----------------------------------------------------------

    # shapes now consistent: concat along batch dimension
    gate_weights_all = torch.cat(gate_weight_accumulator, dim=0)  # (N, K)
    load_balance = gate_weights_all.mean(dim=0)

    # --- NEW: top-1 frequency per expert across the epoch ---
    top1_idx = gate_weights_all.argmax(dim=1)                                # (N,)
    K = gate_weights_all.size(1)
    top1_counts = torch.bincount(top1_idx, minlength=K).float()              # (K,)
    top1_share  = top1_counts / top1_counts.sum().clamp_min(1.0)             # (K,)

    if not config.get("experiment", {}).get("hyper_search", {}).get("enable", False):
        print("\nGate Load & Top-1 share per Expert:")
        total_n = int(top1_counts.sum().item())
        for i in range(K):
            print(
                f"  Expert {i}: "
                f"load={load_balance[i].item():.4f} | "
                f"top1={top1_share[i].item():.4f} "
                f"({int(top1_counts[i].item())}/{total_n})")
 

    final_outputs = torch.cat(all_aggregated_outputs, dim=0)
    final_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(final_outputs, final_targets, dataset_info['metric'])
    metrics['loss'] = total_loss / len(loader.dataset)
    metrics['loss_ce'] = total_ce_loss / len(loader.dataset)
    metrics['loss_reg'] = total_reg_loss / len(loader.dataset)
    # metrics['loss_la'] = total_la_loss / len(loader.dataset)
    # metrics['loss_ea'] = total_ea_loss / len(loader.dataset)
    # metrics['loss_str'] = total_str_loss / len(loader.dataset)
    metrics['loss_div'] = total_div_loss / len(loader.dataset)
    metrics['loss_gate'] = total_gate_loss / len(loader.dataset)
    metrics['load_balance'] = load_balance.tolist()

    # ------- NEW: compute & print epoch-level drop rates -------
    if (acc_node_keep_sum is not None) and (acc_edge_keep_sum is not None):
        node_keep_micro = acc_node_keep_sum / max(acc_node_cnt, 1)
        edge_keep_micro = acc_edge_keep_sum / max(acc_edge_cnt, 1)
        node_drop_micro = (1.0 - node_keep_micro).tolist()
        edge_drop_micro = (1.0 - edge_keep_micro).tolist()

        node_keep_macro = acc_node_keep_pg_sum / max(acc_node_pg_count, 1) if acc_node_keep_pg_sum is not None else None
        edge_keep_macro = acc_edge_keep_pg_sum / max(acc_edge_pg_count, 1) if acc_edge_keep_pg_sum is not None else None
        node_drop_macro = (1.0 - node_keep_macro).tolist() if node_keep_macro is not None else None
        edge_drop_macro = (1.0 - edge_keep_macro).tolist() if edge_keep_macro is not None else None

        # print("\n[Epoch mask drop rates] micro (size-weighted across all items):")
        # for i, (nd, ed) in enumerate(zip(node_drop_micro, edge_drop_micro)):
        #     print(f"  Expert {i}: node={nd:.3f}, edge={ed:.3f}")

        if node_drop_macro is not None:
            print("\n[Epoch mask drop rates] macro (mean across graphs):")
            for i, (nd, ed) in enumerate(zip(node_drop_macro, edge_drop_macro)):
                print(f"  Expert {i}: node={nd:.3f}, edge={ed:.3f}")

        # also return in metrics for logging
        metrics['node_drop_micro'] = node_drop_micro
        metrics['edge_drop_micro'] = edge_drop_micro
        if node_drop_macro is not None:
            metrics['node_drop_macro'] = node_drop_macro
            metrics['edge_drop_macro'] = edge_drop_macro
    # -----------------------------------------------------------

    gc.collect(); torch.cuda.empty_cache()
    return metrics


def evaluate_moe(model, loader, device, metric_type, epoch, config):
    model.eval()
    total_loss = total_ce_loss = total_reg_loss = 0
    total_la_loss = total_ea_loss = total_str_loss = total_div_loss = total_gate_loss = 0
    all_targets = []
    all_aggregated_outputs = []
    all_mv_counts = []  # majority-vote "logits" = per-class vote counts (with tiny tie-breaker)
    all_top1_logits, all_top2u_logits, all_top2w_logits = [], [], []
    all_gap_means = []

    gate_weight_accumulator = []

    if config['model']['parallel']:
        verbose = model.module.verbose
        model.module.set_epoch(epoch)
    else:
        verbose = model.verbose
        model.set_epoch(epoch)

    # ------- NEW: epoch-level accumulators for mask keep/drop -------
    acc_node_keep_sum = None; acc_node_cnt = 0
    acc_edge_keep_sum = None; acc_edge_cnt = 0
    acc_node_keep_pg_sum = None; acc_node_pg_count = 0
    acc_edge_keep_pg_sum = None; acc_edge_pg_count = 0
    # ----------------------------------------------------------------
    # ------- NEW: per-expert logit accumulators (filled after 1st batch) -------
    per_expert_logits_accum = None  # list of length K; each item list of tensors

    pbar = tqdm(loader, desc='Evaluating MoEUIL', leave=False)
    for data in pbar:
        data = data.to(device)
        with autocast():
            aggregated_outputs = model(data)

        batch_size = data.y.size(0)

        # --- FIX: gate_weights is (B, K); no squeeze/transpose ---
        gate_weights = aggregated_outputs['gate_weights']  # (B, K)
        gate_weight_accumulator.append(gate_weights.cpu())

        expert_logits = aggregated_outputs['expert_logits']  # (B, K, C)

        gap_mean = aggregated_outputs['gap_mean']
        all_gap_means.append(gap_mean.cpu())

        # ------- NEW: initialize per-expert accumulators after we know K -------
        if per_expert_logits_accum is None:
            _, K, _ = expert_logits.size()
            per_expert_logits_accum = [[] for _ in range(K)]
        # Accumulate each expert’s logits for this batch
        for k in range(len(per_expert_logits_accum)):
            per_expert_logits_accum[k].append(expert_logits[:, k, :].detach())
        # -----------------------------------------------------------------------

        # --- Majority vote across experts (unweighted) ---
        # Each expert votes for argmax class; count votes per class.
        B, K, C = expert_logits.size()

        # Top-k expert indices by gate (per sample)
        topk = min(2, K)
        topk_idx = gate_weights.topk(k=topk, dim=1).indices        # (B, topk)

        # ---- Top-1: take the single best expert’s logits
        top1_logits = expert_logits[torch.arange(B, device=expert_logits.device).unsqueeze(1),
                                    topk_idx[:, :1], :].squeeze(1) # (B, C)
        all_top1_logits.append(top1_logits.detach())

        # ---- Top-2 (unweighted): mean of the two experts’ logits
        if topk >= 2:
            g2 = expert_logits[torch.arange(B, device=expert_logits.device).unsqueeze(1),
                            topk_idx[:, :2], :]                  # (B, 2, C)
            top2u_logits = g2.mean(dim=1)                           # (B, C)
            all_top2u_logits.append(top2u_logits.detach())

            # ---- Top-2 (gate-weighted): normalize gates over top-2 then weight-sum
            w2 = gate_weights.gather(1, topk_idx[:, :2])            # (B, 2)
            w2 = w2 / (w2.sum(dim=1, keepdim=True) + 1e-8)
            top2w_logits = (w2.unsqueeze(-1) * g2).sum(dim=1)       # (B, C)
            all_top2w_logits.append(top2w_logits.detach())

        per_expert_preds = expert_logits.argmax(dim=2)                   # (B, K)
        vote_counts = F.one_hot(per_expert_preds, num_classes=C).sum(1).float()  # (B, C)

        # Tiny tie-breaker: nudge by gate-weighted logits so ties resolve consistently
        # (Counts dominate; logits are scaled tiny so they only break exact ties.)
        mv_counts = vote_counts + 1e-3 * aggregated_outputs['logits']    # (B, C)
        all_mv_counts.append(mv_counts.detach())

        total_loss += aggregated_outputs['loss_total'].item() * batch_size
        total_ce_loss += aggregated_outputs['loss_ce'].item() * batch_size
        total_reg_loss += aggregated_outputs['loss_reg'].item() * batch_size
        total_la_loss += aggregated_outputs['loss_la'].item() * batch_size
        total_ea_loss += aggregated_outputs['loss_ea'].item() * batch_size
        total_str_loss += aggregated_outputs['loss_str'].item() * batch_size
        total_div_loss += aggregated_outputs['loss_div'].item() * batch_size
        total_gate_loss += aggregated_outputs['loss_gate'].item() * batch_size
        all_targets.append(data.y.detach())
        all_aggregated_outputs.append(aggregated_outputs['logits'].detach())

        # ------- NEW: accumulate mask stats from THIS batch -------
        nm = aggregated_outputs.get('node_masks', None)    # (N, K, 1)
        em = aggregated_outputs.get('edge_masks', None)    # (E, K, 1)
        if (nm is not None) and (em is not None):
            node_batch = data.batch
            edge_batch = data.batch[data.edge_index[0]]

            Xn = nm.detach().float().squeeze(-1)   # (N,K)
            Xe = em.detach().float().squeeze(-1)   # (E,K)

            # micro
            kn = Xn.sum(dim=0).cpu()
            ke = Xe.sum(dim=0).cpu()
            acc_node_keep_sum = kn if acc_node_keep_sum is None else acc_node_keep_sum + kn
            acc_edge_keep_sum = ke if acc_edge_keep_sum is None else acc_edge_keep_sum + ke
            acc_node_cnt += Xn.size(0)
            acc_edge_cnt += Xe.size(0)

            # macro
            G_n = int(node_batch.max().item()) + 1
            G_e = int(edge_batch.max().item()) + 1

            keep_sum_pg_n = torch.zeros(G_n, Xn.size(1), device=Xn.device)
            cnt_sum_pg_n  = torch.zeros(G_n, Xn.size(1), device=Xn.device)
            keep_sum_pg_n.index_add_(0, node_batch, Xn)
            cnt_sum_pg_n.index_add_(0, node_batch, torch.ones_like(Xn))
            keep_pg_n = (keep_sum_pg_n / (cnt_sum_pg_n + 1e-8)).mean(dim=0).cpu()

            keep_sum_pg_e = torch.zeros(G_e, Xe.size(1), device=Xe.device)
            cnt_sum_pg_e  = torch.zeros(G_e, Xe.size(1), device=Xe.device)
            keep_sum_pg_e.index_add_(0, edge_batch, Xe)
            cnt_sum_pg_e.index_add_(0, edge_batch, torch.ones_like(Xe))
            keep_pg_e = (keep_sum_pg_e / (cnt_sum_pg_e + 1e-8)).mean(dim=0).cpu()

            acc_node_keep_pg_sum = keep_pg_n if acc_node_keep_pg_sum is None else acc_node_keep_pg_sum + keep_pg_n
            acc_edge_keep_pg_sum = keep_pg_e if acc_edge_keep_pg_sum is None else acc_edge_keep_pg_sum + keep_pg_e
            acc_node_pg_count += 1
            acc_edge_pg_count += 1
        # ----------------------------------------------------------

    # shapes now consistent: concat along batch dimension
    gate_weights_all = torch.cat(gate_weight_accumulator, dim=0)  # (N, K)
    load_balance = gate_weights_all.mean(dim=0)

    # --- NEW: top-1 frequency per expert across the epoch ---
    top1_idx = gate_weights_all.argmax(dim=1)                                # (N,)
    K = gate_weights_all.size(1)
    top1_counts = torch.bincount(top1_idx, minlength=K).float()              # (K,)
    top1_share  = top1_counts / top1_counts.sum().clamp_min(1.0)             # (K,)

    if not config.get("experiment", {}).get("hyper_search", {}).get("enable", False):
        print("\nGate Load & Top-1 share per Expert:")
        total_n = int(top1_counts.sum().item())
        for i in range(K):
            print(
                f"  Expert {i}: "
                f"load={load_balance[i].item():.4f} | "
                f"top1={top1_share[i].item():.4f} "
                f"({int(top1_counts[i].item())}/{total_n})")

    final_outputs = torch.cat(all_aggregated_outputs, dim=0)
    final_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(final_outputs, final_targets, metric_type)

    # --- Majority vote accuracy ---
    mv_counts_all = torch.cat(all_mv_counts, dim=0)  # (N, C)
    mv_metrics = compute_metrics(mv_counts_all, final_targets, metric_type)
    print("Majority-vote metrics:", {f"mv_{k}": v for k, v in mv_metrics.items()})
    for k, v in mv_metrics.items():
        metrics[f"mv_{k}"] = v

    # --- Top-1 / Top-2 metrics
    if len(all_top1_logits) > 0:
        top1_all = torch.cat(all_top1_logits, dim=0)  # (N, C)
        top1_metrics = compute_metrics(top1_all, final_targets, metric_type)
        print("Top-1-expert metrics:", {f"top1_{k}": v for k, v in top1_metrics.items()})
        for k, v in top1_metrics.items():
            metrics[f"top1_{k}"] = v

    # Only if at least 2 experts exist
    if K >= 2 and len(all_top2u_logits) > 0 and len(all_top2w_logits) > 0:
        top2u_all = torch.cat(all_top2u_logits, dim=0)
        top2w_all = torch.cat(all_top2w_logits, dim=0)

        top2u_metrics = compute_metrics(top2u_all, final_targets, metric_type)
        top2w_metrics = compute_metrics(top2w_all, final_targets, metric_type)

        print("Top-2 (unweighted) metrics:", {f"top2u_{k}": v for k, v in top2u_metrics.items()})
        print("Top-2 (gate-weighted) metrics:", {f"top2w_{k}": v for k, v in top2w_metrics.items()})

        for k, v in top2u_metrics.items():
            metrics[f"top2u_{k}"] = v
        for k, v in top2w_metrics.items():
            metrics[f"top2w_{k}"] = v

    # ------- NEW: Per-expert metrics over the full epoch -------
    gap_mean_avg = sum(all_gap_means) / len(loader.dataset)
    gap_mean_std = torch.tensor(all_gap_means).std()
    print(f"Gap Mean Average: {gap_mean_avg}, Gap Mean Std Dev: {gap_mean_std}")
    # Concatenate per-expert logits and compute metrics against final_targets
    if per_expert_logits_accum is not None:
        print("\nPer-expert metrics:")
        for k in range(K):
            k_all = torch.cat(per_expert_logits_accum[k], dim=0)  # (N, C) or (N, 1)
            k_metrics = compute_metrics(k_all, final_targets, metric_type)
            # Pretty print a compact line; keep primary metric first if present
            primary = k_metrics[metric_type.lower().replace('-', '_')]
            print(f"  Expert {k}:" +
                  " ".join([f"{kk}={vv:.4f}" for kk, vv in list(k_metrics.items())[1:]]))
            # Flatten into overall metrics dict with 'expert{k}_' prefix
            for kk, vv in k_metrics.items():
                metrics[f"expert{k}_{kk}"] = vv

    metrics['loss'] = total_loss / len(loader.dataset)
    metrics['loss_ce'] = total_ce_loss / len(loader.dataset)
    metrics['loss_reg'] = total_reg_loss / len(loader.dataset)
    # metrics['loss_la'] = total_la_loss / len(loader.dataset)
    # metrics['loss_ea'] = total_ea_loss / len(loader.dataset)
    # metrics['loss_str'] = total_str_loss / len(loader.dataset)
    metrics['loss_div'] = total_div_loss / len(loader.dataset)
    metrics['loss_gate'] = total_gate_loss / len(loader.dataset)
    metrics['load_balance'] = load_balance.tolist()


    # ------- NEW: epoch-level drop rates -------
    if (acc_node_keep_sum is not None) and (acc_edge_keep_sum is not None):
        node_keep_micro = acc_node_keep_sum / max(acc_node_cnt, 1)
        edge_keep_micro = acc_edge_keep_sum / max(acc_edge_cnt, 1)
        node_drop_micro = (1.0 - node_keep_micro).tolist()
        edge_drop_micro = (1.0 - edge_keep_micro).tolist()

        node_keep_macro = acc_node_keep_pg_sum / max(acc_node_pg_count, 1) if acc_node_keep_pg_sum is not None else None
        edge_keep_macro = acc_edge_keep_pg_sum / max(acc_edge_pg_count, 1) if acc_edge_keep_pg_sum is not None else None
        node_drop_macro = (1.0 - node_keep_macro).tolist() if node_keep_macro is not None else None
        edge_drop_macro = (1.0 - edge_keep_macro).tolist() if edge_keep_macro is not None else None

        # print("\n[Epoch mask drop rates] micro (size-weighted across all items):")
        # for i, (nd, ed) in enumerate(zip(node_drop_micro, edge_drop_micro)):
        #     print(f"  Expert {i}: node={nd:.3f}, edge={ed:.3f}")

        if node_drop_macro is not None:
            print("\n[Epoch mask drop rates] macro (mean across graphs):")
            for i, (nd, ed) in enumerate(zip(node_drop_macro, edge_drop_macro)):
                print(f"  Expert {i}: node={nd:.3f}, edge={ed:.3f}")

        metrics['node_drop_micro'] = node_drop_micro
        metrics['edge_drop_micro'] = edge_drop_micro
        if node_drop_macro is not None:
            metrics['node_drop_macro'] = node_drop_macro
            metrics['edge_drop_macro'] = edge_drop_macro
    # ------------------------------------------

    gc.collect(); torch.cuda.empty_cache()
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
            # Separate params
            experts_params = list(model.shared.parameters())                   # all experts (encoders, heads, masks, etc.)
            gate_params    = list(model._gate_mlp.parameters())  # gate encoder + MLP
            # Initialize optimizer
            # Build optimizer with param groups
            optimizer = torch.optim.Adam([
                {"params": experts_params, "lr": config['training']['lr'], "weight_decay": config['training']['weight_decay'], "name": "experts"},
                {"params": gate_params,    "lr": config['training']['lr']*0.1, "weight_decay": config['training']['weight_decay'], "name": "gate"},
            ])
            lr_decay_factor = 0.5   # multiply LR by this when patience triggers
            lr_min = 1e-5           # clamp floor; do not decay below this

        
        # Training loop
        if not is_tuning:
            logger.logger.info("Starting training...")
        # primary_metric = 'loss'
        primary_metric = 'accuracy' if metric_type == 'Accuracy' else metric_type.lower().replace('-', '_')
        if primary_metric == 'accuracy' or primary_metric == 'roc_auc':
            best_val_metric = 0
        else:
            best_val_metric = 1000000
        patience_counter = 0
        patience = config['training']['early_stopping']['patience']
        best_epoch = 0  # Track the best epoch
        
        # Adjust epochs if in debug mode
        num_epochs = config['experiment']['debug']['epochs'] if config['experiment']['debug']['enable'] else config['training']['epochs']
        
        for epoch in range(1, num_epochs + 1):
            # Train
            if config['model']['type'] == 'moe':
                train_metrics = train_epoch_moe(model, train_loader, optimizer, dataset_info, device, epoch, config)
                # Validate on OOD validation set
                val_metrics = evaluate_moe(model, val_loader, device, metric_type, epoch, config)                # Validate on in-distribution validation se
                #id_val_metrics = evaluate_moe(model, id_val_loader, device, metric_type, epoch, config)
                id_val_metrics = evaluate_moe(model, test_loader, device, metric_type, epoch, config)
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
            eval_metric = 'accuracy' if metric_type == 'Accuracy' else metric_type.lower().replace('-', '_')
            
            # print(f"val_metrics: {val_metrics}")
            if primary_metric not in val_metrics:
                primary_metric = list(val_metrics.keys())[0]  # Fallback to first metric
                
            current_metric = val_metrics[primary_metric]
            # print(f"current_metric: {current_metric}")
            current_eval_metric = val_metrics[eval_metric]
            # For error metrics like RMSE and MAE, lower is better
            is_better = (current_metric < best_val_metric - config['training']['early_stopping']['min_delta']) if primary_metric in ['rmse', 'mae', 'loss'] else (current_metric > best_val_metric + config['training']['early_stopping']['min_delta'])
            
            if is_better:
                best_val_metric = current_metric
                patience_counter = 0
                best_epoch = epoch  # Update best epoch
                logger.save_model(model, epoch, val_metrics)
                if not is_tuning:
                    logger.logger.info(f"New best model saved with {primary_metric}: {best_val_metric:.4f} and primary metric {eval_metric}: {current_eval_metric:.4f}")
            else:
                # ---- PATIENCE-BASED LR DECAY (no early stop, no cooldown) ----
                patience_counter += 1
                if patience_counter >= patience:
                    # Decay every param group uniformly; clamp to lr_min
                    for i, pg in enumerate(optimizer.param_groups):
                        old_lr = float(pg.get('lr', 0.0))
                        new_lr = max(old_lr * lr_decay_factor, lr_min)
                        if new_lr < old_lr:
                            pg['lr'] = new_lr
                            if not is_tuning:
                                name = pg.get('name', f'group_{i}')
                                logger.logger.info(
                                    f"LR decay @ epoch {epoch}: {name} {old_lr:.6g} -> {new_lr:.6g}"
                                )
                    # Reset patience after a decay so we wait for next window
                    patience_counter = 0
        
        #if is_tuning:
        if False:
            del optimizer
            if 'scaler' in locals():
                del scaler  # if using AMP
            gc.collect()
            torch.cuda.empty_cache()
        
        else:
            # Final evaluation on test sets
            logger.save_model(model, epoch, val_metrics)
            best_val_metric = current_metric
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
            # ---- Gate-only fine-tuning from best checkpoint ----
            try:
                #finetune_gate_only(model, train_loader, val_loader, id_val_loader, dataset_info, device, config, logger, best_val_metric)
                finetune_gate_only(model, train_loader, val_loader, test_loader, dataset_info, device, config, logger, best_val_metric)
            except Exception as e:
                print(f"[WARN] Gate-only fine-tune skipped due to error: {e}")

            # logger.load_best_model(model)
            if config['model']['type'] == 'moe':
                test_ood_metrics = evaluate_moe(model, test_loader, device, metric_type, epoch, config)
                test_id_metrics = evaluate_moe(model, id_test_loader, device, metric_type, epoch, config)
            else:
                test_ood_metrics = evaluate(model, test_loader, device, metric_type)
                test_id_metrics = evaluate(model, id_test_loader, device, metric_type)
            
            # Log test metrics with the best epoch
            logger.log_metrics(test_ood_metrics, best_epoch, phase="test_ood")
            logger.log_metrics(test_id_metrics, best_epoch, phase="test_id")

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
        print(f"Trial completed in {elapsed_time:.2f} seconds with {primary_metric}: {test_ood_metrics[primary_metric]:.4f}")
        # return best_val_metric
        return test_ood_metrics[primary_metric]

    # Calculate average accuracies for test OOD and test ID metrics
    avg_test_ood_primary_metric = sum(metrics[eval_metric] for metrics in all_test_ood_metrics) / len(config['experiment']['seeds'])
    avg_test_id_primary_metric = sum(metrics[eval_metric] for metrics in all_test_id_metrics) / len(config['experiment']['seeds'])
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