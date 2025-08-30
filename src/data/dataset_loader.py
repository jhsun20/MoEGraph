import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
import os.path as osp
import os
import random
import warnings
from torch_geometric.datasets import *
from data.good import *

# Suppress user warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_dataset(config):
    """
    Load and prepare dataset based on configuration.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Dictionary containing train, validation, and test dataloaders
    """
    dataset_config = config['dataset']
    debug_config = config['experiment']['debug']
    is_tuning = config.get("experiment", {}).get("hyper_search", {}).get("enable", False)
    seed = config['experiment'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Ensure dataset directory exists
    dataset_path = dataset_config['path']
    os.makedirs(dataset_path, exist_ok=True)
    
    # Load dataset with error handling
    transform = NormalizeFeatures()
    dataset_name = dataset_config.get('dataset_name')
    task_type = dataset_config.get('task_type')
    shift_type = dataset_config.get('shift_type')

    if dataset_name[:4] == 'GOOD':
        class_name = eval(f'GOOD{dataset_name[4:]}')
        if dataset_config.get('domain') is not None:
            datasets, meta_info = class_name.load(dataset_root=dataset_path,
                                                  shift=shift_type,
                                                  domain=dataset_config.get('domain'))
        else:
            datasets, meta_info = class_name.load(dataset_root=dataset_path,
                                                  shift=shift_type)
        
        if task_type == 'node_classification':
            dataset = datasets
            datasets = {
                'train': dataset,
                'val': dataset,
                'test': dataset
                }
            for key, dataset in datasets.items():
                setattr(dataset.data, 'y', dataset.data.y.view(-1).long())
                setattr(dataset, 'n_classes', len(torch.unique(dataset.data.y)))
                
        else:
            for key, dataset in datasets.items():
                if key in ['task', 'metric']:
                    continue
                if datasets['task'] == 'Binary classification':
                    setattr(dataset, 'n_classes', 2)
                elif datasets['task'] == 'Regression' or datasets['metric'] == 'MAE':
                    setattr(dataset, 'n_classes', 1)
                else:
                    setattr(dataset, 'n_classes', len(torch.unique(dataset.data.y)))
                setattr(dataset.data, 'y', dataset.data.y.view(-1).long())

    # Debug mode: use only a subset of the data
    if debug_config.get('enable', False):
        print(datasets)
        num_samples = min(debug_config.get('num_samples', 100), len(datasets['train']))
            
        datasets['train'] = datasets['train'][torch.randperm(len(datasets['train']))[:num_samples]]
        datasets['val'] = datasets['val'][torch.randperm(len(datasets['val']))[:num_samples]]
        datasets['id_val'] = datasets['id_val'][torch.randperm(len(datasets['id_val']))[:num_samples]]
        datasets['test'] = datasets['test'][torch.randperm(len(datasets['test']))[:num_samples]]
        datasets['id_test'] = datasets['id_test'][torch.randperm(len(datasets['id_test']))[:num_samples]]
        print(f"Debug mode: using {num_samples} samples")
    
    # Create dataloaders
    batch_size = dataset_config.get('batch_size', 32)
    num_workers = dataset_config.get('num_workers', 0)

    # --- TRAIN LOADER ---
    if dataset_name == 'GOODHIV':
        # Extract graph-level labels for stratification
        labels_train = [int(d.y.item()) for d in datasets['train']]
        # choose how many positives per batch you want
        pos_per_batch = max(1, batch_size // 16)  # e.g., ~6% of batch; tune as you like
        train_sampler = StratifiedBatchSampler(labels_train, batch_size=batch_size, pos_per_batch=pos_per_batch)
        train_loader = DataLoader(
            datasets['train'],
            batch_sampler=train_sampler,
            num_workers=num_workers
        )
    else:
        train_loader = DataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        
    val_loader = DataLoader(
        datasets['val'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    id_val_loader = DataLoader(
        datasets['id_val'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        datasets['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    id_test_loader = DataLoader(
        datasets['id_test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # Get dataset info
    dataset_info = {
        'num_features': meta_info.dim_node,
        'num_classes': meta_info.num_classes,
        'num_train_samples': len(datasets['train']),
        'num_val_samples': len(datasets['val']),
        'num_test_samples': len(datasets['test']),
        'dataset_name': dataset_name,
        'shift_type': shift_type,
        'task_type': task_type,
        'metric': datasets['metric'],
        'num_envs': meta_info.num_envs
    }
    
    # Improve print statements for better readability
    if not is_tuning:
        print(f"\n=== Dataset Information ===")
        print(f"Dataset: {dataset_name} ({shift_type})")
        print(f"Task type: {task_type}")
        print(f"Features: {meta_info.dim_node}, Classes: {meta_info.num_classes}")
        print(f"Samples - Train: {len(datasets['train'])}, Val: {len(datasets['val'])}, ID Val: {len(datasets['id_val'])}, Test: {len(datasets['test'])}, ID Test: {len(datasets['id_test'])}")
        print(f"Evaluation metric: {datasets['metric']}")
        print(f"Number of environments: {meta_info.num_envs}")
        print(f"===============================\n")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'id_val_loader': id_val_loader,
        'test_loader': test_loader,
        'id_test_loader': id_test_loader,
        'dataset_info': dataset_info
    }


class StratifiedBatchSampler:
    """
    Guarantees at least `pos_per_batch` positives in every batch.
    For imbalanced binary datasets like GOOD-HIV.
    """
    def __init__(self, labels, batch_size, pos_per_batch=1, drop_last=False):
        import numpy as np
        self.labels = np.array(labels, dtype=int)
        self.pos_idx = np.where(self.labels == 1)[0]
        self.neg_idx = np.where(self.labels == 0)[0]
        self.batch_size = int(batch_size)
        self.pos_per_batch = int(min(pos_per_batch, self.batch_size))
        self.drop_last = drop_last

    def __iter__(self):
        import numpy as np
        # Shuffle pools each epoch
        pos = np.random.permutation(self.pos_idx)
        neg = np.random.permutation(self.neg_idx)
        p_ptr, n_ptr = 0, 0
        # Replenish by wrapping (positives are rare)
        while True:
            if p_ptr + self.pos_per_batch > len(pos):  pos = np.random.permutation(self.pos_idx); p_ptr = 0
            need_neg = self.batch_size - self.pos_per_batch
            if n_ptr + need_neg > len(neg):            neg = np.random.permutation(self.neg_idx); n_ptr = 0
            batch = np.concatenate([pos[p_ptr:p_ptr+self.pos_per_batch],
                                    neg[n_ptr:n_ptr+need_neg]])
            p_ptr += self.pos_per_batch; n_ptr += need_neg
            # local shuffle inside batch
            np.random.shuffle(batch)
            yield batch.tolist()
            # stop when roughly one epoch of negatives is consumed
            if n_ptr >= len(self.neg_idx):
                if not self.drop_last:
                    continue
                break

    def __len__(self):
        # approximate number of batches per epoch (controlled by negatives)
        return max(1, len(self.neg_idx) // max(1, self.batch_size - self.pos_per_batch))
