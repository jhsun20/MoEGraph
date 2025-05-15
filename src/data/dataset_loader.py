import torch
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset, MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
import os.path as osp
import os
import random

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
    
    # Ensure dataset directory exists
    dataset_path = dataset_config['path']
    os.makedirs(dataset_path, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(config['experiment']['seed'])
    np.random.seed(config['experiment']['seed'])
    random.seed(config['experiment']['seed'])
    
    # Load dataset with error handling
    transform = NormalizeFeatures()
    dataset_name = dataset_config.get('dataset_name', 'PROTEINS')
    dataset_type = dataset_config.get('dataset_type', 'tu_dataset')

    if dataset_type == 'tu_dataset':
        dataset = TUDataset(
            root=dataset_path,
            name=dataset_name
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    print(f"Successfully loaded {dataset_name} dataset with {len(dataset)} graphs")
    
    # Debug mode: use only a subset of the data
    if debug_config.get('enable', False):
        num_samples = min(debug_config.get('num_samples', 100), len(dataset))
        
        # Use fixed subset if specified
        if debug_config.get('fixed_subset', False):
            # Deterministic sampling based on seed
            indices = torch.arange(len(dataset))
            indices = indices[torch.randperm(len(indices), generator=torch.Generator().manual_seed(42))[:num_samples]]
        else:
            indices = torch.randperm(len(dataset))[:num_samples]
            
        dataset = dataset[indices]
        print(f"Debug mode: using {num_samples} samples")
    
    # Split dataset
    train_ratio = dataset_config['split'].get('train', 0.8)
    val_ratio = dataset_config['split'].get('val', 0.1)
    test_ratio = dataset_config['split'].get('test', 0.1)
    
    # Calculate actual sizes
    num_samples = len(dataset)
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    num_test = num_samples - num_train - num_val
    
    # Create indices for splits using PyTorch instead of sklearn
    indices = torch.randperm(num_samples)
    train_indices = indices[:num_train].tolist()
    val_indices = indices[num_train:num_train+num_val].tolist()
    test_indices = indices[num_train+num_val:].tolist()
    
    # Create dataset splits
    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    test_dataset = dataset[test_indices]
    
    # Create dataloaders
    batch_size = dataset_config.get('batch_size', 32)
    num_workers = dataset_config.get('num_workers', 0)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # Get dataset info
    num_features = dataset.num_features if hasattr(dataset, 'num_features') else dataset[0].x.size(1)
    num_classes = dataset.num_classes if hasattr(dataset, 'num_classes') else len(set([data.y.item() for data in dataset]))
    
    dataset_info = {
        'num_features': num_features,
        'num_classes': num_classes,
        'num_train_samples': len(train_dataset),
        'num_val_samples': len(val_dataset),
        'num_test_samples': len(test_dataset),
        'dataset_name': dataset_name,
        'dataset_type': dataset_type
    }
    
    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test")
    print(f"Features: {num_features}, Classes: {num_classes}")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'dataset_info': dataset_info
    } 