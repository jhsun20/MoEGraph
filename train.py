import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import sys

# Add project root to path if needed
if os.path.exists('datasets'):
    sys.path.append('.')

# Import the GOOD dataset loader
try:
    from datasets.good_loader import load_good_dataset
    from datasets.config import cfg  # Import config with shift_type and seed
except ImportError:
    raise ImportError("Please ensure the GOOD benchmark and loader are properly installed")

# Import our GNN model
from gnn import GNN

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GNN for graph classification')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--dataset', type=str, default='GOODbbbp', help='Dataset name (must start with GOOD)')
    parser.add_argument('--data_root', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--gnn_type', type=str, default='gin', help='GNN type (gin, gcn, sage)')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Hidden channels')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    return {}

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        
        # Handle both binary and multi-class classification
        if out.size(1) == 1:  # Binary classification
            loss = F.binary_cross_entropy_with_logits(out.view(-1), data.y.float())
        else:  # Multi-class classification
            loss = F.cross_entropy(out, data.y)
            
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device, is_binary=False):
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            
            if is_binary:
                y_scores.append(torch.sigmoid(out).cpu().numpy())
                pred = (out > 0).float().cpu().numpy()
            else:
                pred = out.argmax(dim=1).cpu().numpy()
            
            y_true.append(data.y.cpu().numpy())
            y_pred.append(pred)
    
    y_true = torch.cat([torch.tensor(batch) for batch in y_true], dim=0).numpy()
    y_pred = torch.cat([torch.tensor(batch) for batch in y_pred], dim=0).numpy()
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate ROC-AUC for binary classification
    if is_binary and len(y_scores) > 0:
        y_scores = torch.cat([torch.tensor(batch) for batch in y_scores], dim=0).numpy()
        try:
            auc = roc_auc_score(y_true, y_scores)
            return accuracy, auc
        except:
            return accuracy, 0.0
    
    return accuracy, None

def main():
    args = parse_args()
    
    # Load config file if it exists
    config = load_config(args.config)
    
    # Override config with command line arguments
    for key, value in vars(args).items():
        if key not in config:
            config[key] = value
    
    # Set random seed
    torch.manual_seed(config['seed'])
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # Create directories
    os.makedirs(config.get('log_dir', 'logs'), exist_ok=True)
    os.makedirs(config.get('checkpoint_dir', 'checkpoints'), exist_ok=True)
    os.makedirs(config.get('results_dir', 'results'), exist_ok=True)
    
    # Load dataset using the imported function
    dataset_name = config.get('dataset', 'GOODbbbp')
    data_root = config.get('data_root', './data')
    
    # Set shift type in config if specified
    if 'shift_type' in config:
        cfg.dataset.shift_type = config['shift_type']
    
    # Load the dataset using the imported function
    datasets = load_good_dataset(dataset_name, data_root)
    
    # Get train/val/test splits
    train_dataset = datasets.get('train')
    val_dataset = datasets.get('val')
    id_test_dataset = datasets.get('id_test', datasets.get('test'))  # Fallback to 'test' if 'id_test' not available
    ood_test_dataset = datasets.get('ood_test')
    
    # Create data loaders
    batch_size = config.get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) if train_dataset else None
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
    id_test_loader = DataLoader(id_test_dataset, batch_size=batch_size) if id_test_dataset else None
    ood_test_loader = DataLoader(ood_test_dataset, batch_size=batch_size) if ood_test_dataset else None
    
    # Determine input and output dimensions
    in_channels = train_dataset.data.x.size(1) if train_dataset and hasattr(train_dataset.data, 'x') else 0
    
    # Determine task type and number of classes
    task = datasets.get('task', 'classification')
    is_binary = task == 'Binary classification'
    is_regression = task == 'Regression'
    
    if is_regression:
        out_channels = 1
        is_binary = False
    else:
        out_channels = train_dataset.n_classes if hasattr(train_dataset, 'n_classes') else 1
        is_binary = (out_channels == 1)
    
    # Create model
    model = GNN(
        in_channels=in_channels,
        hidden_channels=config.get('hidden_channels', 64),
        out_channels=out_channels,
        num_layers=config.get('num_layers', 3),
        dropout=config.get('dropout', 0.5),
        gnn_type=config.get('gnn_type', 'gin')
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
    
    # Training loop
    best_val_acc = 0
    checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, f"{dataset_name}_{config.get('gnn_type', 'gin')}_best.pt")
    
    for epoch in range(1, config.get('epochs', 100) + 1):
        train_loss = train(model, train_loader, optimizer, device)
        val_acc, val_auc = evaluate(model, val_loader, device, is_binary)
        
        # Print progress
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}')
            if val_auc is not None:
                print(f'Val AUC: {val_auc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
    
    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))
    
    # Final evaluation
    id_test_acc, id_test_auc = evaluate(model, id_test_loader, device, is_binary)
    ood_test_acc, ood_test_auc = evaluate(model, ood_test_loader, device, is_binary)
    
    # Print results
    print(f'ID Test Acc: {id_test_acc:.4f}')
    if id_test_auc is not None:
        print(f'ID Test AUC: {id_test_auc:.4f}')
    
    print(f'OOD Test Acc: {ood_test_acc:.4f}')
    if ood_test_auc is not None:
        print(f'OOD Test AUC: {ood_test_auc:.4f}')
    
    # Save results
    results = {
        'dataset': dataset_name,
        'shift_type': cfg.dataset.shift_type,
        'gnn_type': config.get('gnn_type', 'gin'),
        'id_test_acc': id_test_acc,
        'id_test_auc': id_test_auc,
        'ood_test_acc': ood_test_acc,
        'ood_test_auc': ood_test_auc
    }
    
    results_dir = config.get('results_dir', 'results')
    results_path = os.path.join(
        results_dir,
        f"{dataset_name}_{cfg.dataset.shift_type}_{config.get('gnn_type', 'gin')}.yaml"
    )
    
    with open(results_path, 'w') as f:
        yaml.dump(results, f)

if __name__ == '__main__':
    main() 