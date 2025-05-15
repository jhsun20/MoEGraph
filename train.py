import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score

# Import the GOOD dataset
try:
    from good_datasets import get_dataset
except ImportError:
    raise ImportError("Please install the GOOD benchmark: pip install good-benchmark")

# Import our GNN model
from gnn import GNN

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GNN for graph classification')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--dataset', type=str, default='good_bbbp', help='Dataset name')
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
            return yaml.safe_load(f)
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
    
    # Load dataset
    dataset = get_dataset(name=config['dataset'])
    
    # Get train/val/test splits
    train_dataset = dataset.get_subset('train')
    val_dataset = dataset.get_subset('val')
    id_test_dataset = dataset.get_subset('id_test')
    ood_test_dataset = dataset.get_subset('ood_test')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    id_test_loader = DataLoader(id_test_dataset, batch_size=config['batch_size'])
    ood_test_loader = DataLoader(ood_test_dataset, batch_size=config['batch_size'])
    
    # Determine input and output dimensions
    in_channels = dataset[0].x.size(1)
    if dataset.task_type == 'classification':
        out_channels = dataset.num_classes
        is_binary = (out_channels == 1)
    else:  # regression
        out_channels = 1
        is_binary = False
    
    # Create model
    model = GNN(
        in_channels=in_channels,
        hidden_channels=config['hidden_channels'],
        out_channels=out_channels,
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        gnn_type=config['gnn_type']
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    # Training loop
    best_val_acc = 0
    for epoch in range(1, config['epochs'] + 1):
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
            torch.save(model.state_dict(), 'best_model.pt')
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    
    # Final evaluation
    id_test_acc, id_test_auc = evaluate(model, id_test_loader, device, is_binary)
    ood_test_acc, ood_test_auc = evaluate(model, ood_test_loader, device, is_binary)
    
    print(f'ID Test Acc: {id_test_acc:.4f}')
    if id_test_auc is not None:
        print(f'ID Test AUC: {id_test_auc:.4f}')
    
    print(f'OOD Test Acc: {ood_test_acc:.4f}')
    if ood_test_auc is not None:
        print(f'OOD Test AUC: {ood_test_auc:.4f}')

if __name__ == '__main__':
    main() 