import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(outputs, targets, threshold=0.5):
    """
    Compute classification metrics.
    
    Args:
        outputs (torch.Tensor): Model outputs
        targets (torch.Tensor): Ground truth labels
        threshold (float): Threshold for binary classification
        
    Returns:
        dict: Dictionary of metrics
    """
    # Convert to numpy arrays
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Get predictions
    if outputs.shape[1] > 1:  # Multi-class
        preds = np.argmax(outputs, axis=1)
    else:  # Binary
        preds = (outputs > threshold).astype(int)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(targets, preds),
        'precision': precision_score(targets, preds, average='macro', zero_division=0),
        'recall': recall_score(targets, preds, average='macro', zero_division=0),
        'f1': f1_score(targets, preds, average='macro', zero_division=0)
    }
    
    return metrics 