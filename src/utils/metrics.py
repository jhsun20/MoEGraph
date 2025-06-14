import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, average_precision_score,
    roc_auc_score
)

from sklearn.preprocessing import label_binarize
from scipy.special import softmax, expit  # Numerically stable versions

def compute_metrics(outputs, targets, metric_type='Accuracy', threshold=0.5):
    """
    Compute metrics based on the dataset's primary metric type.
    
    Args:
        outputs (torch.Tensor): Model outputs
        targets (torch.Tensor): Ground truth labels
        metric_type (str): Type of metric to compute ('Accuracy', 'RMSE', 'MAE', etc.)
        threshold (float): Threshold for binary classification
        
    Returns:
        dict: Dictionary of metrics
    """
    # Convert to numpy arrays
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Always include the metric type in the results
    metrics['metric_type'] = metric_type
    
    # Handle different metric types
    if metric_type == 'Accuracy':
        # Classification metrics - only compute accuracy
        if outputs.shape[1] > 1:  # Multi-class
            preds = np.argmax(outputs, axis=1)
        else:  # Binary
            preds = (outputs > threshold).astype(int)
        
        metrics['accuracy'] = accuracy_score(targets, preds)
    
    elif metric_type == 'F1':
        # Classification metrics - focus on F1
        if outputs.shape[1] > 1:  # Multi-class
            preds = np.argmax(outputs, axis=1)
        else:  # Binary
            preds = (outputs > threshold).astype(int)
        
        metrics['f1'] = f1_score(targets, preds, average='macro', zero_division=0)
        # Include accuracy as a secondary metric
        metrics['accuracy'] = accuracy_score(targets, preds)
    
    elif metric_type == 'ROC-AUC':
        # ROC-AUC specific metrics
        # print(outputs.shape)
        # print(targets.shape)
        if outputs.shape[1] > 2:  # Multi-class
            probs = softmax(outputs, axis=1)
            try:
                metrics['roc_auc'] = roc_auc_score(targets, probs, multi_class='ovo')
            except ValueError:
                # Fallback if ROC-AUC fails (e.g., only one class present)
                print('valueerror multiclass')
                metrics['roc_auc'] = 0.0
        else:  # Binary
            try:
                # For binary classification, use the second column (probability of positive class)
                if outputs.shape[1] == 2:
                    probs = softmax(outputs, axis=1)[:, 1]  # Get probability of positive class
                else:
                    probs = outputs.squeeze()  # If already single column
                # print(probs.shape)
                metrics['roc_auc'] = roc_auc_score(targets, probs)
            except ValueError:
                print('valueerror binary')
                metrics['roc_auc'] = 0.0
        
        # Include accuracy as a secondary metric
        preds = np.argmax(outputs, axis=1)
        metrics['accuracy'] = accuracy_score(targets, preds)
    
    elif metric_type == 'Average Precision':
        # Average Precision specific metrics
        if outputs.shape[1] > 1:  # Multi-class
            preds = np.argmax(outputs, axis=1)
            probs = outputs
            # For multi-class, compute AP for each class and average
            n_classes = outputs.shape[1]
            ap_scores = []
            for i in range(n_classes):
                binary_targets = (targets == i).astype(int)
                try:
                    ap = average_precision_score(binary_targets, probs[:, i])
                    ap_scores.append(ap)
                except ValueError:
                    ap_scores.append(0.0)
            metrics['average_precision'] = np.mean(ap_scores)
        else:  # Binary
            preds = (outputs > threshold).astype(int)
            try:
                metrics['average_precision'] = average_precision_score(targets, outputs)
            except ValueError:
                metrics['average_precision'] = 0.0
        
        # Include accuracy as a secondary metric
        metrics['accuracy'] = accuracy_score(targets, preds)
    
    elif metric_type == 'RMSE':
        # Regression metrics - focus on RMSE
        if outputs.shape[1] > 1:  # Multiple outputs
            mse = mean_squared_error(targets, outputs, multioutput='raw_values')
            metrics['rmse'] = np.sqrt(np.mean(mse))
        else:  # Single output
            metrics['rmse'] = np.sqrt(mean_squared_error(targets, outputs))
    
    elif metric_type == 'MAE':
        # Regression metrics - focus on MAE
        if outputs.shape[1] > 1:  # Multiple outputs
            mae = mean_absolute_error(targets, outputs, multioutput='raw_values')
            metrics['mae'] = np.mean(mae)
        else:  # Single output
            metrics['mae'] = mean_absolute_error(targets, outputs)
    
    else:
        # Default case - compute basic metrics
        if len(outputs.shape) > 1 and outputs.shape[1] > 1:  # Multi-class
            preds = np.argmax(outputs, axis=1)
        else:  # Binary or regression
            if metric_type in ['Accuracy', 'F1', 'ROC-AUC', 'Average Precision']:
                preds = (outputs > threshold).astype(int)
                metrics['accuracy'] = accuracy_score(targets, preds)
            else:
                # For regression, just compute RMSE and MAE
                metrics['rmse'] = np.sqrt(mean_squared_error(targets, outputs))
                metrics['mae'] = mean_absolute_error(targets, outputs)
    
    # Return the metrics
    return metrics 