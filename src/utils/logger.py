import os
import logging
import wandb
import torch
from datetime import datetime

class Logger:
    """Logging utility for training and evaluation."""
    
    def __init__(self, config):
        self.config = config
        self.log_dir = self._setup_log_dir()
        self.logger = self._setup_logger()
        self.use_wandb = config['logging']['wandb']['enable']
        
        # Initialize wandb if enabled
        if self.use_wandb:
            wandb_config = config['logging']['wandb']
            wandb.init(
                project=wandb_config['project'],
                entity=wandb_config['entity'],
                config=config,
                name=f"{config['experiment']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def _setup_log_dir(self):
        """Create log directory with experiment name and timestamp."""
        exp_name = self.config['experiment']['name']
        model_type = self.config['model']['type']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        log_dir = os.path.join(
            "logs", 
            f"{exp_name}_{model_type}_{timestamp}"
        )
        
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    
    def _setup_logger(self):
        """Set up Python logger."""
        logger = logging.getLogger("GNN_Baseline")
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(os.path.join(self.log_dir, "training.log"))
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_metrics(self, metrics, step, phase="train"):
        """Log metrics to both file and wandb if enabled."""
        # Log to file
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"{phase.capitalize()} - Step {step} | {metrics_str}")
        
        # Log to wandb
        if self.use_wandb:
            wandb_metrics = {f"{phase}/{k}": v for k, v in metrics.items()}
            wandb_metrics["step"] = step
            wandb.log(wandb_metrics)
    
    def save_model(self, model, epoch, metrics):
        """Save model checkpoint."""
        if not self.config['logging']['save_model']:
            return
        
        # Create checkpoints directory
        checkpoints_dir = os.path.join(
            "checkpoints", 
            f"{self.config['experiment']['name']}_{self.config['model']['type']}"
        )
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        # Save model
        checkpoint_path = os.path.join(
            checkpoints_dir, 
            f"epoch_{epoch}_val_acc_{metrics['accuracy']:.4f}.pt"
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics
        }, checkpoint_path)
        
        self.logger.info(f"Model saved to {checkpoint_path}")
    
    def close(self):
        """Close logger and wandb."""
        if self.use_wandb:
            wandb.finish() 