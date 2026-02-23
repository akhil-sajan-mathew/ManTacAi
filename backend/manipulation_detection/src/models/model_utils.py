"""Model utilities and configuration management."""

import torch
import torch.nn as nn
import os
import json
import yaml
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from .manipulation_classifier import ManipulationClassifier, ManipulationClassifierWithPooling

logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration class for manipulation detection models."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize model configuration.
        
        Args:
            config_dict: Dictionary containing configuration parameters
        """
        self.config = config_dict
        
        # Model parameters
        self.model_name = config_dict.get('model', {}).get('name', 'distilbert-base-uncased')
        self.num_classes = config_dict.get('model', {}).get('num_classes', 11)
        self.max_length = config_dict.get('model', {}).get('max_length', 512)
        self.dropout_rate = config_dict.get('model', {}).get('dropout_rate', 0.1)
        
        # Training parameters
        self.batch_size = config_dict.get('training', {}).get('batch_size', 16)
        self.learning_rate = config_dict.get('training', {}).get('learning_rate', 2e-5)
        self.num_epochs = config_dict.get('training', {}).get('num_epochs', 5)
        self.warmup_steps = config_dict.get('training', {}).get('warmup_steps', 500)
        self.weight_decay = config_dict.get('training', {}).get('weight_decay', 0.01)
        self.gradient_clip_norm = config_dict.get('training', {}).get('gradient_clip_norm', 1.0)
        
        # Hardware parameters
        self.use_gpu = config_dict.get('hardware', {}).get('use_gpu', True)
        self.mixed_precision = config_dict.get('hardware', {}).get('mixed_precision', True)
        self.num_workers = config_dict.get('hardware', {}).get('num_workers', 4)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ModelConfig':
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            ModelConfig instance
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    @classmethod
    def from_json(cls, config_path: str) -> 'ModelConfig':
        """Load configuration from JSON file.
        
        Args:
            config_path: Path to the JSON configuration file
            
        Returns:
            ModelConfig instance
        """
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config
    
    def save(self, save_path: str):
        """Save configuration to file.
        
        Args:
            save_path: Path to save the configuration
        """
        if save_path.endswith('.yaml') or save_path.endswith('.yml'):
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif save_path.endswith('.json'):
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        else:
            raise ValueError("Configuration file must be .yaml, .yml, or .json")


class ModelFactory:
    """Factory class for creating manipulation detection models."""
    
    @staticmethod
    def create_model(config: ModelConfig, 
                    model_type: str = "standard",
                    pooling_strategy: Optional[str] = None,
                    freeze_base_model: bool = False) -> nn.Module:
        """Create a manipulation detection model.
        
        Args:
            config: Model configuration
            model_type: Type of model ("standard" or "pooling")
            pooling_strategy: Pooling strategy for pooling model
            freeze_base_model: Whether to freeze base model parameters
            
        Returns:
            Initialized model instance
        """
        if model_type == "standard":
            model = ManipulationClassifier(
                model_name=config.model_name,
                num_classes=config.num_classes,
                dropout_rate=config.dropout_rate,
                freeze_base_model=freeze_base_model
            )
        elif model_type == "pooling":
            if pooling_strategy is None:
                pooling_strategy = "cls"
            
            model = ManipulationClassifierWithPooling(
                model_name=config.model_name,
                num_classes=config.num_classes,
                dropout_rate=config.dropout_rate,
                pooling_strategy=pooling_strategy,
                freeze_base_model=freeze_base_model
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Created {model_type} model with {model.get_model_info()['total_parameters']} parameters")
        return model


class ModelCheckpoint:
    """Handles model checkpointing and saving."""
    
    def __init__(self, checkpoint_dir: str = "models/checkpoints"):
        """Initialize model checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       epoch: int,
                       loss: float,
                       metrics: Dict[str, float],
                       config: ModelConfig,
                       is_best: bool = False):
        """Save a model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            epoch: Current epoch number
            loss: Current loss value
            metrics: Dictionary of evaluation metrics
            config: Model configuration
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'metrics': metrics,
            'config': config.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model at epoch {epoch} with loss {loss:.4f}")
        
        logger.info(f"Saved checkpoint at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Dictionary containing checkpoint data
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the latest checkpoint.
        
        Returns:
            Path to the latest checkpoint or None if no checkpoints exist
        """
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("checkpoint_epoch_")]
        if not checkpoints:
            return None
        
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        latest = os.path.join(self.checkpoint_dir, checkpoints[-1])
        return latest
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """Remove old checkpoints, keeping only the last N.
        
        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                      if f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
        
        if len(checkpoints) <= keep_last_n:
            return
        
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        
        # Remove old checkpoints
        for checkpoint in checkpoints[:-keep_last_n]:
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint)
            os.remove(checkpoint_path)
            logger.info(f"Removed old checkpoint: {checkpoint}")


class ModelAnalyzer:
    """Analyzes model architecture and parameters."""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Count model parameters.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params
        }
    
    @staticmethod
    def analyze_model_layers(model: nn.Module) -> List[Dict[str, Any]]:
        """Analyze model layers and their parameters.
        
        Args:
            model: PyTorch model
            
        Returns:
            List of layer information dictionaries
        """
        layer_info = []
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                num_params = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                layer_info.append({
                    'name': name,
                    'type': type(module).__name__,
                    'parameters': num_params,
                    'trainable_parameters': trainable,
                    'frozen': num_params > 0 and trainable == 0
                })
        
        return layer_info
    
    @staticmethod
    def get_model_size_mb(model: nn.Module) -> float:
        """Calculate model size in megabytes.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model size in MB
        """
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb


def load_model_from_config(config_path: str, 
                          model_type: str = "standard",
                          checkpoint_path: Optional[str] = None) -> nn.Module:
    """Load a model from configuration and optionally from checkpoint.
    
    Args:
        config_path: Path to configuration file
        model_type: Type of model to create
        checkpoint_path: Optional path to model checkpoint
        
    Returns:
        Loaded model instance
    """
    # Load configuration
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        config = ModelConfig.from_yaml(config_path)
    else:
        config = ModelConfig.from_json(config_path)
    
    # Create model
    model = ModelFactory.create_model(config, model_type)
    
    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from checkpoint: {checkpoint_path}")
    
    return model