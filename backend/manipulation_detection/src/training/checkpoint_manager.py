"""Enhanced checkpoint management for manipulation detection training."""

import torch
import torch.nn as nn
import os
import json
import shutil
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging
import glob
from pathlib import Path

logger = logging.getLogger(__name__)


class EnhancedCheckpointManager:
    """Enhanced checkpoint manager with versioning and metadata."""
    
    def __init__(self, 
                 checkpoint_dir: str = "models/checkpoints",
                 max_checkpoints: int = 10,
                 save_best_only: bool = False,
                 monitor_metric: str = "val_accuracy",
                 mode: str = "max"):
        """Initialize enhanced checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_only: Whether to save only the best checkpoint
            monitor_metric: Metric to monitor for best checkpoint
            mode: 'max' or 'min' for the monitor metric
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        # Track best metric value
        self.best_metric_value = float('-inf') if mode == 'max' else float('inf')
        
        # Metadata file
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.metadata = self._load_metadata()
        
        logger.info(f"Checkpoint manager initialized:")
        logger.info(f"  Directory: {self.checkpoint_dir}")
        logger.info(f"  Max checkpoints: {self.max_checkpoints}")
        logger.info(f"  Monitor metric: {self.monitor_metric} ({mode})")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            "checkpoints": [],
            "best_checkpoint": None,
            "training_history": []
        }
    
    def _save_metadata(self):
        """Save checkpoint metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _is_better_metric(self, current_value: float) -> bool:
        """Check if current metric value is better than the best."""
        if self.mode == 'max':
            return current_value > self.best_metric_value
        else:
            return current_value < self.best_metric_value
    
    def save_checkpoint(self,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       epoch: int,
                       metrics: Dict[str, float],
                       config: Dict[str, Any],
                       model_name: str = "manipulation_classifier",
                       additional_data: Optional[Dict[str, Any]] = None) -> str:
        """Save a comprehensive checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            epoch: Current epoch number
            metrics: Dictionary of evaluation metrics
            config: Model configuration
            model_name: Name of the model
            additional_data: Additional data to save
            
        Returns:
            Path to the saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{model_name}_epoch_{epoch:03d}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'config': config,
            'timestamp': timestamp,
            'model_name': model_name,
            'pytorch_version': torch.__version__,
            'model_info': self._get_model_info(model)
        }
        
        # Add additional data if provided
        if additional_data:
            checkpoint_data.update(additional_data)
        
        # Check if this is the best checkpoint
        monitor_value = metrics.get(self.monitor_metric, 0.0)
        is_best = self._is_better_metric(monitor_value)
        
        if is_best:
            self.best_metric_value = monitor_value
            checkpoint_data['is_best'] = True
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Update metadata
        checkpoint_info = {
            'path': str(checkpoint_path),
            'epoch': epoch,
            'timestamp': timestamp,
            'metrics': metrics,
            'is_best': is_best,
            'file_size_mb': checkpoint_path.stat().st_size / (1024 * 1024)
        }
        
        self.metadata['checkpoints'].append(checkpoint_info)
        
        if is_best:
            self.metadata['best_checkpoint'] = checkpoint_info
            # Save best model separately
            best_path = self.checkpoint_dir / f"best_{model_name}.pt"
            torch.save(checkpoint_data, best_path)
            logger.info(f"Saved best checkpoint: {best_path}")
        
        # Clean up old checkpoints if needed
        if not self.save_best_only:
            self._cleanup_old_checkpoints()
        
        # Save metadata
        self._save_metadata()
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        logger.info(f"Metrics: {metrics}")
        
        return str(checkpoint_path)
    
    def _get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get model information for checkpoint metadata."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_class': model.__class__.__name__,
            'device': str(next(model.parameters()).device)
        }
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit."""
        if len(self.metadata['checkpoints']) <= self.max_checkpoints:
            return
        
        # Sort checkpoints by epoch (oldest first)
        sorted_checkpoints = sorted(self.metadata['checkpoints'], key=lambda x: x['epoch'])
        
        # Remove oldest checkpoints
        checkpoints_to_remove = sorted_checkpoints[:-self.max_checkpoints]
        
        for checkpoint_info in checkpoints_to_remove:
            checkpoint_path = Path(checkpoint_info['path'])
            if checkpoint_path.exists() and not checkpoint_info.get('is_best', False):
                checkpoint_path.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint_path}")
        
        # Update metadata
        self.metadata['checkpoints'] = sorted_checkpoints[-self.max_checkpoints:]
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file. If None, loads best checkpoint.
            
        Returns:
            Checkpoint data dictionary
        """
        if checkpoint_path is None:
            # Load best checkpoint
            if self.metadata['best_checkpoint']:
                checkpoint_path = self.metadata['best_checkpoint']['path']
            else:
                raise ValueError("No best checkpoint found")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Loaded checkpoint from: {checkpoint_path}")
        logger.info(f"Epoch: {checkpoint['epoch']}, Metrics: {checkpoint['metrics']}")
        
        return checkpoint
    
    def load_best_checkpoint(self) -> Dict[str, Any]:
        """Load the best checkpoint.
        
        Returns:
            Best checkpoint data dictionary
        """
        return self.load_checkpoint(None)
    
    def get_checkpoint_history(self) -> List[Dict[str, Any]]:
        """Get the history of all checkpoints.
        
        Returns:
            List of checkpoint information dictionaries
        """
        return self.metadata['checkpoints']
    
    def get_best_checkpoint_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the best checkpoint.
        
        Returns:
            Best checkpoint information or None
        """
        return self.metadata.get('best_checkpoint')
    
    def restore_model_from_checkpoint(self, 
                                    model: nn.Module,
                                    checkpoint_path: Optional[str] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Restore model from checkpoint.
        
        Args:
            model: Model instance to load weights into
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Tuple of (restored_model, checkpoint_info)
        """
        checkpoint = self.load_checkpoint(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint
    
    def export_checkpoint_summary(self, output_path: str):
        """Export checkpoint summary to file.
        
        Args:
            output_path: Path to save the summary
        """
        summary = {
            'checkpoint_directory': str(self.checkpoint_dir),
            'total_checkpoints': len(self.metadata['checkpoints']),
            'best_checkpoint': self.metadata['best_checkpoint'],
            'monitor_metric': self.monitor_metric,
            'mode': self.mode,
            'checkpoints': self.metadata['checkpoints']
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Exported checkpoint summary to: {output_path}")
    
    def cleanup_all_checkpoints(self, keep_best: bool = True):
        """Remove all checkpoints except optionally the best one.
        
        Args:
            keep_best: Whether to keep the best checkpoint
        """
        for checkpoint_info in self.metadata['checkpoints']:
            if keep_best and checkpoint_info.get('is_best', False):
                continue
            
            checkpoint_path = Path(checkpoint_info['path'])
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info(f"Removed checkpoint: {checkpoint_path}")
        
        # Update metadata
        if keep_best and self.metadata['best_checkpoint']:
            self.metadata['checkpoints'] = [self.metadata['best_checkpoint']]
        else:
            self.metadata['checkpoints'] = []
            self.metadata['best_checkpoint'] = None
        
        self._save_metadata()
        logger.info("Cleaned up all checkpoints")


class AutoCheckpointManager:
    """Automatic checkpoint management with configurable strategies."""
    
    def __init__(self, 
                 checkpoint_manager: EnhancedCheckpointManager,
                 save_frequency: int = 1,
                 save_on_improvement: bool = True,
                 patience: int = 5):
        """Initialize automatic checkpoint manager.
        
        Args:
            checkpoint_manager: Enhanced checkpoint manager instance
            save_frequency: Save checkpoint every N epochs
            save_on_improvement: Save checkpoint when metric improves
            patience: Number of epochs without improvement before stopping
        """
        self.checkpoint_manager = checkpoint_manager
        self.save_frequency = save_frequency
        self.save_on_improvement = save_on_improvement
        self.patience = patience
        
        self.epochs_without_improvement = 0
        self.should_stop_early = False
    
    def should_save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Determine if checkpoint should be saved.
        
        Args:
            epoch: Current epoch
            metrics: Current metrics
            
        Returns:
            True if checkpoint should be saved
        """
        # Always save on specified frequency
        if (epoch + 1) % self.save_frequency == 0:
            return True
        
        # Save on improvement
        if self.save_on_improvement:
            monitor_value = metrics.get(self.checkpoint_manager.monitor_metric, 0.0)
            if self.checkpoint_manager._is_better_metric(monitor_value):
                self.epochs_without_improvement = 0
                return True
            else:
                self.epochs_without_improvement += 1
        
        # Check early stopping
        if self.epochs_without_improvement >= self.patience:
            self.should_stop_early = True
            logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
        
        return False
    
    def should_stop_training(self) -> bool:
        """Check if training should be stopped early.
        
        Returns:
            True if training should be stopped
        """
        return self.should_stop_early