"""Training utilities for manipulation detection model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from datetime import datetime
import os

from ..models.model_utils import ModelConfig, ModelCheckpoint
from ..utils.config import get_label_mapping

logger = logging.getLogger(__name__)


class ManipulationTrainer:
    """Trainer class for manipulation detection models."""
    
    def __init__(self,
                 model: nn.Module,
                 config: ModelConfig,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 checkpoint_dir: str = "models/checkpoints"):
        """Initialize the trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision and torch.cuda.is_available() else None
        
        # Setup checkpointing
        self.checkpoint_manager = ModelCheckpoint(checkpoint_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        # Label mapping for logging
        self.label_mapping = get_label_mapping()
        
        logger.info(f"Trainer initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Mixed precision: {config.mixed_precision}")
        logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup the optimizer with different learning rates for different layers."""
        # Different learning rates for transformer and classifier
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and "transformer" in n],
                "weight_decay": self.config.weight_decay,
                "lr": self.config.learning_rate
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and "transformer" in n],
                "weight_decay": 0.0,
                "lr": self.config.learning_rate
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if "classifier" in n],
                "weight_decay": self.config.weight_decay,
                "lr": self.config.learning_rate * 2  # Higher LR for classifier
            }
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)
        return optimizer
    
    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        total_steps = len(self.train_loader) * self.config.num_epochs
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        return scheduler
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask, labels)
                    loss = outputs['loss']
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                
                # Optimizer step
                self.optimizer.step()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        epoch_loss = total_loss / num_batches
        current_lr = self.optimizer.param_groups[0]['lr']
        
        return {
            'train_loss': epoch_loss,
            'learning_rate': current_lr
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model.
        
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_ids, attention_mask, labels)
                else:
                    outputs = self.model(input_ids, attention_mask, labels)
                
                loss = outputs['loss']
                logits = outputs['logits']
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                
                # Accumulate metrics
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        val_loss = total_loss / len(self.val_loader)
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        # Calculate per-class accuracy
        per_class_accuracy = {}
        for class_id, class_name in self.label_mapping.items():
            class_mask = np.array(all_labels) == class_id
            if class_mask.sum() > 0:
                class_acc = np.mean(np.array(all_predictions)[class_mask] == class_id)
                per_class_accuracy[class_name] = class_acc
        
        return {
            'val_loss': val_loss,
            'val_accuracy': accuracy,
            'per_class_accuracy': per_class_accuracy,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def train(self) -> Dict[str, List[float]]:
        """Train the model for the specified number of epochs.
        
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['train_loss'])
            self.training_history['val_loss'].append(val_metrics['val_loss'])
            self.training_history['val_accuracy'].append(val_metrics['val_accuracy'])
            self.training_history['learning_rate'].append(train_metrics['learning_rate'])
            
            # Check if this is the best model
            is_best = val_metrics['val_accuracy'] > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_metrics['val_accuracy']
                self.best_val_loss = val_metrics['val_loss']
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                loss=val_metrics['val_loss'],
                metrics=val_metrics,
                config=self.config,
                is_best=is_best
            )
            
            # Log epoch results
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}:")
            logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"  Val Accuracy: {val_metrics['val_accuracy']:.4f}")
            logger.info(f"  Learning Rate: {train_metrics['learning_rate']:.2e}")
            
            # Log per-class accuracy
            logger.info("  Per-class accuracy:")
            for class_name, acc in val_metrics['per_class_accuracy'].items():
                logger.info(f"    {class_name}: {acc:.4f}")
        
        logger.info(f"Training completed!")
        logger.info(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return self.training_history
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Update training state
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        
        logger.info(f"Resumed training from epoch {self.current_epoch}")


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = 'min'):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """Check if training should be stopped.
        
        Args:
            score: Current metric score
            
        Returns:
            True if training should be stopped
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        """Check if the score is an improvement."""
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta