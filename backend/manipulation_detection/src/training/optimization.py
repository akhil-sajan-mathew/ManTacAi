"""Optimization and regularization utilities."""

import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    LinearLR, CosineAnnealingLR, ExponentialLR, 
    StepLR, ReduceLROnPlateau, CyclicLR
)
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from typing import Dict, Any, Optional, List, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class OptimizerFactory:
    """Factory for creating optimizers with different configurations."""
    
    @staticmethod
    def create_optimizer(model: nn.Module, 
                        optimizer_type: str = "adamw",
                        learning_rate: float = 2e-5,
                        weight_decay: float = 0.01,
                        **kwargs) -> torch.optim.Optimizer:
        """Create an optimizer for the model.
        
        Args:
            model: Model to optimize
            optimizer_type: Type of optimizer ("adamw", "adam", "sgd")
            learning_rate: Learning rate
            weight_decay: Weight decay coefficient
            **kwargs: Additional optimizer parameters
            
        Returns:
            Configured optimizer
        """
        # Get model parameters with different learning rates
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and "transformer" in n],
                "weight_decay": weight_decay,
                "lr": learning_rate
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay) and "transformer" in n],
                "weight_decay": 0.0,
                "lr": learning_rate
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if "classifier" in n or "dropout" in n],
                "weight_decay": weight_decay,
                "lr": learning_rate * kwargs.get("classifier_lr_multiplier", 2.0)
            }
        ]
        
        if optimizer_type.lower() == "adamw":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                eps=kwargs.get("eps", 1e-8),
                betas=kwargs.get("betas", (0.9, 0.999))
            )
        elif optimizer_type.lower() == "adam":
            optimizer = Adam(
                optimizer_grouped_parameters,
                eps=kwargs.get("eps", 1e-8),
                betas=kwargs.get("betas", (0.9, 0.999))
            )
        elif optimizer_type.lower() == "sgd":
            optimizer = SGD(
                optimizer_grouped_parameters,
                momentum=kwargs.get("momentum", 0.9),
                nesterov=kwargs.get("nesterov", True)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        logger.info(f"Created {optimizer_type} optimizer with {len(optimizer_grouped_parameters)} parameter groups")
        return optimizer


class SchedulerFactory:
    """Factory for creating learning rate schedulers."""
    
    @staticmethod
    def create_scheduler(optimizer: torch.optim.Optimizer,
                        scheduler_type: str = "linear_warmup",
                        num_training_steps: int = 1000,
                        num_warmup_steps: int = 100,
                        **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create a learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            scheduler_type: Type of scheduler
            num_training_steps: Total number of training steps
            num_warmup_steps: Number of warmup steps
            **kwargs: Additional scheduler parameters
            
        Returns:
            Configured scheduler or None
        """
        if scheduler_type == "linear_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_type == "cosine_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=kwargs.get("num_cycles", 0.5)
            )
        elif scheduler_type == "cosine_annealing":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get("T_max", num_training_steps),
                eta_min=kwargs.get("eta_min", 0)
            )
        elif scheduler_type == "exponential":
            scheduler = ExponentialLR(
                optimizer,
                gamma=kwargs.get("gamma", 0.95)
            )
        elif scheduler_type == "step":
            scheduler = StepLR(
                optimizer,
                step_size=kwargs.get("step_size", num_training_steps // 3),
                gamma=kwargs.get("gamma", 0.1)
            )
        elif scheduler_type == "reduce_on_plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get("mode", "min"),
                factor=kwargs.get("factor", 0.5),
                patience=kwargs.get("patience", 5),
                verbose=True
            )
        elif scheduler_type == "cyclic":
            scheduler = CyclicLR(
                optimizer,
                base_lr=kwargs.get("base_lr", 1e-6),
                max_lr=kwargs.get("max_lr", 1e-4),
                step_size_up=kwargs.get("step_size_up", num_training_steps // 10),
                mode=kwargs.get("mode", "triangular2")
            )
        elif scheduler_type == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        logger.info(f"Created {scheduler_type} scheduler")
        return scheduler


class GradientClipping:
    """Gradient clipping utilities."""
    
    @staticmethod
    def clip_grad_norm(model: nn.Module, max_norm: float = 1.0) -> float:
        """Clip gradients by norm.
        
        Args:
            model: Model with gradients
            max_norm: Maximum gradient norm
            
        Returns:
            Total gradient norm before clipping
        """
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    @staticmethod
    def clip_grad_value(model: nn.Module, clip_value: float = 0.5):
        """Clip gradients by value.
        
        Args:
            model: Model with gradients
            clip_value: Maximum gradient value
        """
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)


class RegularizationTechniques:
    """Various regularization techniques."""
    
    @staticmethod
    def apply_dropout_schedule(model: nn.Module, epoch: int, max_epochs: int, 
                             initial_dropout: float = 0.1, final_dropout: float = 0.3):
        """Apply dropout scheduling during training.
        
        Args:
            model: Model to modify
            epoch: Current epoch
            max_epochs: Total number of epochs
            initial_dropout: Starting dropout rate
            final_dropout: Final dropout rate
        """
        # Linear increase in dropout rate
        progress = epoch / max_epochs
        current_dropout = initial_dropout + (final_dropout - initial_dropout) * progress
        
        # Update dropout layers
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = current_dropout
    
    @staticmethod
    def get_weight_decay_schedule(epoch: int, max_epochs: int,
                                initial_wd: float = 0.01, final_wd: float = 0.1) -> float:
        """Get weight decay value for current epoch.
        
        Args:
            epoch: Current epoch
            max_epochs: Total number of epochs
            initial_wd: Starting weight decay
            final_wd: Final weight decay
            
        Returns:
            Weight decay value for current epoch
        """
        progress = epoch / max_epochs
        return initial_wd + (final_wd - initial_wd) * progress


class LossFunction:
    """Custom loss functions for manipulation detection."""
    
    @staticmethod
    def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, 
                  alpha: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
        """Focal loss for handling class imbalance.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            
        Returns:
            Focal loss value
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    @staticmethod
    def label_smoothing_loss(inputs: torch.Tensor, targets: torch.Tensor,
                           smoothing: float = 0.1) -> torch.Tensor:
        """Label smoothing cross entropy loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            smoothing: Label smoothing factor
            
        Returns:
            Label smoothed loss value
        """
        num_classes = inputs.size(-1)
        log_probs = nn.functional.log_softmax(inputs, dim=-1)
        
        # Create smoothed targets
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
        
        loss = -torch.sum(smooth_targets * log_probs, dim=-1)
        return loss.mean()
    
    @staticmethod
    def class_balanced_loss(inputs: torch.Tensor, targets: torch.Tensor,
                          class_weights: torch.Tensor) -> torch.Tensor:
        """Class-balanced cross entropy loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            class_weights: Weight for each class
            
        Returns:
            Class-balanced loss value
        """
        return nn.functional.cross_entropy(inputs, targets, weight=class_weights)


class MixedPrecisionTraining:
    """Mixed precision training utilities."""
    
    def __init__(self, enabled: bool = True):
        """Initialize mixed precision training.
        
        Args:
            enabled: Whether to enable mixed precision
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.enabled else None
        
        if self.enabled:
            logger.info("Mixed precision training enabled")
        else:
            logger.info("Mixed precision training disabled")
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training.
        
        Args:
            loss: Loss tensor
            
        Returns:
            Scaled loss tensor
        """
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer):
        """Step optimizer with mixed precision.
        
        Args:
            optimizer: Optimizer to step
        """
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients for gradient clipping.
        
        Args:
            optimizer: Optimizer with scaled gradients
        """
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)


class AdversarialTraining:
    """Adversarial training for robustness."""
    
    def __init__(self, epsilon: float = 0.01, alpha: float = 0.001, num_steps: int = 3):
        """Initialize adversarial training.
        
        Args:
            epsilon: Maximum perturbation magnitude
            alpha: Step size for perturbation
            num_steps: Number of adversarial steps
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
    
    def generate_adversarial_examples(self, model: nn.Module, 
                                    input_embeddings: torch.Tensor,
                                    attention_mask: torch.Tensor,
                                    labels: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples using PGD.
        
        Args:
            model: Model to attack
            input_embeddings: Input embeddings
            attention_mask: Attention mask
            labels: True labels
            
        Returns:
            Adversarial embeddings
        """
        model.eval()
        
        # Initialize perturbation
        delta = torch.zeros_like(input_embeddings).uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad = True
        
        for _ in range(self.num_steps):
            # Forward pass with perturbed input
            perturbed_embeddings = input_embeddings + delta
            outputs = model.transformer(inputs_embeds=perturbed_embeddings, 
                                      attention_mask=attention_mask)
            
            # Calculate loss
            pooled_output = outputs.last_hidden_state[:, 0, :]
            logits = model.classifier(model.dropout(pooled_output))
            loss = nn.functional.cross_entropy(logits, labels)
            
            # Calculate gradients
            loss.backward()
            
            # Update perturbation
            grad_sign = delta.grad.sign()
            delta.data = delta.data + self.alpha * grad_sign
            delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
            delta.grad.zero_()
        
        model.train()
        return input_embeddings + delta.detach()