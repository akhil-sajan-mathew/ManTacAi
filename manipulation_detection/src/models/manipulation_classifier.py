"""Manipulation detection classifier model."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ManipulationClassifier(nn.Module):
    """Transformer-based classifier for manipulation detection."""
    
    def __init__(self, 
                 model_name: str = "distilbert-base-uncased",
                 num_classes: int = 11,
                 dropout_rate: float = 0.1,
                 freeze_base_model: bool = False):
        """Initialize the manipulation classifier.
        
        Args:
            model_name: Name of the pre-trained transformer model
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
            freeze_base_model: Whether to freeze the base model parameters
        """
        super(ManipulationClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Load pre-trained transformer model
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Freeze base model if specified
        if freeze_base_model:
            for param in self.transformer.parameters():
                param.requires_grad = False
            logger.info("Frozen base model parameters")
        
        # Get hidden size from transformer config
        self.hidden_size = self.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # Initialize classifier weights
        self._init_weights()
        
        logger.info(f"Initialized ManipulationClassifier:")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Hidden size: {self.hidden_size}")
        logger.info(f"  Num classes: {num_classes}")
        logger.info(f"  Dropout rate: {dropout_rate}")
        logger.info(f"  Frozen base: {freeze_base_model}")
    
    def _init_weights(self):
        """Initialize the classifier weights."""
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            input_ids: Token IDs tensor of shape (batch_size, seq_len)
            attention_mask: Attention mask tensor of shape (batch_size, seq_len)
            labels: Optional labels tensor of shape (batch_size,)
            
        Returns:
            Dictionary containing logits and optionally loss
        """
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (first token)
        pooled_output = transformer_outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and classification layer
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        outputs = {"logits": logits}
        
        # Calculate loss if labels are provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            outputs["loss"] = loss
        
        return outputs
    
    def predict(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                return_probabilities: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions on input data.
        
        Args:
            input_ids: Token IDs tensor
            attention_mask: Attention mask tensor
            return_probabilities: Whether to return probabilities or logits
            
        Returns:
            Tuple of (predictions, probabilities/logits)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs["logits"]
            
            if return_probabilities:
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
                return predictions, probabilities
            else:
                predictions = torch.argmax(logits, dim=-1)
                return predictions, logits
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "hidden_size": self.hidden_size,
            "dropout_rate": self.dropout_rate,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params
        }
    
    def save_pretrained(self, save_directory: str):
        """Save the model to a directory.
        
        Args:
            save_directory: Directory to save the model
        """
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # Save config
        config_path = os.path.join(save_directory, "config.json")
        config_dict = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
            "hidden_size": self.hidden_size
        }
        
        import json
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_directory: str) -> 'ManipulationClassifier':
        """Load a model from a directory.
        
        Args:
            model_directory: Directory containing the saved model
            
        Returns:
            Loaded ManipulationClassifier instance
        """
        import os
        import json
        
        # Load config
        config_path = os.path.join(model_directory, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create model instance
        model = cls(
            model_name=config["model_name"],
            num_classes=config["num_classes"],
            dropout_rate=config["dropout_rate"]
        )
        
        # Load state dict
        model_path = os.path.join(model_directory, "pytorch_model.bin")
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        logger.info(f"Model loaded from {model_directory}")
        return model


class ManipulationClassifierWithPooling(ManipulationClassifier):
    """Manipulation classifier with different pooling strategies."""
    
    def __init__(self, 
                 model_name: str = "distilbert-base-uncased",
                 num_classes: int = 11,
                 dropout_rate: float = 0.1,
                 pooling_strategy: str = "cls",
                 freeze_base_model: bool = False):
        """Initialize classifier with pooling options.
        
        Args:
            model_name: Name of the pre-trained transformer model
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
            pooling_strategy: Pooling strategy ("cls", "mean", "max")
            freeze_base_model: Whether to freeze the base model parameters
        """
        super().__init__(model_name, num_classes, dropout_rate, freeze_base_model)
        
        self.pooling_strategy = pooling_strategy
        logger.info(f"Using pooling strategy: {pooling_strategy}")
    
    def _pool_hidden_states(self, 
                           hidden_states: torch.Tensor,
                           attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply pooling to hidden states.
        
        Args:
            hidden_states: Hidden states tensor (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask tensor (batch_size, seq_len)
            
        Returns:
            Pooled representation tensor (batch_size, hidden_size)
        """
        if self.pooling_strategy == "cls":
            # Use [CLS] token (first token)
            return hidden_states[:, 0, :]
        
        elif self.pooling_strategy == "mean":
            # Mean pooling over non-padded tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        
        elif self.pooling_strategy == "max":
            # Max pooling over non-padded tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states[mask_expanded == 0] = -1e9  # Set padded tokens to very negative value
            return torch.max(hidden_states, 1)[0]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with custom pooling.
        
        Args:
            input_ids: Token IDs tensor of shape (batch_size, seq_len)
            attention_mask: Attention mask tensor of shape (batch_size, seq_len)
            labels: Optional labels tensor of shape (batch_size,)
            
        Returns:
            Dictionary containing logits and optionally loss
        """
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Apply pooling strategy
        pooled_output = self._pool_hidden_states(
            transformer_outputs.last_hidden_state,
            attention_mask
        )
        
        # Apply dropout and classification layer
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        outputs = {"logits": logits}
        
        # Calculate loss if labels are provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            outputs["loss"] = loss
        
        return outputs