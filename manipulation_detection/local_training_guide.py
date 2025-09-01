#!/usr/bin/env python3
"""
Local Training Guide for Manipulation Detection Model

This script provides a complete setup for training the manipulation detection model
on your local machine with real data and proper training infrastructure.
"""

import os
import sys
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from datetime import datetime
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# DATASET CLASS FOR LOCAL TRAINING
# =============================================================================

class ManipulationDataset(Dataset):
    """Dataset class for manipulation detection training."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Initialize dataset.
        
        Args:
            texts: List of text strings
            labels: List of label strings or IDs
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label mapping if labels are strings
        if isinstance(labels[0], str):
            unique_labels = sorted(list(set(labels)))
            self.label_to_id = {label: i for i, label in enumerate(unique_labels)}
            self.id_to_label = {i: label for i, label in enumerate(unique_labels)}
            self.label_ids = [self.label_to_id[label] for label in labels]
        else:
            self.label_ids = labels
            self.label_to_id = None
            self.id_to_label = None
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label_id = self.label_ids[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }

# =============================================================================
# MODEL DEFINITION FOR LOCAL TRAINING
# =============================================================================

class LocalManipulationClassifier(nn.Module):
    """Enhanced manipulation classifier for local training."""
    
    def __init__(self, model_name="distilbert-base-uncased", num_classes=11, dropout_rate=0.1):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
        
        # Initialize weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        
        # Default label mapping (can be updated with real data)
        self.labels = [
            "ethical_persuasion", "gaslighting", "guilt_tripping", "love_bombing",
            "threatening", "stonewalling", "projection", "triangulation", 
            "silent_treatment", "emotional_blackmail", "manipulation_through_pity"
        ]
        
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}
        self.id_to_label = {i: label for i, label in enumerate(self.labels)}
    
    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass with optional loss calculation."""
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits
        }
    
    def predict(self, text, tokenizer, device):
        """Make prediction on single text."""
        self.eval()
        
        encoding = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=-1)
        
        predicted_id = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, predicted_id].item()
        predicted_label = self.id_to_label[predicted_id]
        
        return {
            'text': text,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'is_manipulation': predicted_label != 'ethical_persuasion'
        }

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

class LocalTrainer:
    """Trainer class for local model training."""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Setup scheduler
        total_steps = len(train_loader) * config['epochs']
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=total_steps
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_acc = 0.0
        self.best_model_path = None
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            logits = outputs['logits']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Calculate metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Update progress bar
            current_acc = correct_predictions / total_predictions
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                logits = outputs['logits']
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def train(self):
        """Full training loop."""
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {self.config['epochs']}")
        logger.info(f"Learning rate: {self.config['learning_rate']}")
        
        for epoch in range(self.config['epochs']):
            logger.info(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Log results
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_path = f"best_model_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': self.config
                }, self.best_model_path)
                logger.info(f"New best model saved: {self.best_model_path}")
        
        logger.info(f"\nTraining completed! Best validation accuracy: {self.best_val_acc:.4f}")
        return self.history

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_data_from_csv(csv_path):
    """Load data from CSV file."""
    df = pd.read_csv(csv_path)
    
    # Assume columns are 'text' and 'label'
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must have 'text' and 'label' columns")
    
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    logger.info(f"Loaded {len(texts)} examples from {csv_path}")
    logger.info(f"Unique labels: {sorted(set(labels))}")
    
    return texts, labels

def load_data_from_json(json_path):
    """Load data from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Assume format: [{"text": "...", "label": "..."}, ...]
    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]
    
    logger.info(f"Loaded {len(texts)} examples from {json_path}")
    logger.info(f"Unique labels: {sorted(set(labels))}")
    
    return texts, labels

def create_synthetic_data(samples_per_class=100):
    """Create synthetic training data for testing."""
    logger.info("Creating synthetic training data...")
    
    # Enhanced synthetic examples
    synthetic_data = {
        "ethical_persuasion": [
            "I think this approach would benefit both of us.",
            "Let me explain why this solution makes sense.",
            "Consider the advantages of this option.",
            "This seems like a reasonable compromise.",
            "I believe we can find common ground here."
        ],
        "gaslighting": [
            "You're being too sensitive about this.",
            "That never actually happened the way you remember.",
            "You're imagining things again.",
            "You're overreacting to the situation.",
            "You're remembering it completely wrong."
        ],
        "guilt_tripping": [
            "If you really cared, you would do this.",
            "I guess my feelings don't matter to you.",
            "After everything I've done for you.",
            "You're being selfish by not helping.",
            "I'm disappointed in your decision."
        ],
        "love_bombing": [
            "You're absolutely perfect in every way.",
            "I've never met anyone as amazing as you.",
            "You're the most wonderful person alive.",
            "I worship everything about you.",
            "You're flawless and incredible."
        ],
        "threatening": [
            "You'll regret this decision later.",
            "Things won't go well if you refuse.",
            "There will be consequences for this.",
            "You don't want to see what happens next.",
            "You're making a dangerous mistake."
        ],
        "stonewalling": [
            "I'm not discussing this anymore.",
            "This conversation is over.",
            "I refuse to talk about this.",
            "There's nothing more to say.",
            "I won't engage with you on this."
        ],
        "projection": [
            "You're the one with the problem.",
            "You're being controlling, not me.",
            "You're the manipulative one here.",
            "You always start the arguments.",
            "You're projecting your issues onto me."
        ],
        "triangulation": [
            "Everyone else agrees with me.",
            "Your friends think you're wrong too.",
            "Even your family is concerned about you.",
            "Other people have noticed your behavior.",
            "Everyone can see you're the problem."
        ],
        "silent_treatment": [
            "I'm not speaking to you anymore.",
            "Don't expect me to talk to you.",
            "You can talk when you apologize.",
            "I have nothing to say to you.",
            "I'm done communicating with you."
        ],
        "emotional_blackmail": [
            "You're making me want to hurt myself.",
            "If you leave, I don't know what I'll do.",
            "You're causing me so much pain.",
            "I can't handle this stress you're giving me.",
            "You're going to make me have a breakdown."
        ],
        "manipulation_through_pity": [
            "I'm having such a hard time right now.",
            "Nobody else cares about me like you do.",
            "I'm so lonely without your help.",
            "You're all I have left in this world.",
            "I'm struggling and need your support."
        ]
    }
    
    # Generate more examples by variation
    texts = []
    labels = []
    
    for label, base_examples in synthetic_data.items():
        for _ in range(samples_per_class):
            # Use base examples with some variation
            example = np.random.choice(base_examples)
            texts.append(example)
            labels.append(label)
    
    logger.info(f"Created {len(texts)} synthetic examples")
    return texts, labels

# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Manipulation Detection Model Locally')
    parser.add_argument('--data-path', type=str, help='Path to training data (CSV or JSON)')
    parser.add_argument('--model-name', type=str, default='distilbert-base-uncased', 
                       help='Pre-trained model name')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--output-dir', type=str, default='./models', help='Output directory')
    parser.add_argument('--use-synthetic', action='store_true', 
                       help='Use synthetic data for training')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    if args.use_synthetic or args.data_path is None:
        texts, labels = create_synthetic_data(samples_per_class=200)
    else:
        if args.data_path.endswith('.csv'):
            texts, labels = load_data_from_csv(args.data_path)
        elif args.data_path.endswith('.json'):
            texts, labels = load_data_from_json(args.data_path)
        else:
            raise ValueError("Data file must be CSV or JSON")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info(f"Training samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(val_texts)}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = LocalManipulationClassifier(
        model_name=args.model_name,
        num_classes=len(set(labels))
    )
    model.to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets and data loaders
    train_dataset = ManipulationDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = ManipulationDataset(val_texts, val_labels, tokenizer, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Training configuration
    config = {
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': 0.01,
        'warmup_steps': len(train_loader) // 4,
        'batch_size': args.batch_size,
        'max_length': args.max_length,
        'model_name': args.model_name
    }
    
    # Create trainer and train
    trainer = LocalTrainer(model, train_loader, val_loader, device, config)
    history = trainer.train()
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_name': args.model_name,
        'config': config,
        'history': history,
        'label_mapping': train_dataset.label_to_id
    }, final_model_path)
    
    logger.info(f"Final model saved to: {final_model_path}")
    
    # Plot training curves
    plot_training_curves(history, args.output_dir)
    
    # Test the model
    test_model(model, tokenizer, device)
    
    logger.info("Training completed successfully!")

def plot_training_curves(history, output_dir):
    """Plot training curves."""
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.show()

def test_model(model, tokenizer, device):
    """Test the trained model."""
    test_texts = [
        "I think this is a reasonable approach.",
        "You're being too sensitive about this.",
        "If you really loved me, you would understand.",
        "Everyone thinks you're wrong about this.",
        "I'm not talking to you until you apologize."
    ]
    
    logger.info("\nTesting trained model:")
    logger.info("=" * 50)
    
    for text in test_texts:
        result = model.predict(text, tokenizer, device)
        logger.info(f"Text: {text}")
        logger.info(f"Prediction: {result['predicted_label']} (confidence: {result['confidence']:.3f})")
        logger.info("-" * 50)

if __name__ == "__main__":
    main()