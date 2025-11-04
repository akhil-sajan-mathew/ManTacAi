"""
Kaggle GPU Training for Enhanced Critical Splits Dataset

This script is optimized for training on Kaggle with GPU acceleration
using the enhanced_critical_splits_augmented.json dataset.

Copy this entire code into a Kaggle notebook cell and run it.
"""

import warnings

warnings.filterwarnings("ignore")

import os
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from datetime import datetime
from collections import Counter
import gc

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

print("🚀 Kaggle Enhanced Splits Trainer")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )

# =============================================================================
# DATASET CLASS FOR ENHANCED SPLITS
# =============================================================================


class EnhancedSplitsDataset(Dataset):
    """Dataset class for enhanced_critical_splits_augmented.json format."""

    def __init__(self, data_items, tokenizer, max_length=512):
        self.data_items = data_items
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Extract texts and labels
        self.texts = [item["text"] for item in data_items]
        self.labels = [item["manipulation_tactic"] for item in data_items]

        # Create label mapping
        unique_labels = sorted(list(set(self.labels)))
        self.label_to_id = {label: i for i, label in enumerate(unique_labels)}
        self.id_to_label = {i: label for i, label in enumerate(unique_labels)}
        self.label_ids = [self.label_to_id[label] for label in self.labels]

        print(f"✓ Dataset created with {len(self.texts)} samples")
        print(f"✓ Classes: {len(unique_labels)}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label_id = self.label_ids[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label_id, dtype=torch.long),
        }


# =============================================================================
# MODEL DEFINITION
# =============================================================================


class KaggleManipulationClassifier(nn.Module):
    """Optimized manipulation classifier for Kaggle GPU training."""

    def __init__(
        self, model_name="distilbert-base-uncased", num_classes=11, dropout_rate=0.1
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        # Load pre-trained transformer
        self.transformer = AutoModel.from_pretrained(model_name)

        # Classification head with layer normalization (safer than batch norm)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(self.transformer.config.hidden_size)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)

        # Initialize weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass with optional loss calculation."""
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


# =============================================================================
# KAGGLE OPTIMIZED TRAINER
# =============================================================================


class KaggleTrainer:
    """Trainer optimized for Kaggle environment."""

    def __init__(
        self, model, train_loader, val_loader, device, config, class_weights=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        # Setup optimizer with better parameters for Kaggle
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            eps=1e-8,
        )

        # Setup scheduler
        total_steps = len(train_loader) * config["epochs"]
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config["warmup_steps"],
            num_training_steps=total_steps,
        )

        # Setup loss function with optional class weights
        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
            print("✓ Using weighted loss function for imbalanced classes")
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rates": [],
        }

        self.best_val_acc = 0.0
        self.patience_counter = 0

    def train_epoch(self, epoch):
        """Train for one epoch with memory optimization."""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask, labels=None)
            logits = outputs["logits"]

            # Use the loss function (which may include class weights)
            loss = self.loss_fn(logits, labels)

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
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{current_acc:.4f}",
                    "lr": f"{current_lr:.2e}",
                }
            )

            # Memory cleanup every 50 batches
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_predictions

        return avg_loss, accuracy

    def validate(self):
        """Validate the model with memory optimization."""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask, labels=None)
                logits = outputs["logits"]
                loss = self.loss_fn(logits, labels)

                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

                # Store for detailed metrics
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_predictions

        return avg_loss, accuracy, all_predictions, all_labels

    def train(self):
        """Full training loop with early stopping."""
        print(f"\n🎯 Starting training on {self.device}")
        print(f"📊 Training samples: {len(self.train_loader.dataset)}")
        print(f"📊 Validation samples: {len(self.val_loader.dataset)}")
        print(f"⚙️ Epochs: {self.config['epochs']}")
        print(f"⚙️ Batch size: {self.config['batch_size']}")
        print(f"⚙️ Learning rate: {self.config['learning_rate']}")

        for epoch in range(self.config["epochs"]):
            print(f"\n📈 Epoch {epoch + 1}/{self.config['epochs']}")

            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate()

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rates"].append(self.scheduler.get_last_lr()[0])

            # Print results
            print(f"📊 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"📊 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Check for improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0

                # Save best model
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_acc": val_acc,
                        "config": self.config,
                        "label_mapping": self.train_loader.dataset.label_to_id,
                        "id_to_label": self.train_loader.dataset.id_to_label,
                    },
                    "best_kaggle_model.pt",
                )

                print(f"💾 New best model saved! Validation accuracy: {val_acc:.4f}")

                # Print detailed classification report for best model
                if epoch > 0:  # Skip first epoch
                    self.print_classification_report(val_labels, val_preds)
            else:
                self.patience_counter += 1
                print(
                    f"⏳ No improvement. Patience: {self.patience_counter}/{self.config['patience']}"
                )

            # Early stopping
            if self.patience_counter >= self.config["patience"]:
                print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                break

            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()

        print(
            f"\n🎉 Training completed! Best validation accuracy: {self.best_val_acc:.4f}"
        )
        return self.history

    def print_classification_report(self, true_labels, predictions):
        """Print comprehensive classification metrics with confusion matrix."""
        label_names = list(self.train_loader.dataset.id_to_label.values())
        
        print("\n" + "="*80)
        print("� COMPREHENSIVE CLASSIFICATION METRICS")
        print("="*80)
        
        # Calculate all metrics
        accuracy = accuracy_score(true_labels, predictions)
        balanced_acc = balanced_accuracy_score(true_labels, predictions)
        mcc = matthews_corrcoef(true_labels, predictions)
        kappa = cohen_kappa_score(true_labels, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, labels=range(len(label_names)), zero_division=0
        )
        
        # Macro and weighted averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        weighted_precision = np.average(precision, weights=support)
        weighted_recall = np.average(recall, weights=support)
        weighted_f1 = np.average(f1, weights=support)
        
        # Print overall metrics
        print(f"\n{'OVERALL METRICS':<40s}")
        print("-" * 80)
        print(f"{'Accuracy:':<40s} {accuracy:>10.4f}")
        print(f"{'Balanced Accuracy:':<40s} {balanced_acc:>10.4f}")
        print(f"{'Matthews Correlation Coefficient:':<40s} {mcc:>10.4f}")
        print(f"{'Cohen Kappa Score:':<40s} {kappa:>10.4f}")
        print(f"\n{'Macro Precision:':<40s} {macro_precision:>10.4f}")
        print(f"{'Macro Recall:':<40s} {macro_recall:>10.4f}")
        print(f"{'Macro F1-Score:':<40s} {macro_f1:>10.4f}")
        print(f"\n{'Weighted Precision:':<40s} {weighted_precision:>10.4f}")
        print(f"{'Weighted Recall:':<40s} {weighted_recall:>10.4f}")
        print(f"{'Weighted F1-Score:':<40s} {weighted_f1:>10.4f}")
        
        # Print per-class metrics
        print(f"\n{'PER-CLASS METRICS':<40s}")
        print("="*80)
        print(f"{'Class':<30s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
        print("-" * 80)
        
        for i, class_name in enumerate(label_names):
            print(f"{class_name:<30s} "
                  f"{precision[i]:>10.4f} "
                  f"{recall[i]:>10.4f} "
                  f"{f1[i]:>10.4f} "
                  f"{support[i]:>10.0f}")
        
        print("-" * 80)
        print(f"{'MACRO AVERAGE':<30s} "
              f"{macro_precision:>10.4f} "
              f"{macro_recall:>10.4f} "
              f"{macro_f1:>10.4f} "
              f"{sum(support):>10.0f}")
        print(f"{'WEIGHTED AVERAGE':<30s} "
              f"{weighted_precision:>10.4f} "
              f"{weighted_recall:>10.4f} "
              f"{weighted_f1:>10.4f} "
              f"{sum(support):>10.0f}")
        
        # Print confusion matrix
        self.print_confusion_matrix(true_labels, predictions, label_names)
    
    def print_confusion_matrix(self, true_labels, predictions, label_names):
        """Print confusion matrix in text format."""
        cm = confusion_matrix(true_labels, predictions)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        print("\n" + "="*120)
        print("CONFUSION MATRIX (Counts)")
        print("="*120)
        
        # Header
        header_label = "True \\ Predicted"
        print(f"\n{header_label:<25s}", end='')
        for name in label_names:
            print(f"{name[:10]:>11s}", end='')
        print()
        print("-" * 120)
        
        # Rows
        for i, true_name in enumerate(label_names):
            print(f"{true_name:<25s}", end='')
            for j in range(len(label_names)):
                print(f"{cm[i][j]:>11d}", end='')
            print()
        
        print("\n" + "="*120)
        print("CONFUSION MATRIX (Normalized by True Label - Recall per class)")
        print("="*120)
        
        # Header
        header_label = "True \\ Predicted"
        print(f"\n{header_label:<25s}", end='')
        for name in label_names:
            print(f"{name[:10]:>11s}", end='')
        print()
        print("-" * 120)
        
        # Rows
        for i, true_name in enumerate(label_names):
            print(f"{true_name:<25s}", end='')
            for j in range(len(label_names)):
                print(f"{cm_normalized[i][j]:>11.3f}", end='')
            print()
        
        # Plot confusion matrix heatmap
        try:
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            
            # Raw counts
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=label_names, yticklabels=label_names,
                        ax=axes[0], cbar_kws={'label': 'Count'})
            axes[0].set_xlabel('Predicted Label')
            axes[0].set_ylabel('True Label')
            axes[0].set_title('Confusion Matrix (Counts)')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].tick_params(axis='y', rotation=0)
            
            # Normalized
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=label_names, yticklabels=label_names,
                        ax=axes[1], cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
            axes[1].set_xlabel('Predicted Label')
            axes[1].set_ylabel('True Label')
            axes[1].set_title('Confusion Matrix (Normalized by True Label)')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].tick_params(axis='y', rotation=0)
            
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            print("\n✓ Confusion matrix visualization saved to: confusion_matrix.png")
            plt.show()
        except Exception as e:
            print(f"\n⚠️ Could not generate confusion matrix plot: {e}")


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================


def load_enhanced_splits_data(
    json_path="/kaggle/input/psychological-manipulation-detection-dataset/enhanced_critical_splits_augmented.json",
):
    """Load data from enhanced_critical_splits_augmented.json on Kaggle."""
    print(f"📂 Loading data from: {json_path}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {json_path}")
        print("💡 Make sure you've uploaded the dataset to Kaggle and updated the path")
        return None, None, None

    # Extract train, validation, and test sets
    train_data = data["train"]
    val_data = data["validation"]
    test_data = data["test"]

    print(f"✅ Loaded {len(train_data)} training samples")
    print(f"✅ Loaded {len(val_data)} validation samples")
    print(f"✅ Loaded {len(test_data)} test samples")

    # Print class distribution
    train_labels = [item["manipulation_tactic"] for item in train_data]
    label_counts = Counter(train_labels)

    print("\n📊 Training set class distribution:")
    for label, count in sorted(label_counts.items()):
        percentage = count / len(train_data) * 100
        print(f"  {label:25s}: {count:4d} ({percentage:5.1f}%)")

    return train_data, val_data, test_data


def calculate_class_weights(train_data):
    """Calculate class weights for imbalanced dataset."""
    train_labels = [item["manipulation_tactic"] for item in train_data]
    unique_classes = sorted(list(set(train_labels)))
    label_counts = Counter(train_labels)

    total_samples = len(train_labels)
    num_classes = len(unique_classes)

    # Calculate weights using inverse frequency
    weights_dict = {}
    for class_name in unique_classes:
        weight = total_samples / (num_classes * label_counts[class_name])
        weights_dict[class_name] = weight

    print("\n⚖️ Class weights for balancing:")
    for class_name in unique_classes:
        print(f"  {class_name:25s}: {weights_dict[class_name]:8.4f}")

    return weights_dict, unique_classes


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def plot_training_curves(history):
    """Plot training curves optimized for Kaggle."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss plot
    axes[0, 0].plot(history["train_loss"], label="Train Loss", color="blue")
    axes[0, 0].plot(history["val_loss"], label="Validation Loss", color="red")
    axes[0, 0].set_title("Training and Validation Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[0, 1].plot(history["train_acc"], label="Train Accuracy", color="blue")
    axes[0, 1].plot(history["val_acc"], label="Validation Accuracy", color="red")
    axes[0, 1].set_title("Training and Validation Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Learning rate plot
    axes[1, 0].plot(history["learning_rates"], color="green")
    axes[1, 0].set_title("Learning Rate Schedule")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True, alpha=0.3)

    # Validation accuracy zoom
    axes[1, 1].plot(history["val_acc"], color="red", linewidth=2)
    axes[1, 1].set_title("Validation Accuracy (Detailed)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Validation Accuracy")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print final metrics
    print(f"\n📈 Final Training Metrics:")
    print(f"  Best Validation Accuracy: {max(history['val_acc']):.4f}")
    print(f"  Final Training Accuracy: {history['train_acc'][-1]:.4f}")
    print(f"  Final Validation Loss: {history['val_loss'][-1]:.4f}")


def test_model_predictions(model, tokenizer, device, label_mapping):
    """Test the trained model with sample predictions."""
    print("\n🧪 Testing model predictions:")
    print("=" * 50)

    test_texts = [
        "I think this is a reasonable approach to solve our problem.",
        "You're being way too sensitive about this whole situation.",
        "If you really cared about me, you would understand my position.",
        "Everyone else agrees with me that you're being unreasonable.",
        "I'm not going to discuss this topic with you anymore.",
        "You're absolutely perfect and I love everything about you.",
        "You'll regret it if you don't listen to what I'm saying.",
        "Why do you always have to bring up irrelevant things?",
        "You're just like your mother, always complaining about everything.",
        "I'm having such a hard time and really need your help.",
    ]

    model.eval()
    id_to_label = {v: k for k, v in label_mapping.items()}

    for i, text in enumerate(test_texts, 1):
        encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            logits = outputs["logits"]
            probabilities = torch.softmax(logits, dim=-1)
            predicted_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_id].item()
            predicted_label = id_to_label[predicted_id]

        print(f"{i:2d}. Text: '{text[:60]}...'")
        print(f"    Prediction: {predicted_label.replace('_', ' ').title()}")
        print(f"    Confidence: {confidence:.3f}")
        print(
            f"    Is Manipulation: {'Yes' if predicted_label != 'ethical_persuasion' else 'No'}"
        )
        print("-" * 50)


def evaluate_on_test_set(model, test_data, tokenizer, device, label_mapping, max_length=128):
    """Comprehensive evaluation on test set with detailed metrics."""
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE TEST SET EVALUATION")
    print("="*80)
    
    # Create test dataset and loader
    test_dataset = EnhancedSplitsDataset(test_data, tokenizer, max_length)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print(f"\n[Running inference on {len(test_data)} test samples...]")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)
    
    print("✓ Inference complete\n")
    
    # Get label names
    id_to_label = {v: k for k, v in label_mapping.items()}
    label_names = [id_to_label[i] for i in range(len(label_mapping))]
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(label_names)), zero_division=0
    )
    
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)
    
    # Confidence statistics
    max_probs = np.max(y_probs, axis=1)
    avg_confidence = np.mean(max_probs)
    correct_mask = (y_pred == y_true)
    avg_confidence_correct = np.mean(max_probs[correct_mask])
    avg_confidence_incorrect = np.mean(max_probs[~correct_mask]) if np.any(~correct_mask) else 0
    
    # Print overall metrics
    print("="*80)
    print("OVERALL TEST SET METRICS")
    print("="*80)
    print(f"\n{'Metric':<40s} {'Value':>10s}")
    print("-" * 52)
    print(f"{'Accuracy:':<40s} {accuracy:>10.4f}")
    print(f"{'Balanced Accuracy:':<40s} {balanced_acc:>10.4f}")
    print(f"{'Matthews Correlation Coefficient:':<40s} {mcc:>10.4f}")
    print(f"{'Cohen Kappa Score:':<40s} {kappa:>10.4f}")
    print(f"\n{'Macro Precision:':<40s} {macro_precision:>10.4f}")
    print(f"{'Macro Recall:':<40s} {macro_recall:>10.4f}")
    print(f"{'Macro F1-Score:':<40s} {macro_f1:>10.4f}")
    print(f"\n{'Weighted Precision:':<40s} {weighted_precision:>10.4f}")
    print(f"{'Weighted Recall:':<40s} {weighted_recall:>10.4f}")
    print(f"{'Weighted F1-Score:':<40s} {weighted_f1:>10.4f}")
    print(f"\n{'Average Confidence:':<40s} {avg_confidence:>10.4f}")
    print(f"{'Avg Confidence (Correct):':<40s} {avg_confidence_correct:>10.4f}")
    print(f"{'Avg Confidence (Incorrect):':<40s} {avg_confidence_incorrect:>10.4f}")
    
    # Print per-class metrics
    print("\n" + "="*80)
    print("PER-CLASS TEST SET METRICS")
    print("="*80)
    print(f"\n{'Class':<30s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
    print("-" * 80)
    
    for i, class_name in enumerate(label_names):
        print(f"{class_name:<30s} "
              f"{precision[i]:>10.4f} "
              f"{recall[i]:>10.4f} "
              f"{f1[i]:>10.4f} "
              f"{support[i]:>10.0f}")
    
    print("-" * 80)
    print(f"{'MACRO AVERAGE':<30s} "
          f"{macro_precision:>10.4f} "
          f"{macro_recall:>10.4f} "
          f"{macro_f1:>10.4f} "
          f"{sum(support):>10.0f}")
    print(f"{'WEIGHTED AVERAGE':<30s} "
          f"{weighted_precision:>10.4f} "
          f"{weighted_recall:>10.4f} "
          f"{weighted_f1:>10.4f} "
          f"{sum(support):>10.0f}")
    
    # Print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    print("\n" + "="*120)
    print("TEST SET CONFUSION MATRIX (Counts)")
    print("="*120)
    
    # Header
    header_label = "True \\ Predicted"
    print(f"\n{header_label:<25s}", end='')
    for name in label_names:
        print(f"{name[:10]:>11s}", end='')
    print()
    print("-" * 120)
    
    # Rows
    for i, true_name in enumerate(label_names):
        print(f"{true_name:<25s}", end='')
        for j in range(len(label_names)):
            print(f"{cm[i][j]:>11d}", end='')
        print()
    
    print("\n" + "="*120)
    print("TEST SET CONFUSION MATRIX (Normalized - Recall)")
    print("="*120)
    
    # Header
    header_label = "True \\ Predicted"
    print(f"\n{header_label:<25s}", end='')
    for name in label_names:
        print(f"{name[:10]:>11s}", end='')
    print()
    print("-" * 120)
    
    # Rows
    for i, true_name in enumerate(label_names):
        print(f"{true_name:<25s}", end='')
        for j in range(len(label_names)):
            print(f"{cm_normalized[i][j]:>11.3f}", end='')
        print()
    
    # Plot confusion matrix
    try:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=label_names, yticklabels=label_names,
                    ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_xlabel('Predicted Label')
        axes[0].set_ylabel('True Label')
        axes[0].set_title('Test Set Confusion Matrix (Counts)')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].tick_params(axis='y', rotation=0)
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=label_names, yticklabels=label_names,
                    ax=axes[1], cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
        axes[1].set_xlabel('Predicted Label')
        axes[1].set_ylabel('True Label')
        axes[1].set_title('Test Set Confusion Matrix (Normalized)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        plt.savefig('test_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n✓ Test confusion matrix saved to: test_confusion_matrix.png")
        plt.show()
    except Exception as e:
        print(f"\n⚠️ Could not generate confusion matrix plot: {e}")
    
    # Analyze misclassifications
    misclassified = []
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if true_label != pred_label:
            misclassified.append({
                'text': test_data[i]['text'],
                'true_label': label_names[true_label],
                'pred_label': label_names[pred_label],
                'confidence': float(y_probs[i][pred_label]),
                'true_prob': float(y_probs[i][true_label])
            })
    
    misclass_pairs = Counter([
        (label_names[t], label_names[p]) 
        for t, p in zip(y_true, y_pred) 
        if t != p
    ])
    
    print("\n" + "="*80)
    print(f"MISCLASSIFICATION ANALYSIS ({len(misclassified)} errors / {len(y_true)} total)")
    print(f"Error Rate: {len(misclassified)/len(y_true)*100:.2f}%")
    print("="*80)
    
    print("\nTop 10 Most Common Misclassification Patterns:")
    print("-" * 80)
    for (true_label, pred_label), count in misclass_pairs.most_common(10):
        pct = count / len(y_true) * 100
        print(f"  {true_label:<25s} → {pred_label:<25s}: {count:>4d} ({pct:>5.2f}%)")
    
    # Show examples
    print("\nSample Misclassifications (Top 10 by confidence):")
    print("-" * 80)
    misclassified_sorted = sorted(misclassified, key=lambda x: x['confidence'], reverse=True)
    
    for i, error in enumerate(misclassified_sorted[:10], 1):
        print(f"\n{i}. Text: \"{error['text'][:80]}{'...' if len(error['text']) > 80 else ''}\"")
        print(f"   True: {error['true_label']:<25s} (prob: {error['true_prob']:.3f})")
        print(f"   Pred: {error['pred_label']:<25s} (conf: {error['confidence']:.3f})")
    
    # Save detailed metrics
    test_metrics = {
        'overall_metrics': {
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_acc),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'mcc': float(mcc),
            'kappa': float(kappa)
        },
        'per_class_metrics': {
            label_names[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
            for i in range(len(label_names))
        },
        'confusion_matrix': cm.tolist(),
        'total_test_samples': len(y_true),
        'total_errors': len(misclassified),
        'error_rate': float(len(misclassified) / len(y_true))
    }
    
    with open('test_set_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    
    print("\n✓ Detailed test metrics saved to: test_set_metrics.json")
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print(f"  • Test Accuracy: {accuracy:.2%}")
    print(f"  • Macro F1-Score: {macro_f1:.4f}")
    print(f"  • Weighted F1-Score: {weighted_f1:.4f}")
    print(f"  • Error Rate: {len(misclassified)/len(y_true)*100:.2f}%")
    
    # Identify problematic classes
    worst_f1_idx = np.argmin(f1)
    best_f1_idx = np.argmax(f1)
    
    print(f"\n  • Best Performing: {label_names[best_f1_idx]} (F1: {f1[best_f1_idx]:.4f})")
    print(f"  • Worst Performing: {label_names[worst_f1_idx]} (F1: {f1[worst_f1_idx]:.4f})")
    
    return test_metrics


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================


def train_kaggle_model(use_class_weights=True):
    """Main function to train model on Kaggle."""

    # Configuration optimized for Kaggle GPU
    config = {
        "model_name": "distilbert-base-uncased",  # Fast and efficient
        "epochs": 15,
        "batch_size": 32,  # Optimized for Kaggle GPU
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "max_length": 512,
        "warmup_steps": 100,
        "patience": 5,  # Early stopping patience
    }

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    if torch.cuda.is_available():
        print(
            f"🚀 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    try:
        # Load data from the psychological manipulation detection dataset
        train_data, val_data, test_data = load_enhanced_splits_data(
            "/kaggle/input/psychological-manipulation-detection-dataset/enhanced_critical_splits_augmented.json"
        )

        if train_data is None:
            print("❌ Failed to load data. Please check the file path.")
            return False

        # Load tokenizer
        print("🔤 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

        # Create datasets
        print("📦 Creating datasets...")
        train_dataset = EnhancedSplitsDataset(
            train_data, tokenizer, config["max_length"]
        )
        val_dataset = EnhancedSplitsDataset(val_data, tokenizer, config["max_length"])

        # Update config with actual number of classes
        config["num_classes"] = len(train_dataset.label_to_id)
        config["warmup_steps"] = len(train_dataset) // config["batch_size"] // 4

        print(f"📊 Number of classes: {config['num_classes']}")
        print(f"⚙️ Warmup steps: {config['warmup_steps']}")

        # Calculate sample weights for WeightedRandomSampler
        print("\n🎯 Setting up WeightedRandomSampler for balanced batches...")

        # Count samples per class
        class_counts = Counter([item["manipulation_tactic"] for item in train_data])
        print(f"📊 Class distribution in training set:")
        for cls, count in sorted(
            class_counts.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"   {cls:30s}: {count:4d} samples")

        # Calculate class weights (inverse frequency)
        total_samples = len(train_data)
        class_weights_dict = {
            cls: total_samples / count for cls, count in class_counts.items()
        }

        # Create sample weights for each training example
        sample_weights = [
            class_weights_dict[item["manipulation_tactic"]] for item in train_data
        ]
        sample_weights = torch.tensor(sample_weights, dtype=torch.float64)

        # Create WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        print(f"✓ WeightedRandomSampler configured")
        print(f"   This ensures each batch has balanced class representation")
        print(f"   Minority classes will be oversampled during training")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            sampler=sampler,  # Use sampler instead of shuffle
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True,  # Drop last incomplete batch to avoid BatchNorm issues
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=False,  # Keep all validation samples
        )

        # Calculate class weights if requested
        class_weights_tensor = None
        if use_class_weights:
            weights_dict, unique_classes = calculate_class_weights(train_data)
            class_weights_list = [
                weights_dict[class_name]
                for class_name in sorted(train_dataset.label_to_id.keys())
            ]
            class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float32)

        # Create model
        print("🧠 Creating model...")
        model = KaggleManipulationClassifier(
            model_name=config["model_name"], num_classes=config["num_classes"]
        )
        model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")

        # Create trainer and train
        trainer = KaggleTrainer(
            model,
            train_loader,
            val_loader,
            device,
            config,
            class_weights=class_weights_tensor,
        )
        history = trainer.train()

        # Plot results
        plot_training_curves(history)

        # Test predictions (demo samples)
        test_model_predictions(model, tokenizer, device, train_dataset.label_to_id)
        
        # Comprehensive test set evaluation
        print("\n" + "="*80)
        print("📊 Running comprehensive evaluation on test set...")
        print("="*80)
        test_metrics = evaluate_on_test_set(
            model, 
            test_data, 
            tokenizer, 
            device, 
            train_dataset.label_to_id
        )

        # Save final model with metadata
        final_model_data = {
            "model_state_dict": model.state_dict(),
            "tokenizer_name": config["model_name"],
            "config": config,
            "history": history,
            "test_metrics": test_metrics,  # Added comprehensive test metrics
            "label_mapping": train_dataset.label_to_id,
            "id_to_label": train_dataset.id_to_label,
            "best_val_acc": trainer.best_val_acc,
            "training_date": datetime.now().isoformat(),
        }

        torch.save(final_model_data, "final_kaggle_manipulation_model.pt")
        print("💾 Final model saved as 'final_kaggle_manipulation_model.pt'")

        print(f"\n🎉 Training completed successfully!")
        print(f"🏆 Best validation accuracy: {trainer.best_val_acc:.4f}")

        return True

    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


# =============================================================================
# RUN TRAINING
# =============================================================================

if __name__ == "__main__":
    print("🚀 Starting Kaggle Enhanced Splits Training")
    print("Make sure you have:")
    print("1. ✅ GPU enabled in Kaggle notebook settings")
    print("2. ✅ enhanced_critical_splits_augmented.json uploaded as dataset")
    print("3. ✅ Updated the file path in load_enhanced_splits_data()")
    print("\nStarting in 3 seconds...")

    import time

    time.sleep(3)

    success = train_kaggle_model(use_class_weights=True)

    if success:
        print("\n🎉 All done! Your model is ready for use.")
    else:
        print("\n❌ Training failed. Check the error messages above.")
