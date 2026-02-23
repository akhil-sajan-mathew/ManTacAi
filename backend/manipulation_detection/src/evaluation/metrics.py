"""Evaluation metrics for manipulation detection model."""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score
)
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict

from ..utils.config import get_label_mapping

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Comprehensive metrics calculator for manipulation detection."""
    
    def __init__(self, num_classes: int = 11):
        """Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes in the dataset
        """
        self.num_classes = num_classes
        self.label_mapping = get_label_mapping()
        self.class_names = [self.label_mapping[i] for i in range(num_classes)]
    
    def calculate_basic_metrics(self, 
                              y_true: np.ndarray, 
                              y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary containing basic metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def calculate_per_class_metrics(self, 
                                  y_true: np.ndarray, 
                                  y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculate per-class metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary containing per-class metrics
        """
        # Calculate per-class precision, recall, f1
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            per_class_metrics[class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i]),
                'support': int(np.sum(y_true == i))
            }
        
        return per_class_metrics
    
    def calculate_confusion_matrix(self, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Calculate confusion matrix and related metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Tuple of (confusion_matrix, matrix_info)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        # Calculate per-class accuracy from confusion matrix
        per_class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
        per_class_accuracy = np.nan_to_num(per_class_accuracy)
        
        matrix_info = {
            'confusion_matrix': cm,
            'confusion_matrix_normalized': cm_normalized,
            'per_class_accuracy': per_class_accuracy,
            'class_names': self.class_names
        }
        
        return cm, matrix_info
    
    def calculate_classification_report(self, 
                                     y_true: np.ndarray, 
                                     y_pred: np.ndarray) -> Dict[str, Any]:
        """Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report dictionary
        """
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        return report
    
    def calculate_multiclass_auc(self, 
                               y_true: np.ndarray, 
                               y_prob: np.ndarray) -> Dict[str, float]:
        """Calculate AUC metrics for multiclass classification.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary containing AUC metrics
        """
        try:
            # One-vs-rest AUC
            auc_ovr = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            auc_ovr_weighted = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            
            # One-vs-one AUC
            auc_ovo = roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro')
            auc_ovo_weighted = roc_auc_score(y_true, y_prob, multi_class='ovo', average='weighted')
            
            return {
                'auc_ovr_macro': auc_ovr,
                'auc_ovr_weighted': auc_ovr_weighted,
                'auc_ovo_macro': auc_ovo,
                'auc_ovo_weighted': auc_ovo_weighted
            }
        except ValueError as e:
            logger.warning(f"Could not calculate AUC metrics: {e}")
            return {
                'auc_ovr_macro': 0.0,
                'auc_ovr_weighted': 0.0,
                'auc_ovo_macro': 0.0,
                'auc_ovo_weighted': 0.0
            }
    
    def calculate_top_k_accuracy(self, 
                               y_true: np.ndarray, 
                               y_prob: np.ndarray, 
                               k: int = 3) -> float:
        """Calculate top-k accuracy.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            k: Number of top predictions to consider
            
        Returns:
            Top-k accuracy score
        """
        top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
        correct = 0
        
        for i, true_label in enumerate(y_true):
            if true_label in top_k_preds[i]:
                correct += 1
        
        return correct / len(y_true)
    
    def calculate_comprehensive_metrics(self, 
                                     y_true: np.ndarray, 
                                     y_pred: np.ndarray, 
                                     y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['basic_metrics'] = self.calculate_basic_metrics(y_true, y_pred)
        
        # Per-class metrics
        metrics['per_class_metrics'] = self.calculate_per_class_metrics(y_true, y_pred)
        
        # Confusion matrix
        cm, matrix_info = self.calculate_confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix_info'] = matrix_info
        
        # Classification report
        metrics['classification_report'] = self.calculate_classification_report(y_true, y_pred)
        
        # Probability-based metrics
        if y_prob is not None:
            metrics['auc_metrics'] = self.calculate_multiclass_auc(y_true, y_prob)
            metrics['top_3_accuracy'] = self.calculate_top_k_accuracy(y_true, y_prob, k=3)
            metrics['top_5_accuracy'] = self.calculate_top_k_accuracy(y_true, y_prob, k=5)
        
        return metrics


class ManipulationMetrics:
    """Specialized metrics for manipulation detection tasks."""
    
    def __init__(self):
        """Initialize manipulation-specific metrics."""
        self.label_mapping = get_label_mapping()
        self.manipulation_classes = [i for i in range(1, 11)]  # Exclude ethical_persuasion (0)
        self.ethical_class = 0
    
    def calculate_manipulation_detection_metrics(self, 
                                               y_true: np.ndarray, 
                                               y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate binary manipulation vs ethical detection metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary containing manipulation detection metrics
        """
        # Convert to binary: 0 = ethical, 1 = manipulation
        y_true_binary = (y_true != self.ethical_class).astype(int)
        y_pred_binary = (y_pred != self.ethical_class).astype(int)
        
        metrics = {
            'manipulation_accuracy': accuracy_score(y_true_binary, y_pred_binary),
            'manipulation_precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'manipulation_recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'manipulation_f1': f1_score(y_true_binary, y_pred_binary, zero_division=0)
        }
        
        return metrics
    
    def calculate_manipulation_type_accuracy(self, 
                                           y_true: np.ndarray, 
                                           y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate accuracy for each manipulation type.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary containing per-manipulation-type accuracy
        """
        manipulation_accuracy = {}
        
        for class_id in self.manipulation_classes:
            class_name = self.label_mapping[class_id]
            
            # Get samples of this manipulation type
            class_mask = y_true == class_id
            if class_mask.sum() > 0:
                class_accuracy = accuracy_score(
                    y_true[class_mask], 
                    y_pred[class_mask]
                )
                manipulation_accuracy[f"{class_name}_accuracy"] = class_accuracy
        
        return manipulation_accuracy
    
    def calculate_false_positive_analysis(self, 
                                        y_true: np.ndarray, 
                                        y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze false positives (ethical classified as manipulation).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary containing false positive analysis
        """
        # Find ethical samples incorrectly classified as manipulation
        ethical_mask = y_true == self.ethical_class
        ethical_true = y_true[ethical_mask]
        ethical_pred = y_pred[ethical_mask]
        
        # Count false positives by manipulation type
        fp_by_type = defaultdict(int)
        total_ethical = ethical_mask.sum()
        
        for pred_label in ethical_pred:
            if pred_label != self.ethical_class:
                manipulation_type = self.label_mapping[pred_label]
                fp_by_type[manipulation_type] += 1
        
        # Calculate false positive rates
        fp_rates = {}
        for manipulation_type, count in fp_by_type.items():
            fp_rates[f"fp_rate_{manipulation_type}"] = count / total_ethical if total_ethical > 0 else 0.0
        
        analysis = {
            'total_ethical_samples': int(total_ethical),
            'total_false_positives': int(sum(fp_by_type.values())),
            'overall_fp_rate': sum(fp_by_type.values()) / total_ethical if total_ethical > 0 else 0.0,
            'fp_by_manipulation_type': dict(fp_by_type),
            'fp_rates_by_type': fp_rates
        }
        
        return analysis


class ModelEvaluator:
    """Complete model evaluation pipeline."""
    
    def __init__(self, model, device: str = 'cpu'):
        """Initialize model evaluator.
        
        Args:
            model: Trained model to evaluate
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.metrics_calculator = MetricsCalculator()
        self.manipulation_metrics = ManipulationMetrics()
    
    def evaluate_model(self, 
                      data_loader, 
                      return_predictions: bool = True) -> Dict[str, Any]:
        """Evaluate model on a dataset.
        
        Args:
            data_loader: DataLoader containing evaluation data
            return_predictions: Whether to return predictions and probabilities
            
        Returns:
            Dictionary containing evaluation results
        """
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                logits = outputs['logits']
                
                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
                
                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)
        
        # Calculate comprehensive metrics
        results = self.metrics_calculator.calculate_comprehensive_metrics(
            y_true, y_pred, y_prob
        )
        
        # Add manipulation-specific metrics
        results['manipulation_metrics'] = self.manipulation_metrics.calculate_manipulation_detection_metrics(
            y_true, y_pred
        )
        results['manipulation_type_accuracy'] = self.manipulation_metrics.calculate_manipulation_type_accuracy(
            y_true, y_pred
        )
        results['false_positive_analysis'] = self.manipulation_metrics.calculate_false_positive_analysis(
            y_true, y_pred
        )
        
        # Add predictions if requested
        if return_predictions:
            results['predictions'] = {
                'y_true': y_true,
                'y_pred': y_pred,
                'y_prob': y_prob
            }
        
        return results
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """Print a summary of evaluation results.
        
        Args:
            results: Results dictionary from evaluate_model
        """
        basic_metrics = results['basic_metrics']
        manipulation_metrics = results['manipulation_metrics']
        
        print("\n" + "="*50)
        print("MODEL EVALUATION SUMMARY")
        print("="*50)
        
        print(f"\nOverall Performance:")
        print(f"  Accuracy: {basic_metrics['accuracy']:.4f}")
        print(f"  F1-Score (Macro): {basic_metrics['f1_macro']:.4f}")
        print(f"  F1-Score (Weighted): {basic_metrics['f1_weighted']:.4f}")
        
        print(f"\nManipulation Detection:")
        print(f"  Manipulation Accuracy: {manipulation_metrics['manipulation_accuracy']:.4f}")
        print(f"  Manipulation Precision: {manipulation_metrics['manipulation_precision']:.4f}")
        print(f"  Manipulation Recall: {manipulation_metrics['manipulation_recall']:.4f}")
        print(f"  Manipulation F1-Score: {manipulation_metrics['manipulation_f1']:.4f}")
        
        if 'auc_metrics' in results:
            auc_metrics = results['auc_metrics']
            print(f"\nAUC Metrics:")
            print(f"  AUC-ROC (OvR): {auc_metrics['auc_ovr_macro']:.4f}")
            print(f"  AUC-ROC (OvO): {auc_metrics['auc_ovo_macro']:.4f}")
        
        print(f"\nPer-Class Performance:")
        per_class = results['per_class_metrics']
        for class_name, metrics in per_class.items():
            print(f"  {class_name:20s}: F1={metrics['f1_score']:.3f}, "
                  f"Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}")
        
        print("="*50)