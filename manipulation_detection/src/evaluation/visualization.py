"""Visualization utilities for model evaluation."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import os
from pathlib import Path
import logging

from ..utils.config import get_label_mapping

logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


class EvaluationVisualizer:
    """Creates visualizations for model evaluation results."""
    
    def __init__(self, save_dir: str = "plots", figsize: Tuple[int, int] = (10, 8)):
        """Initialize the visualizer.
        
        Args:
            save_dir: Directory to save plots
            figsize: Default figure size
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.label_mapping = get_label_mapping()
        self.class_names = [self.label_mapping[i] for i in range(len(self.label_mapping))]
    
    def plot_confusion_matrix(self, 
                            confusion_matrix: np.ndarray,
                            class_names: Optional[List[str]] = None,
                            normalize: bool = True,
                            title: str = "Confusion Matrix",
                            save_name: str = "confusion_matrix.png") -> plt.Figure:
        """Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names
            normalize: Whether to normalize the matrix
            title: Plot title
            save_name: Filename to save the plot
            
        Returns:
            Matplotlib figure
        """
        if class_names is None:
            class_names = self.class_names
        
        # Normalize if requested
        if normalize:
            cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)
            fmt = '.2f'
        else:
            cm = confusion_matrix
            fmt = 'd'
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Normalized Count' if normalize else 'Count'})
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix plot: {save_path}")
        
        return fig
    
    def plot_per_class_metrics(self, 
                             per_class_metrics: Dict[str, Dict[str, float]],
                             metrics: List[str] = ['precision', 'recall', 'f1_score'],
                             title: str = "Per-Class Performance",
                             save_name: str = "per_class_metrics.png") -> plt.Figure:
        """Plot per-class performance metrics.
        
        Args:
            per_class_metrics: Dictionary of per-class metrics
            metrics: List of metrics to plot
            title: Plot title
            save_name: Filename to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Prepare data
        classes = list(per_class_metrics.keys())
        data = {metric: [per_class_metrics[cls][metric] for cls in classes] 
                for metric in metrics}
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Set positions for bars
        x = np.arange(len(classes))
        width = 0.25
        
        # Plot bars for each metric
        for i, metric in enumerate(metrics):
            offset = (i - len(metrics)/2 + 0.5) * width
            bars = ax.bar(x + offset, data[metric], width, 
                         label=metric.replace('_', ' ').title(), alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Classes', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved per-class metrics plot: {save_path}")
        
        return fig
    
    def plot_training_history(self, 
                            training_history: Dict[str, List[float]],
                            title: str = "Training History",
                            save_name: str = "training_history.png") -> plt.Figure:
        """Plot training history curves.
        
        Args:
            training_history: Dictionary containing training metrics over epochs
            title: Plot title
            save_name: Filename to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        epochs = range(1, len(training_history['train_loss']) + 1)
        
        # Plot 1: Loss curves
        axes[0].plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0].plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_title('Loss Curves', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Accuracy curve
        axes[1].plot(epochs, training_history['val_accuracy'], 'g-', label='Validation Accuracy', linewidth=2)
        axes[1].set_title('Validation Accuracy', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Learning rate
        axes[2].plot(epochs, training_history['learning_rate'], 'm-', label='Learning Rate', linewidth=2)
        axes[2].set_title('Learning Rate Schedule', fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_yscale('log')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Loss difference (overfitting indicator)
        loss_diff = np.array(training_history['val_loss']) - np.array(training_history['train_loss'])
        axes[3].plot(epochs, loss_diff, 'orange', label='Val Loss - Train Loss', linewidth=2)
        axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[3].set_title('Overfitting Indicator', fontweight='bold')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Loss Difference')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history plot: {save_path}")
        
        return fig
    
    def plot_class_distribution(self, 
                              y_true: np.ndarray,
                              title: str = "Class Distribution",
                              save_name: str = "class_distribution.png") -> plt.Figure:
        """Plot class distribution in the dataset.
        
        Args:
            y_true: True labels array
            title: Plot title
            save_name: Filename to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Count classes
        unique, counts = np.unique(y_true, return_counts=True)
        class_names = [self.class_names[i] for i in unique]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot
        bars = ax1.bar(class_names, counts, alpha=0.8, color=sns.color_palette("husl", len(class_names)))
        ax1.set_title('Class Distribution (Counts)', fontweight='bold')
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(counts, labels=class_names, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Class Distribution (Percentages)', fontweight='bold')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved class distribution plot: {save_path}")
        
        return fig
    
    def plot_prediction_confidence(self, 
                                 y_prob: np.ndarray,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 title: str = "Prediction Confidence Analysis",
                                 save_name: str = "prediction_confidence.png") -> plt.Figure:
        """Plot prediction confidence analysis.
        
        Args:
            y_prob: Predicted probabilities
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            save_name: Filename to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Get max probabilities (confidence scores)
        confidence_scores = np.max(y_prob, axis=1)
        correct_predictions = (y_true == y_pred)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Confidence distribution for correct vs incorrect predictions
        axes[0, 0].hist(confidence_scores[correct_predictions], bins=30, alpha=0.7, 
                       label='Correct', color='green', density=True)
        axes[0, 0].hist(confidence_scores[~correct_predictions], bins=30, alpha=0.7, 
                       label='Incorrect', color='red', density=True)
        axes[0, 0].set_title('Confidence Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Confidence Score')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Accuracy vs confidence bins
        confidence_bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_counts = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (confidence_scores >= confidence_bins[i]) & (confidence_scores < confidence_bins[i + 1])
            if mask.sum() > 0:
                bin_accuracy = correct_predictions[mask].mean()
                bin_accuracies.append(bin_accuracy)
                bin_counts.append(mask.sum())
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)
        
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        axes[0, 1].bar(bin_centers, bin_accuracies, width=0.08, alpha=0.8)
        axes[0, 1].plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        axes[0, 1].set_title('Calibration Plot', fontweight='bold')
        axes[0, 1].set_xlabel('Confidence Score')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Per-class confidence
        class_confidences = []
        for class_id in range(len(self.class_names)):
            class_mask = y_true == class_id
            if class_mask.sum() > 0:
                class_conf = confidence_scores[class_mask].mean()
                class_confidences.append(class_conf)
            else:
                class_confidences.append(0)
        
        axes[1, 0].bar(self.class_names, class_confidences, alpha=0.8)
        axes[1, 0].set_title('Average Confidence by Class', fontweight='bold')
        axes[1, 0].set_xlabel('Classes')
        axes[1, 0].set_ylabel('Average Confidence')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Confidence vs accuracy scatter
        axes[1, 1].scatter(confidence_scores, correct_predictions.astype(float), 
                          alpha=0.5, s=10)
        axes[1, 1].set_title('Confidence vs Correctness', fontweight='bold')
        axes[1, 1].set_xlabel('Confidence Score')
        axes[1, 1].set_ylabel('Correct (1) / Incorrect (0)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved prediction confidence plot: {save_path}")
        
        return fig
    
    def create_evaluation_report(self, 
                               evaluation_results: Dict[str, Any],
                               training_history: Optional[Dict[str, List[float]]] = None,
                               save_name: str = "evaluation_report") -> Dict[str, plt.Figure]:
        """Create a comprehensive evaluation report with all visualizations.
        
        Args:
            evaluation_results: Results from model evaluation
            training_history: Training history (optional)
            save_name: Base name for saved files
            
        Returns:
            Dictionary of created figures
        """
        figures = {}
        
        # Confusion matrix
        cm_info = evaluation_results['confusion_matrix_info']
        figures['confusion_matrix'] = self.plot_confusion_matrix(
            cm_info['confusion_matrix'],
            normalize=True,
            save_name=f"{save_name}_confusion_matrix.png"
        )
        
        # Per-class metrics
        figures['per_class_metrics'] = self.plot_per_class_metrics(
            evaluation_results['per_class_metrics'],
            save_name=f"{save_name}_per_class_metrics.png"
        )
        
        # Class distribution
        if 'predictions' in evaluation_results:
            y_true = evaluation_results['predictions']['y_true']
            figures['class_distribution'] = self.plot_class_distribution(
                y_true,
                save_name=f"{save_name}_class_distribution.png"
            )
            
            # Prediction confidence
            y_prob = evaluation_results['predictions']['y_prob']
            y_pred = evaluation_results['predictions']['y_pred']
            figures['prediction_confidence'] = self.plot_prediction_confidence(
                y_prob, y_true, y_pred,
                save_name=f"{save_name}_prediction_confidence.png"
            )
        
        # Training history (if provided)
        if training_history:
            figures['training_history'] = self.plot_training_history(
                training_history,
                save_name=f"{save_name}_training_history.png"
            )
        
        logger.info(f"Created comprehensive evaluation report with {len(figures)} plots")
        return figures


class ReportGenerator:
    """Generates comprehensive evaluation reports."""
    
    def __init__(self, save_dir: str = "reports"):
        """Initialize report generator.
        
        Args:
            save_dir: Directory to save reports
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_text_report(self, 
                           evaluation_results: Dict[str, Any],
                           model_info: Dict[str, Any],
                           save_name: str = "evaluation_report.txt") -> str:
        """Generate a detailed text report.
        
        Args:
            evaluation_results: Results from model evaluation
            model_info: Model information
            save_name: Filename for the report
            
        Returns:
            Path to the saved report
        """
        report_path = self.save_dir / save_name
        
        with open(report_path, 'w') as f:
            f.write("MANIPULATION DETECTION MODEL - EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Model information
            f.write("MODEL INFORMATION\n")
            f.write("-" * 20 + "\n")
            for key, value in model_info.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Overall metrics
            basic_metrics = evaluation_results['basic_metrics']
            f.write("OVERALL PERFORMANCE\n")
            f.write("-" * 20 + "\n")
            f.write(f"Accuracy: {basic_metrics['accuracy']:.4f}\n")
            f.write(f"Precision (Macro): {basic_metrics['precision_macro']:.4f}\n")
            f.write(f"Recall (Macro): {basic_metrics['recall_macro']:.4f}\n")
            f.write(f"F1-Score (Macro): {basic_metrics['f1_macro']:.4f}\n")
            f.write(f"F1-Score (Weighted): {basic_metrics['f1_weighted']:.4f}\n")
            f.write("\n")
            
            # Manipulation detection metrics
            manip_metrics = evaluation_results['manipulation_metrics']
            f.write("MANIPULATION DETECTION PERFORMANCE\n")
            f.write("-" * 35 + "\n")
            f.write(f"Manipulation Accuracy: {manip_metrics['manipulation_accuracy']:.4f}\n")
            f.write(f"Manipulation Precision: {manip_metrics['manipulation_precision']:.4f}\n")
            f.write(f"Manipulation Recall: {manip_metrics['manipulation_recall']:.4f}\n")
            f.write(f"Manipulation F1-Score: {manip_metrics['manipulation_f1']:.4f}\n")
            f.write("\n")
            
            # Per-class performance
            per_class = evaluation_results['per_class_metrics']
            f.write("PER-CLASS PERFORMANCE\n")
            f.write("-" * 25 + "\n")
            f.write(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
            f.write("-" * 70 + "\n")
            for class_name, metrics in per_class.items():
                f.write(f"{class_name:<20} {metrics['precision']:<10.3f} "
                       f"{metrics['recall']:<10.3f} {metrics['f1_score']:<10.3f} "
                       f"{metrics['support']:<10d}\n")
            f.write("\n")
            
            # False positive analysis
            fp_analysis = evaluation_results['false_positive_analysis']
            f.write("FALSE POSITIVE ANALYSIS\n")
            f.write("-" * 25 + "\n")
            f.write(f"Total Ethical Samples: {fp_analysis['total_ethical_samples']}\n")
            f.write(f"Total False Positives: {fp_analysis['total_false_positives']}\n")
            f.write(f"Overall FP Rate: {fp_analysis['overall_fp_rate']:.4f}\n")
            f.write("\nFalse Positives by Manipulation Type:\n")
            for manip_type, count in fp_analysis['fp_by_manipulation_type'].items():
                f.write(f"  {manip_type}: {count}\n")
            f.write("\n")
        
        logger.info(f"Generated text report: {report_path}")
        return str(report_path)