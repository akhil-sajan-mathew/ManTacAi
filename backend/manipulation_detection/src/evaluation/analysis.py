"""Advanced model analysis tools for manipulation detection."""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
import logging
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.config import get_label_mapping

logger = logging.getLogger(__name__)


class ModelAnalyzer:
    """Advanced analysis tools for model performance and behavior."""
    
    def __init__(self, model, tokenizer, device: str = 'cpu'):
        """Initialize model analyzer.
        
        Args:
            model: Trained model to analyze
            tokenizer: Tokenizer used for the model
            device: Device to run analysis on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.label_mapping = get_label_mapping()
        self.class_names = [self.label_mapping[i] for i in range(len(self.label_mapping))]
    
    def analyze_misclassifications(self, 
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 texts: List[str],
                                 y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze misclassified examples in detail.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            texts: Original text samples
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary containing misclassification analysis
        """
        misclassified_mask = y_true != y_pred
        misclassified_indices = np.where(misclassified_mask)[0]
        
        analysis = {
            'total_misclassified': len(misclassified_indices),
            'misclassification_rate': len(misclassified_indices) / len(y_true),
            'misclassified_examples': [],
            'confusion_pairs': defaultdict(int),
            'most_confused_classes': [],
            'confidence_analysis': {}
        }
        
        # Analyze each misclassified example
        for idx in misclassified_indices:
            true_class = self.class_names[y_true[idx]]
            pred_class = self.class_names[y_pred[idx]]
            confidence = y_prob[idx].max() if y_prob is not None else None
            
            example = {
                'index': int(idx),
                'text': texts[idx],
                'true_class': true_class,
                'predicted_class': pred_class,
                'confidence': float(confidence) if confidence is not None else None
            }
            
            if y_prob is not None:
                # Add top-3 predictions
                top_3_indices = np.argsort(y_prob[idx])[-3:][::-1]
                example['top_3_predictions'] = [
                    {
                        'class': self.class_names[i],
                        'probability': float(y_prob[idx][i])
                    } for i in top_3_indices
                ]
            
            analysis['misclassified_examples'].append(example)
            # Use string key for JSON compatibility
            analysis['confusion_pairs'][f"{true_class} -> {pred_class}"] += 1
        
        # Find most confused class pairs
        sorted_pairs = sorted(analysis['confusion_pairs'].items(), 
                            key=lambda x: x[1], reverse=True)
        analysis['most_confused_classes'] = sorted_pairs[:10]
        
        # Confidence analysis for misclassified examples
        if y_prob is not None:
            misclassified_confidences = y_prob[misclassified_mask].max(axis=1)
            correct_confidences = y_prob[~misclassified_mask].max(axis=1)
            
            analysis['confidence_analysis'] = {
                'avg_confidence_misclassified': float(misclassified_confidences.mean()),
                'avg_confidence_correct': float(correct_confidences.mean()),
                'high_confidence_errors': int(np.sum(misclassified_confidences > 0.8)),
                'low_confidence_correct': int(np.sum(correct_confidences < 0.5))
            }
        
        return analysis
    
    def analyze_class_performance(self, 
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze performance for each class in detail.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary containing per-class analysis
        """
        analysis = {}
        
        for class_id, class_name in self.label_mapping.items():
            class_mask = y_true == class_id
            class_samples = class_mask.sum()
            
            if class_samples == 0:
                continue
            
            # Basic metrics
            class_true = y_true[class_mask]
            class_pred = y_pred[class_mask]
            
            correct_predictions = (class_true == class_pred).sum()
            accuracy = correct_predictions / class_samples
            
            # Confusion analysis
            predicted_as = Counter(class_pred)
            most_confused_with = predicted_as.most_common(3)
            
            class_analysis = {
                'total_samples': int(class_samples),
                'correct_predictions': int(correct_predictions),
                'accuracy': float(accuracy),
                'most_confused_with': [
                    {
                        'class': self.class_names[pred_class],
                        'count': count,
                        'percentage': count / class_samples * 100
                    } for pred_class, count in most_confused_with
                ],
                'error_rate': float(1 - accuracy)
            }
            
            # Confidence analysis
            if y_prob is not None:
                class_confidences = y_prob[class_mask, class_id]
                class_analysis.update({
                    'avg_confidence': float(class_confidences.mean()),
                    'min_confidence': float(class_confidences.min()),
                    'max_confidence': float(class_confidences.max()),
                    'std_confidence': float(class_confidences.std())
                })
            
            analysis[class_name] = class_analysis
        
        return analysis
    
    def find_difficult_examples(self, 
                              y_true: np.ndarray,
                              y_prob: np.ndarray,
                              texts: List[str],
                              top_k: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        """Find the most difficult examples for the model.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            texts: Original text samples
            top_k: Number of difficult examples to return per category
            
        Returns:
            Dictionary containing difficult examples by category
        """
        # Calculate confidence for true class
        true_class_probs = y_prob[np.arange(len(y_true)), y_true]
        
        # Find examples with lowest confidence for true class
        lowest_confidence_indices = np.argsort(true_class_probs)[:top_k]
        
        # Find examples with highest entropy (most uncertain)
        entropy = -np.sum(y_prob * np.log(y_prob + 1e-8), axis=1)
        highest_entropy_indices = np.argsort(entropy)[-top_k:]
        
        # Find examples where top-2 predictions are very close
        sorted_probs = np.sort(y_prob, axis=1)
        prob_differences = sorted_probs[:, -1] - sorted_probs[:, -2]
        closest_predictions_indices = np.argsort(prob_differences)[:top_k]
        
        difficult_examples = {
            'lowest_confidence': [],
            'highest_entropy': [],
            'closest_predictions': []
        }
        
        # Process each category
        for category, indices in [
            ('lowest_confidence', lowest_confidence_indices),
            ('highest_entropy', highest_entropy_indices),
            ('closest_predictions', closest_predictions_indices)
        ]:
            for idx in indices:
                true_class = self.class_names[y_true[idx]]
                pred_class = self.class_names[np.argmax(y_prob[idx])]
                
                example = {
                    'index': int(idx),
                    'text': texts[idx],
                    'true_class': true_class,
                    'predicted_class': pred_class,
                    'true_class_probability': float(y_prob[idx, y_true[idx]]),
                    'predicted_class_probability': float(y_prob[idx].max()),
                    'entropy': float(entropy[idx])
                }
                
                if category == 'closest_predictions':
                    example['probability_difference'] = float(prob_differences[idx])
                
                difficult_examples[category].append(example)
        
        return difficult_examples
    
    def analyze_prediction_patterns(self, 
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  texts: List[str]) -> Dict[str, Any]:
        """Analyze patterns in model predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            texts: Original text samples
            
        Returns:
            Dictionary containing prediction pattern analysis
        """
        analysis = {
            'text_length_analysis': {},
            'common_words_analysis': {},
            'prediction_bias': {}
        }
        
        # Text length analysis
        text_lengths = [len(text.split()) for text in texts]
        
        # Group by prediction correctness
        correct_mask = y_true == y_pred
        correct_lengths = [text_lengths[i] for i in range(len(texts)) if correct_mask[i]]
        incorrect_lengths = [text_lengths[i] for i in range(len(texts)) if not correct_mask[i]]
        
        analysis['text_length_analysis'] = {
            'avg_length_correct': np.mean(correct_lengths) if correct_lengths else 0,
            'avg_length_incorrect': np.mean(incorrect_lengths) if incorrect_lengths else 0,
            'length_correlation_with_accuracy': np.corrcoef(text_lengths, correct_mask.astype(int))[0, 1]
        }
        
        # Prediction bias analysis
        pred_counts = Counter(y_pred)
        true_counts = Counter(y_true)
        
        bias_analysis = {}
        for class_id, class_name in self.label_mapping.items():
            predicted_freq = pred_counts.get(class_id, 0) / len(y_pred)
            true_freq = true_counts.get(class_id, 0) / len(y_true)
            bias = predicted_freq - true_freq
            
            bias_analysis[class_name] = {
                'predicted_frequency': predicted_freq,
                'true_frequency': true_freq,
                'bias': bias,
                'overpredict': bias > 0.05,
                'underpredict': bias < -0.05
            }
        
        analysis['prediction_bias'] = bias_analysis
        
        return analysis
    
    def extract_embeddings(self, 
                         texts: List[str],
                         batch_size: int = 32) -> np.ndarray:
        """Extract embeddings from the model for given texts.
        
        Args:
            texts: List of text samples
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings
        """
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch
                encoding = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Get transformer outputs
                transformer_outputs = self.model.transformer(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Use [CLS] token embeddings
                batch_embeddings = transformer_outputs.last_hidden_state[:, 0, :]
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def visualize_embeddings(self, 
                           embeddings: np.ndarray,
                           labels: np.ndarray,
                           method: str = 'tsne',
                           save_path: Optional[str] = None) -> plt.Figure:
        """Visualize embeddings using dimensionality reduction.
        
        Args:
            embeddings: High-dimensional embeddings
            labels: Labels for coloring
            method: Dimensionality reduction method ('tsne' or 'pca')
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Reduce dimensionality
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        elif method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError("Method must be 'tsne' or 'pca'")
        
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot each class
        for class_id, class_name in self.label_mapping.items():
            mask = labels == class_id
            if mask.sum() > 0:
                ax.scatter(reduced_embeddings[mask, 0], 
                          reduced_embeddings[mask, 1],
                          label=class_name, alpha=0.7, s=30)
        
        ax.set_title(f'Embedding Visualization ({method.upper()})', fontsize=16, fontweight='bold')
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved embedding visualization: {save_path}")
        
        return fig
    
    def compare_model_configurations(self, 
                                   results_list: List[Dict[str, Any]],
                                   config_names: List[str]) -> Dict[str, Any]:
        """Compare results from different model configurations.
        
        Args:
            results_list: List of evaluation results dictionaries
            config_names: Names for each configuration
            
        Returns:
            Dictionary containing comparison analysis
        """
        comparison = {
            'summary_table': {},
            'best_performing': {},
            'metric_comparison': {}
        }
        
        # Extract key metrics for comparison
        metrics_to_compare = [
            'accuracy', 'f1_macro', 'f1_weighted', 
            'precision_macro', 'recall_macro'
        ]
        
        summary_data = {}
        for metric in metrics_to_compare:
            summary_data[metric] = []
            for results in results_list:
                value = results['basic_metrics'][metric]
                summary_data[metric].append(value)
        
        # Create summary table
        comparison['summary_table'] = pd.DataFrame(summary_data, index=config_names)
        
        # Find best performing configuration for each metric
        for metric in metrics_to_compare:
            best_idx = np.argmax(summary_data[metric])
            comparison['best_performing'][metric] = {
                'configuration': config_names[best_idx],
                'value': summary_data[metric][best_idx]
            }
        
        # Calculate improvement percentages
        if len(results_list) > 1:
            baseline_idx = 0  # Use first configuration as baseline
            improvements = {}
            
            for i, config_name in enumerate(config_names[1:], 1):
                improvements[config_name] = {}
                for metric in metrics_to_compare:
                    baseline_value = summary_data[metric][baseline_idx]
                    current_value = summary_data[metric][i]
                    improvement = ((current_value - baseline_value) / baseline_value) * 100
                    improvements[config_name][metric] = improvement
            
            comparison['improvements_over_baseline'] = improvements
        
        return comparison


class ErrorAnalyzer:
    """Specialized analyzer for error patterns and failure modes."""
    
    def __init__(self):
        """Initialize error analyzer."""
        self.label_mapping = get_label_mapping()
        self.class_names = [self.label_mapping[i] for i in range(len(self.label_mapping))]
    
    def analyze_systematic_errors(self, 
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                texts: List[str]) -> Dict[str, Any]:
        """Analyze systematic error patterns.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            texts: Original text samples
            
        Returns:
            Dictionary containing systematic error analysis
        """
        errors = {
            'directional_confusion': {},
            'error_clusters': {},
            'text_pattern_errors': {}
        }
        
        # Directional confusion matrix analysis
        cm = confusion_matrix(y_true, y_pred)
        
        for i, true_class in enumerate(self.class_names):
            for j, pred_class in enumerate(self.class_names):
                if i != j and cm[i, j] > 0:
                    error_rate = cm[i, j] / cm[i].sum()
                    if error_rate > 0.1:  # Significant error rate
                        errors['directional_confusion'][f"{true_class}_to_{pred_class}"] = {
                            'count': int(cm[i, j]),
                            'error_rate': float(error_rate),
                            'severity': 'high' if error_rate > 0.3 else 'medium'
                        }
        
        # Find common error patterns in text
        misclassified_mask = y_true != y_pred
        error_texts = [texts[i] for i in range(len(texts)) if misclassified_mask[i]]
        
        # Simple pattern analysis (can be extended)
        common_words_in_errors = Counter()
        for text in error_texts:
            words = text.lower().split()
            common_words_in_errors.update(words)
        
        errors['text_pattern_errors'] = {
            'most_common_words_in_errors': common_words_in_errors.most_common(20),
            'total_error_texts': len(error_texts)
        }
        
        return errors