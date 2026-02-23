"""Unit tests for evaluation components."""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.evaluation.metrics import MetricsCalculator, ManipulationMetrics, ModelEvaluator
from src.evaluation.visualization import EvaluationVisualizer, ReportGenerator
from src.evaluation.analysis import ModelAnalyzer, ErrorAnalyzer


class TestMetricsCalculator(unittest.TestCase):
    """Test cases for MetricsCalculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = MetricsCalculator(num_classes=11)
        
        # Sample data
        self.y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        self.y_pred = np.array([0, 1, 1, 0, 2, 2, 1, 1, 2, 0])
        self.y_prob = np.random.rand(10, 11)
        self.y_prob = self.y_prob / self.y_prob.sum(axis=1, keepdims=True)  # Normalize
    
    def test_calculate_basic_metrics(self):
        """Test basic metrics calculation."""
        metrics = self.calculator.calculate_basic_metrics(self.y_true, self.y_pred)
        
        # Check required metrics
        required_metrics = [
            'accuracy', 'precision_macro', 'precision_micro', 'precision_weighted',
            'recall_macro', 'recall_micro', 'recall_weighted',
            'f1_macro', 'f1_micro', 'f1_weighted'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)
            self.assertTrue(0 <= metrics[metric] <= 1)
    
    def test_calculate_per_class_metrics(self):
        """Test per-class metrics calculation."""
        per_class = self.calculator.calculate_per_class_metrics(self.y_true, self.y_pred)
        
        # Check structure
        self.assertIsInstance(per_class, dict)
        
        # Check that we have metrics for classes present in y_true
        unique_classes = np.unique(self.y_true)
        for class_id in unique_classes:
            class_name = self.calculator.class_names[class_id]
            self.assertIn(class_name, per_class)
            
            class_metrics = per_class[class_name]
            self.assertIn('precision', class_metrics)
            self.assertIn('recall', class_metrics)
            self.assertIn('f1_score', class_metrics)
            self.assertIn('support', class_metrics)
    
    def test_calculate_confusion_matrix(self):
        """Test confusion matrix calculation."""
        cm, matrix_info = self.calculator.calculate_confusion_matrix(self.y_true, self.y_pred)
        
        # Check confusion matrix shape
        num_unique_classes = len(np.unique(np.concatenate([self.y_true, self.y_pred])))
        self.assertEqual(cm.shape, (num_unique_classes, num_unique_classes))
        
        # Check matrix info
        self.assertIn('confusion_matrix', matrix_info)
        self.assertIn('confusion_matrix_normalized', matrix_info)
        self.assertIn('per_class_accuracy', matrix_info)
        self.assertIn('class_names', matrix_info)
    
    def test_calculate_multiclass_auc(self):
        """Test multiclass AUC calculation."""
        auc_metrics = self.calculator.calculate_multiclass_auc(self.y_true, self.y_prob)
        
        expected_metrics = ['auc_ovr_macro', 'auc_ovr_weighted', 'auc_ovo_macro', 'auc_ovo_weighted']
        
        for metric in expected_metrics:
            self.assertIn(metric, auc_metrics)
            self.assertIsInstance(auc_metrics[metric], float)
    
    def test_calculate_top_k_accuracy(self):
        """Test top-k accuracy calculation."""
        top_3_acc = self.calculator.calculate_top_k_accuracy(self.y_true, self.y_prob, k=3)
        
        self.assertIsInstance(top_3_acc, float)
        self.assertTrue(0 <= top_3_acc <= 1)
    
    def test_comprehensive_metrics(self):
        """Test comprehensive metrics calculation."""
        metrics = self.calculator.calculate_comprehensive_metrics(
            self.y_true, self.y_pred, self.y_prob
        )
        
        # Check all sections are present
        expected_sections = [
            'basic_metrics', 'per_class_metrics', 'confusion_matrix_info',
            'classification_report', 'auc_metrics', 'top_3_accuracy', 'top_5_accuracy'
        ]
        
        for section in expected_sections:
            self.assertIn(section, metrics)


class TestManipulationMetrics(unittest.TestCase):
    """Test cases for ManipulationMetrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = ManipulationMetrics()
        
        # Sample data with ethical (0) and manipulation classes (1-10)
        self.y_true = np.array([0, 1, 2, 0, 3, 4, 0, 5, 6, 0])
        self.y_pred = np.array([0, 1, 1, 1, 3, 4, 0, 5, 6, 0])
    
    def test_manipulation_detection_metrics(self):
        """Test manipulation vs ethical detection metrics."""
        metrics = self.metrics.calculate_manipulation_detection_metrics(self.y_true, self.y_pred)
        
        expected_metrics = [
            'manipulation_accuracy', 'manipulation_precision', 
            'manipulation_recall', 'manipulation_f1'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)
            self.assertTrue(0 <= metrics[metric] <= 1)
    
    def test_manipulation_type_accuracy(self):
        """Test per-manipulation-type accuracy."""
        accuracy_metrics = self.metrics.calculate_manipulation_type_accuracy(self.y_true, self.y_pred)
        
        self.assertIsInstance(accuracy_metrics, dict)
        
        # Check that we have accuracy for manipulation classes present in data
        manipulation_classes = [i for i in np.unique(self.y_true) if i != 0]
        for class_id in manipulation_classes:
            class_name = self.metrics.label_mapping[class_id]
            accuracy_key = f"{class_name}_accuracy"
            if accuracy_key in accuracy_metrics:
                self.assertIsInstance(accuracy_metrics[accuracy_key], float)
    
    def test_false_positive_analysis(self):
        """Test false positive analysis."""
        fp_analysis = self.metrics.calculate_false_positive_analysis(self.y_true, self.y_pred)
        
        expected_keys = [
            'total_ethical_samples', 'total_false_positives', 'overall_fp_rate',
            'fp_by_manipulation_type', 'fp_rates_by_type'
        ]
        
        for key in expected_keys:
            self.assertIn(key, fp_analysis)


class TestEvaluationVisualizer(unittest.TestCase):
    """Test cases for EvaluationVisualizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = EvaluationVisualizer(save_dir=self.temp_dir)
        
        # Sample data
        self.confusion_matrix = np.random.randint(0, 10, (11, 11))
        self.per_class_metrics = {
            'ethical_persuasion': {'precision': 0.8, 'recall': 0.9, 'f1_score': 0.85},
            'gaslighting': {'precision': 0.7, 'recall': 0.8, 'f1_score': 0.75}
        }
        self.training_history = {
            'train_loss': [0.8, 0.6, 0.4, 0.3, 0.2],
            'val_loss': [0.9, 0.7, 0.5, 0.4, 0.3],
            'val_accuracy': [0.6, 0.7, 0.8, 0.85, 0.9],
            'learning_rate': [2e-5, 1.8e-5, 1.6e-5, 1.4e-5, 1.2e-5]
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_plot_confusion_matrix(self):
        """Test confusion matrix plotting."""
        fig = self.visualizer.plot_confusion_matrix(
            self.confusion_matrix,
            save_name="test_confusion_matrix.png"
        )
        
        # Check that figure was created
        self.assertIsNotNone(fig)
        
        # Check that file was saved
        save_path = os.path.join(self.temp_dir, "test_confusion_matrix.png")
        self.assertTrue(os.path.exists(save_path))
    
    def test_plot_per_class_metrics(self):
        """Test per-class metrics plotting."""
        fig = self.visualizer.plot_per_class_metrics(
            self.per_class_metrics,
            save_name="test_per_class_metrics.png"
        )
        
        self.assertIsNotNone(fig)
        
        save_path = os.path.join(self.temp_dir, "test_per_class_metrics.png")
        self.assertTrue(os.path.exists(save_path))
    
    def test_plot_training_history(self):
        """Test training history plotting."""
        fig = self.visualizer.plot_training_history(
            self.training_history,
            save_name="test_training_history.png"
        )
        
        self.assertIsNotNone(fig)
        
        save_path = os.path.join(self.temp_dir, "test_training_history.png")
        self.assertTrue(os.path.exists(save_path))
    
    def test_plot_class_distribution(self):
        """Test class distribution plotting."""
        y_true = np.random.randint(0, 11, 100)
        
        fig = self.visualizer.plot_class_distribution(
            y_true,
            save_name="test_class_distribution.png"
        )
        
        self.assertIsNotNone(fig)
        
        save_path = os.path.join(self.temp_dir, "test_class_distribution.png")
        self.assertTrue(os.path.exists(save_path))


class TestReportGenerator(unittest.TestCase):
    """Test cases for ReportGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ReportGenerator(save_dir=self.temp_dir)
        
        # Sample evaluation results
        self.evaluation_results = {
            'basic_metrics': {
                'accuracy': 0.85,
                'precision_macro': 0.8,
                'recall_macro': 0.82,
                'f1_macro': 0.81,
                'f1_weighted': 0.83
            },
            'manipulation_metrics': {
                'manipulation_accuracy': 0.87,
                'manipulation_precision': 0.85,
                'manipulation_recall': 0.88,
                'manipulation_f1': 0.86
            },
            'per_class_metrics': {
                'ethical_persuasion': {
                    'precision': 0.9, 'recall': 0.85, 'f1_score': 0.87, 'support': 100
                },
                'gaslighting': {
                    'precision': 0.8, 'recall': 0.82, 'f1_score': 0.81, 'support': 50
                }
            },
            'false_positive_analysis': {
                'total_ethical_samples': 100,
                'total_false_positives': 15,
                'overall_fp_rate': 0.15,
                'fp_by_manipulation_type': {'gaslighting': 8, 'guilt_tripping': 7}
            }
        }
        
        self.model_info = {
            'model_name': 'distilbert-base-uncased',
            'num_classes': 11,
            'total_parameters': 66955787
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_generate_text_report(self):
        """Test text report generation."""
        report_path = self.generator.generate_text_report(
            self.evaluation_results,
            self.model_info,
            save_name="test_report.txt"
        )
        
        # Check that report was created
        self.assertTrue(os.path.exists(report_path))
        
        # Check report content
        with open(report_path, 'r') as f:
            content = f.read()
        
        # Check that key sections are present
        self.assertIn('MANIPULATION DETECTION MODEL', content)
        self.assertIn('MODEL INFORMATION', content)
        self.assertIn('OVERALL PERFORMANCE', content)
        self.assertIn('PER-CLASS PERFORMANCE', content)


class TestModelAnalyzer(unittest.TestCase):
    """Test cases for ModelAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock model and tokenizer
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        
        self.analyzer = ModelAnalyzer(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device='cpu'
        )
        
        # Sample data
        self.y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        self.y_pred = np.array([0, 1, 1, 0, 2, 2, 1, 1, 2, 0])
        self.texts = [f"Sample text {i}" for i in range(10)]
        self.y_prob = np.random.rand(10, 11)
        self.y_prob = self.y_prob / self.y_prob.sum(axis=1, keepdims=True)
    
    def test_analyze_misclassifications(self):
        """Test misclassification analysis."""
        analysis = self.analyzer.analyze_misclassifications(
            self.y_true, self.y_pred, self.texts, self.y_prob
        )
        
        expected_keys = [
            'total_misclassified', 'misclassification_rate', 'misclassified_examples',
            'confusion_pairs', 'most_confused_classes', 'confidence_analysis'
        ]
        
        for key in expected_keys:
            self.assertIn(key, analysis)
        
        # Check types
        self.assertIsInstance(analysis['total_misclassified'], int)
        self.assertIsInstance(analysis['misclassification_rate'], float)
        self.assertIsInstance(analysis['misclassified_examples'], list)
    
    def test_analyze_class_performance(self):
        """Test class performance analysis."""
        analysis = self.analyzer.analyze_class_performance(
            self.y_true, self.y_pred, self.y_prob
        )
        
        self.assertIsInstance(analysis, dict)
        
        # Check that we have analysis for classes present in data
        unique_classes = np.unique(self.y_true)
        for class_id in unique_classes:
            class_name = self.analyzer.label_mapping[class_id]
            if class_name in analysis:
                class_analysis = analysis[class_name]
                
                expected_keys = [
                    'total_samples', 'correct_predictions', 'accuracy',
                    'most_confused_with', 'error_rate'
                ]
                
                for key in expected_keys:
                    self.assertIn(key, class_analysis)
    
    def test_find_difficult_examples(self):
        """Test finding difficult examples."""
        difficult = self.analyzer.find_difficult_examples(
            self.y_true, self.y_prob, self.texts, top_k=5
        )
        
        expected_categories = ['lowest_confidence', 'highest_entropy', 'closest_predictions']
        
        for category in expected_categories:
            self.assertIn(category, difficult)
            self.assertIsInstance(difficult[category], list)
            self.assertLessEqual(len(difficult[category]), 5)


class TestErrorAnalyzer(unittest.TestCase):
    """Test cases for ErrorAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ErrorAnalyzer()
        
        # Sample data
        self.y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        self.y_pred = np.array([0, 1, 1, 0, 2, 2, 1, 1, 2, 0])
        self.texts = [f"Sample text {i}" for i in range(10)]
    
    def test_analyze_systematic_errors(self):
        """Test systematic error analysis."""
        analysis = self.analyzer.analyze_systematic_errors(
            self.y_true, self.y_pred, self.texts
        )
        
        expected_keys = ['directional_confusion', 'error_clusters', 'text_pattern_errors']
        
        for key in expected_keys:
            self.assertIn(key, analysis)
        
        # Check text pattern errors
        text_errors = analysis['text_pattern_errors']
        self.assertIn('most_common_words_in_errors', text_errors)
        self.assertIn('total_error_texts', text_errors)


if __name__ == '__main__':
    unittest.main()