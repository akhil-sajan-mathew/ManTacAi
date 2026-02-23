"""Unit tests for inference components."""

import unittest
import torch
import tempfile
import os
import json
from unittest.mock import patch, MagicMock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.inference.predictor import ManipulationPredictor, BatchPredictor
from src.inference.batch_processor import EnhancedBatchProcessor, ParallelBatchProcessor, ResultsAnalyzer
from src.inference.deployment import DeploymentInference, InferenceAPI


class TestManipulationPredictor(unittest.TestCase):
    """Test cases for ManipulationPredictor."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock model and tokenizer
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        
        # Setup model outputs
        self.mock_model.eval.return_value = None
        self.mock_model.to.return_value = self.mock_model
        
        # Mock tokenizer outputs
        mock_encoding = {
            'input_ids': torch.randint(0, 1000, (1, 10)),
            'attention_mask': torch.ones(1, 10)
        }
        self.mock_tokenizer.return_value = mock_encoding
        
        # Mock model forward pass
        mock_logits = torch.randn(1, 11)
        self.mock_model.return_value = {'logits': mock_logits}
        
        self.predictor = ManipulationPredictor(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device='cpu',
            confidence_threshold=0.5
        )
    
    def test_predictor_initialization(self):
        """Test predictor initialization."""
        self.assertEqual(self.predictor.device, torch.device('cpu'))
        self.assertEqual(self.predictor.confidence_threshold, 0.5)
        self.assertIsNotNone(self.predictor.preprocessor)
        self.assertIsNotNone(self.predictor.label_mapping)
    
    def test_predict_single_valid_text(self):
        """Test single text prediction with valid input."""
        test_text = "This is a test message."
        
        result = self.predictor.predict_single(test_text)
        
        # Check result structure
        expected_keys = [
            'predicted_class', 'predicted_class_id', 'confidence',
            'is_manipulation', 'high_confidence', 'original_text', 'processed_text'
        ]
        
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check types
        self.assertIsInstance(result['predicted_class'], str)
        self.assertIsInstance(result['predicted_class_id'], int)
        self.assertIsInstance(result['confidence'], float)
        self.assertIsInstance(result['is_manipulation'], bool)
    
    def test_predict_single_empty_text(self):
        """Test single text prediction with empty input."""
        result = self.predictor.predict_single("")
        
        self.assertEqual(result['predicted_class'], 'unknown')
        self.assertEqual(result['confidence'], 0.0)
        self.assertIn('warning', result)
    
    def test_predict_single_with_probabilities(self):
        """Test single text prediction with probabilities."""
        test_text = "This is a test message."
        
        result = self.predictor.predict_single(test_text, return_probabilities=True)
        
        self.assertIn('class_probabilities', result)
        self.assertIsInstance(result['class_probabilities'], dict)
        self.assertEqual(len(result['class_probabilities']), 11)
    
    def test_predict_single_with_top_k(self):
        """Test single text prediction with top-k results."""
        test_text = "This is a test message."
        
        result = self.predictor.predict_single(test_text, top_k=3)
        
        self.assertIn('top_k_predictions', result)
        self.assertIsInstance(result['top_k_predictions'], list)
        self.assertLessEqual(len(result['top_k_predictions']), 3)
    
    def test_predict_with_explanation(self):
        """Test prediction with explanation."""
        test_text = "This is a test message."
        
        result = self.predictor.predict_with_explanation(test_text)
        
        self.assertIn('explanation', result)
        self.assertIn('risk_assessment', result)
        self.assertIsInstance(result['explanation'], str)
        self.assertIsInstance(result['risk_assessment'], dict)
    
    def test_set_confidence_threshold(self):
        """Test setting confidence threshold."""
        new_threshold = 0.8
        self.predictor.set_confidence_threshold(new_threshold)
        
        self.assertEqual(self.predictor.confidence_threshold, new_threshold)
        
        # Test invalid threshold
        with self.assertRaises(ValueError):
            self.predictor.set_confidence_threshold(1.5)
    
    def test_get_model_info(self):
        """Test getting model information."""
        info = self.predictor.get_model_info()
        
        expected_keys = [
            'model_class', 'device', 'confidence_threshold',
            'num_classes', 'class_names', 'model_parameters'
        ]
        
        for key in expected_keys:
            self.assertIn(key, info)


class TestBatchPredictor(unittest.TestCase):
    """Test cases for BatchPredictor."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock predictor
        self.mock_predictor = MagicMock()
        self.mock_predictor.preprocessor.clean_text.side_effect = lambda x: x
        self.mock_predictor.tokenizer.return_value = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
        self.mock_predictor.model.return_value = {
            'logits': torch.randn(2, 11)
        }
        self.mock_predictor.device = 'cpu'
        self.mock_predictor.id_to_label = {i: f"class_{i}" for i in range(11)}
        self.mock_predictor.confidence_threshold = 0.5
        
        self.batch_predictor = BatchPredictor(self.mock_predictor, batch_size=2)
    
    def test_batch_prediction(self):
        """Test batch prediction."""
        texts = ["Text 1", "Text 2", "Text 3"]
        
        results = self.batch_predictor.predict_batch(texts)
        
        self.assertEqual(len(results), len(texts))
        
        for result in results:
            self.assertIn('predicted_class', result)
            self.assertIn('confidence', result)
            self.assertIn('is_manipulation', result)
    
    def test_batch_prediction_with_empty_texts(self):
        """Test batch prediction with empty texts."""
        texts = ["Valid text", "", "   ", "Another valid text"]
        
        results = self.batch_predictor.predict_batch(texts)
        
        self.assertEqual(len(results), len(texts))
        
        # Check that empty texts are handled
        for i, result in enumerate(results):
            if not texts[i].strip():
                self.assertEqual(result['predicted_class'], 'unknown')


class TestEnhancedBatchProcessor(unittest.TestCase):
    """Test cases for EnhancedBatchProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_predictor = MagicMock()
        self.processor = EnhancedBatchProcessor(self.mock_predictor, batch_size=2)
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_csv_file(self):
        """Test loading CSV file."""
        # Create test CSV
        csv_path = os.path.join(self.temp_dir, "test.csv")
        with open(csv_path, 'w') as f:
            f.write("text,label\n")
            f.write("Sample text 1,ethical\n")
            f.write("Sample text 2,manipulation\n")
        
        texts, metadata = self.processor._load_file(csv_path, 'text')
        
        self.assertEqual(len(texts), 2)
        self.assertEqual(texts[0], "Sample text 1")
        self.assertIsNotNone(metadata)
        self.assertEqual(len(metadata), 2)
    
    def test_load_json_file(self):
        """Test loading JSON file."""
        # Create test JSON
        json_path = os.path.join(self.temp_dir, "test.json")
        data = [
            {"text": "Sample text 1", "label": "ethical"},
            {"text": "Sample text 2", "label": "manipulation"}
        ]
        with open(json_path, 'w') as f:
            json.dump(data, f)
        
        texts, metadata = self.processor._load_file(json_path, 'text')
        
        self.assertEqual(len(texts), 2)
        self.assertEqual(texts[0], "Sample text 1")
        self.assertIsNotNone(metadata)
    
    def test_load_txt_file(self):
        """Test loading TXT file."""
        # Create test TXT
        txt_path = os.path.join(self.temp_dir, "test.txt")
        with open(txt_path, 'w') as f:
            f.write("Sample text 1\n")
            f.write("Sample text 2\n")
        
        texts, metadata = self.processor._load_file(txt_path, 'text')
        
        self.assertEqual(len(texts), 2)
        self.assertEqual(texts[0], "Sample text 1")
        self.assertIsNone(metadata)
    
    def test_calculate_processing_stats(self):
        """Test processing statistics calculation."""
        results = [
            {'predicted_class': 'ethical_persuasion', 'confidence': 0.8, 'is_manipulation': False, 'high_confidence': True},
            {'predicted_class': 'gaslighting', 'confidence': 0.9, 'is_manipulation': True, 'high_confidence': True},
            {'predicted_class': 'guilt_tripping', 'confidence': 0.7, 'is_manipulation': True, 'high_confidence': True}
        ]
        
        stats = self.processor._calculate_processing_stats(results, 1.0)
        
        expected_keys = [
            'total_texts', 'processing_time_seconds', 'texts_per_second',
            'manipulation_detected', 'manipulation_rate', 'high_confidence_predictions',
            'high_confidence_rate', 'average_confidence', 'class_distribution'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        self.assertEqual(stats['total_texts'], 3)
        self.assertEqual(stats['manipulation_detected'], 2)
        self.assertAlmostEqual(stats['manipulation_rate'], 2/3)


class TestResultsAnalyzer(unittest.TestCase):
    """Test cases for ResultsAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ResultsAnalyzer()
        
        self.sample_results = [
            {'predicted_class': 'ethical_persuasion', 'confidence': 0.8, 'is_manipulation': False},
            {'predicted_class': 'gaslighting', 'confidence': 0.9, 'is_manipulation': True},
            {'predicted_class': 'guilt_tripping', 'confidence': 0.7, 'is_manipulation': True},
            {'predicted_class': 'threatening_intimidation', 'confidence': 0.85, 'is_manipulation': True}
        ]
    
    def test_analyze_batch_results(self):
        """Test batch results analysis."""
        analysis = self.analyzer.analyze_batch_results(self.sample_results)
        
        expected_sections = [
            'summary', 'confidence_stats', 'class_distribution', 'risk_assessment'
        ]
        
        for section in expected_sections:
            self.assertIn(section, analysis)
        
        # Check summary
        summary = analysis['summary']
        self.assertEqual(summary['total_texts'], 4)
        self.assertEqual(summary['manipulation_detected'], 3)
        self.assertAlmostEqual(summary['manipulation_rate'], 0.75)
    
    def test_generate_summary_report(self):
        """Test summary report generation."""
        report = self.analyzer.generate_summary_report(self.sample_results)
        
        self.assertIsInstance(report, str)
        self.assertIn('BATCH PROCESSING SUMMARY REPORT', report)
        self.assertIn('OVERVIEW:', report)
        self.assertIn('CONFIDENCE ANALYSIS:', report)
        self.assertIn('CLASS DISTRIBUTION:', report)
    
    def test_analyze_empty_results(self):
        """Test analysis with empty results."""
        analysis = self.analyzer.analyze_batch_results([])
        
        self.assertIn('error', analysis)


class TestDeploymentInference(unittest.TestCase):
    """Test cases for DeploymentInference."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock checkpoint
        self.checkpoint_path = os.path.join(self.temp_dir, "model.pt")
        mock_checkpoint = {
            'model_state_dict': {},
            'config': {
                'model': {
                    'name': 'distilbert-base-uncased',
                    'num_classes': 11,
                    'dropout_rate': 0.1
                }
            }
        }
        torch.save(mock_checkpoint, self.checkpoint_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('src.inference.deployment.ManipulationClassifier')
    @patch('src.inference.deployment.AutoTokenizer')
    def test_deployment_inference_initialization(self, mock_tokenizer, mock_model):
        """Test deployment inference initialization."""
        # Setup mocks
        mock_model.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        
        inference = DeploymentInference(
            model_path=self.checkpoint_path,
            device='cpu',
            optimize_for_inference=False
        )
        
        self.assertEqual(inference.device, torch.device('cpu'))
        self.assertIsNotNone(inference.model)
        self.assertIsNotNone(inference.tokenizer)
    
    @patch('src.inference.deployment.ManipulationClassifier')
    @patch('src.inference.deployment.AutoTokenizer')
    def test_predict_method(self, mock_tokenizer, mock_model):
        """Test prediction method."""
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_model_instance.return_value = {'logits': torch.randn(1, 11)}
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 10)),
            'attention_mask': torch.ones(1, 10)
        }
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        inference = DeploymentInference(
            model_path=self.checkpoint_path,
            device='cpu',
            optimize_for_inference=False
        )
        
        result = inference.predict("Test message")
        
        expected_keys = [
            'predicted_class', 'confidence', 'is_manipulation',
            'inference_time_ms', 'timestamp'
        ]
        
        for key in expected_keys:
            self.assertIn(key, result)
    
    @patch('src.inference.deployment.ManipulationClassifier')
    @patch('src.inference.deployment.AutoTokenizer')
    def test_health_check(self, mock_tokenizer, mock_model):
        """Test health check functionality."""
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_model_instance.return_value = {'logits': torch.randn(1, 11)}
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 10)),
            'attention_mask': torch.ones(1, 10)
        }
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        inference = DeploymentInference(
            model_path=self.checkpoint_path,
            device='cpu',
            optimize_for_inference=False
        )
        
        health = inference.health_check()
        
        expected_keys = [
            'status', 'test_prediction_success', 'health_check_time_ms',
            'model_loaded', 'tokenizer_loaded', 'device', 'timestamp'
        ]
        
        for key in expected_keys:
            self.assertIn(key, health)


class TestInferenceAPI(unittest.TestCase):
    """Test cases for InferenceAPI."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_inference = MagicMock()
        self.api = InferenceAPI(self.mock_inference)
    
    def test_predict_text_valid(self):
        """Test API text prediction with valid input."""
        # Mock inference response
        self.mock_inference.predict.return_value = {
            'predicted_class': 'ethical_persuasion',
            'confidence': 0.8,
            'is_manipulation': False
        }
        
        response = self.api.predict_text("Test message")
        
        self.assertTrue(response['success'])
        self.assertIsNone(response['error'])
        self.assertIsNotNone(response['result'])
    
    def test_predict_text_empty(self):
        """Test API text prediction with empty input."""
        response = self.api.predict_text("")
        
        self.assertFalse(response['success'])
        self.assertIsNotNone(response['error'])
        self.assertIsNone(response['result'])
    
    def test_predict_batch_valid(self):
        """Test API batch prediction with valid input."""
        # Mock inference response
        self.mock_inference.predict_batch.return_value = [
            {'predicted_class': 'ethical_persuasion', 'confidence': 0.8},
            {'predicted_class': 'gaslighting', 'confidence': 0.9}
        ]
        
        response = self.api.predict_batch(["Text 1", "Text 2"])
        
        self.assertTrue(response['success'])
        self.assertIsNone(response['error'])
        self.assertIsNotNone(response['results'])
        self.assertEqual(response['count'], 2)
    
    def test_get_model_info(self):
        """Test getting model information via API."""
        # Mock inference stats
        self.mock_inference.get_stats.return_value = {
            'model_info': {'device': 'cpu', 'num_classes': 11}
        }
        self.mock_inference.label_mapping = {0: 'ethical_persuasion', 1: 'gaslighting'}
        
        response = self.api.get_model_info()
        
        self.assertTrue(response['success'])
        self.assertIn('model_info', response)
        self.assertIn('class_names', response)


if __name__ == '__main__':
    unittest.main()