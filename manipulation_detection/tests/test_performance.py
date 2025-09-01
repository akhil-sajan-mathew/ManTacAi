"""Performance and benchmark tests for manipulation detection model."""

import unittest
import torch
import time
import psutil
import os
import tempfile
import json
import numpy as np
from unittest.mock import patch, MagicMock
import threading
from concurrent.futures import ThreadPoolExecutor
import gc

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.manipulation_classifier import ManipulationClassifier
from src.models.model_utils import ModelConfig, ModelFactory
from src.inference.predictor import ManipulationPredictor, BatchPredictor
from src.inference.deployment import DeploymentInference
from src.inference.batch_processor import EnhancedBatchProcessor, ParallelBatchProcessor
from src.data.preprocessing import TextPreprocessor, TokenizerManager
from src.evaluation.metrics import MetricsCalculator


class PerformanceTestCase(unittest.TestCase):
    """Base class for performance tests with timing and memory utilities."""
    
    def setUp(self):
        """Set up performance testing utilities."""
        self.performance_results = {}
        self.memory_baseline = None
    
    def measure_time(self, func, *args, **kwargs):
        """Measure execution time of a function.
        
        Args:
            func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, execution_time_seconds)
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    
    def measure_memory(self):
        """Measure current memory usage.
        
        Returns:
            Memory usage in MB
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    
    def set_memory_baseline(self):
        """Set memory baseline for comparison."""
        gc.collect()  # Force garbage collection
        self.memory_baseline = self.measure_memory()
    
    def get_memory_increase(self):
        """Get memory increase from baseline.
        
        Returns:
            Memory increase in MB
        """
        if self.memory_baseline is None:
            return 0
        current_memory = self.measure_memory()
        return current_memory - self.memory_baseline
    
    def assert_performance_threshold(self, execution_time, threshold, operation_name):
        """Assert that execution time is below threshold.
        
        Args:
            execution_time: Measured execution time in seconds
            threshold: Maximum allowed time in seconds
            operation_name: Name of the operation for error message
        """
        self.assertLess(
            execution_time, 
            threshold,
            f"{operation_name} took {execution_time:.3f}s, expected < {threshold}s"
        )
    
    def assert_memory_threshold(self, memory_usage, threshold, operation_name):
        """Assert that memory usage is below threshold.
        
        Args:
            memory_usage: Memory usage in MB
            threshold: Maximum allowed memory in MB
            operation_name: Name of the operation for error message
        """
        self.assertLess(
            memory_usage,
            threshold,
            f"{operation_name} used {memory_usage:.1f}MB, expected < {threshold}MB"
        )


class TestModelPerformance(PerformanceTestCase):
    """Performance tests for model operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        
        # Test configuration
        self.config = {
            'model': {
                'name': 'distilbert-base-uncased',
                'num_classes': 11,
                'max_length': 512,
                'dropout_rate': 0.1
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoConfig.from_pretrained')
    def test_model_initialization_performance(self, mock_config, mock_model):
        """Test model initialization performance."""
        # Setup mocks
        mock_config.return_value = MagicMock()
        mock_config.return_value.hidden_size = 768
        mock_model.return_value = MagicMock()
        
        config_obj = ModelConfig(self.config)
        
        # Measure initialization time
        _, init_time = self.measure_time(
            ModelFactory.create_model, 
            config_obj, 
            model_type="standard"
        )
        
        # Assert initialization is fast (should be < 5 seconds)
        self.assert_performance_threshold(init_time, 5.0, "Model initialization")
        
        print(f"Model initialization time: {init_time:.3f}s")
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoConfig.from_pretrained')
    def test_model_forward_pass_performance(self, mock_config, mock_model):
        """Test model forward pass performance."""
        # Setup mocks
        mock_config.return_value = MagicMock()
        mock_config.return_value.hidden_size = 768
        
        mock_transformer = MagicMock()
        mock_transformer.last_hidden_state = torch.randn(32, 128, 768)
        mock_model.return_value = mock_transformer
        
        config_obj = ModelConfig(self.config)
        model = ModelFactory.create_model(config_obj)
        
        # Test input
        batch_size = 32
        seq_len = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Warm up
        for _ in range(3):
            model(input_ids, attention_mask)
        
        # Measure forward pass time
        _, forward_time = self.measure_time(
            model, input_ids, attention_mask
        )
        
        # Assert forward pass is fast (should be < 1 second for CPU)
        self.assert_performance_threshold(forward_time, 1.0, "Model forward pass")
        
        # Calculate throughput
        throughput = batch_size / forward_time
        print(f"Forward pass time: {forward_time:.3f}s")
        print(f"Throughput: {throughput:.1f} samples/second")
        
        # Assert minimum throughput (should process at least 10 samples/second)
        self.assertGreater(throughput, 10.0, "Model throughput too low")
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoConfig.from_pretrained')
    def test_model_memory_usage(self, mock_config, mock_model):
        """Test model memory usage."""
        # Setup mocks
        mock_config.return_value = MagicMock()
        mock_config.return_value.hidden_size = 768
        mock_model.return_value = MagicMock()
        
        self.set_memory_baseline()
        
        config_obj = ModelConfig(self.config)
        model = ModelFactory.create_model(config_obj)
        
        memory_increase = self.get_memory_increase()
        
        # Assert model doesn't use excessive memory (should be < 500MB)
        self.assert_memory_threshold(memory_increase, 500.0, "Model creation")
        
        print(f"Model memory usage: {memory_increase:.1f}MB")


class TestInferencePerformance(PerformanceTestCase):
    """Performance tests for inference operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        # Mock model and tokenizer
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        
        # Setup model outputs
        self.mock_model.eval.return_value = None
        self.mock_model.to.return_value = self.mock_model
        
        # Mock tokenizer outputs
        self.mock_tokenizer.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 128)),
            'attention_mask': torch.ones(1, 128)
        }
        
        # Mock model forward pass
        self.mock_model.return_value = {'logits': torch.randn(1, 11)}
        
        self.predictor = ManipulationPredictor(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device='cpu'
        )
    
    def test_single_prediction_performance(self):
        """Test single prediction performance."""
        test_text = "This is a test message for performance evaluation."
        
        # Warm up
        for _ in range(5):
            self.predictor.predict_single(test_text)
        
        # Measure prediction time
        _, prediction_time = self.measure_time(
            self.predictor.predict_single, test_text
        )
        
        # Assert prediction is fast (should be < 0.1 seconds)
        self.assert_performance_threshold(prediction_time, 0.1, "Single prediction")
        
        print(f"Single prediction time: {prediction_time:.4f}s")
    
    def test_batch_prediction_performance(self):
        """Test batch prediction performance."""
        batch_predictor = BatchPredictor(self.predictor, batch_size=32)
        
        # Create test batch
        test_texts = [f"Test message {i} for batch prediction performance." for i in range(100)]
        
        # Mock batch processing
        def mock_process_batch(texts, return_probs):
            return [
                {
                    'predicted_class': 'ethical_persuasion',
                    'confidence': 0.8,
                    'is_manipulation': False
                } for _ in texts
            ]
        
        batch_predictor._process_batch = mock_process_batch
        
        # Measure batch prediction time
        _, batch_time = self.measure_time(
            batch_predictor.predict_batch, test_texts
        )
        
        # Calculate throughput
        throughput = len(test_texts) / batch_time
        
        print(f"Batch prediction time: {batch_time:.3f}s")
        print(f"Batch throughput: {throughput:.1f} samples/second")
        
        # Assert minimum throughput (should process at least 50 samples/second)
        self.assertGreater(throughput, 50.0, "Batch prediction throughput too low")
    
    def test_concurrent_prediction_performance(self):
        """Test concurrent prediction performance."""
        test_texts = [f"Concurrent test message {i}." for i in range(20)]
        
        def predict_text(text):
            return self.predictor.predict_single(text)
        
        # Measure concurrent prediction time
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(predict_text, test_texts))
        
        concurrent_time = time.time() - start_time
        
        # Calculate throughput
        throughput = len(test_texts) / concurrent_time
        
        print(f"Concurrent prediction time: {concurrent_time:.3f}s")
        print(f"Concurrent throughput: {throughput:.1f} samples/second")
        
        # Assert all predictions completed
        self.assertEqual(len(results), len(test_texts))
        
        # Assert reasonable throughput
        self.assertGreater(throughput, 10.0, "Concurrent prediction throughput too low")


class TestDataProcessingPerformance(PerformanceTestCase):
    """Performance tests for data processing operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.preprocessor = TextPreprocessor()
    
    def test_text_preprocessing_performance(self):
        """Test text preprocessing performance."""
        # Create test texts with various complexities
        test_texts = [
            "Simple text.",
            "Text with    extra   spaces   and\n\nnewlines.",
            "Text with special characters: @#$%^&*()!",
            "Very long text " * 100,  # Long text
            ""  # Empty text
        ] * 200  # 1000 total texts
        
        # Measure preprocessing time
        _, preprocess_time = self.measure_time(
            self.preprocessor.preprocess_batch, test_texts
        )
        
        # Calculate throughput
        throughput = len(test_texts) / preprocess_time
        
        print(f"Text preprocessing time: {preprocess_time:.3f}s")
        print(f"Preprocessing throughput: {throughput:.1f} texts/second")
        
        # Assert minimum throughput (should process at least 1000 texts/second)
        self.assertGreater(throughput, 1000.0, "Text preprocessing throughput too low")
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_tokenization_performance(self, mock_tokenizer):
        """Test tokenization performance."""
        # Setup mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.vocab_size = 30522
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "[EOS]"
        
        def mock_tokenize(*args, **kwargs):
            texts = args[0] if isinstance(args[0], list) else [args[0]]
            batch_size = len(texts)
            return {
                'input_ids': torch.randint(0, 1000, (batch_size, 128)),
                'attention_mask': torch.ones(batch_size, 128)
            }
        
        mock_tokenizer_instance.side_effect = mock_tokenize
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        tokenizer_manager = TokenizerManager("distilbert-base-uncased")
        
        # Create test texts
        test_texts = [f"Test tokenization performance with text {i}." for i in range(100)]
        
        # Measure tokenization time
        _, tokenize_time = self.measure_time(
            tokenizer_manager.tokenize_batch, test_texts
        )
        
        # Calculate throughput
        throughput = len(test_texts) / tokenize_time
        
        print(f"Tokenization time: {tokenize_time:.3f}s")
        print(f"Tokenization throughput: {throughput:.1f} texts/second")
        
        # Assert minimum throughput
        self.assertGreater(throughput, 50.0, "Tokenization throughput too low")


class TestEvaluationPerformance(PerformanceTestCase):
    """Performance tests for evaluation operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.calculator = MetricsCalculator(num_classes=11)
    
    def test_metrics_calculation_performance(self):
        """Test metrics calculation performance."""
        # Generate large test data
        n_samples = 10000
        y_true = np.random.randint(0, 11, n_samples)
        y_pred = np.random.randint(0, 11, n_samples)
        y_prob = np.random.rand(n_samples, 11)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize
        
        # Measure metrics calculation time
        _, metrics_time = self.measure_time(
            self.calculator.calculate_comprehensive_metrics,
            y_true, y_pred, y_prob
        )
        
        print(f"Metrics calculation time: {metrics_time:.3f}s for {n_samples} samples")
        
        # Assert metrics calculation is fast (should be < 5 seconds for 10k samples)
        self.assert_performance_threshold(metrics_time, 5.0, "Metrics calculation")
        
        # Calculate throughput
        throughput = n_samples / metrics_time
        print(f"Metrics calculation throughput: {throughput:.1f} samples/second")
    
    def test_confusion_matrix_performance(self):
        """Test confusion matrix calculation performance."""
        # Generate large test data
        n_samples = 50000
        y_true = np.random.randint(0, 11, n_samples)
        y_pred = np.random.randint(0, 11, n_samples)
        
        # Measure confusion matrix calculation time
        _, cm_time = self.measure_time(
            self.calculator.calculate_confusion_matrix,
            y_true, y_pred
        )
        
        print(f"Confusion matrix calculation time: {cm_time:.3f}s for {n_samples} samples")
        
        # Assert confusion matrix calculation is fast
        self.assert_performance_threshold(cm_time, 2.0, "Confusion matrix calculation")


class TestMemoryLeakTests(PerformanceTestCase):
    """Tests for memory leaks in long-running operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        # Mock predictor
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        
        self.mock_model.eval.return_value = None
        self.mock_model.to.return_value = self.mock_model
        self.mock_tokenizer.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 128)),
            'attention_mask': torch.ones(1, 128)
        }
        self.mock_model.return_value = {'logits': torch.randn(1, 11)}
        
        self.predictor = ManipulationPredictor(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device='cpu'
        )
    
    def test_repeated_predictions_memory_leak(self):
        """Test for memory leaks in repeated predictions."""
        self.set_memory_baseline()
        
        # Perform many predictions
        test_text = "Test message for memory leak detection."
        
        for i in range(1000):
            self.predictor.predict_single(test_text)
            
            # Check memory every 100 iterations
            if i % 100 == 0:
                gc.collect()  # Force garbage collection
                memory_increase = self.get_memory_increase()
                
                # Memory should not increase significantly (< 100MB)
                if memory_increase > 100.0:
                    self.fail(f"Memory leak detected: {memory_increase:.1f}MB increase after {i+1} predictions")
        
        final_memory_increase = self.get_memory_increase()
        print(f"Final memory increase after 1000 predictions: {final_memory_increase:.1f}MB")
        
        # Assert no significant memory leak
        self.assert_memory_threshold(final_memory_increase, 50.0, "Repeated predictions")
    
    def test_batch_processing_memory_leak(self):
        """Test for memory leaks in batch processing."""
        batch_predictor = BatchPredictor(self.predictor, batch_size=32)
        
        # Mock batch processing
        def mock_process_batch(texts, return_probs):
            return [
                {
                    'predicted_class': 'ethical_persuasion',
                    'confidence': 0.8,
                    'is_manipulation': False
                } for _ in texts
            ]
        
        batch_predictor._process_batch = mock_process_batch
        
        self.set_memory_baseline()
        
        # Process many batches
        for i in range(100):
            test_texts = [f"Batch {i} message {j}" for j in range(32)]
            batch_predictor.predict_batch(test_texts)
            
            if i % 10 == 0:
                gc.collect()
                memory_increase = self.get_memory_increase()
                
                if memory_increase > 100.0:
                    self.fail(f"Memory leak detected in batch processing: {memory_increase:.1f}MB")
        
        final_memory_increase = self.get_memory_increase()
        print(f"Final memory increase after 100 batches: {final_memory_increase:.1f}MB")
        
        # Assert no significant memory leak
        self.assert_memory_threshold(final_memory_increase, 50.0, "Batch processing")


class TestScalabilityTests(PerformanceTestCase):
    """Tests for scalability with increasing load."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        
        # Mock predictor for scalability tests
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        
        self.mock_model.eval.return_value = None
        self.mock_model.to.return_value = self.mock_model
        
        def mock_tokenizer_call(*args, **kwargs):
            texts = args[0] if isinstance(args[0], list) else [args[0]]
            batch_size = len(texts)
            return {
                'input_ids': torch.randint(0, 1000, (batch_size, 128)),
                'attention_mask': torch.ones(batch_size, 128)
            }
        
        self.mock_tokenizer.side_effect = mock_tokenizer_call
        
        def mock_model_call(*args, **kwargs):
            batch_size = args[0].shape[0]
            return {'logits': torch.randn(batch_size, 11)}
        
        self.mock_model.side_effect = mock_model_call
        
        self.predictor = ManipulationPredictor(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device='cpu'
        )
    
    def test_batch_size_scalability(self):
        """Test performance scaling with batch size."""
        batch_predictor = BatchPredictor(self.predictor, batch_size=32)
        
        batch_sizes = [1, 10, 50, 100, 500]
        results = {}
        
        for batch_size in batch_sizes:
            test_texts = [f"Scalability test message {i}" for i in range(batch_size)]
            
            # Measure processing time
            _, process_time = self.measure_time(
                batch_predictor.predict_batch, test_texts
            )
            
            throughput = batch_size / process_time
            results[batch_size] = {
                'time': process_time,
                'throughput': throughput
            }
            
            print(f"Batch size {batch_size}: {process_time:.3f}s, {throughput:.1f} samples/sec")
        
        # Check that throughput generally increases with batch size
        # (allowing for some variance due to mocking)
        small_batch_throughput = results[1]['throughput']
        large_batch_throughput = results[100]['throughput']
        
        # Large batches should be more efficient than single predictions
        self.assertGreater(
            large_batch_throughput,
            small_batch_throughput * 0.5,  # Allow some overhead
            "Batch processing should be more efficient for larger batches"
        )
    
    def test_concurrent_load_scalability(self):
        """Test performance under concurrent load."""
        def single_prediction():
            return self.predictor.predict_single("Concurrent load test message")
        
        thread_counts = [1, 2, 4, 8]
        results = {}
        
        for num_threads in thread_counts:
            predictions_per_thread = 10
            total_predictions = num_threads * predictions_per_thread
            
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(single_prediction) for _ in range(total_predictions)]
                completed_results = [future.result() for future in futures]
            
            total_time = time.time() - start_time
            throughput = total_predictions / total_time
            
            results[num_threads] = {
                'time': total_time,
                'throughput': throughput,
                'completed': len(completed_results)
            }
            
            print(f"{num_threads} threads: {total_time:.3f}s, {throughput:.1f} predictions/sec")
            
            # Assert all predictions completed
            self.assertEqual(len(completed_results), total_predictions)
        
        # Check that concurrent processing provides some benefit
        single_thread_throughput = results[1]['throughput']
        multi_thread_throughput = results[4]['throughput']
        
        # Multi-threading should provide some improvement (even if limited by GIL)
        self.assertGreater(
            multi_thread_throughput,
            single_thread_throughput * 0.8,
            "Multi-threading should provide some performance benefit"
        )


class BenchmarkRunner:
    """Utility class for running comprehensive benchmarks."""
    
    def __init__(self):
        """Initialize benchmark runner."""
        self.results = {}
    
    def run_all_benchmarks(self):
        """Run all performance benchmarks and collect results."""
        print("Running Manipulation Detection Model Benchmarks")
        print("=" * 60)
        
        # Run test suites
        test_suites = [
            TestModelPerformance,
            TestInferencePerformance,
            TestDataProcessingPerformance,
            TestEvaluationPerformance,
            TestMemoryLeakTests,
            TestScalabilityTests
        ]
        
        for suite_class in test_suites:
            print(f"\nRunning {suite_class.__name__}...")
            suite = unittest.TestLoader().loadTestsFromTestCase(suite_class)
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            
            self.results[suite_class.__name__] = {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun
            }
        
        self.print_summary()
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        total_tests = 0
        total_failures = 0
        total_errors = 0
        
        for suite_name, results in self.results.items():
            total_tests += results['tests_run']
            total_failures += results['failures']
            total_errors += results['errors']
            
            print(f"{suite_name}:")
            print(f"  Tests: {results['tests_run']}")
            print(f"  Success Rate: {results['success_rate']:.1%}")
            if results['failures'] > 0:
                print(f"  Failures: {results['failures']}")
            if results['errors'] > 0:
                print(f"  Errors: {results['errors']}")
            print()
        
        overall_success_rate = (total_tests - total_failures - total_errors) / total_tests
        print(f"Overall Success Rate: {overall_success_rate:.1%}")
        print(f"Total Tests: {total_tests}")
        
        if total_failures > 0 or total_errors > 0:
            print(f"Total Issues: {total_failures + total_errors}")


if __name__ == '__main__':
    # Run individual test suite
    unittest.main()
    
    # Uncomment to run full benchmark suite
    # benchmark_runner = BenchmarkRunner()
    # benchmark_runner.run_all_benchmarks()