"""Integration tests for the manipulation detection training pipeline."""

import unittest
import torch
import tempfile
import os
import json
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import shutil

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.dataset_loader import DatasetLoader
from src.data.preprocessing import ManipulationDataset, TokenizerManager, create_dataset_from_dataframe
from src.data.data_loaders import DataLoaderManager, create_data_loaders_from_config
from src.models.manipulation_classifier import ManipulationClassifier
from src.models.model_utils import ModelConfig, ModelFactory, ModelCheckpoint
from src.training.trainer import ManipulationTrainer
from src.training.optimization import OptimizerFactory, SchedulerFactory
from src.evaluation.metrics import MetricsCalculator, ModelEvaluator
from src.inference.predictor import ManipulationPredictor


class TestEndToEndTrainingPipeline(unittest.TestCase):
    """Integration tests for the complete training pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_dir = os.path.join(self.temp_dir, "dataset")
        os.makedirs(self.dataset_dir)
        
        # Create minimal test dataset
        self.test_data = [
            {"text": "This is ethical persuasion example 1", "label": "ethical_persuasion", "split": "train"},
            {"text": "This is ethical persuasion example 2", "label": "ethical_persuasion", "split": "train"},
            {"text": "This is gaslighting example 1", "label": "gaslighting", "split": "train"},
            {"text": "This is gaslighting example 2", "label": "gaslighting", "split": "train"},
            {"text": "This is guilt tripping example", "label": "guilt_tripping", "split": "train"},
            {"text": "This is deflection example", "label": "deflection", "split": "train"},
            {"text": "Validation ethical example", "label": "ethical_persuasion", "split": "validation"},
            {"text": "Validation gaslighting example", "label": "gaslighting", "split": "validation"},
            {"text": "Test ethical example", "label": "ethical_persuasion", "split": "test"},
            {"text": "Test gaslighting example", "label": "gaslighting", "split": "test"}
        ]
        
        # Save test dataset
        self.dataset_file = os.path.join(self.dataset_dir, "test_dataset.json")
        with open(self.dataset_file, 'w') as f:
            json.dump(self.test_data, f)
        
        # Create test configuration
        self.config = {
            'model': {
                'name': 'distilbert-base-uncased',
                'num_classes': 11,
                'max_length': 128,  # Smaller for testing
                'dropout_rate': 0.1
            },
            'training': {
                'batch_size': 2,  # Small batch size for testing
                'learning_rate': 2e-5,
                'num_epochs': 1,  # Single epoch for testing
                'warmup_steps': 2,
                'weight_decay': 0.01,
                'gradient_clip_norm': 1.0
            },
            'data': {
                'dataset_path': self.dataset_dir,
                'train_file': 'test_dataset.json',
                'validation_split': 0.2,
                'test_split': 0.1
            },
            'hardware': {
                'use_gpu': False,  # Use CPU for testing
                'mixed_precision': False,
                'num_workers': 0  # No multiprocessing for testing
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_data_loading_pipeline(self, mock_tokenizer, mock_config, mock_model):
        """Test the complete data loading pipeline."""
        # Setup mocks
        mock_config.return_value = MagicMock()
        mock_config.return_value.hidden_size = 768
        mock_model.return_value = MagicMock()
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.vocab_size = 30522
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "[EOS]"
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 10)),
            'attention_mask': torch.ones(1, 10)
        }
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Test dataset loading
        loader = DatasetLoader(self.dataset_dir)
        df = loader.load_dataset('test_dataset.json')
        
        self.assertEqual(len(df), 10)
        self.assertIn('label_id', df.columns)
        
        # Test data splits
        train_df, val_df, test_df = loader.get_splits(df)
        
        self.assertEqual(len(train_df), 6)
        self.assertEqual(len(val_df), 2)
        self.assertEqual(len(test_df), 2)
        
        # Test data loader creation
        config_obj = ModelConfig(self.config)
        data_manager = DataLoaderManager(self.config)
        
        train_loader, val_loader, test_loader = data_manager.create_data_loaders()
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)
        
        # Test batch loading
        for batch in train_loader:
            self.assertIn('input_ids', batch)
            self.assertIn('attention_mask', batch)
            self.assertIn('labels', batch)
            break  # Just test first batch
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_model_creation_pipeline(self, mock_tokenizer, mock_config, mock_model):
        """Test model creation and configuration."""
        # Setup mocks
        mock_config.return_value = MagicMock()
        mock_config.return_value.hidden_size = 768
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        
        # Test model configuration
        config_obj = ModelConfig(self.config)
        
        self.assertEqual(config_obj.model_name, 'distilbert-base-uncased')
        self.assertEqual(config_obj.num_classes, 11)
        self.assertEqual(config_obj.batch_size, 2)
        
        # Test model factory
        model = ModelFactory.create_model(config_obj, model_type="standard")
        
        self.assertIsInstance(model, ManipulationClassifier)
        self.assertEqual(model.num_classes, 11)
        
        # Test model info
        model_info = model.get_model_info()
        self.assertIn('total_parameters', model_info)
        self.assertIn('trainable_parameters', model_info)
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_optimizer_and_scheduler_creation(self, mock_tokenizer, mock_config, mock_model):
        """Test optimizer and scheduler creation."""
        # Setup mocks
        mock_config.return_value = MagicMock()
        mock_config.return_value.hidden_size = 768
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        
        # Create model
        config_obj = ModelConfig(self.config)
        model = ModelFactory.create_model(config_obj)
        
        # Test optimizer creation
        optimizer = OptimizerFactory.create_optimizer(
            model=model,
            optimizer_type="adamw",
            learning_rate=config_obj.learning_rate,
            weight_decay=config_obj.weight_decay
        )
        
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        
        # Test scheduler creation
        num_training_steps = 10  # Small number for testing
        scheduler = SchedulerFactory.create_scheduler(
            optimizer=optimizer,
            scheduler_type="linear_warmup",
            num_training_steps=num_training_steps,
            num_warmup_steps=config_obj.warmup_steps
        )
        
        self.assertIsNotNone(scheduler)
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_checkpoint_management(self, mock_tokenizer, mock_config, mock_model):
        """Test checkpoint saving and loading."""
        # Setup mocks
        mock_config.return_value = MagicMock()
        mock_config.return_value.hidden_size = 768
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        
        # Create model and optimizer
        config_obj = ModelConfig(self.config)
        model = ModelFactory.create_model(config_obj)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Test checkpoint manager
        checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        checkpoint_manager = ModelCheckpoint(checkpoint_dir)
        
        # Save checkpoint
        metrics = {'accuracy': 0.8, 'loss': 0.5}
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=0,
            loss=0.5,
            metrics=metrics,
            config=config_obj
        )
        
        # Check if checkpoint was saved
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        self.assertTrue(len(checkpoint_files) > 0)
        
        # Test loading checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
        loaded_checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path)
        
        self.assertIn('model_state_dict', loaded_checkpoint)
        self.assertIn('optimizer_state_dict', loaded_checkpoint)
        self.assertIn('metrics', loaded_checkpoint)
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_training_integration(self, mock_tokenizer, mock_config, mock_model):
        """Test integration of training components."""
        # Setup mocks
        mock_config.return_value = MagicMock()
        mock_config.return_value.hidden_size = 768
        
        # Mock transformer with proper output structure
        mock_transformer = MagicMock()
        mock_transformer.last_hidden_state = torch.randn(2, 10, 768)
        mock_model.return_value = mock_transformer
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.vocab_size = 30522
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "[EOS]"
        
        # Mock tokenizer call to return proper tensors
        def mock_tokenizer_call(*args, **kwargs):
            batch_size = len(args[0]) if isinstance(args[0], list) else 1
            return {
                'input_ids': torch.randint(0, 1000, (batch_size, 10)),
                'attention_mask': torch.ones(batch_size, 10)
            }
        
        mock_tokenizer_instance.side_effect = mock_tokenizer_call
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Create components
        config_obj = ModelConfig(self.config)
        
        # Create data loaders
        data_manager = DataLoaderManager(self.config)
        train_loader, val_loader, test_loader = data_manager.create_data_loaders()
        
        # Create model
        model = ModelFactory.create_model(config_obj)
        
        # Create trainer
        checkpoint_dir = os.path.join(self.temp_dir, "training_checkpoints")
        trainer = ManipulationTrainer(
            model=model,
            config=config_obj,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=checkpoint_dir
        )
        
        # Test single training epoch (mock to avoid actual training)
        with patch.object(trainer, 'train_epoch') as mock_train_epoch, \
             patch.object(trainer, 'validate') as mock_validate:
            
            mock_train_epoch.return_value = {'train_loss': 0.5, 'learning_rate': 2e-5}
            mock_validate.return_value = {
                'val_loss': 0.6, 
                'val_accuracy': 0.7,
                'per_class_accuracy': {'ethical_persuasion': 0.8, 'gaslighting': 0.6},
                'predictions': [0, 1, 0, 1],
                'labels': [0, 1, 1, 0]
            }
            
            # Run training
            history = trainer.train()
            
            # Check training history
            self.assertIn('train_loss', history)
            self.assertIn('val_loss', history)
            self.assertIn('val_accuracy', history)
            self.assertEqual(len(history['train_loss']), 1)  # One epoch
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_evaluation_integration(self, mock_tokenizer, mock_config, mock_model):
        """Test integration of evaluation components."""
        # Setup mocks
        mock_config.return_value = MagicMock()
        mock_config.return_value.hidden_size = 768
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        
        # Create model
        config_obj = ModelConfig(self.config)
        model = ModelFactory.create_model(config_obj)
        
        # Test metrics calculator
        calculator = MetricsCalculator(num_classes=11)
        
        # Sample predictions
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 2])
        y_prob = np.random.rand(5, 11)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        
        # Test comprehensive metrics
        metrics = calculator.calculate_comprehensive_metrics(y_true, y_pred, y_prob)
        
        self.assertIn('basic_metrics', metrics)
        self.assertIn('per_class_metrics', metrics)
        self.assertIn('confusion_matrix_info', metrics)
        
        # Test model evaluator (mock the evaluation)
        evaluator = ModelEvaluator(model, device='cpu')
        
        # Mock data loader
        mock_data_loader = MagicMock()
        mock_batch = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10),
            'labels': torch.tensor([0, 1])
        }
        mock_data_loader.__iter__.return_value = [mock_batch]
        
        # Mock model output
        with patch.object(model, 'forward') as mock_forward:
            mock_forward.return_value = {
                'logits': torch.randn(2, 11)
            }
            
            results = evaluator.evaluate_model(mock_data_loader)
            
            self.assertIn('basic_metrics', results)
            self.assertIn('manipulation_metrics', results)
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_inference_integration(self, mock_tokenizer, mock_config, mock_model):
        """Test integration of inference components."""
        # Setup mocks
        mock_config.return_value = MagicMock()
        mock_config.return_value.hidden_size = 768
        
        mock_transformer = MagicMock()
        mock_transformer.last_hidden_state = torch.randn(1, 10, 768)
        mock_model.return_value = mock_transformer
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 10)),
            'attention_mask': torch.ones(1, 10)
        }
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Create model
        config_obj = ModelConfig(self.config)
        model = ModelFactory.create_model(config_obj)
        
        # Create predictor
        predictor = ManipulationPredictor(
            model=model,
            tokenizer=mock_tokenizer_instance,
            device='cpu'
        )
        
        # Test prediction
        result = predictor.predict_single("This is a test message.")
        
        self.assertIn('predicted_class', result)
        self.assertIn('confidence', result)
        self.assertIn('is_manipulation', result)
        
        # Test batch prediction
        from src.inference.predictor import BatchPredictor
        batch_predictor = BatchPredictor(predictor, batch_size=2)
        
        batch_results = batch_predictor.predict_batch([
            "Test message 1",
            "Test message 2"
        ])
        
        self.assertEqual(len(batch_results), 2)
        for result in batch_results:
            self.assertIn('predicted_class', result)
            self.assertIn('confidence', result)


class TestConfigurationIntegration(unittest.TestCase):
    """Integration tests for configuration management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_yaml_config_integration(self):
        """Test YAML configuration integration."""
        import yaml
        
        config_dict = {
            'model': {
                'name': 'distilbert-base-uncased',
                'num_classes': 11,
                'max_length': 512,
                'dropout_rate': 0.1
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 2e-5,
                'num_epochs': 5
            },
            'data': {
                'dataset_path': 'dataset/',
                'train_file': 'train.json'
            },
            'hardware': {
                'use_gpu': True,
                'mixed_precision': True
            }
        }
        
        # Save config to YAML
        config_path = os.path.join(self.temp_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        
        # Load config
        config = ModelConfig.from_yaml(config_path)
        
        # Test config values
        self.assertEqual(config.model_name, 'distilbert-base-uncased')
        self.assertEqual(config.num_classes, 11)
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.learning_rate, 2e-5)
        self.assertTrue(config.use_gpu)
        self.assertTrue(config.mixed_precision)
    
    def test_json_config_integration(self):
        """Test JSON configuration integration."""
        config_dict = {
            'model': {
                'name': 'roberta-base',
                'num_classes': 5,
                'dropout_rate': 0.2
            },
            'training': {
                'batch_size': 32,
                'num_epochs': 10
            }
        }
        
        # Save config to JSON
        config_path = os.path.join(self.temp_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config_dict, f)
        
        # Load config
        config = ModelConfig.from_json(config_path)
        
        # Test config values
        self.assertEqual(config.model_name, 'roberta-base')
        self.assertEqual(config.num_classes, 5)
        self.assertEqual(config.batch_size, 32)


class TestErrorHandlingIntegration(unittest.TestCase):
    """Integration tests for error handling across components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_missing_dataset_file(self):
        """Test handling of missing dataset file."""
        loader = DatasetLoader(self.temp_dir)
        
        with self.assertRaises(FileNotFoundError):
            loader.load_dataset("nonexistent.json")
    
    def test_invalid_dataset_format(self):
        """Test handling of invalid dataset format."""
        # Create invalid JSON file
        invalid_file = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_file, 'w') as f:
            f.write("invalid json content")
        
        loader = DatasetLoader(self.temp_dir)
        
        with self.assertRaises(ValueError):
            loader.load_dataset("invalid.json")
    
    def test_missing_config_file(self):
        """Test handling of missing configuration file."""
        with self.assertRaises(FileNotFoundError):
            ModelConfig.from_yaml("nonexistent.yaml")
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoConfig.from_pretrained')
    def test_model_creation_error_handling(self, mock_config, mock_model):
        """Test error handling in model creation."""
        # Mock failure in model loading
        mock_model.side_effect = Exception("Model loading failed")
        
        config_dict = {
            'model': {'name': 'invalid-model', 'num_classes': 11, 'dropout_rate': 0.1}
        }
        config = ModelConfig(config_dict)
        
        with self.assertRaises(Exception):
            ModelFactory.create_model(config)


if __name__ == '__main__':
    unittest.main()