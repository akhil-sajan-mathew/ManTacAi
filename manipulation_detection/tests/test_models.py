"""Unit tests for model components."""

import unittest
import torch
import tempfile
import os
from unittest.mock import patch, MagicMock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.manipulation_classifier import ManipulationClassifier, ManipulationClassifierWithPooling
from src.models.model_utils import ModelConfig, ModelFactory, ModelCheckpoint, ModelAnalyzer


class TestManipulationClassifier(unittest.TestCase):
    """Test cases for ManipulationClassifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_name = "distilbert-base-uncased"
        self.num_classes = 11
        self.dropout_rate = 0.1
        
    @patch('src.models.manipulation_classifier.AutoModel')
    @patch('src.models.manipulation_classifier.AutoConfig')
    def test_model_initialization(self, mock_config, mock_model):
        """Test model initialization."""
        # Mock the transformer components
        mock_config.from_pretrained.return_value = MagicMock()
        mock_config.from_pretrained.return_value.hidden_size = 768
        mock_model.from_pretrained.return_value = MagicMock()
        
        model = ManipulationClassifier(
            model_name=self.model_name,
            num_classes=self.num_classes,
            dropout_rate=self.dropout_rate
        )
        
        self.assertEqual(model.model_name, self.model_name)
        self.assertEqual(model.num_classes, self.num_classes)
        self.assertEqual(model.dropout_rate, self.dropout_rate)
        self.assertEqual(model.hidden_size, 768)
    
    @patch('src.models.manipulation_classifier.AutoModel')
    @patch('src.models.manipulation_classifier.AutoConfig')
    def test_forward_pass(self, mock_config, mock_model):
        """Test forward pass."""
        # Setup mocks
        mock_config.from_pretrained.return_value = MagicMock()
        mock_config.from_pretrained.return_value.hidden_size = 768
        
        mock_transformer = MagicMock()
        mock_transformer.last_hidden_state = torch.randn(2, 10, 768)
        mock_model.from_pretrained.return_value = mock_transformer
        
        model = ManipulationClassifier(
            model_name=self.model_name,
            num_classes=self.num_classes,
            dropout_rate=self.dropout_rate
        )
        
        # Test input
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, self.num_classes, (batch_size,))
        
        # Forward pass
        outputs = model(input_ids, attention_mask, labels)
        
        # Check outputs
        self.assertIn('logits', outputs)
        self.assertIn('loss', outputs)
        self.assertEqual(outputs['logits'].shape, (batch_size, self.num_classes))
    
    @patch('src.models.manipulation_classifier.AutoModel')
    @patch('src.models.manipulation_classifier.AutoConfig')
    def test_predict_method(self, mock_config, mock_model):
        """Test predict method."""
        # Setup mocks
        mock_config.from_pretrained.return_value = MagicMock()
        mock_config.from_pretrained.return_value.hidden_size = 768
        
        mock_transformer = MagicMock()
        mock_transformer.last_hidden_state = torch.randn(1, 10, 768)
        mock_model.from_pretrained.return_value = mock_transformer
        
        model = ManipulationClassifier()
        
        # Test input
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones(1, 10)
        
        # Predict
        predictions, probabilities = model.predict(input_ids, attention_mask)
        
        # Check outputs
        self.assertEqual(predictions.shape, (1,))
        self.assertEqual(probabilities.shape, (1, self.num_classes))
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(1)))
    
    def test_get_model_info(self):
        """Test get_model_info method."""
        with patch('src.models.manipulation_classifier.AutoModel'), \
             patch('src.models.manipulation_classifier.AutoConfig') as mock_config:
            
            mock_config.from_pretrained.return_value = MagicMock()
            mock_config.from_pretrained.return_value.hidden_size = 768
            
            model = ManipulationClassifier()
            info = model.get_model_info()
            
            self.assertIn('model_name', info)
            self.assertIn('num_classes', info)
            self.assertIn('total_parameters', info)
            self.assertIn('trainable_parameters', info)


class TestManipulationClassifierWithPooling(unittest.TestCase):
    """Test cases for ManipulationClassifierWithPooling."""
    
    @patch('src.models.manipulation_classifier.AutoModel')
    @patch('src.models.manipulation_classifier.AutoConfig')
    def test_pooling_strategies(self, mock_config, mock_model):
        """Test different pooling strategies."""
        # Setup mocks
        mock_config.from_pretrained.return_value = MagicMock()
        mock_config.from_pretrained.return_value.hidden_size = 768
        mock_model.from_pretrained.return_value = MagicMock()
        
        pooling_strategies = ['cls', 'mean', 'max']
        
        for strategy in pooling_strategies:
            with self.subTest(strategy=strategy):
                model = ManipulationClassifierWithPooling(pooling_strategy=strategy)
                self.assertEqual(model.pooling_strategy, strategy)
    
    @patch('src.models.manipulation_classifier.AutoModel')
    @patch('src.models.manipulation_classifier.AutoConfig')
    def test_pooling_forward_pass(self, mock_config, mock_model):
        """Test forward pass with pooling."""
        # Setup mocks
        mock_config.from_pretrained.return_value = MagicMock()
        mock_config.from_pretrained.return_value.hidden_size = 768
        
        mock_transformer = MagicMock()
        mock_transformer.last_hidden_state = torch.randn(2, 10, 768)
        mock_model.from_pretrained.return_value = mock_transformer
        
        model = ManipulationClassifierWithPooling(pooling_strategy='mean')
        
        # Test input
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        
        # Forward pass
        outputs = model(input_ids, attention_mask)
        
        # Check outputs
        self.assertIn('logits', outputs)
        self.assertEqual(outputs['logits'].shape, (2, 11))


class TestModelConfig(unittest.TestCase):
    """Test cases for ModelConfig."""
    
    def test_config_initialization(self):
        """Test configuration initialization."""
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
            }
        }
        
        config = ModelConfig(config_dict)
        
        self.assertEqual(config.model_name, 'distilbert-base-uncased')
        self.assertEqual(config.num_classes, 11)
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.learning_rate, 2e-5)
    
    def test_config_from_yaml(self):
        """Test loading configuration from YAML."""
        config_dict = {
            'model': {'name': 'test-model', 'num_classes': 5},
            'training': {'batch_size': 8}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_dict, f)
            temp_path = f.name
        
        try:
            config = ModelConfig.from_yaml(temp_path)
            self.assertEqual(config.model_name, 'test-model')
            self.assertEqual(config.num_classes, 5)
        finally:
            os.unlink(temp_path)
    
    def test_config_to_dict(self):
        """Test converting configuration to dictionary."""
        config_dict = {'model': {'name': 'test'}, 'training': {'batch_size': 16}}
        config = ModelConfig(config_dict)
        
        result_dict = config.to_dict()
        self.assertEqual(result_dict, config_dict)


class TestModelFactory(unittest.TestCase):
    """Test cases for ModelFactory."""
    
    def test_create_standard_model(self):
        """Test creating standard model."""
        config_dict = {
            'model': {
                'name': 'distilbert-base-uncased',
                'num_classes': 11,
                'dropout_rate': 0.1
            }
        }
        config = ModelConfig(config_dict)
        
        with patch('src.models.manipulation_classifier.AutoModel'), \
             patch('src.models.manipulation_classifier.AutoConfig') as mock_config:
            
            mock_config.from_pretrained.return_value = MagicMock()
            mock_config.from_pretrained.return_value.hidden_size = 768
            
            model = ModelFactory.create_model(config, model_type="standard")
            
            self.assertIsInstance(model, ManipulationClassifier)
    
    def test_create_pooling_model(self):
        """Test creating pooling model."""
        config_dict = {
            'model': {
                'name': 'distilbert-base-uncased',
                'num_classes': 11,
                'dropout_rate': 0.1
            }
        }
        config = ModelConfig(config_dict)
        
        with patch('src.models.manipulation_classifier.AutoModel'), \
             patch('src.models.manipulation_classifier.AutoConfig') as mock_config:
            
            mock_config.from_pretrained.return_value = MagicMock()
            mock_config.from_pretrained.return_value.hidden_size = 768
            
            model = ModelFactory.create_model(
                config, 
                model_type="pooling", 
                pooling_strategy="mean"
            )
            
            self.assertIsInstance(model, ManipulationClassifierWithPooling)
            self.assertEqual(model.pooling_strategy, "mean")


class TestModelCheckpoint(unittest.TestCase):
    """Test cases for ModelCheckpoint."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = ModelCheckpoint(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_checkpoint_initialization(self):
        """Test checkpoint manager initialization."""
        self.assertTrue(os.path.exists(self.temp_dir))
        self.assertEqual(self.checkpoint_manager.checkpoint_dir, self.temp_dir)
    
    @patch('src.models.manipulation_classifier.AutoModel')
    @patch('src.models.manipulation_classifier.AutoConfig')
    def test_save_checkpoint(self, mock_config, mock_model):
        """Test saving checkpoint."""
        # Setup mocks
        mock_config.from_pretrained.return_value = MagicMock()
        mock_config.from_pretrained.return_value.hidden_size = 768
        mock_model.from_pretrained.return_value = MagicMock()
        
        # Create model and optimizer
        model = ManipulationClassifier()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Mock config
        config = ModelConfig({'model': {'name': 'test'}, 'training': {'batch_size': 16}})
        
        # Save checkpoint
        metrics = {'accuracy': 0.85, 'loss': 0.5}
        self.checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=1,
            loss=0.5,
            metrics=metrics,
            config=config
        )
        
        # Check if checkpoint file was created
        checkpoint_files = [f for f in os.listdir(self.temp_dir) if f.startswith('checkpoint_epoch_')]
        self.assertTrue(len(checkpoint_files) > 0)


class TestModelAnalyzer(unittest.TestCase):
    """Test cases for ModelAnalyzer."""
    
    @patch('src.models.manipulation_classifier.AutoModel')
    @patch('src.models.manipulation_classifier.AutoConfig')
    def test_count_parameters(self, mock_config, mock_model):
        """Test parameter counting."""
        # Setup mocks
        mock_config.from_pretrained.return_value = MagicMock()
        mock_config.from_pretrained.return_value.hidden_size = 768
        mock_model.from_pretrained.return_value = MagicMock()
        
        model = ManipulationClassifier()
        
        param_counts = ModelAnalyzer.count_parameters(model)
        
        self.assertIn('total_parameters', param_counts)
        self.assertIn('trainable_parameters', param_counts)
        self.assertIn('frozen_parameters', param_counts)
        self.assertIsInstance(param_counts['total_parameters'], int)
    
    @patch('src.models.manipulation_classifier.AutoModel')
    @patch('src.models.manipulation_classifier.AutoConfig')
    def test_analyze_model_layers(self, mock_config, mock_model):
        """Test model layer analysis."""
        # Setup mocks
        mock_config.from_pretrained.return_value = MagicMock()
        mock_config.from_pretrained.return_value.hidden_size = 768
        mock_model.from_pretrained.return_value = MagicMock()
        
        model = ManipulationClassifier()
        
        layer_info = ModelAnalyzer.analyze_model_layers(model)
        
        self.assertIsInstance(layer_info, list)
        if layer_info:  # If there are layers
            self.assertIn('name', layer_info[0])
            self.assertIn('type', layer_info[0])
            self.assertIn('parameters', layer_info[0])
    
    @patch('src.models.manipulation_classifier.AutoModel')
    @patch('src.models.manipulation_classifier.AutoConfig')
    def test_get_model_size_mb(self, mock_config, mock_model):
        """Test model size calculation."""
        # Setup mocks
        mock_config.from_pretrained.return_value = MagicMock()
        mock_config.from_pretrained.return_value.hidden_size = 768
        mock_model.from_pretrained.return_value = MagicMock()
        
        model = ManipulationClassifier()
        
        size_mb = ModelAnalyzer.get_model_size_mb(model)
        
        self.assertIsInstance(size_mb, float)
        self.assertGreater(size_mb, 0)


if __name__ == '__main__':
    unittest.main()