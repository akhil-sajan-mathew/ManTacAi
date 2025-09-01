"""Unit tests for data loading functionality."""

import unittest
import tempfile
import json
import os
import pandas as pd
from unittest.mock import patch, MagicMock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.dataset_loader import DatasetLoader, load_manipulation_dataset
from src.data.preprocessing import TextPreprocessor, ManipulationDataset, TokenizerManager
from src.utils.config import get_label_mapping, get_reverse_label_mapping


class TestDatasetLoader(unittest.TestCase):
    """Test cases for DatasetLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = [
            {"text": "This is a test message", "label": "ethical_persuasion", "split": "train"},
            {"text": "Another test message", "label": "gaslighting", "split": "validation"},
            {"text": "Third test message", "label": "guilt_tripping", "split": "test"}
        ]
        
        # Create sample dataset file
        self.dataset_file = os.path.join(self.temp_dir, "test_dataset.json")
        with open(self.dataset_file, 'w') as f:
            json.dump(self.sample_data, f)
    
    def test_load_dataset_success(self):
        """Test successful dataset loading."""
        loader = DatasetLoader(self.temp_dir)
        df = loader.load_dataset("test_dataset.json")
        
        self.assertEqual(len(df), 3)
        self.assertIn('label_id', df.columns)
        self.assertEqual(df.iloc[0]['label_id'], 0)  # ethical_persuasion
        self.assertEqual(df.iloc[1]['label_id'], 1)  # gaslighting
    
    def test_load_dataset_file_not_found(self):
        """Test handling of missing dataset file."""
        loader = DatasetLoader(self.temp_dir)
        
        with self.assertRaises(FileNotFoundError):
            loader.load_dataset("nonexistent.json")
    
    def test_get_splits(self):
        """Test dataset splitting functionality."""
        loader = DatasetLoader(self.temp_dir)
        df = loader.load_dataset("test_dataset.json")
        train_df, val_df, test_df = loader.get_splits(df)
        
        self.assertEqual(len(train_df), 1)
        self.assertEqual(len(val_df), 1)
        self.assertEqual(len(test_df), 1)
        self.assertEqual(train_df.iloc[0]['split'], 'train')
        self.assertEqual(val_df.iloc[0]['split'], 'validation')
        self.assertEqual(test_df.iloc[0]['split'], 'test')
    
    def test_validate_dataset(self):
        """Test dataset validation."""
        loader = DatasetLoader(self.temp_dir)
        df = loader.load_dataset("test_dataset.json")
        stats = loader.validate_dataset(df)
        
        self.assertEqual(stats['total_examples'], 3)
        self.assertEqual(stats['num_classes'], 3)
        self.assertEqual(stats['missing_text'], 0)
        self.assertEqual(stats['empty_text'], 0)


class TestTextPreprocessor(unittest.TestCase):
    """Test cases for TextPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor()
    
    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        text = "  This is a test   with extra spaces  "
        cleaned = self.preprocessor.clean_text(text)
        self.assertEqual(cleaned, "This is a test with extra spaces")
    
    def test_clean_text_punctuation(self):
        """Test punctuation handling."""
        text = "What???? Really!!!!! No way....."
        cleaned = self.preprocessor.clean_text(text)
        self.assertEqual(cleaned, "What??? Really!!! No way...")
    
    def test_clean_text_quotes(self):
        """Test quote normalization."""
        text = ""This is a 'test' message""
        cleaned = self.preprocessor.clean_text(text)
        self.assertEqual(cleaned, '"This is a \'test\' message"')
    
    def test_preprocess_batch(self):
        """Test batch preprocessing."""
        texts = ["  Text 1  ", "Text 2???", "  Text 3  "]
        cleaned = self.preprocessor.preprocess_batch(texts)
        expected = ["Text 1", "Text 2???", "Text 3"]
        self.assertEqual(cleaned, expected)


class TestTokenizerManager(unittest.TestCase):
    """Test cases for TokenizerManager class."""
    
    @patch('src.data.preprocessing.AutoTokenizer')
    def test_tokenizer_initialization(self, mock_tokenizer):
        """Test tokenizer initialization."""
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value.vocab_size = 30522
        mock_tokenizer.from_pretrained.return_value.pad_token = None
        mock_tokenizer.from_pretrained.return_value.eos_token = "[EOS]"
        
        manager = TokenizerManager("test-model")
        
        mock_tokenizer.from_pretrained.assert_called_once_with("test-model")
        self.assertEqual(manager.model_name, "test-model")
    
    @patch('src.data.preprocessing.AutoTokenizer')
    def test_get_token_statistics(self, mock_tokenizer):
        """Test token statistics calculation."""
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.encode.side_effect = [
            [1, 2, 3, 4, 5],  # 5 tokens
            [1, 2, 3],        # 3 tokens
            [1, 2, 3, 4, 5, 6, 7, 8]  # 8 tokens
        ]
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        manager = TokenizerManager("test-model")
        texts = ["Text 1", "Text 2", "Text 3"]
        stats = manager.get_token_statistics(texts)
        
        self.assertAlmostEqual(stats['mean_length'], 5.33, places=1)
        self.assertEqual(stats['max_length'], 8)
        self.assertEqual(stats['min_length'], 3)


class TestLabelMapping(unittest.TestCase):
    """Test cases for label mapping utilities."""
    
    def test_get_label_mapping(self):
        """Test label mapping retrieval."""
        mapping = get_label_mapping()
        
        self.assertEqual(len(mapping), 11)
        self.assertEqual(mapping[0], "ethical_persuasion")
        self.assertEqual(mapping[1], "gaslighting")
        self.assertIn("whataboutism", mapping.values())
    
    def test_get_reverse_label_mapping(self):
        """Test reverse label mapping."""
        reverse_mapping = get_reverse_label_mapping()
        
        self.assertEqual(len(reverse_mapping), 11)
        self.assertEqual(reverse_mapping["ethical_persuasion"], 0)
        self.assertEqual(reverse_mapping["gaslighting"], 1)
        self.assertIn("whataboutism", reverse_mapping.keys())


if __name__ == '__main__':
    unittest.main()