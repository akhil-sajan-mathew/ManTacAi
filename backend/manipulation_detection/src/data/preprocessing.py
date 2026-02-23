"""Text preprocessing and tokenization utilities."""

import re
import string
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Handles text cleaning and preprocessing."""
    
    def __init__(self):
        """Initialize the text preprocessor."""
        pass
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Handle common encoding issues
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove excessive punctuation (more than 3 consecutive)
        text = re.sub(r'([!?.]){4,}', r'\1\1\1', text)
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts.
        
        Args:
            texts: List of raw text strings
            
        Returns:
            List of cleaned text strings
        """
        return [self.clean_text(text) for text in texts]


class ManipulationDataset(Dataset):
    """PyTorch Dataset for manipulation detection."""
    
    def __init__(self, 
                 texts: List[str], 
                 labels: List[int], 
                 tokenizer: AutoTokenizer,
                 max_length: int = 512,
                 preprocessor: TextPreprocessor = None):
        """Initialize the dataset.
        
        Args:
            texts: List of text strings
            labels: List of label integers
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            preprocessor: Text preprocessor instance
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessor = preprocessor or TextPreprocessor()
        
        # Preprocess all texts
        self.processed_texts = self.preprocessor.preprocess_batch(self.texts)
        
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing tokenized inputs and labels
        """
        text = self.processed_texts[idx]
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class TokenizerManager:
    """Manages tokenizer operations and batch processing."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """Initialize the tokenizer manager.
        
        Args:
            model_name: Name of the pre-trained model for tokenizer
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info(f"Initialized tokenizer for {model_name}")
        logger.info(f"Vocab size: {self.tokenizer.vocab_size}")
    
    def tokenize_batch(self, 
                      texts: List[str], 
                      max_length: int = 512,
                      padding: str = 'max_length',
                      truncation: bool = True) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate long sequences
            
        Returns:
            Dictionary containing tokenized inputs
        """
        encoding = self.tokenizer(
            texts,
            truncation=truncation,
            padding=padding,
            max_length=max_length,
            return_tensors='pt'
        )
        
        return encoding
    
    def get_token_statistics(self, texts: List[str]) -> Dict[str, float]:
        """Get tokenization statistics for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary containing tokenization statistics
        """
        token_lengths = []
        
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            token_lengths.append(len(tokens))
        
        stats = {
            'mean_length': sum(token_lengths) / len(token_lengths),
            'max_length': max(token_lengths),
            'min_length': min(token_lengths),
            'median_length': sorted(token_lengths)[len(token_lengths) // 2],
            'over_512': sum(1 for length in token_lengths if length > 512),
            'over_256': sum(1 for length in token_lengths if length > 256)
        }
        
        stats['percent_over_512'] = (stats['over_512'] / len(token_lengths)) * 100
        stats['percent_over_256'] = (stats['over_256'] / len(token_lengths)) * 100
        
        return stats


def create_dataset_from_dataframe(df: pd.DataFrame,
                                tokenizer: AutoTokenizer,
                                max_length: int = 512,
                                text_column: str = 'text',
                                label_column: str = 'label_id') -> ManipulationDataset:
    """Create a ManipulationDataset from a pandas DataFrame.
    
    Args:
        df: DataFrame containing text and labels
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        text_column: Name of the text column
        label_column: Name of the label column
        
    Returns:
        ManipulationDataset instance
    """
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    
    return ManipulationDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length
    )