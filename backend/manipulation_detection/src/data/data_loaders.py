"""Data loader utilities for manipulation detection training."""

from typing import Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from collections import Counter
import logging

from .dataset_loader import DatasetLoader
from .preprocessing import ManipulationDataset, TokenizerManager, create_dataset_from_dataframe

logger = logging.getLogger(__name__)


class DataLoaderManager:
    """Manages creation and configuration of data loaders."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data loader manager.
        
        Args:
            config: Configuration dictionary containing data and training parameters
        """
        self.config = config
        self.dataset_loader = DatasetLoader(config['data']['dataset_path'])
        self.tokenizer_manager = TokenizerManager(config['model']['name'])
        
    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test data loaders.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Load dataset
        df = self.dataset_loader.load_dataset(self.config['data']['train_file'])
        train_df, val_df, test_df = self.dataset_loader.get_splits(df)
        
        # Get tokenizer
        tokenizer = self.tokenizer_manager.tokenizer
        max_length = self.config['model']['max_length']
        
        # Create datasets
        train_dataset = create_dataset_from_dataframe(train_df, tokenizer, max_length)
        val_dataset = create_dataset_from_dataframe(val_df, tokenizer, max_length)
        test_dataset = create_dataset_from_dataframe(test_df, tokenizer, max_length)
        
        # Get batch size and num_workers
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['hardware']['num_workers']
        
        # Create weighted sampler for training to handle class imbalance
        train_sampler = self._create_weighted_sampler(train_df['label_id'].tolist())
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True if self.config['hardware']['use_gpu'] else False,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.config['hardware']['use_gpu'] else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.config['hardware']['use_gpu'] else False
        )
        
        logger.info(f"Created data loaders:")
        logger.info(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
        logger.info(f"  Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
        logger.info(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")
        
        return train_loader, val_loader, test_loader
    
    def _create_weighted_sampler(self, labels: list) -> WeightedRandomSampler:
        """Create a weighted random sampler to handle class imbalance.
        
        Args:
            labels: List of label integers
            
        Returns:
            WeightedRandomSampler instance
        """
        # Count class frequencies
        class_counts = Counter(labels)
        num_classes = len(class_counts)
        
        # Calculate weights (inverse frequency)
        total_samples = len(labels)
        class_weights = {}
        for class_id, count in class_counts.items():
            class_weights[class_id] = total_samples / (num_classes * count)
        
        # Create sample weights
        sample_weights = [class_weights[label] for label in labels]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        logger.info(f"Created weighted sampler with class weights: {class_weights}")
        
        return sampler
    
    def get_tokenizer_stats(self) -> Dict[str, Any]:
        """Get tokenization statistics for the dataset.
        
        Returns:
            Dictionary containing tokenization statistics
        """
        df = self.dataset_loader.load_dataset(self.config['data']['train_file'])
        texts = df['text'].tolist()
        
        stats = self.tokenizer_manager.get_token_statistics(texts)
        
        logger.info("Tokenization statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        return stats


def collate_fn(batch):
    """Custom collate function for batching.
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Batched tensors
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


class DataAugmentation:
    """Data augmentation techniques for text classification."""
    
    def __init__(self):
        """Initialize data augmentation."""
        pass
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """Replace n words with synonyms (placeholder implementation).
        
        Args:
            text: Input text
            n: Number of words to replace
            
        Returns:
            Augmented text
        """
        # This is a placeholder - in practice, you'd use libraries like nltk
        # or more sophisticated augmentation techniques
        return text
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """Insert n random words (placeholder implementation).
        
        Args:
            text: Input text
            n: Number of words to insert
            
        Returns:
            Augmented text
        """
        # Placeholder implementation
        return text
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words with probability p.
        
        Args:
            text: Input text
            p: Probability of deleting each word
            
        Returns:
            Augmented text
        """
        words = text.split()
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if np.random.random() > p:
                new_words.append(word)
        
        # If all words are deleted, return original text
        if len(new_words) == 0:
            return text
        
        return ' '.join(new_words)


def create_data_loaders_from_config(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Convenience function to create data loaders from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    manager = DataLoaderManager(config)
    return manager.create_data_loaders()