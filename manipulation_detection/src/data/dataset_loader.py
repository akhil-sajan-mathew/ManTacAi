"""Dataset loading utilities for manipulation detection."""

import json
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional
import logging
from ..utils.config import get_reverse_label_mapping

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Handles loading and preprocessing of the manipulation detection dataset."""
    
    def __init__(self, dataset_path: str = "dataset/"):
        """Initialize the dataset loader.
        
        Args:
            dataset_path: Path to the directory containing dataset files
        """
        self.dataset_path = dataset_path
        self.label_mapping = get_reverse_label_mapping()
        
    def load_dataset(self, filename: str = "enhanced_critical_splits.json") -> pd.DataFrame:
        """Load the dataset from JSON file.
        
        Args:
            filename: Name of the dataset file
            
        Returns:
            DataFrame containing the loaded dataset
            
        Raises:
            FileNotFoundError: If the dataset file is not found
            ValueError: If the dataset format is invalid
        """
        file_path = os.path.join(self.dataset_path, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Validate required columns
            required_columns = ['text', 'label', 'split']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Add label_id column
            df['label_id'] = df['label'].map(self.label_mapping)
            
            # Check for unmapped labels
            unmapped = df[df['label_id'].isna()]
            if not unmapped.empty:
                logger.warning(f"Found {len(unmapped)} examples with unmapped labels")
                df = df.dropna(subset=['label_id'])
            
            df['label_id'] = df['label_id'].astype(int)
            
            logger.info(f"Loaded dataset with {len(df)} examples")
            logger.info(f"Splits: {df['split'].value_counts().to_dict()}")
            logger.info(f"Labels: {df['label'].value_counts().to_dict()}")
            
            return df
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in dataset file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}")
    
    def get_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split the dataset into train, validation, and test sets.
        
        Args:
            df: DataFrame containing the full dataset
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_df = df[df['split'] == 'train'].copy()
        val_df = df[df['split'] == 'validation'].copy()
        test_df = df[df['split'] == 'test'].copy()
        
        logger.info(f"Train set: {len(train_df)} examples")
        logger.info(f"Validation set: {len(val_df)} examples") 
        logger.info(f"Test set: {len(test_df)} examples")
        
        return train_df, val_df, test_df
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, any]:
        """Validate the dataset and return statistics.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary containing validation statistics
        """
        stats = {
            'total_examples': len(df),
            'num_classes': df['label_id'].nunique(),
            'class_distribution': df['label'].value_counts().to_dict(),
            'split_distribution': df['split'].value_counts().to_dict(),
            'missing_text': df['text'].isna().sum(),
            'empty_text': (df['text'].str.strip() == '').sum(),
            'avg_text_length': df['text'].str.len().mean(),
            'max_text_length': df['text'].str.len().max(),
            'min_text_length': df['text'].str.len().min()
        }
        
        # Check for class imbalance
        class_counts = df['label'].value_counts()
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        stats['class_imbalance_ratio'] = imbalance_ratio
        
        # Validation warnings
        warnings = []
        if stats['missing_text'] > 0:
            warnings.append(f"Found {stats['missing_text']} examples with missing text")
        if stats['empty_text'] > 0:
            warnings.append(f"Found {stats['empty_text']} examples with empty text")
        if imbalance_ratio > 3.0:
            warnings.append(f"High class imbalance detected (ratio: {imbalance_ratio:.2f})")
        
        stats['warnings'] = warnings
        
        return stats


def load_manipulation_dataset(dataset_path: str = "dataset/", 
                            filename: str = "enhanced_critical_splits.json") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience function to load and split the manipulation detection dataset.
    
    Args:
        dataset_path: Path to dataset directory
        filename: Dataset filename
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    loader = DatasetLoader(dataset_path)
    df = loader.load_dataset(filename)
    return loader.get_splits(df)