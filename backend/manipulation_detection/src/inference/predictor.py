"""Inference and prediction utilities for manipulation detection."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from transformers import AutoTokenizer
import logging
from pathlib import Path
import json

from ..models.manipulation_classifier import ManipulationClassifier
from ..data.preprocessing import TextPreprocessor
from ..utils.config import get_label_mapping

logger = logging.getLogger(__name__)


class ManipulationPredictor:
    """Single text prediction interface for manipulation detection."""
    
    def __init__(self, 
                 model: ManipulationClassifier,
                 tokenizer: AutoTokenizer,
                 device: str = 'cpu',
                 confidence_threshold: float = 0.5):
        """Initialize the predictor.
        
        Args:
            model: Trained manipulation detection model
            tokenizer: Tokenizer used for the model
            device: Device to run inference on
            confidence_threshold: Minimum confidence for predictions
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Move model to device and set to eval mode
        self.model.to(device)
        self.model.eval()
        
        # Initialize text preprocessor
        self.preprocessor = TextPreprocessor()
        
        # Get label mappings
        self.label_mapping = get_label_mapping()
        self.id_to_label = {v: k for k, v in self.label_mapping.items()}
        
        logger.info(f"ManipulationPredictor initialized on {device}")
        logger.info(f"Confidence threshold: {confidence_threshold}")
    
    def predict_single(self, 
                      text: str,
                      return_probabilities: bool = True,
                      top_k: int = 3) -> Dict[str, Any]:
        """Predict manipulation tactic for a single text.
        
        Args:
            text: Input text to classify
            return_probabilities: Whether to return class probabilities
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary containing prediction results
        """
        # Preprocess text
        cleaned_text = self.preprocessor.clean_text(text)
        
        if not cleaned_text.strip():
            return {
                'predicted_class': 'unknown',
                'predicted_class_id': -1,
                'confidence': 0.0,
                'is_manipulation': False,
                'warning': 'Empty or invalid text input'
            }
        
        # Tokenize
        encoding = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']
            probabilities = F.softmax(logits, dim=-1)
        
        # Get prediction results
        predicted_id = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, predicted_id].item()
        predicted_class = self.id_to_label[predicted_id]
        
        # Determine if it's manipulation (not ethical_persuasion)
        is_manipulation = predicted_class != 'ethical_persuasion'
        
        result = {
            'predicted_class': predicted_class,
            'predicted_class_id': predicted_id,
            'confidence': confidence,
            'is_manipulation': is_manipulation,
            'high_confidence': confidence >= self.confidence_threshold,
            'original_text': text,
            'processed_text': cleaned_text
        }
        
        # Add probabilities if requested
        if return_probabilities:
            all_probs = probabilities[0].cpu().numpy()
            result['class_probabilities'] = {
                self.id_to_label[i]: float(prob) 
                for i, prob in enumerate(all_probs)
            }
        
        # Add top-k predictions
        if top_k > 1:
            top_k_indices = torch.topk(probabilities[0], k=min(top_k, len(self.label_mapping))).indices
            top_k_predictions = []
            
            for idx in top_k_indices:
                class_id = idx.item()
                class_name = self.id_to_label[class_id]
                prob = probabilities[0, class_id].item()
                
                top_k_predictions.append({
                    'class': class_name,
                    'class_id': class_id,
                    'probability': prob
                })
            
            result['top_k_predictions'] = top_k_predictions
        
        return result
    
    def predict_with_explanation(self, text: str) -> Dict[str, Any]:
        """Predict with additional explanation and confidence analysis.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary containing prediction with explanation
        """
        # Get basic prediction
        result = self.predict_single(text, return_probabilities=True, top_k=3)
        
        # Add explanation based on confidence and prediction
        explanation = self._generate_explanation(result)
        result['explanation'] = explanation
        
        # Add risk assessment
        risk_assessment = self._assess_risk(result)
        result['risk_assessment'] = risk_assessment
        
        return result
    
    def _generate_explanation(self, prediction_result: Dict[str, Any]) -> str:
        """Generate human-readable explanation for the prediction.
        
        Args:
            prediction_result: Result from predict_single
            
        Returns:
            Explanation string
        """
        predicted_class = prediction_result['predicted_class']
        confidence = prediction_result['confidence']
        is_manipulation = prediction_result['is_manipulation']
        
        if not is_manipulation:
            if confidence >= 0.8:
                return f"This text appears to be ethical persuasion with high confidence ({confidence:.1%}). No manipulation tactics detected."
            elif confidence >= 0.6:
                return f"This text likely represents ethical persuasion ({confidence:.1%}), though some uncertainty remains."
            else:
                return f"Low confidence classification as ethical persuasion ({confidence:.1%}). Manual review recommended."
        else:
            manipulation_type = predicted_class.replace('_', ' ').title()
            if confidence >= 0.8:
                return f"High confidence detection of {manipulation_type} ({confidence:.1%}). This appears to be a manipulation tactic."
            elif confidence >= 0.6:
                return f"Moderate confidence detection of {manipulation_type} ({confidence:.1%}). Likely manipulation tactic."
            else:
                return f"Low confidence detection of {manipulation_type} ({confidence:.1%}). Uncertain classification - manual review recommended."
    
    def _assess_risk(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk level based on prediction results.
        
        Args:
            prediction_result: Result from predict_single
            
        Returns:
            Risk assessment dictionary
        """
        predicted_class = prediction_result['predicted_class']
        confidence = prediction_result['confidence']
        is_manipulation = prediction_result['is_manipulation']
        
        # Define risk levels for different manipulation types
        high_risk_tactics = ['gaslighting', 'threatening_intimidation', 'belittling_ridicule']
        medium_risk_tactics = ['guilt_tripping', 'love_bombing', 'passive_aggression']
        low_risk_tactics = ['deflection', 'stonewalling', 'appeal_to_emotion', 'whataboutism']
        
        if not is_manipulation:
            risk_level = 'low'
            risk_score = 0.1
        elif predicted_class in high_risk_tactics:
            risk_level = 'high'
            risk_score = 0.7 + (confidence * 0.3)
        elif predicted_class in medium_risk_tactics:
            risk_level = 'medium'
            risk_score = 0.4 + (confidence * 0.3)
        else:
            risk_level = 'low'
            risk_score = 0.2 + (confidence * 0.2)
        
        # Adjust based on confidence
        if confidence < 0.5:
            risk_level = 'uncertain'
            risk_score *= 0.5
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'requires_attention': risk_level in ['high', 'medium'] and confidence > 0.6,
            'requires_review': confidence < 0.5 or risk_level == 'uncertain'
        }
    
    def set_confidence_threshold(self, threshold: float):
        """Update the confidence threshold.
        
        Args:
            threshold: New confidence threshold (0.0 to 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
        self.confidence_threshold = threshold
        logger.info(f"Updated confidence threshold to {threshold}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_class': self.model.__class__.__name__,
            'device': str(self.device),
            'confidence_threshold': self.confidence_threshold,
            'num_classes': len(self.label_mapping),
            'class_names': list(self.label_mapping.values()),
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        }


class BatchPredictor:
    """Batch prediction interface for processing multiple texts efficiently."""
    
    def __init__(self, 
                 predictor: ManipulationPredictor,
                 batch_size: int = 32):
        """Initialize batch predictor.
        
        Args:
            predictor: ManipulationPredictor instance
            batch_size: Batch size for processing
        """
        self.predictor = predictor
        self.batch_size = batch_size
        
    def predict_batch(self, 
                     texts: List[str],
                     return_probabilities: bool = True) -> List[Dict[str, Any]]:
        """Predict manipulation tactics for a batch of texts.
        
        Args:
            texts: List of input texts
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_results = self._process_batch(batch_texts, return_probabilities)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(self, 
                      texts: List[str],
                      return_probabilities: bool) -> List[Dict[str, Any]]:
        """Process a single batch of texts.
        
        Args:
            texts: Batch of input texts
            return_probabilities: Whether to return probabilities
            
        Returns:
            List of prediction results for the batch
        """
        # Preprocess all texts
        cleaned_texts = [self.predictor.preprocessor.clean_text(text) for text in texts]
        
        # Filter out empty texts
        valid_indices = [i for i, text in enumerate(cleaned_texts) if text.strip()]
        valid_texts = [cleaned_texts[i] for i in valid_indices]
        
        if not valid_texts:
            # Return empty results for all texts
            return [{
                'predicted_class': 'unknown',
                'predicted_class_id': -1,
                'confidence': 0.0,
                'is_manipulation': False,
                'warning': 'Empty or invalid text input'
            } for _ in texts]
        
        # Tokenize batch
        encoding = self.predictor.tokenizer(
            valid_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.predictor.device)
        attention_mask = encoding['attention_mask'].to(self.predictor.device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.predictor.model(input_ids, attention_mask)
            logits = outputs['logits']
            probabilities = F.softmax(logits, dim=-1)
        
        # Process results
        batch_results = []
        valid_idx = 0
        
        for i, text in enumerate(texts):
            if i in valid_indices:
                # Valid text - get prediction
                predicted_id = torch.argmax(probabilities[valid_idx], dim=-1).item()
                confidence = probabilities[valid_idx, predicted_id].item()
                predicted_class = self.predictor.id_to_label[predicted_id]
                is_manipulation = predicted_class != 'ethical_persuasion'
                
                result = {
                    'predicted_class': predicted_class,
                    'predicted_class_id': predicted_id,
                    'confidence': confidence,
                    'is_manipulation': is_manipulation,
                    'high_confidence': confidence >= self.predictor.confidence_threshold,
                    'original_text': text,
                    'processed_text': cleaned_texts[i]
                }
                
                if return_probabilities:
                    all_probs = probabilities[valid_idx].cpu().numpy()
                    result['class_probabilities'] = {
                        self.predictor.id_to_label[j]: float(prob) 
                        for j, prob in enumerate(all_probs)
                    }
                
                batch_results.append(result)
                valid_idx += 1
            else:
                # Invalid text
                batch_results.append({
                    'predicted_class': 'unknown',
                    'predicted_class_id': -1,
                    'confidence': 0.0,
                    'is_manipulation': False,
                    'warning': 'Empty or invalid text input',
                    'original_text': text,
                    'processed_text': cleaned_texts[i]
                })
        
        return batch_results


def load_predictor_from_checkpoint(checkpoint_path: str,
                                 config_path: str,
                                 device: str = 'cpu',
                                 confidence_threshold: float = 0.5) -> ManipulationPredictor:
    """Load a predictor from a saved checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        config_path: Path to the model configuration
        device: Device to load the model on
        confidence_threshold: Confidence threshold for predictions
        
    Returns:
        Loaded ManipulationPredictor instance
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load configuration
    if config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Create model
    model = ManipulationClassifier(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        dropout_rate=config['model']['dropout_rate']
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Create predictor
    predictor = ManipulationPredictor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        confidence_threshold=confidence_threshold
    )
    
    logger.info(f"Loaded predictor from checkpoint: {checkpoint_path}")
    return predictor