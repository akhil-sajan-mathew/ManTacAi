"""Deployment-ready inference module for manipulation detection."""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import Dict, List, Any, Optional, Union
import json
import logging
from pathlib import Path
import time
from datetime import datetime
import os

from .predictor import ManipulationPredictor
from ..models.manipulation_classifier import ManipulationClassifier
from ..data.preprocessing import TextPreprocessor
from ..utils.config import get_label_mapping

logger = logging.getLogger(__name__)


class DeploymentInference:
    """Production-ready inference class with optimizations and error handling."""
    
    def __init__(self, 
                 model_path: str,
                 config_path: Optional[str] = None,
                 device: str = 'auto',
                 optimize_for_inference: bool = True,
                 enable_logging: bool = True):
        """Initialize deployment inference.
        
        Args:
            model_path: Path to the saved model or checkpoint
            config_path: Path to model configuration (optional)
            device: Device to run inference ('auto', 'cpu', 'cuda')
            optimize_for_inference: Whether to optimize model for inference
            enable_logging: Whether to enable inference logging
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        self.enable_logging = enable_logging
        
        # Setup device
        self.device = self._setup_device(device)
        
        # Load model and tokenizer
        self.model, self.tokenizer, self.config = self._load_model_and_tokenizer()
        
        # Optimize for inference
        if optimize_for_inference:
            self._optimize_model()
        
        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.label_mapping = get_label_mapping()
        self.id_to_label = {v: k for k, v in self.label_mapping.items()}
        
        # Performance tracking
        self.inference_stats = {
            'total_predictions': 0,
            'total_inference_time': 0.0,
            'average_inference_time': 0.0,
            'predictions_per_second': 0.0
        }
        
        logger.info(f"DeploymentInference initialized on {self.device}")
        logger.info(f"Model loaded from: {self.model_path}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device.
        
        Args:
            device: Device specification
            
        Returns:
            PyTorch device
        """
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info("CUDA available, using GPU")
            else:
                device = 'cpu'
                logger.info("CUDA not available, using CPU")
        
        return torch.device(device)
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer from saved files.
        
        Returns:
            Tuple of (model, tokenizer, config)
        """
        # Load checkpoint or model
        if self.model_path.suffix == '.pt':
            # Load from checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if 'config' in checkpoint:
                config = checkpoint['config']
            elif self.config_path and self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    if self.config_path.suffix == '.json':
                        config = json.load(f)
                    else:
                        import yaml
                        config = yaml.safe_load(f)
            else:
                raise ValueError("No configuration found in checkpoint or config file")
            
            # Create model
            model = ManipulationClassifier(
                model_name=config['model']['name'],
                num_classes=config['model']['num_classes'],
                dropout_rate=config['model']['dropout_rate']
            )
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else:
            # Load from saved model directory
            model = ManipulationClassifier.from_pretrained(str(self.model_path))
            
            # Load config
            config_file = self.model_path / "config.json"
            with open(config_file, 'r') as f:
                config = json.load(f)
        
        # Load tokenizer
        model_name = config.get('model', {}).get('name', 'distilbert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Move model to device
        model.to(self.device)
        model.eval()
        
        return model, tokenizer, config
    
    def _optimize_model(self):
        """Optimize model for inference."""
        # Disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Enable inference mode optimizations
        if hasattr(torch, 'inference_mode'):
            self.model = torch.jit.optimize_for_inference(self.model)
        
        # Compile model if available (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                logger.info("Model compiled for optimized inference")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        logger.info("Model optimized for inference")
    
    def predict(self, 
               text: str,
               return_probabilities: bool = False,
               return_top_k: int = 1) -> Dict[str, Any]:
        """Make a prediction on input text.
        
        Args:
            text: Input text to classify
            return_probabilities: Whether to return class probabilities
            return_top_k: Number of top predictions to return
            
        Returns:
            Dictionary containing prediction results
        """
        start_time = time.time()
        
        try:
            # Preprocess text
            cleaned_text = self.preprocessor.clean_text(text)
            
            if not cleaned_text.strip():
                return {
                    'predicted_class': 'unknown',
                    'confidence': 0.0,
                    'is_manipulation': False,
                    'error': 'Empty or invalid text input',
                    'inference_time_ms': 0.0
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
            
            # Get results
            predicted_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_id].item()
            predicted_class = self.id_to_label[predicted_id]
            
            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Update stats
            self._update_stats(inference_time / 1000)  # Convert back to seconds for stats
            
            # Build result
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'is_manipulation': predicted_class != 'ethical_persuasion',
                'inference_time_ms': inference_time,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add probabilities if requested
            if return_probabilities:
                all_probs = probabilities[0].cpu().numpy()
                result['class_probabilities'] = {
                    self.id_to_label[i]: float(prob) 
                    for i, prob in enumerate(all_probs)
                }
            
            # Add top-k predictions
            if return_top_k > 1:
                top_k_indices = torch.topk(probabilities[0], k=min(return_top_k, len(self.label_mapping))).indices
                top_k_predictions = []
                
                for idx in top_k_indices:
                    class_id = idx.item()
                    class_name = self.id_to_label[class_id]
                    prob = probabilities[0, class_id].item()
                    
                    top_k_predictions.append({
                        'class': class_name,
                        'probability': prob
                    })
                
                result['top_k_predictions'] = top_k_predictions
            
            # Log prediction if enabled
            if self.enable_logging:
                logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.3f}, time: {inference_time:.1f}ms)")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                'predicted_class': 'error',
                'confidence': 0.0,
                'is_manipulation': False,
                'error': str(e),
                'inference_time_ms': (time.time() - start_time) * 1000
            }
    
    def predict_batch(self, 
                     texts: List[str],
                     batch_size: int = 32,
                     return_probabilities: bool = False) -> List[Dict[str, Any]]:
        """Make predictions on a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            return_probabilities: Whether to return probabilities
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = self._process_batch(batch_texts, return_probabilities)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(self, 
                      texts: List[str],
                      return_probabilities: bool) -> List[Dict[str, Any]]:
        """Process a single batch of texts.
        
        Args:
            texts: Batch of texts
            return_probabilities: Whether to return probabilities
            
        Returns:
            List of prediction results
        """
        start_time = time.time()
        
        try:
            # Preprocess texts
            cleaned_texts = [self.preprocessor.clean_text(text) for text in texts]
            
            # Filter valid texts
            valid_indices = [i for i, text in enumerate(cleaned_texts) if text.strip()]
            valid_texts = [cleaned_texts[i] for i in valid_indices]
            
            if not valid_texts:
                return [{
                    'predicted_class': 'unknown',
                    'confidence': 0.0,
                    'is_manipulation': False,
                    'error': 'Empty or invalid text input'
                } for _ in texts]
            
            # Tokenize batch
            encoding = self.tokenizer(
                valid_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                logits = outputs['logits']
                probabilities = F.softmax(logits, dim=-1)
            
            # Process results
            batch_results = []
            valid_idx = 0
            inference_time = (time.time() - start_time) * 1000
            
            for i, text in enumerate(texts):
                if i in valid_indices:
                    predicted_id = torch.argmax(probabilities[valid_idx], dim=-1).item()
                    confidence = probabilities[valid_idx, predicted_id].item()
                    predicted_class = self.id_to_label[predicted_id]
                    
                    result = {
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'is_manipulation': predicted_class != 'ethical_persuasion',
                        'inference_time_ms': inference_time / len(texts),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    if return_probabilities:
                        all_probs = probabilities[valid_idx].cpu().numpy()
                        result['class_probabilities'] = {
                            self.id_to_label[j]: float(prob) 
                            for j, prob in enumerate(all_probs)
                        }
                    
                    batch_results.append(result)
                    valid_idx += 1
                else:
                    batch_results.append({
                        'predicted_class': 'unknown',
                        'confidence': 0.0,
                        'is_manipulation': False,
                        'error': 'Empty or invalid text input'
                    })
            
            # Update stats
            self._update_stats((time.time() - start_time), len(texts))
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            return [{
                'predicted_class': 'error',
                'confidence': 0.0,
                'is_manipulation': False,
                'error': str(e)
            } for _ in texts]
    
    def _update_stats(self, inference_time: float, num_predictions: int = 1):
        """Update inference statistics.
        
        Args:
            inference_time: Time taken for inference in seconds
            num_predictions: Number of predictions made
        """
        self.inference_stats['total_predictions'] += num_predictions
        self.inference_stats['total_inference_time'] += inference_time
        
        if self.inference_stats['total_predictions'] > 0:
            self.inference_stats['average_inference_time'] = (
                self.inference_stats['total_inference_time'] / 
                self.inference_stats['total_predictions']
            )
            self.inference_stats['predictions_per_second'] = (
                self.inference_stats['total_predictions'] / 
                self.inference_stats['total_inference_time']
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics.
        
        Returns:
            Dictionary containing inference statistics
        """
        return {
            **self.inference_stats,
            'model_info': {
                'device': str(self.device),
                'model_path': str(self.model_path),
                'num_classes': len(self.label_mapping),
                'model_parameters': sum(p.numel() for p in self.model.parameters())
            }
        }
    
    def reset_stats(self):
        """Reset inference statistics."""
        self.inference_stats = {
            'total_predictions': 0,
            'total_inference_time': 0.0,
            'average_inference_time': 0.0,
            'predictions_per_second': 0.0
        }
        logger.info("Inference statistics reset")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the inference system.
        
        Returns:
            Dictionary containing health check results
        """
        try:
            # Test prediction
            test_text = "This is a test message for health check."
            start_time = time.time()
            result = self.predict(test_text)
            health_check_time = time.time() - start_time
            
            # Check if prediction was successful
            success = 'error' not in result and result['predicted_class'] != 'error'
            
            return {
                'status': 'healthy' if success else 'unhealthy',
                'test_prediction_success': success,
                'health_check_time_ms': health_check_time * 1000,
                'model_loaded': self.model is not None,
                'tokenizer_loaded': self.tokenizer is not None,
                'device': str(self.device),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


class InferenceAPI:
    """Simple API wrapper for deployment inference."""
    
    def __init__(self, deployment_inference: DeploymentInference):
        """Initialize API wrapper.
        
        Args:
            deployment_inference: DeploymentInference instance
        """
        self.inference = deployment_inference
    
    def predict_text(self, 
                    text: str,
                    include_probabilities: bool = False,
                    top_k: int = 1) -> Dict[str, Any]:
        """API endpoint for text prediction.
        
        Args:
            text: Input text
            include_probabilities: Whether to include class probabilities
            top_k: Number of top predictions to return
            
        Returns:
            API response dictionary
        """
        if not text or not text.strip():
            return {
                'success': False,
                'error': 'Empty text input',
                'result': None
            }
        
        try:
            result = self.inference.predict(
                text=text,
                return_probabilities=include_probabilities,
                return_top_k=top_k
            )
            
            return {
                'success': True,
                'error': None,
                'result': result
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'result': None
            }
    
    def predict_batch(self, 
                     texts: List[str],
                     batch_size: int = 32,
                     include_probabilities: bool = False) -> Dict[str, Any]:
        """API endpoint for batch prediction.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            include_probabilities: Whether to include probabilities
            
        Returns:
            API response dictionary
        """
        if not texts:
            return {
                'success': False,
                'error': 'Empty text list',
                'results': None
            }
        
        try:
            results = self.inference.predict_batch(
                texts=texts,
                batch_size=batch_size,
                return_probabilities=include_probabilities
            )
            
            return {
                'success': True,
                'error': None,
                'results': results,
                'count': len(results)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'results': None
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Model information dictionary
        """
        return {
            'success': True,
            'model_info': self.inference.get_stats()['model_info'],
            'class_names': list(self.inference.label_mapping.values()),
            'inference_stats': self.inference.get_stats()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check.
        
        Returns:
            Health check results
        """
        return self.inference.health_check()


def create_deployment_inference(model_path: str,
                              config_path: Optional[str] = None,
                              device: str = 'auto') -> DeploymentInference:
    """Factory function to create deployment inference.
    
    Args:
        model_path: Path to the model
        config_path: Path to configuration file
        device: Device to use
        
    Returns:
        DeploymentInference instance
    """
    return DeploymentInference(
        model_path=model_path,
        config_path=config_path,
        device=device,
        optimize_for_inference=True,
        enable_logging=True
    )