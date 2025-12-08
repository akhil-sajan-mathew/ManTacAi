import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import os

logger = logging.getLogger(__name__)

class ManipulationModel:
    def __init__(self, model_path=None, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.id2label = {}
        self.label2id = {}
        
        self.load_model()

    def load_model(self):
        """Load the model and tokenizer."""
        try:
            # Default to base model if no path provided or path doesn't exist
            if not self.model_path or not os.path.exists(self.model_path):
                logger.warning(f"Model path {self.model_path} not found. Loading base emotion model for testing.")
                checkpoint = "j-hartmann/emotion-english-distilroberta-base"
            else:
                checkpoint = self.model_path

            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id
            
            logger.info(f"Model loaded from {checkpoint} on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e

    def predict(self, text):
        """
        Predict manipulation tactics for a given text.
        Returns:
            dict: {label: probability, ...}
        """
        if not text:
            return {}

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Convert to dict
        result = {}
        for i, prob in enumerate(probs[0]):
            label = self.id2label.get(i, str(i))
            result[label] = float(prob)
            
        return result

    def predict_batch(self, texts):
        """Batch prediction."""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results
