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
                raise FileNotFoundError(
                    f"Custom model not found at '{self.model_path}'. "
                    f"ManTacAi requires the 18-label manipulation tactic model. "
                    f"The fallback emotion model has incompatible labels."
                )
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

    def predict(self, text, return_embedding=False):
        """
        Predict manipulation tactics for a given text.
        Args:
            text (str): Input text
            return_embedding (bool): If True, returns (probabilities, embedding_vector)
        Returns:
            dict or tuple: {label: probability} or ({label: prob}, embedding)
        """
        if not text:
            return ({}, None) if return_embedding else {}

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Extract CLS token embedding (Batch index 0, Sequence index 0, All features)
            # shape: [1, 768]
            embedding = None
            if return_embedding:
                # Last layer hidden state: outputs.hidden_states[-1]
                # CLS token is at index 0 for RoBERTa/BERT
                embedding = outputs.hidden_states[-1][:, 0, :].cpu().numpy()[0]
            
        # Convert to dict
        result = {}
        for i, prob in enumerate(probs[0]):
            label = self.id2label.get(i, str(i))
            result[label] = float(prob)
            
        if return_embedding:
            return result, embedding
        return result

    def predict_batch(self, texts):
        """True batched prediction — single tokenizer + model forward pass."""
        if not texts:
            return []

        # Filter empty texts, remember indices
        valid_indices = [i for i, t in enumerate(texts) if t]
        valid_texts = [texts[i] for i in valid_indices]

        if not valid_texts:
            return [{} for _ in texts]

        inputs = self.tokenizer(
            valid_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Build results for valid texts
        valid_results = []
        for prob_row in probs:
            result = {}
            for i, prob in enumerate(prob_row):
                label = self.id2label.get(i, str(i))
                result[label] = float(prob)
            valid_results.append(result)

        # Map back to original positions (empty texts get empty dict)
        all_results = [{} for _ in texts]
        for idx, res in zip(valid_indices, valid_results):
            all_results[idx] = res

        return all_results
