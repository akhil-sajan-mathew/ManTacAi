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

    def predict_with_context(self, text, context_messages=None, return_embedding=False):
        """
        Dual-pass prediction with sliding context window.
        
        Pass 1 (isolated):   classify target text alone → baseline prediction
        Pass 2 (contextual): classify with preceding messages prepended → context-aware prediction
        
        Args:
            text (str): Target message to classify
            context_messages (list[str]): Last N messages (oldest first), typically 3-4
            return_embedding (bool): Whether to return the CLS embedding from Pass 1
            
        Returns:
            tuple: (isolated_preds, contextual_preds, embedding_or_None)
                   - isolated_preds: {label: prob} from classifying text alone
                   - contextual_preds: {label: prob} from classifying with context
                   - embedding: CLS embedding from Pass 1 (for semantic echo detection)
        """
        # Pass 1: Isolated prediction (always needed as baseline)
        if return_embedding:
            isolated_preds, embedding = self.predict(text, return_embedding=True)
        else:
            isolated_preds = self.predict(text, return_embedding=False)
            embedding = None
        
        # If no context provided, contextual = isolated
        if not context_messages or len(context_messages) == 0:
            return isolated_preds, isolated_preds, embedding
        
        # Pass 2: Contextual prediction
        context_input = self._build_context_input(text, context_messages)
        contextual_preds = self.predict(context_input, return_embedding=False)
        
        return isolated_preds, contextual_preds, embedding

    def _build_context_input(self, target_text, context_messages, max_tokens=512):
        """
        Builds a context-aware input string with smart token-budget truncation.
        
        Format: "[CTX] msg1 </s> msg2 </s> msg3 </s> [CUR] target_text"
        
        Token budget:
          - Target gets at least 256 tokens (non-negotiable)
          - Context fills the remaining budget, newest messages first
          - Oversized context messages keep only first + last sentence
          
        Args:
            target_text (str): The message being classified
            context_messages (list[str]): Previous messages, oldest first
            max_tokens (int): Model's max sequence length
            
        Returns:
            str: Formatted input string
        """
        sep = self.tokenizer.sep_token  # </s> for RoBERTa
        
        # Compute target token count
        target_tokens = self.tokenizer.encode(target_text, add_special_tokens=False)
        target_token_count = len(target_tokens)
        
        # Reserve space: target gets at least 256 tokens, plus special tokens overhead (~4)
        min_target_budget = 256
        special_tokens_overhead = 4  # <s>, </s>, and markers
        target_budget = max(min_target_budget, target_token_count)
        context_budget = max_tokens - target_budget - special_tokens_overhead
        
        if context_budget <= 10:
            # Not enough room for meaningful context, skip
            return target_text
        
        # Fill context budget from newest → oldest (reversed iteration)
        selected_context = []
        tokens_used = 0
        per_msg_budget = context_budget // max(len(context_messages), 1)
        
        for msg in reversed(context_messages):
            msg_tokens = self.tokenizer.encode(msg, add_special_tokens=False)
            msg_token_count = len(msg_tokens)
            
            # Check if this message fits in remaining budget
            remaining = context_budget - tokens_used
            if remaining <= 5:
                break  # No more room
            
            if msg_token_count <= remaining:
                # Fits entirely
                selected_context.insert(0, msg)
                tokens_used += msg_token_count + 1  # +1 for sep token
            elif remaining >= 20:
                # Too long but we have some room — keep first + last sentence
                truncated = self._truncate_keep_edges(msg, remaining)
                selected_context.insert(0, truncated)
                tokens_used += remaining
                break  # Budget exhausted
            else:
                break  # Not enough room for a useful truncation
        
        if not selected_context:
            return target_text
        
        # Build final string: [CTX] msg1 </s> msg2 </s> [CUR] target
        context_str = f" {sep} ".join(selected_context)
        return f"[CTX] {context_str} {sep} [CUR] {target_text}"

    def _truncate_keep_edges(self, text, max_tokens):
        """
        Truncates a long message by keeping the first and last sentence.
        The first sentence typically contains the 'hook' and the last contains
        the 'threat' or 'conclusion' — both carry the highest forensic signal.
        """
        sentences = text.replace('! ', '!|').replace('. ', '.|').replace('? ', '?|').split('|')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 2:
            # Already short enough or can't split further, just hard-truncate
            tokens = self.tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        first = sentences[0]
        last = sentences[-1]
        combined = f"{first} ... {last}"
        
        # Verify it fits
        combined_tokens = self.tokenizer.encode(combined, add_special_tokens=False)
        if len(combined_tokens) <= max_tokens:
            return combined
        
        # Still too long, hard-truncate
        tokens = self.tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

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
