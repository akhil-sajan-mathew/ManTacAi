import spacy
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPProcessor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NLPProcessor, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        try:
            logger.info("Loading spaCy model 'en_core_web_sm'...")
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"]) # Disable heavy components
            logger.info("spaCy model loaded successfully.")
        except OSError:
            logger.error("Model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'")
            self.nlp = None

    def lemmatize_text(self, text):
        """
        Returns a string of space-separated lemmas.
        Example: "He was killing me" -> "he be kill I"
        """
        if not self.nlp or not text:
            return text
        
        doc = self.nlp(text.lower())
        
        # We keep all tokens (including stop words) to preserve sentence structure 
        # for multi-word regex patterns (e.g. "i will kill you").
        lemmas = [token.lemma_ for token in doc]
        return " ".join(lemmas).lower()

# Global instance
nlp_processor = NLPProcessor()
