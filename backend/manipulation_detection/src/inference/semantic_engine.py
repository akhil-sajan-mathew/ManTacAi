
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class SemanticAnalyzer:
    """
    Analyzes vector embeddings to detect high-level concepts 
    (e.g., Violence, Self-Harm) that might be phrased implicitly.
    Uses a CONTRASTIVE approach: compares similarity to danger centroids
    vs. a safe centroid, using the difference as the signal.
    """
    def __init__(self):
        self.danger_centroids: Dict[str, np.ndarray] = {}
        self.safe_centroid: np.ndarray = None
        self.contrastive_threshold = 0.50  # Calibrated: danger-safe > 0.50
        
        # Seed phrases: Danger Concepts
        self.danger_seeds = {
            "PHYSICAL_VIOLENCE": [
                "physical violence", "murder and death", "severe bodily harm",
                "kill someone", "beat up and hurt", "stab and shoot",
                "destroy a life", "violent attack", 
                "sleep with the fishes", "end of the road for you",
                # Phase 24: Outcome-based implicit threat phrases
                "you won't survive this", "something bad will happen to you",
                "make you regret it", "you will pay for this"
            ],
            "SELF_HARM": [
                "suicide", "killing myself", "end my life",
                "self destruction", "better off dead", "hurt myself"
            ],
            # Phase 24: Restriction-of-movement / coercive control
            "COERCIVE_RESTRICTION": [
                "I will stop you from leaving",
                "you are not allowed to go",
                "block the exit door",
                "prevent you from escaping",
                "consequences if you leave",
                "you won't make it out of here",
                "you can't leave this house",
                "I won't let you go"
            ]
        }
        
        # Seed phrases: Safe Baseline (ordinary conversation)
        self.safe_seeds = [
            "I am going to the store",
            "Let us have dinner tonight",
            "The weather is nice today",
            "I need to sleep now",
            "Can you pick up the kids",
            "I will be home late",
            "Good morning how are you"
        ]

    def _normalize(self, vec):
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(vec)
        if norm > 0:
            return vec / norm
        return vec

    def compute_centroids(self, model):
        """
        Generates reference vectors using the provided model.
        Must be called on startup.
        """
        logger.info("Computing Semantic Centroids (Contrastive Mode)...")
        try:
            # 1. Compute Danger Centroids
            for concept, phrases in self.danger_seeds.items():
                vectors = []
                for phrase in phrases:
                    _, emb = model.predict(phrase, return_embedding=True)
                    if emb is not None:
                        vectors.append(emb)
                
                if vectors:
                    centroid = np.mean(np.array(vectors), axis=0)
                    self.danger_centroids[concept] = self._normalize(centroid)
                    logger.info(f"Danger Centroid: {concept} ({len(vectors)} seeds)")
                else:
                    logger.warning(f"No vectors generated for concept {concept}")
            
            # 2. Compute Safe Centroid
            safe_vectors = []
            for phrase in self.safe_seeds:
                _, emb = model.predict(phrase, return_embedding=True)
                if emb is not None:
                    safe_vectors.append(emb)
            
            if safe_vectors:
                self.safe_centroid = self._normalize(np.mean(np.array(safe_vectors), axis=0))
                logger.info(f"Safe Centroid computed ({len(safe_vectors)} seeds)")
            else:
                logger.warning("No safe centroid could be computed!")
                    
        except Exception as e:
            logger.error(f"Failed to compute centroids: {e}")

    def check_similarity(self, input_embedding: np.ndarray) -> Tuple[float, str]:
        """
        Contrastive similarity check.
        Computes: max(danger_similarity) - safe_similarity
        Returns: (contrastive_score, concept_name)
        
        A positive contrastive score means the input is closer to danger than to safety.
        The higher the score, the more confident we are.
        """
        if not self.danger_centroids or input_embedding is None or self.safe_centroid is None:
            return 0.0, "None"

        input_vec = self._normalize(input_embedding)

        # Calculate safe similarity
        safe_sim = float(np.dot(input_vec, self.safe_centroid))

        # Calculate max danger similarity
        max_danger_sim = 0.0
        max_concept = "None"

        for concept, centroid in self.danger_centroids.items():
            sim = float(np.dot(input_vec, centroid))
            if sim > max_danger_sim:
                max_danger_sim = sim
                max_concept = concept

        # Contrastive Score: How much MORE similar to danger than to safe
        contrastive_score = max_danger_sim - safe_sim
        
        return contrastive_score, max_concept
