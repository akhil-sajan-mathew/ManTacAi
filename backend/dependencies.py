"""
FastAPI dependency injection providers for ManTacAi.
Centralizes model/engine initialization and provides them via Depends().
"""
import os
from functools import lru_cache

from inference.model import ManipulationModel
from inference.semantic_engine import SemanticAnalyzer
from utils.context_engine import ContextEngine

base_dir = os.path.dirname(os.path.abspath(__file__))


@lru_cache()
def get_detector_model() -> ManipulationModel:
    """Load and cache the manipulation detection model (singleton)."""
    model_path = os.path.join(base_dir, "manipulation_tactic_detector_model")
    try:
        model = ManipulationModel(model_path=model_path)
        print("✅ Model Loaded Successfully")
        return model
    except Exception as e:
        print(f"❌ FATAL: Custom model failed to load. Error: {e}")
        raise RuntimeError(
            f"ManTacAi requires the custom manipulation tactic model at {model_path}. "
            f"The default emotion model has incompatible labels and cannot be used as a fallback."
        ) from e


@lru_cache()
def get_semantic_analyzer() -> SemanticAnalyzer:
    """Load and cache the semantic analyzer with computed centroids (singleton)."""
    analyzer = SemanticAnalyzer()
    analyzer.compute_centroids(get_detector_model())
    return analyzer


def get_context_engine() -> ContextEngine:
    """Get the global persistent context engine."""
    return _global_context_engine


def get_stateless_context_engine() -> ContextEngine:
    """Get a fresh, non-persistent context engine for stateless requests."""
    return ContextEngine(persistence_file=None)


# Initialize global context engine at import time
_context_path = os.path.join(base_dir, "context_state.json")
_global_context_engine = ContextEngine(persistence_file=_context_path)
