"""Configuration utilities for the manipulation detection model."""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str = "config/model_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def get_label_mapping() -> Dict[int, str]:
    """Get the label mapping for manipulation tactics.
    
    Returns:
        Dictionary mapping label IDs to tactic names
    """
    return {
        0: "ethical_persuasion",
        1: "gaslighting", 
        2: "guilt_tripping",
        3: "deflection",
        4: "stonewalling",
        5: "belittling_ridicule",
        6: "love_bombing",
        7: "threatening_intimidation", 
        8: "passive_aggression",
        9: "appeal_to_emotion",
        10: "whataboutism"
    }


def get_reverse_label_mapping() -> Dict[str, int]:
    """Get reverse label mapping from tactic names to IDs.
    
    Returns:
        Dictionary mapping tactic names to label IDs
    """
    label_map = get_label_mapping()
    return {v: k for k, v in label_map.items()}