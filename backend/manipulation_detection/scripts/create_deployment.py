#!/usr/bin/env python3
"""
Deployment Package Creation Script

This script creates a complete deployment package for the manipulation detection model.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
import torch
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.deployment.model_export import ModelExporter, ModelVersionManager
from src.models.model_utils import ModelConfig
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main deployment package creation function."""
    parser = argparse.ArgumentParser(description='Create Deployment Package')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config-path', type=str, required=True, help='Path to model configuration')
    parser.add_argument('--output-dir', type=str, default='deployment', help='Output directory')
    
    args = parser.parse_args()
    
    logger.info("Creating deployment package")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Output: {args.output_dir}")
    
    try:
        # Create output directory
        deployment_dir = Path(args.output_dir)
        deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        if args.config_path.endswith('.yaml'):
            config = ModelConfig.from_yaml(args.config_path)
        else:
            config = ModelConfig.from_json(args.config_path)
        
        # Load model
        checkpoint = torch.load(args.model_path, map_location='cpu')
        from src.models.model_utils import ModelFactory
        model = ModelFactory.create_model(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Create exporter
        exporter = ModelExporter(model, tokenizer, config.to_dict())
        
        # Create deployment package
        package_path = exporter.create_deployment_package(str(deployment_dir))
        
        logger.info(f"Deployment package created at: {package_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)