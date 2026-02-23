#!/usr/bin/env python3
"""
Full Pipeline Execution Script

This script runs the complete end-to-end training and evaluation pipeline
for the manipulation detection model.

Usage:
    python scripts/run_full_pipeline.py --config config/full_pipeline_config.yaml
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
import torch
from datetime import datetime
import yaml

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.model_utils import ModelConfig
from src.data.data_loaders import create_data_loaders_from_config
from src.training.trainer import ModelTrainer
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.visualization import EvaluationVisualizer
from src.inference.predictor import ManipulationPredictor
from src.deployment.model_export import ModelExporter, ModelVersionManager
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('full_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FullPipelineRunner:
    """Runs the complete training and evaluation pipeline."""
    
    def __init__(self, config_path: str, output_dir: str = "pipeline_results"):
        """Initialize pipeline runner.
        
        Args:
            config_path: Path to pipeline configuration file
            output_dir: Output directory for results
        """
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup directories
        self.models_dir = self.output_dir / 'models'
        self.results_dir = self.output_dir / 'results'
        self.logs_dir = self.output_dir / 'logs'
        
        for dir_path in [self.models_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        logger.info(f"Pipeline initialized with output directory: {self.output_dir}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load pipeline configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        logger.info(f"Loaded configuration from: {config_path}")
        return config
    
    def run_training(self) -> str:
        """Run model training.
        
        Returns:
            Path to best model checkpoint
        """
        logger.info("Starting model training...")
        
        # Create model config
        model_config = ModelConfig.from_dict(self.config['model'])
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders_from_config(self.config['data'])
        
        # Create trainer
        trainer = ModelTrainer(
            config=model_config,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=str(self.models_dir),
            **self.config.get('training', {})
        )
        
        # Train model
        training_results = trainer.train()
        
        # Save training results
        results_path = self.results_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2)
        
        logger.info(f"Training completed. Results saved to: {results_path}")
        logger.info(f"Best model saved to: {training_results['best_model_path']}")
        
        return training_results['best_model_path']
    
    def run_evaluation(self, model_path: str) -> dict:
        """Run comprehensive model evaluation.
        
        Args:
            model_path: Path to trained model
            
        Returns:
            Evaluation results dictionary
        """
        logger.info("Starting model evaluation...")
        
        # Load model and tokenizer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        model_config = ModelConfig.from_dict(self.config['model'])
        
        from src.models.model_utils import ModelFactory
        model = ModelFactory.create_model(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders_from_config(self.config['data'])
        
        # Create evaluator
        evaluator = ModelEvaluator(model, device)
        
        # Evaluate on all datasets
        evaluation_results = {}
        
        datasets = {
            'train': train_loader,
            'validation': val_loader,
            'test': test_loader
        }
        
        for dataset_name, data_loader in datasets.items():
            logger.info(f"Evaluating on {dataset_name} set...")
            results = evaluator.evaluate_model(data_loader, return_predictions=True)
            evaluation_results[dataset_name] = results
            
            logger.info(f"{dataset_name} Results:")
            logger.info(f"  Accuracy: {results['basic_metrics']['accuracy']:.4f}")
            logger.info(f"  F1-Score: {results['basic_metrics']['f1_macro']:.4f}")
            logger.info(f"  Manipulation Accuracy: {results['manipulation_metrics']['manipulation_accuracy']:.4f}")
        
        # Save evaluation results
        eval_results_path = self.results_dir / 'evaluation_results.json'
        
        # Convert numpy arrays for JSON serialization
        def convert_for_json(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        json_results = convert_for_json(evaluation_results)
        
        with open(eval_results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Evaluation completed. Results saved to: {eval_results_path}")
        
        return evaluation_results
    
    def generate_visualizations(self, evaluation_results: dict, training_history: dict = None):
        """Generate evaluation visualizations.
        
        Args:
            evaluation_results: Results from evaluation
            training_history: Optional training history
        """
        logger.info("Generating visualizations...")
        
        plots_dir = self.results_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        visualizer = EvaluationVisualizer(save_dir=str(plots_dir))
        
        # Generate plots for each dataset
        for dataset_name, results in evaluation_results.items():
            logger.info(f"Creating plots for {dataset_name} set...")
            
            figures = visualizer.create_evaluation_report(
                results,
                training_history,
                save_name=f"{dataset_name}_evaluation"
            )
            
            logger.info(f"Generated {len(figures)} plots for {dataset_name}")
        
        logger.info(f"Visualizations saved to: {plots_dir}")
    
    def test_inference_performance(self, model_path: str) -> dict:
        """Test inference performance.
        
        Args:
            model_path: Path to trained model
            
        Returns:
            Performance test results
        """
        logger.info("Testing inference performance...")
        
        # Load model and tokenizer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        checkpoint = torch.load(model_path, map_location=device)
        model_config = ModelConfig.from_dict(self.config['model'])
        
        from src.models.model_utils import ModelFactory
        model = ModelFactory.create_model(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
        
        # Create predictor
        predictor = ManipulationPredictor(model, tokenizer, device)
        
        # Test texts
        test_texts = [
            "I think we should consider this option because it benefits everyone.",
            "You're being way too sensitive about this, that never actually happened.",
            "If you really cared about me, you would do this without me having to ask.",
            "I'm not going to discuss this topic anymore, the conversation is over.",
            "You're absolutely amazing and perfect in every way, I love everything about you.",
            "Maybe you should think about what you did to cause this situation.",
            "Everyone else agrees with me on this, you're the only one who thinks differently.",
            "Fine, whatever you want, I guess my opinion doesn't matter anyway.",
            "This is a reasonable request that I'm making in a respectful manner.",
            "You always do this, you never listen to what I'm trying to tell you."
        ]
        
        # Performance testing
        import time
        
        # Warm up
        for _ in range(5):
            predictor.predict_single(test_texts[0])
        
        # Time individual predictions
        inference_times = []
        predictions = []
        
        for text in test_texts:
            start_time = time.time()
            result = predictor.predict_single(text, return_probabilities=True)
            inference_time = time.time() - start_time
            
            inference_times.append(inference_time)
            predictions.append({
                'text': text,
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence'],
                'is_manipulation': result['is_manipulation'],
                'inference_time_ms': inference_time * 1000
            })
        
        # Calculate statistics
        avg_inference_time = sum(inference_times) / len(inference_times)
        throughput = 1.0 / avg_inference_time
        
        performance_results = {
            'average_inference_time_ms': avg_inference_time * 1000,
            'throughput_samples_per_second': throughput,
            'total_samples_tested': len(test_texts),
            'device': str(device),
            'sample_predictions': predictions
        }
        
        # Save performance results
        perf_results_path = self.results_dir / 'inference_performance.json'
        with open(perf_results_path, 'w') as f:
            json.dump(performance_results, f, indent=2)
        
        logger.info(f"Inference performance test completed:")
        logger.info(f"  Average inference time: {avg_inference_time*1000:.2f}ms")
        logger.info(f"  Throughput: {throughput:.1f} samples/second")
        logger.info(f"  Results saved to: {perf_results_path}")
        
        return performance_results
    
    def validate_accuracy_requirements(self, evaluation_results: dict) -> bool:
        """Validate that model meets accuracy requirements.
        
        Args:
            evaluation_results: Evaluation results
            
        Returns:
            True if requirements are met
        """
        logger.info("Validating accuracy requirements...")
        
        # Get test set results
        test_results = evaluation_results.get('test', {})
        basic_metrics = test_results.get('basic_metrics', {})
        manip_metrics = test_results.get('manipulation_metrics', {})
        
        # Check requirements (85%+ overall accuracy)
        overall_accuracy = basic_metrics.get('accuracy', 0.0)
        manipulation_accuracy = manip_metrics.get('manipulation_accuracy', 0.0)
        
        requirements_met = True
        
        if overall_accuracy < 0.85:
            logger.warning(f"Overall accuracy ({overall_accuracy:.3f}) below requirement (0.85)")
            requirements_met = False
        else:
            logger.info(f"✓ Overall accuracy requirement met: {overall_accuracy:.3f}")
        
        if manipulation_accuracy < 0.80:
            logger.warning(f"Manipulation detection accuracy ({manipulation_accuracy:.3f}) below target (0.80)")
        else:
            logger.info(f"✓ Manipulation detection accuracy: {manipulation_accuracy:.3f}")
        
        return requirements_met
    
    def create_deployment_package(self, model_path: str) -> str:
        """Create deployment package.
        
        Args:
            model_path: Path to trained model
            
        Returns:
            Path to deployment package
        """
        logger.info("Creating deployment package...")
        
        # Load model and tokenizer
        checkpoint = torch.load(model_path, map_location='cpu')
        model_config = ModelConfig.from_dict(self.config['model'])
        
        from src.models.model_utils import ModelFactory
        model = ModelFactory.create_model(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
        
        # Create exporter
        exporter = ModelExporter(model, tokenizer, model_config.to_dict())
        
        # Create deployment package
        deployment_dir = self.output_dir / 'deployment'
        package_path = exporter.create_deployment_package(str(deployment_dir))
        
        logger.info(f"Deployment package created at: {package_path}")
        
        return package_path
    
    def generate_final_report(self, evaluation_results: dict, performance_results: dict, 
                            requirements_met: bool, model_path: str, deployment_path: str):
        """Generate final comprehensive report.
        
        Args:
            evaluation_results: Evaluation results
            performance_results: Performance test results
            requirements_met: Whether accuracy requirements were met
            model_path: Path to trained model
            deployment_path: Path to deployment package
        """
        logger.info("Generating final report...")
        
        # Get test results
        test_results = evaluation_results.get('test', {})
        basic_metrics = test_results.get('basic_metrics', {})
        manip_metrics = test_results.get('manipulation_metrics', {})
        
        report_lines = []
        report_lines.append("MANIPULATION DETECTION MODEL - FULL PIPELINE REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"Pipeline Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Configuration: {self.config_path}")
        report_lines.append("")
        
        # Model Information
        report_lines.append("MODEL INFORMATION")
        report_lines.append("-" * 20)
        report_lines.append(f"Model Type: {self.config['model']['model_name']}")
        report_lines.append(f"Number of Classes: {self.config['model']['num_classes']}")
        report_lines.append(f"Max Sequence Length: {self.config['model']['max_length']}")
        report_lines.append(f"Model Path: {model_path}")
        report_lines.append("")
        
        # Performance Results
        report_lines.append("PERFORMANCE RESULTS")
        report_lines.append("-" * 20)
        report_lines.append(f"Overall Accuracy: {basic_metrics.get('accuracy', 0):.4f}")
        report_lines.append(f"F1-Score (Macro): {basic_metrics.get('f1_macro', 0):.4f}")
        report_lines.append(f"F1-Score (Weighted): {basic_metrics.get('f1_weighted', 0):.4f}")
        report_lines.append(f"Manipulation Detection Accuracy: {manip_metrics.get('manipulation_accuracy', 0):.4f}")
        report_lines.append(f"Manipulation Precision: {manip_metrics.get('manipulation_precision', 0):.4f}")
        report_lines.append(f"Manipulation Recall: {manip_metrics.get('manipulation_recall', 0):.4f}")
        report_lines.append("")
        
        # Inference Performance
        report_lines.append("INFERENCE PERFORMANCE")
        report_lines.append("-" * 20)
        report_lines.append(f"Average Inference Time: {performance_results['average_inference_time_ms']:.2f}ms")
        report_lines.append(f"Throughput: {performance_results['throughput_samples_per_second']:.1f} samples/second")
        report_lines.append(f"Device: {performance_results['device']}")
        report_lines.append("")
        
        # Requirements Validation
        report_lines.append("REQUIREMENTS VALIDATION")
        report_lines.append("-" * 25)
        status = "✓ PASSED" if requirements_met else "✗ FAILED"
        report_lines.append(f"Accuracy Requirements (85%+): {status}")
        report_lines.append(f"Current Accuracy: {basic_metrics.get('accuracy', 0):.4f}")
        report_lines.append("")
        
        # Deployment Information
        report_lines.append("DEPLOYMENT INFORMATION")
        report_lines.append("-" * 22)
        report_lines.append(f"Deployment Package: {deployment_path}")
        report_lines.append("Available Formats:")
        report_lines.append("  - PyTorch (.pt)")
        report_lines.append("  - ONNX (.onnx)")
        report_lines.append("  - Quantized (.pt)")
        report_lines.append("")
        
        # Files Generated
        report_lines.append("FILES GENERATED")
        report_lines.append("-" * 15)
        report_lines.append(f"Models: {self.models_dir}")
        report_lines.append(f"Evaluation Results: {self.results_dir}")
        report_lines.append(f"Visualizations: {self.results_dir / 'plots'}")
        report_lines.append(f"Deployment Package: {deployment_path}")
        report_lines.append("")
        
        # Next Steps
        report_lines.append("NEXT STEPS")
        report_lines.append("-" * 10)
        if requirements_met:
            report_lines.append("✓ Model is ready for production deployment")
            report_lines.append("✓ Use deployment package for integration")
            report_lines.append("✓ Monitor performance in production")
        else:
            report_lines.append("⚠ Model requires further training or tuning")
            report_lines.append("⚠ Review evaluation results and adjust hyperparameters")
            report_lines.append("⚠ Consider data augmentation or additional training data")
        
        report_lines.append("")
        report_lines.append("=" * 70)
        
        # Save report
        report_path = self.output_dir / 'FINAL_REPORT.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Also print to console
        print("\n" + '\n'.join(report_lines))
        
        logger.info(f"Final report saved to: {report_path}")
    
    def run_full_pipeline(self):
        """Run the complete pipeline."""
        logger.info("Starting full pipeline execution...")
        
        try:
            # Step 1: Training
            model_path = self.run_training()
            
            # Step 2: Evaluation
            evaluation_results = self.run_evaluation(model_path)
            
            # Step 3: Generate visualizations
            self.generate_visualizations(evaluation_results)
            
            # Step 4: Test inference performance
            performance_results = self.test_inference_performance(model_path)
            
            # Step 5: Validate requirements
            requirements_met = self.validate_accuracy_requirements(evaluation_results)
            
            # Step 6: Create deployment package
            deployment_path = self.create_deployment_package(model_path)
            
            # Step 7: Generate final report
            self.generate_final_report(
                evaluation_results, 
                performance_results, 
                requirements_met, 
                model_path, 
                deployment_path
            )
            
            logger.info("Full pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False


def create_default_config(config_path: str):
    """Create a default pipeline configuration file.
    
    Args:
        config_path: Path to save configuration
    """
    default_config = {
        "model": {
            "model_name": "distilbert-base-uncased",
            "num_classes": 11,
            "max_length": 512,
            "dropout_rate": 0.1,
            "hidden_size": 768
        },
        "data": {
            "train_file": "data/processed/train.json",
            "val_file": "data/processed/validation.json",
            "test_file": "data/processed/test.json",
            "batch_size": 16,
            "max_length": 512,
            "num_workers": 4
        },
        "training": {
            "num_epochs": 10,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_steps": 500,
            "gradient_clip_norm": 1.0,
            "early_stopping_patience": 3,
            "save_strategy": "best",
            "evaluation_strategy": "epoch",
            "logging_steps": 100
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, indent=2)
    
    logger.info(f"Default configuration saved to: {config_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run Full Training and Evaluation Pipeline')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline configuration file')
    parser.add_argument('--output-dir', type=str, default='pipeline_results',
                       help='Output directory for results')
    parser.add_argument('--create-default-config', type=str,
                       help='Create default configuration file at specified path')
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_default_config:
        create_default_config(args.create_default_config)
        return
    
    logger.info("Starting full pipeline execution")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Run pipeline
    runner = FullPipelineRunner(args.config, args.output_dir)
    success = runner.run_full_pipeline()
    
    if success:
        logger.info("Pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()