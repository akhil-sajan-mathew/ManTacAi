#!/usr/bin/env python3
"""
Evaluation and Analysis Script for Manipulation Detection Model

This script evaluates a trained model and generates comprehensive analysis reports.

Usage:
    python scripts/evaluate.py --model-path models/best_model.pt --config-path config/model_config.yaml
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.model_utils import ModelConfig, load_model_from_config
from src.data.data_loaders import create_data_loaders_from_config
from src.evaluation.metrics import ModelEvaluator, MetricsCalculator, ManipulationMetrics
from src.evaluation.visualization import EvaluationVisualizer, ReportGenerator
from src.evaluation.analysis import ModelAnalyzer, ErrorAnalyzer
from src.inference.predictor import ManipulationPredictor
from src.inference.deployment import DeploymentInference
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_output_directories(output_dir: str):
    """Create output directories for evaluation results.
    
    Args:
        output_dir: Base output directory
    """
    directories = [
        'plots',
        'reports', 
        'analysis',
        'predictions',
        'embeddings'
    ]
    
    for directory in directories:
        dir_path = os.path.join(output_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")


def load_model_and_tokenizer(model_path: str, config_path: str, device: str):
    """Load model and tokenizer for evaluation.
    
    Args:
        model_path: Path to model checkpoint or directory
        config_path: Path to configuration file
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer, config)
    """
    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Loading config from: {config_path}")
    
    # Load model
    if model_path.endswith('.pt'):
        # Load from checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = ModelConfig.from_yaml(config_path)
        else:
            config = ModelConfig.from_json(config_path)
        
        from src.models.model_utils import ModelFactory
        model = ModelFactory.create_model(config)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Load from directory
        model = load_model_from_config(config_path, checkpoint_path=model_path)
        config = ModelConfig.from_yaml(config_path) if config_path.endswith('.yaml') else ModelConfig.from_json(config_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully on {device}")
    logger.info(f"Model info: {model.get_model_info()}")
    
    return model, tokenizer, config


def evaluate_on_dataset(model, data_loader, device: str, dataset_name: str):
    """Evaluate model on a dataset.
    
    Args:
        model: Trained model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        dataset_name: Name of the dataset (for logging)
        
    Returns:
        Evaluation results dictionary
    """
    logger.info(f"Evaluating on {dataset_name} set ({len(data_loader)} batches)...")
    
    evaluator = ModelEvaluator(model, device)
    results = evaluator.evaluate_model(data_loader, return_predictions=True)
    
    # Print summary
    logger.info(f"{dataset_name} Results:")
    logger.info(f"  Accuracy: {results['basic_metrics']['accuracy']:.4f}")
    logger.info(f"  F1-Score (Macro): {results['basic_metrics']['f1_macro']:.4f}")
    logger.info(f"  Manipulation Accuracy: {results['manipulation_metrics']['manipulation_accuracy']:.4f}")
    
    return results


def perform_error_analysis(model, tokenizer, test_results, output_dir: str):
    """Perform detailed error analysis.
    
    Args:
        model: Trained model
        tokenizer: Model tokenizer
        test_results: Test evaluation results
        output_dir: Output directory for analysis
    """
    logger.info("Performing error analysis...")
    
    # Extract predictions and texts (we'll need to modify this based on actual data structure)
    y_true = test_results['predictions']['y_true']
    y_pred = test_results['predictions']['y_pred']
    y_prob = test_results['predictions']['y_prob']
    
    # For now, create dummy texts - in practice, you'd pass the actual texts
    texts = [f"Sample text {i}" for i in range(len(y_true))]
    
    # Initialize analyzers
    model_analyzer = ModelAnalyzer(model, tokenizer, device='cpu')
    error_analyzer = ErrorAnalyzer()
    
    # Misclassification analysis
    misclass_analysis = model_analyzer.analyze_misclassifications(y_true, y_pred, texts, y_prob)
    
    # Class performance analysis
    class_analysis = model_analyzer.analyze_class_performance(y_true, y_pred, y_prob)
    
    # Find difficult examples
    difficult_examples = model_analyzer.find_difficult_examples(y_true, y_prob, texts, top_k=20)
    
    # Prediction patterns
    pattern_analysis = model_analyzer.analyze_prediction_patterns(y_true, y_pred, texts)
    
    # Systematic errors
    systematic_errors = error_analyzer.analyze_systematic_errors(y_true, y_pred, texts)
    
    # Aggregate results
    analysis_results = {
        'misclassification': misclass_analysis,
        'class_performance': class_analysis,
        'difficult_examples': difficult_examples,
        'prediction_patterns': pattern_analysis,
        'systematic_errors': systematic_errors
    }
    
    # Save analysis results
    analysis_path = os.path.join(output_dir, 'analysis', 'error_analysis.json')
    with open(analysis_path, 'w') as f:
        # Convert numpy types to python types for serialization
        def convert_numpy(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                np.int16, np.int32, np.int64, np.uint8,
                np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, 
                np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return obj
            
        json.dump(analysis_results, f, indent=4, default=convert_numpy)
        
    logger.info(f"Error analysis saved to: {analysis_path}")
    
    return analysis_results


def generate_visualizations(results_dict, output_dir: str, training_history=None):
    """Generate comprehensive visualizations.
    
    Args:
        results_dict: Dictionary of evaluation results for different datasets
        output_dir: Output directory for plots
        training_history: Optional training history for training curves
    """
    logger.info("Generating visualizations...")
    
    visualizer = EvaluationVisualizer(save_dir=os.path.join(output_dir, 'plots'))
    
    # Generate plots for each dataset
    for dataset_name, results in results_dict.items():
        logger.info(f"Creating plots for {dataset_name} set...")
        
        figures = visualizer.create_evaluation_report(
            results,
            training_history,
            save_name=f"{dataset_name}_evaluation"
        )
        
        logger.info(f"Generated {len(figures)} plots for {dataset_name}")
    
    # Create comparison plots if multiple datasets
    if len(results_dict) > 1:
        logger.info("Creating comparison visualizations...")
        
        # Compare basic metrics across datasets
        comparison_data = {}
        for dataset_name, results in results_dict.items():
            comparison_data[dataset_name] = results['basic_metrics']
        
        # Save comparison data
        comparison_path = os.path.join(output_dir, 'analysis', 'dataset_comparison.json')
        with open(comparison_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        logger.info(f"Dataset comparison saved to: {comparison_path}")


def generate_reports(results_dict, model_info, output_dir: str, analysis_results=None):
    """Generate comprehensive text reports.
    
    Args:
        results_dict: Dictionary of evaluation results
        model_info: Model information dictionary
        output_dir: Output directory for reports
        analysis_results: Optional error analysis results
    """
    logger.info("Generating reports...")
    
    report_generator = ReportGenerator(save_dir=os.path.join(output_dir, 'reports'))
    
    # Generate report for each dataset
    for dataset_name, results in results_dict.items():
        report_path = report_generator.generate_text_report(
            results,
            model_info,
            save_name=f"{dataset_name}_evaluation_report.txt"
        )
        logger.info(f"Generated report for {dataset_name}: {report_path}")
    
    # Generate comprehensive summary report
    summary_report = []
    summary_report.append("MANIPULATION DETECTION MODEL - COMPREHENSIVE EVALUATION")
    summary_report.append("=" * 70)
    summary_report.append(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_report.append("")
    
    # Model information
    summary_report.append("MODEL INFORMATION")
    summary_report.append("-" * 20)
    for key, value in model_info.items():
        summary_report.append(f"{key}: {value}")
    summary_report.append("")
    
    # Results summary
    summary_report.append("EVALUATION SUMMARY")
    summary_report.append("-" * 20)
    
    for dataset_name, results in results_dict.items():
        summary_report.append(f"\n{dataset_name.upper()} SET:")
        basic_metrics = results['basic_metrics']
        manip_metrics = results['manipulation_metrics']
        
        summary_report.append(f"  Overall Accuracy: {basic_metrics['accuracy']:.4f}")
        summary_report.append(f"  F1-Score (Macro): {basic_metrics['f1_macro']:.4f}")
        summary_report.append(f"  F1-Score (Weighted): {basic_metrics['f1_weighted']:.4f}")
        summary_report.append(f"  Manipulation Detection Accuracy: {manip_metrics['manipulation_accuracy']:.4f}")
        summary_report.append(f"  Manipulation Precision: {manip_metrics['manipulation_precision']:.4f}")
        summary_report.append(f"  Manipulation Recall: {manip_metrics['manipulation_recall']:.4f}")
    
    # Error analysis summary
    if analysis_results:
        summary_report.append("\nERROR ANALYSIS HIGHLIGHTS")
        summary_report.append("-" * 25)
        
        misclass = analysis_results['misclassification']
        summary_report.append(f"Total Misclassifications: {misclass['total_misclassified']}")
        summary_report.append(f"Misclassification Rate: {misclass['misclassification_rate']:.4f}")
        
        if 'most_confused_classes' in misclass and misclass['most_confused_classes']:
            summary_report.append("\nMost Confused Class Pairs:")
            for (true_class, pred_class), count in misclass['most_confused_classes'][:5]:
                summary_report.append(f"  {true_class} → {pred_class}: {count} errors")
    
    summary_report.append("")
    summary_report.append("=" * 70)
    
    # Save summary report
    summary_path = os.path.join(output_dir, 'reports', 'comprehensive_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_report))
    
    logger.info(f"Comprehensive summary saved to: {summary_path}")


def test_inference_performance(model, tokenizer, device: str, output_dir: str):
    """Test inference performance and create sample predictions.
    
    Args:
        model: Trained model
        tokenizer: Model tokenizer
        device: Device to run inference on
        output_dir: Output directory
    """
    logger.info("Testing inference performance...")
    
    # Create predictor
    predictor = ManipulationPredictor(model, tokenizer, device)
    
    # Sample texts for testing
    test_texts = [
        "I think we should consider this option because it benefits everyone involved.",
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
    
    # Test single predictions
    predictions = []
    inference_times = []
    
    for i, text in enumerate(test_texts):
        import time
        start_time = time.time()
        
        result = predictor.predict_single(text, return_probabilities=True, top_k=3)
        
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        predictions.append({
            'text': text,
            'predicted_class': result['predicted_class'],
            'confidence': result['confidence'],
            'is_manipulation': result['is_manipulation'],
            'top_3_predictions': result.get('top_k_predictions', []),
            'inference_time_ms': inference_time * 1000
        })
        
        logger.info(f"Sample {i+1}: {result['predicted_class']} (conf: {result['confidence']:.3f})")
    
    # Performance statistics
    avg_inference_time = sum(inference_times) / len(inference_times)
    throughput = 1.0 / avg_inference_time
    
    performance_stats = {
        'average_inference_time_ms': avg_inference_time * 1000,
        'throughput_samples_per_second': throughput,
        'total_samples': len(test_texts),
        'device': str(device)
    }
    
    # Save predictions and performance
    predictions_data = {
        'sample_predictions': predictions,
        'performance_statistics': performance_stats,
        'timestamp': datetime.now().isoformat()
    }
    
    predictions_path = os.path.join(output_dir, 'predictions', 'sample_predictions.json')
    with open(predictions_path, 'w') as f:
        json.dump(predictions_data, f, indent=2)
    
    logger.info(f"Sample predictions saved to: {predictions_path}")
    logger.info(f"Average inference time: {avg_inference_time*1000:.2f}ms")
    logger.info(f"Throughput: {throughput:.1f} samples/second")
    
    return predictions_data


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Manipulation Detection Model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to model checkpoint or directory')
    parser.add_argument('--config-path', type=str, required=True,
                       help='Path to model configuration file')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--datasets', type=str, nargs='+', default=['test'],
                       choices=['train', 'validation', 'test', 'all'],
                       help='Datasets to evaluate on')
    parser.add_argument('--skip-error-analysis', action='store_true',
                       help='Skip detailed error analysis')
    parser.add_argument('--skip-visualizations', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--training-history', type=str,
                       help='Path to training history JSON file')
    
    args = parser.parse_args()
    
    logger.info("Starting model evaluation")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Config: {args.config_path}")
    logger.info(f"Output: {args.output_dir}")
    
    # Setup output directories
    setup_output_directories(args.output_dir)
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    try:
        # Load model and tokenizer
        model, tokenizer, config = load_model_and_tokenizer(args.model_path, args.config_path, device)
        
        # Load data loaders
        logger.info("Loading data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders_from_config(config.to_dict())
        
        # Determine which datasets to evaluate
        if 'all' in args.datasets:
            datasets_to_eval = ['train', 'validation', 'test']
        else:
            datasets_to_eval = args.datasets
        
        # Map dataset names to loaders
        loader_map = {
            'train': train_loader,
            'validation': val_loader,
            'test': test_loader
        }
        
        # Evaluate on specified datasets
        results_dict = {}
        for dataset_name in datasets_to_eval:
            if dataset_name in loader_map:
                results = evaluate_on_dataset(model, loader_map[dataset_name], device, dataset_name)
                results_dict[dataset_name] = results
        
        # Get model info
        model_info = model.get_model_info()
        model_info.update({
            'evaluation_date': datetime.now().isoformat(),
            'device_used': device,
            'datasets_evaluated': list(results_dict.keys())
        })
        
        # Perform error analysis on test set
        analysis_results = None
        if not args.skip_error_analysis and 'test' in results_dict:
            analysis_results = perform_error_analysis(model, tokenizer, results_dict['test'], args.output_dir)
        
        # Load training history if provided
        training_history = None
        if args.training_history and os.path.exists(args.training_history):
            with open(args.training_history, 'r') as f:
                training_history = json.load(f)
            logger.info(f"Loaded training history from: {args.training_history}")
        
        # Generate visualizations
        if not args.skip_visualizations:
            generate_visualizations(results_dict, args.output_dir, training_history)
        
        # Generate reports
        generate_reports(results_dict, model_info, args.output_dir, analysis_results)
        
        # Test inference performance
        inference_results = test_inference_performance(model, tokenizer, device, args.output_dir)
        
        # Save complete evaluation results
        complete_results = {
            'model_info': model_info,
            'evaluation_results': results_dict,
            'error_analysis': analysis_results,
            'inference_performance': inference_results,
            'evaluation_config': {
                'model_path': args.model_path,
                'config_path': args.config_path,
                'device': device,
                'datasets_evaluated': datasets_to_eval
            }
        }
        
        results_path = os.path.join(args.output_dir, 'complete_evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(complete_results, f, indent=2)
        
        logger.info(f"Complete evaluation results saved to: {results_path}")
        
        # Print final summary
        logger.info("\n" + "="*60)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        for dataset_name, results in results_dict.items():
            logger.info(f"{dataset_name.upper()} SET RESULTS:")
            logger.info(f"  Accuracy: {results['basic_metrics']['accuracy']:.4f}")
            logger.info(f"  F1-Score: {results['basic_metrics']['f1_macro']:.4f}")
            logger.info(f"  Manipulation Detection: {results['manipulation_metrics']['manipulation_accuracy']:.4f}")
        
        logger.info(f"\nResults saved to: {args.output_dir}")
        logger.info("Check the following directories:")
        logger.info("  - plots/ - Evaluation visualizations")
        logger.info("  - reports/ - Detailed text reports")
        logger.info("  - analysis/ - Error analysis results")
        logger.info("  - predictions/ - Sample predictions")
        
        return True
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)