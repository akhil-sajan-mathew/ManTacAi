#!/usr/bin/env python3
"""
Inference Demonstration Script for Manipulation Detection Model

This script demonstrates how to use the trained model for predictions.
Supports single text prediction, batch processing, and interactive mode.

Usage:
    # Single prediction
    python scripts/predict.py --model-path models/best_model.pt --text "Your text here"
    
    # Batch prediction from file
    python scripts/predict.py --model-path models/best_model.pt --input-file texts.txt
    
    # Interactive mode
    python scripts/predict.py --model-path models/best_model.pt --interactive
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
import torch
import pandas as pd
from datetime import datetime
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.inference.deployment import DeploymentInference, InferenceAPI
from src.inference.predictor import ManipulationPredictor
from src.inference.batch_processor import EnhancedBatchProcessor, ResultsAnalyzer
from src.models.model_utils import ModelConfig
from src.utils.config import get_label_mapping
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_inference_model(model_path: str, config_path: str = None, device: str = 'auto'):
    """Load model for inference.
    
    Args:
        model_path: Path to model checkpoint or directory
        config_path: Path to configuration file (optional)
        device: Device to use for inference
        
    Returns:
        DeploymentInference instance
    """
    logger.info(f"Loading model from: {model_path}")
    
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Using device: {device}")
    
    # Load model
    inference = DeploymentInference(
        model_path=model_path,
        config_path=config_path,
        device=device,
        optimize_for_inference=True
    )
    
    # Test model health
    health = inference.health_check()
    if health['status'] != 'healthy':
        logger.warning(f"Model health check failed: {health}")
    else:
        logger.info("Model loaded successfully and passed health check")
    
    return inference


def predict_single_text(inference: DeploymentInference, text: str, verbose: bool = True):
    """Predict manipulation tactic for a single text.
    
    Args:
        inference: DeploymentInference instance
        text: Input text to analyze
        verbose: Whether to print detailed results
        
    Returns:
        Prediction result dictionary
    """
    result = inference.predict(
        text=text,
        return_probabilities=True,
        return_top_k=3
    )
    
    if verbose:
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"Text: '{text}'")
        print(f"\nPredicted Class: {result['predicted_class'].replace('_', ' ').title()}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Is Manipulation: {'Yes' if result['is_manipulation'] else 'No'}")
        print(f"Inference Time: {result['inference_time_ms']:.1f}ms")
        
        if 'top_k_predictions' in result:
            print("\nTop 3 Predictions:")
            for i, pred in enumerate(result['top_k_predictions'], 1):
                class_name = pred['class'].replace('_', ' ').title()
                print(f"  {i}. {class_name}: {pred['probability']:.3f}")
        
        if 'class_probabilities' in result:
            print("\nAll Class Probabilities:")
            sorted_probs = sorted(result['class_probabilities'].items(), 
                                key=lambda x: x[1], reverse=True)
            for class_name, prob in sorted_probs:
                display_name = class_name.replace('_', ' ').title()
                print(f"  {display_name:25s}: {prob:.3f}")
        
        print("="*60)
    
    return result


def predict_batch_from_file(inference: DeploymentInference, input_file: str, output_file: str = None):
    """Predict manipulation tactics for texts from a file.
    
    Args:
        inference: DeploymentInference instance
        input_file: Path to input file (txt, csv, or json)
        output_file: Path to output file (optional)
        
    Returns:
        List of prediction results
    """
    logger.info(f"Processing batch file: {input_file}")
    
    # Create batch processor
    predictor = ManipulationPredictor(
        model=inference.model,
        tokenizer=inference.tokenizer,
        device=inference.device
    )
    
    processor = EnhancedBatchProcessor(predictor, batch_size=32)
    
    # Determine output file
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_predictions.json")
    
    # Process file
    start_time = time.time()
    stats = processor.process_file(
        input_file=input_file,
        output_file=output_file,
        text_column='text',
        include_probabilities=True,
        progress_bar=True
    )
    processing_time = time.time() - start_time
    
    logger.info(f"Batch processing completed in {processing_time:.2f} seconds")
    logger.info(f"Results saved to: {output_file}")
    
    # Print statistics
    print("\n" + "="*60)
    print("BATCH PROCESSING RESULTS")
    print("="*60)
    print(f"Total texts processed: {stats['total_texts']:,}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Throughput: {stats['texts_per_second']:.1f} texts/second")
    print(f"Manipulation detected: {stats['manipulation_detected']:,} ({stats['manipulation_rate']:.1%})")
    print(f"High confidence predictions: {stats['high_confidence_predictions']:,} ({stats['high_confidence_rate']:.1%})")
    print(f"Average confidence: {stats['average_confidence']:.3f}")
    
    print("\nClass Distribution:")
    for class_name, count in stats['class_distribution'].items():
        display_name = class_name.replace('_', ' ').title()
        percentage = count / stats['total_texts'] * 100
        print(f"  {display_name:25s}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nResults saved to: {output_file}")
    print("="*60)
    
    return stats


def interactive_mode(inference: DeploymentInference):
    """Run interactive prediction mode.
    
    Args:
        inference: DeploymentInference instance
    """
    print("\n" + "="*60)
    print("INTERACTIVE MANIPULATION DETECTION")
    print("="*60)
    print("Enter text to analyze for manipulation tactics.")
    print("Commands:")
    print("  'quit' or 'exit' - Exit interactive mode")
    print("  'help' - Show this help message")
    print("  'stats' - Show model statistics")
    print("  'examples' - Show example texts")
    print("="*60)
    
    # Get model info
    model_stats = inference.get_stats()
    
    while True:
        try:
            # Get user input
            text = input("\nEnter text (or command): ").strip()
            
            if not text:
                continue
            
            # Handle commands
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            elif text.lower() == 'help':
                print("\nCommands:")
                print("  'quit' or 'exit' - Exit interactive mode")
                print("  'help' - Show this help message")
                print("  'stats' - Show model statistics")
                print("  'examples' - Show example texts")
                continue
            
            elif text.lower() == 'stats':
                print("\nModel Statistics:")
                print(f"  Total predictions made: {model_stats['total_predictions']}")
                print(f"  Average inference time: {model_stats['average_inference_time']*1000:.2f}ms")
                print(f"  Predictions per second: {model_stats['predictions_per_second']:.1f}")
                print(f"  Device: {model_stats['model_info']['device']}")
                print(f"  Model parameters: {model_stats['model_info']['model_parameters']:,}")
                continue
            
            elif text.lower() == 'examples':
                examples = [
                    ("Ethical Persuasion", "I think we should consider this option because it benefits everyone."),
                    ("Gaslighting", "You're being too sensitive, that never actually happened."),
                    ("Guilt Tripping", "If you really cared about me, you would do this for me."),
                    ("Stonewalling", "I'm not going to discuss this anymore."),
                    ("Love Bombing", "You're absolutely perfect and amazing in every single way."),
                    ("Threatening", "You'll regret it if you don't do what I'm asking.")
                ]
                
                print("\nExample texts to try:")
                for category, example in examples:
                    print(f"  {category}: '{example}'")
                continue
            
            # Make prediction
            result = predict_single_text(inference, text, verbose=True)
            
            # Update stats
            model_stats = inference.get_stats()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            logger.error(f"Prediction error: {str(e)}")


def benchmark_performance(inference: DeploymentInference, num_samples: int = 100):
    """Benchmark inference performance.
    
    Args:
        inference: DeploymentInference instance
        num_samples: Number of samples to benchmark
    """
    logger.info(f"Benchmarking performance with {num_samples} samples...")
    
    # Generate test texts
    test_texts = [
        f"This is a benchmark test message number {i} for performance evaluation."
        for i in range(num_samples)
    ]
    
    # Warm up
    for _ in range(5):
        inference.predict(test_texts[0])
    
    # Benchmark single predictions
    single_times = []
    for text in test_texts[:min(50, num_samples)]:
        start_time = time.time()
        inference.predict(text)
        single_times.append(time.time() - start_time)
    
    # Benchmark batch predictions
    batch_start = time.time()
    batch_results = inference.predict_batch(test_texts, batch_size=32)
    batch_time = time.time() - batch_start
    
    # Calculate statistics
    avg_single_time = sum(single_times) / len(single_times)
    single_throughput = 1.0 / avg_single_time
    batch_throughput = len(test_texts) / batch_time
    
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK RESULTS")
    print("="*60)
    print(f"Single Prediction Performance:")
    print(f"  Average time: {avg_single_time*1000:.2f}ms")
    print(f"  Throughput: {single_throughput:.1f} predictions/second")
    
    print(f"\nBatch Prediction Performance:")
    print(f"  Total time: {batch_time:.2f}s for {len(test_texts)} samples")
    print(f"  Throughput: {batch_throughput:.1f} predictions/second")
    print(f"  Speedup: {batch_throughput/single_throughput:.1f}x faster than single predictions")
    
    print(f"\nModel Statistics:")
    stats = inference.get_stats()
    print(f"  Device: {stats['model_info']['device']}")
    print(f"  Model parameters: {stats['model_info']['model_parameters']:,}")
    print(f"  Total predictions made: {stats['total_predictions']}")
    print("="*60)
    
    return {
        'single_prediction_time_ms': avg_single_time * 1000,
        'single_throughput': single_throughput,
        'batch_time_seconds': batch_time,
        'batch_throughput': batch_throughput,
        'speedup_factor': batch_throughput / single_throughput
    }


def analyze_predictions(predictions_file: str):
    """Analyze predictions from a batch processing file.
    
    Args:
        predictions_file: Path to predictions JSON file
    """
    logger.info(f"Analyzing predictions from: {predictions_file}")
    
    # Load predictions
    with open(predictions_file, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        predictions = data
    else:
        predictions = data.get('predictions', data)
    
    # Analyze results
    analyzer = ResultsAnalyzer()
    analysis = analyzer.analyze_batch_results(predictions)
    
    # Generate summary report
    report = analyzer.generate_summary_report(predictions)
    
    print("\n" + report)
    
    # Save analysis
    analysis_file = predictions_file.replace('.json', '_analysis.json')
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"Analysis saved to: {analysis_file}")
    
    return analysis


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Manipulation Detection Model Inference')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to model checkpoint or directory')
    parser.add_argument('--config-path', type=str,
                       help='Path to model configuration file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--text', type=str,
                           help='Single text to analyze')
    input_group.add_argument('--input-file', type=str,
                           help='File containing texts to analyze')
    input_group.add_argument('--interactive', action='store_true',
                           help='Run in interactive mode')
    input_group.add_argument('--benchmark', action='store_true',
                           help='Run performance benchmark')
    input_group.add_argument('--analyze', type=str,
                           help='Analyze existing predictions file')
    
    # Output options
    parser.add_argument('--output-file', type=str,
                       help='Output file for batch predictions')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--save-results', action='store_true',
                       help='Save prediction results to file')
    
    # Benchmark options
    parser.add_argument('--benchmark-samples', type=int, default=100,
                       help='Number of samples for benchmarking')
    
    args = parser.parse_args()
    
    # Handle analysis mode
    if args.analyze:
        analyze_predictions(args.analyze)
        return
    
    logger.info("Starting Manipulation Detection Inference")
    logger.info(f"Model: {args.model_path}")
    
    try:
        # Load model
        inference = load_inference_model(args.model_path, args.config_path, args.device)
        
        # Handle different modes
        if args.text:
            # Single text prediction
            result = predict_single_text(inference, args.text, verbose=True)
            
            if args.save_results:
                output_file = args.output_file or 'single_prediction.json'
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Result saved to: {output_file}")
        
        elif args.input_file:
            # Batch prediction from file
            stats = predict_batch_from_file(inference, args.input_file, args.output_file)
            
        elif args.interactive:
            # Interactive mode
            interactive_mode(inference)
        
        elif args.benchmark:
            # Performance benchmark
            benchmark_results = benchmark_performance(inference, args.benchmark_samples)
            
            if args.save_results:
                output_file = args.output_file or 'benchmark_results.json'
                with open(output_file, 'w') as f:
                    json.dump(benchmark_results, f, indent=2)
                logger.info(f"Benchmark results saved to: {output_file}")
        
        else:
            # Default: show help and run interactive mode
            parser.print_help()
            print("\nNo input specified. Starting interactive mode...")
            interactive_mode(inference)
        
        logger.info("Inference completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)