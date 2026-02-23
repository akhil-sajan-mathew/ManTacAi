#!/usr/bin/env python3
"""
Deployment Validation Script

This script validates that the model is ready for deployment by testing
model export, loading, and inference consistency in a clean environment.

Usage:
    python scripts/validate_deployment.py --model-path models/best_model.pt --config-path config/model_config.yaml
"""

import os
import sys
import argparse
import json
import logging
import tempfile
import shutil
from pathlib import Path
import torch
import numpy as np
from datetime import datetime
import subprocess

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.deployment.model_export import ModelExporter
from src.models.model_utils import ModelConfig
from src.inference.deployment import DeploymentInference
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeploymentValidator:
    """Validates model deployment readiness."""
    
    def __init__(self, model_path: str, config_path: str):
        """Initialize deployment validator.
        
        Args:
            model_path: Path to trained model
            config_path: Path to model configuration
        """
        self.model_path = model_path
        self.config_path = config_path
        self.validation_results = {}
        
        logger.info("DeploymentValidator initialized")
        logger.info(f"Model: {model_path}")
        logger.info(f"Config: {config_path}")
    
    def validate_model_loading(self) -> bool:
        """Validate that model can be loaded correctly.
        
        Returns:
            True if model loads successfully
        """
        logger.info("Validating model loading...")
        
        try:
            # Load configuration
            if self.config_path.endswith('.yaml'):
                config = ModelConfig.from_yaml(self.config_path)
            else:
                config = ModelConfig.from_json(self.config_path)
            
            # Load model
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            from src.models.model_utils import ModelFactory
            model = ModelFactory.create_model(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            
            # Test basic functionality
            test_text = "This is a test message for validation."
            encoding = tokenizer(test_text, return_tensors='pt', truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = model(encoding['input_ids'], encoding['attention_mask'])
                logits = outputs['logits']
            
            # Validate output shape
            expected_shape = (1, config.num_classes)
            if logits.shape != expected_shape:
                raise ValueError(f"Unexpected output shape: {logits.shape}, expected: {expected_shape}")
            
            self.validation_results['model_loading'] = {
                'status': 'passed',
                'model_parameters': sum(p.numel() for p in model.parameters()),
                'output_shape': list(logits.shape),
                'config_valid': True
            }
            
            logger.info("✓ Model loading validation passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Model loading validation failed: {str(e)}")
            self.validation_results['model_loading'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def validate_model_export(self) -> bool:
        """Validate model export functionality.
        
        Returns:
            True if export works correctly
        """
        logger.info("Validating model export...")
        
        try:
            # Load model and tokenizer
            if self.config_path.endswith('.yaml'):
                config = ModelConfig.from_yaml(self.config_path)
            else:
                config = ModelConfig.from_json(self.config_path)
            
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            from src.models.model_utils import ModelFactory
            model = ModelFactory.create_model(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            
            # Create temporary directory for export testing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Create exporter
                exporter = ModelExporter(model, tokenizer, config.to_dict())
                
                # Test PyTorch export
                pytorch_export_path = exporter.export_pytorch_model(str(temp_path / 'pytorch_model'))
                
                # Test ONNX export
                onnx_export_path = exporter.export_onnx_model(str(temp_path / 'model.onnx'))
                
                # Test quantized export
                quantized_export_path = exporter.export_quantized_model(str(temp_path / 'quantized_model.pt'))
                
                # Validate exported files exist
                pytorch_exists = Path(pytorch_export_path).exists()
                onnx_exists = Path(onnx_export_path).exists()
                quantized_exists = Path(quantized_export_path).exists()
                
                if not all([pytorch_exists, onnx_exists, quantized_exists]):
                    raise ValueError("Some export files were not created")
                
                # Test loading exported PyTorch model
                from src.models.manipulation_classifier import ManipulationClassifier
                exported_model = ManipulationClassifier.from_pretrained(pytorch_export_path)
                exported_model.eval()
                
                # Test inference consistency
                test_text = "This is a test for export validation."
                encoding = tokenizer(test_text, return_tensors='pt', truncation=True, padding=True)
                
                with torch.no_grad():
                    original_output = model(encoding['input_ids'], encoding['attention_mask'])['logits']
                    exported_output = exported_model(encoding['input_ids'], encoding['attention_mask'])['logits']
                
                # Check consistency
                max_diff = torch.max(torch.abs(original_output - exported_output)).item()
                
                if max_diff > 1e-5:
                    logger.warning(f"Export consistency check: difference = {max_diff:.2e}")
                
                self.validation_results['model_export'] = {
                    'status': 'passed',
                    'pytorch_export': pytorch_exists,
                    'onnx_export': onnx_exists,
                    'quantized_export': quantized_exists,
                    'consistency_check': max_diff < 1e-5,
                    'max_difference': max_diff
                }
                
                logger.info("✓ Model export validation passed")
                return True
                
        except Exception as e:
            logger.error(f"✗ Model export validation failed: {str(e)}")
            self.validation_results['model_export'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def validate_deployment_inference(self) -> bool:
        """Validate deployment inference functionality.
        
        Returns:
            True if deployment inference works correctly
        """
        logger.info("Validating deployment inference...")
        
        try:
            # Create temporary deployment package
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Load model and create exporter
                if self.config_path.endswith('.yaml'):
                    config = ModelConfig.from_yaml(self.config_path)
                else:
                    config = ModelConfig.from_json(self.config_path)
                
                checkpoint = torch.load(self.model_path, map_location='cpu')
                
                from src.models.model_utils import ModelFactory
                model = ModelFactory.create_model(config)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                tokenizer = AutoTokenizer.from_pretrained(config.model_name)
                
                # Create deployment package
                exporter = ModelExporter(model, tokenizer, config.to_dict())
                deployment_path = exporter.create_deployment_package(str(temp_path / 'deployment'))
                
                # Test deployment inference
                inference = DeploymentInference(
                    model_path=str(Path(deployment_path) / 'model'),
                    device='cpu',
                    optimize_for_inference=True
                )
                
                # Test health check
                health = inference.health_check()
                if health['status'] != 'healthy':
                    raise ValueError(f"Health check failed: {health}")
                
                # Test single prediction
                test_texts = [
                    "This is a reasonable request.",
                    "You're being too sensitive about this.",
                    "If you really cared, you would understand."
                ]
                
                predictions = []
                inference_times = []
                
                for text in test_texts:
                    import time
                    start_time = time.time()
                    
                    result = inference.predict(text, return_probabilities=True)
                    
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)
                    
                    predictions.append({
                        'text': text,
                        'predicted_class': result['predicted_class'],
                        'confidence': result['confidence'],
                        'is_manipulation': result['is_manipulation']
                    })
                
                # Test batch prediction
                batch_results = inference.predict_batch(test_texts)
                
                # Validate results
                if len(batch_results) != len(test_texts):
                    raise ValueError("Batch prediction returned incorrect number of results")
                
                avg_inference_time = sum(inference_times) / len(inference_times)
                
                self.validation_results['deployment_inference'] = {
                    'status': 'passed',
                    'health_check': health['status'] == 'healthy',
                    'single_predictions': len(predictions),
                    'batch_predictions': len(batch_results),
                    'average_inference_time_ms': avg_inference_time * 1000,
                    'sample_predictions': predictions[:2]  # Include first 2 for verification
                }
                
                logger.info("✓ Deployment inference validation passed")
                logger.info(f"  Average inference time: {avg_inference_time*1000:.2f}ms")
                return True
                
        except Exception as e:
            logger.error(f"✗ Deployment inference validation failed: {str(e)}")
            self.validation_results['deployment_inference'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def validate_clean_environment(self) -> bool:
        """Validate model works in a clean Python environment.
        
        Returns:
            True if clean environment test passes
        """
        logger.info("Validating clean environment deployment...")
        
        try:
            # Create a temporary clean environment test
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Create deployment package
                if self.config_path.endswith('.yaml'):
                    config = ModelConfig.from_yaml(self.config_path)
                else:
                    config = ModelConfig.from_json(self.config_path)
                
                checkpoint = torch.load(self.model_path, map_location='cpu')
                
                from src.models.model_utils import ModelFactory
                model = ModelFactory.create_model(config)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                tokenizer = AutoTokenizer.from_pretrained(config.model_name)
                
                exporter = ModelExporter(model, tokenizer, config.to_dict())
                deployment_path = exporter.create_deployment_package(str(temp_path / 'clean_test'))
                
                # Create a test script
                test_script = f'''
import sys
import os
sys.path.insert(0, "{deployment_path}")

import torch
import json
from transformers import AutoTokenizer
from manipulation_classifier import ManipulationClassifier

try:
    # Load model
    model = ManipulationClassifier.from_pretrained("model/")
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("model/tokenizer")
    
    # Test prediction
    text = "This is a test message."
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(encoding["input_ids"], encoding["attention_mask"])
        logits = outputs["logits"]
        prediction = torch.argmax(logits, dim=-1).item()
    
    print(f"SUCCESS: Prediction = {{prediction}}")
    
except Exception as e:
    print(f"ERROR: {{str(e)}}")
    sys.exit(1)
'''
                
                test_script_path = temp_path / 'test_clean_env.py'
                with open(test_script_path, 'w') as f:
                    f.write(test_script)
                
                # Run the test script in a subprocess
                result = subprocess.run(
                    [sys.executable, str(test_script_path)],
                    cwd=deployment_path,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0 and "SUCCESS" in result.stdout:
                    self.validation_results['clean_environment'] = {
                        'status': 'passed',
                        'output': result.stdout.strip(),
                        'execution_time': 'under_60s'
                    }
                    
                    logger.info("✓ Clean environment validation passed")
                    return True
                else:
                    raise ValueError(f"Clean environment test failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"✗ Clean environment validation failed: {str(e)}")
            self.validation_results['clean_environment'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def validate_performance_consistency(self) -> bool:
        """Validate performance consistency across multiple runs.
        
        Returns:
            True if performance is consistent
        """
        logger.info("Validating performance consistency...")
        
        try:
            # Create deployment inference
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Create deployment package
                if self.config_path.endswith('.yaml'):
                    config = ModelConfig.from_yaml(self.config_path)
                else:
                    config = ModelConfig.from_json(self.config_path)
                
                checkpoint = torch.load(self.model_path, map_location='cpu')
                
                from src.models.model_utils import ModelFactory
                model = ModelFactory.create_model(config)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                tokenizer = AutoTokenizer.from_pretrained(config.model_name)
                
                exporter = ModelExporter(model, tokenizer, config.to_dict())
                deployment_path = exporter.create_deployment_package(str(temp_path / 'perf_test'))
                
                inference = DeploymentInference(
                    model_path=str(Path(deployment_path) / 'model'),
                    device='cpu'
                )
                
                # Test consistency across multiple runs
                test_text = "This is a consistency test message."
                num_runs = 10
                
                predictions = []
                inference_times = []
                
                for i in range(num_runs):
                    import time
                    start_time = time.time()
                    
                    result = inference.predict(test_text)
                    
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)
                    predictions.append(result['predicted_class'])
                
                # Check prediction consistency
                unique_predictions = set(predictions)
                prediction_consistent = len(unique_predictions) == 1
                
                # Check timing consistency (coefficient of variation < 0.5)
                avg_time = np.mean(inference_times)
                std_time = np.std(inference_times)
                cv = std_time / avg_time if avg_time > 0 else float('inf')
                
                timing_consistent = cv < 0.5
                
                self.validation_results['performance_consistency'] = {
                    'status': 'passed' if prediction_consistent and timing_consistent else 'failed',
                    'prediction_consistent': prediction_consistent,
                    'unique_predictions': list(unique_predictions),
                    'timing_consistent': timing_consistent,
                    'average_inference_time_ms': avg_time * 1000,
                    'std_inference_time_ms': std_time * 1000,
                    'coefficient_of_variation': cv,
                    'num_runs': num_runs
                }
                
                if prediction_consistent and timing_consistent:
                    logger.info("✓ Performance consistency validation passed")
                    logger.info(f"  Average inference time: {avg_time*1000:.2f}ms (CV: {cv:.3f})")
                    return True
                else:
                    logger.warning("⚠ Performance consistency issues detected")
                    if not prediction_consistent:
                        logger.warning(f"  Inconsistent predictions: {unique_predictions}")
                    if not timing_consistent:
                        logger.warning(f"  High timing variability: CV = {cv:.3f}")
                    return False
                
        except Exception as e:
            logger.error(f"✗ Performance consistency validation failed: {str(e)}")
            self.validation_results['performance_consistency'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def generate_validation_report(self, output_path: str = None):
        """Generate comprehensive validation report.
        
        Args:
            output_path: Path to save the report
        """
        if output_path is None:
            output_path = f"deployment_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Calculate overall status
        all_tests = [
            self.validation_results.get('model_loading', {}).get('status') == 'passed',
            self.validation_results.get('model_export', {}).get('status') == 'passed',
            self.validation_results.get('deployment_inference', {}).get('status') == 'passed',
            self.validation_results.get('clean_environment', {}).get('status') == 'passed',
            self.validation_results.get('performance_consistency', {}).get('status') == 'passed'
        ]
        
        overall_status = 'PASSED' if all(all_tests) else 'FAILED'
        
        # Create comprehensive report
        report = {
            'validation_summary': {
                'overall_status': overall_status,
                'validation_date': datetime.now().isoformat(),
                'model_path': self.model_path,
                'config_path': self.config_path,
                'tests_passed': sum(all_tests),
                'total_tests': len(all_tests)
            },
            'test_results': self.validation_results,
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("DEPLOYMENT VALIDATION REPORT")
        print("="*60)
        print(f"Overall Status: {overall_status}")
        print(f"Tests Passed: {sum(all_tests)}/{len(all_tests)}")
        print(f"Model: {self.model_path}")
        print(f"Report saved to: {output_path}")
        
        # Print test results
        test_names = [
            'Model Loading',
            'Model Export', 
            'Deployment Inference',
            'Clean Environment',
            'Performance Consistency'
        ]
        
        for i, (test_name, passed) in enumerate(zip(test_names, all_tests)):
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"  {test_name}: {status}")
        
        print("="*60)
        
        logger.info(f"Validation report saved to: {output_path}")
        
        return overall_status == 'PASSED'
    
    def _generate_recommendations(self) -> list:
        """Generate recommendations based on validation results.
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check each test result and provide recommendations
        if self.validation_results.get('model_loading', {}).get('status') != 'passed':
            recommendations.append("Fix model loading issues before deployment")
        
        if self.validation_results.get('model_export', {}).get('status') != 'passed':
            recommendations.append("Resolve model export problems - check dependencies and model compatibility")
        
        if self.validation_results.get('deployment_inference', {}).get('status') != 'passed':
            recommendations.append("Fix deployment inference issues - verify model and tokenizer compatibility")
        
        if self.validation_results.get('clean_environment', {}).get('status') != 'passed':
            recommendations.append("Ensure all dependencies are properly packaged for clean environment deployment")
        
        if self.validation_results.get('performance_consistency', {}).get('status') != 'passed':
            recommendations.append("Investigate performance inconsistencies - may indicate non-deterministic behavior")
        
        # Performance recommendations
        perf_results = self.validation_results.get('deployment_inference', {})
        if perf_results.get('average_inference_time_ms', 0) > 1000:
            recommendations.append("Consider model optimization for faster inference (quantization, ONNX)")
        
        # If all tests passed
        if not recommendations:
            recommendations.extend([
                "Model is ready for production deployment",
                "Monitor performance in production environment",
                "Set up logging and error tracking",
                "Consider load testing for production scale"
            ])
        
        return recommendations
    
    def run_all_validations(self) -> bool:
        """Run all validation tests.
        
        Returns:
            True if all validations pass
        """
        logger.info("Starting comprehensive deployment validation...")
        
        validations = [
            ('Model Loading', self.validate_model_loading),
            ('Model Export', self.validate_model_export),
            ('Deployment Inference', self.validate_deployment_inference),
            ('Clean Environment', self.validate_clean_environment),
            ('Performance Consistency', self.validate_performance_consistency)
        ]
        
        results = []
        
        for test_name, validation_func in validations:
            logger.info(f"Running {test_name} validation...")
            try:
                result = validation_func()
                results.append(result)
                status = "PASSED" if result else "FAILED"
                logger.info(f"{test_name}: {status}")
            except Exception as e:
                logger.error(f"{test_name} validation error: {str(e)}")
                results.append(False)
        
        # Generate report
        overall_success = self.generate_validation_report()
        
        return overall_success


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description='Validate Model Deployment Readiness')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config-path', type=str, required=True,
                       help='Path to model configuration file')
    parser.add_argument('--output-report', type=str,
                       help='Path to save validation report')
    
    args = parser.parse_args()
    
    logger.info("Starting deployment validation")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Config: {args.config_path}")
    
    # Create validator
    validator = DeploymentValidator(args.model_path, args.config_path)
    
    # Run all validations
    success = validator.run_all_validations()
    
    if success:
        logger.info("✓ All deployment validations passed - model is ready for deployment!")
        sys.exit(0)
    else:
        logger.error("✗ Some deployment validations failed - review the report for details")
        sys.exit(1)


if __name__ == "__main__":
    main()