"""Enhanced batch processing utilities for manipulation detection."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Iterator
from pathlib import Path
import json
import csv
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .predictor import ManipulationPredictor, BatchPredictor

logger = logging.getLogger(__name__)


class EnhancedBatchProcessor:
    """Enhanced batch processor with file I/O and parallel processing."""
    
    def __init__(self, 
                 predictor: ManipulationPredictor,
                 batch_size: int = 32,
                 max_workers: int = 4):
        """Initialize enhanced batch processor.
        
        Args:
            predictor: ManipulationPredictor instance
            batch_size: Batch size for processing
            max_workers: Maximum number of worker threads
        """
        self.predictor = predictor
        self.batch_predictor = BatchPredictor(predictor, batch_size)
        self.batch_size = batch_size
        self.max_workers = max_workers
        
    def process_file(self, 
                    input_file: str,
                    output_file: str,
                    text_column: str = 'text',
                    include_probabilities: bool = True,
                    progress_bar: bool = True) -> Dict[str, Any]:
        """Process a file containing texts and save results.
        
        Args:
            input_file: Path to input file (CSV, JSON, or TXT)
            output_file: Path to output file
            text_column: Name of the text column (for CSV/JSON)
            include_probabilities: Whether to include class probabilities
            progress_bar: Whether to show progress bar
            
        Returns:
            Dictionary containing processing statistics
        """
        start_time = time.time()
        
        # Load data
        texts, metadata = self._load_file(input_file, text_column)
        
        logger.info(f"Processing {len(texts)} texts from {input_file}")
        
        # Process in batches with progress bar
        all_results = []
        
        if progress_bar:
            batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
            for batch in tqdm(batches, desc="Processing batches"):
                batch_results = self.batch_predictor.predict_batch(batch, include_probabilities)
                all_results.extend(batch_results)
        else:
            all_results = self.batch_predictor.predict_batch(texts, include_probabilities)
        
        # Combine with metadata if available
        if metadata:
            for i, result in enumerate(all_results):
                result.update(metadata[i])
        
        # Save results
        self._save_results(all_results, output_file)
        
        # Calculate statistics
        processing_time = time.time() - start_time
        stats = self._calculate_processing_stats(all_results, processing_time)
        
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        logger.info(f"Results saved to {output_file}")
        
        return stats
    
    def _load_file(self, file_path: str, text_column: str) -> tuple[List[str], Optional[List[Dict]]]:
        """Load texts from various file formats.
        
        Args:
            file_path: Path to the input file
            text_column: Name of the text column
            
        Returns:
            Tuple of (texts, metadata)
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
            texts = df[text_column].astype(str).tolist()
            metadata = df.drop(columns=[text_column]).to_dict('records')
            return texts, metadata
            
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                if isinstance(data[0], dict):
                    texts = [item[text_column] for item in data]
                    metadata = [{k: v for k, v in item.items() if k != text_column} for item in data]
                    return texts, metadata
                else:
                    return data, None
            else:
                raise ValueError("JSON file must contain a list of texts or objects")
                
        elif file_path.suffix.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            return texts, None
            
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _save_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save results to file.
        
        Args:
            results: List of prediction results
            output_file: Path to output file
        """
        output_path = Path(output_file)
        
        if output_path.suffix.lower() == '.csv':
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            
        elif output_path.suffix.lower() == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")
    
    def _calculate_processing_stats(self, 
                                  results: List[Dict[str, Any]], 
                                  processing_time: float) -> Dict[str, Any]:
        """Calculate processing statistics.
        
        Args:
            results: List of prediction results
            processing_time: Total processing time in seconds
            
        Returns:
            Dictionary containing statistics
        """
        total_texts = len(results)
        manipulation_count = sum(1 for r in results if r.get('is_manipulation', False))
        high_confidence_count = sum(1 for r in results if r.get('high_confidence', False))
        
        # Count predictions by class
        class_counts = {}
        confidence_sum = 0
        
        for result in results:
            predicted_class = result.get('predicted_class', 'unknown')
            class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
            confidence_sum += result.get('confidence', 0)
        
        stats = {
            'total_texts': total_texts,
            'processing_time_seconds': processing_time,
            'texts_per_second': total_texts / processing_time if processing_time > 0 else 0,
            'manipulation_detected': manipulation_count,
            'manipulation_rate': manipulation_count / total_texts if total_texts > 0 else 0,
            'high_confidence_predictions': high_confidence_count,
            'high_confidence_rate': high_confidence_count / total_texts if total_texts > 0 else 0,
            'average_confidence': confidence_sum / total_texts if total_texts > 0 else 0,
            'class_distribution': class_counts
        }
        
        return stats
    
    def process_streaming(self, 
                         texts: Iterator[str],
                         output_callback: callable,
                         buffer_size: int = 100) -> Dict[str, Any]:
        """Process texts in streaming fashion.
        
        Args:
            texts: Iterator of text strings
            output_callback: Function to call with each batch of results
            buffer_size: Size of the processing buffer
            
        Returns:
            Dictionary containing processing statistics
        """
        start_time = time.time()
        buffer = []
        total_processed = 0
        
        for text in texts:
            buffer.append(text)
            
            if len(buffer) >= buffer_size:
                # Process buffer
                results = self.batch_predictor.predict_batch(buffer, return_probabilities=True)
                output_callback(results)
                
                total_processed += len(buffer)
                buffer = []
        
        # Process remaining texts in buffer
        if buffer:
            results = self.batch_predictor.predict_batch(buffer, return_probabilities=True)
            output_callback(results)
            total_processed += len(buffer)
        
        processing_time = time.time() - start_time
        
        return {
            'total_processed': total_processed,
            'processing_time_seconds': processing_time,
            'texts_per_second': total_processed / processing_time if processing_time > 0 else 0
        }


class ParallelBatchProcessor:
    """Parallel batch processor for high-throughput processing."""
    
    def __init__(self, 
                 predictor: ManipulationPredictor,
                 num_workers: int = 4,
                 batch_size: int = 32):
        """Initialize parallel batch processor.
        
        Args:
            predictor: ManipulationPredictor instance
            num_workers: Number of parallel workers
            batch_size: Batch size per worker
        """
        self.predictor = predictor
        self.num_workers = num_workers
        self.batch_size = batch_size
    
    def process_large_dataset(self, 
                            texts: List[str],
                            progress_bar: bool = True) -> List[Dict[str, Any]]:
        """Process large dataset using parallel workers.
        
        Args:
            texts: List of input texts
            progress_bar: Whether to show progress bar
            
        Returns:
            List of prediction results
        """
        # Split texts into chunks for parallel processing
        chunk_size = len(texts) // self.num_workers
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit tasks
            future_to_chunk = {
                executor.submit(self._process_chunk, chunk): i 
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results with progress bar
            if progress_bar:
                futures = tqdm(as_completed(future_to_chunk), 
                             total=len(chunks), 
                             desc="Processing chunks")
            else:
                futures = as_completed(future_to_chunk)
            
            chunk_results = {}
            for future in futures:
                chunk_idx = future_to_chunk[future]
                try:
                    results = future.result()
                    chunk_results[chunk_idx] = results
                except Exception as exc:
                    logger.error(f'Chunk {chunk_idx} generated an exception: {exc}')
                    chunk_results[chunk_idx] = []
        
        # Combine results in order
        for i in range(len(chunks)):
            all_results.extend(chunk_results.get(i, []))
        
        return all_results
    
    def _process_chunk(self, chunk: List[str]) -> List[Dict[str, Any]]:
        """Process a chunk of texts.
        
        Args:
            chunk: List of texts to process
            
        Returns:
            List of prediction results
        """
        batch_processor = BatchPredictor(self.predictor, self.batch_size)
        return batch_processor.predict_batch(chunk, return_probabilities=True)


class ResultsAnalyzer:
    """Analyzer for batch processing results."""
    
    def __init__(self):
        """Initialize results analyzer."""
        pass
    
    def analyze_batch_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze batch processing results.
        
        Args:
            results: List of prediction results
            
        Returns:
            Dictionary containing analysis
        """
        if not results:
            return {'error': 'No results to analyze'}
        
        total_count = len(results)
        manipulation_count = sum(1 for r in results if r.get('is_manipulation', False))
        
        # Confidence analysis
        confidences = [r.get('confidence', 0) for r in results]
        
        # Class distribution
        class_counts = {}
        for result in results:
            predicted_class = result.get('predicted_class', 'unknown')
            class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
        
        # Risk analysis
        high_risk_count = 0
        medium_risk_count = 0
        
        for result in results:
            predicted_class = result.get('predicted_class', '')
            confidence = result.get('confidence', 0)
            
            if predicted_class in ['gaslighting', 'threatening_intimidation', 'belittling_ridicule']:
                if confidence > 0.7:
                    high_risk_count += 1
            elif predicted_class in ['guilt_tripping', 'love_bombing', 'passive_aggression']:
                if confidence > 0.7:
                    medium_risk_count += 1
        
        analysis = {
            'summary': {
                'total_texts': total_count,
                'manipulation_detected': manipulation_count,
                'manipulation_rate': manipulation_count / total_count,
                'ethical_persuasion': total_count - manipulation_count,
                'ethical_rate': (total_count - manipulation_count) / total_count
            },
            'confidence_stats': {
                'mean_confidence': np.mean(confidences),
                'median_confidence': np.median(confidences),
                'std_confidence': np.std(confidences),
                'min_confidence': np.min(confidences),
                'max_confidence': np.max(confidences),
                'high_confidence_count': sum(1 for c in confidences if c >= 0.8),
                'low_confidence_count': sum(1 for c in confidences if c < 0.5)
            },
            'class_distribution': class_counts,
            'risk_assessment': {
                'high_risk_detections': high_risk_count,
                'medium_risk_detections': medium_risk_count,
                'requires_immediate_attention': high_risk_count,
                'requires_monitoring': medium_risk_count
            }
        }
        
        return analysis
    
    def generate_summary_report(self, 
                              results: List[Dict[str, Any]],
                              output_file: Optional[str] = None) -> str:
        """Generate a summary report from batch results.
        
        Args:
            results: List of prediction results
            output_file: Optional file to save the report
            
        Returns:
            Summary report as string
        """
        analysis = self.analyze_batch_results(results)
        
        report = []
        report.append("BATCH PROCESSING SUMMARY REPORT")
        report.append("=" * 40)
        report.append("")
        
        # Summary statistics
        summary = analysis['summary']
        report.append("OVERVIEW:")
        report.append(f"  Total texts processed: {summary['total_texts']:,}")
        report.append(f"  Manipulation detected: {summary['manipulation_detected']:,} ({summary['manipulation_rate']:.1%})")
        report.append(f"  Ethical persuasion: {summary['ethical_persuasion']:,} ({summary['ethical_rate']:.1%})")
        report.append("")
        
        # Confidence statistics
        conf_stats = analysis['confidence_stats']
        report.append("CONFIDENCE ANALYSIS:")
        report.append(f"  Average confidence: {conf_stats['mean_confidence']:.3f}")
        report.append(f"  High confidence predictions (≥0.8): {conf_stats['high_confidence_count']:,}")
        report.append(f"  Low confidence predictions (<0.5): {conf_stats['low_confidence_count']:,}")
        report.append("")
        
        # Class distribution
        report.append("CLASS DISTRIBUTION:")
        class_dist = analysis['class_distribution']
        for class_name, count in sorted(class_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = count / summary['total_texts'] * 100
            report.append(f"  {class_name}: {count:,} ({percentage:.1f}%)")
        report.append("")
        
        # Risk assessment
        risk = analysis['risk_assessment']
        report.append("RISK ASSESSMENT:")
        report.append(f"  High-risk detections: {risk['high_risk_detections']:,}")
        report.append(f"  Medium-risk detections: {risk['medium_risk_detections']:,}")
        report.append(f"  Requires immediate attention: {risk['requires_immediate_attention']:,}")
        report.append("")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Summary report saved to {output_file}")
        
        return report_text