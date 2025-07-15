"""
Custom Airflow operator for quality validation operations.

This operator provides integration between Airflow and quality validation
components for ensuring output quality in the text-to-audiobook pipeline.
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.hooks.base import BaseHook


class QualityValidationOperator(BaseOperator):
    """
    Custom operator for quality validation operations.
    
    This operator integrates with validation components to ensure
    output quality meets specified thresholds.
    """
    
    template_fields = ['structured_segments', 'validation_config', 'quality_thresholds']
    template_ext = ['.json']
    ui_color = '#e377c2'
    
    @apply_defaults
    def __init__(
        self,
        structured_segments: List[Dict[str, Any]],
        validation_config: Dict[str, Any] = None,
        quality_thresholds: Dict[str, float] = None,
        validation_type: str = "comprehensive",
        fail_on_threshold: bool = True,
        output_key: str = "quality_validation_result",
        *args,
        **kwargs
    ):
        """
        Initialize the QualityValidationOperator.
        
        Args:
            structured_segments: List of structured segments to validate
            validation_config: Validation configuration options
            quality_thresholds: Quality thresholds for validation
            validation_type: Type of validation to perform
            fail_on_threshold: Whether to fail task if threshold not met
            output_key: Key for storing output in XCom
        """
        super().__init__(*args, **kwargs)
        
        self.structured_segments = structured_segments
        self.validation_config = validation_config or {}
        self.quality_thresholds = quality_thresholds or {
            'overall_quality': 95.0,
            'speaker_consistency': 90.0,
            'attribution_confidence': 85.0,
            'error_rate': 5.0
        }
        self.validation_type = validation_type
        self.fail_on_threshold = fail_on_threshold
        self.output_key = output_key
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the quality validation operation.
        
        Args:
            context: Airflow execution context
            
        Returns:
            Dictionary containing validation results
        """
        try:
            job_id = context['dag_run'].run_id
            self.logger.info(f"Starting quality validation for job: {job_id}")
            
            # Import validation components
            import sys
            sys.path.append('/opt/airflow/dags')
            
            # Route to appropriate validation method
            if self.validation_type == "comprehensive":
                result = self._comprehensive_validation(job_id)
            elif self.validation_type == "speaker_consistency":
                result = self._speaker_consistency_validation(job_id)
            elif self.validation_type == "attribution_quality":
                result = self._attribution_quality_validation(job_id)
            elif self.validation_type == "structural_validation":
                result = self._structural_validation(job_id)
            else:
                raise AirflowException(f"Unknown validation type: {self.validation_type}")
            
            # Check thresholds
            threshold_results = self._check_thresholds(result)
            result['threshold_results'] = threshold_results
            
            # Store results in XCom
            self.xcom_push(context, key=self.output_key, value=result)
            
            # Handle threshold failures
            if self.fail_on_threshold and not threshold_results['all_passed']:
                failed_thresholds = [
                    f"{k}: {v['actual']:.2f} < {v['threshold']:.2f}"
                    for k, v in threshold_results['details'].items()
                    if not v['passed']
                ]
                
                self.logger.error(f"Quality thresholds failed: {failed_thresholds}")
                raise AirflowException(f"Quality validation failed: {failed_thresholds}")
            
            self.logger.info(
                f"Quality validation completed for job {job_id}: "
                f"overall score {result['overall_quality_score']:.2f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Quality validation operation failed: {str(e)}")
            
            # Store error information
            error_result = {
                'job_id': context['dag_run'].run_id,
                'validation_type': self.validation_type,
                'error_message': str(e),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }
            
            self.xcom_push(context, key=f"{self.output_key}_error", value=error_result)
            
            raise AirflowException(f"Quality validation operation failed: {str(e)}")
    
    def _comprehensive_validation(self, job_id: str) -> Dict[str, Any]:
        """Perform comprehensive quality validation."""
        from src.validation.validator import SimplifiedValidator
        
        # Create validator
        validator = SimplifiedValidator()
        
        # Perform validation
        start_time = time.time()
        validation_result = validator.validate_structured_segments(self.structured_segments)
        processing_time = time.time() - start_time
        
        return {
            'job_id': job_id,
            'validation_type': 'comprehensive',
            'validation_result': validation_result,
            'overall_quality_score': validation_result.get('quality_score', 0.0),
            'total_segments': len(self.structured_segments),
            'processing_time': processing_time,
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }
    
    def _speaker_consistency_validation(self, job_id: str) -> Dict[str, Any]:
        """Perform speaker consistency validation."""
        start_time = time.time()
        
        # Analyze speaker consistency
        speaker_stats = {}
        speaker_transitions = []
        
        for i, segment in enumerate(self.structured_segments):
            speaker = segment.get('speaker', 'UNKNOWN')
            
            # Track speaker statistics
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    'count': 0,
                    'total_length': 0,
                    'segments': []
                }
            
            speaker_stats[speaker]['count'] += 1
            speaker_stats[speaker]['total_length'] += len(segment.get('text', ''))
            speaker_stats[speaker]['segments'].append(i)
            
            # Track speaker transitions
            if i > 0:
                prev_speaker = self.structured_segments[i-1].get('speaker', 'UNKNOWN')
                if prev_speaker != speaker:
                    speaker_transitions.append({
                        'from': prev_speaker,
                        'to': speaker,
                        'segment_index': i
                    })
        
        # Calculate consistency metrics
        total_segments = len(self.structured_segments)
        unique_speakers = len(speaker_stats)
        transition_rate = len(speaker_transitions) / max(1, total_segments - 1)
        
        # Calculate consistency score
        consistency_score = 100.0 * (1 - transition_rate)
        
        processing_time = time.time() - start_time
        
        return {
            'job_id': job_id,
            'validation_type': 'speaker_consistency',
            'speaker_stats': speaker_stats,
            'speaker_transitions': speaker_transitions,
            'unique_speakers': unique_speakers,
            'transition_rate': transition_rate,
            'consistency_score': consistency_score,
            'overall_quality_score': consistency_score,
            'total_segments': total_segments,
            'processing_time': processing_time,
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }
    
    def _attribution_quality_validation(self, job_id: str) -> Dict[str, Any]:
        """Perform attribution quality validation."""
        start_time = time.time()
        
        # Analyze attribution quality
        attribution_stats = {
            'total_segments': len(self.structured_segments),
            'attributed_segments': 0,
            'ambiguous_segments': 0,
            'error_segments': 0,
            'high_confidence_segments': 0
        }
        
        confidence_scores = []
        
        for segment in self.structured_segments:
            speaker = segment.get('speaker', 'UNKNOWN')
            confidence = segment.get('confidence', 0.0)
            
            confidence_scores.append(confidence)
            
            if speaker == 'AMBIGUOUS':
                attribution_stats['ambiguous_segments'] += 1
            elif speaker in ['ERROR', 'PROCESSING_ERROR', 'INFERENCE_ERROR']:
                attribution_stats['error_segments'] += 1
            else:
                attribution_stats['attributed_segments'] += 1
                
                if confidence > 0.8:
                    attribution_stats['high_confidence_segments'] += 1
        
        # Calculate quality metrics
        total_segments = attribution_stats['total_segments']
        attribution_rate = attribution_stats['attributed_segments'] / max(1, total_segments) * 100
        error_rate = attribution_stats['error_segments'] / max(1, total_segments) * 100
        confidence_rate = attribution_stats['high_confidence_segments'] / max(1, total_segments) * 100
        
        avg_confidence = sum(confidence_scores) / max(1, len(confidence_scores))
        
        # Calculate overall quality score
        quality_score = (attribution_rate * 0.4 + confidence_rate * 0.4 + (100 - error_rate) * 0.2)
        
        processing_time = time.time() - start_time
        
        return {
            'job_id': job_id,
            'validation_type': 'attribution_quality',
            'attribution_stats': attribution_stats,
            'attribution_rate': attribution_rate,
            'error_rate': error_rate,
            'confidence_rate': confidence_rate,
            'average_confidence': avg_confidence,
            'overall_quality_score': quality_score,
            'total_segments': total_segments,
            'processing_time': processing_time,
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }
    
    def _structural_validation(self, job_id: str) -> Dict[str, Any]:
        """Perform structural validation."""
        start_time = time.time()
        
        # Validate segment structure
        structure_issues = []
        required_fields = ['speaker', 'text']
        
        for i, segment in enumerate(self.structured_segments):
            # Check required fields
            for field in required_fields:
                if field not in segment:
                    structure_issues.append({
                        'segment_index': i,
                        'issue': f'Missing required field: {field}',
                        'severity': 'error'
                    })
            
            # Check text content
            text = segment.get('text', '')
            if not text or not text.strip():
                structure_issues.append({
                    'segment_index': i,
                    'issue': 'Empty or whitespace-only text',
                    'severity': 'warning'
                })
            
            # Check speaker format
            speaker = segment.get('speaker', '')
            if not speaker or len(speaker) < 2:
                structure_issues.append({
                    'segment_index': i,
                    'issue': f'Invalid speaker format: {speaker}',
                    'severity': 'warning'
                })
        
        # Calculate structural quality
        error_count = sum(1 for issue in structure_issues if issue['severity'] == 'error')
        warning_count = sum(1 for issue in structure_issues if issue['severity'] == 'warning')
        
        total_segments = len(self.structured_segments)
        error_rate = error_count / max(1, total_segments) * 100
        warning_rate = warning_count / max(1, total_segments) * 100
        
        # Calculate structural quality score
        structural_score = 100.0 - (error_rate * 2 + warning_rate * 0.5)
        structural_score = max(0, min(100, structural_score))
        
        processing_time = time.time() - start_time
        
        return {
            'job_id': job_id,
            'validation_type': 'structural_validation',
            'structure_issues': structure_issues,
            'error_count': error_count,
            'warning_count': warning_count,
            'error_rate': error_rate,
            'warning_rate': warning_rate,
            'structural_score': structural_score,
            'overall_quality_score': structural_score,
            'total_segments': total_segments,
            'processing_time': processing_time,
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }
    
    def _check_thresholds(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Check quality thresholds."""
        threshold_results = {
            'all_passed': True,
            'details': {}
        }
        
        # Map result keys to threshold keys
        result_to_threshold = {
            'overall_quality_score': 'overall_quality',
            'consistency_score': 'speaker_consistency',
            'average_confidence': 'attribution_confidence',
            'error_rate': 'error_rate'
        }
        
        for result_key, threshold_key in result_to_threshold.items():
            if result_key in result and threshold_key in self.quality_thresholds:
                actual_value = result[result_key]
                threshold_value = self.quality_thresholds[threshold_key]
                
                # For error rate, check if actual is below threshold
                if threshold_key == 'error_rate':
                    passed = actual_value <= threshold_value
                else:
                    passed = actual_value >= threshold_value
                
                threshold_results['details'][threshold_key] = {
                    'actual': actual_value,
                    'threshold': threshold_value,
                    'passed': passed
                }
                
                if not passed:
                    threshold_results['all_passed'] = False
        
        return threshold_results


class BatchQualityValidationOperator(BaseOperator):
    """
    Operator for batch quality validation operations.
    
    This operator validates multiple batches of structured segments.
    """
    
    template_fields = ['segment_batches', 'validation_config', 'quality_thresholds']
    template_ext = ['.json']
    ui_color = '#ff7f0e'
    
    @apply_defaults
    def __init__(
        self,
        segment_batches: List[List[Dict[str, Any]]],
        validation_config: Dict[str, Any] = None,
        quality_thresholds: Dict[str, float] = None,
        validation_type: str = "comprehensive",
        fail_on_threshold: bool = True,
        max_workers: int = 4,
        output_key: str = "batch_validation_result",
        *args,
        **kwargs
    ):
        """
        Initialize the BatchQualityValidationOperator.
        
        Args:
            segment_batches: List of segment batches to validate
            validation_config: Validation configuration
            quality_thresholds: Quality thresholds
            validation_type: Type of validation
            fail_on_threshold: Whether to fail on threshold
            max_workers: Maximum parallel workers
            output_key: Key for storing output in XCom
        """
        super().__init__(*args, **kwargs)
        
        self.segment_batches = segment_batches
        self.validation_config = validation_config or {}
        self.quality_thresholds = quality_thresholds or {
            'overall_quality': 95.0,
            'speaker_consistency': 90.0,
            'attribution_confidence': 85.0,
            'error_rate': 5.0
        }
        self.validation_type = validation_type
        self.fail_on_threshold = fail_on_threshold
        self.max_workers = max_workers
        self.output_key = output_key
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the batch quality validation operation.
        
        Args:
            context: Airflow execution context
            
        Returns:
            Dictionary containing batch validation results
        """
        try:
            job_id = context['dag_run'].run_id
            self.logger.info(f"Starting batch quality validation for job: {job_id}")
            
            # Import concurrent processing
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # Process batches in parallel
            start_time = time.time()
            batch_results = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(self._validate_batch, i, batch, job_id): i
                    for i, batch in enumerate(self.segment_batches)
                }
                
                # Collect results
                for future in as_completed(future_to_batch):
                    batch_index = future_to_batch[future]
                    try:
                        batch_result = future.result()
                        batch_results.append(batch_result)
                        self.logger.info(f"Batch {batch_index} validation completed")
                    except Exception as e:
                        self.logger.error(f"Batch {batch_index} validation failed: {str(e)}")
                        batch_results.append({
                            'batch_index': batch_index,
                            'error': str(e),
                            'status': 'failed'
                        })
            
            processing_time = time.time() - start_time
            
            # Aggregate results
            total_segments = sum(len(batch) for batch in self.segment_batches)
            successful_batches = sum(1 for result in batch_results if result.get('status') == 'completed')
            
            # Calculate overall quality score
            quality_scores = [
                result.get('overall_quality_score', 0.0)
                for result in batch_results
                if result.get('status') == 'completed'
            ]
            
            overall_quality_score = sum(quality_scores) / max(1, len(quality_scores))
            
            result = {
                'job_id': job_id,
                'validation_type': self.validation_type,
                'batch_results': batch_results,
                'total_batches': len(self.segment_batches),
                'successful_batches': successful_batches,
                'total_segments': total_segments,
                'overall_quality_score': overall_quality_score,
                'processing_time': processing_time,
                'status': 'completed',
                'timestamp': datetime.now().isoformat()
            }
            
            # Store results in XCom
            self.xcom_push(context, key=self.output_key, value=result)
            
            self.logger.info(
                f"Batch quality validation completed: {successful_batches}/{len(self.segment_batches)} batches successful"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Batch quality validation operation failed: {str(e)}")
            
            # Store error information
            error_result = {
                'job_id': context['dag_run'].run_id,
                'validation_type': self.validation_type,
                'error_message': str(e),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }
            
            self.xcom_push(context, key=f"{self.output_key}_error", value=error_result)
            
            raise AirflowException(f"Batch quality validation operation failed: {str(e)}")
    
    def _validate_batch(self, batch_index: int, batch: List[Dict[str, Any]], job_id: str) -> Dict[str, Any]:
        """Validate a single batch of segments."""
        try:
            # Create individual validation operator
            validation_operator = QualityValidationOperator(
                task_id=f"validate_batch_{batch_index}",
                structured_segments=batch,
                validation_config=self.validation_config,
                quality_thresholds=self.quality_thresholds,
                validation_type=self.validation_type,
                fail_on_threshold=False  # Don't fail individual batches
            )
            
            # Create mock context
            mock_context = {
                'dag_run': type('obj', (object,), {'run_id': f"{job_id}_batch_{batch_index}"})()
            }
            
            # Execute batch validation
            result = validation_operator.execute(mock_context)
            
            return {
                'batch_index': batch_index,
                'batch_size': len(batch),
                'result': result,
                'overall_quality_score': result.get('overall_quality_score', 0.0),
                'status': 'completed'
            }
            
        except Exception as e:
            self.logger.error(f"Error validating batch {batch_index}: {str(e)}")
            return {
                'batch_index': batch_index,
                'batch_size': len(batch),
                'error': str(e),
                'overall_quality_score': 0.0,
                'status': 'failed'
            }


# Utility functions for creating operators
def create_quality_validation_operator(
    task_id: str,
    structured_segments: List[Dict[str, Any]],
    validation_config: Dict[str, Any] = None,
    quality_thresholds: Dict[str, float] = None,
    validation_type: str = "comprehensive",
    fail_on_threshold: bool = True,
    dag=None
) -> QualityValidationOperator:
    """
    Factory function for creating QualityValidationOperator instances.
    
    Args:
        task_id: Unique task identifier
        structured_segments: Segments to validate
        validation_config: Validation configuration
        quality_thresholds: Quality thresholds
        validation_type: Type of validation
        fail_on_threshold: Whether to fail on threshold
        dag: DAG instance
        
    Returns:
        Configured QualityValidationOperator instance
    """
    return QualityValidationOperator(
        task_id=task_id,
        structured_segments=structured_segments,
        validation_config=validation_config,
        quality_thresholds=quality_thresholds,
        validation_type=validation_type,
        fail_on_threshold=fail_on_threshold,
        dag=dag
    )


def create_batch_quality_validation_operator(
    task_id: str,
    segment_batches: List[List[Dict[str, Any]]],
    validation_config: Dict[str, Any] = None,
    quality_thresholds: Dict[str, float] = None,
    validation_type: str = "comprehensive",
    fail_on_threshold: bool = True,
    max_workers: int = 4,
    dag=None
) -> BatchQualityValidationOperator:
    """
    Factory function for creating BatchQualityValidationOperator instances.
    
    Args:
        task_id: Unique task identifier
        segment_batches: Segment batches to validate
        validation_config: Validation configuration
        quality_thresholds: Quality thresholds
        validation_type: Type of validation
        fail_on_threshold: Whether to fail on threshold
        max_workers: Maximum workers
        dag: DAG instance
        
    Returns:
        Configured BatchQualityValidationOperator instance
    """
    return BatchQualityValidationOperator(
        task_id=task_id,
        segment_batches=segment_batches,
        validation_config=validation_config,
        quality_thresholds=quality_thresholds,
        validation_type=validation_type,
        fail_on_threshold=fail_on_threshold,
        max_workers=max_workers,
        dag=dag
    )