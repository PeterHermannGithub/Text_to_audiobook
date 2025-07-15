"""
Custom Airflow operator for LLM processing operations.

This operator provides integration between Airflow and LLM processing
components for speaker attribution and text analysis.
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.exceptions import AirflowException
from airflow.hooks.base import BaseHook


class LLMProcessingOperator(BaseOperator):
    """
    Custom operator for LLM processing operations.
    
    This operator integrates with LLM components to perform speaker attribution
    and other text analysis tasks.
    """
    
    template_fields = ['text_segments', 'llm_config', 'processing_options']
    template_ext = ['.json']
    ui_color = '#17becf'
    
    @apply_defaults
    def __init__(
        self,
        text_segments: List[Dict[str, Any]],
        llm_config: Dict[str, Any] = None,
        processing_options: Dict[str, Any] = None,
        operation_type: str = "speaker_attribution",
        output_key: str = "llm_processing_result",
        timeout_seconds: int = 1800,
        *args,
        **kwargs
    ):
        """
        Initialize the LLMProcessingOperator.
        
        Args:
            text_segments: List of text segments to process
            llm_config: LLM configuration (engine, model, etc.)
            processing_options: Processing configuration options
            operation_type: Type of LLM operation to perform
            output_key: Key for storing output in XCom
            timeout_seconds: Timeout for LLM operation
        """
        super().__init__(*args, **kwargs)
        
        self.text_segments = text_segments
        self.llm_config = llm_config or {}
        self.processing_options = processing_options or {}
        self.operation_type = operation_type
        self.output_key = output_key
        self.timeout_seconds = timeout_seconds
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the LLM processing operation.
        
        Args:
            context: Airflow execution context
            
        Returns:
            Dictionary containing LLM processing results
        """
        try:
            job_id = context['dag_run'].run_id
            self.logger.info(f"Starting LLM processing for job: {job_id}")
            
            # Import LLM components
            import sys
            sys.path.append('/opt/airflow/dags')
            
            # Route to appropriate processing method
            if self.operation_type == "speaker_attribution":
                result = self._process_speaker_attribution(job_id)
            elif self.operation_type == "text_classification":
                result = self._process_text_classification(job_id)
            elif self.operation_type == "content_analysis":
                result = self._process_content_analysis(job_id)
            elif self.operation_type == "distributed_inference":
                result = self._process_distributed_inference(job_id)
            else:
                raise AirflowException(f"Unknown operation type: {self.operation_type}")
            
            # Store results in XCom
            self.xcom_push(context, key=self.output_key, value=result)
            
            self.logger.info(
                f"LLM processing completed for job {job_id}: {result['total_segments']} segments processed"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"LLM processing operation failed: {str(e)}")
            
            # Store error information
            error_result = {
                'job_id': context['dag_run'].run_id,
                'operation_type': self.operation_type,
                'error_message': str(e),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }
            
            self.xcom_push(context, key=f"{self.output_key}_error", value=error_result)
            
            raise AirflowException(f"LLM processing operation failed: {str(e)}")
    
    def _process_speaker_attribution(self, job_id: str) -> Dict[str, Any]:
        """Process speaker attribution using LLM orchestrator."""
        from src.attribution.llm.orchestrator import LLMOrchestrator
        
        # Create LLM orchestrator
        orchestrator = LLMOrchestrator(
            engine=self.llm_config.get('engine', 'local'),
            model=self.llm_config.get('model', 'mistral'),
            **self.llm_config
        )
        
        # Process segments
        start_time = time.time()
        processed_segments = []
        
        for segment in self.text_segments:
            try:
                # Process segment with LLM
                result = orchestrator.process_segment(
                    segment=segment,
                    metadata=self.processing_options.get('metadata', {}),
                    processing_options=self.processing_options
                )
                
                processed_segments.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing segment: {str(e)}")
                # Add segment with error marker
                error_segment = segment.copy()
                error_segment['speaker'] = 'PROCESSING_ERROR'
                error_segment['error'] = str(e)
                processed_segments.append(error_segment)
        
        processing_time = time.time() - start_time
        
        return {
            'job_id': job_id,
            'operation_type': 'speaker_attribution',
            'processed_segments': processed_segments,
            'total_segments': len(processed_segments),
            'processing_time': processing_time,
            'llm_config': self.llm_config,
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }
    
    def _process_text_classification(self, job_id: str) -> Dict[str, Any]:
        """Process text classification using LLM client."""
        from src.llm_pool.llm_client import LLMClient
        
        # Create LLM client
        client = LLMClient(
            engine=self.llm_config.get('engine', 'local'),
            model=self.llm_config.get('model', 'mistral')
        )
        
        # Process segments
        start_time = time.time()
        classifications = []
        
        for segment in self.text_segments:
            try:
                # Classify segment
                classification = client.classify_text(
                    text=segment.get('text', ''),
                    model_config=self.llm_config
                )
                
                classifications.append({
                    'segment_id': segment.get('id', ''),
                    'text': segment.get('text', ''),
                    'classification': classification,
                    'confidence': 0.8  # Placeholder
                })
                
            except Exception as e:
                self.logger.error(f"Error classifying segment: {str(e)}")
                classifications.append({
                    'segment_id': segment.get('id', ''),
                    'text': segment.get('text', ''),
                    'classification': 'ERROR',
                    'error': str(e)
                })
        
        processing_time = time.time() - start_time
        
        return {
            'job_id': job_id,
            'operation_type': 'text_classification',
            'classifications': classifications,
            'total_segments': len(classifications),
            'processing_time': processing_time,
            'llm_config': self.llm_config,
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }
    
    def _process_content_analysis(self, job_id: str) -> Dict[str, Any]:
        """Process content analysis using text preprocessor."""
        from src.text_processing.preprocessor import TextPreprocessor
        
        # Create preprocessor
        preprocessor = TextPreprocessor()
        
        # Combine all text for analysis
        combined_text = '\n'.join(segment.get('text', '') for segment in self.text_segments)
        
        # Analyze content
        start_time = time.time()
        analysis = preprocessor.analyze(combined_text)
        processing_time = time.time() - start_time
        
        return {
            'job_id': job_id,
            'operation_type': 'content_analysis',
            'analysis': analysis,
            'total_segments': len(self.text_segments),
            'processing_time': processing_time,
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }
    
    def _process_distributed_inference(self, job_id: str) -> Dict[str, Any]:
        """Process distributed inference using LLM pool."""
        from src.llm_pool.llm_pool_manager import get_pool_manager
        
        # Get pool manager
        pool_manager = get_pool_manager()
        
        # Process segments using pool
        start_time = time.time()
        processed_segments = []
        
        for segment in self.text_segments:
            try:
                # Get available client
                client = pool_manager.get_client()
                
                # Process segment
                result = client.process_segment(
                    segment=segment,
                    processing_options=self.processing_options
                )
                
                processed_segments.append(result)
                
                # Return client to pool
                pool_manager.return_client(client)
                
            except Exception as e:
                self.logger.error(f"Error in distributed inference: {str(e)}")
                # Add segment with error marker
                error_segment = segment.copy()
                error_segment['speaker'] = 'INFERENCE_ERROR'
                error_segment['error'] = str(e)
                processed_segments.append(error_segment)
        
        processing_time = time.time() - start_time
        
        # Get pool status
        pool_status = pool_manager.get_pool_status()
        
        return {
            'job_id': job_id,
            'operation_type': 'distributed_inference',
            'processed_segments': processed_segments,
            'total_segments': len(processed_segments),
            'processing_time': processing_time,
            'pool_status': pool_status,
            'llm_config': self.llm_config,
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }


class BatchLLMProcessingOperator(BaseOperator):
    """
    Operator for batch LLM processing operations.
    
    This operator processes multiple batches of text segments in parallel.
    """
    
    template_fields = ['text_batches', 'llm_config', 'processing_options']
    template_ext = ['.json']
    ui_color = '#bcbd22'
    
    @apply_defaults
    def __init__(
        self,
        text_batches: List[List[Dict[str, Any]]],
        llm_config: Dict[str, Any] = None,
        processing_options: Dict[str, Any] = None,
        operation_type: str = "speaker_attribution",
        batch_size: int = 10,
        max_workers: int = 4,
        output_key: str = "batch_llm_result",
        timeout_seconds: int = 3600,
        *args,
        **kwargs
    ):
        """
        Initialize the BatchLLMProcessingOperator.
        
        Args:
            text_batches: List of text segment batches
            llm_config: LLM configuration
            processing_options: Processing configuration options
            operation_type: Type of LLM operation
            batch_size: Size of each processing batch
            max_workers: Maximum number of parallel workers
            output_key: Key for storing output in XCom
            timeout_seconds: Timeout for batch operation
        """
        super().__init__(*args, **kwargs)
        
        self.text_batches = text_batches
        self.llm_config = llm_config or {}
        self.processing_options = processing_options or {}
        self.operation_type = operation_type
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.output_key = output_key
        self.timeout_seconds = timeout_seconds
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the batch LLM processing operation.
        
        Args:
            context: Airflow execution context
            
        Returns:
            Dictionary containing batch processing results
        """
        try:
            job_id = context['dag_run'].run_id
            self.logger.info(f"Starting batch LLM processing for job: {job_id}")
            
            # Import concurrent processing
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # Process batches in parallel
            start_time = time.time()
            all_results = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(self._process_batch, i, batch, job_id): i
                    for i, batch in enumerate(self.text_batches)
                }
                
                # Collect results
                for future in as_completed(future_to_batch):
                    batch_index = future_to_batch[future]
                    try:
                        batch_result = future.result()
                        all_results.append(batch_result)
                        self.logger.info(f"Batch {batch_index} completed successfully")
                    except Exception as e:
                        self.logger.error(f"Batch {batch_index} failed: {str(e)}")
                        all_results.append({
                            'batch_index': batch_index,
                            'error': str(e),
                            'status': 'failed'
                        })
            
            processing_time = time.time() - start_time
            
            # Aggregate results
            total_segments = sum(len(batch) for batch in self.text_batches)
            successful_batches = sum(1 for result in all_results if result.get('status') == 'completed')
            
            result = {
                'job_id': job_id,
                'operation_type': self.operation_type,
                'batch_results': all_results,
                'total_batches': len(self.text_batches),
                'successful_batches': successful_batches,
                'total_segments': total_segments,
                'processing_time': processing_time,
                'llm_config': self.llm_config,
                'status': 'completed',
                'timestamp': datetime.now().isoformat()
            }
            
            # Store results in XCom
            self.xcom_push(context, key=self.output_key, value=result)
            
            self.logger.info(
                f"Batch LLM processing completed: {successful_batches}/{len(self.text_batches)} batches successful"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Batch LLM processing operation failed: {str(e)}")
            
            # Store error information
            error_result = {
                'job_id': context['dag_run'].run_id,
                'operation_type': self.operation_type,
                'error_message': str(e),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }
            
            self.xcom_push(context, key=f"{self.output_key}_error", value=error_result)
            
            raise AirflowException(f"Batch LLM processing operation failed: {str(e)}")
    
    def _process_batch(self, batch_index: int, batch: List[Dict[str, Any]], job_id: str) -> Dict[str, Any]:
        """Process a single batch of text segments."""
        try:
            # Create individual LLM processing operator
            llm_operator = LLMProcessingOperator(
                task_id=f"batch_{batch_index}",
                text_segments=batch,
                llm_config=self.llm_config,
                processing_options=self.processing_options,
                operation_type=self.operation_type
            )
            
            # Create mock context
            mock_context = {
                'dag_run': type('obj', (object,), {'run_id': f"{job_id}_batch_{batch_index}"})()
            }
            
            # Execute batch processing
            result = llm_operator.execute(mock_context)
            
            return {
                'batch_index': batch_index,
                'batch_size': len(batch),
                'result': result,
                'status': 'completed'
            }
            
        except Exception as e:
            self.logger.error(f"Error processing batch {batch_index}: {str(e)}")
            return {
                'batch_index': batch_index,
                'batch_size': len(batch),
                'error': str(e),
                'status': 'failed'
            }


class LLMHealthCheckOperator(BaseOperator):
    """
    Operator for checking LLM system health.
    
    This operator performs health checks on LLM components before processing.
    """
    
    template_fields = ['llm_config']
    template_ext = ['.json']
    ui_color = '#8c564b'
    
    @apply_defaults
    def __init__(
        self,
        llm_config: Dict[str, Any] = None,
        check_components: List[str] = None,
        output_key: str = "llm_health_result",
        *args,
        **kwargs
    ):
        """
        Initialize the LLMHealthCheckOperator.
        
        Args:
            llm_config: LLM configuration
            check_components: List of components to check
            output_key: Key for storing output in XCom
        """
        super().__init__(*args, **kwargs)
        
        self.llm_config = llm_config or {}
        self.check_components = check_components or ['orchestrator', 'pool', 'client']
        self.output_key = output_key
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the LLM health check operation.
        
        Args:
            context: Airflow execution context
            
        Returns:
            Dictionary containing health check results
        """
        try:
            job_id = context['dag_run'].run_id
            self.logger.info(f"Starting LLM health check for job: {job_id}")
            
            # Import health check components
            import sys
            sys.path.append('/opt/airflow/dags')
            
            health_results = {}
            
            # Check orchestrator
            if 'orchestrator' in self.check_components:
                health_results['orchestrator'] = self._check_orchestrator_health()
            
            # Check pool manager
            if 'pool' in self.check_components:
                health_results['pool'] = self._check_pool_health()
            
            # Check client
            if 'client' in self.check_components:
                health_results['client'] = self._check_client_health()
            
            # Determine overall health
            overall_healthy = all(
                result.get('healthy', False) for result in health_results.values()
            )
            
            result = {
                'job_id': job_id,
                'health_results': health_results,
                'overall_healthy': overall_healthy,
                'check_components': self.check_components,
                'llm_config': self.llm_config,
                'status': 'completed',
                'timestamp': datetime.now().isoformat()
            }
            
            # Store results in XCom
            self.xcom_push(context, key=self.output_key, value=result)
            
            if overall_healthy:
                self.logger.info("LLM health check passed")
            else:
                self.logger.warning("LLM health check failed")
                raise AirflowException("LLM health check failed")
            
            return result
            
        except Exception as e:
            self.logger.error(f"LLM health check operation failed: {str(e)}")
            
            # Store error information
            error_result = {
                'job_id': context['dag_run'].run_id,
                'error_message': str(e),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }
            
            self.xcom_push(context, key=f"{self.output_key}_error", value=error_result)
            
            raise AirflowException(f"LLM health check operation failed: {str(e)}")
    
    def _check_orchestrator_health(self) -> Dict[str, Any]:
        """Check orchestrator health."""
        try:
            from src.attribution.llm.orchestrator import LLMOrchestrator
            
            orchestrator = LLMOrchestrator(
                engine=self.llm_config.get('engine', 'local'),
                model=self.llm_config.get('model', 'mistral')
            )
            
            # Test basic functionality
            test_result = orchestrator.test_connection()
            
            return {
                'healthy': test_result.get('success', False),
                'details': test_result,
                'component': 'orchestrator'
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'component': 'orchestrator'
            }
    
    def _check_pool_health(self) -> Dict[str, Any]:
        """Check pool manager health."""
        try:
            from src.llm_pool.llm_pool_manager import get_pool_manager
            
            pool_manager = get_pool_manager()
            pool_status = pool_manager.get_pool_status()
            
            return {
                'healthy': pool_status.get('healthy_instances', 0) > 0,
                'details': pool_status,
                'component': 'pool'
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'component': 'pool'
            }
    
    def _check_client_health(self) -> Dict[str, Any]:
        """Check LLM client health."""
        try:
            from src.llm_pool.llm_client import LLMClient
            
            client = LLMClient(
                engine=self.llm_config.get('engine', 'local'),
                model=self.llm_config.get('model', 'mistral')
            )
            
            health_status = client.health_check()
            
            return {
                'healthy': health_status.get('overall_health') == 'healthy',
                'details': health_status,
                'component': 'client'
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'component': 'client'
            }


# Utility functions for creating operators
def create_llm_processing_operator(
    task_id: str,
    text_segments: List[Dict[str, Any]],
    llm_config: Dict[str, Any] = None,
    processing_options: Dict[str, Any] = None,
    operation_type: str = "speaker_attribution",
    dag=None
) -> LLMProcessingOperator:
    """
    Factory function for creating LLMProcessingOperator instances.
    
    Args:
        task_id: Unique task identifier
        text_segments: Text segments to process
        llm_config: LLM configuration
        processing_options: Processing options
        operation_type: Type of operation
        dag: DAG instance
        
    Returns:
        Configured LLMProcessingOperator instance
    """
    return LLMProcessingOperator(
        task_id=task_id,
        text_segments=text_segments,
        llm_config=llm_config,
        processing_options=processing_options,
        operation_type=operation_type,
        dag=dag
    )


def create_batch_llm_processing_operator(
    task_id: str,
    text_batches: List[List[Dict[str, Any]]],
    llm_config: Dict[str, Any] = None,
    processing_options: Dict[str, Any] = None,
    operation_type: str = "speaker_attribution",
    batch_size: int = 10,
    max_workers: int = 4,
    dag=None
) -> BatchLLMProcessingOperator:
    """
    Factory function for creating BatchLLMProcessingOperator instances.
    
    Args:
        task_id: Unique task identifier
        text_batches: Text segment batches
        llm_config: LLM configuration
        processing_options: Processing options
        operation_type: Type of operation
        batch_size: Batch size
        max_workers: Maximum workers
        dag: DAG instance
        
    Returns:
        Configured BatchLLMProcessingOperator instance
    """
    return BatchLLMProcessingOperator(
        task_id=task_id,
        text_batches=text_batches,
        llm_config=llm_config,
        processing_options=processing_options,
        operation_type=operation_type,
        batch_size=batch_size,
        max_workers=max_workers,
        dag=dag
    )


def create_llm_health_check_operator(
    task_id: str,
    llm_config: Dict[str, Any] = None,
    check_components: List[str] = None,
    dag=None
) -> LLMHealthCheckOperator:
    """
    Factory function for creating LLMHealthCheckOperator instances.
    
    Args:
        task_id: Unique task identifier
        llm_config: LLM configuration
        check_components: Components to check
        dag: DAG instance
        
    Returns:
        Configured LLMHealthCheckOperator instance
    """
    return LLMHealthCheckOperator(
        task_id=task_id,
        llm_config=llm_config,
        check_components=check_components,
        dag=dag
    )