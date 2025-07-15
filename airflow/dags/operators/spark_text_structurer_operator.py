"""
Custom Airflow operator for Spark text structuring.

This operator provides integration between Airflow and the SparkTextStructurer
for distributed text processing in the text-to-audiobook pipeline.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.exceptions import AirflowException
from airflow.hooks.base import BaseHook
from airflow.providers.spark.hooks.spark_submit import SparkSubmitHook

from pyspark.sql import SparkSession


class SparkTextStructurerOperator(BaseOperator):
    """
    Custom operator for distributed text structuring using Spark.
    
    This operator integrates with the SparkTextStructurer to process text
    using distributed computing resources.
    """
    
    template_fields = ['text_content', 'processing_options', 'spark_config']
    template_ext = ['.json']
    ui_color = '#ff7f0e'
    
    @apply_defaults
    def __init__(
        self,
        text_content: str,
        processing_options: Dict[str, Any] = None,
        spark_environment: str = "local",
        spark_config: Dict[str, Any] = None,
        output_key: str = "structured_segments",
        conn_id: str = "spark_default",
        *args,
        **kwargs
    ):
        """
        Initialize the SparkTextStructurerOperator.
        
        Args:
            text_content: The text content to process
            processing_options: Processing configuration options
            spark_environment: Spark environment ("local", "cluster", "kubernetes")
            spark_config: Additional Spark configuration
            output_key: Key for storing output in XCom
            conn_id: Airflow connection ID for Spark
        """
        super().__init__(*args, **kwargs)
        
        self.text_content = text_content
        self.processing_options = processing_options or {}
        self.spark_environment = spark_environment
        self.spark_config = spark_config or {}
        self.output_key = output_key
        self.conn_id = conn_id
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the Spark text structuring operation.
        
        Args:
            context: Airflow execution context
            
        Returns:
            Dictionary containing structured segments and processing metrics
        """
        try:
            job_id = context['dag_run'].run_id
            self.logger.info(f"Starting Spark text structuring for job {job_id}")
            
            # Import SparkTextStructurer
            import sys
            sys.path.append('/opt/airflow/dags')
            from src.spark.spark_text_structurer import SparkTextStructurerContext
            
            # Create Spark configuration
            spark_config = self._create_spark_config()
            
            # Initialize and execute text structuring
            with SparkTextStructurerContext(
                environment=self.spark_environment,
                config=spark_config
            ) as structurer:
                
                # Log system health before processing
                health_status = structurer.health_check()
                self.logger.info(f"System health: {health_status}")
                
                if health_status['overall_health'] != 'healthy':
                    raise AirflowException(f"System health check failed: {health_status}")
                
                # Process text with distributed Spark processing
                start_time = datetime.now()
                
                structured_segments = structurer.structure_text(
                    text_content=self.text_content,
                    processing_options=self.processing_options
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Get processing metrics
                processing_metrics = structurer.get_processing_metrics()
                processing_metrics['airflow_processing_time'] = processing_time
                
                # Prepare result
                result = {
                    'structured_segments': structured_segments,
                    'processing_metrics': processing_metrics,
                    'job_id': job_id,
                    'spark_environment': self.spark_environment,
                    'processing_time': processing_time,
                    'total_segments': len(structured_segments),
                    'unique_speakers': len(set(seg['speaker'] for seg in structured_segments)),
                    'status': 'completed',
                    'timestamp': datetime.now().isoformat()
                }
                
                # Store results in XCom
                self.xcom_push(context, key=self.output_key, value=result)
                
                self.logger.info(
                    f"Spark text structuring completed: {len(structured_segments)} segments, "
                    f"{processing_time:.2f}s processing time"
                )
                
                return result
                
        except Exception as e:
            self.logger.error(f"Spark text structuring failed: {str(e)}")
            
            # Store error information
            error_result = {
                'status': 'failed',
                'error_message': str(e),
                'job_id': context['dag_run'].run_id,
                'timestamp': datetime.now().isoformat()
            }
            
            self.xcom_push(context, key=f"{self.output_key}_error", value=error_result)
            
            raise AirflowException(f"Spark text structuring failed: {str(e)}")
    
    def _create_spark_config(self) -> Dict[str, Any]:
        """Create Spark configuration from operator parameters."""
        config = {
            'spark': {
                'app.name': f'text-to-audiobook-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                'sql.adaptive.enabled': 'true',
                'sql.adaptive.coalescePartitions.enabled': 'true',
                'dynamicAllocation.enabled': 'true',
                'dynamicAllocation.minExecutors': '1',
                'dynamicAllocation.maxExecutors': '10',
                'executor.memory': '2g',
                'executor.cores': '2',
                'driver.memory': '1g',
                'driver.maxResultSize': '1g'
            },
            'processing': self.processing_options,
            'environment': self.spark_environment
        }
        
        # Merge with provided spark_config
        if self.spark_config:
            config.update(self.spark_config)
        
        return config
    
    def on_kill(self):
        """Handle task cancellation."""
        self.logger.info("Spark text structuring task cancelled")
        # Add cleanup logic here if needed


class SparkTextStructurerSubmitOperator(BaseOperator):
    """
    Alternative operator that uses Spark Submit for processing.
    
    This operator submits a Spark job using spark-submit for environments
    where direct Spark integration is not available.
    """
    
    template_fields = ['application_args', 'spark_config']
    template_ext = ['.json']
    ui_color = '#ff9900'
    
    @apply_defaults
    def __init__(
        self,
        application: str,
        text_content: str,
        processing_options: Dict[str, Any] = None,
        spark_config: Dict[str, Any] = None,
        output_key: str = "structured_segments",
        conn_id: str = "spark_default",
        *args,
        **kwargs
    ):
        """
        Initialize the SparkTextStructurerSubmitOperator.
        
        Args:
            application: Path to the Spark application script
            text_content: The text content to process
            processing_options: Processing configuration options
            spark_config: Spark configuration
            output_key: Key for storing output in XCom
            conn_id: Airflow connection ID for Spark
        """
        super().__init__(*args, **kwargs)
        
        self.application = application
        self.text_content = text_content
        self.processing_options = processing_options or {}
        self.spark_config = spark_config or {}
        self.output_key = output_key
        self.conn_id = conn_id
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the Spark job using spark-submit.
        
        Args:
            context: Airflow execution context
            
        Returns:
            Dictionary containing structured segments and processing metrics
        """
        try:
            job_id = context['dag_run'].run_id
            self.logger.info(f"Submitting Spark job for text structuring: {job_id}")
            
            # Prepare application arguments
            application_args = [
                '--text-content', self.text_content,
                '--processing-options', json.dumps(self.processing_options),
                '--job-id', job_id,
                '--output-key', self.output_key
            ]
            
            # Create Spark submit hook
            spark_submit_hook = SparkSubmitHook(
                conn_id=self.conn_id,
                application=self.application,
                application_args=application_args,
                conf=self.spark_config
            )
            
            # Submit and monitor the job
            spark_submit_hook.submit()
            
            # Get results (this would need to be implemented based on your output mechanism)
            result = self._get_job_results(job_id)
            
            # Store results in XCom
            self.xcom_push(context, key=self.output_key, value=result)
            
            self.logger.info(f"Spark job completed successfully: {job_id}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Spark job submission failed: {str(e)}")
            raise AirflowException(f"Spark job submission failed: {str(e)}")
    
    def _get_job_results(self, job_id: str) -> Dict[str, Any]:
        """
        Get results from completed Spark job.
        
        Args:
            job_id: Job ID for the Spark job
            
        Returns:
            Dictionary containing job results
        """
        # This is a placeholder implementation
        # In practice, you would read results from a shared storage system
        # like HDFS, S3, or a database
        
        import os
        import tempfile
        
        results_file = os.path.join(tempfile.gettempdir(), f"spark_results_{job_id}.json")
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                return json.load(f)
        else:
            raise AirflowException(f"Results file not found: {results_file}")


class DistributedTextProcessingOperator(BaseOperator):
    """
    High-level operator for distributed text processing.
    
    This operator provides a simplified interface for common text processing
    workflows using the distributed architecture.
    """
    
    template_fields = ['input_file_path', 'processing_config']
    template_ext = ['.json']
    ui_color = '#2ca02c'
    
    @apply_defaults
    def __init__(
        self,
        input_file_path: str,
        processing_config: Dict[str, Any] = None,
        output_format: str = "json",
        output_key: str = "processing_result",
        *args,
        **kwargs
    ):
        """
        Initialize the DistributedTextProcessingOperator.
        
        Args:
            input_file_path: Path to the input file
            processing_config: Complete processing configuration
            output_format: Output format ("json", "csv", "parquet")
            output_key: Key for storing output in XCom
        """
        super().__init__(*args, **kwargs)
        
        self.input_file_path = input_file_path
        self.processing_config = processing_config or {}
        self.output_format = output_format
        self.output_key = output_key
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete distributed text processing pipeline.
        
        Args:
            context: Airflow execution context
            
        Returns:
            Dictionary containing complete processing results
        """
        try:
            job_id = context['dag_run'].run_id
            self.logger.info(f"Starting distributed text processing: {job_id}")
            
            # Import required modules
            import sys
            sys.path.append('/opt/airflow/dags')
            
            from src.text_processing.text_extractor import TextExtractor
            from src.spark.spark_text_structurer import SparkTextStructurerContext
            from src.output.output_formatter import OutputFormatter
            
            # Phase 1: Text extraction
            self.logger.info("Phase 1: Text extraction")
            extractor = TextExtractor()
            extracted_text = extractor.extract(self.input_file_path)
            
            # Phase 2: Distributed processing
            self.logger.info("Phase 2: Distributed processing")
            with SparkTextStructurerContext(
                environment=self.processing_config.get('spark_environment', 'local'),
                config=self.processing_config
            ) as structurer:
                
                structured_segments = structurer.structure_text(
                    text_content=extracted_text,
                    processing_options=self.processing_config
                )
                
                processing_metrics = structurer.get_processing_metrics()
            
            # Phase 3: Output formatting
            self.logger.info("Phase 3: Output formatting")
            formatter = OutputFormatter()
            formatted_output = formatter.format_segments(structured_segments)
            
            # Prepare final result
            result = {
                'input_file_path': self.input_file_path,
                'job_id': job_id,
                'structured_segments': formatted_output,
                'processing_metrics': processing_metrics,
                'total_segments': len(formatted_output),
                'unique_speakers': len(set(seg['speaker'] for seg in formatted_output)),
                'output_format': self.output_format,
                'status': 'completed',
                'timestamp': datetime.now().isoformat()
            }
            
            # Store results in XCom
            self.xcom_push(context, key=self.output_key, value=result)
            
            self.logger.info(
                f"Distributed text processing completed: {len(formatted_output)} segments"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Distributed text processing failed: {str(e)}")
            raise AirflowException(f"Distributed text processing failed: {str(e)}")


# Utility functions for creating operators
def create_spark_text_structurer_operator(
    task_id: str,
    text_content: str,
    processing_options: Dict[str, Any] = None,
    spark_environment: str = "local",
    dag=None
) -> SparkTextStructurerOperator:
    """
    Factory function for creating SparkTextStructurerOperator instances.
    
    Args:
        task_id: Unique task identifier
        text_content: Text content to process
        processing_options: Processing configuration
        spark_environment: Spark environment
        dag: DAG instance
        
    Returns:
        Configured SparkTextStructurerOperator instance
    """
    return SparkTextStructurerOperator(
        task_id=task_id,
        text_content=text_content,
        processing_options=processing_options,
        spark_environment=spark_environment,
        dag=dag
    )


def create_distributed_processing_operator(
    task_id: str,
    input_file_path: str,
    processing_config: Dict[str, Any] = None,
    dag=None
) -> DistributedTextProcessingOperator:
    """
    Factory function for creating DistributedTextProcessingOperator instances.
    
    Args:
        task_id: Unique task identifier
        input_file_path: Path to input file
        processing_config: Complete processing configuration
        dag: DAG instance
        
    Returns:
        Configured DistributedTextProcessingOperator instance
    """
    return DistributedTextProcessingOperator(
        task_id=task_id,
        input_file_path=input_file_path,
        processing_config=processing_config,
        dag=dag
    )