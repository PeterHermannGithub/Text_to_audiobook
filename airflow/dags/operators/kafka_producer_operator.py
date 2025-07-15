"""
Custom Airflow operator for Kafka producer operations.

This operator provides integration between Airflow and Kafka producers
for event-driven processing in the text-to-audiobook pipeline.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.exceptions import AirflowException
from airflow.hooks.base import BaseHook


class KafkaProducerOperator(BaseOperator):
    """
    Custom operator for sending messages to Kafka topics.
    
    This operator integrates with the Kafka producers to send messages
    for event-driven processing workflows.
    """
    
    template_fields = ['message_data', 'topic', 'key']
    template_ext = ['.json']
    ui_color = '#1f77b4'
    
    @apply_defaults
    def __init__(
        self,
        topic: str,
        message_data: Dict[str, Any],
        key: Optional[str] = None,
        producer_config: Dict[str, Any] = None,
        message_type: str = "generic",
        output_key: str = "kafka_result",
        conn_id: str = "kafka_default",
        *args,
        **kwargs
    ):
        """
        Initialize the KafkaProducerOperator.
        
        Args:
            topic: Kafka topic to send message to
            message_data: Message data to send
            key: Message key for partitioning
            producer_config: Kafka producer configuration
            message_type: Type of message being sent
            output_key: Key for storing output in XCom
            conn_id: Airflow connection ID for Kafka
        """
        super().__init__(*args, **kwargs)
        
        self.topic = topic
        self.message_data = message_data
        self.key = key
        self.producer_config = producer_config or {}
        self.message_type = message_type
        self.output_key = output_key
        self.conn_id = conn_id
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the Kafka producer operation.
        
        Args:
            context: Airflow execution context
            
        Returns:
            Dictionary containing producer result information
        """
        try:
            job_id = context['dag_run'].run_id
            self.logger.info(f"Sending message to Kafka topic '{self.topic}' for job {job_id}")
            
            # Import Kafka producer
            import sys
            sys.path.append('/opt/airflow/dags')
            from src.kafka.producers.file_upload_producer import FileUploadProducer
            from src.kafka.producers.chunk_producer import ChunkProducer
            
            # Create producer based on message type
            if self.message_type == "file_upload":
                producer = FileUploadProducer(self.producer_config)
            elif self.message_type == "chunk_processing":
                producer = ChunkProducer(self.producer_config)
            else:
                # Generic producer
                from kafka import KafkaProducer
                from src.kafka.kafka_config import KafkaConfig
                
                producer_config = KafkaConfig.get_producer_config()
                producer_config.update(self.producer_config)
                producer = KafkaProducer(**producer_config)
            
            # Send message
            start_time = datetime.now()
            
            if hasattr(producer, '_send_message'):
                # Use custom producer _send_message method
                success = producer._send_message(
                    topic=self.topic,
                    message=self.message_data,
                    key=self.key or job_id
                )
            else:
                # Use standard Kafka producer
                future = producer.send(
                    topic=self.topic,
                    value=self.message_data,
                    key=self.key or job_id
                )
                
                # Wait for confirmation
                record_metadata = future.get(timeout=30)
                success = True
                
                self.logger.info(
                    f"Message sent to {self.topic}: partition={record_metadata.partition}, "
                    f"offset={record_metadata.offset}"
                )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare result
            result = {
                'topic': self.topic,
                'message_type': self.message_type,
                'key': self.key or job_id,
                'job_id': job_id,
                'success': success,
                'processing_time': processing_time,
                'message_size': len(json.dumps(self.message_data)),
                'timestamp': datetime.now().isoformat(),
                'status': 'sent' if success else 'failed'
            }
            
            # Close producer
            if hasattr(producer, 'close'):
                producer.close()
            
            # Store results in XCom
            self.xcom_push(context, key=self.output_key, value=result)
            
            self.logger.info(
                f"Kafka message sent successfully to '{self.topic}' in {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Kafka producer operation failed: {str(e)}")
            
            # Store error information
            error_result = {
                'topic': self.topic,
                'message_type': self.message_type,
                'job_id': context['dag_run'].run_id,
                'success': False,
                'error_message': str(e),
                'timestamp': datetime.now().isoformat(),
                'status': 'failed'
            }
            
            self.xcom_push(context, key=f"{self.output_key}_error", value=error_result)
            
            raise AirflowException(f"Kafka producer operation failed: {str(e)}")


class FileUploadProducerOperator(BaseOperator):
    """
    Specialized operator for file upload events.
    
    This operator uses the FileUploadProducer to handle file processing
    events in the text-to-audiobook pipeline.
    """
    
    template_fields = ['file_path', 'user_id', 'processing_options']
    template_ext = ['.json']
    ui_color = '#ff7f0e'
    
    @apply_defaults
    def __init__(
        self,
        file_path: str,
        user_id: str,
        processing_options: Dict[str, Any] = None,
        priority: int = 0,
        producer_config: Dict[str, Any] = None,
        output_key: str = "file_upload_result",
        *args,
        **kwargs
    ):
        """
        Initialize the FileUploadProducerOperator.
        
        Args:
            file_path: Path to the file to process
            user_id: User ID for the processing request
            processing_options: Processing configuration options
            priority: Processing priority
            producer_config: Kafka producer configuration
            output_key: Key for storing output in XCom
        """
        super().__init__(*args, **kwargs)
        
        self.file_path = file_path
        self.user_id = user_id
        self.processing_options = processing_options or {}
        self.priority = priority
        self.producer_config = producer_config or {}
        self.output_key = output_key
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the file upload producer operation.
        
        Args:
            context: Airflow execution context
            
        Returns:
            Dictionary containing file upload result
        """
        try:
            job_id = context['dag_run'].run_id
            self.logger.info(f"Submitting file for processing: {self.file_path}")
            
            # Import FileUploadProducer
            import sys
            sys.path.append('/opt/airflow/dags')
            from src.kafka.producers.file_upload_producer import FileUploadProducer
            
            # Create and use producer
            with FileUploadProducer(self.producer_config) as producer:
                
                # Submit file for processing
                returned_job_id = producer.submit_file_for_processing(
                    file_path=self.file_path,
                    user_id=self.user_id,
                    processing_options=self.processing_options,
                    priority=self.priority
                )
                
                # Get producer metrics
                producer_metrics = producer.get_producer_metrics()
                
                # Prepare result
                result = {
                    'file_path': self.file_path,
                    'user_id': self.user_id,
                    'job_id': returned_job_id,
                    'airflow_job_id': job_id,
                    'priority': self.priority,
                    'processing_options': self.processing_options,
                    'producer_metrics': producer_metrics,
                    'status': 'submitted',
                    'timestamp': datetime.now().isoformat()
                }
                
                # Store results in XCom
                self.xcom_push(context, key=self.output_key, value=result)
                
                self.logger.info(
                    f"File upload submitted successfully: {self.file_path} -> {returned_job_id}"
                )
                
                return result
                
        except Exception as e:
            self.logger.error(f"File upload producer operation failed: {str(e)}")
            
            # Store error information
            error_result = {
                'file_path': self.file_path,
                'user_id': self.user_id,
                'job_id': context['dag_run'].run_id,
                'error_message': str(e),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }
            
            self.xcom_push(context, key=f"{self.output_key}_error", value=error_result)
            
            raise AirflowException(f"File upload producer operation failed: {str(e)}")


class ChunkProducerOperator(BaseOperator):
    """
    Specialized operator for chunk processing events.
    
    This operator uses the ChunkProducer to handle text chunk distribution
    in the distributed processing pipeline.
    """
    
    template_fields = ['text_content', 'processing_config']
    template_ext = ['.json']
    ui_color = '#2ca02c'
    
    @apply_defaults
    def __init__(
        self,
        text_content: str,
        processing_config: Dict[str, Any] = None,
        priority: int = 0,
        producer_config: Dict[str, Any] = None,
        output_key: str = "chunk_processing_result",
        *args,
        **kwargs
    ):
        """
        Initialize the ChunkProducerOperator.
        
        Args:
            text_content: Text content to process
            processing_config: Processing configuration
            priority: Processing priority
            producer_config: Kafka producer configuration
            output_key: Key for storing output in XCom
        """
        super().__init__(*args, **kwargs)
        
        self.text_content = text_content
        self.processing_config = processing_config or {}
        self.priority = priority
        self.producer_config = producer_config or {}
        self.output_key = output_key
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the chunk producer operation.
        
        Args:
            context: Airflow execution context
            
        Returns:
            Dictionary containing chunk processing result
        """
        try:
            job_id = context['dag_run'].run_id
            self.logger.info(f"Processing text chunks for job: {job_id}")
            
            # Import ChunkProducer
            import sys
            sys.path.append('/opt/airflow/dags')
            from src.kafka.producers.chunk_producer import DistributedChunkProcessor
            
            # Create and use processor
            with DistributedChunkProcessor(self.producer_config) as processor:
                
                # Process text distributed
                returned_job_id = processor.process_text_distributed(
                    text_content=self.text_content,
                    job_id=job_id,
                    processing_config=self.processing_config,
                    priority=self.priority
                )
                
                # Get processor metrics
                processor_metrics = processor.chunk_producer.get_metrics()
                
                # Prepare result
                result = {
                    'text_content_length': len(self.text_content),
                    'job_id': returned_job_id,
                    'airflow_job_id': job_id,
                    'priority': self.priority,
                    'processing_config': self.processing_config,
                    'processor_metrics': processor_metrics,
                    'status': 'submitted',
                    'timestamp': datetime.now().isoformat()
                }
                
                # Store results in XCom
                self.xcom_push(context, key=self.output_key, value=result)
                
                self.logger.info(
                    f"Chunk processing submitted successfully: {len(self.text_content)} chars -> {returned_job_id}"
                )
                
                return result
                
        except Exception as e:
            self.logger.error(f"Chunk producer operation failed: {str(e)}")
            
            # Store error information
            error_result = {
                'text_content_length': len(self.text_content),
                'job_id': context['dag_run'].run_id,
                'error_message': str(e),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }
            
            self.xcom_push(context, key=f"{self.output_key}_error", value=error_result)
            
            raise AirflowException(f"Chunk producer operation failed: {str(e)}")


class BatchFileUploadOperator(BaseOperator):
    """
    Operator for batch file upload operations.
    
    This operator handles multiple file uploads in a single operation.
    """
    
    template_fields = ['file_paths', 'user_id', 'processing_options']
    template_ext = ['.json']
    ui_color = '#d62728'
    
    @apply_defaults
    def __init__(
        self,
        file_paths: List[str],
        user_id: str,
        processing_options: Dict[str, Any] = None,
        priority: int = 0,
        producer_config: Dict[str, Any] = None,
        output_key: str = "batch_upload_result",
        *args,
        **kwargs
    ):
        """
        Initialize the BatchFileUploadOperator.
        
        Args:
            file_paths: List of file paths to process
            user_id: User ID for the processing request
            processing_options: Processing configuration options
            priority: Processing priority
            producer_config: Kafka producer configuration
            output_key: Key for storing output in XCom
        """
        super().__init__(*args, **kwargs)
        
        self.file_paths = file_paths
        self.user_id = user_id
        self.processing_options = processing_options or {}
        self.priority = priority
        self.producer_config = producer_config or {}
        self.output_key = output_key
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the batch file upload operation.
        
        Args:
            context: Airflow execution context
            
        Returns:
            Dictionary containing batch upload result
        """
        try:
            job_id = context['dag_run'].run_id
            self.logger.info(f"Submitting {len(self.file_paths)} files for batch processing")
            
            # Import FileUploadProducer
            import sys
            sys.path.append('/opt/airflow/dags')
            from src.kafka.producers.file_upload_producer import FileUploadProducer
            
            # Create and use producer
            with FileUploadProducer(self.producer_config) as producer:
                
                # Submit batch files
                job_ids = producer.submit_batch_files(
                    file_paths=self.file_paths,
                    user_id=self.user_id,
                    processing_options=self.processing_options,
                    priority=self.priority
                )
                
                # Get producer metrics
                producer_metrics = producer.get_producer_metrics()
                
                # Prepare result
                result = {
                    'file_paths': self.file_paths,
                    'user_id': self.user_id,
                    'job_ids': job_ids,
                    'airflow_job_id': job_id,
                    'priority': self.priority,
                    'processing_options': self.processing_options,
                    'producer_metrics': producer_metrics,
                    'total_files': len(self.file_paths),
                    'successful_submissions': len(job_ids),
                    'status': 'submitted',
                    'timestamp': datetime.now().isoformat()
                }
                
                # Store results in XCom
                self.xcom_push(context, key=self.output_key, value=result)
                
                self.logger.info(
                    f"Batch file upload completed: {len(job_ids)}/{len(self.file_paths)} files submitted"
                )
                
                return result
                
        except Exception as e:
            self.logger.error(f"Batch file upload operation failed: {str(e)}")
            
            # Store error information
            error_result = {
                'file_paths': self.file_paths,
                'user_id': self.user_id,
                'job_id': context['dag_run'].run_id,
                'error_message': str(e),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }
            
            self.xcom_push(context, key=f"{self.output_key}_error", value=error_result)
            
            raise AirflowException(f"Batch file upload operation failed: {str(e)}")


# Utility functions for creating operators
def create_kafka_producer_operator(
    task_id: str,
    topic: str,
    message_data: Dict[str, Any],
    key: Optional[str] = None,
    message_type: str = "generic",
    dag=None
) -> KafkaProducerOperator:
    """
    Factory function for creating KafkaProducerOperator instances.
    
    Args:
        task_id: Unique task identifier
        topic: Kafka topic
        message_data: Message data
        key: Message key
        message_type: Message type
        dag: DAG instance
        
    Returns:
        Configured KafkaProducerOperator instance
    """
    return KafkaProducerOperator(
        task_id=task_id,
        topic=topic,
        message_data=message_data,
        key=key,
        message_type=message_type,
        dag=dag
    )


def create_file_upload_producer_operator(
    task_id: str,
    file_path: str,
    user_id: str,
    processing_options: Dict[str, Any] = None,
    priority: int = 0,
    dag=None
) -> FileUploadProducerOperator:
    """
    Factory function for creating FileUploadProducerOperator instances.
    
    Args:
        task_id: Unique task identifier
        file_path: File path to process
        user_id: User ID
        processing_options: Processing options
        priority: Processing priority
        dag: DAG instance
        
    Returns:
        Configured FileUploadProducerOperator instance
    """
    return FileUploadProducerOperator(
        task_id=task_id,
        file_path=file_path,
        user_id=user_id,
        processing_options=processing_options,
        priority=priority,
        dag=dag
    )