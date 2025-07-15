"""
Custom Airflow operator for Kafka consumer operations.

This operator provides integration between Airflow and Kafka consumers
for event-driven processing in the text-to-audiobook pipeline.
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.sensors.base import BaseSensorOperator
from airflow.hooks.base import BaseHook


class KafkaConsumerOperator(BaseOperator):
    """
    Custom operator for consuming messages from Kafka topics.
    
    This operator integrates with the Kafka consumers to process messages
    in event-driven processing workflows.
    """
    
    template_fields = ['topic', 'consumer_config', 'processing_options']
    template_ext = ['.json']
    ui_color = '#9467bd'
    
    @apply_defaults
    def __init__(
        self,
        topic: str,
        consumer_type: str = "text_extraction",
        consumer_config: Dict[str, Any] = None,
        processing_options: Dict[str, Any] = None,
        max_messages: int = 100,
        timeout_seconds: int = 300,
        output_key: str = "consumer_result",
        conn_id: str = "kafka_default",
        *args,
        **kwargs
    ):
        """
        Initialize the KafkaConsumerOperator.
        
        Args:
            topic: Kafka topic to consume from
            consumer_type: Type of consumer ("text_extraction", "llm_classification")
            consumer_config: Kafka consumer configuration
            processing_options: Processing configuration options
            max_messages: Maximum number of messages to process
            timeout_seconds: Timeout for consumer operation
            output_key: Key for storing output in XCom
            conn_id: Airflow connection ID for Kafka
        """
        super().__init__(*args, **kwargs)
        
        self.topic = topic
        self.consumer_type = consumer_type
        self.consumer_config = consumer_config or {}
        self.processing_options = processing_options or {}
        self.max_messages = max_messages
        self.timeout_seconds = timeout_seconds
        self.output_key = output_key
        self.conn_id = conn_id
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the Kafka consumer operation.
        
        Args:
            context: Airflow execution context
            
        Returns:
            Dictionary containing consumer result information
        """
        try:
            job_id = context['dag_run'].run_id
            self.logger.info(f"Starting Kafka consumer for topic '{self.topic}' (job: {job_id})")
            
            # Import appropriate consumer
            import sys
            sys.path.append('/opt/airflow/dags')
            
            consumer = self._create_consumer()
            
            # Start consumer
            consumer.start(num_workers=2)
            
            # Wait for processing to complete or timeout
            start_time = time.time()
            messages_processed = 0
            
            try:
                while (time.time() - start_time) < self.timeout_seconds and messages_processed < self.max_messages:
                    # Check consumer metrics
                    metrics = consumer.get_metrics()
                    messages_processed = metrics.get('messages_processed', 0)
                    
                    if messages_processed >= self.max_messages:
                        self.logger.info(f"Reached max messages limit: {self.max_messages}")
                        break
                    
                    # Sleep briefly before checking again
                    time.sleep(1)
                
                # Get final metrics
                final_metrics = consumer.get_metrics()
                health_status = consumer.health_check()
                
                # Prepare result
                result = {
                    'topic': self.topic,
                    'consumer_type': self.consumer_type,
                    'job_id': job_id,
                    'messages_processed': final_metrics.get('messages_processed', 0),
                    'messages_failed': final_metrics.get('messages_failed', 0),
                    'processing_time': time.time() - start_time,
                    'health_status': health_status,
                    'final_metrics': final_metrics,
                    'status': 'completed',
                    'timestamp': datetime.now().isoformat()
                }
                
                # Store results in XCom
                self.xcom_push(context, key=self.output_key, value=result)
                
                self.logger.info(
                    f"Kafka consumer completed: {result['messages_processed']} messages processed "
                    f"in {result['processing_time']:.2f}s"
                )
                
                return result
                
            finally:
                # Stop consumer
                consumer.stop()
                
        except Exception as e:
            self.logger.error(f"Kafka consumer operation failed: {str(e)}")
            
            # Store error information
            error_result = {
                'topic': self.topic,
                'consumer_type': self.consumer_type,
                'job_id': context['dag_run'].run_id,
                'error_message': str(e),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }
            
            self.xcom_push(context, key=f"{self.output_key}_error", value=error_result)
            
            raise AirflowException(f"Kafka consumer operation failed: {str(e)}")
    
    def _create_consumer(self):
        """Create the appropriate consumer based on consumer_type."""
        if self.consumer_type == "text_extraction":
            from src.kafka.consumers.text_extraction_consumer import TextExtractionConsumer
            return TextExtractionConsumer(self.consumer_config)
        elif self.consumer_type == "llm_classification":
            from src.kafka.consumers.llm_consumer import LLMConsumer
            return LLMConsumer(self.consumer_config)
        else:
            raise AirflowException(f"Unknown consumer type: {self.consumer_type}")


class TextExtractionConsumerOperator(BaseOperator):
    """
    Specialized operator for text extraction consumer.
    
    This operator uses the TextExtractionConsumer to process file upload
    events and extract text content.
    """
    
    template_fields = ['consumer_config', 'processing_options']
    template_ext = ['.json']
    ui_color = '#ff7f0e'
    
    @apply_defaults
    def __init__(
        self,
        consumer_config: Dict[str, Any] = None,
        processing_options: Dict[str, Any] = None,
        max_messages: int = 50,
        timeout_seconds: int = 600,
        output_key: str = "text_extraction_result",
        *args,
        **kwargs
    ):
        """
        Initialize the TextExtractionConsumerOperator.
        
        Args:
            consumer_config: Kafka consumer configuration
            processing_options: Text extraction processing options
            max_messages: Maximum number of messages to process
            timeout_seconds: Timeout for consumer operation
            output_key: Key for storing output in XCom
        """
        super().__init__(*args, **kwargs)
        
        self.consumer_config = consumer_config or {}
        self.processing_options = processing_options or {}
        self.max_messages = max_messages
        self.timeout_seconds = timeout_seconds
        self.output_key = output_key
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the text extraction consumer operation.
        
        Args:
            context: Airflow execution context
            
        Returns:
            Dictionary containing text extraction results
        """
        try:
            job_id = context['dag_run'].run_id
            self.logger.info(f"Starting text extraction consumer for job: {job_id}")
            
            # Import TextExtractionConsumer
            import sys
            sys.path.append('/opt/airflow/dags')
            from src.kafka.consumers.text_extraction_consumer import TextExtractionConsumer
            
            # Create callback to collect results
            extraction_results = []
            
            def extraction_callback(job_id: str, result_data: Dict[str, Any]):
                extraction_results.append({
                    'job_id': job_id,
                    'result_data': result_data,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Create and start consumer
            with TextExtractionConsumer(self.consumer_config, extraction_callback) as consumer:
                consumer.start(num_workers=2)
                
                # Wait for processing to complete
                start_time = time.time()
                
                while (time.time() - start_time) < self.timeout_seconds:
                    metrics = consumer.get_metrics()
                    
                    if metrics['messages_processed'] >= self.max_messages:
                        self.logger.info(f"Reached max messages limit: {self.max_messages}")
                        break
                    
                    if len(extraction_results) > 0:
                        self.logger.info(f"Received {len(extraction_results)} extraction results")
                        
                    time.sleep(2)
                
                # Get final metrics
                final_metrics = consumer.get_metrics()
                health_status = consumer.health_check()
                
                # Prepare result
                result = {
                    'job_id': job_id,
                    'messages_processed': final_metrics.get('messages_processed', 0),
                    'messages_failed': final_metrics.get('messages_failed', 0),
                    'extraction_results': extraction_results,
                    'processing_time': time.time() - start_time,
                    'health_status': health_status,
                    'final_metrics': final_metrics,
                    'status': 'completed',
                    'timestamp': datetime.now().isoformat()
                }
                
                # Store results in XCom
                self.xcom_push(context, key=self.output_key, value=result)
                
                self.logger.info(
                    f"Text extraction consumer completed: {len(extraction_results)} extractions, "
                    f"{result['messages_processed']} messages processed"
                )
                
                return result
                
        except Exception as e:
            self.logger.error(f"Text extraction consumer operation failed: {str(e)}")
            
            # Store error information
            error_result = {
                'job_id': context['dag_run'].run_id,
                'error_message': str(e),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }
            
            self.xcom_push(context, key=f"{self.output_key}_error", value=error_result)
            
            raise AirflowException(f"Text extraction consumer operation failed: {str(e)}")


class LLMConsumerOperator(BaseOperator):
    """
    Specialized operator for LLM classification consumer.
    
    This operator uses the LLMConsumer to process LLM classification
    requests and return speaker attributions.
    """
    
    template_fields = ['consumer_config', 'processing_options']
    template_ext = ['.json']
    ui_color = '#2ca02c'
    
    @apply_defaults
    def __init__(
        self,
        consumer_config: Dict[str, Any] = None,
        processing_options: Dict[str, Any] = None,
        max_messages: int = 100,
        timeout_seconds: int = 1200,
        output_key: str = "llm_classification_result",
        *args,
        **kwargs
    ):
        """
        Initialize the LLMConsumerOperator.
        
        Args:
            consumer_config: Kafka consumer configuration
            processing_options: LLM processing options
            max_messages: Maximum number of messages to process
            timeout_seconds: Timeout for consumer operation
            output_key: Key for storing output in XCom
        """
        super().__init__(*args, **kwargs)
        
        self.consumer_config = consumer_config or {}
        self.processing_options = processing_options or {}
        self.max_messages = max_messages
        self.timeout_seconds = timeout_seconds
        self.output_key = output_key
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the LLM consumer operation.
        
        Args:
            context: Airflow execution context
            
        Returns:
            Dictionary containing LLM classification results
        """
        try:
            job_id = context['dag_run'].run_id
            self.logger.info(f"Starting LLM classification consumer for job: {job_id}")
            
            # Import LLMConsumer
            import sys
            sys.path.append('/opt/airflow/dags')
            from src.kafka.consumers.llm_consumer import LLMConsumer
            
            # Create callback to collect results
            classification_results = []
            
            def classification_callback(job_id: str, result_data: Dict[str, Any]):
                classification_results.append({
                    'job_id': job_id,
                    'result_data': result_data,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Create and start consumer
            with LLMConsumer(self.consumer_config, classification_callback) as consumer:
                consumer.start(num_workers=4)
                
                # Wait for processing to complete
                start_time = time.time()
                
                while (time.time() - start_time) < self.timeout_seconds:
                    metrics = consumer.get_metrics()
                    
                    if metrics['messages_processed'] >= self.max_messages:
                        self.logger.info(f"Reached max messages limit: {self.max_messages}")
                        break
                    
                    if len(classification_results) > 0:
                        self.logger.info(f"Received {len(classification_results)} classification results")
                        
                    time.sleep(2)
                
                # Get final metrics
                final_metrics = consumer.get_metrics()
                health_status = consumer.health_check()
                
                # Prepare result
                result = {
                    'job_id': job_id,
                    'messages_processed': final_metrics.get('messages_processed', 0),
                    'messages_failed': final_metrics.get('messages_failed', 0),
                    'classification_results': classification_results,
                    'processing_time': time.time() - start_time,
                    'health_status': health_status,
                    'final_metrics': final_metrics,
                    'total_classifications': final_metrics.get('classifications_generated', 0),
                    'status': 'completed',
                    'timestamp': datetime.now().isoformat()
                }
                
                # Store results in XCom
                self.xcom_push(context, key=self.output_key, value=result)
                
                self.logger.info(
                    f"LLM classification consumer completed: {len(classification_results)} results, "
                    f"{result['messages_processed']} messages processed, "
                    f"{result['total_classifications']} classifications generated"
                )
                
                return result
                
        except Exception as e:
            self.logger.error(f"LLM consumer operation failed: {str(e)}")
            
            # Store error information
            error_result = {
                'job_id': context['dag_run'].run_id,
                'error_message': str(e),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }
            
            self.xcom_push(context, key=f"{self.output_key}_error", value=error_result)
            
            raise AirflowException(f"LLM consumer operation failed: {str(e)}")


class KafkaMessageSensor(BaseSensorOperator):
    """
    Sensor that waits for messages on a Kafka topic.
    
    This sensor monitors a Kafka topic and triggers when messages are available.
    """
    
    template_fields = ['topic', 'consumer_config']
    ui_color = '#e377c2'
    
    @apply_defaults
    def __init__(
        self,
        topic: str,
        consumer_config: Dict[str, Any] = None,
        min_messages: int = 1,
        timeout_seconds: int = 60,
        poke_interval: int = 10,
        *args,
        **kwargs
    ):
        """
        Initialize the KafkaMessageSensor.
        
        Args:
            topic: Kafka topic to monitor
            consumer_config: Kafka consumer configuration
            min_messages: Minimum number of messages to wait for
            timeout_seconds: Timeout for sensor operation
            poke_interval: Interval between checks
        """
        super().__init__(
            poke_interval=poke_interval,
            timeout=timeout_seconds,
            *args,
            **kwargs
        )
        
        self.topic = topic
        self.consumer_config = consumer_config or {}
        self.min_messages = min_messages
        
        self.logger = logging.getLogger(__name__)
    
    def poke(self, context: Dict[str, Any]) -> bool:
        """
        Check if messages are available on the Kafka topic.
        
        Args:
            context: Airflow execution context
            
        Returns:
            True if messages are available, False otherwise
        """
        try:
            # Import Kafka consumer
            import sys
            sys.path.append('/opt/airflow/dags')
            from kafka import KafkaConsumer
            from src.kafka.kafka_config import KafkaConfig
            
            # Create consumer configuration
            consumer_config = KafkaConfig.get_consumer_config("sensor_group")
            consumer_config.update(self.consumer_config)
            consumer_config['auto_offset_reset'] = 'earliest'
            consumer_config['enable_auto_commit'] = False
            
            # Create consumer
            consumer = KafkaConsumer(self.topic, **consumer_config)
            
            try:
                # Poll for messages
                message_batch = consumer.poll(timeout_ms=5000)
                
                # Count messages
                message_count = sum(len(messages) for messages in message_batch.values())
                
                self.logger.info(f"Found {message_count} messages on topic '{self.topic}'")
                
                return message_count >= self.min_messages
                
            finally:
                consumer.close()
                
        except Exception as e:
            self.logger.error(f"Error checking Kafka topic '{self.topic}': {str(e)}")
            return False


# Utility functions for creating operators
def create_kafka_consumer_operator(
    task_id: str,
    topic: str,
    consumer_type: str = "text_extraction",
    consumer_config: Dict[str, Any] = None,
    max_messages: int = 100,
    timeout_seconds: int = 300,
    dag=None
) -> KafkaConsumerOperator:
    """
    Factory function for creating KafkaConsumerOperator instances.
    
    Args:
        task_id: Unique task identifier
        topic: Kafka topic to consume from
        consumer_type: Type of consumer
        consumer_config: Consumer configuration
        max_messages: Maximum messages to process
        timeout_seconds: Timeout for operation
        dag: DAG instance
        
    Returns:
        Configured KafkaConsumerOperator instance
    """
    return KafkaConsumerOperator(
        task_id=task_id,
        topic=topic,
        consumer_type=consumer_type,
        consumer_config=consumer_config,
        max_messages=max_messages,
        timeout_seconds=timeout_seconds,
        dag=dag
    )


def create_text_extraction_consumer_operator(
    task_id: str,
    consumer_config: Dict[str, Any] = None,
    processing_options: Dict[str, Any] = None,
    max_messages: int = 50,
    timeout_seconds: int = 600,
    dag=None
) -> TextExtractionConsumerOperator:
    """
    Factory function for creating TextExtractionConsumerOperator instances.
    
    Args:
        task_id: Unique task identifier
        consumer_config: Consumer configuration
        processing_options: Processing options
        max_messages: Maximum messages to process
        timeout_seconds: Timeout for operation
        dag: DAG instance
        
    Returns:
        Configured TextExtractionConsumerOperator instance
    """
    return TextExtractionConsumerOperator(
        task_id=task_id,
        consumer_config=consumer_config,
        processing_options=processing_options,
        max_messages=max_messages,
        timeout_seconds=timeout_seconds,
        dag=dag
    )


def create_llm_consumer_operator(
    task_id: str,
    consumer_config: Dict[str, Any] = None,
    processing_options: Dict[str, Any] = None,
    max_messages: int = 100,
    timeout_seconds: int = 1200,
    dag=None
) -> LLMConsumerOperator:
    """
    Factory function for creating LLMConsumerOperator instances.
    
    Args:
        task_id: Unique task identifier
        consumer_config: Consumer configuration
        processing_options: Processing options
        max_messages: Maximum messages to process
        timeout_seconds: Timeout for operation
        dag: DAG instance
        
    Returns:
        Configured LLMConsumerOperator instance
    """
    return LLMConsumerOperator(
        task_id=task_id,
        consumer_config=consumer_config,
        processing_options=processing_options,
        max_messages=max_messages,
        timeout_seconds=timeout_seconds,
        dag=dag
    )


def create_kafka_message_sensor(
    task_id: str,
    topic: str,
    consumer_config: Dict[str, Any] = None,
    min_messages: int = 1,
    timeout_seconds: int = 60,
    poke_interval: int = 10,
    dag=None
) -> KafkaMessageSensor:
    """
    Factory function for creating KafkaMessageSensor instances.
    
    Args:
        task_id: Unique task identifier
        topic: Kafka topic to monitor
        consumer_config: Consumer configuration
        min_messages: Minimum messages to wait for
        timeout_seconds: Timeout for sensor
        poke_interval: Interval between checks
        dag: DAG instance
        
    Returns:
        Configured KafkaMessageSensor instance
    """
    return KafkaMessageSensor(
        task_id=task_id,
        topic=topic,
        consumer_config=consumer_config,
        min_messages=min_messages,
        timeout_seconds=timeout_seconds,
        poke_interval=poke_interval,
        dag=dag
    )