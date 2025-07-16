"""
Distributed text processing consumer for the complete text-to-audiobook pipeline.

This consumer integrates with the DistributedPipelineOrchestrator to provide
end-to-end distributed text processing with Kafka, Spark, and LLM Pool integration.
"""

import logging
import time
import json
import traceback
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime
import uuid

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

from ..kafka_config import KafkaConfig, ConsumerGroups, MessageRouting
from ..schemas.message_schemas import (
    MessageValidator, TextExtractionRequest, TextExtractionResult,
    ProcessingStatus, ErrorEvent, StatusUpdate, MessageFactory
)
from ...monitoring.prometheus_metrics import get_metrics_collector
from ...cache.redis_cache import RedisCacheManager


class DistributedTextProcessor:
    """
    Distributed text processing consumer that integrates with the complete pipeline.
    
    This consumer:
    1. Consumes text processing requests from Kafka
    2. Processes them using the DistributedPipelineOrchestrator
    3. Publishes results and status updates back to Kafka
    4. Handles error recovery and retry logic
    """
    
    def __init__(self, orchestrator=None, config: Dict[str, Any] = None):
        """
        Initialize distributed text processor.
        
        Args:
            orchestrator: DistributedPipelineOrchestrator instance
            config: Optional configuration override
        """
        self.orchestrator = orchestrator
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Kafka components
        self.consumer = None
        self.producer = None
        self.running = False
        self.worker_threads = []
        self.routing = MessageRouting.get_routing_config()
        
        # Processing configuration
        self.max_workers = self.config.get('max_workers', 5)
        self.batch_size = self.config.get('batch_size', 10)
        self.processing_timeout = self.config.get('processing_timeout', 300)
        self.retry_attempts = self.config.get('retry_attempts', 3)
        self.retry_delay = self.config.get('retry_delay', 5)
        
        # Monitoring and metrics
        self.metrics_collector = get_metrics_collector()
        self.cache_manager = None
        
        # Processing state
        self.active_jobs = {}
        self.processing_stats = {
            'total_processed': 0,
            'successful_processed': 0,
            'failed_processed': 0,
            'avg_processing_time': 0.0
        }
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize Kafka consumer, producer, and other components."""
        try:
            # Initialize Kafka consumer
            kafka_config = KafkaConfig.get_consumer_config()
            self.consumer = KafkaConsumer(
                self.routing['topics']['text_processing_requests'],
                bootstrap_servers=kafka_config['bootstrap_servers'],
                group_id=ConsumerGroups.TEXT_PROCESSING_GROUP,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda m: m.decode('utf-8') if m else None,
                **kafka_config['consumer_config']
            )
            
            # Initialize Kafka producer
            producer_config = KafkaConfig.get_producer_config()
            self.producer = KafkaProducer(
                bootstrap_servers=producer_config['bootstrap_servers'],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                **producer_config['producer_config']
            )
            
            # Initialize cache manager
            try:
                self.cache_manager = RedisCacheManager(namespace='distributed_processor')
            except Exception as e:
                self.logger.warning(f"Failed to initialize cache manager: {e}")
            
            self.logger.info("Distributed text processor components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def start_processing(self):
        """Start the distributed text processing consumer."""
        try:
            self.running = True
            self.logger.info("Starting distributed text processing consumer...")
            
            # Start worker threads
            for i in range(self.max_workers):
                thread = threading.Thread(
                    target=self._worker_thread,
                    name=f"distributed-processor-{i}",
                    daemon=True
                )
                thread.start()
                self.worker_threads.append(thread)
            
            # Start main processing loop
            self._main_processing_loop()
            
        except Exception as e:
            self.logger.error(f"Error in distributed text processor: {e}")
            self.stop_processing()
    
    def stop_processing(self):
        """Stop the distributed text processing consumer."""
        try:
            self.logger.info("Stopping distributed text processing consumer...")
            self.running = False
            
            # Wait for worker threads to finish
            for thread in self.worker_threads:
                thread.join(timeout=10)
            
            # Close Kafka connections
            if self.consumer:
                self.consumer.close()
            if self.producer:
                self.producer.close()
            
            # Close cache manager
            if self.cache_manager:
                self.cache_manager.close()
            
            self.logger.info("Distributed text processor stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping distributed text processor: {e}")
    
    def _main_processing_loop(self):
        """Main processing loop that consumes messages from Kafka."""
        try:
            while self.running:
                try:
                    # Poll for messages
                    messages = self.consumer.poll(timeout_ms=1000, max_records=self.batch_size)
                    
                    if not messages:
                        continue
                    
                    # Process messages
                    for topic_partition, message_list in messages.items():
                        for message in message_list:
                            if not self.running:
                                break
                            
                            self._handle_message(message)
                    
                    # Commit offsets
                    self.consumer.commit()
                    
                except KafkaError as e:
                    self.logger.error(f"Kafka error in processing loop: {e}")
                    time.sleep(self.retry_delay)
                
                except Exception as e:
                    self.logger.error(f"Unexpected error in processing loop: {e}")
                    time.sleep(self.retry_delay)
        
        except Exception as e:
            self.logger.error(f"Fatal error in main processing loop: {e}")
            self.running = False
    
    def _handle_message(self, message):
        """Handle a single Kafka message."""
        try:
            # Parse message
            job_id = message.key
            request_data = message.value
            
            # Validate message
            if not self._validate_message(request_data):
                self.logger.warning(f"Invalid message format for job {job_id}")
                return
            
            # Add to active jobs
            self.active_jobs[job_id] = {
                'start_time': time.time(),
                'status': 'queued',
                'request_data': request_data
            }
            
            # Send status update
            self._send_status_update(job_id, 'queued', 'Job queued for processing')
            
            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.record_kafka_message('text_processing_requests', 'received')
            
            self.logger.info(f"Received text processing request for job {job_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
    
    def _worker_thread(self):
        """Worker thread that processes queued jobs."""
        thread_name = threading.current_thread().name
        
        while self.running:
            try:
                # Find queued jobs
                queued_jobs = [
                    job_id for job_id, job_info in self.active_jobs.items()
                    if job_info['status'] == 'queued'
                ]
                
                if not queued_jobs:
                    time.sleep(1)
                    continue
                
                # Process the first queued job
                job_id = queued_jobs[0]
                self._process_job(job_id, thread_name)
                
            except Exception as e:
                self.logger.error(f"Error in worker thread {thread_name}: {e}")
                time.sleep(5)
    
    def _process_job(self, job_id: str, thread_name: str):
        """Process a single job using the distributed orchestrator."""
        try:
            job_info = self.active_jobs.get(job_id)
            if not job_info:
                return
            
            # Update job status
            job_info['status'] = 'processing'
            self._send_status_update(job_id, 'processing', f'Processing started by {thread_name}')
            
            # Extract text from request
            request_data = job_info['request_data']
            text_content = request_data.get('text_content', '')
            
            if not text_content:
                self._handle_job_error(job_id, "No text content in request")
                return
            
            # Check cache for existing results
            cached_result = None
            if self.cache_manager:
                cached_result = self.cache_manager.get('processing_results', job_id)
            
            if cached_result:
                self.logger.info(f"Found cached result for job {job_id}")
                self._send_processing_result(job_id, cached_result, from_cache=True)
                return
            
            # Process using distributed orchestrator
            if self.orchestrator:
                start_time = time.time()
                
                # Process text
                result = self.orchestrator.process_text(text_content, job_id)
                
                processing_time = time.time() - start_time
                
                # Update job info
                job_info['processing_time'] = processing_time
                job_info['result'] = result
                
                if result.success:
                    # Send successful result
                    self._send_processing_result(job_id, result)
                    
                    # Cache result
                    if self.cache_manager:
                        self.cache_manager.set('processing_results', job_id, result)
                    
                    # Update statistics
                    self.processing_stats['successful_processed'] += 1
                    
                    self.logger.info(f"Successfully processed job {job_id} in {processing_time:.2f}s")
                    
                else:
                    # Handle processing failure
                    self._handle_job_error(job_id, result.error_details)
            
            else:
                self._handle_job_error(job_id, "Distributed orchestrator not available")
        
        except Exception as e:
            self.logger.error(f"Error processing job {job_id}: {e}")
            self._handle_job_error(job_id, str(e))
        
        finally:
            # Clean up job
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            
            # Update total processed count
            self.processing_stats['total_processed'] += 1
    
    def _handle_job_error(self, job_id: str, error_message: str):
        """Handle job processing error."""
        try:
            # Update job status
            if job_id in self.active_jobs:
                self.active_jobs[job_id]['status'] = 'failed'
                self.active_jobs[job_id]['error'] = error_message
            
            # Send error status update
            self._send_status_update(job_id, 'failed', error_message)
            
            # Send error event
            error_event = MessageFactory.create_error_event(
                job_id=job_id,
                error_type='processing_error',
                error_message=error_message,
                component='distributed_text_processor'
            )
            
            self._send_message(
                self.routing['topics']['error_events'],
                job_id,
                error_event
            )
            
            # Update statistics
            self.processing_stats['failed_processed'] += 1
            
            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.record_processing_request('text_processing', 'failed')
            
            self.logger.error(f"Job {job_id} failed: {error_message}")
            
        except Exception as e:
            self.logger.error(f"Error handling job error for {job_id}: {e}")
    
    def _send_processing_result(self, job_id: str, result, from_cache: bool = False):
        """Send processing result to Kafka."""
        try:
            # Create result message
            result_message = {
                'job_id': job_id,
                'success': result.success,
                'processed_segments': result.processed_segments,
                'processing_time': result.processing_time,
                'performance_metrics': result.performance_metrics,
                'quality_report': result.quality_report,
                'from_cache': from_cache,
                'timestamp': datetime.now().isoformat()
            }
            
            # Send to results topic
            self._send_message(
                self.routing['topics']['text_processing_results'],
                job_id,
                result_message
            )
            
            # Send status update
            self._send_status_update(
                job_id, 
                'completed', 
                f'Processing completed successfully (from_cache: {from_cache})'
            )
            
            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.record_processing_request('text_processing', 'completed')
                self.metrics_collector.record_processing_duration(
                    'text_processing', 'orchestrator', result.processing_time
                )
            
            self.logger.info(f"Sent processing result for job {job_id}")
            
        except Exception as e:
            self.logger.error(f"Error sending processing result for {job_id}: {e}")
    
    def _send_status_update(self, job_id: str, status: str, message: str):
        """Send status update to Kafka."""
        try:
            status_update = MessageFactory.create_status_update(
                job_id=job_id,
                status=status,
                message=message,
                component='distributed_text_processor'
            )
            
            self._send_message(
                self.routing['topics']['status_updates'],
                job_id,
                status_update
            )
            
        except Exception as e:
            self.logger.error(f"Error sending status update for {job_id}: {e}")
    
    def _send_message(self, topic: str, key: str, message: Dict[str, Any]):
        """Send message to Kafka topic."""
        try:
            if self.producer:
                future = self.producer.send(topic, key=key, value=message)
                future.get(timeout=10)  # Wait for send completion
                
                if self.metrics_collector:
                    self.metrics_collector.record_kafka_message(topic, 'sent')
        
        except Exception as e:
            self.logger.error(f"Error sending message to {topic}: {e}")
    
    def _validate_message(self, message: Dict[str, Any]) -> bool:
        """Validate incoming message format."""
        try:
            # Check required fields
            required_fields = ['job_id', 'text_content']
            
            for field in required_fields:
                if field not in message:
                    self.logger.warning(f"Missing required field: {field}")
                    return False
            
            # Check text content is not empty
            if not message['text_content'].strip():
                self.logger.warning("Empty text content")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating message: {e}")
            return False
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        stats = self.processing_stats.copy()
        
        # Add active jobs count
        stats['active_jobs'] = len(self.active_jobs)
        
        # Add queue status
        queued_jobs = sum(1 for job in self.active_jobs.values() if job['status'] == 'queued')
        processing_jobs = sum(1 for job in self.active_jobs.values() if job['status'] == 'processing')
        
        stats['queued_jobs'] = queued_jobs
        stats['processing_jobs'] = processing_jobs
        
        # Calculate success rate
        if stats['total_processed'] > 0:
            stats['success_rate'] = stats['successful_processed'] / stats['total_processed']
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def get_active_jobs(self) -> Dict[str, Any]:
        """Get information about active jobs."""
        return {
            job_id: {
                'status': job_info['status'],
                'start_time': job_info['start_time'],
                'processing_time': job_info.get('processing_time', 0),
                'text_length': len(job_info['request_data'].get('text_content', ''))
            }
            for job_id, job_info in self.active_jobs.items()
        }


class DistributedTextProcessorManager:
    """Manager for distributed text processor instances."""
    
    def __init__(self, orchestrator=None):
        """Initialize manager with orchestrator."""
        self.orchestrator = orchestrator
        self.processor = None
        self.logger = logging.getLogger(__name__)
    
    def start_processor(self, config: Dict[str, Any] = None):
        """Start distributed text processor."""
        try:
            if self.processor:
                self.logger.warning("Processor already running")
                return
            
            self.processor = DistributedTextProcessor(
                orchestrator=self.orchestrator,
                config=config
            )
            
            # Start processing in separate thread
            import threading
            processor_thread = threading.Thread(
                target=self.processor.start_processing,
                daemon=True
            )
            processor_thread.start()
            
            self.logger.info("Distributed text processor started")
            
        except Exception as e:
            self.logger.error(f"Failed to start processor: {e}")
            raise
    
    def stop_processor(self):
        """Stop distributed text processor."""
        try:
            if self.processor:
                self.processor.stop_processing()
                self.processor = None
                self.logger.info("Distributed text processor stopped")
        
        except Exception as e:
            self.logger.error(f"Error stopping processor: {e}")
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        if self.processor:
            return self.processor.get_processing_stats()
        else:
            return {'status': 'not_running'}
    
    def get_active_jobs(self) -> Dict[str, Any]:
        """Get active jobs information."""
        if self.processor:
            return self.processor.get_active_jobs()
        else:
            return {}


# Global manager instance
_processor_manager = None


def get_processor_manager(orchestrator=None) -> DistributedTextProcessorManager:
    """Get the global processor manager instance."""
    global _processor_manager
    
    if _processor_manager is None:
        _processor_manager = DistributedTextProcessorManager(orchestrator)
    
    return _processor_manager