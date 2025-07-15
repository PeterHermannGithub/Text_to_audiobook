"""
Text extraction consumer for distributed processing.

This module consumes text extraction requests from Kafka and processes them
using the existing text extraction pipeline.
"""

import logging
import time
import json
import traceback
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from kafka import KafkaConsumer
from kafka.errors import KafkaError

from ..kafka_config import KafkaConfig, ConsumerGroups, MessageRouting
from ..schemas.message_schemas import (
    MessageValidator, TextExtractionRequest, TextExtractionResult,
    ProcessingStatus, ErrorEvent, StatusUpdate, MessageFactory
)
from ..producers.file_upload_producer import FileUploadProducer
from ...text_processing.text_extractor import TextExtractor


class TextExtractionConsumer:
    """Consumer for text extraction requests."""
    
    def __init__(self, config: Dict[str, Any] = None, 
                 callback: Callable[[str, Dict[str, Any]], None] = None):
        """
        Initialize text extraction consumer.
        
        Args:
            config: Optional configuration override
            callback: Optional callback function for processing results
        """
        self.config = config or {}
        self.callback = callback
        self.logger = logging.getLogger(__name__)
        self.consumer = None
        self.producer = None
        self.running = False
        self.worker_threads = []
        self.routing = MessageRouting.get_routing_config()
        
        # Processing components
        self.text_extractor = TextExtractor()
        
        # Metrics
        self.metrics = {
            'messages_processed': 0,
            'messages_failed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'start_time': time.time(),
            'last_processed_time': 0.0
        }
        
        # Initialize consumer and producer
        self._initialize_consumer()
        self._initialize_producer()
    
    def _initialize_consumer(self):
        """Initialize Kafka consumer."""
        try:
            consumer_config = KafkaConfig.get_consumer_config(
                ConsumerGroups.TEXT_EXTRACTION_GROUP
            )
            
            # Override with custom config if provided
            if self.config:
                consumer_config.update(self.config)
            
            # Subscribe to text extraction requests topic
            topics = [self.routing['file_upload']]
            
            self.consumer = KafkaConsumer(*topics, **consumer_config)
            self.logger.info(f"Text extraction consumer initialized for topics: {topics}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize text extraction consumer: {e}")
            raise
    
    def _initialize_producer(self):
        """Initialize producer for sending results."""
        try:
            self.producer = FileUploadProducer(self.config)
            self.logger.info("Text extraction producer initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize text extraction producer: {e}")
            raise
    
    def start(self, num_workers: int = 4):
        """
        Start the consumer.
        
        Args:
            num_workers: Number of worker threads for processing
        """
        if self.running:
            self.logger.warning("Consumer is already running")
            return
        
        self.running = True
        self.logger.info(f"Starting text extraction consumer with {num_workers} workers")
        
        # Start worker threads
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"TextExtractionWorker-{i}",
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
        
        self.logger.info("Text extraction consumer started successfully")
    
    def stop(self):
        """Stop the consumer."""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Stopping text extraction consumer")
        
        # Wait for workers to finish
        for worker in self.worker_threads:
            worker.join(timeout=10.0)
        
        # Close consumer and producer
        if self.consumer:
            self.consumer.close()
        
        if self.producer:
            self.producer.close()
        
        self.logger.info("Text extraction consumer stopped")
    
    def _worker_loop(self):
        """Worker loop for processing messages."""
        while self.running:
            try:
                # Poll for messages
                message_batch = self.consumer.poll(timeout_ms=1000)
                
                if not message_batch:
                    continue
                
                # Process messages
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        try:
                            self._process_message(message)
                        except Exception as e:
                            self.logger.error(f"Error processing message: {e}")
                            self.logger.error(traceback.format_exc())
                            self.metrics['messages_failed'] += 1
                
                # Commit offsets
                self.consumer.commit()
                
            except Exception as e:
                self.logger.error(f"Error in worker loop: {e}")
                if self.running:
                    time.sleep(1)  # Brief pause before retrying
    
    def _process_message(self, message):
        """Process a single message."""
        try:
            # Parse message
            message_data = message.value
            
            # Validate message
            if not MessageValidator.validate_message(message_data):
                self.logger.error(f"Invalid message format: {message_data}")
                return
            
            # Check message type
            if message_data.get('message_type') != 'text_extraction_request':
                self.logger.debug(f"Skipping message type: {message_data.get('message_type')}")
                return
            
            # Extract request details
            job_id = message_data['job_id']
            file_path = message_data['file_path']
            user_id = message_data['user_id']
            processing_options = message_data.get('processing_options', {})
            
            self.logger.info(f"Processing text extraction request: {job_id}")
            
            start_time = time.time()
            
            # Send status update
            self._send_status_update(
                job_id=job_id,
                status=ProcessingStatus.IN_PROGRESS,
                progress=10.0,
                current_step="Starting text extraction"
            )
            
            # Extract text
            extracted_text = self._extract_text(file_path, processing_options)
            
            # Create metadata
            text_metadata = self._create_text_metadata(extracted_text, processing_options)
            
            # Create file info
            file_info = self._create_file_info(file_path)
            
            # Send extraction result
            self.producer.notify_text_extracted(
                job_id=job_id,
                extracted_text=extracted_text,
                text_metadata=text_metadata,
                file_info=file_info
            )
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(processing_time)
            
            # Send completion status
            self._send_status_update(
                job_id=job_id,
                status=ProcessingStatus.IN_PROGRESS,
                progress=25.0,
                current_step="Text extraction completed"
            )
            
            # Call callback if provided
            if self.callback:
                self.callback(job_id, {
                    'extracted_text': extracted_text,
                    'text_metadata': text_metadata,
                    'file_info': file_info,
                    'processing_time': processing_time
                })
            
            self.logger.info(f"Text extraction completed for job {job_id} in {processing_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error processing text extraction message: {e}")
            self.logger.error(traceback.format_exc())
            
            # Send error event
            job_id = message_data.get('job_id', 'unknown')
            self._send_error_event(
                job_id=job_id,
                error_type="TextExtractionError",
                error_message=str(e),
                error_details={
                    'file_path': message_data.get('file_path'),
                    'user_id': message_data.get('user_id')
                }
            )
            
            # Send failed status
            self._send_status_update(
                job_id=job_id,
                status=ProcessingStatus.FAILED,
                progress=0.0,
                current_step="Text extraction failed"
            )
            
            self.metrics['messages_failed'] += 1
    
    def _extract_text(self, file_path: str, processing_options: Dict[str, Any]) -> str:
        """Extract text from file."""
        try:
            # Use existing text extractor
            extracted_text = self.text_extractor.extract(file_path)
            
            # Apply processing options if specified
            if processing_options:
                extracted_text = self._apply_processing_options(extracted_text, processing_options)
            
            return extracted_text
            
        except Exception as e:
            self.logger.error(f"Error extracting text from {file_path}: {e}")
            raise
    
    def _apply_processing_options(self, text: str, options: Dict[str, Any]) -> str:
        """Apply processing options to extracted text."""
        try:
            # Length limit
            if 'max_length' in options:
                max_length = options['max_length']
                if len(text) > max_length:
                    text = text[:max_length]
                    self.logger.info(f"Text truncated to {max_length} characters")
            
            # Encoding handling
            if 'encoding' in options:
                encoding = options['encoding']
                # Handle encoding if needed
                pass
            
            # Custom filters
            if 'filters' in options:
                filters = options['filters']
                for filter_type in filters:
                    if filter_type == 'remove_empty_lines':
                        text = '\n'.join(line for line in text.split('\n') if line.strip())
                    elif filter_type == 'normalize_whitespace':
                        import re
                        text = re.sub(r'\s+', ' ', text)
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error applying processing options: {e}")
            return text
    
    def _create_text_metadata(self, text: str, processing_options: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for extracted text."""
        try:
            metadata = {
                'character_count': len(text),
                'word_count': len(text.split()),
                'line_count': len(text.split('\n')),
                'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
                'extraction_time': time.time(),
                'processing_options': processing_options
            }
            
            # Add language detection if available
            try:
                from langdetect import detect
                metadata['detected_language'] = detect(text[:1000])  # Use first 1000 chars
            except:
                metadata['detected_language'] = 'unknown'
            
            # Add basic text statistics
            sentences = text.split('.')
            metadata['sentence_count'] = len([s for s in sentences if s.strip()])
            
            if metadata['word_count'] > 0:
                metadata['average_word_length'] = metadata['character_count'] / metadata['word_count']
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error creating text metadata: {e}")
            return {
                'character_count': len(text),
                'extraction_time': time.time(),
                'error': str(e)
            }
    
    def _create_file_info(self, file_path: str) -> Dict[str, Any]:
        """Create file information."""
        try:
            import os
            from pathlib import Path
            
            path = Path(file_path)
            
            file_info = {
                'file_path': file_path,
                'file_name': path.name,
                'file_extension': path.suffix,
                'file_size': os.path.getsize(file_path),
                'file_modified_time': os.path.getmtime(file_path),
                'file_created_time': os.path.getctime(file_path)
            }
            
            # Add MIME type if available
            try:
                import mimetypes
                file_info['mime_type'] = mimetypes.guess_type(file_path)[0]
            except:
                file_info['mime_type'] = 'unknown'
            
            return file_info
            
        except Exception as e:
            self.logger.error(f"Error creating file info: {e}")
            return {
                'file_path': file_path,
                'error': str(e)
            }
    
    def _send_status_update(self, job_id: str, status: ProcessingStatus,
                           progress: float, current_step: str):
        """Send status update."""
        try:
            status_update = MessageFactory.create_status_update(
                status=status,
                progress=progress,
                current_step=current_step,
                job_id=job_id
            )
            
            self.producer._send_message(
                topic=self.routing['status_update'],
                message=status_update,
                key=job_id
            )
            
        except Exception as e:
            self.logger.error(f"Error sending status update: {e}")
    
    def _send_error_event(self, job_id: str, error_type: str, 
                         error_message: str, error_details: Dict[str, Any]):
        """Send error event."""
        try:
            error_event = MessageFactory.create_error_event(
                error_type=error_type,
                error_message=error_message,
                error_details=error_details,
                component="TextExtractionConsumer"
            )
            error_event.job_id = job_id
            
            self.producer._send_message(
                topic=self.routing['error'],
                message=error_event,
                key=job_id
            )
            
        except Exception as e:
            self.logger.error(f"Error sending error event: {e}")
    
    def _update_metrics(self, processing_time: float):
        """Update processing metrics."""
        self.metrics['messages_processed'] += 1
        self.metrics['total_processing_time'] += processing_time
        self.metrics['average_processing_time'] = (
            self.metrics['total_processing_time'] / self.metrics['messages_processed']
        )
        self.metrics['last_processed_time'] = time.time()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get consumer metrics."""
        runtime = time.time() - self.metrics['start_time']
        
        return {
            'messages_processed': self.metrics['messages_processed'],
            'messages_failed': self.metrics['messages_failed'],
            'total_processing_time': self.metrics['total_processing_time'],
            'average_processing_time': self.metrics['average_processing_time'],
            'runtime': runtime,
            'messages_per_second': self.metrics['messages_processed'] / max(runtime, 1),
            'success_rate': (
                self.metrics['messages_processed'] / 
                max(self.metrics['messages_processed'] + self.metrics['messages_failed'], 1)
            ),
            'last_processed_time': self.metrics['last_processed_time'],
            'worker_threads': len(self.worker_threads),
            'running': self.running
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check consumer health."""
        try:
            # Check if consumer is running
            is_running = self.running and self.consumer is not None
            
            # Check if threads are alive
            alive_threads = sum(1 for t in self.worker_threads if t.is_alive())
            
            # Check recent activity
            time_since_last_process = time.time() - self.metrics['last_processed_time']
            recent_activity = time_since_last_process < 300  # 5 minutes
            
            # Overall health
            healthy = is_running and alive_threads > 0
            
            return {
                'healthy': healthy,
                'running': is_running,
                'alive_threads': alive_threads,
                'total_threads': len(self.worker_threads),
                'recent_activity': recent_activity,
                'time_since_last_process': time_since_last_process,
                'metrics': self.get_metrics()
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class BatchTextExtractionConsumer:
    """Consumer for batch text extraction processing."""
    
    def __init__(self, config: Dict[str, Any] = None, max_workers: int = 10):
        """Initialize batch consumer."""
        self.config = config or {}
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Individual consumers for parallel processing
        self.consumers = []
        for i in range(max_workers):
            consumer = TextExtractionConsumer(config)
            self.consumers.append(consumer)
    
    def start(self):
        """Start all consumers."""
        for i, consumer in enumerate(self.consumers):
            consumer.start(num_workers=1)  # Each consumer uses 1 worker
            self.logger.info(f"Started consumer {i}")
    
    def stop(self):
        """Stop all consumers."""
        for consumer in self.consumers:
            consumer.stop()
        
        self.executor.shutdown(wait=True)
        self.logger.info("Batch text extraction consumer stopped")
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics from all consumers."""
        total_processed = sum(c.metrics['messages_processed'] for c in self.consumers)
        total_failed = sum(c.metrics['messages_failed'] for c in self.consumers)
        total_time = sum(c.metrics['total_processing_time'] for c in self.consumers)
        
        return {
            'total_consumers': len(self.consumers),
            'total_processed': total_processed,
            'total_failed': total_failed,
            'total_processing_time': total_time,
            'average_processing_time': total_time / max(total_processed, 1),
            'success_rate': total_processed / max(total_processed + total_failed, 1),
            'consumers_running': sum(1 for c in self.consumers if c.running)
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Utility functions
def start_text_extraction_consumer(config: Dict[str, Any] = None, 
                                  num_workers: int = 4,
                                  callback: Callable[[str, Dict[str, Any]], None] = None) -> TextExtractionConsumer:
    """Start a text extraction consumer."""
    consumer = TextExtractionConsumer(config, callback)
    consumer.start(num_workers)
    return consumer


def start_batch_text_extraction_consumer(config: Dict[str, Any] = None,
                                        max_workers: int = 10) -> BatchTextExtractionConsumer:
    """Start a batch text extraction consumer."""
    consumer = BatchTextExtractionConsumer(config, max_workers)
    consumer.start()
    return consumer