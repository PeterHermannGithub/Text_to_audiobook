"""
LLM consumer for distributed text classification.

This module consumes LLM classification requests from Kafka and processes them
using the LLM pool management system.
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
    MessageValidator, LLMClassificationRequest, LLMClassificationResult,
    ProcessingStatus, ErrorEvent, StatusUpdate, MessageFactory
)
from ...llm_pool.llm_client import SparkLLMClient, LLMClient


class LLMConsumer:
    """Consumer for LLM classification requests."""
    
    def __init__(self, config: Dict[str, Any] = None,
                 callback: Callable[[str, Dict[str, Any]], None] = None):
        """
        Initialize LLM consumer.
        
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
        
        # LLM client
        self.llm_client = LLMClient()
        
        # Metrics
        self.metrics = {
            'messages_processed': 0,
            'messages_failed': 0,
            'total_processing_time': 0.0,
            'total_llm_time': 0.0,
            'average_processing_time': 0.0,
            'average_llm_time': 0.0,
            'start_time': time.time(),
            'last_processed_time': 0.0,
            'classifications_generated': 0,
            'retries_attempted': 0
        }
        
        # Initialize consumer and producer
        self._initialize_consumer()
        self._initialize_producer()
    
    def _initialize_consumer(self):
        """Initialize Kafka consumer."""
        try:
            consumer_config = KafkaConfig.get_consumer_config(
                ConsumerGroups.LLM_CLASSIFICATION_GROUP
            )
            
            # Override with custom config if provided
            if self.config:
                consumer_config.update(self.config)
            
            # Subscribe to LLM classification requests topic
            topics = [self.routing['llm_request']]
            
            self.consumer = KafkaConsumer(*topics, **consumer_config)
            self.logger.info(f"LLM consumer initialized for topics: {topics}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM consumer: {e}")
            raise
    
    def _initialize_producer(self):
        """Initialize producer for sending results."""
        try:
            from kafka import KafkaProducer
            
            producer_config = KafkaConfig.get_producer_config()
            self.producer = KafkaProducer(**producer_config)
            
            self.logger.info("LLM producer initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM producer: {e}")
            raise
    
    def start(self, num_workers: int = 4):
        """
        Start the consumer.
        
        Args:
            num_workers: Number of worker threads for processing
        """
        if self.running:
            self.logger.warning("LLM consumer is already running")
            return
        
        self.running = True
        self.logger.info(f"Starting LLM consumer with {num_workers} workers")
        
        # Start worker threads
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"LLMWorker-{i}",
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
        
        self.logger.info("LLM consumer started successfully")
    
    def stop(self):
        """Stop the consumer."""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Stopping LLM consumer")
        
        # Wait for workers to finish
        for worker in self.worker_threads:
            worker.join(timeout=10.0)
        
        # Close consumer and producer
        if self.consumer:
            self.consumer.close()
        
        if self.producer:
            self.producer.close()
        
        self.logger.info("LLM consumer stopped")
    
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
            if message_data.get('message_type') != 'llm_classification_request':
                self.logger.debug(f"Skipping message type: {message_data.get('message_type')}")
                return
            
            # Extract request details
            job_id = message_data['job_id']
            chunk_id = message_data.get('chunk_id', 'unknown')
            text_lines = message_data['text_lines']
            prompt = message_data['prompt']
            model_config = message_data.get('model_config', {})
            
            self.logger.info(f"Processing LLM classification request: {job_id}/{chunk_id}")
            
            start_time = time.time()
            
            # Send status update
            self._send_status_update(
                job_id=job_id,
                status=ProcessingStatus.IN_PROGRESS,
                progress=50.0,
                current_step=f"Processing LLM classification for chunk {chunk_id}"
            )
            
            # Process with LLM
            llm_start_time = time.time()
            classifications = self._classify_texts(text_lines, prompt, model_config)
            llm_processing_time = time.time() - llm_start_time
            
            # Create confidence scores (placeholder)
            confidence_scores = [0.8] * len(classifications)
            
            # Send result
            self._send_classification_result(
                job_id=job_id,
                chunk_id=chunk_id,
                classifications=classifications,
                confidence_scores=confidence_scores,
                processing_time=llm_processing_time
            )
            
            # Update metrics
            total_processing_time = time.time() - start_time
            self._update_metrics(total_processing_time, llm_processing_time, len(classifications))
            
            # Send completion status
            self._send_status_update(
                job_id=job_id,
                status=ProcessingStatus.IN_PROGRESS,
                progress=75.0,
                current_step=f"LLM classification completed for chunk {chunk_id}"
            )
            
            # Call callback if provided
            if self.callback:
                self.callback(job_id, {
                    'chunk_id': chunk_id,
                    'classifications': classifications,
                    'confidence_scores': confidence_scores,
                    'processing_time': llm_processing_time,
                    'total_processing_time': total_processing_time
                })
            
            self.logger.info(f"LLM classification completed for {job_id}/{chunk_id} in {total_processing_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error processing LLM classification message: {e}")
            self.logger.error(traceback.format_exc())
            
            # Send error event
            job_id = message_data.get('job_id', 'unknown')
            chunk_id = message_data.get('chunk_id', 'unknown')
            self._send_error_event(
                job_id=job_id,
                error_type="LLMClassificationError",
                error_message=str(e),
                error_details={
                    'chunk_id': chunk_id,
                    'text_lines_count': len(message_data.get('text_lines', [])),
                    'model_config': message_data.get('model_config', {})
                }
            )
            
            # Send failed status
            self._send_status_update(
                job_id=job_id,
                status=ProcessingStatus.FAILED,
                progress=0.0,
                current_step=f"LLM classification failed for chunk {chunk_id}"
            )
            
            self.metrics['messages_failed'] += 1
    
    def _classify_texts(self, text_lines: List[str], prompt: str, 
                       model_config: Dict[str, Any]) -> List[str]:
        """Classify text lines using LLM."""
        try:
            # If we have a single prompt, use it directly
            if isinstance(prompt, str):
                response = self.llm_client.classify_text(
                    text=prompt,
                    model_config=model_config
                )
                
                # Parse response to extract classifications
                classifications = self._parse_classification_response(response, len(text_lines))
                
            else:
                # For individual classification requests
                classifications = []
                for text_line in text_lines:
                    try:
                        classification = self.llm_client.classify_with_fallback(
                            text=text_line,
                            fallback_result="AMBIGUOUS",
                            model_config=model_config
                        )
                        classifications.append(classification)
                    except Exception as e:
                        self.logger.warning(f"Failed to classify line, using fallback: {e}")
                        classifications.append("AMBIGUOUS")
            
            return classifications
            
        except Exception as e:
            self.logger.error(f"Error classifying texts: {e}")
            # Return fallback classifications
            return ["AMBIGUOUS"] * len(text_lines)
    
    def _parse_classification_response(self, response: str, expected_count: int) -> List[str]:
        """Parse classification response into list of speakers."""
        try:
            # Try to parse as JSON first
            import json
            parsed = json.loads(response)
            
            if isinstance(parsed, list) and len(parsed) == expected_count:
                return parsed
            elif isinstance(parsed, dict) and 'speakers' in parsed:
                speakers = parsed['speakers']
                if isinstance(speakers, list) and len(speakers) == expected_count:
                    return speakers
        except:
            pass
        
        # Fallback: try to extract from text
        lines = response.strip().split('\n')
        classifications = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                # Try to extract speaker name
                if ':' in line:
                    speaker = line.split(':')[0].strip()
                    classifications.append(speaker)
                else:
                    classifications.append(line)
        
        # Ensure we have the right number of classifications
        if len(classifications) != expected_count:
            self.logger.warning(f"Expected {expected_count} classifications, got {len(classifications)}")
            # Pad or truncate to match expected count
            if len(classifications) < expected_count:
                classifications.extend(["AMBIGUOUS"] * (expected_count - len(classifications)))
            else:
                classifications = classifications[:expected_count]
        
        return classifications
    
    def _send_classification_result(self, job_id: str, chunk_id: str,
                                   classifications: List[str], 
                                   confidence_scores: List[float],
                                   processing_time: float):
        """Send classification result."""
        try:
            result = LLMClassificationResult(
                classifications=classifications,
                confidence_scores=confidence_scores,
                processing_time=processing_time,
                chunk_id=chunk_id,
                job_id=job_id
            )
            
            self._send_message(
                topic=self.routing.get('llm_result', 'llm-classification'),
                message=result,
                key=chunk_id
            )
            
        except Exception as e:
            self.logger.error(f"Error sending classification result: {e}")
    
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
            
            self._send_message(
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
                component="LLMConsumer"
            )
            error_event.job_id = job_id
            
            self._send_message(
                topic=self.routing['error'],
                message=error_event,
                key=job_id
            )
            
        except Exception as e:
            self.logger.error(f"Error sending error event: {e}")
    
    def _send_message(self, topic: str, message: Any, key: str = None):
        """Send message to Kafka topic."""
        try:
            # Convert message to dictionary
            if hasattr(message, 'to_dict'):
                message_dict = message.to_dict()
            else:
                message_dict = message
            
            # Send message
            future = self.producer.send(
                topic=topic,
                value=message_dict,
                key=key
            )
            
            # Wait for confirmation (with timeout)
            record_metadata = future.get(timeout=10)
            
            self.logger.debug(f"Message sent to {topic}: partition={record_metadata.partition}, offset={record_metadata.offset}")
            
        except Exception as e:
            self.logger.error(f"Error sending message to {topic}: {e}")
    
    def _update_metrics(self, processing_time: float, llm_time: float, classification_count: int):
        """Update processing metrics."""
        self.metrics['messages_processed'] += 1
        self.metrics['total_processing_time'] += processing_time
        self.metrics['total_llm_time'] += llm_time
        self.metrics['classifications_generated'] += classification_count
        
        self.metrics['average_processing_time'] = (
            self.metrics['total_processing_time'] / self.metrics['messages_processed']
        )
        self.metrics['average_llm_time'] = (
            self.metrics['total_llm_time'] / self.metrics['messages_processed']
        )
        
        self.metrics['last_processed_time'] = time.time()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get consumer metrics."""
        runtime = time.time() - self.metrics['start_time']
        
        return {
            'messages_processed': self.metrics['messages_processed'],
            'messages_failed': self.metrics['messages_failed'],
            'total_processing_time': self.metrics['total_processing_time'],
            'total_llm_time': self.metrics['total_llm_time'],
            'average_processing_time': self.metrics['average_processing_time'],
            'average_llm_time': self.metrics['average_llm_time'],
            'classifications_generated': self.metrics['classifications_generated'],
            'retries_attempted': self.metrics['retries_attempted'],
            'runtime': runtime,
            'messages_per_second': self.metrics['messages_processed'] / max(runtime, 1),
            'classifications_per_second': self.metrics['classifications_generated'] / max(runtime, 1),
            'success_rate': (
                self.metrics['messages_processed'] / 
                max(self.metrics['messages_processed'] + self.metrics['messages_failed'], 1)
            ),
            'llm_efficiency': self.metrics['total_llm_time'] / max(self.metrics['total_processing_time'], 1),
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
            
            # Check LLM client health
            llm_health = self.llm_client.health_check()
            
            # Check recent activity
            time_since_last_process = time.time() - self.metrics['last_processed_time']
            recent_activity = time_since_last_process < 300  # 5 minutes
            
            # Overall health
            healthy = (is_running and alive_threads > 0 and 
                      llm_health.get('overall_health') == 'healthy')
            
            return {
                'healthy': healthy,
                'running': is_running,
                'alive_threads': alive_threads,
                'total_threads': len(self.worker_threads),
                'recent_activity': recent_activity,
                'time_since_last_process': time_since_last_process,
                'llm_health': llm_health,
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


class BatchLLMConsumer:
    """Consumer for batch LLM processing."""
    
    def __init__(self, config: Dict[str, Any] = None, max_workers: int = 8):
        """Initialize batch LLM consumer."""
        self.config = config or {}
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Individual consumers for parallel processing
        self.consumers = []
        for i in range(max_workers):
            consumer = LLMConsumer(config)
            self.consumers.append(consumer)
    
    def start(self):
        """Start all consumers."""
        for i, consumer in enumerate(self.consumers):
            consumer.start(num_workers=1)  # Each consumer uses 1 worker
            self.logger.info(f"Started LLM consumer {i}")
    
    def stop(self):
        """Stop all consumers."""
        for consumer in self.consumers:
            consumer.stop()
        
        self.executor.shutdown(wait=True)
        self.logger.info("Batch LLM consumer stopped")
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics from all consumers."""
        total_processed = sum(c.metrics['messages_processed'] for c in self.consumers)
        total_failed = sum(c.metrics['messages_failed'] for c in self.consumers)
        total_time = sum(c.metrics['total_processing_time'] for c in self.consumers)
        total_llm_time = sum(c.metrics['total_llm_time'] for c in self.consumers)
        total_classifications = sum(c.metrics['classifications_generated'] for c in self.consumers)
        
        return {
            'total_consumers': len(self.consumers),
            'total_processed': total_processed,
            'total_failed': total_failed,
            'total_processing_time': total_time,
            'total_llm_time': total_llm_time,
            'total_classifications': total_classifications,
            'average_processing_time': total_time / max(total_processed, 1),
            'average_llm_time': total_llm_time / max(total_processed, 1),
            'success_rate': total_processed / max(total_processed + total_failed, 1),
            'llm_efficiency': total_llm_time / max(total_time, 1),
            'consumers_running': sum(1 for c in self.consumers if c.running),
            'classifications_per_second': total_classifications / max(total_time, 1)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all consumers."""
        health_checks = [c.health_check() for c in self.consumers]
        healthy_consumers = sum(1 for h in health_checks if h['healthy'])
        
        return {
            'overall_healthy': healthy_consumers > 0,
            'healthy_consumers': healthy_consumers,
            'total_consumers': len(self.consumers),
            'health_checks': health_checks,
            'aggregate_metrics': self.get_aggregate_metrics()
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Utility functions
def start_llm_consumer(config: Dict[str, Any] = None, num_workers: int = 4,
                      callback: Callable[[str, Dict[str, Any]], None] = None) -> LLMConsumer:
    """Start an LLM consumer."""
    consumer = LLMConsumer(config, callback)
    consumer.start(num_workers)
    return consumer


def start_batch_llm_consumer(config: Dict[str, Any] = None,
                            max_workers: int = 8) -> BatchLLMConsumer:
    """Start a batch LLM consumer."""
    consumer = BatchLLMConsumer(config, max_workers)
    consumer.start()
    return consumer