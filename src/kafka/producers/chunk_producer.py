"""
Chunk producer for distributed text processing.

This module handles the distribution of text chunks across workers
for parallel processing in the text-to-audiobook system.
"""

import json
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from kafka import KafkaProducer
from kafka.errors import KafkaError

from ..kafka_config import KafkaConfig, MessageRouting
from ..schemas.message_schemas import (
    MessageFactory, ChunkProcessing, ProcessingRequest, StatusUpdate,
    ErrorEvent, ProcessingStatus, MessageType
)


@dataclass
class ChunkTask:
    """Represents a chunk processing task."""
    chunk_id: str
    job_id: str
    chunk_index: int
    total_chunks: int
    text_lines: List[str]
    context_lines: List[str]
    metadata: Dict[str, Any]
    context_hint: Dict[str, Any]
    priority: int
    retry_count: int
    max_retries: int
    created_at: float
    
    def should_retry(self) -> bool:
        """Check if chunk should be retried."""
        return self.retry_count < self.max_retries


class ChunkProducer:
    """Producer for distributing text chunks for processing."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize chunk producer."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.producer = None
        self.routing = MessageRouting.get_routing_config()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Metrics
        self.metrics = {
            'chunks_sent': 0,
            'chunks_failed': 0,
            'total_processing_time': 0.0,
            'average_chunk_size': 0.0,
            'retry_count': 0
        }
        
        # Initialize producer
        self._initialize_producer()
    
    def _initialize_producer(self):
        """Initialize Kafka producer."""
        try:
            producer_config = KafkaConfig.get_producer_config()
            
            # Override with custom config if provided
            if self.config:
                producer_config.update(self.config)
            
            self.producer = KafkaProducer(**producer_config)
            self.logger.info("Chunk producer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize chunk producer: {e}")
            raise
    
    def submit_chunks_for_processing(self, chunks: List[ChunkTask],
                                    processing_config: Dict[str, Any] = None) -> bool:
        """
        Submit multiple chunks for processing.
        
        Args:
            chunks: List of chunk tasks to process
            processing_config: Optional processing configuration
            
        Returns:
            True if all chunks submitted successfully
        """
        try:
            if not chunks:
                self.logger.warning("No chunks provided for processing")
                return True
            
            self.logger.info(f"Submitting {len(chunks)} chunks for processing")
            
            # Submit processing request first
            job_id = chunks[0].job_id
            self._send_processing_request(job_id, chunks, processing_config)
            
            # Send status update
            self._send_status_update(
                job_id=job_id,
                status=ProcessingStatus.IN_PROGRESS,
                progress=0.0,
                current_step=f"Submitting {len(chunks)} chunks for processing",
                total_steps=len(chunks)
            )
            
            # Submit chunks in parallel
            success_count = 0
            failed_chunks = []
            
            with ThreadPoolExecutor(max_workers=min(10, len(chunks))) as executor:
                future_to_chunk = {
                    executor.submit(self._submit_single_chunk, chunk): chunk
                    for chunk in chunks
                }
                
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        success = future.result()
                        if success:
                            success_count += 1
                        else:
                            failed_chunks.append(chunk)
                    except Exception as e:
                        self.logger.error(f"Error submitting chunk {chunk.chunk_id}: {e}")
                        failed_chunks.append(chunk)
            
            # Update metrics
            self.metrics['chunks_sent'] += success_count
            self.metrics['chunks_failed'] += len(failed_chunks)
            
            # Handle failed chunks
            if failed_chunks:
                self.logger.warning(f"{len(failed_chunks)} chunks failed to submit")
                self._handle_failed_chunks(failed_chunks)
            
            # Send progress update
            progress = (success_count / len(chunks)) * 100
            self._send_status_update(
                job_id=job_id,
                status=ProcessingStatus.IN_PROGRESS,
                progress=progress,
                current_step=f"Submitted {success_count}/{len(chunks)} chunks",
                total_steps=len(chunks)
            )
            
            self.logger.info(f"Chunk submission completed: {success_count}/{len(chunks)} successful")
            
            return len(failed_chunks) == 0
            
        except Exception as e:
            self.logger.error(f"Error submitting chunks for processing: {e}")
            return False
    
    def _submit_single_chunk(self, chunk: ChunkTask) -> bool:
        """Submit a single chunk for processing."""
        try:
            # Create chunk processing message
            chunk_message = ChunkProcessing(
                chunk_id=chunk.chunk_id,
                chunk_text="\n".join(chunk.text_lines),
                chunk_metadata=chunk.metadata,
                context_hint=chunk.context_hint,
                chunk_index=chunk.chunk_index,
                total_chunks=chunk.total_chunks,
                job_id=chunk.job_id
            )
            
            # Add additional metadata
            chunk_message.metadata.update({
                'priority': chunk.priority,
                'retry_count': chunk.retry_count,
                'max_retries': chunk.max_retries,
                'created_at': chunk.created_at,
                'text_lines_count': len(chunk.text_lines),
                'context_lines_count': len(chunk.context_lines)
            })
            
            # Send to Kafka
            success = self._send_message(
                topic=self.routing['chunk_processing'],
                message=chunk_message,
                key=chunk.chunk_id
            )
            
            if success:
                self.logger.debug(f"Chunk {chunk.chunk_id} submitted successfully")
            else:
                self.logger.error(f"Failed to submit chunk {chunk.chunk_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error submitting chunk {chunk.chunk_id}: {e}")
            return False
    
    def _send_processing_request(self, job_id: str, chunks: List[ChunkTask],
                                processing_config: Dict[str, Any] = None):
        """Send processing request message."""
        try:
            # Create processing request
            processing_request = ProcessingRequest(
                text_content=f"Processing {len(chunks)} chunks",
                processing_config=processing_config or {},
                priority=max(chunk.priority for chunk in chunks),
                job_id=job_id
            )
            
            # Add chunk information
            processing_request.metadata.update({
                'total_chunks': len(chunks),
                'chunk_ids': [chunk.chunk_id for chunk in chunks],
                'total_text_lines': sum(len(chunk.text_lines) for chunk in chunks),
                'average_chunk_size': sum(len(chunk.text_lines) for chunk in chunks) / len(chunks)
            })
            
            # Send to Kafka
            self._send_message(
                topic=self.routing['processing_request'],
                message=processing_request,
                key=job_id
            )
            
            self.logger.info(f"Processing request sent for job {job_id}")
            
        except Exception as e:
            self.logger.error(f"Error sending processing request: {e}")
    
    def _handle_failed_chunks(self, failed_chunks: List[ChunkTask]):
        """Handle failed chunks (retry logic)."""
        for chunk in failed_chunks:
            if chunk.should_retry():
                # Increment retry count
                chunk.retry_count += 1
                
                # Send error event
                error_event = MessageFactory.create_error_event(
                    error_type="ChunkSubmissionFailure",
                    error_message=f"Failed to submit chunk {chunk.chunk_id}",
                    error_details={
                        'chunk_id': chunk.chunk_id,
                        'chunk_index': chunk.chunk_index,
                        'retry_count': chunk.retry_count,
                        'max_retries': chunk.max_retries
                    },
                    component="ChunkProducer"
                )
                error_event.job_id = chunk.job_id
                
                self._send_message(
                    topic=self.routing['error'],
                    message=error_event,
                    key=chunk.chunk_id
                )
                
                # Retry submission
                self.logger.info(f"Retrying chunk {chunk.chunk_id} (attempt {chunk.retry_count})")
                self._submit_single_chunk(chunk)
            else:
                # Max retries exceeded
                self.logger.error(f"Max retries exceeded for chunk {chunk.chunk_id}")
                
                # Send final error event
                error_event = MessageFactory.create_error_event(
                    error_type="ChunkSubmissionFailure",
                    error_message=f"Max retries exceeded for chunk {chunk.chunk_id}",
                    error_details={
                        'chunk_id': chunk.chunk_id,
                        'chunk_index': chunk.chunk_index,
                        'retry_count': chunk.retry_count,
                        'max_retries': chunk.max_retries
                    },
                    component="ChunkProducer"
                )
                error_event.job_id = chunk.job_id
                
                self._send_message(
                    topic=self.routing['error'],
                    message=error_event,
                    key=chunk.chunk_id
                )
    
    def create_chunk_tasks(self, job_id: str, text_windows: List[Dict[str, Any]],
                          metadata: Dict[str, Any], priority: int = 0) -> List[ChunkTask]:
        """
        Create chunk tasks from text windows.
        
        Args:
            job_id: Job ID for the processing
            text_windows: List of text windows from ChunkManager
            metadata: Text metadata
            priority: Processing priority
            
        Returns:
            List of chunk tasks
        """
        try:
            chunk_tasks = []
            
            for i, window in enumerate(text_windows):
                chunk_id = f"{job_id}-chunk-{i}"
                
                chunk_task = ChunkTask(
                    chunk_id=chunk_id,
                    job_id=job_id,
                    chunk_index=i,
                    total_chunks=len(text_windows),
                    text_lines=window['task_lines'],
                    context_lines=window['context_lines'],
                    metadata=metadata,
                    context_hint=window.get('context_hint', {}),
                    priority=priority,
                    retry_count=0,
                    max_retries=3,
                    created_at=time.time()
                )
                
                chunk_tasks.append(chunk_task)
            
            self.logger.info(f"Created {len(chunk_tasks)} chunk tasks for job {job_id}")
            
            return chunk_tasks
            
        except Exception as e:
            self.logger.error(f"Error creating chunk tasks: {e}")
            return []
    
    def notify_chunk_completed(self, chunk_id: str, job_id: str,
                              result_data: Dict[str, Any],
                              processing_time: float) -> bool:
        """
        Notify that a chunk has been completed.
        
        Args:
            chunk_id: ID of the completed chunk
            job_id: Job ID
            result_data: Processing result data
            processing_time: Time taken to process
            
        Returns:
            True if notification sent successfully
        """
        try:
            # Create metrics event
            metrics_event = MessageFactory.create_metrics_event(
                metric_name="chunk_processing_time",
                metric_value=processing_time,
                metric_type="histogram",
                tags={
                    'chunk_id': chunk_id,
                    'job_id': job_id,
                    'component': 'chunk_processor'
                }
            )
            
            # Send metrics
            self._send_message(
                topic=self.routing['metrics'],
                message=metrics_event,
                key=chunk_id
            )
            
            self.logger.debug(f"Chunk completion notification sent for {chunk_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending chunk completion notification: {e}")
            return False
    
    def _send_status_update(self, job_id: str, status: ProcessingStatus,
                           progress: float, current_step: str,
                           total_steps: int = None) -> bool:
        """Send status update message."""
        try:
            status_update = MessageFactory.create_status_update(
                status=status,
                progress=progress,
                current_step=current_step,
                job_id=job_id
            )
            
            if total_steps:
                status_update.total_steps = total_steps
            
            self._send_message(
                topic=self.routing['status_update'],
                message=status_update,
                key=job_id
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending status update: {e}")
            return False
    
    def _send_message(self, topic: str, message: Any, key: str = None) -> bool:
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
            
            return True
            
        except KafkaError as e:
            self.logger.error(f"Kafka error sending message: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get producer metrics."""
        try:
            # Get Kafka producer metrics
            kafka_metrics = {}
            if self.producer:
                kafka_metrics = self.producer.metrics()
            
            # Combine with internal metrics
            return {
                'internal_metrics': self.metrics,
                'kafka_metrics': kafka_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error getting metrics: {e}")
            return {'internal_metrics': self.metrics}
    
    def flush(self, timeout: float = 30.0) -> bool:
        """Flush any pending messages."""
        try:
            if self.producer:
                self.producer.flush(timeout=timeout)
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error flushing producer: {e}")
            return False
    
    def close(self):
        """Close the producer."""
        try:
            if self.producer:
                self.producer.close()
            
            if self.executor:
                self.executor.shutdown(wait=True)
            
            self.logger.info("Chunk producer closed")
            
        except Exception as e:
            self.logger.error(f"Error closing producer: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class DistributedChunkProcessor:
    """High-level interface for distributed chunk processing."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize distributed chunk processor."""
        self.chunk_producer = ChunkProducer(config)
        self.logger = logging.getLogger(__name__)
    
    def process_text_distributed(self, text_content: str, job_id: str = None,
                                processing_config: Dict[str, Any] = None,
                                priority: int = 0) -> str:
        """
        Process text using distributed chunk processing.
        
        Args:
            text_content: Text to process
            job_id: Optional job ID (generated if not provided)
            processing_config: Optional processing configuration
            priority: Processing priority
            
        Returns:
            Job ID for tracking
        """
        try:
            if not job_id:
                job_id = str(uuid.uuid4())
            
            # Extract metadata
            from ...text_processing.preprocessor import TextPreprocessor
            preprocessor = TextPreprocessor()
            metadata = preprocessor.analyze(text_content)
            
            # Create chunks
            from ...text_processing.segmentation.chunking import ChunkManager
            chunk_manager = ChunkManager()
            
            scene_breaks = metadata.get('scene_breaks', [])
            if hasattr(chunk_manager, 'create_sliding_windows'):
                windows = chunk_manager.create_sliding_windows(text_content, scene_breaks, metadata)
            else:
                chunks = chunk_manager.create_chunks(text_content, scene_breaks)
                windows = chunk_manager._convert_chunks_to_windows(chunks)
            
            # Create chunk tasks
            chunk_tasks = self.chunk_producer.create_chunk_tasks(
                job_id=job_id,
                text_windows=windows,
                metadata=metadata,
                priority=priority
            )
            
            # Submit for processing
            success = self.chunk_producer.submit_chunks_for_processing(
                chunks=chunk_tasks,
                processing_config=processing_config
            )
            
            if success:
                self.logger.info(f"Distributed processing initiated for job {job_id}")
                return job_id
            else:
                raise Exception("Failed to submit chunks for processing")
                
        except Exception as e:
            self.logger.error(f"Error in distributed processing: {e}")
            raise
    
    def close(self):
        """Close the processor."""
        self.chunk_producer.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Utility functions
def create_chunk_tasks_from_text(text_content: str, job_id: str = None,
                                priority: int = 0) -> Tuple[str, List[ChunkTask]]:
    """Create chunk tasks from text content."""
    if not job_id:
        job_id = str(uuid.uuid4())
    
    with ChunkProducer() as producer:
        # Extract metadata
        from ...text_processing.preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor()
        metadata = preprocessor.analyze(text_content)
        
        # Create chunks
        from ...text_processing.segmentation.chunking import ChunkManager
        chunk_manager = ChunkManager()
        
        scene_breaks = metadata.get('scene_breaks', [])
        if hasattr(chunk_manager, 'create_sliding_windows'):
            windows = chunk_manager.create_sliding_windows(text_content, scene_breaks, metadata)
        else:
            chunks = chunk_manager.create_chunks(text_content, scene_breaks)
            windows = chunk_manager._convert_chunks_to_windows(chunks)
        
        # Create chunk tasks
        chunk_tasks = producer.create_chunk_tasks(
            job_id=job_id,
            text_windows=windows,
            metadata=metadata,
            priority=priority
        )
        
        return job_id, chunk_tasks


def submit_chunks_for_processing(chunks: List[ChunkTask],
                                processing_config: Dict[str, Any] = None) -> bool:
    """Submit chunks for processing using default producer."""
    with ChunkProducer() as producer:
        return producer.submit_chunks_for_processing(chunks, processing_config)


def process_text_distributed(text_content: str, job_id: str = None,
                            processing_config: Dict[str, Any] = None,
                            priority: int = 0) -> str:
    """Process text using distributed processing."""
    with DistributedChunkProcessor() as processor:
        return processor.process_text_distributed(
            text_content=text_content,
            job_id=job_id,
            processing_config=processing_config,
            priority=priority
        )