"""
File upload producer for Kafka-based text processing pipeline.

This module handles the initial file upload events and text extraction requests
in the distributed text-to-audiobook system.
"""

import json
import logging
import time
import uuid
from typing import Dict, Any, Optional, List
from pathlib import Path
import os

from kafka import KafkaProducer
from kafka.errors import KafkaError

from ..kafka_config import KafkaConfig, MessageRouting
from ..schemas.message_schemas import (
    MessageFactory, TextExtractionRequest, StatusUpdate, ErrorEvent,
    ProcessingStatus, MessageType
)


class FileUploadProducer:
    """Producer for file upload and text extraction events."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize file upload producer."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.producer = None
        self.routing = MessageRouting.get_routing_config()
        
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
            self.logger.info("File upload producer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize file upload producer: {e}")
            raise
    
    def submit_file_for_processing(self, file_path: str, user_id: str, 
                                  processing_options: Dict[str, Any] = None,
                                  priority: int = 0) -> str:
        """
        Submit a file for processing.
        
        Args:
            file_path: Path to the file to process
            user_id: ID of the user submitting the file
            processing_options: Optional processing configuration
            priority: Processing priority (higher = more priority)
            
        Returns:
            Job ID for tracking the processing
        """
        try:
            # Validate file path
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Create extraction request message
            extraction_request = MessageFactory.create_text_extraction_request(
                file_path=file_path,
                user_id=user_id,
                processing_options=processing_options or {}
            )
            
            # Add priority information
            extraction_request.metadata['priority'] = priority
            extraction_request.metadata['file_size'] = os.path.getsize(file_path)
            extraction_request.metadata['file_extension'] = Path(file_path).suffix
            
            # Send to Kafka
            self._send_message(
                topic=self.routing['file_upload'],
                message=extraction_request,
                key=extraction_request.job_id
            )
            
            # Send initial status update
            self._send_status_update(
                job_id=extraction_request.job_id,
                status=ProcessingStatus.PENDING,
                progress=0.0,
                current_step="File submitted for processing"
            )
            
            self.logger.info(f"File submitted for processing: {file_path} (job_id: {extraction_request.job_id})")
            
            return extraction_request.job_id
            
        except Exception as e:
            self.logger.error(f"Error submitting file for processing: {e}")
            
            # Send error event
            error_event = MessageFactory.create_error_event(
                error_type="FileSubmissionError",
                error_message=str(e),
                error_details={'file_path': file_path, 'user_id': user_id},
                component="FileUploadProducer"
            )
            
            self._send_message(
                topic=self.routing['error'],
                message=error_event,
                key=error_event.job_id
            )
            
            raise
    
    def submit_batch_files(self, file_paths: List[str], user_id: str,
                          processing_options: Dict[str, Any] = None,
                          priority: int = 0) -> List[str]:
        """
        Submit multiple files for batch processing.
        
        Args:
            file_paths: List of file paths to process
            user_id: ID of the user submitting the files
            processing_options: Optional processing configuration
            priority: Processing priority
            
        Returns:
            List of job IDs for tracking
        """
        job_ids = []
        
        for file_path in file_paths:
            try:
                job_id = self.submit_file_for_processing(
                    file_path=file_path,
                    user_id=user_id,
                    processing_options=processing_options,
                    priority=priority
                )
                job_ids.append(job_id)
                
            except Exception as e:
                self.logger.error(f"Error submitting file {file_path}: {e}")
                # Continue with other files
        
        self.logger.info(f"Batch submitted: {len(job_ids)} files successfully submitted")
        
        return job_ids
    
    def notify_text_extracted(self, job_id: str, extracted_text: str, 
                             text_metadata: Dict[str, Any], 
                             file_info: Dict[str, Any]) -> bool:
        """
        Notify that text extraction is complete.
        
        Args:
            job_id: Job ID for the extraction
            extracted_text: The extracted text content
            text_metadata: Metadata about the text
            file_info: Information about the source file
            
        Returns:
            True if notification sent successfully
        """
        try:
            from ..schemas.message_schemas import TextExtractionResult
            
            # Create extraction result message
            extraction_result = TextExtractionResult(
                extracted_text=extracted_text,
                text_metadata=text_metadata,
                file_info=file_info,
                job_id=job_id
            )
            
            # Send to Kafka
            self._send_message(
                topic=self.routing['text_extracted'],
                message=extraction_result,
                key=job_id
            )
            
            # Send status update
            self._send_status_update(
                job_id=job_id,
                status=ProcessingStatus.IN_PROGRESS,
                progress=20.0,
                current_step="Text extraction completed"
            )
            
            self.logger.info(f"Text extraction completed notification sent for job {job_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending text extraction notification: {e}")
            return False
    
    def notify_processing_complete(self, job_id: str, result_data: Dict[str, Any],
                                  processing_stats: Dict[str, Any],
                                  output_path: str = None) -> bool:
        """
        Notify that processing is complete.
        
        Args:
            job_id: Job ID for the processing
            result_data: Processing result data
            processing_stats: Statistics about the processing
            output_path: Path to output file (optional)
            
        Returns:
            True if notification sent successfully
        """
        try:
            from ..schemas.message_schemas import ProcessingComplete
            
            # Create processing complete message
            processing_complete = ProcessingComplete(
                status=ProcessingStatus.COMPLETED,
                result_data=result_data,
                processing_stats=processing_stats,
                output_path=output_path,
                job_id=job_id
            )
            
            # Send to Kafka
            self._send_message(
                topic=self.routing['processing_complete'],
                message=processing_complete,
                key=job_id
            )
            
            # Send final status update
            self._send_status_update(
                job_id=job_id,
                status=ProcessingStatus.COMPLETED,
                progress=100.0,
                current_step="Processing completed successfully"
            )
            
            self.logger.info(f"Processing complete notification sent for job {job_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending processing complete notification: {e}")
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
    
    def get_producer_metrics(self) -> Dict[str, Any]:
        """Get producer metrics."""
        if not self.producer:
            return {}
        
        try:
            metrics = self.producer.metrics()
            
            # Extract key metrics
            key_metrics = {}
            for metric_name, metric_value in metrics.items():
                if any(key in metric_name for key in ['record-send-rate', 'record-send-total', 'record-error-rate']):
                    key_metrics[metric_name] = metric_value
            
            return key_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting producer metrics: {e}")
            return {}
    
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
                self.logger.info("File upload producer closed")
                
        except Exception as e:
            self.logger.error(f"Error closing producer: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class TextExtractionProducer:
    """Specialized producer for text extraction events."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize text extraction producer."""
        self.file_upload_producer = FileUploadProducer(config)
        self.logger = logging.getLogger(__name__)
    
    def extract_and_notify(self, file_path: str, user_id: str, 
                          processing_options: Dict[str, Any] = None) -> str:
        """
        Extract text from file and notify via Kafka.
        
        Args:
            file_path: Path to the file to extract text from
            user_id: ID of the user
            processing_options: Optional processing configuration
            
        Returns:
            Job ID for tracking
        """
        try:
            # Submit file for processing
            job_id = self.file_upload_producer.submit_file_for_processing(
                file_path=file_path,
                user_id=user_id,
                processing_options=processing_options
            )
            
            # Perform text extraction
            from ...text_processing.text_extractor import TextExtractor
            
            extractor = TextExtractor()
            extracted_text = extractor.extract(file_path)
            
            # Create metadata
            text_metadata = {
                'character_count': len(extracted_text),
                'word_count': len(extracted_text.split()),
                'line_count': len(extracted_text.split('\n')),
                'extraction_time': time.time()
            }
            
            # Create file info
            file_info = {
                'file_path': file_path,
                'file_size': os.path.getsize(file_path),
                'file_extension': Path(file_path).suffix,
                'file_name': Path(file_path).name
            }
            
            # Notify extraction complete
            self.file_upload_producer.notify_text_extracted(
                job_id=job_id,
                extracted_text=extracted_text,
                text_metadata=text_metadata,
                file_info=file_info
            )
            
            return job_id
            
        except Exception as e:
            self.logger.error(f"Error in extract_and_notify: {e}")
            raise
    
    def close(self):
        """Close the producer."""
        self.file_upload_producer.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Utility functions for easy integration
def submit_file_for_processing(file_path: str, user_id: str,
                              processing_options: Dict[str, Any] = None,
                              priority: int = 0) -> str:
    """Submit a file for processing using default producer."""
    with FileUploadProducer() as producer:
        return producer.submit_file_for_processing(
            file_path=file_path,
            user_id=user_id,
            processing_options=processing_options,
            priority=priority
        )


def extract_text_and_notify(file_path: str, user_id: str,
                           processing_options: Dict[str, Any] = None) -> str:
    """Extract text from file and notify via Kafka."""
    with TextExtractionProducer() as producer:
        return producer.extract_and_notify(
            file_path=file_path,
            user_id=user_id,
            processing_options=processing_options
        )


def submit_batch_files(file_paths: List[str], user_id: str,
                      processing_options: Dict[str, Any] = None,
                      priority: int = 0) -> List[str]:
    """Submit multiple files for batch processing."""
    with FileUploadProducer() as producer:
        return producer.submit_batch_files(
            file_paths=file_paths,
            user_id=user_id,
            processing_options=processing_options,
            priority=priority
        )