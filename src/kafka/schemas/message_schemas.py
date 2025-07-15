"""
Message schemas for Kafka messages in the text-to-audiobook system.

This module defines the structure and validation schemas for all messages
passed through the Kafka event-driven architecture.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
import uuid
from enum import Enum


class MessageType(Enum):
    """Enumeration of message types."""
    TEXT_EXTRACTION_REQUEST = "text_extraction_request"
    TEXT_EXTRACTION_RESULT = "text_extraction_result"
    PROCESSING_REQUEST = "processing_request"
    CHUNK_PROCESSING = "chunk_processing"
    LLM_CLASSIFICATION_REQUEST = "llm_classification_request"
    LLM_CLASSIFICATION_RESULT = "llm_classification_result"
    VALIDATION_RESULT = "validation_result"
    PROCESSING_COMPLETE = "processing_complete"
    ERROR_EVENT = "error_event"
    STATUS_UPDATE = "status_update"
    METRICS_EVENT = "metrics_event"


class ProcessingStatus(Enum):
    """Enumeration of processing statuses."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BaseMessage:
    """Base class for all Kafka messages."""
    
    def __init__(self, message_type: MessageType, job_id: str = None, 
                 timestamp: datetime = None, metadata: Dict[str, Any] = None):
        self.message_type = message_type.value
        self.job_id = job_id or str(uuid.uuid4())
        self.timestamp = timestamp or datetime.utcnow()
        self.metadata = metadata or {}
        self.version = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            'message_type': self.message_type,
            'job_id': self.job_id,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'version': self.version
        }
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseMessage':
        """Create message from dictionary."""
        return cls(
            message_type=MessageType(data['message_type']),
            job_id=data['job_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )


class TextExtractionRequest(BaseMessage):
    """Message for text extraction requests."""
    
    def __init__(self, file_path: str, user_id: str, processing_options: Dict[str, Any] = None,
                 job_id: str = None, timestamp: datetime = None, metadata: Dict[str, Any] = None):
        super().__init__(MessageType.TEXT_EXTRACTION_REQUEST, job_id, timestamp, metadata)
        self.file_path = file_path
        self.user_id = user_id
        self.processing_options = processing_options or {}
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'file_path': self.file_path,
            'user_id': self.user_id,
            'processing_options': self.processing_options
        })
        return data


class TextExtractionResult(BaseMessage):
    """Message for text extraction results."""
    
    def __init__(self, extracted_text: str, text_metadata: Dict[str, Any], 
                 file_info: Dict[str, Any], job_id: str = None, 
                 timestamp: datetime = None, metadata: Dict[str, Any] = None):
        super().__init__(MessageType.TEXT_EXTRACTION_RESULT, job_id, timestamp, metadata)
        self.extracted_text = extracted_text
        self.text_metadata = text_metadata
        self.file_info = file_info
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'extracted_text': self.extracted_text,
            'text_metadata': self.text_metadata,
            'file_info': self.file_info
        })
        return data


class ProcessingRequest(BaseMessage):
    """Message for processing requests."""
    
    def __init__(self, text_content: str, processing_config: Dict[str, Any],
                 priority: int = 0, job_id: str = None, timestamp: datetime = None,
                 metadata: Dict[str, Any] = None):
        super().__init__(MessageType.PROCESSING_REQUEST, job_id, timestamp, metadata)
        self.text_content = text_content
        self.processing_config = processing_config
        self.priority = priority
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'text_content': self.text_content,
            'processing_config': self.processing_config,
            'priority': self.priority
        })
        return data


class ChunkProcessing(BaseMessage):
    """Message for chunk processing."""
    
    def __init__(self, chunk_id: str, chunk_text: str, chunk_metadata: Dict[str, Any],
                 context_hint: Dict[str, Any] = None, chunk_index: int = 0,
                 total_chunks: int = 1, job_id: str = None, timestamp: datetime = None,
                 metadata: Dict[str, Any] = None):
        super().__init__(MessageType.CHUNK_PROCESSING, job_id, timestamp, metadata)
        self.chunk_id = chunk_id
        self.chunk_text = chunk_text
        self.chunk_metadata = chunk_metadata
        self.context_hint = context_hint or {}
        self.chunk_index = chunk_index
        self.total_chunks = total_chunks
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'chunk_id': self.chunk_id,
            'chunk_text': self.chunk_text,
            'chunk_metadata': self.chunk_metadata,
            'context_hint': self.context_hint,
            'chunk_index': self.chunk_index,
            'total_chunks': self.total_chunks
        })
        return data


class LLMClassificationRequest(BaseMessage):
    """Message for LLM classification requests."""
    
    def __init__(self, text_lines: List[str], prompt: str, model_config: Dict[str, Any],
                 chunk_id: str = None, job_id: str = None, timestamp: datetime = None,
                 metadata: Dict[str, Any] = None):
        super().__init__(MessageType.LLM_CLASSIFICATION_REQUEST, job_id, timestamp, metadata)
        self.text_lines = text_lines
        self.prompt = prompt
        self.model_config = model_config
        self.chunk_id = chunk_id
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'text_lines': self.text_lines,
            'prompt': self.prompt,
            'model_config': self.model_config,
            'chunk_id': self.chunk_id
        })
        return data


class LLMClassificationResult(BaseMessage):
    """Message for LLM classification results."""
    
    def __init__(self, classifications: List[str], confidence_scores: List[float] = None,
                 processing_time: float = None, chunk_id: str = None, 
                 job_id: str = None, timestamp: datetime = None,
                 metadata: Dict[str, Any] = None):
        super().__init__(MessageType.LLM_CLASSIFICATION_RESULT, job_id, timestamp, metadata)
        self.classifications = classifications
        self.confidence_scores = confidence_scores or []
        self.processing_time = processing_time
        self.chunk_id = chunk_id
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'classifications': self.classifications,
            'confidence_scores': self.confidence_scores,
            'processing_time': self.processing_time,
            'chunk_id': self.chunk_id
        })
        return data


class ValidationResult(BaseMessage):
    """Message for validation results."""
    
    def __init__(self, validation_score: float, validation_errors: List[Dict[str, Any]],
                 quality_metrics: Dict[str, Any], is_valid: bool = True,
                 job_id: str = None, timestamp: datetime = None,
                 metadata: Dict[str, Any] = None):
        super().__init__(MessageType.VALIDATION_RESULT, job_id, timestamp, metadata)
        self.validation_score = validation_score
        self.validation_errors = validation_errors
        self.quality_metrics = quality_metrics
        self.is_valid = is_valid
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'validation_score': self.validation_score,
            'validation_errors': self.validation_errors,
            'quality_metrics': self.quality_metrics,
            'is_valid': self.is_valid
        })
        return data


class ProcessingComplete(BaseMessage):
    """Message for processing completion."""
    
    def __init__(self, status: ProcessingStatus, result_data: Dict[str, Any],
                 processing_stats: Dict[str, Any], output_path: str = None,
                 job_id: str = None, timestamp: datetime = None,
                 metadata: Dict[str, Any] = None):
        super().__init__(MessageType.PROCESSING_COMPLETE, job_id, timestamp, metadata)
        self.status = status.value
        self.result_data = result_data
        self.processing_stats = processing_stats
        self.output_path = output_path
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'status': self.status,
            'result_data': self.result_data,
            'processing_stats': self.processing_stats,
            'output_path': self.output_path
        })
        return data


class ErrorEvent(BaseMessage):
    """Message for error events."""
    
    def __init__(self, error_type: str, error_message: str, error_details: Dict[str, Any],
                 stack_trace: str = None, component: str = None, severity: str = "ERROR",
                 job_id: str = None, timestamp: datetime = None,
                 metadata: Dict[str, Any] = None):
        super().__init__(MessageType.ERROR_EVENT, job_id, timestamp, metadata)
        self.error_type = error_type
        self.error_message = error_message
        self.error_details = error_details
        self.stack_trace = stack_trace
        self.component = component
        self.severity = severity
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'error_type': self.error_type,
            'error_message': self.error_message,
            'error_details': self.error_details,
            'stack_trace': self.stack_trace,
            'component': self.component,
            'severity': self.severity
        })
        return data


class StatusUpdate(BaseMessage):
    """Message for status updates."""
    
    def __init__(self, status: ProcessingStatus, progress: float, 
                 current_step: str, total_steps: int = None,
                 estimated_completion: datetime = None, job_id: str = None,
                 timestamp: datetime = None, metadata: Dict[str, Any] = None):
        super().__init__(MessageType.STATUS_UPDATE, job_id, timestamp, metadata)
        self.status = status.value
        self.progress = progress
        self.current_step = current_step
        self.total_steps = total_steps
        self.estimated_completion = estimated_completion
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'status': self.status,
            'progress': self.progress,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'estimated_completion': self.estimated_completion.isoformat() if self.estimated_completion else None
        })
        return data


class MetricsEvent(BaseMessage):
    """Message for metrics events."""
    
    def __init__(self, metric_name: str, metric_value: Union[int, float], 
                 metric_type: str, tags: Dict[str, str] = None,
                 job_id: str = None, timestamp: datetime = None,
                 metadata: Dict[str, Any] = None):
        super().__init__(MessageType.METRICS_EVENT, job_id, timestamp, metadata)
        self.metric_name = metric_name
        self.metric_value = metric_value
        self.metric_type = metric_type
        self.tags = tags or {}
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'metric_type': self.metric_type,
            'tags': self.tags
        })
        return data


# Schema validation utilities
class MessageValidator:
    """Validates message schemas."""
    
    @staticmethod
    def validate_base_message(data: Dict[str, Any]) -> bool:
        """Validate base message structure."""
        required_fields = ['message_type', 'job_id', 'timestamp', 'version']
        return all(field in data for field in required_fields)
    
    @staticmethod
    def validate_text_extraction_request(data: Dict[str, Any]) -> bool:
        """Validate text extraction request schema."""
        if not MessageValidator.validate_base_message(data):
            return False
        
        required_fields = ['file_path', 'user_id', 'processing_options']
        return all(field in data for field in required_fields)
    
    @staticmethod
    def validate_chunk_processing(data: Dict[str, Any]) -> bool:
        """Validate chunk processing schema."""
        if not MessageValidator.validate_base_message(data):
            return False
        
        required_fields = ['chunk_id', 'chunk_text', 'chunk_metadata', 'chunk_index', 'total_chunks']
        return all(field in data for field in required_fields)
    
    @staticmethod
    def validate_llm_classification_request(data: Dict[str, Any]) -> bool:
        """Validate LLM classification request schema."""
        if not MessageValidator.validate_base_message(data):
            return False
        
        required_fields = ['text_lines', 'prompt', 'model_config']
        return all(field in data for field in required_fields)
    
    @staticmethod
    def validate_message(data: Dict[str, Any]) -> bool:
        """Validate message based on its type."""
        message_type = data.get('message_type')
        
        if message_type == MessageType.TEXT_EXTRACTION_REQUEST.value:
            return MessageValidator.validate_text_extraction_request(data)
        elif message_type == MessageType.CHUNK_PROCESSING.value:
            return MessageValidator.validate_chunk_processing(data)
        elif message_type == MessageType.LLM_CLASSIFICATION_REQUEST.value:
            return MessageValidator.validate_llm_classification_request(data)
        else:
            return MessageValidator.validate_base_message(data)


# Message factory for creating messages
class MessageFactory:
    """Factory for creating messages."""
    
    @staticmethod
    def create_text_extraction_request(file_path: str, user_id: str, 
                                       processing_options: Dict[str, Any] = None) -> TextExtractionRequest:
        """Create text extraction request message."""
        return TextExtractionRequest(file_path, user_id, processing_options)
    
    @staticmethod
    def create_chunk_processing(chunk_id: str, chunk_text: str, 
                                chunk_metadata: Dict[str, Any]) -> ChunkProcessing:
        """Create chunk processing message."""
        return ChunkProcessing(chunk_id, chunk_text, chunk_metadata)
    
    @staticmethod
    def create_llm_classification_request(text_lines: List[str], prompt: str,
                                          model_config: Dict[str, Any]) -> LLMClassificationRequest:
        """Create LLM classification request message."""
        return LLMClassificationRequest(text_lines, prompt, model_config)
    
    @staticmethod
    def create_error_event(error_type: str, error_message: str, 
                           error_details: Dict[str, Any], component: str = None) -> ErrorEvent:
        """Create error event message."""
        return ErrorEvent(error_type, error_message, error_details, component=component)
    
    @staticmethod
    def create_status_update(status: ProcessingStatus, progress: float,
                             current_step: str, job_id: str) -> StatusUpdate:
        """Create status update message."""
        return StatusUpdate(status, progress, current_step, job_id=job_id)
    
    @staticmethod
    def create_metrics_event(metric_name: str, metric_value: Union[int, float],
                             metric_type: str, tags: Dict[str, str] = None) -> MetricsEvent:
        """Create metrics event message."""
        return MetricsEvent(metric_name, metric_value, metric_type, tags)


# Common message patterns
class MessagePatterns:
    """Common message patterns and utilities."""
    
    @staticmethod
    def create_processing_pipeline_messages(job_id: str, file_path: str, 
                                            user_id: str) -> List[BaseMessage]:
        """Create a complete processing pipeline message sequence."""
        messages = []
        
        # 1. Text extraction request
        extraction_request = MessageFactory.create_text_extraction_request(
            file_path, user_id
        )
        extraction_request.job_id = job_id
        messages.append(extraction_request)
        
        # 2. Status update for started processing
        status_update = MessageFactory.create_status_update(
            ProcessingStatus.IN_PROGRESS, 0.0, "Starting text extraction", job_id
        )
        messages.append(status_update)
        
        return messages
    
    @staticmethod
    def create_error_recovery_messages(original_job_id: str, error_details: Dict[str, Any]) -> List[BaseMessage]:
        """Create error recovery message sequence."""
        messages = []
        
        # 1. Error event
        error_event = MessageFactory.create_error_event(
            "ProcessingError", "Processing failed", error_details
        )
        error_event.job_id = original_job_id
        messages.append(error_event)
        
        # 2. Status update for failed processing
        status_update = MessageFactory.create_status_update(
            ProcessingStatus.FAILED, 0.0, "Processing failed", original_job_id
        )
        messages.append(status_update)
        
        return messages