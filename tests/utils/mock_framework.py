"""
Comprehensive mocking framework for external dependencies.

Provides realistic mocks for LLM services, TTS engines, and other external
dependencies with configurable behavior, failure simulation, and performance
characteristics.
"""

import json
import time
import random
import uuid
from typing import Dict, Any, List, Optional, Callable, Union
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import threading
from enum import Enum
from abc import ABC, abstractmethod


class MockBehavior(Enum):
    """Mock behavior modes."""
    REALISTIC = "realistic"          # Realistic response times and occasional failures
    FAST = "fast"                   # Fast responses for unit testing
    SLOW = "slow"                   # Slow responses for timeout testing
    UNRELIABLE = "unreliable"       # High failure rate for error handling tests
    PERFECT = "perfect"             # Always succeeds with perfect responses


@dataclass
class MockConfiguration:
    """Configuration for mock behavior."""
    behavior: MockBehavior = MockBehavior.REALISTIC
    response_delay_range: tuple = (0.05, 0.2)  # seconds
    failure_rate: float = 0.02  # 2% failure rate
    timeout_probability: float = 0.01  # 1% timeout probability
    api_rate_limit: Optional[int] = None  # requests per minute
    enable_logging: bool = True
    persist_state: bool = False


@dataclass
class APICallRecord:
    """Record of an API call for analysis."""
    timestamp: datetime
    service: str
    method: str
    request_data: Dict[str, Any]
    response_data: Optional[Dict[str, Any]]
    duration: float
    success: bool
    error_message: Optional[str] = None


class BaseMockService(ABC):
    """Base class for mock external services."""
    
    def __init__(self, service_name: str, config: MockConfiguration):
        self.service_name = service_name
        self.config = config
        self.call_history: List[APICallRecord] = []
        self.state: Dict[str, Any] = {}
        self.rate_limiter = RateLimiter(config.api_rate_limit) if config.api_rate_limit else None
        self._lock = threading.Lock()
    
    def _simulate_api_call(self, method: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate an API call with configurable behavior."""
        start_time = time.time()
        
        # Rate limiting
        if self.rate_limiter and not self.rate_limiter.allow_request():
            raise Exception("API rate limit exceeded")
        
        # Simulate processing delay
        delay = self._calculate_delay()
        time.sleep(delay)
        
        # Simulate failures
        if self._should_fail():
            error_msg = self._generate_error_message()
            self._record_call(method, request_data, None, time.time() - start_time, False, error_msg)
            raise Exception(error_msg)
        
        # Generate response
        response_data = self._generate_response(method, request_data)
        duration = time.time() - start_time
        
        # Record call
        self._record_call(method, request_data, response_data, duration, True)
        
        return response_data
    
    def _calculate_delay(self) -> float:
        """Calculate API response delay based on behavior mode."""
        if self.config.behavior == MockBehavior.FAST:
            return random.uniform(0.001, 0.005)
        elif self.config.behavior == MockBehavior.SLOW:
            return random.uniform(1.0, 3.0)
        elif self.config.behavior == MockBehavior.PERFECT:
            return 0.001
        else:  # REALISTIC or UNRELIABLE
            return random.uniform(*self.config.response_delay_range)
    
    def _should_fail(self) -> bool:
        """Determine if the API call should fail."""
        if self.config.behavior == MockBehavior.PERFECT:
            return False
        elif self.config.behavior == MockBehavior.UNRELIABLE:
            return random.random() < 0.2  # 20% failure rate
        else:
            return random.random() < self.config.failure_rate
    
    def _generate_error_message(self) -> str:
        """Generate a realistic error message."""
        errors = [
            "Service temporarily unavailable",
            "Request timeout",
            "Invalid API key",
            "Rate limit exceeded",
            "Internal server error",
            "Bad request format",
            "Network connection failed"
        ]
        return random.choice(errors)
    
    def _record_call(self, method: str, request_data: Dict[str, Any], 
                    response_data: Optional[Dict[str, Any]], duration: float, 
                    success: bool, error_message: Optional[str] = None):
        """Record API call for analysis."""
        with self._lock:
            record = APICallRecord(
                timestamp=datetime.now(),
                service=self.service_name,
                method=method,
                request_data=request_data.copy(),
                response_data=response_data.copy() if response_data else None,
                duration=duration,
                success=success,
                error_message=error_message
            )
            self.call_history.append(record)
    
    @abstractmethod
    def _generate_response(self, method: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate service-specific response."""
        pass
    
    def get_call_history(self) -> List[APICallRecord]:
        """Get call history for analysis."""
        with self._lock:
            return self.call_history.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get call statistics."""
        with self._lock:
            if not self.call_history:
                return {'total_calls': 0}
            
            successful_calls = [c for c in self.call_history if c.success]
            failed_calls = [c for c in self.call_history if not c.success]
            
            return {
                'total_calls': len(self.call_history),
                'successful_calls': len(successful_calls),
                'failed_calls': len(failed_calls),
                'success_rate': len(successful_calls) / len(self.call_history),
                'avg_response_time': sum(c.duration for c in successful_calls) / len(successful_calls) if successful_calls else 0,
                'min_response_time': min(c.duration for c in successful_calls) if successful_calls else 0,
                'max_response_time': max(c.duration for c in successful_calls) if successful_calls else 0
            }
    
    def reset_state(self):
        """Reset service state and call history."""
        with self._lock:
            self.call_history.clear()
            self.state.clear()


class RateLimiter:
    """Simple rate limiter for API mocking."""
    
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.request_times: List[float] = []
        self._lock = threading.Lock()
    
    def allow_request(self) -> bool:
        """Check if request is allowed under rate limit."""
        with self._lock:
            current_time = time.time()
            
            # Remove requests older than 1 minute
            cutoff_time = current_time - 60
            self.request_times = [t for t in self.request_times if t > cutoff_time]
            
            # Check if we're under the limit
            if len(self.request_times) < self.requests_per_minute:
                self.request_times.append(current_time)
                return True
            
            return False


class MockLLMService(BaseMockService):
    """Mock LLM service with realistic behavior."""
    
    def __init__(self, config: MockConfiguration = None):
        super().__init__("llm_service", config or MockConfiguration())
        self.supported_models = ["gpt-3.5-turbo", "gpt-4", "claude-3", "mistral", "llama3"]
        self.speaker_pool = ["narrator", "alice", "bob", "charlie", "diana", "eve", "frank"]
    
    def process_segment(self, segment: Dict[str, Any], model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        """Process a text segment for speaker attribution."""
        request_data = {
            'method': 'process_segment',
            'model': model,
            'segment_id': segment.get('segment_id'),
            'text_length': len(segment.get('text_content', '')),
            'current_speaker': segment.get('speaker')
        }
        
        response = self._simulate_api_call('process_segment', request_data)
        return response
    
    def classify_text_type(self, text: str, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        """Classify text as dialogue, narrative, or mixed."""
        request_data = {
            'method': 'classify_text_type',
            'model': model,
            'text_length': len(text)
        }
        
        response = self._simulate_api_call('classify_text_type', request_data)
        return response
    
    def extract_speakers(self, text: str, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        """Extract potential speakers from text."""
        request_data = {
            'method': 'extract_speakers',
            'model': model,
            'text_length': len(text)
        }
        
        response = self._simulate_api_call('extract_speakers', request_data)
        return response
    
    def _generate_response(self, method: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate LLM service responses."""
        if method == 'process_segment':
            return self._generate_segment_response(request_data)
        elif method == 'classify_text_type':
            return self._generate_classification_response(request_data)
        elif method == 'extract_speakers':
            return self._generate_speaker_extraction_response(request_data)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _generate_segment_response(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate realistic segment processing response."""
        # Simulate speaker assignment logic
        current_speaker = request_data.get('current_speaker', 'AMBIGUOUS')
        
        if current_speaker != 'AMBIGUOUS' and random.random() > 0.1:
            # Keep existing speaker 90% of the time
            assigned_speaker = current_speaker
            confidence = random.uniform(0.85, 0.98)
        else:
            # Assign new speaker
            assigned_speaker = random.choice(self.speaker_pool)
            confidence = random.uniform(0.70, 0.95)
        
        return {
            'segment_id': request_data.get('segment_id', 'unknown'),
            'speaker': assigned_speaker,
            'confidence': confidence,
            'reasoning': f"Assigned based on context analysis and speaker patterns",
            'alternative_speakers': random.sample(
                [s for s in self.speaker_pool if s != assigned_speaker], 
                min(2, len(self.speaker_pool) - 1)
            ),
            'processing_metadata': {
                'model': request_data.get('model', 'unknown'),
                'tokens_processed': request_data.get('text_length', 0) // 4,  # Rough token estimate
                'processing_time_ms': random.uniform(50, 200)
            }
        }
    
    def _generate_classification_response(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text classification response."""
        text_length = request_data.get('text_length', 0)
        
        # Simulate classification based on text length and patterns
        if text_length < 50:
            text_type = random.choice(['dialogue', 'narrative'])
            confidence = random.uniform(0.7, 0.9)
        else:
            text_type = random.choice(['mixed', 'narrative', 'dialogue'])
            confidence = random.uniform(0.8, 0.95)
        
        return {
            'text_type': text_type,
            'confidence': confidence,
            'dialogue_percentage': random.uniform(0.2, 0.8) if text_type != 'narrative' else random.uniform(0.0, 0.1),
            'narrative_percentage': random.uniform(0.2, 0.8) if text_type != 'dialogue' else random.uniform(0.0, 0.1),
            'features_detected': random.sample([
                'quotation_marks', 'speaker_tags', 'descriptive_language', 
                'action_words', 'dialogue_markers'
            ], random.randint(2, 4))
        }
    
    def _generate_speaker_extraction_response(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate speaker extraction response."""
        num_speakers = random.randint(2, 5)
        detected_speakers = random.sample(self.speaker_pool, num_speakers)
        
        return {
            'speakers': [
                {
                    'name': speaker,
                    'confidence': random.uniform(0.7, 0.95),
                    'occurrences': random.randint(1, 10),
                    'context_clues': random.sample([
                        'name_mention', 'dialogue_attribution', 'character_description'
                    ], random.randint(1, 3))
                }
                for speaker in detected_speakers
            ],
            'total_speakers_detected': num_speakers,
            'extraction_confidence': random.uniform(0.8, 0.95)
        }


class MockTTSService(BaseMockService):
    """Mock Text-to-Speech service."""
    
    def __init__(self, config: MockConfiguration = None):
        super().__init__("tts_service", config or MockConfiguration())
        self.voice_models = {
            'narrator': ['david', 'matthew', 'brian'],
            'young_male': ['alex', 'sam', 'ryan'],
            'young_female': ['emma', 'sophia', 'olivia'],
            'adult_male': ['james', 'robert', 'william'],
            'adult_female': ['mary', 'patricia', 'jennifer'],
            'elderly_male': ['john', 'charles', 'thomas'],
            'elderly_female': ['barbara', 'margaret', 'helen']
        }
    
    def synthesize_speech(self, text: str, voice: str, format: str = 'mp3') -> Dict[str, Any]:
        """Synthesize speech from text."""
        request_data = {
            'method': 'synthesize_speech',
            'text_length': len(text),
            'voice': voice,
            'format': format
        }
        
        response = self._simulate_api_call('synthesize_speech', request_data)
        return response
    
    def get_available_voices(self) -> Dict[str, Any]:
        """Get available TTS voices."""
        request_data = {'method': 'get_available_voices'}
        response = self._simulate_api_call('get_available_voices', request_data)
        return response
    
    def estimate_synthesis_time(self, text: str, voice: str) -> Dict[str, Any]:
        """Estimate synthesis time for text."""
        request_data = {
            'method': 'estimate_synthesis_time',
            'text_length': len(text),
            'voice': voice
        }
        
        response = self._simulate_api_call('estimate_synthesis_time', request_data)
        return response
    
    def _generate_response(self, method: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate TTS service responses."""
        if method == 'synthesize_speech':
            return self._generate_synthesis_response(request_data)
        elif method == 'get_available_voices':
            return self._generate_voices_response(request_data)
        elif method == 'estimate_synthesis_time':
            return self._generate_time_estimate_response(request_data)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _generate_synthesis_response(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate speech synthesis response."""
        text_length = request_data.get('text_length', 0)
        voice = request_data.get('voice', 'default')
        format = request_data.get('format', 'mp3')
        
        # Simulate audio file generation
        audio_duration = text_length * 0.01  # ~100 characters per second
        file_size = int(audio_duration * 32000)  # Rough file size estimation
        
        return {
            'audio_url': f"https://mock-tts.com/audio/{uuid.uuid4().hex}.{format}",
            'audio_duration_seconds': audio_duration,
            'file_size_bytes': file_size,
            'voice_used': voice,
            'format': format,
            'synthesis_metadata': {
                'characters_processed': text_length,
                'processing_time_ms': random.uniform(100, 500),
                'voice_quality': random.choice(['standard', 'premium', 'neural'])
            }
        }
    
    def _generate_voices_response(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate available voices response."""
        voices = []
        for category, voice_list in self.voice_models.items():
            for voice in voice_list:
                voices.append({
                    'voice_id': voice,
                    'voice_name': voice.title(),
                    'category': category,
                    'language': 'en-US',
                    'gender': 'male' if 'male' in category else 'female' if 'female' in category else 'neutral',
                    'age_group': category.split('_')[0] if '_' in category else 'adult',
                    'quality': random.choice(['standard', 'premium', 'neural']),
                    'sample_url': f"https://mock-tts.com/samples/{voice}.mp3"
                })
        
        return {
            'voices': voices,
            'total_voices': len(voices),
            'supported_languages': ['en-US', 'en-GB', 'en-AU'],
            'supported_formats': ['mp3', 'wav', 'ogg', 'm4a']
        }
    
    def _generate_time_estimate_response(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate synthesis time estimate response."""
        text_length = request_data.get('text_length', 0)
        
        # Simulate processing time estimation
        base_time = text_length * 0.002  # 2ms per character
        queue_delay = random.uniform(0, 30)  # 0-30 seconds queue time
        total_estimate = base_time + queue_delay
        
        return {
            'estimated_processing_time_seconds': total_estimate,
            'queue_position': random.randint(0, 50),
            'estimated_completion_time': (datetime.now() + timedelta(seconds=total_estimate)).isoformat(),
            'cost_estimate': {
                'characters': text_length,
                'cost_per_character': 0.000016,  # $0.000016 per character
                'total_cost': text_length * 0.000016
            }
        }


class MockCloudStorageService(BaseMockService):
    """Mock cloud storage service for file operations."""
    
    def __init__(self, config: MockConfiguration = None):
        super().__init__("cloud_storage", config or MockConfiguration())
        self.storage_buckets = ['audio-files', 'processed-text', 'temp-storage']
    
    def upload_file(self, file_path: str, bucket: str, key: str) -> Dict[str, Any]:
        """Upload file to cloud storage."""
        request_data = {
            'method': 'upload_file',
            'file_path': file_path,
            'bucket': bucket,
            'key': key,
            'file_size': random.randint(1000, 10000000)  # Random file size
        }
        
        response = self._simulate_api_call('upload_file', request_data)
        return response
    
    def download_file(self, bucket: str, key: str, local_path: str) -> Dict[str, Any]:
        """Download file from cloud storage."""
        request_data = {
            'method': 'download_file',
            'bucket': bucket,
            'key': key,
            'local_path': local_path
        }
        
        response = self._simulate_api_call('download_file', request_data)
        return response
    
    def list_files(self, bucket: str, prefix: str = '') -> Dict[str, Any]:
        """List files in bucket."""
        request_data = {
            'method': 'list_files',
            'bucket': bucket,
            'prefix': prefix
        }
        
        response = self._simulate_api_call('list_files', request_data)
        return response
    
    def _generate_response(self, method: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cloud storage responses."""
        if method == 'upload_file':
            return self._generate_upload_response(request_data)
        elif method == 'download_file':
            return self._generate_download_response(request_data)
        elif method == 'list_files':
            return self._generate_list_response(request_data)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _generate_upload_response(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate file upload response."""
        return {
            'upload_id': uuid.uuid4().hex,
            'bucket': request_data.get('bucket'),
            'key': request_data.get('key'),
            'file_size': request_data.get('file_size'),
            'etag': uuid.uuid4().hex,
            'url': f"https://mock-storage.com/{request_data.get('bucket')}/{request_data.get('key')}",
            'upload_time': datetime.now().isoformat(),
            'storage_class': 'STANDARD'
        }
    
    def _generate_download_response(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate file download response."""
        return {
            'bucket': request_data.get('bucket'),
            'key': request_data.get('key'),
            'local_path': request_data.get('local_path'),
            'file_size': random.randint(1000, 10000000),
            'download_time': datetime.now().isoformat(),
            'content_type': 'audio/mpeg'
        }
    
    def _generate_list_response(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate file list response."""
        num_files = random.randint(5, 50)
        files = []
        
        for i in range(num_files):
            files.append({
                'key': f"{request_data.get('prefix', '')}file_{i:03d}.mp3",
                'size': random.randint(1000, 5000000),
                'last_modified': (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
                'etag': uuid.uuid4().hex,
                'storage_class': random.choice(['STANDARD', 'IA', 'ARCHIVE'])
            })
        
        return {
            'bucket': request_data.get('bucket'),
            'prefix': request_data.get('prefix', ''),
            'files': files,
            'total_files': num_files,
            'total_size': sum(f['size'] for f in files)
        }


class MockServiceRegistry:
    """Registry for managing mock services."""
    
    def __init__(self):
        self.services: Dict[str, BaseMockService] = {}
        self.global_config = MockConfiguration()
    
    def register_service(self, name: str, service: BaseMockService):
        """Register a mock service."""
        self.services[name] = service
    
    def get_service(self, name: str) -> Optional[BaseMockService]:
        """Get a registered service."""
        return self.services.get(name)
    
    def create_llm_service(self, config: MockConfiguration = None) -> MockLLMService:
        """Create and register LLM service."""
        service = MockLLMService(config or self.global_config)
        self.register_service('llm', service)
        return service
    
    def create_tts_service(self, config: MockConfiguration = None) -> MockTTSService:
        """Create and register TTS service."""
        service = MockTTSService(config or self.global_config)
        self.register_service('tts', service)
        return service
    
    def create_storage_service(self, config: MockConfiguration = None) -> MockCloudStorageService:
        """Create and register cloud storage service."""
        service = MockCloudStorageService(config or self.global_config)
        self.register_service('storage', service)
        return service
    
    def reset_all_services(self):
        """Reset all registered services."""
        for service in self.services.values():
            service.reset_state()
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all services."""
        return {
            name: service.get_statistics() 
            for name, service in self.services.items()
        }


# Global service registry
_mock_registry = MockServiceRegistry()


def get_mock_registry() -> MockServiceRegistry:
    """Get the global mock service registry."""
    return _mock_registry


# Convenience functions for common mock setups
def setup_realistic_mocks() -> MockServiceRegistry:
    """Setup realistic mock services for integration testing."""
    registry = get_mock_registry()
    config = MockConfiguration(behavior=MockBehavior.REALISTIC)
    
    registry.create_llm_service(config)
    registry.create_tts_service(config)
    registry.create_storage_service(config)
    
    return registry


def setup_fast_mocks() -> MockServiceRegistry:
    """Setup fast mock services for unit testing."""
    registry = get_mock_registry()
    config = MockConfiguration(behavior=MockBehavior.FAST)
    
    registry.create_llm_service(config)
    registry.create_tts_service(config)
    registry.create_storage_service(config)
    
    return registry


def setup_unreliable_mocks() -> MockServiceRegistry:
    """Setup unreliable mock services for error handling testing."""
    registry = get_mock_registry()
    config = MockConfiguration(behavior=MockBehavior.UNRELIABLE)
    
    registry.create_llm_service(config)
    registry.create_tts_service(config)
    registry.create_storage_service(config)
    
    return registry


def patch_llm_service(service: MockLLMService):
    """Patch LLM service imports with mock."""
    return patch.multiple(
        'src.llm_pool.llm_pool_manager',
        get_pool_manager=lambda: service
    )


def patch_tts_service(service: MockTTSService):
    """Patch TTS service imports with mock."""
    return patch.multiple(
        'src.audio.tts_engines',
        get_tts_engine=lambda: service
    )


def patch_storage_service(service: MockCloudStorageService):
    """Patch cloud storage service imports with mock."""
    return patch.multiple(
        'src.storage.cloud_storage',
        get_storage_client=lambda: service
    )