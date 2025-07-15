"""
Pytest configuration and shared fixtures for the text-to-audiobook test suite.

This module provides comprehensive test fixtures for all components including
Spark sessions, Kafka clients, Redis connections, mock LLM services, and test data.
"""

import pytest
import tempfile
import shutil
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
import logging

# Test data imports
import pandas as pd
import numpy as np

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "spark: marks tests that require Spark"
    )
    config.addinivalue_line(
        "markers", "kafka: marks tests that require Kafka"
    )
    config.addinivalue_line(
        "markers", "redis: marks tests that require Redis"
    )
    config.addinivalue_line(
        "markers", "llm: marks tests that require LLM services"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add unit test marker by default
        if not any(item.iter_markers()):
            item.add_marker(pytest.mark.unit)
        
        # Add markers based on test path
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        if "spark" in str(item.fspath) or "spark" in item.name:
            item.add_marker(pytest.mark.spark)
        
        if "kafka" in str(item.fspath) or "kafka" in item.name:
            item.add_marker(pytest.mark.kafka)
        
        if "redis" in str(item.fspath) or "cache" in str(item.fspath):
            item.add_marker(pytest.mark.redis)
        
        if "llm" in str(item.fspath) or "llm" in item.name:
            item.add_marker(pytest.mark.llm)


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for the test session."""
    temp_path = tempfile.mkdtemp(prefix="text_to_audiobook_test_")
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_data_dir(temp_dir):
    """Create a test data directory with sample files."""
    data_dir = temp_dir / "test_data"
    data_dir.mkdir(exist_ok=True)
    
    # Create sample text files
    (data_dir / "sample.txt").write_text(
        "This is a sample text file for testing. "
        "It contains multiple sentences. "
        "Some dialogue: \"Hello, world!\" said the character."
    )
    
    # Create sample JSON data
    sample_segments = [
        {
            "segment_id": "seg_001",
            "text_content": "This is the first segment.",
            "speaker_id": "narrator",
            "segment_type": "narrative",
            "quality_score": 0.85
        },
        {
            "segment_id": "seg_002", 
            "text_content": "\"Hello, world!\" said the character.",
            "speaker_id": "character_1",
            "segment_type": "dialogue",
            "quality_score": 0.92
        }
    ]
    
    with open(data_dir / "sample_segments.json", "w") as f:
        json.dump(sample_segments, f)
    
    # Create sample speaker data
    speaker_data = [
        {
            "speaker_id": "narrator",
            "name": "Narrator",
            "voice_characteristics": "neutral, clear",
            "dialogue_segments": 15,
            "confidence_scores": "[0.8, 0.85, 0.82, 0.88, 0.9]"
        },
        {
            "speaker_id": "character_1",
            "name": "Character One",
            "voice_characteristics": "young, energetic",
            "dialogue_segments": 8,
            "confidence_scores": "[0.92, 0.89, 0.95, 0.91, 0.88]"
        }
    ]
    
    with open(data_dir / "sample_speakers.json", "w") as f:
        json.dump(speaker_data, f)
    
    return data_dir


@pytest.fixture
def output_dir(temp_dir):
    """Create a temporary output directory."""
    output_path = temp_dir / "output"
    output_path.mkdir(exist_ok=True)
    return output_path


# ============================================================================
# Spark Testing Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def spark_session():
    """Create a Spark session for testing."""
    try:
        from pyspark.sql import SparkSession
        
        spark = SparkSession.builder \
            .appName("TextToAudiobook_Test") \
            .master("local[2]") \
            .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
            .config("spark.sql.adaptive.enabled", "false") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
            .getOrCreate()
        
        # Set log level to reduce noise
        spark.sparkContext.setLogLevel("WARN")
        
        yield spark
        
        spark.stop()
        
    except ImportError:
        pytest.skip("PySpark not available")


@pytest.fixture
def spark_context(spark_session):
    """Get Spark context from session."""
    return spark_session.sparkContext


# ============================================================================
# Kafka Testing Fixtures  
# ============================================================================

@pytest.fixture
def mock_kafka_producer():
    """Create a mock Kafka producer."""
    producer = Mock()
    producer.send.return_value = Mock()
    producer.send.return_value.get.return_value = Mock()
    producer.flush.return_value = None
    producer.close.return_value = None
    return producer


@pytest.fixture
def mock_kafka_consumer():
    """Create a mock Kafka consumer."""
    consumer = Mock()
    consumer.subscribe.return_value = None
    consumer.poll.return_value = {}
    consumer.commit.return_value = None
    consumer.close.return_value = None
    return consumer


@pytest.fixture
def kafka_test_config():
    """Kafka configuration for testing."""
    return {
        'bootstrap_servers': ['localhost:9092'],
        'security_protocol': 'PLAINTEXT',
        'api_version': (0, 10, 1),
        'retries': 3,
        'batch_size': 16384,
        'linger_ms': 10,
        'buffer_memory': 33554432
    }


# ============================================================================
# Redis/Cache Testing Fixtures
# ============================================================================

@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client."""
    redis_mock = Mock()
    redis_mock.ping.return_value = True
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.setex.return_value = True
    redis_mock.delete.return_value = 1
    redis_mock.exists.return_value = False
    redis_mock.keys.return_value = []
    redis_mock.info.return_value = {
        'used_memory': 1024000,
        'used_memory_peak': 2048000
    }
    return redis_mock


@pytest.fixture
def cache_test_config():
    """Cache configuration for testing."""
    return {
        'host': 'localhost',
        'port': 6379,
        'db': 1,  # Use different DB for tests
        'namespace': 'test_text_to_audiobook',
        'default_ttl': 300,  # 5 minutes for tests
        'max_memory_mb': 100
    }


# ============================================================================
# LLM Testing Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_response():
    """Create mock LLM response data."""
    return {
        'choices': [
            {
                'text': 'This is a mock LLM response for testing purposes.',
                'finish_reason': 'stop',
                'index': 0
            }
        ],
        'usage': {
            'prompt_tokens': 10,
            'completion_tokens': 12,
            'total_tokens': 22
        },
        'model': 'test-model',
        'created': int(time.time())
    }


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = Mock()
    client.generate.return_value = {
        'text': 'Mock LLM response',
        'confidence': 0.95,
        'tokens_used': 50
    }
    client.health_check.return_value = True
    return client


@pytest.fixture
def llm_pool_config():
    """LLM pool configuration for testing."""
    return {
        'pool_size': 2,
        'max_concurrent_requests': 4,
        'request_timeout': 30,
        'health_check_interval': 60,
        'retry_attempts': 2,
        'models': {
            'text_analysis': {
                'model_name': 'test-text-model',
                'max_tokens': 1000,
                'temperature': 0.7
            },
            'dialogue_detection': {
                'model_name': 'test-dialogue-model',
                'max_tokens': 500,
                'temperature': 0.3
            }
        }
    }


# ============================================================================
# Monitoring Testing Fixtures
# ============================================================================

@pytest.fixture
def mock_prometheus_metrics():
    """Create mock Prometheus metrics collector."""
    metrics = Mock()
    metrics.record_processing_request.return_value = None
    metrics.record_processing_duration.return_value = None
    metrics.set_queue_size.return_value = None
    metrics.record_spark_job.return_value = None
    metrics.record_kafka_message.return_value = None
    metrics.record_llm_request.return_value = None
    metrics.set_system_health.return_value = None
    metrics.get_metrics.return_value = "# Mock metrics\n"
    return metrics


@pytest.fixture
def monitoring_test_config():
    """Monitoring configuration for testing."""
    return {
        'namespace': 'test_text_to_audiobook',
        'enable_push_gateway': False,
        'push_gateway_url': 'localhost:9091'
    }


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_text_segments():
    """Sample text segments for testing."""
    return [
        {
            'segment_id': 'seg_001',
            'text_content': 'This is the opening paragraph of our story.',
            'speaker_id': 'narrator',
            'segment_type': 'narrative',
            'quality_score': 0.85,
            'confidence_score': 0.92,
            'processing_metadata': '{"tokens": 9, "complexity": 3}'
        },
        {
            'segment_id': 'seg_002',
            'text_content': '"Hello there!" exclaimed the protagonist.',
            'speaker_id': 'protagonist',
            'segment_type': 'dialogue',
            'quality_score': 0.91,
            'confidence_score': 0.88,
            'processing_metadata': '{"tokens": 6, "complexity": 2}'
        },
        {
            'segment_id': 'seg_003',
            'text_content': 'The weather was particularly gloomy that day.',
            'speaker_id': 'narrator',
            'segment_type': 'narrative',
            'quality_score': 0.78,
            'confidence_score': 0.85,
            'processing_metadata': '{"tokens": 8, "complexity": 4}'
        }
    ]


@pytest.fixture
def sample_speaker_data():
    """Sample speaker data for testing."""
    return [
        {
            'speaker_id': 'narrator',
            'name': 'Narrator',
            'voice_characteristics': 'neutral, authoritative',
            'dialogue_segments': 25,
            'confidence_scores': '[0.85, 0.87, 0.82, 0.89, 0.91, 0.86, 0.88]'
        },
        {
            'speaker_id': 'protagonist',
            'name': 'Main Character',
            'voice_characteristics': 'young, enthusiastic',
            'dialogue_segments': 18,
            'confidence_scores': '[0.91, 0.89, 0.93, 0.87, 0.92, 0.90]'
        },
        {
            'speaker_id': 'antagonist',
            'name': 'Villain',
            'voice_characteristics': 'deep, menacing',
            'dialogue_segments': 12,
            'confidence_scores': '[0.88, 0.86, 0.91, 0.89, 0.85]'
        }
    ]


@pytest.fixture
def sample_workload_characteristics():
    """Sample workload characteristics for testing."""
    return {
        'data_size_mb': 50.0,
        'complexity_score': 6.5,
        'parallelism_potential': 0.8,
        'memory_intensive': False,
        'io_intensive': True,
        'cpu_intensive': False,
        'estimated_duration_minutes': 15.0
    }


@pytest.fixture
def sample_processing_job():
    """Sample processing job data."""
    return {
        'job_id': 'test_job_001',
        'input_file': 'sample_book.pdf',
        'job_type': 'pdf_to_audiobook',
        'status': 'processing',
        'created_at': datetime.now().isoformat(),
        'processing_config': {
            'voice_mapping': {
                'narrator': 'voice_001',
                'character_1': 'voice_002'
            },
            'output_format': 'mp3',
            'quality': 'high'
        },
        'segments': [
            {
                'segment_id': 'seg_001',
                'text': 'Chapter 1: The Beginning',
                'speaker': 'narrator',
                'start_time': 0.0,
                'duration': 2.5
            }
        ]
    }


# ============================================================================
# Error Simulation Fixtures
# ============================================================================

@pytest.fixture
def network_error():
    """Simulate network connectivity issues."""
    from requests.exceptions import ConnectionError
    return ConnectionError("Network unreachable")


@pytest.fixture
def timeout_error():
    """Simulate timeout errors."""
    from requests.exceptions import Timeout
    return Timeout("Request timed out")


@pytest.fixture
def resource_error():
    """Simulate resource exhaustion errors."""
    return MemoryError("Out of memory")


# ============================================================================
# Performance Testing Fixtures
# ============================================================================

@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


@pytest.fixture
def load_test_data():
    """Generate load testing data."""
    def generate_data(num_segments=100, text_length=100):
        segments = []
        for i in range(num_segments):
            segment = {
                'segment_id': f'seg_{i:06d}',
                'text_content': f'This is test segment {i} with content. ' * (text_length // 50),
                'speaker_id': f'speaker_{i % 5}',
                'segment_type': 'narrative' if i % 3 == 0 else 'dialogue',
                'quality_score': 0.7 + (i % 30) / 100,
                'confidence_score': 0.8 + (i % 20) / 100
            }
            segments.append(segment)
        return segments
    
    return generate_data


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test."""
    yield
    # Cleanup happens after test execution
    import gc
    gc.collect()


@pytest.fixture(scope="session", autouse=True)
def test_session_setup():
    """Setup and teardown for the entire test session."""
    logger.info("Starting text-to-audiobook test session")
    
    # Setup session-level resources
    yield
    
    # Cleanup session-level resources  
    logger.info("Ending text-to-audiobook test session")


# ============================================================================
# Test Utilities
# ============================================================================

@pytest.fixture
def assert_helpers():
    """Helper functions for common test assertions."""
    class AssertHelpers:
        @staticmethod
        def assert_dict_contains(actual: Dict, expected: Dict):
            """Assert that actual dict contains all key-value pairs from expected."""
            for key, value in expected.items():
                assert key in actual, f"Key '{key}' not found in actual dict"
                assert actual[key] == value, f"Value mismatch for key '{key}': {actual[key]} != {value}"
        
        @staticmethod
        def assert_list_length(actual: List, expected_length: int):
            """Assert list has expected length."""
            assert len(actual) == expected_length, f"List length mismatch: {len(actual)} != {expected_length}"
        
        @staticmethod
        def assert_float_close(actual: float, expected: float, tolerance: float = 0.01):
            """Assert float values are close within tolerance."""
            assert abs(actual - expected) <= tolerance, f"Float values not close: {actual} vs {expected} (tolerance: {tolerance})"
        
        @staticmethod
        def assert_processing_result_valid(result: Dict):
            """Assert processing result has required fields."""
            required_fields = ['job_id', 'status', 'created_at']
            for field in required_fields:
                assert field in result, f"Required field '{field}' missing from result"
    
    return AssertHelpers()


# ============================================================================
# Mock Factory Functions
# ============================================================================

def create_mock_spark_session():
    """Factory function to create mock Spark session."""
    mock_session = Mock()
    mock_session.sparkContext = Mock()
    mock_session.sparkContext.applicationId = "test_app_001"
    mock_session.sparkContext.appName = "test_application"
    mock_session.conf.set.return_value = None
    return mock_session


def create_mock_dataframe(data: List[Dict]):
    """Factory function to create mock Spark DataFrame."""
    mock_df = Mock()
    mock_df.collect.return_value = [Mock(**row) for row in data]
    mock_df.count.return_value = len(data)
    mock_df.cache.return_value = mock_df
    mock_df.unpersist.return_value = None
    return mock_df