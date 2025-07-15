"""
Configuration file for integration tests.

This module provides pytest fixtures and configuration for integration testing
of the distributed text-to-audiobook pipeline.
"""

import pytest
import os
import tempfile
import shutil
import json
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

# Test environment setup
@pytest.fixture(scope='session')
def test_environment():
    """Set up test environment for the entire test session."""
    # Create test directories
    test_dirs = [
        '/tmp/test_input',
        '/tmp/test_output',
        '/tmp/test_logs'
    ]
    
    for test_dir in test_dirs:
        os.makedirs(test_dir, exist_ok=True)
    
    # Set environment variables for testing
    test_env = {
        'AIRFLOW_HOME': '/tmp/test_airflow',
        'SPARK_HOME': '/tmp/test_spark',
        'KAFKA_HOME': '/tmp/test_kafka',
        'LLM_ENGINE': 'local',
        'KAFKA_ENABLED': 'true',
        'SPARK_ENVIRONMENT': 'local',
        'LOG_LEVEL': 'DEBUG',
        'TESTING': 'true'
    }
    
    with patch.dict(os.environ, test_env):
        yield test_env
    
    # Cleanup
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


@pytest.fixture
def sample_text_content():
    """Provide sample text content for testing."""
    return """
    Chapter 1: The Beginning
    
    John: "Hello there, Mary. How are you doing today?"
    Mary: "I'm doing well, thank you for asking. How about you?"
    John: "I'm great! Just finished reading an amazing book about AI."
    
    The conversation continued as they walked through the park.
    
    Mary: "Oh really? What book was it?"
    John: "It was called 'The Future of Machine Learning' by Dr. Smith."
    Mary: "That sounds fascinating! I'd love to read it too."
    
    Narrator: As they spoke, the sun began to set behind the trees.
    
    John: "I can lend it to you if you'd like."
    Mary: "That would be wonderful, thank you!"
    """


@pytest.fixture
def sample_structured_segments():
    """Provide sample structured segments for testing."""
    return [
        {
            'speaker': 'John',
            'text': 'Hello there, Mary. How are you doing today?',
            'line_id': 1,
            'confidence': 0.95
        },
        {
            'speaker': 'Mary',
            'text': 'I\'m doing well, thank you for asking. How about you?',
            'line_id': 2,
            'confidence': 0.92
        },
        {
            'speaker': 'John',
            'text': 'I\'m great! Just finished reading an amazing book about AI.',
            'line_id': 3,
            'confidence': 0.89
        },
        {
            'speaker': 'Narrator',
            'text': 'The conversation continued as they walked through the park.',
            'line_id': 4,
            'confidence': 0.98
        },
        {
            'speaker': 'Mary',
            'text': 'Oh really? What book was it?',
            'line_id': 5,
            'confidence': 0.91
        },
        {
            'speaker': 'John',
            'text': 'It was called \'The Future of Machine Learning\' by Dr. Smith.',
            'line_id': 6,
            'confidence': 0.87
        },
        {
            'speaker': 'Mary',
            'text': 'That sounds fascinating! I\'d love to read it too.',
            'line_id': 7,
            'confidence': 0.94
        },
        {
            'speaker': 'Narrator',
            'text': 'As they spoke, the sun began to set behind the trees.',
            'line_id': 8,
            'confidence': 0.96
        },
        {
            'speaker': 'John',
            'text': 'I can lend it to you if you\'d like.',
            'line_id': 9,
            'confidence': 0.93
        },
        {
            'speaker': 'Mary',
            'text': 'That would be wonderful, thank you!',
            'line_id': 10,
            'confidence': 0.95
        }
    ]


@pytest.fixture
def test_configuration():
    """Provide test configuration."""
    return {
        'spark': {
            'environment': 'local',
            'app_name': 'test_text_to_audiobook',
            'executor_memory': '1g',
            'executor_cores': 1,
            'driver_memory': '512m'
        },
        'kafka': {
            'enabled': True,
            'bootstrap_servers': 'localhost:9092',
            'topics': {
                'file_upload': 'test_file_upload',
                'text_extracted': 'test_text_extracted',
                'chunk_processing': 'test_chunk_processing',
                'llm_request': 'test_llm_request',
                'llm_result': 'test_llm_result',
                'status_update': 'test_status_update',
                'error': 'test_error'
            }
        },
        'llm': {
            'engine': 'local',
            'model': 'mistral',
            'timeout': 30,
            'max_retries': 3
        },
        'processing': {
            'chunk_size': 1000,
            'overlap_size': 200,
            'max_refinement_iterations': 2,
            'enable_contextual_refinement': True
        },
        'quality': {
            'thresholds': {
                'overall_quality': 85.0,
                'speaker_consistency': 80.0,
                'attribution_confidence': 75.0,
                'error_rate': 10.0
            }
        },
        'output': {
            'format': 'json',
            'directory': '/tmp/test_output',
            'include_metadata': True
        }
    }


@pytest.fixture
def mock_spark_session():
    """Provide a mock Spark session for testing."""
    with patch('pyspark.sql.SparkSession') as mock_spark:
        mock_session = Mock()
        mock_spark.builder.config.return_value.getOrCreate.return_value = mock_session
        
        # Mock DataFrame operations
        mock_df = Mock()
        mock_session.createDataFrame.return_value = mock_df
        mock_df.cache.return_value = mock_df
        mock_df.select.return_value = mock_df
        mock_df.collect.return_value = []
        
        # Mock Spark context
        mock_sc = Mock()
        mock_session.sparkContext = mock_sc
        mock_sc.broadcast.return_value = Mock()
        
        yield mock_session


@pytest.fixture
def mock_kafka_producer():
    """Provide a mock Kafka producer for testing."""
    with patch('kafka.KafkaProducer') as mock_producer:
        mock_producer_instance = Mock()
        mock_producer.return_value = mock_producer_instance
        
        # Mock producer methods
        mock_future = Mock()
        mock_metadata = Mock()
        mock_metadata.partition = 0
        mock_metadata.offset = 123
        mock_future.get.return_value = mock_metadata
        mock_producer_instance.send.return_value = mock_future
        
        mock_producer_instance.metrics.return_value = {
            'record-send-rate': 10.0,
            'record-send-total': 100,
            'record-error-rate': 0.1
        }
        
        yield mock_producer_instance


@pytest.fixture
def mock_kafka_consumer():
    """Provide a mock Kafka consumer for testing."""
    with patch('kafka.KafkaConsumer') as mock_consumer:
        mock_consumer_instance = Mock()
        mock_consumer.return_value = mock_consumer_instance
        
        # Mock consumer methods
        mock_consumer_instance.poll.return_value = {}
        mock_consumer_instance.commit.return_value = None
        mock_consumer_instance.close.return_value = None
        
        yield mock_consumer_instance


@pytest.fixture
def mock_llm_client():
    """Provide a mock LLM client for testing."""
    with patch('src.llm_pool.llm_client.LLMClient') as mock_client:
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        # Mock LLM methods
        mock_client_instance.classify_text.return_value = "John"
        mock_client_instance.process_chunk.return_value = {
            'success': True,
            'classifications': ['John', 'Mary', 'Narrator']
        }
        mock_client_instance.health_check.return_value = {
            'overall_health': 'healthy',
            'response_time': 0.5
        }
        
        yield mock_client_instance


@pytest.fixture
def mock_validator():
    """Provide a mock validator for testing."""
    with patch('src.validation.validator.SimplifiedValidator') as mock_validator:
        mock_validator_instance = Mock()
        mock_validator.return_value = mock_validator_instance
        
        # Mock validator methods
        mock_validator_instance.validate_structured_segments.return_value = {
            'quality_score': 95.0,
            'errors': [],
            'warnings': [],
            'speaker_consistency': 92.0,
            'attribution_confidence': 88.0,
            'error_rate': 2.0
        }
        
        yield mock_validator_instance


@pytest.fixture
def mock_text_extractor():
    """Provide a mock text extractor for testing."""
    with patch('src.text_processing.text_extractor.TextExtractor') as mock_extractor:
        mock_extractor_instance = Mock()
        mock_extractor.return_value = mock_extractor_instance
        
        # Mock extractor methods
        def mock_extract(file_path):
            if 'test' in file_path:
                return "This is extracted test content."
            else:
                raise FileNotFoundError(f"File not found: {file_path}")
        
        mock_extractor_instance.extract.side_effect = mock_extract
        
        yield mock_extractor_instance


@pytest.fixture
def test_file_path(tmp_path, sample_text_content):
    """Create a temporary test file with sample content."""
    test_file = tmp_path / "test_book.txt"
    test_file.write_text(sample_text_content)
    return str(test_file)


@pytest.fixture
def airflow_test_context():
    """Provide a mock Airflow context for testing."""
    mock_dag_run = Mock()
    mock_dag_run.run_id = 'test_run_123'
    mock_dag_run.conf = {'file_path': '/tmp/test_file.txt'}
    
    mock_task_instance = Mock()
    mock_task_instance.xcom_push = Mock()
    mock_task_instance.xcom_pull = Mock()
    
    return {
        'dag_run': mock_dag_run,
        'task_instance': mock_task_instance,
        'execution_date': '2023-01-01T00:00:00',
        'task_id': 'test_task'
    }


@pytest.fixture
def integration_test_data():
    """Provide comprehensive test data for integration tests."""
    return {
        'books': [
            {
                'title': 'Test Book 1',
                'content': 'John: "Hello world!" Mary: "Hi there!"',
                'expected_speakers': ['John', 'Mary'],
                'expected_segments': 2
            },
            {
                'title': 'Test Book 2',
                'content': 'Narrator: It was a dark and stormy night. Character: "I must go!"',
                'expected_speakers': ['Narrator', 'Character'],
                'expected_segments': 2
            }
        ],
        'processing_configs': [
            {
                'name': 'fast_processing',
                'chunk_size': 500,
                'overlap_size': 100,
                'max_refinement_iterations': 1
            },
            {
                'name': 'quality_processing',
                'chunk_size': 2000,
                'overlap_size': 400,
                'max_refinement_iterations': 3
            }
        ],
        'quality_thresholds': [
            {
                'name': 'strict',
                'overall_quality': 95.0,
                'speaker_consistency': 90.0,
                'attribution_confidence': 85.0,
                'error_rate': 5.0
            },
            {
                'name': 'relaxed',
                'overall_quality': 80.0,
                'speaker_consistency': 75.0,
                'attribution_confidence': 70.0,
                'error_rate': 15.0
            }
        ]
    }


# Performance testing fixtures
@pytest.fixture
def performance_test_data():
    """Provide data for performance testing."""
    return {
        'small_text': "John: Hello! Mary: Hi there!" * 10,
        'medium_text': "John: Hello! Mary: Hi there!" * 100,
        'large_text': "John: Hello! Mary: Hi there!" * 1000,
        'performance_thresholds': {
            'small_text_processing_time': 1.0,  # seconds
            'medium_text_processing_time': 5.0,  # seconds
            'large_text_processing_time': 30.0,  # seconds
            'memory_usage_limit': 1024 * 1024 * 1024,  # 1GB
            'cpu_usage_limit': 80.0  # percentage
        }
    }


# Custom markers for test categorization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmark tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running tests"
    )
    config.addinivalue_line(
        "markers", "external: marks tests that require external services"
    )


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add integration marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add performance marker to performance tests
        if "performance" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        
        # Add slow marker to tests that might be slow
        if any(keyword in item.name for keyword in ["end_to_end", "benchmark", "large"]):
            item.add_marker(pytest.mark.slow)


# Test reporting hooks
@pytest.fixture(autouse=True)
def test_logging():
    """Set up test logging."""
    import logging
    
    # Configure logging for tests
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress noisy loggers
    logging.getLogger('kafka').setLevel(logging.WARNING)
    logging.getLogger('pyspark').setLevel(logging.WARNING)
    logging.getLogger('py4j').setLevel(logging.WARNING)


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Clean up test files after each test."""
    yield
    
    # Clean up any temporary files created during testing
    test_patterns = [
        '/tmp/test_*',
        '/tmp/airflow_*',
        '/tmp/spark_*',
        '/tmp/kafka_*'
    ]
    
    import glob
    for pattern in test_patterns:
        for file_path in glob.glob(pattern):
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Warning: Could not clean up {file_path}: {e}")


# Database fixtures for Airflow testing
@pytest.fixture
def airflow_db():
    """Set up Airflow database for testing."""
    from airflow.utils.db import create_tables, resetdb
    
    # Reset and create tables
    resetdb()
    create_tables()
    
    yield
    
    # Cleanup would go here if needed