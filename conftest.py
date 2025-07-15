"""
Global pytest configuration and fixtures for text-to-audiobook testing.

This file makes all testing utilities, fixtures, and mock objects available
across the entire test suite.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch
import threading
import time

# Import all testing utilities
from tests.utils.test_fixtures import (
    MockFactory, TestContextManager, get_test_context_manager,
    create_test_segments, create_test_kafka_messages, ResourceMonitor
)
from tests.utils.mock_framework import (
    MockServiceRegistry, get_mock_registry, MockConfiguration, MockBehavior,
    setup_realistic_mocks, setup_fast_mocks, setup_unreliable_mocks
)
from tests.utils.test_data_manager import get_test_data_manager, TestDataGenerator
from tests.performance.test_load_simulation import LoadTestRunner, LoadTestConfiguration


# Global configuration
pytest_plugins = []


def pytest_configure(config):
    """Configure pytest settings and markers."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests across components"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and load tests"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end pipeline tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests (>10 seconds)"
    )
    config.addinivalue_line(
        "markers", "external: Tests requiring external services"
    )
    config.addinivalue_line(
        "markers", "spark: Tests requiring Spark session"
    )
    config.addinivalue_line(
        "markers", "kafka: Tests requiring Kafka services"
    )
    config.addinivalue_line(
        "markers", "redis: Tests requiring Redis services"
    )
    config.addinivalue_line(
        "markers", "llm: Tests requiring LLM services"
    )
    config.addinivalue_line(
        "markers", "distributed: Tests for distributed processing"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers."""
    # Add markers based on test file location
    for item in items:
        test_file = str(item.fspath)
        
        if "/performance/" in test_file:
            item.add_marker(pytest.mark.performance)
        
        if "/integration/" in test_file:
            item.add_marker(pytest.mark.integration)
        
        if "/unit/" in test_file:
            item.add_marker(pytest.mark.unit)
        
        if "test_spark" in test_file or "spark" in test_file:
            item.add_marker(pytest.mark.spark)
        
        if "test_kafka" in test_file or "kafka" in test_file:
            item.add_marker(pytest.mark.kafka)
        
        if "test_redis" in test_file or "redis" in test_file:
            item.add_marker(pytest.mark.redis)
        
        if "test_llm" in test_file or "llm" in test_file:
            item.add_marker(pytest.mark.llm)
        
        if "test_load" in test_file or "load_simulation" in test_file:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session")
def test_session_id():
    """Provide a unique test session ID."""
    import uuid
    return f"test_session_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def test_data_manager():
    """Provide the global test data manager."""
    return get_test_data_manager()


@pytest.fixture(scope="session")
def mock_service_registry():
    """Provide the global mock service registry."""
    registry = get_mock_registry()
    yield registry
    # Cleanup after session
    registry.reset_all_services()


@pytest.fixture(scope="function")
def mock_factory():
    """Provide a fresh mock factory for each test."""
    factory = MockFactory()
    yield factory
    # Cleanup after test
    factory.mock_registry.clear()
    factory.call_history.clear()


@pytest.fixture(scope="function")
def test_context(request):
    """Provide a test execution context with automatic cleanup."""
    manager = get_test_context_manager()
    test_name = request.node.name
    context = manager.create_test_context(test_name)
    
    yield context
    
    # Cleanup
    manager.cleanup_test_context(context)


@pytest.fixture(scope="function")
def temp_workspace(test_context):
    """Provide a temporary workspace directory."""
    return test_context.temp_directory


@pytest.fixture(scope="function")
def resource_monitor():
    """Provide a resource monitor for performance testing."""
    monitor = ResourceMonitor()
    yield monitor
    if monitor.monitoring:
        monitor.stop_monitoring()


# Mock Infrastructure Fixtures
@pytest.fixture(scope="function")
def mock_kafka_infrastructure(mock_factory):
    """Provide complete mock Kafka infrastructure."""
    from tests.utils.mock_framework import MockConfiguration, MockBehavior
    config = MockConfiguration(behavior=MockBehavior.FAST, response_delay_range=(0.001, 0.002))
    
    producer = mock_factory.create_kafka_producer(config)
    consumer = mock_factory.create_kafka_consumer(config)
    
    return {
        'producer': producer,
        'consumer': consumer,
        'factory': mock_factory
    }


@pytest.fixture(scope="function")
def mock_spark_session(mock_factory):
    """Provide a mock Spark session."""
    from tests.utils.mock_framework import MockConfiguration, MockBehavior
    config = MockConfiguration(behavior=MockBehavior.FAST, response_delay_range=(0.01, 0.02))
    return mock_factory.create_spark_session(config)


@pytest.fixture(scope="function")
def mock_redis_client(mock_factory):
    """Provide a mock Redis client."""
    from tests.utils.mock_framework import MockConfiguration, MockBehavior
    config = MockConfiguration(behavior=MockBehavior.FAST, response_delay_range=(0.0001, 0.0002))
    return mock_factory.create_redis_client(config)


@pytest.fixture(scope="function")
def mock_llm_pool(mock_factory):
    """Provide a mock LLM pool manager."""
    from tests.utils.mock_framework import MockConfiguration, MockBehavior
    config = MockConfiguration(behavior=MockBehavior.FAST, response_delay_range=(0.01, 0.05))
    return mock_factory.create_llm_pool_manager(config)


@pytest.fixture(scope="function")
def mock_monitoring_infrastructure(mock_factory):
    """Provide complete mock monitoring infrastructure."""
    from tests.utils.mock_framework import MockConfiguration, MockBehavior
    metrics_config = MockConfiguration(behavior=MockBehavior.FAST, response_delay_range=(0.0001, 0.0001))
    health_config = MockConfiguration(behavior=MockBehavior.FAST, response_delay_range=(0.001, 0.002))
    
    metrics = mock_factory.create_metrics_collector(metrics_config)
    health = mock_factory.create_health_service(health_config)
    
    return {
        'metrics': metrics,
        'health': health,
        'factory': mock_factory
    }


# External Service Mock Fixtures
@pytest.fixture(scope="function")
def mock_llm_service(mock_service_registry):
    """Provide a mock LLM service with realistic behavior."""
    from tests.utils.mock_framework import MockConfiguration, MockBehavior
    config = MockConfiguration(behavior=MockBehavior.FAST, response_delay_range=(0.05, 0.1))
    return mock_service_registry.create_llm_service(config)


@pytest.fixture(scope="function")
def mock_tts_service(mock_service_registry):
    """Provide a mock TTS service with realistic behavior."""
    from tests.utils.mock_framework import MockConfiguration, MockBehavior
    config = MockConfiguration(behavior=MockBehavior.FAST, response_delay_range=(0.1, 0.3))
    return mock_service_registry.create_tts_service(config)


@pytest.fixture(scope="function")
def mock_cloud_storage(mock_service_registry):
    """Provide a mock cloud storage service."""
    from tests.utils.mock_framework import MockConfiguration, MockBehavior
    config = MockConfiguration(behavior=MockBehavior.FAST, response_delay_range=(0.01, 0.05))
    return mock_service_registry.create_storage_service(config)


# Test Data Fixtures
@pytest.fixture(scope="function")
def sample_test_data():
    """Provide sample test data for various scenarios."""
    generator = TestDataGenerator(seed=42)
    
    # Generate different types of test content
    simple_dialogue, _ = generator.generate_book_content(
        chapter_count=1,
        target_word_count=500,
        dialogue_ratio=0.8,
        speaker_count=2
    )
    
    complex_narrative, _ = generator.generate_book_content(
        chapter_count=3,
        target_word_count=2000,
        dialogue_ratio=0.3,
        speaker_count=5
    )
    
    mixed_content, _ = generator.generate_book_content(
        chapter_count=2,
        target_word_count=1000,
        dialogue_ratio=0.5,
        speaker_count=3
    )
    
    return {
        'simple_dialogue': simple_dialogue,
        'complex_narrative': complex_narrative,
        'mixed_content': mixed_content,
        'generator': generator
    }


@pytest.fixture(scope="function")
def test_segments():
    """Provide test segments for component testing."""
    return create_test_segments(count=10, content_type='mixed')


@pytest.fixture(scope="function")
def test_kafka_messages(test_segments):
    """Provide test Kafka messages from segments."""
    return create_test_kafka_messages(test_segments, job_id='test_job_001')


# Performance Testing Fixtures
@pytest.fixture(scope="function")
def load_test_runner():
    """Provide a load test runner."""
    return LoadTestRunner()


@pytest.fixture(scope="function")
def performance_test_config():
    """Provide configuration for performance testing."""
    return {
        'load_levels': ['light', 'moderate'],  # Reduced for faster testing
        'concurrent_users': [1, 5, 10],
        'data_sizes': [100, 500, 1000],  # KB
        'duration_seconds': [10, 30],  # Reduced for faster testing
        'thresholds': {
            'max_response_time': 5.0,
            'min_success_rate': 0.95,
            'max_memory_growth_mb': 500,
            'max_cpu_usage_percent': 80
        }
    }


@pytest.fixture(scope="function")
def integration_test_config():
    """Provide configuration for integration testing."""
    return {
        'components': ['kafka', 'spark', 'redis', 'llm_pool', 'monitoring'],
        'test_scenarios': ['happy_path', 'error_handling'],  # Reduced for faster testing
        'timeouts': {
            'component_startup': 10,  # Reduced for mocked components
            'test_execution': 60,     # Reduced for faster testing
            'cleanup': 10            # Reduced for faster testing
        },
        'retry_config': {
            'max_retries': 2,        # Reduced for faster testing
            'backoff_factor': 1.5,   # Reduced for faster testing
            'initial_delay': 0.5     # Reduced for faster testing
        }
    }


@pytest.fixture(scope="function")
def error_simulation_config():
    """Provide configuration for error simulation testing."""
    return {
        'failure_modes': [
            {'component': 'kafka', 'failure_rate': 0.1, 'error': 'Connection timeout'},
            {'component': 'spark', 'failure_rate': 0.05, 'error': 'Out of memory'},
            {'component': 'redis', 'failure_rate': 0.02, 'error': 'Connection refused'},
            {'component': 'llm', 'failure_rate': 0.1, 'error': 'API rate limit'}  # Reduced for faster testing
        ],
        'recovery_scenarios': [
            'immediate_retry',
            'exponential_backoff'  # Reduced set for faster testing
        ]
    }


# Environment-specific fixtures
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment with proper isolation."""
    # Ensure test isolation
    import os
    os.environ['TESTING'] = 'true'
    os.environ['LOG_LEVEL'] = 'WARNING'  # Reduce log noise during testing
    
    yield
    
    # Cleanup
    if 'TESTING' in os.environ:
        del os.environ['TESTING']


@pytest.fixture(scope="function")
def isolated_env(monkeypatch, temp_workspace):
    """Provide an isolated environment for testing."""
    # Set up isolated paths
    monkeypatch.setenv('TEMP_DIR', str(temp_workspace))
    monkeypatch.setenv('OUTPUT_DIR', str(temp_workspace / 'output'))
    monkeypatch.setenv('CACHE_DIR', str(temp_workspace / 'cache'))
    
    # Create necessary directories
    (temp_workspace / 'output').mkdir(exist_ok=True)
    (temp_workspace / 'cache').mkdir(exist_ok=True)
    (temp_workspace / 'logs').mkdir(exist_ok=True)
    
    return temp_workspace


# Cleanup fixtures
@pytest.fixture(scope="function", autouse=True)
def cleanup_after_test():
    """Automatic cleanup after each test."""
    yield
    
    # Force garbage collection to clean up resources
    import gc
    gc.collect()
    
    # Reset any global state in modules
    try:
        from tests.utils.mock_framework import get_mock_registry
        registry = get_mock_registry()
        registry.reset_all_services()
    except ImportError:
        pass


# Parametrized fixtures for different testing scenarios
@pytest.fixture(params=['fast', 'realistic'])
def mock_behavior_mode(request):
    """Parametrized fixture for different mock behavior modes."""
    from tests.utils.mock_framework import MockBehavior
    
    behavior_map = {
        'fast': MockBehavior.FAST,
        'realistic': MockBehavior.REALISTIC
    }
    
    return behavior_map[request.param]


@pytest.fixture(params=[1, 5, 10])
def concurrent_user_count(request):
    """Parametrized fixture for different concurrency levels."""
    return request.param


@pytest.fixture(params=[100, 500, 1000])
def data_size_kb(request):
    """Parametrized fixture for different data sizes."""
    return request.param


# Skip markers for optional dependencies
def pytest_runtest_setup(item):
    """Setup hook to skip tests based on missing dependencies."""
    # Skip Spark tests if PySpark is not available
    if item.get_closest_marker("spark"):
        try:
            import pyspark
        except ImportError:
            pytest.skip("PySpark not available")
    
    # Skip Kafka tests if kafka-python is not available
    if item.get_closest_marker("kafka"):
        try:
            import kafka
        except ImportError:
            pytest.skip("kafka-python not available")
    
    # Skip Redis tests if redis-py is not available
    if item.get_closest_marker("redis"):
        try:
            import redis
        except ImportError:
            pytest.skip("redis-py not available")
    
    # Skip external tests if marked to skip external dependencies
    if item.get_closest_marker("external"):
        if item.config.getoption("--no-external", default=False):
            pytest.skip("External dependencies disabled")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--no-external",
        action="store_true",
        default=False,
        help="Skip tests that require external services"
    )
    parser.addoption(
        "--load-test",
        action="store_true",
        default=False,
        help="Run load and performance tests"
    )
    parser.addoption(
        "--integration-only",
        action="store_true",
        default=False,
        help="Run only integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    if config.getoption("--integration-only"):
        # Only run integration tests
        skip_non_integration = pytest.mark.skip(reason="--integration-only specified")
        for item in items:
            if not item.get_closest_marker("integration"):
                item.add_marker(skip_non_integration)
    
    if not config.getoption("--load-test"):
        # Skip load tests by default
        skip_load = pytest.mark.skip(reason="Load tests skipped (use --load-test to enable)")
        for item in items:
            if item.get_closest_marker("performance") and "load" in str(item.fspath):
                item.add_marker(skip_load)