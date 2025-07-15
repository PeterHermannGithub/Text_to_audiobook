"""
Comprehensive test fixtures and utilities for component testing.

Provides reusable fixtures, mock objects, and testing utilities for all
components in the text-to-audiobook distributed pipeline.
"""

import pytest
import json
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Generator
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor

from tests.utils.test_data_manager import TestDataGenerator, get_test_data_manager


@dataclass
class MockConfiguration:
    """Configuration for mock objects."""
    response_delay: float = 0.0
    failure_rate: float = 0.0
    failure_exception: Exception = Exception("Mock failure")
    enable_side_effects: bool = True
    call_tracking: bool = True


@dataclass
class TestContext:
    """Test execution context with shared resources."""
    test_id: str
    temp_directory: Path
    mock_registry: Dict[str, Mock]
    cleanup_callbacks: List[Callable]
    start_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MockFactory:
    """Factory for creating consistent mock objects across tests."""
    
    def __init__(self):
        self.mock_registry: Dict[str, Mock] = {}
        self.call_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def create_kafka_producer(self, config: MockConfiguration = None) -> Mock:
        """Create a mock Kafka producer with realistic behavior."""
        config = config or MockConfiguration()
        
        mock_producer = Mock()
        
        def mock_send(topic, value=None, key=None, partition=None, **kwargs):
            if config.response_delay > 0:
                time.sleep(config.response_delay)
            
            if config.failure_rate > 0 and self._should_fail(config.failure_rate):
                raise config.failure_exception
            
            # Create mock future
            mock_future = Mock()
            mock_future.get.return_value = Mock()
            
            if config.call_tracking:
                self._track_call('kafka_producer_send', {
                    'topic': topic,
                    'key': key,
                    'value_size': len(str(value)) if value else 0,
                    'timestamp': time.time()
                })
            
            return mock_future
        
        mock_producer.send.side_effect = mock_send
        mock_producer.flush.return_value = None
        mock_producer.close.return_value = None
        
        self.mock_registry['kafka_producer'] = mock_producer
        return mock_producer
    
    def create_kafka_consumer(self, config: MockConfiguration = None) -> Mock:
        """Create a mock Kafka consumer with realistic behavior."""
        config = config or MockConfiguration()
        
        mock_consumer = Mock()
        
        def mock_poll(timeout_ms=1000, max_records=500):
            if config.response_delay > 0:
                time.sleep(config.response_delay)
            
            if config.failure_rate > 0 and self._should_fail(config.failure_rate):
                raise config.failure_exception
            
            if config.call_tracking:
                self._track_call('kafka_consumer_poll', {
                    'timeout_ms': timeout_ms,
                    'max_records': max_records,
                    'timestamp': time.time()
                })
            
            # Return empty poll result by default
            return {}
        
        mock_consumer.poll.side_effect = mock_poll
        mock_consumer.subscribe.return_value = None
        mock_consumer.commit.return_value = None
        mock_consumer.close.return_value = None
        mock_consumer.assignment.return_value = []
        
        self.mock_registry['kafka_consumer'] = mock_consumer
        return mock_consumer
    
    def create_spark_session(self, config: MockConfiguration = None) -> Mock:
        """Create a mock Spark session with realistic behavior."""
        config = config or MockConfiguration()
        
        mock_spark = Mock()
        mock_context = Mock()
        
        # Spark context setup
        mock_context.applicationId = f"test_app_{uuid.uuid4().hex[:8]}"
        mock_context.defaultParallelism = 4
        mock_context.version = "3.5.0"
        mock_spark.sparkContext = mock_context
        
        def mock_create_dataframe(data, schema=None):
            if config.response_delay > 0:
                time.sleep(config.response_delay)
            
            if config.failure_rate > 0 and self._should_fail(config.failure_rate):
                raise config.failure_exception
            
            # Create mock DataFrame
            mock_df = self.create_spark_dataframe(config)
            
            if config.call_tracking:
                self._track_call('spark_create_dataframe', {
                    'data_size': len(data) if data else 0,
                    'schema': str(schema) if schema else None,
                    'timestamp': time.time()
                })
            
            return mock_df
        
        mock_spark.createDataFrame.side_effect = mock_create_dataframe
        mock_spark.conf.set.return_value = None
        mock_spark.stop.return_value = None
        mock_spark.udf.register.return_value = None
        
        self.mock_registry['spark_session'] = mock_spark
        return mock_spark
    
    def create_spark_dataframe(self, config: MockConfiguration = None) -> Mock:
        """Create a mock Spark DataFrame with realistic operations."""
        config = config or MockConfiguration()
        
        mock_df = Mock()
        
        def mock_collect():
            if config.response_delay > 0:
                time.sleep(config.response_delay)
            
            if config.failure_rate > 0 and self._should_fail(config.failure_rate):
                raise config.failure_exception
            
            if config.call_tracking:
                self._track_call('spark_dataframe_collect', {
                    'timestamp': time.time()
                })
            
            return []
        
        def mock_count():
            if config.response_delay > 0:
                time.sleep(config.response_delay / 2)  # Count is typically faster
            
            if config.failure_rate > 0 and self._should_fail(config.failure_rate):
                raise config.failure_exception
            
            return 100  # Default count
        
        mock_df.collect.side_effect = mock_collect
        mock_df.count.side_effect = mock_count
        mock_df.cache.return_value = mock_df
        mock_df.unpersist.return_value = None
        mock_df.filter.return_value = mock_df
        mock_df.select.return_value = mock_df
        mock_df.withColumn.return_value = mock_df
        mock_df.groupBy.return_value = mock_df
        mock_df.agg.return_value = mock_df
        
        return mock_df
    
    def create_redis_client(self, config: MockConfiguration = None) -> Mock:
        """Create a mock Redis client with realistic behavior."""
        config = config or MockConfiguration()
        
        mock_redis = Mock()
        
        def mock_get(key):
            if config.response_delay > 0:
                time.sleep(config.response_delay)
            
            if config.failure_rate > 0 and self._should_fail(config.failure_rate):
                raise config.failure_exception
            
            if config.call_tracking:
                self._track_call('redis_get', {
                    'key': key,
                    'timestamp': time.time()
                })
            
            return None  # Cache miss by default
        
        def mock_setex(key, ttl, value):
            if config.response_delay > 0:
                time.sleep(config.response_delay)
            
            if config.failure_rate > 0 and self._should_fail(config.failure_rate):
                raise config.failure_exception
            
            if config.call_tracking:
                self._track_call('redis_setex', {
                    'key': key,
                    'ttl': ttl,
                    'value_size': len(str(value)),
                    'timestamp': time.time()
                })
            
            return True
        
        mock_redis.get.side_effect = mock_get
        mock_redis.setex.side_effect = mock_setex
        mock_redis.delete.return_value = 1
        mock_redis.ping.return_value = True
        mock_redis.flushdb.return_value = True
        mock_redis.keys.return_value = []
        
        self.mock_registry['redis_client'] = mock_redis
        return mock_redis
    
    def create_llm_pool_manager(self, config: MockConfiguration = None) -> Mock:
        """Create a mock LLM pool manager with realistic behavior."""
        config = config or MockConfiguration()
        
        mock_pool = Mock()
        
        def mock_process_segment(segment):
            if config.response_delay > 0:
                time.sleep(config.response_delay)
            
            if config.failure_rate > 0 and self._should_fail(config.failure_rate):
                raise config.failure_exception
            
            # Generate realistic response
            result = {
                'segment_id': segment.get('segment_id', 'unknown'),
                'speaker': self._generate_speaker_assignment(),
                'confidence': 0.85 + (hash(str(segment)) % 100) / 1000,  # 0.85-0.95
                'processing_time': config.response_delay or 0.1
            }
            
            if config.call_tracking:
                self._track_call('llm_process_segment', {
                    'segment_id': result['segment_id'],
                    'confidence': result['confidence'],
                    'timestamp': time.time()
                })
            
            return result
        
        mock_pool.process_segment.side_effect = mock_process_segment
        mock_pool.get_pool_stats.return_value = {
            'active_instances': 2,
            'queue_size': 0,
            'total_requests': 0,
            'avg_response_time': config.response_delay or 0.1
        }
        mock_pool.shutdown.return_value = None
        
        self.mock_registry['llm_pool'] = mock_pool
        return mock_pool
    
    def create_metrics_collector(self, config: MockConfiguration = None) -> Mock:
        """Create a mock metrics collector."""
        config = config or MockConfiguration()
        
        mock_metrics = Mock()
        
        def mock_record_metric(*args, **kwargs):
            if config.response_delay > 0:
                time.sleep(config.response_delay)
            
            if config.call_tracking:
                self._track_call('metrics_record', {
                    'args': args,
                    'kwargs': kwargs,
                    'timestamp': time.time()
                })
        
        mock_metrics.record_processing_request.side_effect = mock_record_metric
        mock_metrics.record_processing_duration.side_effect = mock_record_metric
        mock_metrics.record_spark_job.side_effect = mock_record_metric
        mock_metrics.record_kafka_message.side_effect = mock_record_metric
        mock_metrics.record_llm_request.side_effect = mock_record_metric
        mock_metrics.set_system_health.side_effect = mock_record_metric
        
        self.mock_registry['metrics_collector'] = mock_metrics
        return mock_metrics
    
    def create_health_service(self, config: MockConfiguration = None) -> Mock:
        """Create a mock health service."""
        config = config or MockConfiguration()
        
        mock_health = Mock()
        
        def mock_check_component(component):
            if config.response_delay > 0:
                time.sleep(config.response_delay)
            
            if config.failure_rate > 0 and self._should_fail(config.failure_rate):
                return {
                    'status': 'unhealthy',
                    'component': component,
                    'error': str(config.failure_exception),
                    'timestamp': time.time()
                }
            
            return {
                'status': 'healthy',
                'component': component,
                'latency_ms': (config.response_delay or 0.01) * 1000,
                'timestamp': time.time()
            }
        
        mock_health.check_component_health.side_effect = mock_check_component
        mock_health.get_overall_health.return_value = {
            'status': 'healthy',
            'message': 'All components operational',
            'summary': {'healthy': 5, 'degraded': 0, 'unhealthy': 0, 'total': 5},
            'timestamp': time.time()
        }
        
        self.mock_registry['health_service'] = mock_health
        return mock_health
    
    def _should_fail(self, failure_rate: float) -> bool:
        """Determine if a mock should fail based on failure rate."""
        import random
        return random.random() < failure_rate
    
    def _generate_speaker_assignment(self) -> str:
        """Generate a realistic speaker assignment."""
        speakers = ['narrator', 'alice', 'bob', 'charlie', 'diana']
        import random
        return random.choice(speakers)
    
    def _track_call(self, operation: str, data: Dict[str, Any]):
        """Track method calls for analysis."""
        if operation not in self.call_history:
            self.call_history[operation] = []
        self.call_history[operation].append(data)
    
    def get_call_history(self, operation: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get call history for analysis."""
        if operation:
            return {operation: self.call_history.get(operation, [])}
        return self.call_history.copy()
    
    def reset_call_history(self):
        """Reset call tracking history."""
        self.call_history.clear()
    
    def get_mock(self, mock_name: str) -> Optional[Mock]:
        """Get a registered mock by name."""
        return self.mock_registry.get(mock_name)


class TestContextManager:
    """Manages test execution context and cleanup."""
    
    def __init__(self):
        self.active_contexts: Dict[str, TestContext] = {}
        self.mock_factory = MockFactory()
    
    def create_test_context(self, test_name: str, **metadata) -> TestContext:
        """Create a new test execution context."""
        test_id = f"{test_name}_{uuid.uuid4().hex[:8]}"
        temp_dir = Path(tempfile.mkdtemp(prefix=f"test_{test_name}_"))
        
        context = TestContext(
            test_id=test_id,
            temp_directory=temp_dir,
            mock_registry={},
            cleanup_callbacks=[],
            start_time=time.time(),
            metadata=metadata
        )
        
        self.active_contexts[test_id] = context
        return context
    
    def cleanup_test_context(self, context: TestContext):
        """Clean up test context resources."""
        try:
            # Execute cleanup callbacks
            for callback in context.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    print(f"Cleanup callback failed: {e}")
            
            # Remove temporary directory
            if context.temp_directory.exists():
                shutil.rmtree(context.temp_directory, ignore_errors=True)
            
            # Remove from active contexts
            if context.test_id in self.active_contexts:
                del self.active_contexts[context.test_id]
                
        except Exception as e:
            print(f"Context cleanup failed: {e}")
    
    def cleanup_all_contexts(self):
        """Clean up all active test contexts."""
        for context in list(self.active_contexts.values()):
            self.cleanup_test_context(context)


# Global test context manager
_test_context_manager = TestContextManager()


def get_test_context_manager() -> TestContextManager:
    """Get the global test context manager."""
    return _test_context_manager


# Pytest fixtures
@pytest.fixture
def mock_factory():
    """Provide a mock factory for tests."""
    factory = MockFactory()
    yield factory
    factory.mock_registry.clear()
    factory.call_history.clear()


@pytest.fixture
def test_context(request):
    """Provide a test execution context with automatic cleanup."""
    manager = get_test_context_manager()
    test_name = request.node.name
    context = manager.create_test_context(test_name)
    
    yield context
    
    manager.cleanup_test_context(context)


@pytest.fixture
def temp_workspace(test_context):
    """Provide a temporary workspace directory."""
    return test_context.temp_directory


@pytest.fixture
def mock_kafka_infrastructure(mock_factory):
    """Provide complete mock Kafka infrastructure."""
    config = MockConfiguration(response_delay=0.001)  # 1ms latency
    
    producer = mock_factory.create_kafka_producer(config)
    consumer = mock_factory.create_kafka_consumer(config)
    
    return {
        'producer': producer,
        'consumer': consumer,
        'factory': mock_factory
    }


@pytest.fixture
def mock_spark_session(mock_factory):
    """Provide a mock Spark session."""
    config = MockConfiguration(response_delay=0.01)  # 10ms latency
    return mock_factory.create_spark_session(config)


@pytest.fixture
def mock_redis_client(mock_factory):
    """Provide a mock Redis client."""
    config = MockConfiguration(response_delay=0.0001)  # 0.1ms latency
    return mock_factory.create_redis_client(config)


@pytest.fixture
def mock_llm_pool(mock_factory):
    """Provide a mock LLM pool manager."""
    config = MockConfiguration(response_delay=0.1)  # 100ms processing
    return mock_factory.create_llm_pool_manager(config)


@pytest.fixture
def mock_monitoring_infrastructure(mock_factory):
    """Provide complete mock monitoring infrastructure."""
    metrics_config = MockConfiguration(response_delay=0.0001)
    health_config = MockConfiguration(response_delay=0.01)
    
    metrics = mock_factory.create_metrics_collector(metrics_config)
    health = mock_factory.create_health_service(health_config)
    
    return {
        'metrics': metrics,
        'health': health,
        'factory': mock_factory
    }


@pytest.fixture
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


@pytest.fixture
def performance_test_config():
    """Provide configuration for performance testing."""
    return {
        'load_levels': ['light', 'moderate', 'heavy'],
        'concurrent_users': [1, 5, 10, 20],
        'data_sizes': [100, 500, 1000, 5000],  # KB
        'duration_seconds': [30, 60, 120],
        'thresholds': {
            'max_response_time': 5.0,
            'min_success_rate': 0.95,
            'max_memory_growth_mb': 500,
            'max_cpu_usage_percent': 80
        }
    }


@pytest.fixture
def integration_test_config():
    """Provide configuration for integration testing."""
    return {
        'components': ['kafka', 'spark', 'redis', 'llm_pool', 'monitoring'],
        'test_scenarios': ['happy_path', 'error_handling', 'concurrent_load'],
        'timeouts': {
            'component_startup': 30,
            'test_execution': 300,
            'cleanup': 60
        },
        'retry_config': {
            'max_retries': 3,
            'backoff_factor': 2,
            'initial_delay': 1
        }
    }


@pytest.fixture
def error_simulation_config():
    """Provide configuration for error simulation testing."""
    return {
        'failure_modes': [
            {'component': 'kafka', 'failure_rate': 0.1, 'error': 'Connection timeout'},
            {'component': 'spark', 'failure_rate': 0.05, 'error': 'Out of memory'},
            {'component': 'redis', 'failure_rate': 0.02, 'error': 'Connection refused'},
            {'component': 'llm', 'failure_rate': 0.15, 'error': 'API rate limit'}
        ],
        'recovery_scenarios': [
            'immediate_retry',
            'exponential_backoff',
            'circuit_breaker',
            'fallback_service'
        ]
    }


# Utility functions for test setup
def create_test_segments(count: int = 10, content_type: str = 'mixed') -> List[Dict[str, Any]]:
    """Create test segments for testing."""
    generator = TestDataGenerator(seed=42)
    segments = []
    
    for i in range(count):
        if content_type == 'dialogue':
            text = f'"Hello, this is test dialogue {i}," character_{i % 2} said.'
            speaker = f'character_{i % 2}'
            segment_type = 'dialogue'
        elif content_type == 'narrative':
            text = generator.generate_narrative_segment(30, 60)
            speaker = 'narrator'
            segment_type = 'narrative'
        else:  # mixed
            if i % 2 == 0:
                text = generator.generate_narrative_segment(30, 60)
                speaker = 'narrator'
                segment_type = 'narrative'
            else:
                text = f'"This is dialogue segment {i}," character_{i % 3} replied.'
                speaker = f'character_{i % 3}'
                segment_type = 'dialogue'
        
        segment = {
            'segment_id': f'test_seg_{i:03d}',
            'text_content': text,
            'speaker': speaker,
            'segment_type': segment_type,
            'quality_score': 0.8 + (i % 20) / 100,
            'confidence_score': 0.85 + (i % 15) / 100,
            'processing_metadata': json.dumps({
                'tokens': len(text.split()),
                'characters': len(text),
                'complexity': 3 + (i % 5)
            })
        }
        segments.append(segment)
    
    return segments


def create_test_kafka_messages(segments: List[Dict[str, Any]], 
                             job_id: str = 'test_job') -> List[Dict[str, Any]]:
    """Create Kafka messages from test segments."""
    messages = []
    
    for i, segment in enumerate(segments):
        message = {
            'message_type': 'segment_processing',
            'job_id': job_id,
            'segment_id': segment['segment_id'],
            'text_content': segment['text_content'],
            'speaker': segment.get('speaker', 'AMBIGUOUS'),
            'segment_type': segment.get('segment_type', 'unknown'),
            'chunk_index': i // 10,  # 10 segments per chunk
            'total_chunks': (len(segments) + 9) // 10,
            'timestamp': datetime.now().isoformat(),
            'metadata': segment.get('processing_metadata', '{}')
        }
        messages.append(message)
    
    return messages


def simulate_processing_delay(operation_type: str, data_size: int = 0) -> float:
    """Simulate realistic processing delays based on operation type and data size."""
    base_delays = {
        'kafka_send': 0.001,      # 1ms
        'kafka_receive': 0.002,   # 2ms
        'spark_create_df': 0.05,  # 50ms
        'spark_collect': 0.1,     # 100ms
        'redis_get': 0.0001,      # 0.1ms
        'redis_set': 0.0002,      # 0.2ms
        'llm_process': 0.1,       # 100ms
        'validation': 0.01,       # 10ms
        'file_io': 0.005         # 5ms
    }
    
    base_delay = base_delays.get(operation_type, 0.01)
    
    # Add data size factor (logarithmic scaling)
    if data_size > 0:
        import math
        size_factor = 1 + math.log10(max(1, data_size / 1000))  # Scale by KB
        base_delay *= size_factor
    
    # Add some randomness (±20%)
    import random
    variation = 1 + (random.random() - 0.5) * 0.4
    
    return base_delay * variation


def assert_performance_within_bounds(actual_time: float, 
                                   expected_time: float, 
                                   tolerance_percent: float = 20.0,
                                   operation_name: str = "operation"):
    """Assert that performance is within acceptable bounds."""
    tolerance = expected_time * (tolerance_percent / 100.0)
    min_time = expected_time - tolerance
    max_time = expected_time + tolerance
    
    assert min_time <= actual_time <= max_time, \
        f"{operation_name} took {actual_time:.3f}s, expected {expected_time:.3f}s ±{tolerance_percent}%"


def measure_execution_time(func: Callable) -> tuple:
    """Measure execution time of a function."""
    start_time = time.time()
    result = func()
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time


class ResourceMonitor:
    """Monitor system resources during test execution."""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.monitoring = False
        self.resource_data: List[Dict[str, Any]] = []
        self.monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.resource_data.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> List[Dict[str, Any]]:
        """Stop monitoring and return collected data."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        return self.resource_data.copy()
    
    def _monitor_loop(self):
        """Resource monitoring loop."""
        try:
            import psutil
            process = psutil.Process()
            
            while self.monitoring:
                try:
                    memory_info = process.memory_info()
                    cpu_percent = process.cpu_percent()
                    
                    data = {
                        'timestamp': time.time(),
                        'memory_rss_mb': memory_info.rss / 1024 / 1024,
                        'memory_vms_mb': memory_info.vms / 1024 / 1024,
                        'cpu_percent': cpu_percent,
                        'num_threads': process.num_threads()
                    }
                    
                    self.resource_data.append(data)
                    time.sleep(self.interval)
                    
                except Exception:
                    # Continue monitoring even if individual measurements fail
                    time.sleep(self.interval)
                    
        except ImportError:
            # psutil not available, skip monitoring
            pass


@pytest.fixture
def resource_monitor():
    """Provide a resource monitor for performance testing."""
    monitor = ResourceMonitor()
    yield monitor
    if monitor.monitoring:
        monitor.stop_monitoring()