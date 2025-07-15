"""
Unit tests for the Redis-based caching layer.

Tests cover cache operations, serialization strategies, TTL management,
tag-based invalidation, and performance optimization features.
"""

import pytest
import json
import time
import pickle
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.cache.redis_cache import (
    RedisCacheManager,
    CacheEntry,
    MetricTimer,
    get_cache_manager,
    initialize_cache_manager,
    cached,
    cache_llm_response,
    get_cached_llm_response,
    cache_speaker_embedding,
    get_cached_speaker_embedding
)


class TestCacheEntry:
    """Test CacheEntry dataclass."""
    
    def test_cache_entry_creation(self):
        """Test CacheEntry creation and basic properties."""
        created_time = datetime.now()
        expires_time = created_time + timedelta(hours=1)
        
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=created_time,
            expires_at=expires_time,
            access_count=5,
            size_bytes=100,
            tags=["tag1", "tag2"]
        )
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.created_at == created_time
        assert entry.expires_at == expires_time
        assert entry.access_count == 5
        assert entry.size_bytes == 100
        assert entry.tags == ["tag1", "tag2"]
    
    def test_cache_entry_default_values(self):
        """Test CacheEntry default values."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=datetime.now()
        )
        
        assert entry.expires_at is None
        assert entry.access_count == 0
        assert entry.size_bytes == 0
        assert entry.tags == []
    
    def test_cache_entry_is_expired(self):
        """Test expiration checking."""
        # Non-expiring entry
        entry_no_expiry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=datetime.now()
        )
        assert not entry_no_expiry.is_expired()
        
        # Expired entry
        entry_expired = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=datetime.now(),
            expires_at=datetime.now() - timedelta(hours=1)
        )
        assert entry_expired.is_expired()
        
        # Not yet expired entry
        entry_valid = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1)
        )
        assert not entry_valid.is_expired()
    
    def test_cache_entry_to_dict(self):
        """Test CacheEntry conversion to dictionary."""
        created_time = datetime.now()
        expires_time = created_time + timedelta(hours=1)
        
        entry = CacheEntry(
            key="test_key",
            value={"nested": "data"},
            created_at=created_time,
            expires_at=expires_time,
            access_count=3,
            size_bytes=50,
            tags=["test"]
        )
        
        entry_dict = entry.to_dict()
        
        assert isinstance(entry_dict, dict)
        assert entry_dict['value'] == {"nested": "data"}
        assert entry_dict['created_at'] == created_time.isoformat()
        assert entry_dict['expires_at'] == expires_time.isoformat()
        assert entry_dict['access_count'] == 3
        assert entry_dict['size_bytes'] == 50
        assert entry_dict['tags'] == ["test"]
    
    def test_cache_entry_from_dict(self):
        """Test CacheEntry creation from dictionary."""
        created_time = datetime.now()
        expires_time = created_time + timedelta(hours=1)
        
        data = {
            'value': {"test": "data"},
            'created_at': created_time.isoformat(),
            'expires_at': expires_time.isoformat(),
            'access_count': 5,
            'size_bytes': 100,
            'tags': ["tag1", "tag2"]
        }
        
        entry = CacheEntry.from_dict("test_key", data)
        
        assert entry.key == "test_key"
        assert entry.value == {"test": "data"}
        assert entry.created_at == created_time
        assert entry.expires_at == expires_time
        assert entry.access_count == 5
        assert entry.size_bytes == 100
        assert entry.tags == ["tag1", "tag2"]


class TestRedisCacheManager:
    """Test RedisCacheManager main functionality."""
    
    @pytest.fixture
    def mock_redis_client(self):
        """Create mock Redis client."""
        client = Mock()
        client.ping.return_value = True
        client.get.return_value = None
        client.set.return_value = True
        client.setex.return_value = True
        client.delete.return_value = 1
        client.exists.return_value = False
        client.keys.return_value = []
        client.incr.return_value = 1
        client.expire.return_value = True
        client.sadd.return_value = 1
        client.smembers.return_value = set()
        client.config_set.return_value = True
        client.info.return_value = {
            'used_memory': 1024000,
            'used_memory_peak': 2048000
        }
        return client
    
    @pytest.fixture
    def cache_manager(self, mock_redis_client):
        """Create cache manager with mocked Redis client."""
        with patch('src.cache.redis_cache.redis') as mock_redis_module:
            mock_redis_module.Redis.return_value = mock_redis_client
            manager = RedisCacheManager(
                host='localhost',
                port=6379,
                namespace='test_cache',
                default_ttl=300
            )
            return manager
    
    def test_cache_manager_initialization(self, cache_manager):
        """Test cache manager initialization."""
        assert cache_manager.namespace == 'test_cache'
        assert cache_manager.default_ttl == 300
        assert cache_manager.redis_client is not None
        assert isinstance(cache_manager.stats, dict)
        assert 'hits' in cache_manager.stats
        assert 'misses' in cache_manager.stats
        assert 'sets' in cache_manager.stats
        assert 'deletes' in cache_manager.stats
        assert 'errors' in cache_manager.stats
    
    def test_cache_manager_without_redis(self):
        """Test cache manager when Redis is not available."""
        with patch('src.cache.redis_cache.REDIS_AVAILABLE', False):
            manager = RedisCacheManager()
            assert manager.redis_client is None
    
    def test_generate_key(self, cache_manager):
        """Test cache key generation."""
        key1 = cache_manager._generate_key("test_category", "test_id")
        key2 = cache_manager._generate_key("test_category", "test_id")
        key3 = cache_manager._generate_key("test_category", "different_id")
        
        # Same inputs should generate same key
        assert key1 == key2
        
        # Different inputs should generate different keys
        assert key1 != key3
        
        # Key should contain namespace
        assert key1.startswith("test_cache:")
        
        # Test with additional parameters
        key_with_params = cache_manager._generate_key(
            "test_category", "test_id", param1="value1", param2="value2"
        )
        assert key_with_params != key1
    
    def test_serialize_deserialize_json(self, cache_manager):
        """Test JSON serialization and deserialization."""
        strategy = {'serialize': 'json', 'compress': False}
        
        # Test simple data
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        serialized = cache_manager._serialize_value(data, strategy)
        deserialized = cache_manager._deserialize_value(serialized, strategy)
        
        assert deserialized == data
        assert isinstance(serialized, bytes)
    
    def test_serialize_deserialize_pickle(self, cache_manager):
        """Test pickle serialization and deserialization."""
        strategy = {'serialize': 'pickle', 'compress': False}
        
        # Test complex data including custom objects
        class TestClass:
            def __init__(self, value):
                self.value = value
            
            def __eq__(self, other):
                return isinstance(other, TestClass) and self.value == other.value
        
        data = TestClass("test_value")
        serialized = cache_manager._serialize_value(data, strategy)
        deserialized = cache_manager._deserialize_value(serialized, strategy)
        
        assert deserialized == data
        assert isinstance(serialized, bytes)
    
    @patch('gzip.compress')
    @patch('gzip.decompress')
    def test_compression(self, mock_decompress, mock_compress, cache_manager):
        """Test compression and decompression."""
        strategy = {'serialize': 'json', 'compress': True}
        
        data = {"large": "data" * 1000}
        compressed_data = b"compressed"
        mock_compress.return_value = compressed_data
        mock_decompress.return_value = json.dumps(data).encode('utf-8')
        
        # Test compression during serialization
        serialized = cache_manager._serialize_value(data, strategy)
        mock_compress.assert_called_once()
        assert serialized == compressed_data
        
        # Test decompression during deserialization
        deserialized = cache_manager._deserialize_value(compressed_data, strategy)
        mock_decompress.assert_called_once_with(compressed_data)
        assert deserialized == data
    
    def test_cache_set_get(self, cache_manager, mock_redis_client):
        """Test basic cache set and get operations."""
        # Test successful set
        mock_redis_client.setex.return_value = True
        result = cache_manager.set("test_category", "test_id", "test_value")
        assert result is True
        
        # Verify Redis calls
        mock_redis_client.setex.assert_called()
        args = mock_redis_client.setex.call_args[0]
        assert len(args) == 3  # key, ttl, value
        
        # Test get with hit
        test_data = json.dumps("test_value").encode('utf-8')
        mock_redis_client.get.return_value = test_data
        
        value = cache_manager.get("test_category", "test_id")
        assert value == "test_value"
        
        # Verify stats
        assert cache_manager.stats['sets'] > 0
        assert cache_manager.stats['hits'] > 0
    
    def test_cache_miss(self, cache_manager, mock_redis_client):
        """Test cache miss scenario."""
        mock_redis_client.get.return_value = None
        
        value = cache_manager.get("test_category", "nonexistent_id")
        assert value is None
        assert cache_manager.stats['misses'] > 0
    
    def test_cache_delete(self, cache_manager, mock_redis_client):
        """Test cache deletion."""
        mock_redis_client.delete.return_value = 1
        
        result = cache_manager.delete("test_category", "test_id")
        assert result is True
        assert cache_manager.stats['deletes'] > 0
        
        # Verify multiple keys were deleted (main, metadata, access_count)
        mock_redis_client.delete.assert_called_once()
        call_args = mock_redis_client.delete.call_args[0]
        assert len(call_args) == 3
    
    def test_cache_exists(self, cache_manager, mock_redis_client):
        """Test cache existence check."""
        mock_redis_client.exists.return_value = True
        
        result = cache_manager.exists("test_category", "test_id")
        assert result is True
        
        mock_redis_client.exists.assert_called_once()
    
    def test_cache_with_ttl(self, cache_manager, mock_redis_client):
        """Test cache operations with custom TTL."""
        custom_ttl = 1800
        
        cache_manager.set("test_category", "test_id", "test_value", ttl=custom_ttl)
        
        # Verify TTL was used in setex call
        call_args = mock_redis_client.setex.call_args[0]
        assert call_args[1] == custom_ttl  # TTL parameter
    
    def test_cache_with_tags(self, cache_manager, mock_redis_client):
        """Test cache operations with tags."""
        tags = ["tag1", "tag2"]
        
        cache_manager.set("test_category", "test_id", "test_value", tags=tags)
        
        # Verify tag indexes were updated
        mock_redis_client.sadd.assert_called()
        mock_redis_client.expire.assert_called()
    
    def test_invalidate_by_tag(self, cache_manager, mock_redis_client):
        """Test tag-based cache invalidation."""
        # Mock tag members
        test_keys = {b'key1', b'key2', b'key3'}
        mock_redis_client.smembers.return_value = test_keys
        mock_redis_client.delete.return_value = 6  # 3 keys * 2 (with metadata)
        
        count = cache_manager.invalidate_by_tag("test_tag")
        
        assert count == 3  # Number of keys invalidated
        mock_redis_client.smembers.assert_called_once()
        mock_redis_client.delete.assert_called()
    
    def test_clear_namespace(self, cache_manager, mock_redis_client):
        """Test namespace clearing."""
        test_keys = [b'key1', b'key2', b'key3']
        mock_redis_client.keys.return_value = test_keys
        mock_redis_client.delete.return_value = 3
        
        count = cache_manager.clear_namespace()
        
        assert count == 3
        mock_redis_client.keys.assert_called_once()
        mock_redis_client.delete.assert_called_once_with(*test_keys)
    
    def test_get_stats(self, cache_manager, mock_redis_client):
        """Test statistics retrieval."""
        # Perform some operations to generate stats
        cache_manager.set("test", "id1", "value1")
        cache_manager.get("test", "id1") 
        cache_manager.get("test", "nonexistent")
        
        stats = cache_manager.get_stats()
        
        assert isinstance(stats, dict)
        assert 'hits' in stats
        assert 'misses' in stats
        assert 'sets' in stats
        assert 'hit_rate' in stats
        assert 'redis_connected' in stats
        
        # Verify hit rate calculation
        total_requests = stats['hits'] + stats['misses']
        if total_requests > 0:
            expected_hit_rate = stats['hits'] / total_requests
            assert abs(stats['hit_rate'] - expected_hit_rate) < 0.01
    
    def test_health_check(self, cache_manager, mock_redis_client):
        """Test health check functionality."""
        mock_redis_client.ping.return_value = True
        
        health = cache_manager.health_check()
        
        assert isinstance(health, dict)
        assert health['redis_available'] is True
        assert health['connected'] is True
        assert 'latency_ms' in health
        assert health['latency_ms'] is not None
    
    def test_health_check_failure(self, cache_manager, mock_redis_client):
        """Test health check with Redis failure."""
        mock_redis_client.ping.side_effect = Exception("Connection failed")
        
        health = cache_manager.health_check()
        
        assert health['connected'] is False
        assert 'error' in health
    
    def test_error_handling(self, cache_manager, mock_redis_client):
        """Test error handling in cache operations."""
        # Test set error
        mock_redis_client.setex.side_effect = Exception("Redis error")
        
        result = cache_manager.set("test", "id", "value")
        assert result is False
        assert cache_manager.stats['errors'] > 0
        
        # Reset for get error test
        cache_manager.stats['errors'] = 0
        mock_redis_client.get.side_effect = Exception("Redis error")
        
        result = cache_manager.get("test", "id")
        assert result is None
        assert cache_manager.stats['errors'] > 0
    
    def test_pipeline_context_manager(self, cache_manager, mock_redis_client):
        """Test Redis pipeline context manager."""
        mock_pipeline = Mock()
        mock_redis_client.pipeline.return_value = mock_pipeline
        
        with cache_manager.pipeline() as pipe:
            assert pipe is mock_pipeline
        
        mock_pipeline.execute.assert_called_once()
    
    def test_cache_strategies(self, cache_manager):
        """Test different cache strategies."""
        strategies = cache_manager.cache_strategies
        
        # Verify strategies exist
        assert 'llm_response' in strategies
        assert 'speaker_embedding' in strategies
        assert 'processed_segment' in strategies
        assert 'quality_score' in strategies
        assert 'text_analysis' in strategies
        
        # Verify strategy properties
        llm_strategy = strategies['llm_response']
        assert 'ttl' in llm_strategy
        assert 'compress' in llm_strategy
        assert 'serialize' in llm_strategy
        
        # Verify TTLs are reasonable
        assert llm_strategy['ttl'] > 0
        assert strategies['speaker_embedding']['ttl'] > llm_strategy['ttl']  # Longer TTL for embeddings


class TestMetricTimer:
    """Test MetricTimer context manager."""
    
    def test_metric_timer_basic(self):
        """Test basic timer functionality."""
        mock_collector = Mock()
        labels = {'job_type': 'test', 'component': 'validation'}
        
        timer = MetricTimer(mock_collector, 'processing_test', labels)
        
        with timer:
            time.sleep(0.01)  # Small delay
        
        # Verify timer recorded duration
        mock_collector.record_processing_duration.assert_called_once()
        args = mock_collector.record_processing_duration.call_args[0]
        assert args[0] == 'test'  # job_type
        assert args[1] == 'validation'  # component
        assert args[2] > 0  # duration
    
    def test_metric_timer_spark(self):
        """Test timer for Spark metrics."""
        mock_collector = Mock()
        labels = {'job_type': 'spark_test'}
        
        timer = MetricTimer(mock_collector, 'spark_job_timer', labels)
        
        with timer:
            time.sleep(0.01)
        
        mock_collector.record_spark_job.assert_called_once()
        args = mock_collector.record_spark_job.call_args[0]
        assert args[0] == 'spark_test'
        assert args[1] == 'completed'
        assert args[2] > 0
    
    def test_metric_timer_llm(self):
        """Test timer for LLM metrics."""
        mock_collector = Mock()
        labels = {'engine': 'test_engine', 'model': 'test_model'}
        
        timer = MetricTimer(mock_collector, 'llm_request_timer', labels)
        
        with timer:
            time.sleep(0.01)
        
        mock_collector.record_llm_request.assert_called_once()
        args = mock_collector.record_llm_request.call_args[0]
        assert args[0] == 'test_engine'
        assert args[1] == 'test_model'
        assert args[2] == 'completed'
        assert args[3] > 0


class TestCacheDecorators:
    """Test cache decorators and utility functions."""
    
    @patch('src.cache.redis_cache.get_cache_manager')
    def test_cached_decorator(self, mock_get_cache):
        """Test the @cached decorator."""
        mock_cache = Mock()
        mock_cache.get.return_value = None  # Cache miss
        mock_cache.set.return_value = True
        mock_get_cache.return_value = mock_cache
        
        @cached("test_category", ttl=300, tags=["test"])
        def test_function(x, y):
            return x + y
        
        # First call - should execute and cache
        result1 = test_function(1, 2)
        assert result1 == 3
        mock_cache.set.assert_called_once()
        
        # Set up cache hit for second call
        mock_cache.get.return_value = 3
        
        # Second call - should return cached value
        result2 = test_function(1, 2)
        assert result2 == 3
        assert mock_cache.get.call_count == 2  # Both calls check cache
    
    @patch('src.cache.redis_cache.get_cache_manager')
    def test_cache_llm_response(self, mock_get_cache):
        """Test LLM response caching utility."""
        mock_cache = Mock()
        mock_cache.set.return_value = True
        mock_get_cache.return_value = mock_cache
        
        result = cache_llm_response("test prompt", "test model", {"response": "data"})
        assert result is True
        
        mock_cache.set.assert_called_once()
        call_args = mock_cache.set.call_args
        assert call_args[0][0] == 'llm_response'  # category
        assert call_args[1]['ttl'] == 7200  # default TTL
        assert 'llm' in call_args[1]['tags']
    
    @patch('src.cache.redis_cache.get_cache_manager')
    def test_get_cached_llm_response(self, mock_get_cache):
        """Test LLM response retrieval utility."""
        mock_cache = Mock()
        mock_cache.get.return_value = {"response": "cached_data"}
        mock_get_cache.return_value = mock_cache
        
        result = get_cached_llm_response("test prompt", "test model")
        assert result == {"response": "cached_data"}
        
        mock_cache.get.assert_called_once()
        call_args = mock_cache.get.call_args[0]
        assert call_args[0] == 'llm_response'
    
    @patch('src.cache.redis_cache.get_cache_manager')
    def test_cache_speaker_embedding(self, mock_get_cache):
        """Test speaker embedding caching utility."""
        mock_cache = Mock()
        mock_cache.set.return_value = True
        mock_get_cache.return_value = mock_cache
        
        embedding_data = [0.1, 0.2, 0.3, 0.4]
        result = cache_speaker_embedding("speaker_001", embedding_data)
        assert result is True
        
        mock_cache.set.assert_called_once()
        call_args = mock_cache.set.call_args
        assert call_args[0][0] == 'speaker_embedding'
        assert call_args[0][1] == 'speaker_001'
        assert call_args[0][2] == embedding_data
    
    @patch('src.cache.redis_cache.get_cache_manager')
    def test_get_cached_speaker_embedding(self, mock_get_cache):
        """Test speaker embedding retrieval utility."""
        mock_cache = Mock()
        embedding_data = [0.1, 0.2, 0.3, 0.4]
        mock_cache.get.return_value = embedding_data
        mock_get_cache.return_value = mock_cache
        
        result = get_cached_speaker_embedding("speaker_001")
        assert result == embedding_data
        
        mock_cache.get.assert_called_once()


class TestCacheManagerSingleton:
    """Test global cache manager singleton pattern."""
    
    @patch('src.cache.redis_cache.RedisCacheManager')
    def test_get_cache_manager_singleton(self, mock_cache_class):
        """Test that get_cache_manager returns singleton."""
        # Reset global state
        import src.cache.redis_cache
        src.cache.redis_cache._global_cache_manager = None
        
        mock_instance = Mock()
        mock_cache_class.return_value = mock_instance
        
        # First call should create instance
        manager1 = get_cache_manager()
        assert manager1 is mock_instance
        
        # Second call should return same instance
        manager2 = get_cache_manager()
        assert manager2 is manager1
        
        # Only one instance should be created
        mock_cache_class.assert_called_once()
    
    @patch('src.cache.redis_cache.RedisCacheManager')
    def test_initialize_cache_manager(self, mock_cache_class):
        """Test cache manager initialization function."""
        mock_instance = Mock()
        mock_cache_class.return_value = mock_instance
        
        manager = initialize_cache_manager(
            host='test_host',
            port=6380,
            namespace='test_namespace'
        )
        
        assert manager is mock_instance
        mock_cache_class.assert_called_once_with(
            host='test_host',
            port=6380,
            db=0,
            password=None,
            namespace='test_namespace',
            max_memory_mb=1024
        )


class TestCacheIntegration:
    """Integration tests for cache components."""
    
    def test_cache_strategies_integration(self, cache_manager):
        """Test that cache strategies work correctly together."""
        # Test different data types with appropriate strategies
        test_cases = [
            ('llm_response', 'test_llm', {"response": "test"}, 'json'),
            ('speaker_embedding', 'speaker_1', [0.1, 0.2, 0.3], 'pickle'),
            ('processed_segment', 'seg_1', {"text": "content"}, 'json'),
            ('quality_score', 'job_1', 0.85, 'json')
        ]
        
        for category, identifier, value, expected_serialization in test_cases:
            # Set value
            success = cache_manager.set(category, identifier, value)
            assert success is True
            
            # Verify strategy was applied
            strategy = cache_manager.cache_strategies.get(category, {})
            assert strategy.get('serialize') == expected_serialization
    
    def test_error_recovery(self, cache_manager, mock_redis_client):
        """Test error recovery and graceful degradation."""
        # Test that cache continues working after errors
        mock_redis_client.get.side_effect = [Exception("Error"), b'"recovered"']
        
        # First call fails
        result1 = cache_manager.get("test", "id")
        assert result1 is None
        assert cache_manager.stats['errors'] > 0
        
        # Second call succeeds  
        result2 = cache_manager.get("test", "id")
        assert result2 == "recovered"
    
    def test_performance_characteristics(self, cache_manager):
        """Test performance-related characteristics."""
        # Test that operations complete quickly
        start_time = time.time()
        
        for i in range(10):
            cache_manager.set(f"perf_test", f"id_{i}", f"value_{i}")
            cache_manager.get(f"perf_test", f"id_{i}")
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should complete quickly (less than 1 second for 20 operations)
        assert elapsed < 1.0
    
    @pytest.mark.slow
    def test_large_data_handling(self, cache_manager):
        """Test handling of large data objects."""
        # Test with relatively large data
        large_data = {"data": "x" * 10000, "numbers": list(range(1000))}
        
        success = cache_manager.set("large_data", "test", large_data, ttl=60)
        assert success is True
        
        retrieved = cache_manager.get("large_data", "test")
        assert retrieved == large_data