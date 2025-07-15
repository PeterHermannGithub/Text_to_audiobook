"""
Redis-based caching layer for the distributed text-to-audiobook pipeline.

This module provides intelligent caching for LLM responses, speaker embeddings,
processed segments, and other expensive computations to improve performance.
"""

import json
import hashlib
import pickle
import logging
import time
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from functools import wraps
from contextlib import contextmanager

try:
    import redis
    from redis.exceptions import RedisError, ConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    RedisError = Exception
    ConnectionError = Exception

from ..monitoring.prometheus_metrics import get_metrics_collector


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    size_bytes: int = 0
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'value': self.value,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'access_count': self.access_count,
            'size_bytes': self.size_bytes,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, key: str, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        return cls(
            key=key,
            value=data['value'],
            created_at=datetime.fromisoformat(data['created_at']),
            expires_at=datetime.fromisoformat(data['expires_at']) if data['expires_at'] else None,
            access_count=data.get('access_count', 0),
            size_bytes=data.get('size_bytes', 0),
            tags=data.get('tags', [])
        )


class RedisCacheManager:
    """
    Redis-based cache manager with intelligent caching strategies.
    
    Provides caching for:
    - LLM responses and embeddings
    - Processed text segments  
    - Speaker voice characteristics
    - Expensive computation results
    """
    
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None,
                 namespace: str = 'text_to_audiobook',
                 default_ttl: int = 3600,
                 max_memory_mb: int = 1024):
        """
        Initialize Redis cache manager.
        
        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Redis password
            namespace: Cache key namespace
            default_ttl: Default time-to-live in seconds
            max_memory_mb: Maximum memory usage in MB
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.namespace = namespace
        self.default_ttl = default_ttl
        self.max_memory_mb = max_memory_mb
        
        self.logger = logging.getLogger(__name__)
        self.metrics_collector = get_metrics_collector()
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
        
        # Initialize Redis connection
        self.redis_client = self._initialize_redis()
        
        # Cache strategies
        self.cache_strategies = {
            'llm_response': {'ttl': 7200, 'compress': True, 'serialize': 'json'},
            'speaker_embedding': {'ttl': 86400, 'compress': True, 'serialize': 'pickle'},
            'processed_segment': {'ttl': 3600, 'compress': False, 'serialize': 'json'},
            'quality_score': {'ttl': 1800, 'compress': False, 'serialize': 'json'},
            'text_analysis': {'ttl': 3600, 'compress': True, 'serialize': 'pickle'}
        }
        
        self.logger.info(f"Redis cache manager initialized (namespace: {namespace})")
    
    def _initialize_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis connection with fallback."""
        if not REDIS_AVAILABLE:
            self.logger.warning("Redis not available, caching disabled")
            return None
        
        try:
            client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=False,  # Handle binary data
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            client.ping()
            
            # Configure memory policy
            try:
                client.config_set('maxmemory', f'{self.max_memory_mb}mb')
                client.config_set('maxmemory-policy', 'allkeys-lru')
            except RedisError:
                self.logger.warning("Could not configure Redis memory policy")
            
            self.logger.info("Redis connection established")
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            return None
    
    def _generate_key(self, category: str, identifier: str, **kwargs) -> str:
        """Generate namespaced cache key."""
        # Create deterministic key from parameters
        params_str = json.dumps(kwargs, sort_keys=True) if kwargs else ""
        key_data = f"{category}:{identifier}:{params_str}"
        
        # Hash for consistent length and avoid special characters
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()[:16]
        
        return f"{self.namespace}:{category}:{key_hash}"
    
    def _serialize_value(self, value: Any, strategy: Dict[str, Any]) -> bytes:
        """Serialize value based on strategy."""
        serialize_method = strategy.get('serialize', 'json')
        
        if serialize_method == 'json':
            serialized = json.dumps(value, default=str).encode('utf-8')
        elif serialize_method == 'pickle':
            serialized = pickle.dumps(value)
        else:
            # Fallback to string representation
            serialized = str(value).encode('utf-8')
        
        # Apply compression if enabled
        if strategy.get('compress', False):
            try:
                import gzip
                serialized = gzip.compress(serialized)
            except ImportError:
                self.logger.warning("gzip not available, skipping compression")
        
        return serialized
    
    def _deserialize_value(self, data: bytes, strategy: Dict[str, Any]) -> Any:
        """Deserialize value based on strategy."""
        # Decompress if needed
        if strategy.get('compress', False):
            try:
                import gzip
                data = gzip.decompress(data)
            except ImportError:
                pass
        
        serialize_method = strategy.get('serialize', 'json')
        
        if serialize_method == 'json':
            return json.loads(data.decode('utf-8'))
        elif serialize_method == 'pickle':
            return pickle.loads(data)
        else:
            return data.decode('utf-8')
    
    def get(self, category: str, identifier: str, **kwargs) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            category: Cache category (e.g., 'llm_response', 'speaker_embedding')
            identifier: Unique identifier for the cached item
            **kwargs: Additional parameters for key generation
            
        Returns:
            Cached value or None if not found/expired
        """
        if not self.redis_client:
            return None
        
        key = self._generate_key(category, identifier, **kwargs)
        strategy = self.cache_strategies.get(category, {})
        
        try:
            data = self.redis_client.get(key)
            
            if data is None:
                self.stats['misses'] += 1
                self.metrics_collector.record_kafka_message('cache', 'miss')
                return None
            
            # Deserialize value
            value = self._deserialize_value(data, strategy)
            
            # Update access statistics
            self.stats['hits'] += 1
            self.metrics_collector.record_kafka_message('cache', 'hit')
            
            # Update access count
            try:
                access_key = f"{key}:access_count"
                self.redis_client.incr(access_key)
                self.redis_client.expire(access_key, strategy.get('ttl', self.default_ttl))
            except RedisError:
                pass
            
            self.logger.debug(f"Cache hit: {category}:{identifier}")
            return value
            
        except Exception as e:
            self.logger.error(f"Cache get error for {key}: {e}")
            self.stats['errors'] += 1
            return None
    
    def set(self, category: str, identifier: str, value: Any, 
            ttl: Optional[int] = None, tags: List[str] = None, **kwargs) -> bool:
        """
        Set value in cache.
        
        Args:
            category: Cache category
            identifier: Unique identifier
            value: Value to cache
            ttl: Time-to-live in seconds (None for strategy default)
            tags: Tags for cache entry organization
            **kwargs: Additional parameters for key generation
            
        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            return False
        
        key = self._generate_key(category, identifier, **kwargs)
        strategy = self.cache_strategies.get(category, {})
        ttl = ttl or strategy.get('ttl', self.default_ttl)
        
        try:
            # Serialize value
            serialized_data = self._serialize_value(value, strategy)
            
            # Store in Redis with TTL
            success = self.redis_client.setex(key, ttl, serialized_data)
            
            if success:
                self.stats['sets'] += 1
                self.metrics_collector.record_kafka_message('cache', 'set')
                
                # Store metadata
                metadata = {
                    'created_at': datetime.now().isoformat(),
                    'expires_at': (datetime.now() + timedelta(seconds=ttl)).isoformat(),
                    'size_bytes': len(serialized_data),
                    'tags': tags or [],
                    'access_count': 0
                }
                
                metadata_key = f"{key}:metadata"
                self.redis_client.setex(metadata_key, ttl, json.dumps(metadata))
                
                # Add to tag indexes
                if tags:
                    self._update_tag_indexes(key, tags, ttl)
                
                self.logger.debug(f"Cache set: {category}:{identifier} ({len(serialized_data)} bytes)")
                return True
            else:
                self.logger.warning(f"Failed to set cache: {key}")
                return False
                
        except Exception as e:
            self.logger.error(f"Cache set error for {key}: {e}")
            self.stats['errors'] += 1
            return False
    
    def delete(self, category: str, identifier: str, **kwargs) -> bool:
        """Delete value from cache."""
        if not self.redis_client:
            return False
        
        key = self._generate_key(category, identifier, **kwargs)
        
        try:
            # Delete main key, metadata, and access count
            keys_to_delete = [key, f"{key}:metadata", f"{key}:access_count"]
            deleted = self.redis_client.delete(*keys_to_delete)
            
            if deleted > 0:
                self.stats['deletes'] += 1
                self.logger.debug(f"Cache delete: {category}:{identifier}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Cache delete error for {key}: {e}")
            self.stats['errors'] += 1
            return False
    
    def exists(self, category: str, identifier: str, **kwargs) -> bool:
        """Check if key exists in cache."""
        if not self.redis_client:
            return False
        
        key = self._generate_key(category, identifier, **kwargs)
        
        try:
            return bool(self.redis_client.exists(key))
        except Exception:
            return False
    
    def _update_tag_indexes(self, key: str, tags: List[str], ttl: int):
        """Update tag indexes for cache organization."""
        try:
            for tag in tags:
                tag_key = f"{self.namespace}:tag:{tag}"
                self.redis_client.sadd(tag_key, key)
                self.redis_client.expire(tag_key, ttl)
        except RedisError:
            pass
    
    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all cache entries with a specific tag."""
        if not self.redis_client:
            return 0
        
        try:
            tag_key = f"{self.namespace}:tag:{tag}"
            keys = self.redis_client.smembers(tag_key)
            
            if keys:
                # Delete all keys and their metadata
                all_keys = []
                for key in keys:
                    all_keys.extend([key, f"{key}:metadata", f"{key}:access_count"])
                
                deleted = self.redis_client.delete(*all_keys)
                self.redis_client.delete(tag_key)
                
                self.logger.info(f"Invalidated {len(keys)} cache entries with tag: {tag}")
                return len(keys)
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Tag invalidation error for {tag}: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.stats.copy()
        
        if self.redis_client:
            try:
                info = self.redis_client.info('memory')
                stats.update({
                    'redis_memory_used': info.get('used_memory', 0),
                    'redis_memory_peak': info.get('used_memory_peak', 0),
                    'redis_connected': True
                })
            except Exception:
                stats['redis_connected'] = False
        else:
            stats['redis_connected'] = False
        
        # Calculate hit rate
        total_requests = stats['hits'] + stats['misses']
        stats['hit_rate'] = stats['hits'] / total_requests if total_requests > 0 else 0
        
        return stats
    
    def clear_namespace(self) -> int:
        """Clear all cache entries in the namespace."""
        if not self.redis_client:
            return 0
        
        try:
            pattern = f"{self.namespace}:*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                deleted = self.redis_client.delete(*keys)
                self.logger.info(f"Cleared {deleted} cache entries from namespace: {self.namespace}")
                return deleted
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Namespace clear error: {e}")
            return 0
    
    @contextmanager
    def pipeline(self):
        """Context manager for Redis pipeline operations."""
        if not self.redis_client:
            yield None
            return
        
        pipe = self.redis_client.pipeline()
        try:
            yield pipe
            pipe.execute()
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            pipe.reset()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Redis connection."""
        health = {
            'redis_available': REDIS_AVAILABLE,
            'connected': False,
            'latency_ms': None,
            'memory_usage': None
        }
        
        if not self.redis_client:
            return health
        
        try:
            start_time = time.time()
            self.redis_client.ping()
            latency = (time.time() - start_time) * 1000
            
            health.update({
                'connected': True,
                'latency_ms': round(latency, 2)
            })
            
            # Get memory info
            try:
                info = self.redis_client.info('memory')
                health['memory_usage'] = {
                    'used_mb': round(info.get('used_memory', 0) / 1024 / 1024, 2),
                    'peak_mb': round(info.get('used_memory_peak', 0) / 1024 / 1024, 2),
                    'limit_mb': self.max_memory_mb
                }
            except Exception:
                pass
            
        except Exception as e:
            health['error'] = str(e)
        
        return health


# Cache decorators
def cached(category: str, ttl: Optional[int] = None, tags: List[str] = None):
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache_manager()
            
            # Generate cache identifier from function name and arguments
            func_name = func.__name__
            identifier = hashlib.sha256(
                json.dumps([args, kwargs], sort_keys=True, default=str).encode()
            ).hexdigest()[:16]
            
            # Try to get from cache
            result = cache.get(category, f"{func_name}:{identifier}")
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(category, f"{func_name}:{identifier}", result, ttl=ttl, tags=tags)
            
            return result
        return wrapper
    return decorator


# Global cache manager instance
_global_cache_manager = None


def get_cache_manager() -> RedisCacheManager:
    """Get the global cache manager instance."""
    global _global_cache_manager
    
    if _global_cache_manager is None:
        _global_cache_manager = RedisCacheManager()
    
    return _global_cache_manager


def initialize_cache_manager(
    host: str = 'localhost',
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None,
    namespace: str = 'text_to_audiobook',
    max_memory_mb: int = 1024
) -> RedisCacheManager:
    """Initialize the global cache manager."""
    global _global_cache_manager
    
    _global_cache_manager = RedisCacheManager(
        host=host,
        port=port,
        db=db,
        password=password,
        namespace=namespace,
        max_memory_mb=max_memory_mb
    )
    
    return _global_cache_manager


# Convenience functions for common cache operations
def cache_llm_response(prompt: str, model: str, response: Any, ttl: int = 7200) -> bool:
    """Cache LLM response."""
    cache = get_cache_manager()
    identifier = hashlib.sha256(f"{prompt}:{model}".encode()).hexdigest()
    return cache.set('llm_response', identifier, response, ttl=ttl, tags=['llm', model])


def get_cached_llm_response(prompt: str, model: str) -> Optional[Any]:
    """Get cached LLM response."""
    cache = get_cache_manager()
    identifier = hashlib.sha256(f"{prompt}:{model}".encode()).hexdigest()
    return cache.get('llm_response', identifier)


def cache_speaker_embedding(speaker_id: str, embedding: Any, ttl: int = 86400) -> bool:
    """Cache speaker embedding."""
    cache = get_cache_manager()
    return cache.set('speaker_embedding', speaker_id, embedding, ttl=ttl, tags=['speaker', 'embedding'])


def get_cached_speaker_embedding(speaker_id: str) -> Optional[Any]:
    """Get cached speaker embedding."""
    cache = get_cache_manager()
    return cache.get('speaker_embedding', speaker_id)