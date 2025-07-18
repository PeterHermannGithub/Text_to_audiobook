import hashlib
import json
import time
import zlib
import logging
from typing import Dict, Optional, Any, Tuple
from collections import OrderedDict
from dataclasses import dataclass, asdict
from config import settings

# Type aliases
CacheKey = str
CacheValue = Any
TextMetadata = Dict[str, Any]


@dataclass
class CacheEntry:
    """Represents a single cache entry with metadata."""
    response: str
    timestamp: float
    hits: int
    compressed: bool = False
    
    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        return time.time() - self.timestamp > settings.LLM_CACHE_TTL_SECONDS
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class LRUCache:
    """
    Least Recently Used cache implementation with TTL support.
    
    This cache automatically removes expired entries and enforces size limits
    using LRU eviction policy for optimal memory usage.
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: OrderedDict[CacheKey, CacheEntry] = OrderedDict()
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: CacheKey) -> Optional[str]:
        """Get value from cache, returns None if not found or expired."""
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        
        # Check if entry is expired
        if entry.is_expired():
            del self._cache[key]
            self.logger.debug(f"Cache entry expired: {key[:8]}...")
            return None
        
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        
        # Increment hit count
        entry.hits += 1
        
        # Decompress if needed
        if entry.compressed:
            try:
                response = zlib.decompress(entry.response.encode('utf-8')).decode('utf-8')
            except Exception as e:
                self.logger.error(f"Failed to decompress cache entry: {e}")
                del self._cache[key]
                return None
        else:
            response = entry.response
        
        self.logger.debug(f"Cache hit: {key[:8]}... (hits: {entry.hits})")
        return response
    
    def put(self, key: CacheKey, value: str) -> None:
        """Put value in cache with automatic compression and LRU eviction."""
        try:
            # Compress if enabled and beneficial
            compressed = False
            if settings.LLM_CACHE_COMPRESS_RESPONSES and len(value) > 1000:
                try:
                    compressed_value = zlib.compress(value.encode('utf-8')).decode('utf-8')
                    # Only use compression if it actually saves space
                    if len(compressed_value) < len(value) * 0.8:
                        value = compressed_value
                        compressed = True
                except Exception as e:
                    self.logger.warning(f"Compression failed, storing uncompressed: {e}")
            
            # Create cache entry
            entry = CacheEntry(
                response=value,
                timestamp=time.time(),
                hits=0,
                compressed=compressed
            )
            
            # Remove old entry if exists
            if key in self._cache:
                del self._cache[key]
            
            # Add new entry
            self._cache[key] = entry
            
            # Enforce size limit
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self.logger.debug(f"Evicted cache entry: {oldest_key[:8]}...")
            
            self.logger.debug(f"Cache put: {key[:8]}... (compressed: {compressed})")
            
        except Exception as e:
            self.logger.error(f"Failed to put cache entry: {e}")
    
    def clear_expired(self) -> int:
        """Clear all expired entries and return count of removed entries."""
        expired_keys = []
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            self.logger.info(f"Cleared {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(entry.hits for entry in self._cache.values())
        compressed_entries = sum(1 for entry in self._cache.values() if entry.compressed)
        
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'total_hits': total_hits,
            'compressed_entries': compressed_entries,
            'compression_ratio': compressed_entries / len(self._cache) if self._cache else 0
        }


class LLMCacheManager:
    """
    Manages LLM response caching with hash-based keys and intelligent cache invalidation.
    
    This cache manager provides sophisticated caching for LLM responses based on
    content hashes, metadata, and context to maximize cache hit rates while
    ensuring response accuracy.
    
    Features:
        - Hash-based cache keys for content deduplication
        - TTL-based cache expiration
        - LRU eviction policy for memory management
        - Optional response compression
        - Metadata-aware cache key generation
        - Cache hit/miss statistics
        - Async-compatible design
    
    Architecture:
        The cache manager uses a two-level approach:
        1. Content hashing for deduplication
        2. LRU cache for memory management
        
        Cache keys are generated from:
        - Prompt content (normalized)
        - Text metadata (if enabled)
        - Context hints
        - LLM configuration parameters
    """
    
    def __init__(self):
        self.cache = LRUCache(max_size=settings.LLM_CACHE_MAX_SIZE)
        self.logger = logging.getLogger(__name__)
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_requests = 0
        
        # Initialize cache
        self.logger.info(f"LLM Cache Manager initialized (max_size: {settings.LLM_CACHE_MAX_SIZE}, ttl: {settings.LLM_CACHE_TTL_SECONDS}s)")
    
    def _generate_cache_key(self, prompt: str, text_metadata: Optional[TextMetadata] = None, 
                          context_hint: Optional[str] = None, engine: str = "local", 
                          model: Optional[str] = None) -> CacheKey:
        """
        Generate a hash-based cache key from prompt and metadata.
        
        The cache key includes:
        - Normalized prompt content
        - Relevant metadata (if enabled)
        - Context hints
        - Engine and model configuration
        
        Args:
            prompt: The LLM prompt
            text_metadata: Optional metadata for context
            context_hint: Optional context hint
            engine: LLM engine ('local' or 'gcp')
            model: Model name
            
        Returns:
            Hash-based cache key
        """
        # Normalize prompt (remove extra whitespace, normalize line endings)
        normalized_prompt = ' '.join(prompt.split())
        
        # Build cache key components
        key_components = [
            f"prompt:{normalized_prompt}",
            f"engine:{engine}",
            f"model:{model or 'default'}"
        ]
        
        # Add metadata if enabled
        if settings.LLM_CACHE_INCLUDE_METADATA and text_metadata:
            # Include only relevant metadata for cache key
            relevant_metadata = {
                'pov_type': text_metadata.get('pov_analysis', {}).get('type'),
                'character_count': len(text_metadata.get('character_profiles', [])),
                'format_type': text_metadata.get('format_type'),
                'content_type': text_metadata.get('content_type')
            }
            
            # Remove None values
            relevant_metadata = {k: v for k, v in relevant_metadata.items() if v is not None}
            
            if relevant_metadata:
                metadata_str = json.dumps(relevant_metadata, sort_keys=True)
                key_components.append(f"metadata:{metadata_str}")
        
        # Add context hint if provided
        if context_hint:
            # Normalize context hint
            normalized_context = ' '.join(str(context_hint).split())
            key_components.append(f"context:{normalized_context}")
        
        # Create hash
        combined_key = "|".join(key_components)
        hash_obj = hashlib.sha256(combined_key.encode('utf-8'))
        cache_key = hash_obj.hexdigest()[:settings.LLM_CACHE_HASH_LENGTH]
        
        return cache_key
    
    def get_cached_response(self, prompt: str, text_metadata: Optional[TextMetadata] = None, 
                          context_hint: Optional[str] = None, engine: str = "local", 
                          model: Optional[str] = None) -> Optional[str]:
        """
        Get cached LLM response if available.
        
        Args:
            prompt: The LLM prompt
            text_metadata: Optional metadata for context
            context_hint: Optional context hint
            engine: LLM engine ('local' or 'gcp')
            model: Model name
            
        Returns:
            Cached response or None if not found
        """
        if not settings.LLM_CACHE_ENABLED:
            return None
        
        self._total_requests += 1
        
        try:
            cache_key = self._generate_cache_key(prompt, text_metadata, context_hint, engine, model)
            cached_response = self.cache.get(cache_key)
            
            if cached_response is not None:
                self._cache_hits += 1
                self.logger.debug(f"Cache hit for key: {cache_key}")
                return cached_response
            else:
                self._cache_misses += 1
                self.logger.debug(f"Cache miss for key: {cache_key}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting cached response: {e}")
            return None
    
    def cache_response(self, prompt: str, response: str, text_metadata: Optional[TextMetadata] = None, 
                      context_hint: Optional[str] = None, engine: str = "local", 
                      model: Optional[str] = None) -> None:
        """
        Cache an LLM response.
        
        Args:
            prompt: The LLM prompt
            response: The LLM response
            text_metadata: Optional metadata for context
            context_hint: Optional context hint
            engine: LLM engine ('local' or 'gcp')
            model: Model name
        """
        if not settings.LLM_CACHE_ENABLED:
            return
        
        try:
            cache_key = self._generate_cache_key(prompt, text_metadata, context_hint, engine, model)
            self.cache.put(cache_key, response)
            self.logger.debug(f"Cached response for key: {cache_key}")
            
        except Exception as e:
            self.logger.error(f"Error caching response: {e}")
    
    def clear_expired_entries(self) -> int:
        """Clear expired cache entries and return count."""
        return self.cache.clear_expired()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        cache_stats = self.cache.stats()
        
        hit_rate = (self._cache_hits / self._total_requests) if self._total_requests > 0 else 0
        
        return {
            'enabled': settings.LLM_CACHE_ENABLED,
            'total_requests': self._total_requests,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': cache_stats['size'],
            'max_cache_size': cache_stats['max_size'],
            'total_cached_hits': cache_stats['total_hits'],
            'compressed_entries': cache_stats['compressed_entries'],
            'compression_ratio': cache_stats['compression_ratio'],
            'ttl_seconds': settings.LLM_CACHE_TTL_SECONDS
        }
    
    def invalidate_cache(self, pattern: Optional[str] = None) -> None:
        """
        Invalidate cache entries matching a pattern.
        
        Args:
            pattern: Optional pattern to match (if None, clears all)
        """
        if pattern is None:
            # Clear all cache
            self.cache._cache.clear()
            self.logger.info("Cleared all cache entries")
        else:
            # Pattern-based invalidation would require storing original keys
            # For now, just clear expired entries
            expired_count = self.clear_expired_entries()
            self.logger.info(f"Cleared {expired_count} expired entries")
    
    def get_cache_key_for_debug(self, prompt: str, text_metadata: Optional[TextMetadata] = None, 
                               context_hint: Optional[str] = None, engine: str = "local", 
                               model: Optional[str] = None) -> str:
        """Get cache key for debugging purposes."""
        return self._generate_cache_key(prompt, text_metadata, context_hint, engine, model)