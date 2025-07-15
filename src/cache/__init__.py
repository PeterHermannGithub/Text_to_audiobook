"""
Caching layer for the distributed text-to-audiobook pipeline.

This package provides intelligent caching capabilities for improving performance:
- Redis-based distributed caching
- LLM response caching
- Speaker embedding caching
- Processed segment caching
- Cache decorators and utilities
"""

from .redis_cache import (
    RedisCacheManager,
    CacheEntry,
    get_cache_manager,
    initialize_cache_manager,
    cached,
    cache_llm_response,
    get_cached_llm_response,
    cache_speaker_embedding,
    get_cached_speaker_embedding
)

__all__ = [
    'RedisCacheManager',
    'CacheEntry',
    'get_cache_manager',
    'initialize_cache_manager',
    'cached',
    'cache_llm_response',
    'get_cached_llm_response',
    'cache_speaker_embedding',
    'get_cached_speaker_embedding'
]

__version__ = '1.0.0'