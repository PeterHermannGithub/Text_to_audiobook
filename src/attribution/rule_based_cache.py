import json
import time
import logging
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from .cache_keys import CacheKeyGenerator
from .llm.cache_manager import LRUCache
from config import settings


class RuleBasedCacheManager:
    """
    Manages caching for rule-based attribution operations.
    
    This cache manager provides specialized caching for deterministic operations
    in the rule-based attribution pipeline, including pattern matching, fuzzy
    string matching, character detection, and confidence scoring.
    
    Features:
        - Specialized cache keys for different operation types
        - Hierarchical caching for different granularities
        - TTL-based expiration for cache freshness
        - Performance monitoring and statistics
        - Memory-efficient storage with compression
    
    Cache Levels:
        - Line-level: Individual line attribution results
        - Batch-level: Batch processing results
        - Pattern-level: Pattern matching results
        - Fuzzy-level: Fuzzy matching results
        - Character-level: Character detection results
    """
    
    def __init__(self, max_size: int = None):
        self.max_size = max_size or settings.LLM_CACHE_MAX_SIZE
        self.cache = LRUCache(max_size=self.max_size)
        self.logger = logging.getLogger(__name__)
        
        # Statistics tracking
        self._line_cache_hits = 0
        self._line_cache_misses = 0
        self._batch_cache_hits = 0
        self._batch_cache_misses = 0
        self._pattern_cache_hits = 0
        self._pattern_cache_misses = 0
        self._fuzzy_cache_hits = 0
        self._fuzzy_cache_misses = 0
        
        self.logger.info(f"Rule-based cache manager initialized (max_size: {self.max_size})")
    
    def get_line_attribution(self, text: str, character_names: Set[str], 
                           is_script_like: bool = False) -> Optional[Tuple[str, float]]:
        """
        Get cached line attribution result.
        
        Args:
            text: Text line to check
            character_names: Set of known character names
            is_script_like: Whether content is script-like
            
        Returns:
            Cached (speaker, confidence) tuple or None if not cached
        """
        if not settings.LLM_CACHE_ENABLED:
            return None
            
        cache_key = CacheKeyGenerator.generate_rule_based_line_key(
            text, character_names, is_script_like
        )
        
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self._line_cache_hits += 1
            self.logger.debug(f"Rule-based line cache hit: {cache_key[:16]}...")
            
            # Deserialize result
            try:
                result_data = json.loads(cached_result)
                return result_data.get('speaker'), result_data.get('confidence', 0.0)
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.error(f"Failed to deserialize cached line result: {e}")
                return None
        else:
            self._line_cache_misses += 1
            self.logger.debug(f"Rule-based line cache miss: {cache_key[:16]}...")
            return None
    
    def cache_line_attribution(self, text: str, character_names: Set[str], 
                             speaker: str, confidence: float, is_script_like: bool = False) -> None:
        """
        Cache line attribution result.
        
        Args:
            text: Text line that was processed
            character_names: Set of known character names
            speaker: Attributed speaker
            confidence: Attribution confidence score
            is_script_like: Whether content is script-like
        """
        if not settings.LLM_CACHE_ENABLED:
            return
            
        cache_key = CacheKeyGenerator.generate_rule_based_line_key(
            text, character_names, is_script_like
        )
        
        # Serialize result
        result_data = {
            'speaker': speaker,
            'confidence': confidence,
            'timestamp': time.time()
        }
        
        try:
            cached_value = json.dumps(result_data)
            self.cache.put(cache_key, cached_value)
            self.logger.debug(f"Cached line attribution: {cache_key[:16]}...")
        except Exception as e:
            self.logger.error(f"Failed to cache line attribution: {e}")
    
    def get_batch_attribution(self, lines: List[str], character_names: Set[str], 
                            metadata: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached batch attribution result.
        
        Args:
            lines: List of text lines
            character_names: Set of known character names
            metadata: Text metadata
            
        Returns:
            Cached batch attribution result or None if not cached
        """
        if not settings.LLM_CACHE_ENABLED:
            return None
            
        cache_key = CacheKeyGenerator.generate_rule_based_batch_key(
            lines, character_names, metadata
        )
        
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self._batch_cache_hits += 1
            self.logger.debug(f"Rule-based batch cache hit: {cache_key[:16]}...")
            
            try:
                result_data = json.loads(cached_result)
                return result_data.get('attributed_lines', [])
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.error(f"Failed to deserialize cached batch result: {e}")
                return None
        else:
            self._batch_cache_misses += 1
            self.logger.debug(f"Rule-based batch cache miss: {cache_key[:16]}...")
            return None
    
    def cache_batch_attribution(self, lines: List[str], character_names: Set[str], 
                              metadata: Dict[str, Any], attributed_lines: List[Dict[str, Any]]) -> None:
        """
        Cache batch attribution result.
        
        Args:
            lines: List of text lines that were processed
            character_names: Set of known character names
            metadata: Text metadata
            attributed_lines: Attribution results
        """
        if not settings.LLM_CACHE_ENABLED:
            return
            
        cache_key = CacheKeyGenerator.generate_rule_based_batch_key(
            lines, character_names, metadata
        )
        
        # Serialize result
        result_data = {
            'attributed_lines': attributed_lines,
            'timestamp': time.time(),
            'line_count': len(lines)
        }
        
        try:
            cached_value = json.dumps(result_data)
            self.cache.put(cache_key, cached_value)
            self.logger.debug(f"Cached batch attribution: {cache_key[:16]}...")
        except Exception as e:
            self.logger.error(f"Failed to cache batch attribution: {e}")
    
    def get_fuzzy_match_result(self, text: str, character_names: Set[str], 
                             threshold: int = 80) -> Optional[Tuple[str, int]]:
        """
        Get cached fuzzy matching result.
        
        Args:
            text: Text to match
            character_names: Set of character names
            threshold: Fuzzy matching threshold
            
        Returns:
            Cached (best_match, score) tuple or None if not cached
        """
        if not settings.LLM_CACHE_ENABLED:
            return None
            
        cache_key = CacheKeyGenerator.generate_fuzzy_match_key(
            text, character_names, threshold
        )
        
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self._fuzzy_cache_hits += 1
            self.logger.debug(f"Fuzzy match cache hit: {cache_key[:16]}...")
            
            try:
                result_data = json.loads(cached_result)
                return result_data.get('best_match'), result_data.get('score', 0)
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.error(f"Failed to deserialize cached fuzzy result: {e}")
                return None
        else:
            self._fuzzy_cache_misses += 1
            self.logger.debug(f"Fuzzy match cache miss: {cache_key[:16]}...")
            return None
    
    def cache_fuzzy_match_result(self, text: str, character_names: Set[str], 
                               best_match: str, score: int, threshold: int = 80) -> None:
        """
        Cache fuzzy matching result.
        
        Args:
            text: Text that was matched
            character_names: Set of character names
            best_match: Best fuzzy match found
            score: Fuzzy match score
            threshold: Fuzzy matching threshold
        """
        if not settings.LLM_CACHE_ENABLED:
            return
            
        cache_key = CacheKeyGenerator.generate_fuzzy_match_key(
            text, character_names, threshold
        )
        
        # Serialize result
        result_data = {
            'best_match': best_match,
            'score': score,
            'timestamp': time.time()
        }
        
        try:
            cached_value = json.dumps(result_data)
            self.cache.put(cache_key, cached_value)
            self.logger.debug(f"Cached fuzzy match result: {cache_key[:16]}...")
        except Exception as e:
            self.logger.error(f"Failed to cache fuzzy match result: {e}")
    
    def get_pattern_match_result(self, text: str, pattern_type: str, 
                               additional_context: Optional[str] = None) -> Optional[bool]:
        """
        Get cached pattern matching result.
        
        Args:
            text: Text to check pattern against
            pattern_type: Type of pattern
            additional_context: Additional context
            
        Returns:
            Cached pattern match result or None if not cached
        """
        if not settings.LLM_CACHE_ENABLED:
            return None
            
        cache_key = CacheKeyGenerator.generate_pattern_match_key(
            text, pattern_type, additional_context
        )
        
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self._pattern_cache_hits += 1
            self.logger.debug(f"Pattern match cache hit: {cache_key[:16]}...")
            
            try:
                result_data = json.loads(cached_result)
                return result_data.get('matches', False)
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.error(f"Failed to deserialize cached pattern result: {e}")
                return None
        else:
            self._pattern_cache_misses += 1
            self.logger.debug(f"Pattern match cache miss: {cache_key[:16]}...")
            return None
    
    def cache_pattern_match_result(self, text: str, pattern_type: str, matches: bool, 
                                 additional_context: Optional[str] = None) -> None:
        """
        Cache pattern matching result.
        
        Args:
            text: Text that was checked
            pattern_type: Type of pattern
            matches: Whether pattern matched
            additional_context: Additional context
        """
        if not settings.LLM_CACHE_ENABLED:
            return
            
        cache_key = CacheKeyGenerator.generate_pattern_match_key(
            text, pattern_type, additional_context
        )
        
        # Serialize result
        result_data = {
            'matches': matches,
            'timestamp': time.time()
        }
        
        try:
            cached_value = json.dumps(result_data)
            self.cache.put(cache_key, cached_value)
            self.logger.debug(f"Cached pattern match result: {cache_key[:16]}...")
        except Exception as e:
            self.logger.error(f"Failed to cache pattern match result: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        base_stats = self.cache.stats()
        
        total_requests = (self._line_cache_hits + self._line_cache_misses + 
                         self._batch_cache_hits + self._batch_cache_misses +
                         self._pattern_cache_hits + self._pattern_cache_misses +
                         self._fuzzy_cache_hits + self._fuzzy_cache_misses)
        
        total_hits = (self._line_cache_hits + self._batch_cache_hits + 
                     self._pattern_cache_hits + self._fuzzy_cache_hits)
        
        hit_rate = (total_hits / total_requests) if total_requests > 0 else 0
        
        return {
            'enabled': settings.LLM_CACHE_ENABLED,
            'total_requests': total_requests,
            'total_hits': total_hits,
            'hit_rate': hit_rate,
            'cache_size': base_stats['size'],
            'max_cache_size': base_stats['max_size'],
            'line_cache_hits': self._line_cache_hits,
            'line_cache_misses': self._line_cache_misses,
            'batch_cache_hits': self._batch_cache_hits,
            'batch_cache_misses': self._batch_cache_misses,
            'pattern_cache_hits': self._pattern_cache_hits,
            'pattern_cache_misses': self._pattern_cache_misses,
            'fuzzy_cache_hits': self._fuzzy_cache_hits,
            'fuzzy_cache_misses': self._fuzzy_cache_misses,
            'compressed_entries': base_stats['compressed_entries'],
            'compression_ratio': base_stats['compression_ratio']
        }
    
    def clear_cache(self) -> None:
        """Clear all cached rule-based results."""
        self.cache._cache.clear()
        self.logger.info("Rule-based cache cleared")
    
    def clear_expired_entries(self) -> int:
        """Clear expired cache entries and return count."""
        return self.cache.clear_expired()
    
    def get_cache_key_for_debug(self, operation_type: str, **kwargs) -> str:
        """Get cache key for debugging purposes."""
        if operation_type == "line":
            return CacheKeyGenerator.generate_rule_based_line_key(
                kwargs.get('text', ''), kwargs.get('character_names', set()),
                kwargs.get('is_script_like', False)
            )
        elif operation_type == "batch":
            return CacheKeyGenerator.generate_rule_based_batch_key(
                kwargs.get('lines', []), kwargs.get('character_names', set()),
                kwargs.get('metadata', {})
            )
        elif operation_type == "fuzzy":
            return CacheKeyGenerator.generate_fuzzy_match_key(
                kwargs.get('text', ''), kwargs.get('character_names', set()),
                kwargs.get('threshold', 80)
            )
        elif operation_type == "pattern":
            return CacheKeyGenerator.generate_pattern_match_key(
                kwargs.get('text', ''), kwargs.get('pattern_type', ''),
                kwargs.get('additional_context')
            )
        else:
            return f"unknown_operation:{operation_type}"