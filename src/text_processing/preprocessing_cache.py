import json
import time
import zlib
import pickle
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from ..attribution.cache_keys import CacheKeyGenerator
from ..attribution.llm.cache_manager import LRUCache
from config import settings

class PreprocessingCacheManager:
    """
    Manages caching for expensive text preprocessing operations.
    
    This cache manager provides specialized caching for NLP preprocessing results,
    including spaCy document caching, character profile caching, POV analysis,
    scene break detection, and document structure analysis.
    
    Features:
        - spaCy document serialization and compression
        - Character profile caching with fuzzy matching consolidation
        - POV analysis caching with narrator discovery
        - Scene break detection caching
        - Document structure analysis caching
        - Performance monitoring and statistics
        - TTL-based expiration for cache freshness
    
    Cache Levels:
        - Full preprocessing: Complete analysis results
        - spaCy docs: Serialized spaCy doc objects
        - Character profiles: Extracted character profiles
        - POV analysis: Point of view analysis results
        - Scene breaks: Scene break detection results
        - Document structure: Document structure analysis results
    """
    
    def __init__(self, max_size: int = None):
        self.max_size = max_size or settings.PREPROCESSING_CACHE_MAX_SIZE
        self.cache = LRUCache(max_size=self.max_size)
        self.logger = logging.getLogger(__name__)
        
        # Statistics tracking
        self._full_preprocessing_hits = 0
        self._full_preprocessing_misses = 0
        self._spacy_cache_hits = 0
        self._spacy_cache_misses = 0
        self._character_profile_hits = 0
        self._character_profile_misses = 0
        self._pov_analysis_hits = 0
        self._pov_analysis_misses = 0
        self._scene_break_hits = 0
        self._scene_break_misses = 0
        self._document_structure_hits = 0
        self._document_structure_misses = 0
        
        self.logger.info(f"Preprocessing cache manager initialized (max_size: {self.max_size})")
    
    def get_full_preprocessing_result(self, text: str, spacy_model: str = "en_core_web_sm") -> Optional[Dict[str, Any]]:
        """
        Get cached full preprocessing result.
        
        Args:
            text: Text to check for cached preprocessing
            spacy_model: spaCy model name used for processing
            
        Returns:
            Cached preprocessing result or None if not cached
        """
        if not settings.PREPROCESSING_CACHE_ENABLED:
            return None
            
        cache_key = CacheKeyGenerator.generate_full_preprocessing_key(text, spacy_model)
        
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self._full_preprocessing_hits += 1
            self.logger.debug(f"Full preprocessing cache hit: {cache_key[:16]}...")
            
            try:
                result_data = json.loads(cached_result)
                return result_data
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.error(f"Failed to deserialize cached full preprocessing result: {e}")
                return None
        else:
            self._full_preprocessing_misses += 1
            self.logger.debug(f"Full preprocessing cache miss: {cache_key[:16]}...")
            return None
    
    def cache_full_preprocessing_result(self, text: str, result: Dict[str, Any], 
                                      spacy_model: str = "en_core_web_sm") -> None:
        """
        Cache full preprocessing result.
        
        Args:
            text: Text that was processed
            result: Full preprocessing result
            spacy_model: spaCy model name used for processing
        """
        if not settings.PREPROCESSING_CACHE_ENABLED:
            return
            
        cache_key = CacheKeyGenerator.generate_full_preprocessing_key(text, spacy_model)
        
        # Create cacheable result (exclude non-serializable data)
        cacheable_result = self._make_result_cacheable(result)
        
        # Serialize result
        result_data = {
            'result': cacheable_result,
            'timestamp': time.time(),
            'spacy_model': spacy_model
        }
        
        try:
            cached_value = json.dumps(result_data)
            self.cache.put(cache_key, cached_value)
            self.logger.debug(f"Cached full preprocessing result: {cache_key[:16]}...")
        except Exception as e:
            self.logger.error(f"Failed to cache full preprocessing result: {e}")
    
    def get_spacy_doc_bytes(self, text: str, spacy_model: str = "en_core_web_sm") -> Optional[bytes]:
        """
        Get cached spaCy document bytes.
        
        Args:
            text: Text to check for cached spaCy doc
            spacy_model: spaCy model name used for processing
            
        Returns:
            Cached spaCy doc bytes or None if not cached
        """
        if not settings.SPACY_CACHE_ENABLED or not settings.SPACY_CACHE_SERIALIZE_DOCS:
            return None
            
        cache_key = CacheKeyGenerator.generate_spacy_processing_key(text, spacy_model)
        
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self._spacy_cache_hits += 1
            self.logger.debug(f"spaCy cache hit: {cache_key[:16]}...")
            
            try:
                # Deserialize and decompress if needed
                if settings.SPACY_CACHE_COMPRESS_DOCS:
                    doc_bytes = zlib.decompress(cached_result.encode('latin-1'))
                else:
                    doc_bytes = cached_result.encode('latin-1')
                return doc_bytes
            except Exception as e:
                self.logger.error(f"Failed to deserialize cached spaCy doc: {e}")
                return None
        else:
            self._spacy_cache_misses += 1
            self.logger.debug(f"spaCy cache miss: {cache_key[:16]}...")
            return None
    
    def cache_spacy_doc_bytes(self, text: str, doc_bytes: bytes, spacy_model: str = "en_core_web_sm") -> None:
        """
        Cache spaCy document bytes.
        
        Args:
            text: Text that was processed
            doc_bytes: Serialized spaCy document bytes
            spacy_model: spaCy model name used for processing
        """
        if not settings.SPACY_CACHE_ENABLED or not settings.SPACY_CACHE_SERIALIZE_DOCS:
            return
            
        cache_key = CacheKeyGenerator.generate_spacy_processing_key(text, spacy_model)
        
        try:
            # Compress doc bytes if enabled
            if settings.SPACY_CACHE_COMPRESS_DOCS:
                compressed_bytes = zlib.compress(doc_bytes)
                cached_value = compressed_bytes.decode('latin-1')
            else:
                cached_value = doc_bytes.decode('latin-1')
            
            self.cache.put(cache_key, cached_value)
            self.logger.debug(f"Cached spaCy doc: {cache_key[:16]}... (size: {len(doc_bytes)} bytes)")
        except Exception as e:
            self.logger.error(f"Failed to cache spaCy doc: {e}")
    
    def get_character_profiles(self, text: str, spacy_model: str = "en_core_web_sm") -> Optional[List[Dict[str, Any]]]:
        """
        Get cached character profiles.
        
        Args:
            text: Text to check for cached character profiles
            spacy_model: spaCy model name used for processing
            
        Returns:
            Cached character profiles or None if not cached
        """
        if not settings.CHARACTER_PROFILE_CACHE_ENABLED:
            return None
            
        cache_key = CacheKeyGenerator.generate_character_profile_key(text, spacy_model)
        
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self._character_profile_hits += 1
            self.logger.debug(f"Character profile cache hit: {cache_key[:16]}...")
            
            try:
                result_data = json.loads(cached_result)
                return result_data.get('profiles', [])
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.error(f"Failed to deserialize cached character profiles: {e}")
                return None
        else:
            self._character_profile_misses += 1
            self.logger.debug(f"Character profile cache miss: {cache_key[:16]}...")
            return None
    
    def cache_character_profiles(self, text: str, profiles: List[Dict[str, Any]], 
                               spacy_model: str = "en_core_web_sm") -> None:
        """
        Cache character profiles.
        
        Args:
            text: Text that was processed
            profiles: Character profiles to cache
            spacy_model: spaCy model name used for processing
        """
        if not settings.CHARACTER_PROFILE_CACHE_ENABLED:
            return
            
        cache_key = CacheKeyGenerator.generate_character_profile_key(text, spacy_model)
        
        # Serialize result
        result_data = {
            'profiles': profiles,
            'timestamp': time.time(),
            'spacy_model': spacy_model
        }
        
        try:
            cached_value = json.dumps(result_data)
            self.cache.put(cache_key, cached_value)
            self.logger.debug(f"Cached character profiles: {cache_key[:16]}... ({len(profiles)} profiles)")
        except Exception as e:
            self.logger.error(f"Failed to cache character profiles: {e}")
    
    def get_pov_analysis(self, text: str, sample_size: int = 2000) -> Optional[Dict[str, Any]]:
        """
        Get cached POV analysis result.
        
        Args:
            text: Text to check for cached POV analysis
            sample_size: Number of words analyzed for POV detection
            
        Returns:
            Cached POV analysis result or None if not cached
        """
        if not settings.POV_ANALYSIS_CACHE_ENABLED:
            return None
            
        cache_key = CacheKeyGenerator.generate_pov_analysis_key(text, sample_size)
        
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self._pov_analysis_hits += 1
            self.logger.debug(f"POV analysis cache hit: {cache_key[:16]}...")
            
            try:
                result_data = json.loads(cached_result)
                return result_data.get('pov_analysis', {})
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.error(f"Failed to deserialize cached POV analysis: {e}")
                return None
        else:
            self._pov_analysis_misses += 1
            self.logger.debug(f"POV analysis cache miss: {cache_key[:16]}...")
            return None
    
    def cache_pov_analysis(self, text: str, pov_analysis: Dict[str, Any], sample_size: int = 2000) -> None:
        """
        Cache POV analysis result.
        
        Args:
            text: Text that was processed
            pov_analysis: POV analysis result to cache
            sample_size: Number of words analyzed for POV detection
        """
        if not settings.POV_ANALYSIS_CACHE_ENABLED:
            return
            
        cache_key = CacheKeyGenerator.generate_pov_analysis_key(text, sample_size)
        
        # Serialize result
        result_data = {
            'pov_analysis': pov_analysis,
            'timestamp': time.time(),
            'sample_size': sample_size
        }
        
        try:
            cached_value = json.dumps(result_data)
            self.cache.put(cache_key, cached_value)
            self.logger.debug(f"Cached POV analysis: {cache_key[:16]}... (type: {pov_analysis.get('type', 'unknown')})")
        except Exception as e:
            self.logger.error(f"Failed to cache POV analysis: {e}")
    
    def get_scene_breaks(self, text: str) -> Optional[List[int]]:
        """
        Get cached scene break detection result.
        
        Args:
            text: Text to check for cached scene breaks
            
        Returns:
            Cached scene break positions or None if not cached
        """
        if not settings.SCENE_BREAK_CACHE_ENABLED:
            return None
            
        cache_key = CacheKeyGenerator.generate_scene_break_key(text)
        
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self._scene_break_hits += 1
            self.logger.debug(f"Scene break cache hit: {cache_key[:16]}...")
            
            try:
                result_data = json.loads(cached_result)
                return result_data.get('scene_breaks', [])
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.error(f"Failed to deserialize cached scene breaks: {e}")
                return None
        else:
            self._scene_break_misses += 1
            self.logger.debug(f"Scene break cache miss: {cache_key[:16]}...")
            return None
    
    def cache_scene_breaks(self, text: str, scene_breaks: List[int]) -> None:
        """
        Cache scene break detection result.
        
        Args:
            text: Text that was processed
            scene_breaks: Scene break positions to cache
        """
        if not settings.SCENE_BREAK_CACHE_ENABLED:
            return
            
        cache_key = CacheKeyGenerator.generate_scene_break_key(text)
        
        # Serialize result
        result_data = {
            'scene_breaks': scene_breaks,
            'timestamp': time.time()
        }
        
        try:
            cached_value = json.dumps(result_data)
            self.cache.put(cache_key, cached_value)
            self.logger.debug(f"Cached scene breaks: {cache_key[:16]}... ({len(scene_breaks)} breaks)")
        except Exception as e:
            self.logger.error(f"Failed to cache scene breaks: {e}")
    
    def get_document_structure(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Get cached document structure analysis result.
        
        Args:
            text: Text to check for cached document structure
            
        Returns:
            Cached document structure analysis or None if not cached
        """
        if not settings.DOCUMENT_STRUCTURE_CACHE_ENABLED:
            return None
            
        cache_key = CacheKeyGenerator.generate_document_structure_key(text)
        
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self._document_structure_hits += 1
            self.logger.debug(f"Document structure cache hit: {cache_key[:16]}...")
            
            try:
                result_data = json.loads(cached_result)
                return result_data.get('document_structure', {})
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.error(f"Failed to deserialize cached document structure: {e}")
                return None
        else:
            self._document_structure_misses += 1
            self.logger.debug(f"Document structure cache miss: {cache_key[:16]}...")
            return None
    
    def cache_document_structure(self, text: str, document_structure: Dict[str, Any]) -> None:
        """
        Cache document structure analysis result.
        
        Args:
            text: Text that was processed
            document_structure: Document structure analysis to cache
        """
        if not settings.DOCUMENT_STRUCTURE_CACHE_ENABLED:
            return
            
        cache_key = CacheKeyGenerator.generate_document_structure_key(text)
        
        # Serialize result
        result_data = {
            'document_structure': document_structure,
            'timestamp': time.time()
        }
        
        try:
            cached_value = json.dumps(result_data)
            self.cache.put(cache_key, cached_value)
            self.logger.debug(f"Cached document structure: {cache_key[:16]}... (genre: {document_structure.get('estimated_genre', 'unknown')})")
        except Exception as e:
            self.logger.error(f"Failed to cache document structure: {e}")
    
    def _make_result_cacheable(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make preprocessing result cacheable by removing non-serializable data.
        
        Args:
            result: Original preprocessing result
            
        Returns:
            Cacheable version of the result
        """
        cacheable_result = result.copy()
        
        # Convert sets to lists for JSON serialization
        if 'dialogue_markers' in cacheable_result:
            cacheable_result['dialogue_markers'] = list(cacheable_result['dialogue_markers'])
        
        if 'potential_character_names' in cacheable_result:
            cacheable_result['potential_character_names'] = list(cacheable_result['potential_character_names'])
        
        return cacheable_result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        base_stats = self.cache.stats()
        
        # Calculate totals
        total_requests = (
            self._full_preprocessing_hits + self._full_preprocessing_misses +
            self._spacy_cache_hits + self._spacy_cache_misses +
            self._character_profile_hits + self._character_profile_misses +
            self._pov_analysis_hits + self._pov_analysis_misses +
            self._scene_break_hits + self._scene_break_misses +
            self._document_structure_hits + self._document_structure_misses
        )
        
        total_hits = (
            self._full_preprocessing_hits + self._spacy_cache_hits +
            self._character_profile_hits + self._pov_analysis_hits +
            self._scene_break_hits + self._document_structure_hits
        )
        
        hit_rate = (total_hits / total_requests) if total_requests > 0 else 0
        
        return {
            'enabled': settings.PREPROCESSING_CACHE_ENABLED,
            'total_requests': total_requests,
            'total_hits': total_hits,
            'hit_rate': hit_rate,
            'cache_size': base_stats['size'],
            'max_cache_size': base_stats['max_size'],
            'full_preprocessing_hits': self._full_preprocessing_hits,
            'full_preprocessing_misses': self._full_preprocessing_misses,
            'spacy_cache_hits': self._spacy_cache_hits,
            'spacy_cache_misses': self._spacy_cache_misses,
            'character_profile_hits': self._character_profile_hits,
            'character_profile_misses': self._character_profile_misses,
            'pov_analysis_hits': self._pov_analysis_hits,
            'pov_analysis_misses': self._pov_analysis_misses,
            'scene_break_hits': self._scene_break_hits,
            'scene_break_misses': self._scene_break_misses,
            'document_structure_hits': self._document_structure_hits,
            'document_structure_misses': self._document_structure_misses,
            'compressed_entries': base_stats['compressed_entries'],
            'compression_ratio': base_stats['compression_ratio']
        }
    
    def clear_cache(self) -> None:
        """Clear all cached preprocessing results."""
        self.cache._cache.clear()
        self.logger.info("Preprocessing cache cleared")
    
    def clear_expired_entries(self) -> int:
        """Clear expired cache entries and return count."""
        return self.cache.clear_expired()
    
    def get_cache_key_for_debug(self, operation_type: str, **kwargs) -> str:
        """Get cache key for debugging purposes."""
        if operation_type == "full_preprocessing":
            return CacheKeyGenerator.generate_full_preprocessing_key(
                kwargs.get('text', ''), kwargs.get('spacy_model', 'en_core_web_sm')
            )
        elif operation_type == "spacy_doc":
            return CacheKeyGenerator.generate_spacy_processing_key(
                kwargs.get('text', ''), kwargs.get('spacy_model', 'en_core_web_sm')
            )
        elif operation_type == "character_profiles":
            return CacheKeyGenerator.generate_character_profile_key(
                kwargs.get('text', ''), kwargs.get('spacy_model', 'en_core_web_sm')
            )
        elif operation_type == "pov_analysis":
            return CacheKeyGenerator.generate_pov_analysis_key(
                kwargs.get('text', ''), kwargs.get('sample_size', 2000)
            )
        elif operation_type == "scene_breaks":
            return CacheKeyGenerator.generate_scene_break_key(kwargs.get('text', ''))
        elif operation_type == "document_structure":
            return CacheKeyGenerator.generate_document_structure_key(kwargs.get('text', ''))
        else:
            return f"unknown_operation:{operation_type}"