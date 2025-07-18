import hashlib
import json
from typing import Dict, Any, Optional, Set, List
from config import settings


class CacheKeyGenerator:
    """
    Generates cache keys for different types of operations in the text processing pipeline.
    
    This class provides specialized cache key generation for various operations including
    rule-based attribution, LLM processing, and text preprocessing to maximize cache hit
    rates while ensuring cache correctness.
    
    Key Generation Strategies:
        - Content-based hashing for text processing operations
        - Metadata-aware keys for context-sensitive operations
        - Version-aware keys for algorithm evolution
        - Hierarchical keys for different cache levels
    """
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for consistent cache keys."""
        return ' '.join(text.strip().split())
    
    @staticmethod
    def _hash_content(content: str, length: int = 16) -> str:
        """Generate hash for content with specified length."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:length]
    
    @staticmethod
    def _serialize_metadata(metadata: Dict[str, Any]) -> str:
        """Serialize metadata to consistent string format."""
        # Extract only cache-relevant metadata
        cache_metadata = {
            'is_script_like': metadata.get('is_script_like', False),
            'format_type': metadata.get('format_type'),
            'content_type': metadata.get('content_type'),
            'pov_type': metadata.get('pov_analysis', {}).get('type'),
            'character_count': len(metadata.get('potential_character_names', set()))
        }
        
        # Remove None values
        cache_metadata = {k: v for k, v in cache_metadata.items() if v is not None}
        
        return json.dumps(cache_metadata, sort_keys=True)
    
    @staticmethod
    def _serialize_character_names(character_names: Set[str]) -> str:
        """Serialize character names for cache keys."""
        if not character_names:
            return "no_characters"
        
        # Sort for consistent ordering
        sorted_names = sorted(character_names)
        return json.dumps(sorted_names)
    
    @classmethod
    def generate_rule_based_line_key(cls, text: str, character_names: Set[str], 
                                    is_script_like: bool = False) -> str:
        """
        Generate cache key for rule-based attribution of a single line.
        
        Args:
            text: The text line to process
            character_names: Set of known character names
            is_script_like: Whether the content is script-like
            
        Returns:
            Cache key for this specific line attribution
        """
        # Normalize text for consistent caching
        normalized_text = cls._normalize_text(text)
        
        # Create key components
        components = [
            f"rule_line:{cls._hash_content(normalized_text, 12)}",
            f"chars:{cls._hash_content(cls._serialize_character_names(character_names), 8)}",
            f"script:{is_script_like}",
            f"v:1.0"  # Version for cache invalidation
        ]
        
        return "|".join(components)
    
    @classmethod
    def generate_rule_based_batch_key(cls, lines: List[str], character_names: Set[str], 
                                     metadata: Dict[str, Any]) -> str:
        """
        Generate cache key for batch rule-based attribution processing.
        
        Args:
            lines: List of text lines to process
            character_names: Set of known character names
            metadata: Text metadata for context
            
        Returns:
            Cache key for batch processing
        """
        # Create content hash from all lines
        combined_text = "\n".join(cls._normalize_text(line) for line in lines)
        content_hash = cls._hash_content(combined_text, 16)
        
        # Create metadata hash
        metadata_hash = cls._hash_content(cls._serialize_metadata(metadata), 8)
        
        # Create character names hash
        chars_hash = cls._hash_content(cls._serialize_character_names(character_names), 8)
        
        components = [
            f"rule_batch:{content_hash}",
            f"meta:{metadata_hash}",
            f"chars:{chars_hash}",
            f"v:1.0"
        ]
        
        return "|".join(components)
    
    @classmethod
    def generate_fuzzy_match_key(cls, text: str, character_names: Set[str], 
                                threshold: int = 80) -> str:
        """
        Generate cache key for fuzzy matching operations.
        
        Args:
            text: Text to match against
            character_names: Set of character names for matching
            threshold: Fuzzy matching threshold
            
        Returns:
            Cache key for fuzzy matching result
        """
        normalized_text = cls._normalize_text(text)
        chars_hash = cls._hash_content(cls._serialize_character_names(character_names), 8)
        
        components = [
            f"fuzzy:{cls._hash_content(normalized_text, 12)}",
            f"chars:{chars_hash}",
            f"thresh:{threshold}",
            f"v:1.0"
        ]
        
        return "|".join(components)
    
    @classmethod
    def generate_pattern_match_key(cls, text: str, pattern_type: str, 
                                  additional_context: Optional[str] = None) -> str:
        """
        Generate cache key for pattern matching operations.
        
        Args:
            text: Text to match patterns against
            pattern_type: Type of pattern (script, dialogue, narrative, etc.)
            additional_context: Additional context for pattern matching
            
        Returns:
            Cache key for pattern matching result
        """
        normalized_text = cls._normalize_text(text)
        
        components = [
            f"pattern:{cls._hash_content(normalized_text, 12)}",
            f"type:{pattern_type}",
            f"v:1.0"
        ]
        
        if additional_context:
            context_hash = cls._hash_content(additional_context, 6)
            components.append(f"ctx:{context_hash}")
        
        return "|".join(components)
    
    @classmethod
    def generate_character_detection_key(cls, text: str, metadata: Dict[str, Any]) -> str:
        """
        Generate cache key for character name detection operations.
        
        Args:
            text: Text to detect characters in
            metadata: Metadata for context
            
        Returns:
            Cache key for character detection result
        """
        normalized_text = cls._normalize_text(text)
        metadata_hash = cls._hash_content(cls._serialize_metadata(metadata), 8)
        
        components = [
            f"char_detect:{cls._hash_content(normalized_text, 12)}",
            f"meta:{metadata_hash}",
            f"v:1.0"
        ]
        
        return "|".join(components)
    
    @classmethod
    def generate_confidence_score_key(cls, attribution_result: Dict[str, Any], 
                                     metadata: Dict[str, Any]) -> str:
        """
        Generate cache key for confidence scoring operations.
        
        Args:
            attribution_result: Result of attribution operation
            metadata: Metadata for context
            
        Returns:
            Cache key for confidence scoring result
        """
        # Serialize attribution result
        result_str = json.dumps(attribution_result, sort_keys=True)
        result_hash = cls._hash_content(result_str, 12)
        
        metadata_hash = cls._hash_content(cls._serialize_metadata(metadata), 8)
        
        components = [
            f"confidence:{result_hash}",
            f"meta:{metadata_hash}",
            f"v:1.0"
        ]
        
        return "|".join(components)
    
    @classmethod
    def generate_spacy_processing_key(cls, text: str, model_name: str = "en_core_web_sm") -> str:
        """
        Generate cache key for spaCy NLP processing results.
        
        Args:
            text: Text to process with spaCy
            model_name: spaCy model name
            
        Returns:
            Cache key for spaCy processing result
        """
        normalized_text = cls._normalize_text(text)
        
        components = [
            f"spacy:{cls._hash_content(normalized_text, 12)}",
            f"model:{model_name}",
            f"v:1.0"
        ]
        
        return "|".join(components)
    
    @classmethod
    def generate_preprocessing_key(cls, text: str, preprocessing_config: Dict[str, Any]) -> str:
        """
        Generate cache key for text preprocessing operations.
        
        Args:
            text: Text to preprocess
            preprocessing_config: Configuration for preprocessing
            
        Returns:
            Cache key for preprocessing result
        """
        normalized_text = cls._normalize_text(text)
        
        # Hash the preprocessing configuration
        config_str = json.dumps(preprocessing_config, sort_keys=True)
        config_hash = cls._hash_content(config_str, 8)
        
        components = [
            f"preprocess:{cls._hash_content(normalized_text, 12)}",
            f"config:{config_hash}",
            f"v:1.0"
        ]
        
        return "|".join(components)
    
    @classmethod
    def generate_character_profile_key(cls, text: str, spacy_model: str = "en_core_web_sm") -> str:
        """
        Generate cache key for character profile extraction.
        
        Args:
            text: Text to extract character profiles from
            spacy_model: spaCy model name used for NLP processing
            
        Returns:
            Cache key for character profile extraction result
        """
        normalized_text = cls._normalize_text(text)
        
        components = [
            f"char_profiles:{cls._hash_content(normalized_text, 12)}",
            f"model:{spacy_model}",
            f"v:1.0"
        ]
        
        return "|".join(components)
    
    @classmethod
    def generate_pov_analysis_key(cls, text: str, sample_size: int = 2000) -> str:
        """
        Generate cache key for POV analysis operations.
        
        Args:
            text: Text to analyze for POV
            sample_size: Number of words to analyze for POV detection
            
        Returns:
            Cache key for POV analysis result
        """
        # Use first N words for POV analysis cache key
        words = text.split()[:sample_size]
        sample_text = ' '.join(words)
        normalized_text = cls._normalize_text(sample_text)
        
        components = [
            f"pov:{cls._hash_content(normalized_text, 12)}",
            f"sample:{sample_size}",
            f"v:1.0"
        ]
        
        return "|".join(components)
    
    @classmethod
    def generate_scene_break_key(cls, text: str) -> str:
        """
        Generate cache key for scene break detection operations.
        
        Args:
            text: Text to detect scene breaks in
            
        Returns:
            Cache key for scene break detection result
        """
        normalized_text = cls._normalize_text(text)
        
        components = [
            f"scene_breaks:{cls._hash_content(normalized_text, 12)}",
            f"v:1.0"
        ]
        
        return "|".join(components)
    
    @classmethod
    def generate_document_structure_key(cls, text: str) -> str:
        """
        Generate cache key for document structure analysis operations.
        
        Args:
            text: Text to analyze document structure
            
        Returns:
            Cache key for document structure analysis result
        """
        normalized_text = cls._normalize_text(text)
        
        components = [
            f"doc_structure:{cls._hash_content(normalized_text, 12)}",
            f"v:1.0"
        ]
        
        return "|".join(components)
    
    @classmethod
    def generate_full_preprocessing_key(cls, text: str, spacy_model: str = "en_core_web_sm") -> str:
        """
        Generate cache key for complete preprocessing analysis.
        
        Args:
            text: Text to preprocess
            spacy_model: spaCy model name used for NLP processing
            
        Returns:
            Cache key for complete preprocessing result
        """
        normalized_text = cls._normalize_text(text)
        
        components = [
            f"full_preprocess:{cls._hash_content(normalized_text, 12)}",
            f"model:{spacy_model}",
            f"v:1.0"
        ]
        
        return "|".join(components)