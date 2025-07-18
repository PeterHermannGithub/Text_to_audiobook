#!/usr/bin/env python3
"""
Test script for preprocessing cache functionality.

This script validates that the preprocessing cache system works correctly,
including spaCy document caching, character profile caching, POV analysis
caching, scene break caching, and document structure caching.
"""

import os
import sys
import time
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from text_processing.preprocessor import TextPreprocessor
from text_processing.preprocessing_cache import PreprocessingCacheManager
from attribution.cache_keys import CacheKeyGenerator
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PreprocessingCacheTest:
    """Test suite for preprocessing cache functionality."""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        
        # Sample texts for testing
        self.sample_texts = {
            'simple_dialogue': '''
"Hello there," Alice said with a smile.
Bob nodded in response. "How are you today?"
"I'm doing well, thank you," Alice replied.
The sun was shining brightly outside.
''',
            'script_format': '''
ALICE: Good morning, Bob!
BOB: Good morning, Alice. Ready for the meeting?
ALICE: Absolutely. Let's go.
Enter CHARLIE from stage left.
CHARLIE: Sorry I'm late, everyone.
''',
            'mixed_content': '''
Chapter 1: The Beginning

"This is the start of our story," the narrator began.
Alice walked into the room. She was excited about the day ahead.
"Bob, are you ready?" she asked.
Bob looked up from his book. "Yes, I think so."
The room was filled with anticipation.
''',
            'first_person': '''
My name is John, and this is my story.
I walked into the coffee shop that morning.
"Can I help you?" the barista asked.
"I'll have a coffee, please," I replied.
The aroma of fresh coffee filled the air.
'''
        }
        
        # Initialize spaCy model (mock for testing)
        self.nlp_model = None
        try:
            import spacy
            self.nlp_model = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not available. Using mock model for testing.")
            self.nlp_model = MockSpacyModel()
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor(self.nlp_model)
        
        # Initialize cache manager
        self.cache_manager = PreprocessingCacheManager()
        
        logger.info("PreprocessingCacheTest initialized")
    
    def run_all_tests(self):
        """Run all preprocessing cache tests."""
        logger.info("="*60)
        logger.info("PREPROCESSING CACHE TEST SUITE")
        logger.info("="*60)
        
        # Test cache key generation
        self.test_cache_key_generation()
        
        # Test cache hit/miss behavior
        self.test_cache_hit_miss_behavior()
        
        # Test spaCy document caching
        self.test_spacy_document_caching()
        
        # Test character profile caching
        self.test_character_profile_caching()
        
        # Test POV analysis caching
        self.test_pov_analysis_caching()
        
        # Test scene break caching
        self.test_scene_break_caching()
        
        # Test document structure caching
        self.test_document_structure_caching()
        
        # Test full preprocessing result caching
        self.test_full_preprocessing_caching()
        
        # Test cache statistics
        self.test_cache_statistics()
        
        # Test cache performance
        self.test_cache_performance()
        
        # Print results
        self.print_test_results()
    
    def test_cache_key_generation(self):
        """Test cache key generation for different operations."""
        logger.info("\n🔑 Testing cache key generation...")
        
        try:
            text = self.sample_texts['simple_dialogue']
            
            # Test different cache key types
            keys = {
                'full_preprocessing': CacheKeyGenerator.generate_full_preprocessing_key(text),
                'spacy_processing': CacheKeyGenerator.generate_spacy_processing_key(text),
                'character_profile': CacheKeyGenerator.generate_character_profile_key(text),
                'pov_analysis': CacheKeyGenerator.generate_pov_analysis_key(text),
                'scene_break': CacheKeyGenerator.generate_scene_break_key(text),
                'document_structure': CacheKeyGenerator.generate_document_structure_key(text)
            }
            
            # Verify keys are unique and properly formatted
            for key_type, key in keys.items():
                assert key is not None, f"{key_type} key is None"
                assert len(key) > 10, f"{key_type} key is too short: {key}"
                assert '|' in key, f"{key_type} key missing separator: {key}"
                assert key.endswith('v:1.0'), f"{key_type} key missing version: {key}"
                logger.info(f"   ✓ {key_type}: {key[:30]}...")
            
            # Test key consistency
            key1 = CacheKeyGenerator.generate_full_preprocessing_key(text)
            key2 = CacheKeyGenerator.generate_full_preprocessing_key(text)
            assert key1 == key2, "Cache keys should be consistent for same input"
            
            # Test key uniqueness
            different_text = self.sample_texts['script_format']
            key3 = CacheKeyGenerator.generate_full_preprocessing_key(different_text)
            assert key1 != key3, "Cache keys should be different for different inputs"
            
            self.tests_passed += 1
            logger.info("   ✅ Cache key generation tests passed")
            
        except Exception as e:
            self.tests_failed += 1
            logger.error(f"   ❌ Cache key generation tests failed: {e}")
    
    def test_cache_hit_miss_behavior(self):
        """Test basic cache hit/miss behavior."""
        logger.info("\n🎯 Testing cache hit/miss behavior...")
        
        try:
            text = self.sample_texts['simple_dialogue']
            cache_key = CacheKeyGenerator.generate_full_preprocessing_key(text)
            
            # Test cache miss
            result = self.cache_manager.get_full_preprocessing_result(text)
            assert result is None, "Cache should be empty initially"
            
            # Cache a result
            test_result = {'test': 'data', 'cached': True}
            self.cache_manager.cache_full_preprocessing_result(text, test_result)
            
            # Test cache hit
            cached_result = self.cache_manager.get_full_preprocessing_result(text)
            assert cached_result is not None, "Cache should contain the result"
            assert cached_result['result']['test'] == 'data', "Cached result should match"
            
            # Test cache statistics
            stats = self.cache_manager.get_cache_stats()
            assert stats['full_preprocessing_hits'] > 0, "Should have cache hits"
            assert stats['full_preprocessing_misses'] > 0, "Should have cache misses"
            
            self.tests_passed += 1
            logger.info("   ✅ Cache hit/miss behavior tests passed")
            
        except Exception as e:
            self.tests_failed += 1
            logger.error(f"   ❌ Cache hit/miss behavior tests failed: {e}")
    
    def test_spacy_document_caching(self):
        """Test spaCy document caching functionality."""
        logger.info("\n🔬 Testing spaCy document caching...")
        
        try:
            text = self.sample_texts['simple_dialogue']
            
            # Test spaCy doc caching
            doc = self.preprocessor._get_spacy_doc_with_cache(text)
            assert doc is not None, "spaCy doc should not be None"
            
            # Test cache hit (should be faster second time)
            start_time = time.time()
            doc2 = self.preprocessor._get_spacy_doc_with_cache(text)
            cache_time = time.time() - start_time
            
            assert doc2 is not None, "Cached spaCy doc should not be None"
            assert cache_time < 0.1, f"Cached retrieval should be fast: {cache_time:.3f}s"
            
            # Test cache statistics
            stats = self.cache_manager.get_cache_stats()
            assert stats['spacy_cache_hits'] > 0, "Should have spaCy cache hits"
            
            self.tests_passed += 1
            logger.info("   ✅ spaCy document caching tests passed")
            
        except Exception as e:
            self.tests_failed += 1
            logger.error(f"   ❌ spaCy document caching tests failed: {e}")
    
    def test_character_profile_caching(self):
        """Test character profile caching functionality."""
        logger.info("\n👥 Testing character profile caching...")
        
        try:
            text = self.sample_texts['simple_dialogue']
            
            # Test character profile caching
            profiles1 = self.preprocessor._extract_character_profiles_with_cache(text, None)
            assert isinstance(profiles1, dict), "Character profiles should be a dictionary"
            
            # Test cache hit
            start_time = time.time()
            profiles2 = self.preprocessor._extract_character_profiles_with_cache(text, None)
            cache_time = time.time() - start_time
            
            assert isinstance(profiles2, dict), "Cached character profiles should be a dictionary"
            assert len(profiles1) == len(profiles2), "Profile counts should match"
            assert cache_time < 0.1, f"Cached retrieval should be fast: {cache_time:.3f}s"
            
            # Test cache statistics
            stats = self.cache_manager.get_cache_stats()
            assert stats['character_profile_hits'] > 0, "Should have character profile cache hits"
            
            self.tests_passed += 1
            logger.info("   ✅ Character profile caching tests passed")
            
        except Exception as e:
            self.tests_failed += 1
            logger.error(f"   ❌ Character profile caching tests failed: {e}")
    
    def test_pov_analysis_caching(self):
        """Test POV analysis caching functionality."""
        logger.info("\n📖 Testing POV analysis caching...")
        
        try:
            text = self.sample_texts['first_person']
            
            # Test POV analysis caching
            pov1 = self.preprocessor._analyze_pov_profile_with_cache(text)
            assert isinstance(pov1, dict), "POV analysis should be a dictionary"
            assert 'type' in pov1, "POV analysis should have a type"
            
            # Test cache hit
            start_time = time.time()
            pov2 = self.preprocessor._analyze_pov_profile_with_cache(text)
            cache_time = time.time() - start_time
            
            assert isinstance(pov2, dict), "Cached POV analysis should be a dictionary"
            assert pov1['type'] == pov2['type'], "POV types should match"
            assert cache_time < 0.1, f"Cached retrieval should be fast: {cache_time:.3f}s"
            
            # Test cache statistics
            stats = self.cache_manager.get_cache_stats()
            assert stats['pov_analysis_hits'] > 0, "Should have POV analysis cache hits"
            
            self.tests_passed += 1
            logger.info("   ✅ POV analysis caching tests passed")
            
        except Exception as e:
            self.tests_failed += 1
            logger.error(f"   ❌ POV analysis caching tests failed: {e}")
    
    def test_scene_break_caching(self):
        """Test scene break caching functionality."""
        logger.info("\n🎬 Testing scene break caching...")
        
        try:
            text = self.sample_texts['mixed_content']
            
            # Test scene break caching
            breaks1 = self.preprocessor._detect_scene_breaks_with_cache(text)
            assert isinstance(breaks1, list), "Scene breaks should be a list"
            
            # Test cache hit
            start_time = time.time()
            breaks2 = self.preprocessor._detect_scene_breaks_with_cache(text)
            cache_time = time.time() - start_time
            
            assert isinstance(breaks2, list), "Cached scene breaks should be a list"
            assert len(breaks1) == len(breaks2), "Scene break counts should match"
            assert cache_time < 0.1, f"Cached retrieval should be fast: {cache_time:.3f}s"
            
            # Test cache statistics
            stats = self.cache_manager.get_cache_stats()
            assert stats['scene_break_hits'] > 0, "Should have scene break cache hits"
            
            self.tests_passed += 1
            logger.info("   ✅ Scene break caching tests passed")
            
        except Exception as e:
            self.tests_failed += 1
            logger.error(f"   ❌ Scene break caching tests failed: {e}")
    
    def test_document_structure_caching(self):
        """Test document structure caching functionality."""
        logger.info("\n📄 Testing document structure caching...")
        
        try:
            text = self.sample_texts['mixed_content']
            
            # Test document structure caching
            structure1 = self.preprocessor._analyze_document_structure_with_cache(text)
            assert isinstance(structure1, dict), "Document structure should be a dictionary"
            assert 'total_length' in structure1, "Document structure should have total_length"
            
            # Test cache hit
            start_time = time.time()
            structure2 = self.preprocessor._analyze_document_structure_with_cache(text)
            cache_time = time.time() - start_time
            
            assert isinstance(structure2, dict), "Cached document structure should be a dictionary"
            assert structure1['total_length'] == structure2['total_length'], "Structures should match"
            assert cache_time < 0.1, f"Cached retrieval should be fast: {cache_time:.3f}s"
            
            # Test cache statistics
            stats = self.cache_manager.get_cache_stats()
            assert stats['document_structure_hits'] > 0, "Should have document structure cache hits"
            
            self.tests_passed += 1
            logger.info("   ✅ Document structure caching tests passed")
            
        except Exception as e:
            self.tests_failed += 1
            logger.error(f"   ❌ Document structure caching tests failed: {e}")
    
    def test_full_preprocessing_caching(self):
        """Test full preprocessing result caching."""
        logger.info("\n🔄 Testing full preprocessing caching...")
        
        try:
            text = self.sample_texts['simple_dialogue']
            
            # Clear cache to ensure fresh test
            self.preprocessor.cache_manager.clear_cache()
            
            # Test full preprocessing (should be cache miss)
            start_time = time.time()
            result1 = self.preprocessor.analyze(text)
            first_time = time.time() - start_time
            
            assert isinstance(result1, dict), "Preprocessing result should be a dictionary"
            assert 'dialogue_markers' in result1, "Result should have dialogue_markers"
            assert 'character_profiles' in result1, "Result should have character_profiles"
            
            # Test full preprocessing again (should be cache hit)
            start_time = time.time()
            result2 = self.preprocessor.analyze(text)
            second_time = time.time() - start_time
            
            assert isinstance(result2, dict), "Cached preprocessing result should be a dictionary"
            assert result1.keys() == result2.keys(), "Result keys should match"
            assert second_time < first_time, f"Cached call should be faster: {second_time:.3f}s vs {first_time:.3f}s"
            
            # Verify cache hit occurred
            stats = self.cache_manager.get_cache_stats()
            assert stats['full_preprocessing_hits'] > 0, "Should have full preprocessing cache hits"
            
            self.tests_passed += 1
            logger.info("   ✅ Full preprocessing caching tests passed")
            
        except Exception as e:
            self.tests_failed += 1
            logger.error(f"   ❌ Full preprocessing caching tests failed: {e}")
    
    def test_cache_statistics(self):
        """Test cache statistics functionality."""
        logger.info("\n📊 Testing cache statistics...")
        
        try:
            # Get cache statistics
            stats = self.cache_manager.get_cache_stats()
            
            # Verify required fields
            required_fields = [
                'enabled', 'total_requests', 'total_hits', 'hit_rate',
                'cache_size', 'max_cache_size', 'full_preprocessing_hits',
                'spacy_cache_hits', 'character_profile_hits', 'pov_analysis_hits',
                'scene_break_hits', 'document_structure_hits'
            ]
            
            for field in required_fields:
                assert field in stats, f"Statistics should have {field}"
                assert isinstance(stats[field], (int, float, bool)), f"{field} should be numeric or boolean"
            
            # Verify hit rate calculation
            if stats['total_requests'] > 0:
                expected_hit_rate = stats['total_hits'] / stats['total_requests']
                assert abs(stats['hit_rate'] - expected_hit_rate) < 0.001, "Hit rate calculation should be correct"
            
            self.tests_passed += 1
            logger.info("   ✅ Cache statistics tests passed")
            
        except Exception as e:
            self.tests_failed += 1
            logger.error(f"   ❌ Cache statistics tests failed: {e}")
    
    def test_cache_performance(self):
        """Test cache performance improvements."""
        logger.info("\n⚡ Testing cache performance...")
        
        try:
            text = self.sample_texts['mixed_content']
            
            # Clear cache
            self.preprocessor.cache_manager.clear_cache()
            
            # Time multiple operations without cache
            start_time = time.time()
            for i in range(3):
                self.preprocessor.analyze(text + f" {i}")  # Unique text each time
            no_cache_time = time.time() - start_time
            
            # Time operations with cache (same text)
            start_time = time.time()
            for i in range(3):
                self.preprocessor.analyze(text)  # Same text for cache hits
            cache_time = time.time() - start_time
            
            # Cache should provide significant speedup
            speedup = no_cache_time / cache_time if cache_time > 0 else 1
            logger.info(f"   Performance improvement: {speedup:.2f}x speedup")
            logger.info(f"   No cache: {no_cache_time:.3f}s, With cache: {cache_time:.3f}s")
            
            # Verify cache hits occurred
            stats = self.cache_manager.get_cache_stats()
            assert stats['total_hits'] > 0, "Should have cache hits for performance test"
            
            self.tests_passed += 1
            logger.info("   ✅ Cache performance tests passed")
            
        except Exception as e:
            self.tests_failed += 1
            logger.error(f"   ❌ Cache performance tests failed: {e}")
    
    def print_test_results(self):
        """Print final test results."""
        logger.info("\n" + "="*60)
        logger.info("PREPROCESSING CACHE TEST RESULTS")
        logger.info("="*60)
        
        total_tests = self.tests_passed + self.tests_failed
        success_rate = (self.tests_passed / total_tests) * 100 if total_tests > 0 else 0
        
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Tests passed: {self.tests_passed}")
        logger.info(f"Tests failed: {self.tests_failed}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        
        if self.tests_failed == 0:
            logger.info("🎉 All preprocessing cache tests passed!")
        else:
            logger.error(f"❌ {self.tests_failed} tests failed")
        
        # Display final cache statistics
        logger.info("\n📊 Final Cache Statistics:")
        stats = self.cache_manager.get_cache_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"   {key}: {value:.3f}")
            else:
                logger.info(f"   {key}: {value}")


class MockSpacyModel:
    """Mock spaCy model for testing when spaCy is not available."""
    
    def __init__(self):
        self.meta = {'name': 'mock_model'}
    
    def __call__(self, text):
        return MockDoc(text)
    
    def from_bytes(self, data):
        return MockDoc("cached_doc")
    
    def to_bytes(self):
        return b"mock_doc_bytes"


class MockDoc:
    """Mock spaCy Doc object for testing."""
    
    def __init__(self, text):
        self.text = text
        self.ents = []
        self.sents = []
    
    def to_bytes(self):
        return b"mock_doc_bytes"


def main():
    """Run the preprocessing cache tests."""
    test_suite = PreprocessingCacheTest()
    test_suite.run_all_tests()
    
    # Return exit code based on test results
    return 0 if test_suite.tests_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())