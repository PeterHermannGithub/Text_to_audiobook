#!/usr/bin/env python3
"""
Simple test script for preprocessing cache functionality.

This script validates core cache functionality without requiring spaCy.
"""

import os
import sys
import time
import json

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from text_processing.preprocessing_cache import PreprocessingCacheManager
from attribution.cache_keys import CacheKeyGenerator

class SimpleCacheTest:
    """Simple test suite for preprocessing cache functionality."""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.cache_manager = PreprocessingCacheManager()
        
    def run_tests(self):
        """Run all cache tests."""
        print("="*60)
        print("PREPROCESSING CACHE VALIDATION TEST")
        print("="*60)
        
        self.test_cache_key_generation()
        self.test_basic_cache_operations()
        self.test_cache_statistics()
        self.test_cache_performance()
        
        self.print_results()
        
    def test_cache_key_generation(self):
        """Test cache key generation."""
        print("\n🔑 Testing cache key generation...")
        
        try:
            text = "This is a test text for cache key generation."
            
            # Test different cache key types
            keys = {
                'full_preprocessing': CacheKeyGenerator.generate_full_preprocessing_key(text),
                'spacy_processing': CacheKeyGenerator.generate_spacy_processing_key(text),
                'character_profile': CacheKeyGenerator.generate_character_profile_key(text),
                'pov_analysis': CacheKeyGenerator.generate_pov_analysis_key(text),
                'scene_break': CacheKeyGenerator.generate_scene_break_key(text),
                'document_structure': CacheKeyGenerator.generate_document_structure_key(text)
            }
            
            for key_type, key in keys.items():
                assert key is not None, f"{key_type} key is None"
                assert len(key) > 10, f"{key_type} key too short"
                assert '|' in key, f"{key_type} key missing separator"
                print(f"   ✓ {key_type}: {key[:40]}...")
                
            # Test consistency
            key1 = CacheKeyGenerator.generate_full_preprocessing_key(text)
            key2 = CacheKeyGenerator.generate_full_preprocessing_key(text)
            assert key1 == key2, "Keys should be consistent"
            
            self.tests_passed += 1
            print("   ✅ Cache key generation tests passed")
            
        except Exception as e:
            self.tests_failed += 1
            print(f"   ❌ Cache key generation failed: {e}")
    
    def test_basic_cache_operations(self):
        """Test basic cache operations."""
        print("\n💾 Testing basic cache operations...")
        
        try:
            text = "Sample text for testing cache operations."
            
            # Test cache miss
            result = self.cache_manager.get_full_preprocessing_result(text)
            assert result is None, "Cache should be empty initially"
            
            # Test caching
            test_data = {'test': 'data', 'timestamp': time.time()}
            self.cache_manager.cache_full_preprocessing_result(text, test_data)
            
            # Test cache hit
            cached_result = self.cache_manager.get_full_preprocessing_result(text)
            assert cached_result is not None, "Cache should contain result"
            assert cached_result['result']['test'] == 'data', "Cached data should match"
            
            # Test different cache types
            test_profiles = [{'name': 'Alice', 'confidence': 0.9}]
            self.cache_manager.cache_character_profiles(text, test_profiles)
            
            profiles = self.cache_manager.get_character_profiles(text)
            assert profiles is not None, "Character profiles should be cached"
            assert len(profiles) == 1, "Should have one profile"
            
            # Test POV analysis
            pov_data = {'type': 'FIRST_PERSON', 'confidence': 0.8}
            self.cache_manager.cache_pov_analysis(text, pov_data)
            
            pov_result = self.cache_manager.get_pov_analysis(text)
            assert pov_result is not None, "POV analysis should be cached"
            assert pov_result['type'] == 'FIRST_PERSON', "POV type should match"
            
            # Test scene breaks
            scene_breaks = [100, 200, 300]
            self.cache_manager.cache_scene_breaks(text, scene_breaks)
            
            cached_breaks = self.cache_manager.get_scene_breaks(text)
            assert cached_breaks is not None, "Scene breaks should be cached"
            assert len(cached_breaks) == 3, "Should have three scene breaks"
            
            # Test document structure
            doc_structure = {'total_length': 100, 'paragraph_count': 5}
            self.cache_manager.cache_document_structure(text, doc_structure)
            
            cached_structure = self.cache_manager.get_document_structure(text)
            assert cached_structure is not None, "Document structure should be cached"
            assert cached_structure['total_length'] == 100, "Structure should match"
            
            self.tests_passed += 1
            print("   ✅ Basic cache operations tests passed")
            
        except Exception as e:
            self.tests_failed += 1
            print(f"   ❌ Basic cache operations failed: {e}")
    
    def test_cache_statistics(self):
        """Test cache statistics."""
        print("\n📊 Testing cache statistics...")
        
        try:
            stats = self.cache_manager.get_cache_stats()
            
            # Check required fields
            required_fields = [
                'enabled', 'total_requests', 'total_hits', 'hit_rate',
                'cache_size', 'max_cache_size'
            ]
            
            for field in required_fields:
                assert field in stats, f"Missing field: {field}"
                
            # Verify we have some activity
            assert stats['total_requests'] > 0, "Should have cache requests"
            assert stats['total_hits'] > 0, "Should have cache hits"
            assert 0 <= stats['hit_rate'] <= 1, "Hit rate should be between 0 and 1"
            
            print(f"   📈 Cache statistics:")
            print(f"      Total requests: {stats['total_requests']}")
            print(f"      Total hits: {stats['total_hits']}")
            print(f"      Hit rate: {stats['hit_rate']:.2%}")
            print(f"      Cache size: {stats['cache_size']}")
            
            self.tests_passed += 1
            print("   ✅ Cache statistics tests passed")
            
        except Exception as e:
            self.tests_failed += 1
            print(f"   ❌ Cache statistics failed: {e}")
    
    def test_cache_performance(self):
        """Test cache performance."""
        print("\n⚡ Testing cache performance...")
        
        try:
            # Clear cache
            self.cache_manager.clear_cache()
            
            # Test data
            test_data = {'performance': 'test', 'data': list(range(100))}
            text = "Performance test text with some content."
            
            # Test cache miss timing
            start_time = time.time()
            result = self.cache_manager.get_full_preprocessing_result(text)
            miss_time = time.time() - start_time
            assert result is None, "Should be cache miss"
            
            # Cache the data
            self.cache_manager.cache_full_preprocessing_result(text, test_data)
            
            # Test cache hit timing
            start_time = time.time()
            cached_result = self.cache_manager.get_full_preprocessing_result(text)
            hit_time = time.time() - start_time
            
            assert cached_result is not None, "Should be cache hit"
            assert hit_time < 0.01, f"Cache hit should be fast: {hit_time:.4f}s"
            
            print(f"   ⏱️  Cache miss time: {miss_time:.4f}s")
            print(f"   ⏱️  Cache hit time: {hit_time:.4f}s")
            print(f"   🚀 Performance improvement: {miss_time/hit_time:.1f}x faster")
            
            self.tests_passed += 1
            print("   ✅ Cache performance tests passed")
            
        except Exception as e:
            self.tests_failed += 1
            print(f"   ❌ Cache performance failed: {e}")
    
    def print_results(self):
        """Print test results."""
        print("\n" + "="*60)
        print("CACHE VALIDATION TEST RESULTS")
        print("="*60)
        
        total_tests = self.tests_passed + self.tests_failed
        success_rate = (self.tests_passed / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"Total tests: {total_tests}")
        print(f"Tests passed: {self.tests_passed}")
        print(f"Tests failed: {self.tests_failed}")
        print(f"Success rate: {success_rate:.1f}%")
        
        if self.tests_failed == 0:
            print("🎉 All cache validation tests passed!")
            print("✅ Preprocessing cache is working correctly!")
        else:
            print(f"❌ {self.tests_failed} tests failed")
            
        # Final cache statistics
        print("\n📊 Final Cache Statistics:")
        stats = self.cache_manager.get_cache_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
        return self.tests_failed == 0

def main():
    """Run the cache validation tests."""
    test_suite = SimpleCacheTest()
    success = test_suite.run_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())