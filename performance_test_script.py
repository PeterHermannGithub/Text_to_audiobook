#!/usr/bin/env python3
"""
Performance testing script for text-to-audiobook system
Tests various configurations and measures resource usage
"""
import subprocess
import time
import json
import sys
import os

def run_test(test_name, command, timeout=300):
    """Run a test and measure performance"""
    print(f"\n=== {test_name} ===")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Duration: {duration:.2f} seconds")
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout[-1000:])  # Last 1000 chars
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr[-1000:])  # Last 1000 chars
            
        return {
            'test_name': test_name,
            'duration': duration,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.TimeoutExpired:
        print(f"Test timed out after {timeout} seconds")
        return {
            'test_name': test_name,
            'duration': timeout,
            'return_code': -1,
            'timeout': True
        }
    except Exception as e:
        print(f"Test failed with error: {e}")
        return {
            'test_name': test_name,
            'duration': 0,
            'return_code': -1,
            'error': str(e)
        }

def get_resource_usage():
    """Get current resource usage of containers"""
    try:
        result = subprocess.run(
            'docker stats --no-stream --format "{{.Container}},{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}}"',
            shell=True,
            capture_output=True,
            text=True
        )
        return result.stdout
    except Exception as e:
        return f"Error getting resource usage: {e}"

def main():
    """Main performance testing function"""
    results = []
    
    # Test files to use
    test_files = [
        'PrideandPrejudice_small.txt',
        'PrideandPrejudice_medium.txt'
    ]
    
    # Configuration parameters to test
    chunk_sizes = [500, 1000, 2000, 4000]
    quality_thresholds = [0.7, 0.8, 0.9]
    
    print("=== PERFORMANCE TESTING STARTING ===")
    print("Getting initial resource usage...")
    print(get_resource_usage())
    
    # Test 1: Basic text extraction performance
    for test_file in test_files:
        test_name = f"Text Extraction - {test_file}"
        command = f'docker exec text_to_audiobook_app python -c "from src.text_processing.text_extractor import TextExtractor; extractor = TextExtractor(); text = extractor.extract(\'/app/input/{test_file}\'); print(f\'Extracted {{len(text)}} characters\')"'
        result = run_test(test_name, command, timeout=60)
        results.append(result)
    
    # Test 2: Rule-based attribution performance (no LLM required)
    for test_file in test_files:
        test_name = f"Rule-based Attribution - {test_file}"
        command = f'docker exec text_to_audiobook_app python -c "from src.attribution.rule_based_attributor import RuleBasedAttributor; from src.text_processing.text_extractor import TextExtractor; extractor = TextExtractor(); text = extractor.extract(\'/app/input/{test_file}\'); attributor = RuleBasedAttributor(); segments = text.split(\'\\n\'); result = attributor.attribute_segments(segments); print(f\'Processed {{len(segments)}} segments, {{len([s for s in result if s[\\\"confidence\\\"] > 0.8])}} high-confidence attributions\')"'
        result = run_test(test_name, command, timeout=120)
        results.append(result)
    
    # Test 3: Chunk size impact testing
    for chunk_size in chunk_sizes:
        test_name = f"Chunk Size {chunk_size} - Small File"
        command = f'docker exec text_to_audiobook_app python app.py input/PrideandPrejudice_small.txt --chunk-size {chunk_size} --skip-voice-casting --debug-llm'
        result = run_test(test_name, command, timeout=180)
        results.append(result)
    
    # Test 4: Quality threshold testing
    for quality_threshold in quality_thresholds:
        test_name = f"Quality Threshold {quality_threshold} - Small File"
        command = f'docker exec text_to_audiobook_app python app.py input/PrideandPrejudice_small.txt --quality-threshold {quality_threshold} --skip-voice-casting'
        result = run_test(test_name, command, timeout=180)
        results.append(result)
    
    print("\n=== FINAL RESOURCE USAGE ===")
    print(get_resource_usage())
    
    # Save results to file
    with open('/mnt/c/Dev/Projects/text_to_audiobook/performance_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n=== PERFORMANCE TESTING COMPLETED ===")
    print("Results saved to performance_test_results.json")
    
    # Summary
    print("\n=== SUMMARY ===")
    for result in results:
        if 'duration' in result:
            print(f"{result['test_name']}: {result['duration']:.2f}s (RC: {result['return_code']})")

if __name__ == "__main__":
    main()