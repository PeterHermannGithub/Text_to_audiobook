#!/usr/bin/env python3
"""
Test Runner for Text_to_audiobook Quality Validation Suite

This script runs comprehensive tests to validate the quality improvements
implemented in January 2025, including regression testing and performance
benchmarking.

Usage:
    python tests/run_tests.py [--verbose] [--performance] [--regression] [--all]
"""

import unittest
import sys
import os
import argparse
import time
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from test_ground_truth import GroundTruthTestSuite

class TestRunner:
    """Comprehensive test runner with reporting capabilities."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'skipped_tests': 0,
            'execution_time': 0.0,
            'test_details': []
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites and return comprehensive results."""
        print("🚀 Starting Text_to_audiobook Quality Validation Test Suite")
        print("="*70)
        
        start_time = time.time()
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(GroundTruthTestSuite)
        
        # Create custom test runner with detailed results
        runner = unittest.TextTestRunner(
            verbosity=2 if self.verbose else 1,
            stream=sys.stdout,
            buffer=True
        )
        
        # Run tests
        result = runner.run(suite)
        
        # Process results
        self.results['total_tests'] = result.testsRun
        self.results['passed_tests'] = result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
        self.results['failed_tests'] = len(result.failures) + len(result.errors)
        self.results['skipped_tests'] = len(result.skipped)
        self.results['execution_time'] = time.time() - start_time
        
        # Add detailed failure information
        for test, traceback in result.failures:
            self.results['test_details'].append({
                'test': str(test),
                'status': 'FAILED',
                'error': traceback
            })
        
        for test, traceback in result.errors:
            self.results['test_details'].append({
                'test': str(test),
                'status': 'ERROR',
                'error': traceback
            })
        
        for test, reason in result.skipped:
            self.results['test_details'].append({
                'test': str(test),
                'status': 'SKIPPED',
                'error': reason
            })
        
        return self.results
    
    def run_regression_tests(self) -> Dict[str, Any]:
        """Run only regression tests to verify quality improvements."""
        print("🔄 Running Regression Tests for Quality Improvements")
        print("="*50)
        
        # Create targeted test suite for regression
        suite = unittest.TestSuite()
        
        # Add specific regression tests
        suite.addTest(GroundTruthTestSuite('test_quality_validation_system'))
        suite.addTest(GroundTruthTestSuite('test_rule_based_attribution'))
        suite.addTest(GroundTruthTestSuite('test_unfixable_recovery_system'))
        suite.addTest(GroundTruthTestSuite('test_output_formatting_cleanup'))
        suite.addTest(GroundTruthTestSuite('test_regression_quality_thresholds'))
        
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)
        
        return self._process_results(result)
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance benchmark tests."""
        print("⚡ Running Performance Benchmark Tests")
        print("="*40)
        
        suite = unittest.TestSuite()
        suite.addTest(GroundTruthTestSuite('test_performance_benchmarks'))
        
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)
        
        return self._process_results(result)
    
    def _process_results(self, result) -> Dict[str, Any]:
        """Process unittest results into summary dictionary."""
        return {
            'total_tests': result.testsRun,
            'passed_tests': result.testsRun - len(result.failures) - len(result.errors),
            'failed_tests': len(result.failures) + len(result.errors),
            'skipped_tests': len(result.skipped),
            'success_rate': ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
        }
    
    def print_summary(self, results: Dict[str, Any]):
        """Print comprehensive test summary."""
        print("\n" + "="*70)
        print("📊 TEST EXECUTION SUMMARY")
        print("="*70)
        
        print(f"Total Tests Run: {results['total_tests']}")
        print(f"✅ Passed: {results['passed_tests']}")
        print(f"❌ Failed: {results['failed_tests']}")
        print(f"⏭️  Skipped: {results['skipped_tests']}")
        
        if 'execution_time' in results:
            print(f"⏱️  Execution Time: {results['execution_time']:.2f} seconds")
        
        success_rate = (results['passed_tests'] / results['total_tests'] * 100) if results['total_tests'] > 0 else 0
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        # Quality assessment
        if success_rate >= 95:
            print("🎉 EXCELLENT: Quality improvements are working perfectly!")
        elif success_rate >= 85:
            print("✅ GOOD: Most quality improvements are working well.")
        elif success_rate >= 70:
            print("⚠️  WARNING: Some quality improvements need attention.")
        else:
            print("🚨 CRITICAL: Quality improvements require immediate review.")
        
        # Print detailed failures if any
        if results.get('test_details'):
            print("\n" + "="*70)
            print("📋 DETAILED TEST RESULTS")
            print("="*70)
            
            for detail in results['test_details']:
                if detail['status'] in ['FAILED', 'ERROR']:
                    print(f"\n❌ {detail['test']}")
                    print(f"   Status: {detail['status']}")
                    if self.verbose:
                        print(f"   Error: {detail['error'][:200]}...")
                elif detail['status'] == 'SKIPPED':
                    print(f"\n⏭️  {detail['test']} (SKIPPED: {detail['error']})")


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description='Text_to_audiobook Quality Test Runner')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose output')
    parser.add_argument('--performance', '-p', action='store_true',
                       help='Run only performance tests')
    parser.add_argument('--regression', '-r', action='store_true',
                       help='Run only regression tests')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Run all tests (default)')
    
    args = parser.parse_args()
    
    # Default to all tests if no specific test type specified
    if not any([args.performance, args.regression]):
        args.all = True
    
    runner = TestRunner(verbose=args.verbose)
    
    try:
        if args.all:
            results = runner.run_all_tests()
        elif args.regression:
            results = runner.run_regression_tests()
        elif args.performance:
            results = runner.run_performance_tests()
        
        runner.print_summary(results)
        
        # Exit with appropriate code
        if results['failed_tests'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Test execution interrupted by user.")
        sys.exit(2)
    except Exception as e:
        print(f"\n\n💥 Test execution failed with error: {e}")
        sys.exit(3)


if __name__ == '__main__':
    main()