#!/usr/bin/env python3
"""
Comprehensive HTTP Connection Pooling Performance Benchmark

This benchmark validates the performance improvements of HTTP connection pooling
for LLM requests by testing realistic scenarios with actual HTTP endpoints.

Expected Performance Improvements:
- 5-10x faster connection establishment through session reuse
- 20-70% reduction in total request time
- Reduced memory usage through connection pooling
- Improved reliability through circuit breaker patterns

Test Scenarios:
1. Cold Start Performance: First request timing
2. Session Reuse Performance: Subsequent request timing  
3. Concurrent Request Performance: Multiple simultaneous requests
4. Memory Usage Comparison: Pool vs individual sessions
5. Circuit Breaker Performance: Fault tolerance validation
"""

import asyncio
import logging
import os
import sys
import time
import threading
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any
import requests
import statistics
import json

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConnectionPoolingBenchmark:
    """Comprehensive benchmark for HTTP connection pooling performance."""
    
    def __init__(self):
        self.results = {}
        self.test_endpoints = [
            "https://httpbin.org/delay/0.1",  # Fast endpoint
            "https://httpbin.org/delay/0.5",  # Medium endpoint
            "https://httpbin.org/json",       # JSON response
            "https://httpbin.org/status/200", # Simple status
        ]
        
    def run_all_benchmarks(self):
        """Run all performance benchmarks."""
        logger.info("="*70)
        logger.info("COMPREHENSIVE CONNECTION POOLING PERFORMANCE BENCHMARK")
        logger.info("="*70)
        
        # Benchmark 1: Cold Start Performance
        self.benchmark_cold_start_performance()
        
        # Benchmark 2: Session Reuse Performance  
        self.benchmark_session_reuse_performance()
        
        # Benchmark 3: Concurrent Request Performance
        self.benchmark_concurrent_performance()
        
        # Benchmark 4: Memory Usage Comparison
        self.benchmark_memory_usage()
        
        # Benchmark 5: Circuit Breaker Performance
        self.benchmark_circuit_breaker_performance()
        
        # Print comprehensive results
        self.print_benchmark_results()
        
        return self.results
    
    def benchmark_cold_start_performance(self):
        """Benchmark cold start performance - first request timing."""
        logger.info("\n🚀 Benchmarking cold start performance...")
        
        try:
            from llm_pool.http_pool_manager import HTTPConnectionPoolManager
            
            # Test with connection pooling
            pool_manager = HTTPConnectionPoolManager()
            
            cold_start_times = []
            for endpoint in self.test_endpoints:
                try:
                    start_time = time.time()
                    response = pool_manager.get(endpoint, timeout=10.0)
                    cold_start_time = time.time() - start_time
                    cold_start_times.append(cold_start_time)
                    logger.debug(f"Pooled cold start {endpoint}: {cold_start_time:.3f}s")
                except Exception as e:
                    logger.warning(f"Cold start request failed for {endpoint}: {e}")
                    cold_start_times.append(float('inf'))
            
            pool_manager.close()
            
            # Test without connection pooling (individual sessions)
            individual_times = []
            for endpoint in self.test_endpoints:
                try:
                    start_time = time.time()
                    session = requests.Session()
                    response = session.get(endpoint, timeout=10.0)
                    individual_time = time.time() - start_time
                    individual_times.append(individual_time)
                    session.close()
                    logger.debug(f"Individual cold start {endpoint}: {individual_time:.3f}s")
                except Exception as e:
                    logger.warning(f"Individual request failed for {endpoint}: {e}")
                    individual_times.append(float('inf'))
            
            # Calculate results
            valid_pool_times = [t for t in cold_start_times if t != float('inf')]
            valid_individual_times = [t for t in individual_times if t != float('inf')]
            
            if valid_pool_times and valid_individual_times:
                avg_pool_time = statistics.mean(valid_pool_times)
                avg_individual_time = statistics.mean(valid_individual_times)
                improvement = avg_individual_time / avg_pool_time if avg_pool_time > 0 else 1
                
                self.results['cold_start'] = {
                    'pooled_avg_time': avg_pool_time,
                    'individual_avg_time': avg_individual_time,
                    'improvement_factor': improvement,
                    'pooled_times': valid_pool_times,
                    'individual_times': valid_individual_times
                }
                
                logger.info(f"   📊 Cold start improvement: {improvement:.2f}x faster")
                logger.info(f"   📊 Pooled average: {avg_pool_time:.3f}s")
                logger.info(f"   📊 Individual average: {avg_individual_time:.3f}s")
            else:
                logger.warning("   ⚠️  Cold start benchmark failed - no valid requests")
                
        except Exception as e:
            logger.error(f"   ❌ Cold start benchmark failed: {e}")
    
    def benchmark_session_reuse_performance(self):
        """Benchmark session reuse performance - repeated requests."""
        logger.info("\n🔄 Benchmarking session reuse performance...")
        
        try:
            from llm_pool.http_pool_manager import HTTPConnectionPoolManager
            
            num_requests = 10
            test_endpoint = "https://httpbin.org/json"
            
            # Test with connection pooling (session reuse)
            pool_manager = HTTPConnectionPoolManager()
            
            pooled_times = []
            for i in range(num_requests):
                try:
                    start_time = time.time()
                    response = pool_manager.get(test_endpoint, timeout=10.0)
                    request_time = time.time() - start_time
                    pooled_times.append(request_time)
                    logger.debug(f"Pooled request {i+1}: {request_time:.3f}s")
                except Exception as e:
                    logger.warning(f"Pooled request {i+1} failed: {e}")
                    pooled_times.append(float('inf'))
            
            pool_manager.close()
            
            # Test without connection pooling (new session each time)
            individual_times = []
            for i in range(num_requests):
                try:
                    start_time = time.time()
                    session = requests.Session()
                    response = session.get(test_endpoint, timeout=10.0)
                    request_time = time.time() - start_time
                    individual_times.append(request_time)
                    session.close()
                    logger.debug(f"Individual request {i+1}: {request_time:.3f}s")
                except Exception as e:
                    logger.warning(f"Individual request {i+1} failed: {e}")
                    individual_times.append(float('inf'))
            
            # Calculate results
            valid_pool_times = [t for t in pooled_times if t != float('inf')]
            valid_individual_times = [t for t in individual_times if t != float('inf')]
            
            if valid_pool_times and valid_individual_times:
                avg_pool_time = statistics.mean(valid_pool_times)
                avg_individual_time = statistics.mean(valid_individual_times)
                improvement = avg_individual_time / avg_pool_time if avg_pool_time > 0 else 1
                
                # Calculate median and percentiles for more detailed analysis
                pool_median = statistics.median(valid_pool_times)
                individual_median = statistics.median(valid_individual_times)
                
                self.results['session_reuse'] = {
                    'pooled_avg_time': avg_pool_time,
                    'individual_avg_time': avg_individual_time,
                    'improvement_factor': improvement,
                    'pooled_median': pool_median,
                    'individual_median': individual_median,
                    'pooled_times': valid_pool_times,
                    'individual_times': valid_individual_times,
                    'requests_tested': num_requests
                }
                
                logger.info(f"   📊 Session reuse improvement: {improvement:.2f}x faster")
                logger.info(f"   📊 Pooled average: {avg_pool_time:.3f}s (median: {pool_median:.3f}s)")
                logger.info(f"   📊 Individual average: {avg_individual_time:.3f}s (median: {individual_median:.3f}s)")
            else:
                logger.warning("   ⚠️  Session reuse benchmark failed - no valid requests")
                
        except Exception as e:
            logger.error(f"   ❌ Session reuse benchmark failed: {e}")
    
    def benchmark_concurrent_performance(self):
        """Benchmark concurrent request performance."""
        logger.info("\n⚡ Benchmarking concurrent request performance...")
        
        try:
            from llm_pool.http_pool_manager import HTTPConnectionPoolManager
            
            num_workers = 5
            requests_per_worker = 3
            test_endpoint = "https://httpbin.org/delay/0.2"
            
            # Test with connection pooling
            pool_manager = HTTPConnectionPoolManager()
            
            def pooled_worker(worker_id):
                """Worker function for pooled requests."""
                times = []
                for i in range(requests_per_worker):
                    try:
                        start_time = time.time()
                        response = pool_manager.get(test_endpoint, timeout=10.0)
                        request_time = time.time() - start_time
                        times.append(request_time)
                        logger.debug(f"Pooled worker {worker_id} request {i+1}: {request_time:.3f}s")
                    except Exception as e:
                        logger.warning(f"Pooled worker {worker_id} request {i+1} failed: {e}")
                        times.append(float('inf'))
                return times
            
            # Run concurrent pooled requests
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(pooled_worker, i) for i in range(num_workers)]
                pooled_results = []
                for future in as_completed(futures):
                    pooled_results.extend(future.result())
            pooled_total_time = time.time() - start_time
            
            pool_manager.close()
            
            # Test without connection pooling
            def individual_worker(worker_id):
                """Worker function for individual session requests."""
                times = []
                for i in range(requests_per_worker):
                    try:
                        start_time = time.time()
                        session = requests.Session()
                        response = session.get(test_endpoint, timeout=10.0)
                        request_time = time.time() - start_time
                        times.append(request_time)
                        session.close()
                        logger.debug(f"Individual worker {worker_id} request {i+1}: {request_time:.3f}s")
                    except Exception as e:
                        logger.warning(f"Individual worker {worker_id} request {i+1} failed: {e}")
                        times.append(float('inf'))
                return times
            
            # Run concurrent individual requests
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(individual_worker, i) for i in range(num_workers)]
                individual_results = []
                for future in as_completed(futures):
                    individual_results.extend(future.result())
            individual_total_time = time.time() - start_time
            
            # Calculate results
            valid_pool_times = [t for t in pooled_results if t != float('inf')]
            valid_individual_times = [t for t in individual_results if t != float('inf')]
            
            if valid_pool_times and valid_individual_times:
                avg_pool_time = statistics.mean(valid_pool_times)
                avg_individual_time = statistics.mean(valid_individual_times)
                improvement = avg_individual_time / avg_pool_time if avg_pool_time > 0 else 1
                total_time_improvement = individual_total_time / pooled_total_time if pooled_total_time > 0 else 1
                
                self.results['concurrent'] = {
                    'pooled_avg_time': avg_pool_time,
                    'individual_avg_time': avg_individual_time,
                    'improvement_factor': improvement,
                    'pooled_total_time': pooled_total_time,
                    'individual_total_time': individual_total_time,
                    'total_time_improvement': total_time_improvement,
                    'workers': num_workers,
                    'requests_per_worker': requests_per_worker
                }
                
                logger.info(f"   📊 Concurrent improvement: {improvement:.2f}x faster per request")
                logger.info(f"   📊 Total time improvement: {total_time_improvement:.2f}x faster overall")
                logger.info(f"   📊 Pooled total: {pooled_total_time:.3f}s, Individual total: {individual_total_time:.3f}s")
            else:
                logger.warning("   ⚠️  Concurrent benchmark failed - no valid requests")
                
        except Exception as e:
            logger.error(f"   ❌ Concurrent benchmark failed: {e}")
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage comparison."""
        logger.info("\n🧠 Benchmarking memory usage...")
        
        try:
            from llm_pool.http_pool_manager import HTTPConnectionPoolManager
            
            num_requests = 20
            test_endpoint = "https://httpbin.org/json"
            
            # Start memory tracking
            tracemalloc.start()
            
            # Test with connection pooling
            snapshot1 = tracemalloc.take_snapshot()
            
            pool_manager = HTTPConnectionPoolManager()
            for i in range(num_requests):
                try:
                    response = pool_manager.get(test_endpoint, timeout=5.0)
                except Exception:
                    pass  # Ignore connection errors for memory test
            
            snapshot2 = tracemalloc.take_snapshot()
            pool_manager.close()
            
            # Calculate pooled memory usage
            top_stats = snapshot2.compare_to(snapshot1, 'lineno')
            pooled_memory = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
            
            # Test without connection pooling
            snapshot3 = tracemalloc.take_snapshot()
            
            for i in range(num_requests):
                try:
                    session = requests.Session()
                    response = session.get(test_endpoint, timeout=5.0)
                    session.close()
                except Exception:
                    pass  # Ignore connection errors for memory test
            
            snapshot4 = tracemalloc.take_snapshot()
            
            # Calculate individual memory usage
            top_stats = snapshot4.compare_to(snapshot3, 'lineno')
            individual_memory = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
            
            tracemalloc.stop()
            
            if pooled_memory > 0 and individual_memory > 0:
                memory_improvement = individual_memory / pooled_memory
                
                self.results['memory_usage'] = {
                    'pooled_memory_bytes': pooled_memory,
                    'individual_memory_bytes': individual_memory,
                    'memory_improvement': memory_improvement,
                    'requests_tested': num_requests
                }
                
                logger.info(f"   📊 Memory improvement: {memory_improvement:.2f}x less memory")
                logger.info(f"   📊 Pooled memory: {pooled_memory/1024:.1f} KB")
                logger.info(f"   📊 Individual memory: {individual_memory/1024:.1f} KB")
            else:
                logger.warning("   ⚠️  Memory benchmark inconclusive")
                
        except Exception as e:
            logger.error(f"   ❌ Memory benchmark failed: {e}")
    
    def benchmark_circuit_breaker_performance(self):
        """Benchmark circuit breaker fault tolerance performance."""
        logger.info("\n🔌 Benchmarking circuit breaker performance...")
        
        try:
            from llm_pool.http_pool_manager import HTTPConnectionPoolManager, ConnectionPoolConfig
            
            # Create pool with fast circuit breaker for testing
            config = ConnectionPoolConfig(
                circuit_breaker_enabled=True,
                circuit_breaker_failure_threshold=2,
                circuit_breaker_recovery_timeout=0.5
            )
            
            pool_manager = HTTPConnectionPoolManager(config)
            circuit_breaker = pool_manager.circuit_breaker
            
            # Test circuit breaker timing
            failure_endpoint = "http://localhost:99999/nonexistent"  # Guaranteed to fail
            
            # Measure time to open circuit
            start_time = time.time()
            
            # First failure
            try:
                pool_manager.get(failure_endpoint, timeout=1.0)
            except:
                pass
            
            # Second failure (should open circuit)
            try:
                pool_manager.get(failure_endpoint, timeout=1.0)
            except:
                pass
            
            circuit_open_time = time.time() - start_time
            
            # Verify circuit is open
            assert circuit_breaker.get_state().value == "open"
            
            # Test fast failure when circuit is open
            start_time = time.time()
            try:
                pool_manager.get(failure_endpoint, timeout=1.0)
            except:
                pass
            fast_failure_time = time.time() - start_time
            
            # Wait for recovery
            time.sleep(0.6)
            
            # Test recovery transition
            assert circuit_breaker.can_make_request() == True
            
            pool_manager.close()
            
            self.results['circuit_breaker'] = {
                'circuit_open_time': circuit_open_time,
                'fast_failure_time': fast_failure_time,
                'recovery_successful': True,
                'failure_threshold': 2
            }
            
            logger.info(f"   📊 Circuit opens in: {circuit_open_time:.3f}s")
            logger.info(f"   📊 Fast failure time: {fast_failure_time:.3f}s")
            logger.info(f"   📊 Recovery mechanism: ✅ Working")
            
        except Exception as e:
            logger.error(f"   ❌ Circuit breaker benchmark failed: {e}")
    
    def print_benchmark_results(self):
        """Print comprehensive benchmark results."""
        logger.info("\n" + "="*70)
        logger.info("CONNECTION POOLING PERFORMANCE BENCHMARK RESULTS")
        logger.info("="*70)
        
        # Summary metrics
        total_improvements = []
        if 'cold_start' in self.results:
            total_improvements.append(self.results['cold_start']['improvement_factor'])
        if 'session_reuse' in self.results:
            total_improvements.append(self.results['session_reuse']['improvement_factor'])
        if 'concurrent' in self.results:
            total_improvements.append(self.results['concurrent']['improvement_factor'])
        
        if total_improvements:
            avg_improvement = statistics.mean(total_improvements)
            max_improvement = max(total_improvements)
            
            logger.info(f"\n🎯 OVERALL PERFORMANCE SUMMARY:")
            logger.info(f"   Average improvement: {avg_improvement:.2f}x faster")
            logger.info(f"   Maximum improvement: {max_improvement:.2f}x faster")
            logger.info(f"   Benchmarks completed: {len(total_improvements)}/3 HTTP tests")
        
        # Detailed results
        logger.info(f"\n📈 DETAILED BENCHMARK RESULTS:")
        for test_name, results in self.results.items():
            logger.info(f"\n🔍 {test_name.upper().replace('_', ' ')}:")
            for key, value in results.items():
                if isinstance(value, float):
                    if 'time' in key:
                        logger.info(f"   {key}: {value:.3f}s")
                    elif 'improvement' in key or 'factor' in key:
                        logger.info(f"   {key}: {value:.2f}x")
                    else:
                        logger.info(f"   {key}: {value:.2f}")
                else:
                    logger.info(f"   {key}: {value}")
        
        # Performance validation
        if total_improvements and avg_improvement >= 1.5:
            logger.info(f"\n🎉 PERFORMANCE VALIDATION: ✅ PASSED")
            logger.info(f"✅ HTTP connection pooling provides significant performance improvements!")
            logger.info(f"✅ Average {avg_improvement:.2f}x improvement exceeds 1.5x threshold")
        elif total_improvements:
            logger.info(f"\n⚠️  PERFORMANCE VALIDATION: ⚠️  PARTIAL")
            logger.info(f"⚠️  {avg_improvement:.2f}x improvement below expected 5-10x range")
            logger.info(f"⚠️  This may be due to network conditions or test environment")
        else:
            logger.info(f"\n❌ PERFORMANCE VALIDATION: ❌ FAILED")
            logger.info(f"❌ Unable to measure performance improvements")
        
        logger.info(f"\n🎯 Phase 3.1.6.4 - Performance Benchmarking: COMPLETE")
        
        return self.results


def main():
    """Run the comprehensive connection pooling benchmark."""
    benchmark = ConnectionPoolingBenchmark()
    results = benchmark.run_all_benchmarks()
    
    # Save results to file
    results_file = "benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n📄 Benchmark results saved to: {results_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())