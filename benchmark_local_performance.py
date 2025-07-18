#!/usr/bin/env python3
"""
Local HTTP Connection Pooling Performance Benchmark

This benchmark validates the performance improvements of HTTP connection pooling
for LLM requests by testing local performance characteristics without external dependencies.

Focus Areas:
1. Session Creation/Reuse Performance
2. Memory Efficiency
3. Connection Pool Management Overhead
4. Circuit Breaker Response Times
5. Statistics Collection Performance

Test Methodology:
- Uses local timing measurements for session operations
- Tests connection pool manager overhead
- Validates memory efficiency improvements
- Measures circuit breaker response times
"""

import logging
import os
import sys
import time
import threading
import tracemalloc
from typing import Dict, List, Optional, Any
import statistics
import json
import requests

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocalPerformanceBenchmark:
    """Local performance benchmark for HTTP connection pooling."""
    
    def __init__(self):
        self.results = {}
        
    def run_all_benchmarks(self):
        """Run all local performance benchmarks."""
        logger.info("="*70)
        logger.info("LOCAL CONNECTION POOLING PERFORMANCE BENCHMARK")
        logger.info("="*70)
        
        # Benchmark 1: Session Creation Performance
        self.benchmark_session_creation()
        
        # Benchmark 2: Connection Pool Overhead
        self.benchmark_pool_overhead()
        
        # Benchmark 3: Memory Efficiency
        self.benchmark_memory_efficiency()
        
        # Benchmark 4: Circuit Breaker Performance
        self.benchmark_circuit_breaker_timing()
        
        # Benchmark 5: Statistics Collection Performance
        self.benchmark_statistics_performance()
        
        # Print comprehensive results
        self.print_benchmark_results()
        
        return self.results
    
    def benchmark_session_creation(self):
        """Benchmark session creation and reuse performance."""
        logger.info("\n🚀 Benchmarking session creation performance...")
        
        try:
            from llm_pool.http_pool_manager import HTTPConnectionPoolManager
            
            num_operations = 1000
            
            # Test connection pool session reuse
            pool_manager = HTTPConnectionPoolManager()
            
            start_time = time.time()
            for i in range(num_operations):
                session = pool_manager.get_session("http://localhost:11434")
                # Simulate session usage
                pass
            pooled_time = time.time() - start_time
            
            pool_manager.close()
            
            # Test individual session creation
            start_time = time.time()
            for i in range(num_operations):
                session = requests.Session()
                session.close()
            individual_time = time.time() - start_time
            
            # Calculate results
            improvement = individual_time / pooled_time if pooled_time > 0 else 1
            
            self.results['session_creation'] = {
                'pooled_time': pooled_time,
                'individual_time': individual_time,
                'improvement_factor': improvement,
                'operations': num_operations,
                'pooled_ops_per_sec': num_operations / pooled_time if pooled_time > 0 else 0,
                'individual_ops_per_sec': num_operations / individual_time if individual_time > 0 else 0
            }
            
            logger.info(f"   📊 Session creation improvement: {improvement:.2f}x faster")
            logger.info(f"   📊 Pooled: {num_operations / pooled_time:.0f} ops/sec")
            logger.info(f"   📊 Individual: {num_operations / individual_time:.0f} ops/sec")
            
        except Exception as e:
            logger.error(f"   ❌ Session creation benchmark failed: {e}")
    
    def benchmark_pool_overhead(self):
        """Benchmark connection pool management overhead."""
        logger.info("\n⚙️  Benchmarking connection pool overhead...")
        
        try:
            from llm_pool.http_pool_manager import HTTPConnectionPoolManager
            
            num_operations = 10000
            
            # Test pool manager operations
            pool_manager = HTTPConnectionPoolManager()
            
            start_time = time.time()
            for i in range(num_operations):
                # Test various pool operations
                stats = pool_manager.get_stats()
                health = pool_manager.get_health_status()
                session_stats = pool_manager.get_session_stats()
            pool_overhead_time = time.time() - start_time
            
            pool_manager.close()
            
            # Test equivalent operations without pool manager
            mock_stats = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'success_rate': 100.0,
                'failure_rate': 0.0,
                'average_response_time': 0.0,
                'active_connections': 0,
                'pool_sessions': 1
            }
            
            start_time = time.time()
            for i in range(num_operations):
                # Simulate equivalent operations
                _ = dict(mock_stats)
                _ = {'healthy': True, 'issues': []}
                _ = {'total_sessions': 1, 'healthy_sessions': 1}
            direct_time = time.time() - start_time
            
            # Calculate overhead
            overhead_factor = pool_overhead_time / direct_time if direct_time > 0 else 1
            
            self.results['pool_overhead'] = {
                'pool_time': pool_overhead_time,
                'direct_time': direct_time,
                'overhead_factor': overhead_factor,
                'operations': num_operations,
                'overhead_per_operation_us': (pool_overhead_time - direct_time) / num_operations * 1000000
            }
            
            logger.info(f"   📊 Pool overhead: {overhead_factor:.2f}x (minimal is good)")
            logger.info(f"   📊 Overhead per operation: {(pool_overhead_time - direct_time) / num_operations * 1000000:.1f} μs")
            
        except Exception as e:
            logger.error(f"   ❌ Pool overhead benchmark failed: {e}")
    
    def benchmark_memory_efficiency(self):
        """Benchmark memory efficiency of connection pooling."""
        logger.info("\n🧠 Benchmarking memory efficiency...")
        
        try:
            from llm_pool.http_pool_manager import HTTPConnectionPoolManager
            
            num_sessions = 100
            
            # Start memory tracking
            tracemalloc.start()
            
            # Test pooled session memory usage
            snapshot1 = tracemalloc.take_snapshot()
            
            pool_manager = HTTPConnectionPoolManager()
            sessions = []
            for i in range(num_sessions):
                session = pool_manager.get_session(f"http://localhost:{11434 + i % 10}")
                sessions.append(session)
            
            snapshot2 = tracemalloc.take_snapshot()
            pool_manager.close()
            
            # Calculate pooled memory usage
            top_stats = snapshot2.compare_to(snapshot1, 'lineno')
            pooled_memory = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
            
            # Test individual session memory usage
            snapshot3 = tracemalloc.take_snapshot()
            
            individual_sessions = []
            for i in range(num_sessions):
                session = requests.Session()
                individual_sessions.append(session)
            
            snapshot4 = tracemalloc.take_snapshot()
            
            # Cleanup individual sessions
            for session in individual_sessions:
                session.close()
            
            # Calculate individual memory usage
            top_stats = snapshot4.compare_to(snapshot3, 'lineno')
            individual_memory = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
            
            tracemalloc.stop()
            
            # Calculate efficiency
            memory_efficiency = individual_memory / pooled_memory if pooled_memory > 0 else 1
            
            self.results['memory_efficiency'] = {
                'pooled_memory_bytes': pooled_memory,
                'individual_memory_bytes': individual_memory,
                'memory_efficiency': memory_efficiency,
                'sessions_tested': num_sessions,
                'memory_per_session_pooled': pooled_memory / num_sessions if num_sessions > 0 else 0,
                'memory_per_session_individual': individual_memory / num_sessions if num_sessions > 0 else 0
            }
            
            logger.info(f"   📊 Memory efficiency: {memory_efficiency:.2f}x less memory")
            logger.info(f"   📊 Pooled: {pooled_memory/1024:.1f} KB ({pooled_memory/num_sessions:.0f} bytes/session)")
            logger.info(f"   📊 Individual: {individual_memory/1024:.1f} KB ({individual_memory/num_sessions:.0f} bytes/session)")
            
        except Exception as e:
            logger.error(f"   ❌ Memory efficiency benchmark failed: {e}")
    
    def benchmark_circuit_breaker_timing(self):
        """Benchmark circuit breaker response times."""
        logger.info("\n🔌 Benchmarking circuit breaker timing...")
        
        try:
            from llm_pool.http_pool_manager import HTTPConnectionPoolManager, ConnectionPoolConfig
            
            # Create pool with fast circuit breaker
            config = ConnectionPoolConfig(
                circuit_breaker_enabled=True,
                circuit_breaker_failure_threshold=2,
                circuit_breaker_recovery_timeout=0.1
            )
            
            pool_manager = HTTPConnectionPoolManager(config)
            circuit_breaker = pool_manager.circuit_breaker
            
            # Test normal state performance
            start_time = time.time()
            can_request = circuit_breaker.can_make_request()
            normal_check_time = time.time() - start_time
            
            # Force circuit to open
            circuit_breaker.record_failure()
            circuit_breaker.record_failure()
            
            # Test open state performance
            start_time = time.time()
            can_request = circuit_breaker.can_make_request()
            open_check_time = time.time() - start_time
            
            # Test state transition timing
            start_time = time.time()
            circuit_breaker.record_success()
            transition_time = time.time() - start_time
            
            # Test recovery timing
            time.sleep(0.11)  # Wait for recovery timeout
            start_time = time.time()
            can_request = circuit_breaker.can_make_request()
            recovery_check_time = time.time() - start_time
            
            pool_manager.close()
            
            self.results['circuit_breaker_timing'] = {
                'normal_check_time_us': normal_check_time * 1000000,
                'open_check_time_us': open_check_time * 1000000,
                'transition_time_us': transition_time * 1000000,
                'recovery_check_time_us': recovery_check_time * 1000000,
                'avg_check_time_us': (normal_check_time + open_check_time + recovery_check_time) / 3 * 1000000
            }
            
            logger.info(f"   📊 Normal state check: {normal_check_time * 1000000:.1f} μs")
            logger.info(f"   📊 Open state check: {open_check_time * 1000000:.1f} μs")
            logger.info(f"   📊 State transition: {transition_time * 1000000:.1f} μs")
            logger.info(f"   📊 Recovery check: {recovery_check_time * 1000000:.1f} μs")
            
        except Exception as e:
            logger.error(f"   ❌ Circuit breaker timing benchmark failed: {e}")
    
    def benchmark_statistics_performance(self):
        """Benchmark statistics collection performance."""
        logger.info("\n📊 Benchmarking statistics performance...")
        
        try:
            from llm_pool.http_pool_manager import HTTPConnectionPoolManager
            
            num_operations = 10000
            
            pool_manager = HTTPConnectionPoolManager()
            
            # Simulate some requests to generate statistics
            for i in range(10):
                pool_manager.stats.update_request_stats(0.1, True, False)
            
            # Test statistics collection performance
            start_time = time.time()
            for i in range(num_operations):
                stats = pool_manager.get_stats()
                session_stats = pool_manager.get_session_stats()
                health_status = pool_manager.get_health_status()
                monitoring_report = pool_manager.get_monitoring_report()
            stats_time = time.time() - start_time
            
            # Test statistics update performance
            start_time = time.time()
            for i in range(num_operations):
                pool_manager.stats.update_request_stats(0.1 + (i % 100) / 1000, i % 2 == 0, False)
            update_time = time.time() - start_time
            
            pool_manager.close()
            
            self.results['statistics_performance'] = {
                'collection_time': stats_time,
                'update_time': update_time,
                'operations': num_operations,
                'collection_ops_per_sec': num_operations / stats_time if stats_time > 0 else 0,
                'update_ops_per_sec': num_operations / update_time if update_time > 0 else 0,
                'collection_time_per_op_us': stats_time / num_operations * 1000000,
                'update_time_per_op_us': update_time / num_operations * 1000000
            }
            
            logger.info(f"   📊 Stats collection: {num_operations / stats_time:.0f} ops/sec")
            logger.info(f"   📊 Stats update: {num_operations / update_time:.0f} ops/sec")
            logger.info(f"   📊 Collection time per op: {stats_time / num_operations * 1000000:.1f} μs")
            logger.info(f"   📊 Update time per op: {update_time / num_operations * 1000000:.1f} μs")
            
        except Exception as e:
            logger.error(f"   ❌ Statistics performance benchmark failed: {e}")
    
    def print_benchmark_results(self):
        """Print comprehensive benchmark results."""
        logger.info("\n" + "="*70)
        logger.info("LOCAL PERFORMANCE BENCHMARK RESULTS")
        logger.info("="*70)
        
        # Calculate key performance indicators
        performance_indicators = []
        
        if 'session_creation' in self.results:
            improvement = self.results['session_creation']['improvement_factor']
            performance_indicators.append(('Session Creation', improvement))
        
        if 'memory_efficiency' in self.results:
            efficiency = self.results['memory_efficiency']['memory_efficiency']
            performance_indicators.append(('Memory Efficiency', efficiency))
        
        if 'pool_overhead' in self.results:
            overhead = self.results['pool_overhead']['overhead_factor']
            performance_indicators.append(('Pool Overhead', overhead))
        
        # Print summary
        logger.info(f"\n🎯 PERFORMANCE SUMMARY:")
        for name, value in performance_indicators:
            if 'overhead' in name.lower():
                status = "✅ Minimal" if value <= 2.0 else "⚠️  High"
                logger.info(f"   {name}: {value:.2f}x {status}")
            else:
                status = "✅ Good" if value >= 1.5 else "⚠️  Minimal" if value >= 1.1 else "❌ Poor"
                logger.info(f"   {name}: {value:.2f}x {status}")
        
        # Detailed results
        logger.info(f"\n📈 DETAILED RESULTS:")
        for test_name, results in self.results.items():
            logger.info(f"\n🔍 {test_name.upper().replace('_', ' ')}:")
            for key, value in results.items():
                if isinstance(value, float):
                    if 'time' in key and 'us' in key:
                        logger.info(f"   {key}: {value:.1f} μs")
                    elif 'time' in key:
                        logger.info(f"   {key}: {value:.3f}s")
                    elif 'ops_per_sec' in key:
                        logger.info(f"   {key}: {value:.0f} ops/sec")
                    elif 'bytes' in key:
                        logger.info(f"   {key}: {value/1024:.1f} KB")
                    elif 'factor' in key or 'efficiency' in key:
                        logger.info(f"   {key}: {value:.2f}x")
                    else:
                        logger.info(f"   {key}: {value:.2f}")
                else:
                    logger.info(f"   {key}: {value}")
        
        # Overall assessment
        if performance_indicators:
            avg_improvement = statistics.mean([v for _, v in performance_indicators if 'overhead' not in _])
            
            if avg_improvement >= 2.0:
                logger.info(f"\n🎉 PERFORMANCE ASSESSMENT: ✅ EXCELLENT")
                logger.info(f"✅ Connection pooling provides significant performance benefits!")
                logger.info(f"✅ Average {avg_improvement:.2f}x improvement across key metrics")
            elif avg_improvement >= 1.5:
                logger.info(f"\n✅ PERFORMANCE ASSESSMENT: ✅ GOOD")
                logger.info(f"✅ Connection pooling provides measurable performance benefits!")
                logger.info(f"✅ Average {avg_improvement:.2f}x improvement validates implementation")
            elif avg_improvement >= 1.1:
                logger.info(f"\n⚠️  PERFORMANCE ASSESSMENT: ⚠️  ADEQUATE")
                logger.info(f"⚠️  Connection pooling provides some performance benefits")
                logger.info(f"⚠️  Average {avg_improvement:.2f}x improvement is modest but positive")
            else:
                logger.info(f"\n❌ PERFORMANCE ASSESSMENT: ❌ POOR")
                logger.info(f"❌ Connection pooling shows minimal performance benefits")
        
        logger.info(f"\n🎯 Phase 3.1.6.4 - Local Performance Validation: COMPLETE")
        
        return self.results


def main():
    """Run the local performance benchmark."""
    benchmark = LocalPerformanceBenchmark()
    results = benchmark.run_all_benchmarks()
    
    # Save results to file
    results_file = "local_benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n📄 Local benchmark results saved to: {results_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())