"""
Performance testing framework with load simulation for text-to-audiobook pipeline.

Provides comprehensive load testing, performance benchmarking, and stress testing
capabilities for distributed components including Spark, Kafka, and LLM processing.
"""

import pytest
import time
import json
import threading
import statistics
from typing import Dict, Any, List, Optional, Callable
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import resource
import psutil
import gc

from tests.utils.test_data_manager import get_test_data_manager, TestDataGenerator
from src.monitoring.prometheus_metrics import get_metrics_collector


@dataclass
class LoadTestConfiguration:
    """Configuration for load testing scenarios."""
    test_name: str
    concurrent_users: int
    total_requests: int
    duration_seconds: Optional[int] = None
    ramp_up_seconds: int = 30
    target_throughput: Optional[float] = None
    max_response_time: float = 5.0
    success_rate_threshold: float = 0.95
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    test_data_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceResult:
    """Individual performance test result."""
    test_id: str
    start_time: float
    end_time: float
    success: bool
    response_time: float
    error_message: Optional[str] = None
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadTestReport:
    """Comprehensive load test report."""
    test_name: str
    configuration: LoadTestConfiguration
    start_time: datetime
    end_time: datetime
    total_duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    
    # Performance statistics
    response_times: List[float]
    min_response_time: float
    max_response_time: float
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    
    # Throughput statistics
    requests_per_second: float
    peak_throughput: float
    
    # Resource statistics
    peak_memory_usage: float
    peak_cpu_usage: float
    resource_usage_timeline: List[Dict[str, Any]]
    
    # Error analysis
    error_breakdown: Dict[str, int]
    
    # Performance trends
    throughput_timeline: List[Dict[str, Any]]
    response_time_timeline: List[Dict[str, Any]]


class LoadTestRunner:
    """
    Load test runner with advanced simulation capabilities.
    
    Provides comprehensive load testing with realistic user behavior simulation,
    resource monitoring, and detailed performance analysis.
    """
    
    def __init__(self, metrics_collector=None):
        """
        Initialize load test runner.
        
        Args:
            metrics_collector: Metrics collector for monitoring
        """
        self.metrics_collector = metrics_collector or get_metrics_collector()
        self.active_tests: Dict[str, threading.Event] = {}
        self.results_cache: Dict[str, List[PerformanceResult]] = {}
        self.resource_monitor_interval = 1.0
        
    def run_load_test(self, 
                     config: LoadTestConfiguration,
                     test_function: Callable,
                     **test_kwargs) -> LoadTestReport:
        """
        Execute a load test with the specified configuration.
        
        Args:
            config: Load test configuration
            test_function: Function to execute under load
            **test_kwargs: Additional arguments for test function
            
        Returns:
            Comprehensive load test report
        """
        
        test_start = datetime.now()
        stop_event = threading.Event()
        self.active_tests[config.test_name] = stop_event
        
        # Initialize result collection
        results: List[PerformanceResult] = []
        results_lock = threading.Lock()
        
        # Initialize resource monitoring
        resource_timeline = []
        resource_lock = threading.Lock()
        
        # Start resource monitoring thread
        resource_monitor = threading.Thread(
            target=self._monitor_resources,
            args=(stop_event, resource_timeline, resource_lock),
            daemon=True
        )
        resource_monitor.start()
        
        try:
            # Execute load test
            if config.duration_seconds:
                # Duration-based load test
                results = self._run_duration_based_test(
                    config, test_function, stop_event, results_lock, test_kwargs
                )
            else:
                # Request-count-based load test
                results = self._run_request_based_test(
                    config, test_function, stop_event, results_lock, test_kwargs
                )
            
        finally:
            # Stop monitoring
            stop_event.set()
            resource_monitor.join(timeout=5.0)
            
            # Clean up
            if config.test_name in self.active_tests:
                del self.active_tests[config.test_name]
        
        test_end = datetime.now()
        
        # Generate comprehensive report
        report = self._generate_load_test_report(
            config, test_start, test_end, results, resource_timeline
        )
        
        # Cache results for analysis
        self.results_cache[config.test_name] = results
        
        return report
    
    def _run_duration_based_test(self,
                               config: LoadTestConfiguration,
                               test_function: Callable,
                               stop_event: threading.Event,
                               results_lock: threading.Lock,
                               test_kwargs: Dict[str, Any]) -> List[PerformanceResult]:
        """Run duration-based load test."""
        
        results = []
        
        # Calculate timing
        end_time = time.time() + config.duration_seconds
        ramp_up_end = time.time() + config.ramp_up_seconds
        
        # Worker function
        def worker(worker_id: int):
            worker_results = []
            request_count = 0
            
            while time.time() < end_time and not stop_event.is_set():
                # Ramp-up logic
                current_time = time.time()
                if current_time < ramp_up_end:
                    # Calculate current concurrency during ramp-up
                    progress = (current_time - (end_time - config.duration_seconds)) / config.ramp_up_seconds
                    current_concurrency = int(config.concurrent_users * progress)
                    if worker_id >= current_concurrency:
                        time.sleep(0.1)
                        continue
                
                # Execute test
                result = self._execute_single_test(
                    f"{config.test_name}_{worker_id}_{request_count:06d}",
                    test_function,
                    test_kwargs
                )
                worker_results.append(result)
                request_count += 1
                
                # Apply throttling if target throughput is specified
                if config.target_throughput:
                    expected_interval = 1.0 / (config.target_throughput / config.concurrent_users)
                    time.sleep(max(0, expected_interval - result.response_time))
            
            with results_lock:
                results.extend(worker_results)
        
        # Launch worker threads
        with ThreadPoolExecutor(max_workers=config.concurrent_users) as executor:
            futures = [
                executor.submit(worker, i)
                for i in range(config.concurrent_users)
            ]
            
            # Wait for completion
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Worker failed: {e}")
        
        return results
    
    def _run_request_based_test(self,
                              config: LoadTestConfiguration,
                              test_function: Callable,
                              stop_event: threading.Event,
                              results_lock: threading.Lock,
                              test_kwargs: Dict[str, Any]) -> List[PerformanceResult]:
        """Run request-count-based load test."""
        
        results = []
        
        # Calculate requests per worker
        requests_per_worker = config.total_requests // config.concurrent_users
        remaining_requests = config.total_requests % config.concurrent_users
        
        def worker(worker_id: int, request_count: int):
            worker_results = []
            
            # Ramp-up delay
            ramp_up_delay = (config.ramp_up_seconds / config.concurrent_users) * worker_id
            time.sleep(ramp_up_delay)
            
            for i in range(request_count):
                if stop_event.is_set():
                    break
                
                result = self._execute_single_test(
                    f"{config.test_name}_{worker_id}_{i:06d}",
                    test_function,
                    test_kwargs
                )
                worker_results.append(result)
                
                # Apply throttling if target throughput is specified
                if config.target_throughput:
                    expected_interval = 1.0 / (config.target_throughput / config.concurrent_users)
                    time.sleep(max(0, expected_interval - result.response_time))
            
            with results_lock:
                results.extend(worker_results)
        
        # Launch worker threads
        with ThreadPoolExecutor(max_workers=config.concurrent_users) as executor:
            futures = []
            
            for i in range(config.concurrent_users):
                # Distribute remaining requests among first workers
                worker_requests = requests_per_worker
                if i < remaining_requests:
                    worker_requests += 1
                
                futures.append(
                    executor.submit(worker, i, worker_requests)
                )
            
            # Wait for completion
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Worker failed: {e}")
        
        return results
    
    def _execute_single_test(self,
                           test_id: str,
                           test_function: Callable,
                           test_kwargs: Dict[str, Any]) -> PerformanceResult:
        """Execute a single test iteration."""
        
        start_time = time.time()
        success = False
        error_message = None
        
        # Capture resource usage before test
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Execute test function
            result = test_function(**test_kwargs)
            success = True
            
        except Exception as e:
            error_message = str(e)
            success = False
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Capture resource usage after test
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        return PerformanceResult(
            test_id=test_id,
            start_time=start_time,
            end_time=end_time,
            success=success,
            response_time=response_time,
            error_message=error_message,
            resource_usage={
                'memory_before_mb': memory_before,
                'memory_after_mb': memory_after,
                'memory_delta_mb': memory_after - memory_before
            }
        )
    
    def _monitor_resources(self,
                         stop_event: threading.Event,
                         resource_timeline: List[Dict[str, Any]],
                         resource_lock: threading.Lock):
        """Monitor system resources during load test."""
        
        while not stop_event.is_set():
            try:
                # Get current resource usage
                current_time = time.time()
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                resource_data = {
                    'timestamp': current_time,
                    'cpu_percent': cpu_percent,
                    'memory_total_gb': memory.total / 1024**3,
                    'memory_available_gb': memory.available / 1024**3,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / 1024**3
                }
                
                # Add disk I/O if available
                try:
                    disk_io = psutil.disk_io_counters()
                    if disk_io:
                        resource_data.update({
                            'disk_read_bytes': disk_io.read_bytes,
                            'disk_write_bytes': disk_io.write_bytes
                        })
                except:
                    pass
                
                # Add network I/O if available
                try:
                    net_io = psutil.net_io_counters()
                    if net_io:
                        resource_data.update({
                            'network_bytes_sent': net_io.bytes_sent,
                            'network_bytes_recv': net_io.bytes_recv
                        })
                except:
                    pass
                
                with resource_lock:
                    resource_timeline.append(resource_data)
                
                time.sleep(self.resource_monitor_interval)
                
            except Exception:
                # Continue monitoring even if individual measurements fail
                time.sleep(self.resource_monitor_interval)
    
    def _generate_load_test_report(self,
                                 config: LoadTestConfiguration,
                                 start_time: datetime,
                                 end_time: datetime,
                                 results: List[PerformanceResult],
                                 resource_timeline: List[Dict[str, Any]]) -> LoadTestReport:
        """Generate comprehensive load test report."""
        
        total_duration = (end_time - start_time).total_seconds()
        
        # Filter successful and failed results
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        # Calculate basic statistics
        total_requests = len(results)
        successful_requests = len(successful_results)
        failed_requests = len(failed_results)
        success_rate = successful_requests / total_requests if total_requests > 0 else 0.0
        
        # Response time statistics
        response_times = [r.response_time for r in successful_results]
        
        if response_times:
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            avg_response_time = statistics.mean(response_times)
            p50_response_time = statistics.median(response_times)
            
            sorted_times = sorted(response_times)
            p95_idx = int(len(sorted_times) * 0.95)
            p99_idx = int(len(sorted_times) * 0.99)
            p95_response_time = sorted_times[p95_idx] if p95_idx < len(sorted_times) else max_response_time
            p99_response_time = sorted_times[p99_idx] if p99_idx < len(sorted_times) else max_response_time
        else:
            min_response_time = max_response_time = avg_response_time = 0.0
            p50_response_time = p95_response_time = p99_response_time = 0.0
        
        # Throughput statistics
        requests_per_second = total_requests / total_duration if total_duration > 0 else 0.0
        
        # Calculate peak throughput (requests per second in best 10-second window)
        peak_throughput = self._calculate_peak_throughput(results)
        
        # Resource statistics
        peak_memory_usage = 0.0
        peak_cpu_usage = 0.0
        
        if resource_timeline:
            peak_memory_usage = max(r['memory_percent'] for r in resource_timeline)
            peak_cpu_usage = max(r['cpu_percent'] for r in resource_timeline)
        
        # Error breakdown
        error_breakdown = {}
        for result in failed_results:
            error_type = type(Exception(result.error_message)).__name__ if result.error_message else 'Unknown'
            error_breakdown[error_type] = error_breakdown.get(error_type, 0) + 1
        
        # Generate timeline data
        throughput_timeline = self._generate_throughput_timeline(results, total_duration)
        response_time_timeline = self._generate_response_time_timeline(successful_results, total_duration)
        
        return LoadTestReport(
            test_name=config.test_name,
            configuration=config,
            start_time=start_time,
            end_time=end_time,
            total_duration=total_duration,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            success_rate=success_rate,
            response_times=response_times,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            avg_response_time=avg_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=requests_per_second,
            peak_throughput=peak_throughput,
            peak_memory_usage=peak_memory_usage,
            peak_cpu_usage=peak_cpu_usage,
            resource_usage_timeline=resource_timeline,
            error_breakdown=error_breakdown,
            throughput_timeline=throughput_timeline,
            response_time_timeline=response_time_timeline
        )
    
    def _calculate_peak_throughput(self, results: List[PerformanceResult]) -> float:
        """Calculate peak throughput over a sliding window."""
        
        if not results:
            return 0.0
        
        # Sort results by start time
        sorted_results = sorted(results, key=lambda r: r.start_time)
        
        window_size = 10.0  # 10-second window
        max_throughput = 0.0
        
        for i, result in enumerate(sorted_results):
            window_start = result.start_time
            window_end = window_start + window_size
            
            # Count requests in this window
            requests_in_window = 0
            for r in sorted_results[i:]:
                if r.start_time >= window_end:
                    break
                requests_in_window += 1
            
            throughput = requests_in_window / window_size
            max_throughput = max(max_throughput, throughput)
        
        return max_throughput
    
    def _generate_throughput_timeline(self, 
                                    results: List[PerformanceResult], 
                                    total_duration: float) -> List[Dict[str, Any]]:
        """Generate throughput timeline data."""
        
        if not results:
            return []
        
        # Create time buckets (10-second intervals)
        bucket_size = 10.0
        num_buckets = int(total_duration / bucket_size) + 1
        
        timeline = []
        sorted_results = sorted(results, key=lambda r: r.start_time)
        min_start_time = sorted_results[0].start_time
        
        for i in range(num_buckets):
            bucket_start = min_start_time + (i * bucket_size)
            bucket_end = bucket_start + bucket_size
            
            # Count requests in this bucket
            requests_count = sum(
                1 for r in sorted_results 
                if bucket_start <= r.start_time < bucket_end
            )
            
            throughput = requests_count / bucket_size
            
            timeline.append({
                'time_offset': i * bucket_size,
                'timestamp': bucket_start,
                'requests_count': requests_count,
                'throughput': throughput
            })
        
        return timeline
    
    def _generate_response_time_timeline(self,
                                       results: List[PerformanceResult],
                                       total_duration: float) -> List[Dict[str, Any]]:
        """Generate response time timeline data."""
        
        if not results:
            return []
        
        # Create time buckets (10-second intervals)
        bucket_size = 10.0
        num_buckets = int(total_duration / bucket_size) + 1
        
        timeline = []
        sorted_results = sorted(results, key=lambda r: r.start_time)
        min_start_time = sorted_results[0].start_time
        
        for i in range(num_buckets):
            bucket_start = min_start_time + (i * bucket_size)
            bucket_end = bucket_start + bucket_size
            
            # Get response times in this bucket
            bucket_response_times = [
                r.response_time for r in sorted_results 
                if bucket_start <= r.start_time < bucket_end
            ]
            
            if bucket_response_times:
                avg_response_time = statistics.mean(bucket_response_times)
                min_response_time = min(bucket_response_times)
                max_response_time = max(bucket_response_times)
                p95_response_time = statistics.quantiles(bucket_response_times, n=20)[18] if len(bucket_response_times) >= 20 else max_response_time
            else:
                avg_response_time = min_response_time = max_response_time = p95_response_time = 0.0
            
            timeline.append({
                'time_offset': i * bucket_size,
                'timestamp': bucket_start,
                'sample_count': len(bucket_response_times),
                'avg_response_time': avg_response_time,
                'min_response_time': min_response_time,
                'max_response_time': max_response_time,
                'p95_response_time': p95_response_time
            })
        
        return timeline
    
    def compare_load_test_results(self, 
                                 baseline_report: LoadTestReport,
                                 current_report: LoadTestReport) -> Dict[str, Any]:
        """Compare two load test reports for performance regression analysis."""
        
        comparison = {
            'test_comparison': {
                'baseline_test': baseline_report.test_name,
                'current_test': current_report.test_name,
                'comparison_timestamp': datetime.now().isoformat()
            },
            'performance_deltas': {},
            'regression_analysis': {},
            'recommendations': []
        }
        
        # Calculate performance deltas
        deltas = comparison['performance_deltas']
        
        deltas['success_rate_delta'] = current_report.success_rate - baseline_report.success_rate
        deltas['avg_response_time_delta'] = current_report.avg_response_time - baseline_report.avg_response_time
        deltas['p95_response_time_delta'] = current_report.p95_response_time - baseline_report.p95_response_time
        deltas['throughput_delta'] = current_report.requests_per_second - baseline_report.requests_per_second
        deltas['peak_memory_delta'] = current_report.peak_memory_usage - baseline_report.peak_memory_usage
        deltas['peak_cpu_delta'] = current_report.peak_cpu_usage - baseline_report.peak_cpu_usage
        
        # Regression analysis
        regression = comparison['regression_analysis']
        
        # Success rate regression
        if deltas['success_rate_delta'] < -0.05:  # 5% decrease
            regression['success_rate'] = 'regression'
        elif deltas['success_rate_delta'] > 0.02:  # 2% improvement
            regression['success_rate'] = 'improvement'
        else:
            regression['success_rate'] = 'stable'
        
        # Response time regression
        if deltas['avg_response_time_delta'] > 0.5:  # 500ms increase
            regression['response_time'] = 'regression'
        elif deltas['avg_response_time_delta'] < -0.1:  # 100ms improvement
            regression['response_time'] = 'improvement'
        else:
            regression['response_time'] = 'stable'
        
        # Throughput regression
        throughput_change_percent = (deltas['throughput_delta'] / baseline_report.requests_per_second) * 100 if baseline_report.requests_per_second > 0 else 0
        if throughput_change_percent < -10:  # 10% decrease
            regression['throughput'] = 'regression'
        elif throughput_change_percent > 5:  # 5% improvement
            regression['throughput'] = 'improvement'
        else:
            regression['throughput'] = 'stable'
        
        # Generate recommendations
        recommendations = comparison['recommendations']
        
        if regression['success_rate'] == 'regression':
            recommendations.append('Investigate error rate increase - check logs for failure patterns')
        
        if regression['response_time'] == 'regression':
            recommendations.append('Response time degradation detected - profile bottlenecks and optimize critical paths')
        
        if regression['throughput'] == 'regression':
            recommendations.append('Throughput regression detected - check resource utilization and scaling configuration')
        
        if deltas['peak_memory_delta'] > 1000:  # 1GB increase
            recommendations.append('Memory usage increased significantly - investigate memory leaks or optimize memory allocation')
        
        if deltas['peak_cpu_delta'] > 20:  # 20% increase
            recommendations.append('CPU usage increased significantly - optimize CPU-intensive operations')
        
        return comparison


# Predefined load test scenarios
class LoadTestScenarios:
    """Predefined load test scenarios for common use cases."""
    
    @staticmethod
    def light_load() -> LoadTestConfiguration:
        """Light load scenario for development testing."""
        return LoadTestConfiguration(
            test_name="light_load",
            concurrent_users=5,
            total_requests=100,
            ramp_up_seconds=10,
            max_response_time=2.0,
            success_rate_threshold=0.98
        )
    
    @staticmethod
    def moderate_load() -> LoadTestConfiguration:
        """Moderate load scenario for staging testing."""
        return LoadTestConfiguration(
            test_name="moderate_load",
            concurrent_users=20,
            total_requests=1000,
            ramp_up_seconds=30,
            max_response_time=3.0,
            success_rate_threshold=0.95
        )
    
    @staticmethod
    def heavy_load() -> LoadTestConfiguration:
        """Heavy load scenario for production capacity testing."""
        return LoadTestConfiguration(
            test_name="heavy_load",
            concurrent_users=50,
            total_requests=5000,
            ramp_up_seconds=60,
            max_response_time=5.0,
            success_rate_threshold=0.90
        )
    
    @staticmethod
    def stress_test() -> LoadTestConfiguration:
        """Stress test scenario to find breaking points."""
        return LoadTestConfiguration(
            test_name="stress_test",
            concurrent_users=100,
            duration_seconds=300,  # 5 minutes
            ramp_up_seconds=120,
            max_response_time=10.0,
            success_rate_threshold=0.80
        )
    
    @staticmethod
    def endurance_test() -> LoadTestConfiguration:
        """Endurance test for long-running stability."""
        return LoadTestConfiguration(
            test_name="endurance_test",
            concurrent_users=25,
            duration_seconds=1800,  # 30 minutes
            ramp_up_seconds=300,
            max_response_time=5.0,
            success_rate_threshold=0.95
        )
    
    @staticmethod
    def spike_test() -> LoadTestConfiguration:
        """Spike test for sudden load increases."""
        return LoadTestConfiguration(
            test_name="spike_test",
            concurrent_users=100,
            total_requests=2000,
            ramp_up_seconds=5,  # Very fast ramp-up
            max_response_time=8.0,
            success_rate_threshold=0.85
        )


def create_performance_test_suite() -> Dict[str, LoadTestConfiguration]:
    """Create a comprehensive performance test suite."""
    
    return {
        'light': LoadTestScenarios.light_load(),
        'moderate': LoadTestScenarios.moderate_load(),
        'heavy': LoadTestScenarios.heavy_load(),
        'stress': LoadTestScenarios.stress_test(),
        'endurance': LoadTestScenarios.endurance_test(),
        'spike': LoadTestScenarios.spike_test()
    }