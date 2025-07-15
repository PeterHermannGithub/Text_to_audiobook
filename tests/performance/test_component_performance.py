"""
Component-specific performance tests for text-to-audiobook pipeline.

Tests individual component performance characteristics, resource usage,
and optimization opportunities across the distributed architecture.
"""

import pytest
import time
import json
import statistics
from typing import Dict, Any, List, Callable
from unittest.mock import Mock, patch
from datetime import datetime
import tempfile
from pathlib import Path

from tests.performance.test_load_simulation import LoadTestRunner, LoadTestConfiguration, PerformanceResult
from tests.utils.test_data_manager import get_test_data_manager, TestDataGenerator
from src.monitoring.prometheus_metrics import get_metrics_collector


@pytest.mark.performance
@pytest.mark.component
class TestSparkComponentPerformance:
    """Performance tests for Spark components."""
    
    @pytest.fixture
    def spark_performance_config(self):
        """Configuration for Spark performance testing."""
        return {
            'data_sizes': [100, 500, 1000, 5000],  # MB
            'executor_configs': [
                {'instances': 2, 'cores': 2, 'memory': '2g'},
                {'instances': 4, 'cores': 2, 'memory': '2g'},
                {'instances': 2, 'cores': 4, 'memory': '4g'}
            ],
            'partition_counts': [10, 20, 50, 100],
            'performance_thresholds': {
                'max_processing_time_per_mb': 0.5,  # seconds
                'min_throughput_mb_per_second': 2.0,
                'max_memory_overhead_percent': 150.0
            }
        }
    
    def test_spark_text_structuring_performance(self, spark_performance_config):
        """Test Spark text structuring performance across different data sizes."""
        
        with patch('src.spark.distributed_validation.SparkSession') as mock_spark_class:
            mock_spark_session = Mock()
            mock_spark_class.getActiveSession.return_value = mock_spark_session
            
            # Mock DataFrame operations for performance testing
            mock_df = Mock()
            mock_df.cache.return_value = mock_df
            mock_df.unpersist.return_value = None
            mock_df.count.return_value = 1000
            mock_spark_session.createDataFrame.return_value = mock_df
            
            from src.spark.distributed_validation import DistributedValidationEngine
            
            performance_results = []
            
            for data_size_mb in spark_performance_config['data_sizes']:
                # Generate test data
                test_segments = self._generate_test_segments_for_size(data_size_mb)
                
                start_time = time.time()
                
                # Test Spark processing
                validation_engine = DistributedValidationEngine(mock_spark_session)
                
                with patch.object(validation_engine, '_perform_segment_validation', return_value=mock_df):
                    with patch.object(validation_engine, '_apply_segment_refinements', return_value=mock_df):
                        # Mock collect to return appropriate results
                        mock_validation_rows = [Mock() for _ in range(len(test_segments))]
                        for i, row in enumerate(mock_validation_rows):
                            row.segment_id = f'seg_{i:03d}'
                            row.original_quality_score = 0.85
                            row.refined_quality_score = 0.90
                            row.validation_issues = '[]'
                            row.refinement_applied = False
                            row.processing_time = 0.1
                        
                        mock_df.collect.return_value = mock_validation_rows
                        
                        validation_results = validation_engine.validate_text_segments(test_segments)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Calculate performance metrics
                throughput = data_size_mb / processing_time
                processing_time_per_mb = processing_time / data_size_mb
                
                performance_results.append({
                    'data_size_mb': data_size_mb,
                    'processing_time': processing_time,
                    'throughput_mb_per_second': throughput,
                    'processing_time_per_mb': processing_time_per_mb,
                    'segments_processed': len(validation_results)
                })
                
                # Verify performance thresholds
                thresholds = spark_performance_config['performance_thresholds']
                assert processing_time_per_mb <= thresholds['max_processing_time_per_mb'], \
                    f"Processing time per MB too high: {processing_time_per_mb}s"
                assert throughput >= thresholds['min_throughput_mb_per_second'], \
                    f"Throughput too low: {throughput} MB/s"
            
            # Verify performance scaling characteristics
            self._verify_spark_scaling_characteristics(performance_results)
    
    def test_spark_resource_optimization_performance(self, spark_performance_config):
        """Test Spark resource optimization performance."""
        
        with patch('src.spark.resource_optimizer.SparkSession') as mock_spark_class:
            with patch('src.spark.resource_optimizer.psutil') as mock_psutil:
                
                # Setup mocks
                mock_spark_session = Mock()
                mock_spark_class.getActiveSession.return_value = mock_spark_session
                
                # Mock system resources
                mock_psutil.cpu_count.return_value = 8
                mock_psutil.cpu_percent.return_value = 25.0
                mock_memory = Mock()
                mock_memory.total = 16 * 1024**3  # 16GB
                mock_memory.available = 12 * 1024**3  # 12GB
                mock_memory.percent = 25.0
                mock_psutil.virtual_memory.return_value = mock_memory
                
                from src.spark.resource_optimizer import SparkResourceOptimizer
                
                optimizer = SparkResourceOptimizer(mock_spark_session)
                optimization_results = []
                
                for config in spark_performance_config['executor_configs']:
                    for data_size in spark_performance_config['data_sizes']:
                        
                        start_time = time.time()
                        
                        # Test workload analysis
                        workload = optimizer.analyze_workload(
                            data_size_mb=data_size,
                            processing_type='text_validation',
                            estimated_complexity=5.0
                        )
                        
                        # Test optimization
                        allocation = optimizer.optimize_allocation(workload, 'balanced')
                        
                        # Test allocation application
                        success = optimizer.apply_allocation(allocation)
                        
                        end_time = time.time()
                        optimization_time = end_time - start_time
                        
                        optimization_results.append({
                            'data_size_mb': data_size,
                            'executor_config': config,
                            'optimization_time': optimization_time,
                            'allocation_success': success,
                            'executor_instances': allocation.executor_instances,
                            'executor_cores': allocation.executor_cores,
                            'executor_memory_gb': allocation.executor_memory_gb
                        })
                        
                        # Verify optimization time is reasonable
                        assert optimization_time < 1.0, f"Optimization took too long: {optimization_time}s"
                        assert success is True, "Allocation application failed"
                
                # Verify optimization results are sensible
                self._verify_optimization_results(optimization_results)
    
    def _generate_test_segments_for_size(self, target_size_mb: float) -> List[Dict[str, Any]]:
        """Generate test segments for a target data size."""
        
        # Estimate average segment size (1KB per segment)
        avg_segment_size_kb = 1.0
        target_segments = int((target_size_mb * 1024) / avg_segment_size_kb)
        
        generator = TestDataGenerator(seed=42)
        test_segments = []
        
        for i in range(target_segments):
            segment = {
                'segment_id': f'perf_seg_{i:06d}',
                'text_content': generator.generate_narrative_segment(50, 100),
                'speaker_id': 'narrator' if i % 3 == 0 else f'character_{(i % 2) + 1}',
                'segment_type': 'narrative' if i % 3 == 0 else 'dialogue',
                'quality_score': 0.85 + (i % 10) * 0.01,
                'confidence_score': 0.90 + (i % 10) * 0.005,
                'processing_metadata': json.dumps({'tokens': 15 + (i % 10), 'complexity': 3 + (i % 5)})
            }
            test_segments.append(segment)
        
        return test_segments
    
    def _verify_spark_scaling_characteristics(self, results: List[Dict[str, Any]]):
        """Verify that Spark processing scales appropriately with data size."""
        
        # Verify linear or sub-linear scaling
        data_sizes = [r['data_size_mb'] for r in results]
        processing_times = [r['processing_time'] for r in results]
        
        # Calculate scaling efficiency (should not degrade significantly)
        for i in range(1, len(results)):
            size_ratio = data_sizes[i] / data_sizes[i-1]
            time_ratio = processing_times[i] / processing_times[i-1]
            
            # Processing time should scale linearly or better
            scaling_efficiency = size_ratio / time_ratio
            assert scaling_efficiency >= 0.8, f"Poor scaling efficiency: {scaling_efficiency}"
    
    def _verify_optimization_results(self, results: List[Dict[str, Any]]):
        """Verify optimization results are sensible."""
        
        for result in results:
            # Verify resource allocation is reasonable
            assert result['executor_instances'] >= 1
            assert result['executor_instances'] <= 20  # Reasonable upper bound
            assert result['executor_cores'] >= 1
            assert result['executor_cores'] <= 8  # Reasonable upper bound
            assert result['executor_memory_gb'] >= 1
            assert result['executor_memory_gb'] <= 32  # Reasonable upper bound


@pytest.mark.performance
@pytest.mark.component
class TestKafkaComponentPerformance:
    """Performance tests for Kafka components."""
    
    def test_kafka_producer_throughput(self):
        """Test Kafka producer throughput performance."""
        
        with patch('src.kafka.producers.chunk_producer.KafkaProducer') as mock_producer_class:
            
            # Mock producer with timing simulation
            mock_producer = Mock()
            mock_producer_class.return_value = mock_producer
            
            # Simulate send latency
            def mock_send(*args, **kwargs):
                time.sleep(0.001)  # 1ms latency
                future = Mock()
                future.get.return_value = Mock()
                return future
            
            mock_producer.send.side_effect = mock_send
            mock_producer.flush.return_value = None
            
            from src.kafka.producers.chunk_producer import ChunkProducer
            
            producer = ChunkProducer()
            
            # Test different message sizes
            message_sizes = [1, 10, 100, 1000]  # KB
            throughput_results = []
            
            for size_kb in message_sizes:
                # Generate test chunk
                test_chunk = {
                    'chunk_id': f'perf_chunk_{size_kb}kb',
                    'job_id': 'perf_test',
                    'text_content': 'x' * (size_kb * 1024),  # Create data of specified size
                    'chunk_index': 0,
                    'total_chunks': 1
                }
                
                # Measure throughput over multiple sends
                num_messages = 100
                start_time = time.time()
                
                for i in range(num_messages):
                    producer.send_chunk(test_chunk)
                
                end_time = time.time()
                total_time = end_time - start_time
                
                messages_per_second = num_messages / total_time
                throughput_mb_per_second = (num_messages * size_kb) / (total_time * 1024)
                
                throughput_results.append({
                    'message_size_kb': size_kb,
                    'messages_per_second': messages_per_second,
                    'throughput_mb_per_second': throughput_mb_per_second,
                    'avg_latency_ms': (total_time / num_messages) * 1000
                })
                
                # Verify performance thresholds
                assert messages_per_second >= 50, f"Low throughput for {size_kb}KB: {messages_per_second} msg/s"
                assert throughput_mb_per_second >= 0.1, f"Low throughput: {throughput_mb_per_second} MB/s"
            
            # Verify throughput characteristics
            self._verify_kafka_throughput_characteristics(throughput_results)
    
    def test_kafka_consumer_processing_performance(self):
        """Test Kafka consumer processing performance."""
        
        with patch('src.kafka.consumers.text_extraction_consumer.KafkaConsumer') as mock_consumer_class:
            
            mock_consumer = Mock()
            mock_consumer_class.return_value = mock_consumer
            
            # Generate test messages
            test_messages = []
            for i in range(1000):
                mock_message = Mock()
                mock_message.value = json.dumps({
                    'chunk_id': f'test_chunk_{i:06d}',
                    'job_id': 'performance_test',
                    'text_content': f'This is test content for chunk {i}.' * 10,
                    'chunk_index': i,
                    'total_chunks': 1000
                }).encode('utf-8')
                mock_message.key = f'test_chunk_{i:06d}'.encode('utf-8')
                test_messages.append(mock_message)
            
            # Mock poll to return messages in batches
            batch_size = 50
            batches = [test_messages[i:i + batch_size] for i in range(0, len(test_messages), batch_size)]
            
            poll_responses = []
            for batch in batches:
                poll_responses.append({'test_topic': batch})
            poll_responses.append({})  # Empty response to end polling
            
            mock_consumer.poll.side_effect = poll_responses
            mock_consumer.commit.return_value = None
            
            from src.kafka.consumers.text_extraction_consumer import TextExtractionConsumer
            
            consumer = TextExtractionConsumer()
            
            # Measure processing performance
            start_time = time.time()
            processed_messages = 0
            
            # Mock message processing
            def mock_process_chunk(chunk_data):
                # Simulate processing time
                time.sleep(0.001)  # 1ms per message
                return {'processed': True}
            
            with patch.object(consumer, 'process_chunk', side_effect=mock_process_chunk):
                
                # Process all batches
                for batch_messages in poll_responses[:-1]:  # Skip empty response
                    if batch_messages:
                        for topic, messages in batch_messages.items():
                            for message in messages:
                                chunk_data = json.loads(message.value.decode('utf-8'))
                                consumer.process_chunk(chunk_data)
                                processed_messages += 1
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate performance metrics
            messages_per_second = processed_messages / total_time
            avg_processing_time_ms = (total_time / processed_messages) * 1000
            
            # Verify performance thresholds
            assert messages_per_second >= 100, f"Low consumer throughput: {messages_per_second} msg/s"
            assert avg_processing_time_ms <= 50, f"High processing latency: {avg_processing_time_ms}ms"
            assert processed_messages == len(test_messages), "Not all messages were processed"
    
    def _verify_kafka_throughput_characteristics(self, results: List[Dict[str, Any]]):
        """Verify Kafka throughput characteristics."""
        
        # Verify throughput scales appropriately with message size
        for i in range(1, len(results)):
            current = results[i]
            previous = results[i-1]
            
            # Larger messages should have lower message rate but similar or higher data rate
            size_ratio = current['message_size_kb'] / previous['message_size_kb']
            msg_rate_ratio = current['messages_per_second'] / previous['messages_per_second']
            
            # Message rate should decrease as size increases (due to overhead)
            assert msg_rate_ratio <= 1.2, f"Message rate increased unexpectedly: {msg_rate_ratio}"


@pytest.mark.performance
@pytest.mark.component
class TestCacheComponentPerformance:
    """Performance tests for caching components."""
    
    def test_redis_cache_performance(self):
        """Test Redis cache performance characteristics."""
        
        with patch('src.cache.redis_cache.redis') as mock_redis_module:
            
            # Mock Redis client with timing simulation
            mock_redis_client = Mock()
            mock_redis_module.Redis.return_value = mock_redis_client
            
            # Simulate Redis latency
            def mock_get(*args, **kwargs):
                time.sleep(0.0001)  # 0.1ms latency
                return None  # Cache miss
            
            def mock_setex(*args, **kwargs):
                time.sleep(0.0001)  # 0.1ms latency
                return True
            
            mock_redis_client.ping.return_value = True
            mock_redis_client.get.side_effect = mock_get
            mock_redis_client.setex.side_effect = mock_setex
            mock_redis_client.delete.return_value = 1
            
            from src.cache.redis_cache import RedisCacheManager
            
            cache_manager = RedisCacheManager(namespace='performance_test')
            
            # Test different data sizes
            data_sizes = [1, 10, 100, 1000]  # KB
            cache_results = []
            
            for size_kb in data_sizes:
                # Generate test data
                test_data = {
                    'content': 'x' * (size_kb * 1024),
                    'metadata': {'size_kb': size_kb, 'timestamp': time.time()}
                }
                
                # Test set performance
                num_operations = 100
                
                start_time = time.time()
                for i in range(num_operations):
                    success = cache_manager.set('test_data', f'key_{i}', test_data, ttl=3600)
                    assert success is True
                set_time = time.time() - start_time
                
                # Test get performance (cache miss scenario)
                start_time = time.time()
                for i in range(num_operations):
                    result = cache_manager.get('test_data', f'key_{i}')
                get_time = time.time() - start_time
                
                set_ops_per_second = num_operations / set_time
                get_ops_per_second = num_operations / get_time
                
                cache_results.append({
                    'data_size_kb': size_kb,
                    'set_ops_per_second': set_ops_per_second,
                    'get_ops_per_second': get_ops_per_second,
                    'avg_set_latency_ms': (set_time / num_operations) * 1000,
                    'avg_get_latency_ms': (get_time / num_operations) * 1000
                })
                
                # Verify performance thresholds
                assert set_ops_per_second >= 1000, f"Low cache set throughput: {set_ops_per_second} ops/s"
                assert get_ops_per_second >= 1000, f"Low cache get throughput: {get_ops_per_second} ops/s"
            
            # Verify cache performance characteristics
            self._verify_cache_performance_characteristics(cache_results)
    
    def _verify_cache_performance_characteristics(self, results: List[Dict[str, Any]]):
        """Verify cache performance characteristics."""
        
        # Verify performance degrades gracefully with data size
        for i in range(1, len(results)):
            current = results[i]
            previous = results[i-1]
            
            # Performance should not degrade dramatically
            set_perf_ratio = current['set_ops_per_second'] / previous['set_ops_per_second']
            get_perf_ratio = current['get_ops_per_second'] / previous['get_ops_per_second']
            
            assert set_perf_ratio >= 0.1, f"Cache set performance degraded too much: {set_perf_ratio}"
            assert get_perf_ratio >= 0.1, f"Cache get performance degraded too much: {get_perf_ratio}"


@pytest.mark.performance
@pytest.mark.component
class TestLLMComponentPerformance:
    """Performance tests for LLM processing components."""
    
    def test_llm_pool_performance(self):
        """Test LLM pool performance and resource utilization."""
        
        with patch('src.llm_pool.llm_pool_manager.get_pool_manager') as mock_get_pool:
            
            # Mock LLM pool manager
            mock_pool_manager = Mock()
            mock_get_pool.return_value = mock_pool_manager
            
            # Simulate LLM processing latency
            def mock_process_segment(segment):
                time.sleep(0.1)  # 100ms processing time
                return {
                    'segment_id': segment['segment_id'],
                    'speaker': 'character_1',
                    'confidence': 0.95
                }
            
            mock_pool_manager.process_segment.side_effect = mock_process_segment
            mock_pool_manager.get_pool_stats.return_value = {
                'active_instances': 2,
                'queue_size': 0,
                'total_requests': 0,
                'avg_response_time': 0.1
            }
            
            # Test concurrent LLM processing
            test_segments = []
            for i in range(50):
                segment = {
                    'segment_id': f'llm_perf_seg_{i:03d}',
                    'text_content': f'This is test segment {i} for LLM performance testing.',
                    'speaker': 'AMBIGUOUS',
                    'segment_type': 'dialogue'
                }
                test_segments.append(segment)
            
            # Test sequential processing
            start_time = time.time()
            sequential_results = []
            
            for segment in test_segments:
                result = mock_pool_manager.process_segment(segment)
                sequential_results.append(result)
            
            sequential_time = time.time() - start_time
            
            # Test concurrent processing simulation
            start_time = time.time()
            
            # Simulate concurrent processing with ThreadPoolExecutor
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            concurrent_results = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(mock_pool_manager.process_segment, segment)
                    for segment in test_segments
                ]
                
                for future in as_completed(futures):
                    result = future.result()
                    concurrent_results.append(result)
            
            concurrent_time = time.time() - start_time
            
            # Calculate performance metrics
            sequential_throughput = len(test_segments) / sequential_time
            concurrent_throughput = len(test_segments) / concurrent_time
            concurrency_speedup = sequential_time / concurrent_time
            
            # Verify performance characteristics
            assert len(sequential_results) == len(test_segments)
            assert len(concurrent_results) == len(test_segments)
            assert concurrent_throughput > sequential_throughput
            assert concurrency_speedup >= 2.0, f"Poor concurrency speedup: {concurrency_speedup}"
            
            # Verify throughput thresholds
            assert sequential_throughput >= 8, f"Low sequential throughput: {sequential_throughput} seg/s"
            assert concurrent_throughput >= 20, f"Low concurrent throughput: {concurrent_throughput} seg/s"


@pytest.mark.performance
@pytest.mark.component
class TestMonitoringPerformance:
    """Performance tests for monitoring and metrics components."""
    
    def test_metrics_collection_performance(self):
        """Test metrics collection performance overhead."""
        
        with patch('src.monitoring.prometheus_metrics.get_metrics_collector') as mock_get_metrics:
            
            mock_metrics = Mock()
            mock_get_metrics.return_value = mock_metrics
            
            # Simulate metrics collection overhead
            def mock_record_metric(*args, **kwargs):
                time.sleep(0.00001)  # 0.01ms overhead
            
            mock_metrics.record_processing_request.side_effect = mock_record_metric
            mock_metrics.record_processing_duration.side_effect = mock_record_metric
            mock_metrics.record_spark_job.side_effect = mock_record_metric
            mock_metrics.record_kafka_message.side_effect = mock_record_metric
            
            metrics_collector = mock_get_metrics()
            
            # Test metrics collection overhead
            num_operations = 10000
            
            # Baseline without metrics
            start_time = time.time()
            for i in range(num_operations):
                # Simulate application work
                result = i * 2 + 1
            baseline_time = time.time() - start_time
            
            # With metrics collection
            start_time = time.time()
            for i in range(num_operations):
                # Simulate application work
                result = i * 2 + 1
                
                # Collect metrics
                metrics_collector.record_processing_request('test_operation', 'started')
                metrics_collector.record_processing_duration('test_operation', 'component', 0.001)
            
            with_metrics_time = time.time() - start_time
            
            # Calculate overhead
            overhead_time = with_metrics_time - baseline_time
            overhead_percent = (overhead_time / baseline_time) * 100
            overhead_per_operation_us = (overhead_time / num_operations) * 1000000
            
            # Verify acceptable overhead
            assert overhead_percent <= 10, f"Metrics overhead too high: {overhead_percent}%"
            assert overhead_per_operation_us <= 50, f"Per-operation overhead too high: {overhead_per_operation_us}μs"
    
    def test_health_check_performance(self):
        """Test health check performance."""
        
        with patch('src.monitoring.health_checks.get_health_service') as mock_get_health:
            
            mock_health_service = Mock()
            mock_get_health.return_value = mock_health_service
            
            # Mock health check latency
            def mock_health_check():
                time.sleep(0.01)  # 10ms health check
                return {
                    'status': 'healthy',
                    'component': 'test_component',
                    'latency_ms': 10,
                    'timestamp': time.time()
                }
            
            mock_health_service.check_component_health.side_effect = lambda component: mock_health_check()
            mock_health_service.get_overall_health.return_value = {
                'status': 'healthy',
                'message': 'All components operational',
                'summary': {'healthy': 5, 'degraded': 0, 'unhealthy': 0, 'total': 5},
                'timestamp': time.time()
            }
            
            health_service = mock_get_health()
            
            # Test individual health checks
            components = ['kafka', 'spark', 'redis', 'llm_pool', 'validation']
            
            start_time = time.time()
            for component in components:
                health_result = health_service.check_component_health(component)
                assert health_result['status'] == 'healthy'
            individual_checks_time = time.time() - start_time
            
            # Test overall health check
            start_time = time.time()
            overall_health = health_service.get_overall_health()
            overall_check_time = time.time() - start_time
            
            # Verify performance thresholds
            avg_component_check_time = individual_checks_time / len(components)
            assert avg_component_check_time <= 0.05, f"Component health check too slow: {avg_component_check_time}s"
            assert overall_check_time <= 0.1, f"Overall health check too slow: {overall_check_time}s"
            assert overall_health['status'] == 'healthy'


def create_component_performance_test_suite() -> Dict[str, Callable]:
    """Create a comprehensive component performance test suite."""
    
    return {
        'spark_validation': TestSparkComponentPerformance().test_spark_text_structuring_performance,
        'spark_optimization': TestSparkComponentPerformance().test_spark_resource_optimization_performance,
        'kafka_producer': TestKafkaComponentPerformance().test_kafka_producer_throughput,
        'kafka_consumer': TestKafkaComponentPerformance().test_kafka_consumer_processing_performance,
        'redis_cache': TestCacheComponentPerformance().test_redis_cache_performance,
        'llm_pool': TestLLMComponentPerformance().test_llm_pool_performance,
        'metrics_collection': TestMonitoringPerformance().test_metrics_collection_performance,
        'health_checks': TestMonitoringPerformance().test_health_check_performance
    }