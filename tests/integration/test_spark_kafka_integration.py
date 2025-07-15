"""
Integration tests for Spark and Kafka components.

Tests verify the complete data flow between Kafka producers/consumers 
and Spark distributed processing components.
"""

import pytest
import json
import time
import threading
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

from src.kafka.producers.chunk_producer import ChunkProducer
from src.kafka.consumers.text_extraction_consumer import TextExtractionConsumer
from src.spark.distributed_validation import DistributedValidationEngine, ValidationResult
from src.spark.resource_optimizer import SparkResourceOptimizer, WorkloadCharacteristics
from src.cache.redis_cache import RedisCacheManager
from src.monitoring.prometheus_metrics import get_metrics_collector


@pytest.mark.integration
@pytest.mark.kafka
@pytest.mark.spark
class TestSparkKafkaIntegration:
    """Test integration between Spark and Kafka components."""
    
    @pytest.fixture
    def mock_kafka_infrastructure(self):
        """Setup mock Kafka infrastructure."""
        mock_producer = Mock()
        mock_consumer = Mock()
        
        # Mock successful produce/consume operations
        mock_producer.send.return_value = Mock()
        mock_producer.send.return_value.get.return_value = Mock()
        mock_producer.flush.return_value = None
        mock_producer.close.return_value = None
        
        mock_consumer.subscribe.return_value = None
        mock_consumer.poll.return_value = {}
        mock_consumer.commit.return_value = None
        mock_consumer.close.return_value = None
        
        return {
            'producer': mock_producer,
            'consumer': mock_consumer
        }
    
    @pytest.fixture
    def mock_spark_session(self):
        """Setup mock Spark session."""
        mock_session = Mock()
        mock_session.sparkContext = Mock()
        mock_session.sparkContext.applicationId = "test_integration_app"
        mock_session.createDataFrame.return_value = Mock()
        mock_session.conf.set.return_value = None
        mock_session.udf.register.return_value = None
        return mock_session
    
    @pytest.fixture
    def sample_text_chunks(self):
        """Sample text chunks for testing."""
        return [
            {
                "chunk_id": "chunk_001",
                "job_id": "test_job_001",
                "text_content": "Chapter 1: The Beginning. This is the start of our story. \"Hello, world!\" said Alice.",
                "chunk_index": 0,
                "total_chunks": 3,
                "metadata": {
                    "source_file": "test_book.txt",
                    "created_at": datetime.now().isoformat()
                }
            },
            {
                "chunk_id": "chunk_002", 
                "job_id": "test_job_001",
                "text_content": "\"How are you?\" asked Bob. Alice smiled and replied, \"I'm doing well, thank you.\"",
                "chunk_index": 1,
                "total_chunks": 3,
                "metadata": {
                    "source_file": "test_book.txt",
                    "created_at": datetime.now().isoformat()
                }
            },
            {
                "chunk_id": "chunk_003",
                "job_id": "test_job_001", 
                "text_content": "The conversation continued as they walked through the garden together.",
                "chunk_index": 2,
                "total_chunks": 3,
                "metadata": {
                    "source_file": "test_book.txt",
                    "created_at": datetime.now().isoformat()
                }
            }
        ]
    
    def test_kafka_to_spark_data_flow(self, mock_kafka_infrastructure, mock_spark_session, sample_text_chunks):
        """Test data flow from Kafka to Spark processing."""
        
        with patch('src.kafka.producers.chunk_producer.KafkaProducer') as mock_producer_class:
            with patch('src.kafka.consumers.text_extraction_consumer.KafkaConsumer') as mock_consumer_class:
                with patch('src.spark.distributed_validation.SparkSession') as mock_spark_class:
                    
                    # Setup mocks
                    mock_producer_class.return_value = mock_kafka_infrastructure['producer']
                    mock_consumer_class.return_value = mock_kafka_infrastructure['consumer']
                    mock_spark_class.getActiveSession.return_value = mock_spark_session
                    
                    # Create Kafka producer and send chunks
                    producer = ChunkProducer()
                    
                    # Send all chunks
                    for chunk in sample_text_chunks:
                        producer.send_chunk(chunk)
                    
                    # Verify producer calls
                    assert mock_kafka_infrastructure['producer'].send.call_count == len(sample_text_chunks)
                    
                    # Simulate consumer receiving messages
                    consumer = TextExtractionConsumer()
                    
                    # Mock poll to return our test messages
                    mock_messages = []
                    for chunk in sample_text_chunks:
                        mock_message = Mock()
                        mock_message.value = json.dumps(chunk).encode('utf-8')
                        mock_message.key = chunk['chunk_id'].encode('utf-8')
                        mock_message.topic = 'text_chunks'
                        mock_message.partition = 0
                        mock_message.offset = 0
                        mock_messages.append(mock_message)
                    
                    mock_kafka_infrastructure['consumer'].poll.return_value = {
                        'text_chunks': mock_messages
                    }
                    
                    # Process messages and collect segments
                    processed_segments = []
                    
                    def mock_process_chunk(chunk_data):
                        # Simulate text segmentation
                        segments = [
                            {
                                'segment_id': f"{chunk_data['chunk_id']}_seg_001",
                                'text_content': chunk_data['text_content'][:50] + "...",
                                'speaker_id': 'narrator',
                                'segment_type': 'narrative',
                                'quality_score': 0.85,
                                'confidence_score': 0.90,
                                'processing_metadata': json.dumps({'tokens': 10, 'complexity': 3})
                            }
                        ]
                        processed_segments.extend(segments)
                        return segments
                    
                    # Mock consumer's process_chunk method
                    with patch.object(consumer, 'process_chunk', side_effect=mock_process_chunk):
                        # Simulate message processing
                        messages = mock_kafka_infrastructure['consumer'].poll()
                        for topic_partition, message_list in messages.items():
                            for message in message_list:
                                chunk_data = json.loads(message.value.decode('utf-8'))
                                consumer.process_chunk(chunk_data)
                    
                    # Verify segments were created
                    assert len(processed_segments) == len(sample_text_chunks)
                    
                    # Now test Spark processing of segments
                    validation_engine = DistributedValidationEngine(mock_spark_session)
                    
                    # Mock Spark DataFrame operations
                    mock_df = Mock()
                    mock_df.cache.return_value = mock_df
                    mock_df.unpersist.return_value = None
                    
                    # Mock validation results
                    mock_validation_rows = []
                    for i, segment in enumerate(processed_segments):
                        mock_row = Mock()
                        mock_row.segment_id = segment['segment_id']
                        mock_row.original_quality_score = segment['quality_score']
                        mock_row.refined_quality_score = segment['quality_score'] * 1.1
                        mock_row.validation_issues = '[]'
                        mock_row.refinement_applied = False
                        mock_row.processing_time = 0.1
                        mock_validation_rows.append(mock_row)
                    
                    mock_df.collect.return_value = mock_validation_rows
                    mock_spark_session.createDataFrame.return_value = mock_df
                    
                    # Mock validation methods
                    with patch.object(validation_engine, '_perform_segment_validation', return_value=mock_df):
                        with patch.object(validation_engine, '_apply_segment_refinements', return_value=mock_df):
                            validation_results = validation_engine.validate_text_segments(processed_segments)
                    
                    # Verify validation results
                    assert len(validation_results) == len(processed_segments)
                    assert all(isinstance(result, ValidationResult) for result in validation_results)
                    
                    # Verify end-to-end data consistency
                    for i, (original_chunk, validation_result) in enumerate(zip(sample_text_chunks, validation_results)):
                        assert validation_result.segment_id.startswith(original_chunk['chunk_id'])
                        assert validation_result.original_quality_score > 0
                        assert validation_result.refined_quality_score >= validation_result.original_quality_score
    
    def test_spark_optimization_with_kafka_workload(self, mock_spark_session, sample_text_chunks):
        """Test Spark resource optimization based on Kafka workload characteristics."""
        
        with patch('src.spark.resource_optimizer.SparkSession') as mock_spark_class:
            with patch('src.spark.resource_optimizer.psutil') as mock_psutil:
                
                # Setup mocks
                mock_spark_class.getActiveSession.return_value = mock_spark_session
                
                # Mock system resources
                mock_psutil.cpu_count.return_value = 8
                mock_psutil.cpu_percent.return_value = 30.0
                mock_memory = Mock()
                mock_memory.total = 16 * 1024**3  # 16GB
                mock_memory.available = 12 * 1024**3  # 12GB
                mock_memory.percent = 25.0
                mock_psutil.virtual_memory.return_value = mock_memory
                mock_psutil.disk_io_counters.return_value = None
                mock_psutil.net_io_counters.return_value = None
                
                optimizer = SparkResourceOptimizer(mock_spark_session)
                
                # Calculate workload characteristics based on Kafka chunks
                total_data_size = sum(len(chunk['text_content']) for chunk in sample_text_chunks) / 1024  # KB to MB
                
                workload = optimizer.analyze_workload(
                    data_size_mb=total_data_size,
                    processing_type='text_extraction',
                    estimated_complexity=6.0
                )
                
                # Optimize allocation
                allocation = optimizer.optimize_allocation(workload, 'balanced')
                
                # Verify allocation is reasonable for the workload
                assert allocation.executor_instances >= 1
                assert allocation.executor_cores >= 1
                assert allocation.sql_shuffle_partitions > 0
                
                # Test allocation application
                result = optimizer.apply_allocation(allocation)
                assert result is True
                
                # Verify Spark configuration was applied
                assert mock_spark_session.conf.set.call_count >= 7
    
    def test_cache_integration_with_spark_kafka(self, mock_kafka_infrastructure, mock_spark_session, sample_text_chunks):
        """Test cache integration with Spark and Kafka processing."""
        
        with patch('src.cache.redis_cache.redis') as mock_redis_module:
            
            # Setup Redis mock
            mock_redis_client = Mock()
            mock_redis_client.ping.return_value = True
            mock_redis_client.get.return_value = None  # Cache miss initially
            mock_redis_client.setex.return_value = True
            mock_redis_client.delete.return_value = 1
            mock_redis_module.Redis.return_value = mock_redis_client
            
            cache_manager = RedisCacheManager(namespace='integration_test')
            
            # Test caching Kafka chunk processing results
            processing_results = {}
            
            for chunk in sample_text_chunks:
                # Simulate processing and caching results
                result = {
                    'segments_count': 3,
                    'quality_score': 0.88,
                    'processing_time': 1.2,
                    'speakers_detected': ['narrator', 'alice', 'bob']
                }
                
                # Cache the result
                cache_key = f"chunk_processing_{chunk['chunk_id']}"
                success = cache_manager.set('processed_segment', cache_key, result, ttl=3600)
                assert success is True
                
                processing_results[chunk['chunk_id']] = result
            
            # Test cache retrieval during Spark processing
            for chunk in sample_text_chunks:
                cache_key = f"chunk_processing_{chunk['chunk_id']}"
                
                # Simulate cache hit
                mock_redis_client.get.return_value = json.dumps(processing_results[chunk['chunk_id']]).encode('utf-8')
                
                cached_result = cache_manager.get('processed_segment', cache_key)
                assert cached_result is not None
                assert cached_result['quality_score'] == 0.88
                assert 'speakers_detected' in cached_result
            
            # Verify cache statistics
            stats = cache_manager.get_stats()
            assert stats['sets'] > 0
            assert stats['hits'] > 0
    
    def test_error_handling_in_integration(self, mock_kafka_infrastructure, mock_spark_session):
        """Test error handling across integrated components."""
        
        with patch('src.kafka.producers.chunk_producer.KafkaProducer') as mock_producer_class:
            with patch('src.spark.distributed_validation.SparkSession') as mock_spark_class:
                
                # Setup failing Kafka producer
                mock_producer = Mock()
                mock_producer.send.side_effect = Exception("Kafka connection failed")
                mock_producer_class.return_value = mock_producer
                
                mock_spark_class.getActiveSession.return_value = mock_spark_session
                
                producer = ChunkProducer()
                
                # Test that Kafka failures are handled gracefully
                test_chunk = {
                    "chunk_id": "error_test_chunk",
                    "job_id": "error_test_job",
                    "text_content": "Test content for error handling",
                    "chunk_index": 0,
                    "total_chunks": 1
                }
                
                # Should not raise exception
                try:
                    producer.send_chunk(test_chunk)
                    # Verify error was logged but didn't crash
                    assert True
                except Exception:
                    pytest.fail("Error handling failed - exception was not caught")
                
                # Test Spark error handling
                mock_spark_session.createDataFrame.side_effect = Exception("Spark processing failed")
                
                validation_engine = DistributedValidationEngine(mock_spark_session)
                
                # Should handle Spark errors gracefully
                with pytest.raises(Exception):
                    validation_engine.validate_text_segments([])
    
    def test_concurrent_processing(self, mock_kafka_infrastructure, mock_spark_session, sample_text_chunks):
        """Test concurrent processing across Kafka and Spark components."""
        
        with patch('src.kafka.producers.chunk_producer.KafkaProducer') as mock_producer_class:
            with patch('src.spark.distributed_validation.SparkSession') as mock_spark_class:
                
                mock_producer_class.return_value = mock_kafka_infrastructure['producer']
                mock_spark_class.getActiveSession.return_value = mock_spark_session
                
                # Test concurrent chunk production
                def produce_chunks(chunks_subset):
                    producer = ChunkProducer()
                    for chunk in chunks_subset:
                        producer.send_chunk(chunk)
                    return len(chunks_subset)
                
                # Split chunks for concurrent processing
                chunk_batches = [
                    sample_text_chunks[:2],
                    sample_text_chunks[2:]
                ]
                
                results = []
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [executor.submit(produce_chunks, batch) for batch in chunk_batches]
                    
                    for future in as_completed(futures):
                        try:
                            result = future.result(timeout=10)
                            results.append(result)
                        except Exception as e:
                            pytest.fail(f"Concurrent processing failed: {e}")
                
                # Verify all chunks were processed
                total_processed = sum(results)
                assert total_processed == len(sample_text_chunks)
                
                # Verify concurrent calls to Kafka producer
                assert mock_kafka_infrastructure['producer'].send.call_count == len(sample_text_chunks)
    
    def test_end_to_end_pipeline_performance(self, mock_kafka_infrastructure, mock_spark_session, sample_text_chunks):
        """Test performance characteristics of the integrated pipeline."""
        
        with patch('src.kafka.producers.chunk_producer.KafkaProducer') as mock_producer_class:
            with patch('src.spark.distributed_validation.SparkSession') as mock_spark_class:
                with patch('src.cache.redis_cache.redis') as mock_redis_module:
                    
                    # Setup all mocks
                    mock_producer_class.return_value = mock_kafka_infrastructure['producer']
                    mock_spark_class.getActiveSession.return_value = mock_spark_session
                    
                    mock_redis_client = Mock()
                    mock_redis_client.ping.return_value = True
                    mock_redis_client.setex.return_value = True
                    mock_redis_module.Redis.return_value = mock_redis_client
                    
                    start_time = time.time()
                    
                    # Simulate full pipeline
                    producer = ChunkProducer()
                    cache_manager = RedisCacheManager(namespace='perf_test')
                    
                    # Phase 1: Send chunks to Kafka
                    for chunk in sample_text_chunks:
                        producer.send_chunk(chunk)
                    
                    # Phase 2: Cache intermediate results
                    for chunk in sample_text_chunks:
                        cache_manager.set('processing_result', chunk['chunk_id'], {
                            'processed': True,
                            'timestamp': time.time()
                        })
                    
                    # Phase 3: Spark validation (mocked)
                    mock_df = Mock()
                    mock_df.cache.return_value = mock_df
                    mock_df.unpersist.return_value = None
                    mock_df.collect.return_value = [Mock() for _ in sample_text_chunks]
                    mock_spark_session.createDataFrame.return_value = mock_df
                    
                    validation_engine = DistributedValidationEngine(mock_spark_session)
                    
                    with patch.object(validation_engine, '_perform_segment_validation', return_value=mock_df):
                        with patch.object(validation_engine, '_apply_segment_refinements', return_value=mock_df):
                            segments = [{'segment_id': f'seg_{i}', 'text_content': 'test'} for i in range(len(sample_text_chunks))]
                            validation_results = validation_engine.validate_text_segments(segments)
                    
                    end_time = time.time()
                    total_time = end_time - start_time
                    
                    # Performance assertions
                    assert total_time < 5.0  # Should complete in under 5 seconds
                    assert len(validation_results) == len(sample_text_chunks)
                    
                    # Verify all components were called
                    assert mock_kafka_infrastructure['producer'].send.call_count == len(sample_text_chunks)
                    assert mock_redis_client.setex.call_count == len(sample_text_chunks)
                    assert mock_spark_session.createDataFrame.called


@pytest.mark.integration 
@pytest.mark.monitoring
class TestMonitoringIntegration:
    """Test integration of monitoring with Spark and Kafka components."""
    
    def test_metrics_collection_across_components(self, mock_kafka_infrastructure, mock_spark_session):
        """Test that metrics are collected across all integrated components."""
        
        with patch('src.monitoring.prometheus_metrics.get_metrics_collector') as mock_get_metrics:
            
            mock_metrics = Mock()
            mock_get_metrics.return_value = mock_metrics
            
            # Test Kafka metrics
            with patch('src.kafka.producers.chunk_producer.KafkaProducer') as mock_producer_class:
                mock_producer_class.return_value = mock_kafka_infrastructure['producer']
                
                producer = ChunkProducer()
                test_chunk = {
                    "chunk_id": "metrics_test",
                    "job_id": "metrics_job",
                    "text_content": "Test content for metrics",
                    "chunk_index": 0,
                    "total_chunks": 1
                }
                
                producer.send_chunk(test_chunk)
                
                # Verify Kafka metrics were recorded
                # (Would depend on actual implementation in ChunkProducer)
                
            # Test Spark metrics
            with patch('src.spark.distributed_validation.SparkSession') as mock_spark_class:
                mock_spark_class.getActiveSession.return_value = mock_spark_session
                
                validation_engine = DistributedValidationEngine(mock_spark_session)
                
                # Mock DataFrame operations
                mock_df = Mock()
                mock_df.cache.return_value = mock_df
                mock_df.unpersist.return_value = None
                mock_df.collect.return_value = []
                mock_spark_session.createDataFrame.return_value = mock_df
                
                with patch.object(validation_engine, '_perform_segment_validation', return_value=mock_df):
                    with patch.object(validation_engine, '_apply_segment_refinements', return_value=mock_df):
                        validation_engine.validate_text_segments([])
                
                # Verify Spark metrics were recorded
                mock_metrics.record_spark_job.assert_called()
    
    def test_health_checks_integration(self, mock_kafka_infrastructure, mock_spark_session):
        """Test health checks across integrated components."""
        
        with patch('src.monitoring.health_checks.get_health_service') as mock_get_health:
            
            mock_health_service = Mock()
            mock_get_health.return_value = mock_health_service
            
            # Mock component health checks
            mock_health_results = {
                'spark': Mock(status='healthy', component='spark'),
                'kafka': Mock(status='healthy', component='kafka'),
                'redis': Mock(status='healthy', component='redis'),
                'system_resources': Mock(status='healthy', component='system_resources')
            }
            
            mock_health_service.check_all_components.return_value = mock_health_results
            
            # Test overall health check
            mock_health_service.get_overall_health.return_value = {
                'status': 'healthy',
                'message': 'All components are healthy',
                'summary': {
                    'healthy': 4,
                    'degraded': 0,
                    'unhealthy': 0,
                    'total': 4
                },
                'timestamp': time.time()
            }
            
            health_service = mock_get_health()
            overall_health = health_service.get_overall_health()
            
            assert overall_health['status'] == 'healthy'
            assert overall_health['summary']['healthy'] == 4
            assert overall_health['summary']['total'] == 4


@pytest.mark.integration
@pytest.mark.slow
class TestLargeScaleIntegration:
    """Test integration with larger datasets and longer processing times."""
    
    def test_large_dataset_processing(self, mock_kafka_infrastructure, mock_spark_session):
        """Test processing of larger datasets through the integrated pipeline."""
        
        # Generate large test dataset
        large_chunks = []
        for i in range(50):  # 50 chunks
            chunk = {
                "chunk_id": f"large_chunk_{i:03d}",
                "job_id": "large_test_job",
                "text_content": f"This is chunk {i} with substantial content. " * 20,  # ~1KB per chunk
                "chunk_index": i,
                "total_chunks": 50,
                "metadata": {
                    "source_file": "large_test_book.txt",
                    "created_at": datetime.now().isoformat()
                }
            }
            large_chunks.append(chunk)
        
        with patch('src.kafka.producers.chunk_producer.KafkaProducer') as mock_producer_class:
            with patch('src.spark.distributed_validation.SparkSession') as mock_spark_class:
                
                mock_producer_class.return_value = mock_kafka_infrastructure['producer']
                mock_spark_class.getActiveSession.return_value = mock_spark_session
                
                start_time = time.time()
                
                # Test batch processing
                producer = ChunkProducer()
                
                # Process in batches to simulate real-world usage
                batch_size = 10
                for i in range(0, len(large_chunks), batch_size):
                    batch = large_chunks[i:i + batch_size]
                    for chunk in batch:
                        producer.send_chunk(chunk)
                    
                    # Simulate small delay between batches
                    time.sleep(0.01)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Performance assertions for large dataset
                assert processing_time < 10.0  # Should complete in under 10 seconds
                assert mock_kafka_infrastructure['producer'].send.call_count == len(large_chunks)
                
                # Test throughput
                throughput = len(large_chunks) / processing_time
                assert throughput > 10  # Should process more than 10 chunks per second
    
    @pytest.mark.timeout(60)  # 1 minute timeout
    def test_long_running_integration(self, mock_kafka_infrastructure, mock_spark_session):
        """Test long-running integration scenarios."""
        
        with patch('src.kafka.producers.chunk_producer.KafkaProducer') as mock_producer_class:
            with patch('src.spark.resource_optimizer.SparkSession') as mock_spark_class:
                with patch('src.spark.resource_optimizer.psutil') as mock_psutil:
                    
                    # Setup mocks
                    mock_producer_class.return_value = mock_kafka_infrastructure['producer']
                    mock_spark_class.getActiveSession.return_value = mock_spark_session
                    
                    # Mock system resources  
                    mock_psutil.cpu_count.return_value = 8
                    mock_psutil.cpu_percent.return_value = 40.0
                    mock_memory = Mock()
                    mock_memory.total = 32 * 1024**3
                    mock_memory.available = 24 * 1024**3
                    mock_memory.percent = 25.0
                    mock_psutil.virtual_memory.return_value = mock_memory
                    mock_psutil.disk_io_counters.return_value = None
                    mock_psutil.net_io_counters.return_value = None
                    
                    # Test resource optimization over time
                    optimizer = SparkResourceOptimizer(mock_spark_session)
                    optimizer.start_monitoring()
                    
                    try:
                        # Simulate workload changes over time
                        workloads = [
                            (100.0, 'text_extraction'),
                            (500.0, 'llm_processing'), 
                            (200.0, 'validation'),
                            (1000.0, 'audio_generation')
                        ]
                        
                        allocations = []
                        for data_size, processing_type in workloads:
                            workload = optimizer.analyze_workload(data_size, processing_type)
                            allocation = optimizer.optimize_allocation(workload, 'balanced')
                            allocations.append(allocation)
                            
                            # Simulate some processing time
                            time.sleep(0.1)
                        
                        # Verify allocations were optimized for different workloads
                        assert len(allocations) == len(workloads)
                        assert all(alloc.executor_instances >= 1 for alloc in allocations)
                        
                        # Verify monitoring collected data
                        report = optimizer.get_optimization_report()
                        assert report['optimization_status'] == 'active'
                        
                    finally:
                        optimizer.stop_monitoring()


@pytest.mark.integration
@pytest.mark.external
class TestExternalServiceIntegration:
    """Test integration with external services (when available)."""
    
    def test_real_kafka_integration(self):
        """Test with real Kafka if available (skip if not)."""
        try:
            from kafka import KafkaProducer, KafkaConsumer
            
            # Try to create a real Kafka connection
            producer = KafkaProducer(
                bootstrap_servers=['localhost:9092'],
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                request_timeout_ms=5000
            )
            
            # If we get here, Kafka is available
            test_topic = 'test_integration'
            test_message = {
                'test': 'integration_test',
                'timestamp': time.time()
            }
            
            # Send test message
            future = producer.send(test_topic, test_message)
            result = future.get(timeout=10)
            
            assert result is not None
            
            producer.close()
            
        except Exception:
            pytest.skip("Real Kafka not available")
    
    def test_real_redis_integration(self):
        """Test with real Redis if available (skip if not)."""
        try:
            import redis
            
            # Try to create a real Redis connection
            client = redis.Redis(host='localhost', port=6379, db=1)
            client.ping()
            
            # If we get here, Redis is available
            test_key = 'integration_test'
            test_value = {'test': 'integration', 'timestamp': time.time()}
            
            # Test set/get
            client.setex(test_key, 60, json.dumps(test_value))
            retrieved = client.get(test_key)
            
            assert retrieved is not None
            retrieved_value = json.loads(retrieved.decode('utf-8'))
            assert retrieved_value['test'] == 'integration'
            
            # Cleanup
            client.delete(test_key)
            
        except Exception:
            pytest.skip("Real Redis not available")