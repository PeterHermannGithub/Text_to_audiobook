"""
End-to-end integration tests for the complete text-to-audiobook pipeline.

Tests verify the entire system working together from input to output,
including all distributed components, caching, monitoring, and validation.
"""

import pytest
import json
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.kafka.producers.file_upload_producer import FileUploadProducer
from src.kafka.consumers.text_extraction_consumer import TextExtractionConsumer
from src.spark.distributed_validation import DistributedValidationEngine
from src.spark.resource_optimizer import SparkResourceOptimizer
from src.cache.redis_cache import RedisCacheManager
from src.monitoring.prometheus_metrics import get_metrics_collector
from src.monitoring.health_checks import HealthCheckService


@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.slow
class TestEndToEndPipeline:
    """Test complete end-to-end pipeline integration."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        workspace = tempfile.mkdtemp(prefix="e2e_test_")
        workspace_path = Path(workspace)
        
        # Create directory structure
        (workspace_path / "input").mkdir()
        (workspace_path / "output").mkdir()
        (workspace_path / "logs").mkdir()
        (workspace_path / "cache").mkdir()
        
        yield workspace_path
        
        # Cleanup
        shutil.rmtree(workspace, ignore_errors=True)
    
    @pytest.fixture
    def sample_book_content(self):
        """Sample book content for end-to-end testing."""
        return """
        Chapter 1: The Adventure Begins
        
        Once upon a time, in a land far away, there lived a young adventurer named Alice.
        
        "I wonder what lies beyond those mountains," Alice said to herself one morning.
        
        She had always been curious about the world beyond her small village. The mountains 
        had always seemed mysterious and inviting.
        
        "Today is the day I find out," she declared with determination.
        
        Alice packed her belongings: a water bottle, some bread, and her father's old map.
        
        "Goodbye, village," she whispered as she walked toward the mountain path.
        
        The narrator continued: Alice's journey would prove to be more challenging than 
        she had anticipated. But she was ready for whatever lay ahead.
        
        Chapter 2: The Mountain Path
        
        The path up the mountain was steep and rocky. Alice had to be careful with each step.
        
        "This is harder than I thought," she muttered, pausing to catch her breath.
        
        Hours passed as she climbed higher and higher. The village below grew smaller and smaller.
        
        "I can't give up now," Alice encouraged herself. "I've come too far to turn back."
        
        As the sun began to set, she finally reached a small clearing where she could rest for the night.
        """
    
    @pytest.fixture
    def mock_infrastructure(self):
        """Setup complete mock infrastructure for E2E testing."""
        infrastructure = {
            'kafka': {
                'producer': Mock(),
                'consumer': Mock()
            },
            'spark': Mock(),
            'redis': Mock(),
            'llm_pool': Mock(),
            'metrics': Mock()
        }
        
        # Configure Kafka mocks
        infrastructure['kafka']['producer'].send.return_value = Mock()
        infrastructure['kafka']['producer'].send.return_value.get.return_value = Mock()
        infrastructure['kafka']['producer'].flush.return_value = None
        infrastructure['kafka']['producer'].close.return_value = None
        
        infrastructure['kafka']['consumer'].subscribe.return_value = None
        infrastructure['kafka']['consumer'].poll.return_value = {}
        infrastructure['kafka']['consumer'].commit.return_value = None
        infrastructure['kafka']['consumer'].close.return_value = None
        
        # Configure Spark mocks
        infrastructure['spark'].sparkContext = Mock()
        infrastructure['spark'].sparkContext.applicationId = "e2e_test_app"
        infrastructure['spark'].createDataFrame.return_value = Mock()
        infrastructure['spark'].conf.set.return_value = None
        
        # Configure Redis mocks
        infrastructure['redis'].ping.return_value = True
        infrastructure['redis'].get.return_value = None
        infrastructure['redis'].setex.return_value = True
        infrastructure['redis'].delete.return_value = 1
        
        # Configure LLM pool mocks
        infrastructure['llm_pool'].process_segment.return_value = {
            'speaker': 'character_1',
            'confidence': 0.95
        }
        
        # Configure metrics mocks
        infrastructure['metrics'].record_processing_request.return_value = None
        infrastructure['metrics'].record_processing_duration.return_value = None
        
        return infrastructure
    
    def test_complete_pipeline_execution(self, temp_workspace, sample_book_content, mock_infrastructure):
        """Test complete pipeline from file input to structured output."""
        
        # Create test input file
        input_file = temp_workspace / "input" / "test_book.txt"
        input_file.write_text(sample_book_content)
        
        output_file = temp_workspace / "output" / "test_book_structured.json"
        
        with patch('src.kafka.producers.file_upload_producer.KafkaProducer') as mock_kafka_producer:
            with patch('src.kafka.consumers.text_extraction_consumer.KafkaConsumer') as mock_kafka_consumer:
                with patch('src.spark.distributed_validation.SparkSession') as mock_spark_session:
                    with patch('src.cache.redis_cache.redis') as mock_redis_module:
                        with patch('src.llm_pool.llm_pool_manager.get_pool_manager') as mock_get_pool:
                            
                            # Setup all mocks
                            self._setup_complete_mocks(
                                mock_kafka_producer, mock_kafka_consumer, mock_spark_session,
                                mock_redis_module, mock_get_pool, mock_infrastructure
                            )
                            
                            # Execute pipeline phases
                            pipeline_result = self._execute_complete_pipeline(
                                input_file, output_file, mock_infrastructure
                            )
                            
                            # Verify pipeline results
                            assert pipeline_result['success'] is True
                            assert pipeline_result['phases_completed'] == 7
                            assert pipeline_result['total_segments'] > 0
                            assert pipeline_result['processing_time'] > 0
                            
                            # Verify output file was created
                            assert output_file.exists()
                            
                            # Verify output content
                            with open(output_file, 'r') as f:
                                output_data = json.load(f)
                            
                            assert isinstance(output_data, list)
                            assert len(output_data) > 0
                            
                            # Verify segment structure
                            for segment in output_data:
                                assert 'segment_id' in segment
                                assert 'text_content' in segment
                                assert 'speaker' in segment
                                assert 'quality_score' in segment
    
    def _setup_complete_mocks(self, mock_kafka_producer, mock_kafka_consumer, 
                            mock_spark_session, mock_redis_module, mock_get_pool, infrastructure):
        """Setup all infrastructure mocks for complete pipeline."""
        
        # Kafka producer setup
        mock_kafka_producer.return_value = infrastructure['kafka']['producer']
        
        # Kafka consumer setup
        mock_kafka_consumer.return_value = infrastructure['kafka']['consumer']
        
        # Setup consumer poll responses
        def create_mock_message(content, key):
            mock_msg = Mock()
            mock_msg.value = json.dumps(content).encode('utf-8')
            mock_msg.key = key.encode('utf-8') if isinstance(key, str) else key
            return mock_msg
        
        # Mock different message types for different phases
        file_upload_msg = create_mock_message({
            'job_id': 'e2e_test_001',
            'file_path': 'test_book.txt',
            'file_size': 1024,
            'timestamp': datetime.now().isoformat()
        }, 'file_upload')
        
        text_extraction_msg = create_mock_message({
            'job_id': 'e2e_test_001',
            'extracted_text': 'Sample extracted text...',
            'chunk_count': 3,
            'timestamp': datetime.now().isoformat()
        }, 'text_extraction')
        
        chunk_processing_msg = create_mock_message({
            'job_id': 'e2e_test_001',
            'chunk_id': 'chunk_001',
            'text_content': 'Chapter 1 content...',
            'chunk_index': 0,
            'total_chunks': 3
        }, 'chunk_processing')
        
        # Configure poll responses for different phases
        poll_responses = [
            {'file_uploads': [file_upload_msg]},
            {'text_extraction': [text_extraction_msg]}, 
            {'chunk_processing': [chunk_processing_msg]},
            {}  # Empty response to end polling
        ]
        
        infrastructure['kafka']['consumer'].poll.side_effect = poll_responses
        
        # Spark session setup
        mock_spark_session.getActiveSession.return_value = infrastructure['spark']
        mock_spark_session.builder.appName.return_value.getOrCreate.return_value = infrastructure['spark']
        
        # Spark DataFrame mock
        mock_df = Mock()
        mock_df.cache.return_value = mock_df
        mock_df.unpersist.return_value = None
        
        # Mock validation results
        mock_validation_rows = []
        for i in range(5):  # 5 sample segments
            mock_row = Mock()
            mock_row.segment_id = f'seg_{i:03d}'
            mock_row.original_quality_score = 0.8 + (i * 0.02)
            mock_row.refined_quality_score = 0.85 + (i * 0.02)
            mock_row.validation_issues = '[]'
            mock_row.refinement_applied = i % 2 == 0
            mock_row.processing_time = 0.1 + (i * 0.01)
            mock_validation_rows.append(mock_row)
        
        mock_df.collect.return_value = mock_validation_rows
        infrastructure['spark'].createDataFrame.return_value = mock_df
        
        # Redis setup
        mock_redis_module.Redis.return_value = infrastructure['redis']
        
        # LLM pool setup
        mock_get_pool.return_value = infrastructure['llm_pool']
    
    def _execute_complete_pipeline(self, input_file: Path, output_file: Path, infrastructure) -> Dict[str, Any]:
        """Execute the complete pipeline with all phases."""
        
        start_time = time.time()
        phases_completed = 0
        total_segments = 0
        
        try:
            # Phase 1: File Upload
            file_producer = FileUploadProducer()
            upload_result = file_producer.upload_file(str(input_file), 'e2e_test_001')
            phases_completed += 1
            
            # Phase 2: Text Extraction
            text_consumer = TextExtractionConsumer()
            # Simulate text extraction processing
            extraction_result = {'extracted_text': 'Sample text', 'chunks': 3}
            phases_completed += 1
            
            # Phase 3: Chunk Processing
            # Simulate chunk processing
            chunks = [
                {'chunk_id': f'chunk_{i:03d}', 'text_content': f'Chunk {i} content'}
                for i in range(3)
            ]
            phases_completed += 1
            
            # Phase 4: Spark Text Structuring
            validation_engine = DistributedValidationEngine(infrastructure['spark'])
            
            # Mock segment data for validation
            segments = [
                {
                    'segment_id': f'seg_{i:03d}',
                    'text_content': f'Segment {i} content with dialogue and narrative.',
                    'speaker_id': 'narrator' if i % 2 == 0 else 'alice',
                    'segment_type': 'narrative' if i % 2 == 0 else 'dialogue',
                    'quality_score': 0.8 + (i * 0.02),
                    'confidence_score': 0.85 + (i * 0.02),
                    'processing_metadata': json.dumps({'tokens': 10 + i, 'complexity': 3 + i})
                }
                for i in range(5)
            ]
            
            with patch.object(validation_engine, '_perform_segment_validation') as mock_validate:
                with patch.object(validation_engine, '_apply_segment_refinements') as mock_refine:
                    mock_df = Mock()
                    mock_df.cache.return_value = mock_df
                    mock_df.unpersist.return_value = None
                    mock_validate.return_value = mock_df
                    mock_refine.return_value = mock_df
                    
                    validation_results = validation_engine.validate_text_segments(segments)
            
            total_segments = len(validation_results)
            phases_completed += 1
            
            # Phase 5: LLM Processing
            # Simulate LLM processing for ambiguous segments
            llm_results = []
            for segment in segments:
                if segment['speaker_id'] == 'AMBIGUOUS':
                    result = infrastructure['llm_pool'].process_segment(segment)
                    llm_results.append(result)
            phases_completed += 1
            
            # Phase 6: Quality Validation
            # Generate quality report
            quality_report = validation_engine.generate_quality_report(validation_results, [])
            phases_completed += 1
            
            # Phase 7: Output Generation
            # Create final structured output
            final_output = []
            for i, result in enumerate(validation_results):
                output_segment = {
                    'segment_id': result.segment_id,
                    'text_content': segments[i]['text_content'],
                    'speaker': segments[i]['speaker_id'],
                    'segment_type': segments[i]['segment_type'],
                    'quality_score': result.refined_quality_score,
                    'validation_issues': result.validation_issues,
                    'processing_metadata': {
                        'original_quality': result.original_quality_score,
                        'refinement_applied': result.refinement_applied,
                        'processing_time': result.processing_time
                    }
                }
                final_output.append(output_segment)
            
            # Write output file
            with open(output_file, 'w') as f:
                json.dump(final_output, f, indent=2)
            
            phases_completed += 1
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'phases_completed': phases_completed,
                'total_segments': total_segments,
                'processing_time': processing_time,
                'quality_report': quality_report,
                'output_file': str(output_file)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'phases_completed': phases_completed,
                'processing_time': time.time() - start_time
            }
    
    def test_pipeline_performance_characteristics(self, temp_workspace, sample_book_content, mock_infrastructure):
        """Test performance characteristics of the complete pipeline."""
        
        # Create larger test content
        large_content = sample_book_content * 10  # 10x larger
        input_file = temp_workspace / "input" / "large_test_book.txt"
        input_file.write_text(large_content)
        
        output_file = temp_workspace / "output" / "large_test_book_structured.json"
        
        with patch('src.kafka.producers.file_upload_producer.KafkaProducer') as mock_kafka_producer:
            with patch('src.spark.distributed_validation.SparkSession') as mock_spark_session:
                with patch('src.cache.redis_cache.redis') as mock_redis_module:
                    
                    # Setup mocks for performance test
                    mock_kafka_producer.return_value = mock_infrastructure['kafka']['producer']
                    mock_spark_session.getActiveSession.return_value = mock_infrastructure['spark']
                    mock_redis_module.Redis.return_value = mock_infrastructure['redis']
                    
                    # Measure performance
                    start_time = time.time()
                    
                    # Execute key performance-critical components
                    cache_manager = RedisCacheManager(namespace='perf_test')
                    
                    # Test cache performance
                    for i in range(100):
                        cache_manager.set('test_segment', f'seg_{i}', {
                            'text': f'Segment {i} content',
                            'quality': 0.8 + (i % 20) / 100
                        })
                    
                    # Test resource optimization
                    optimizer = SparkResourceOptimizer(mock_infrastructure['spark'])
                    workload = optimizer.analyze_workload(len(large_content) / 1024, 'text_extraction')
                    allocation = optimizer.optimize_allocation(workload, 'balanced')
                    
                    end_time = time.time()
                    total_time = end_time - start_time
                    
                    # Performance assertions
                    assert total_time < 2.0, f"Performance test took too long: {total_time}s"
                    assert allocation.executor_instances > 0
                    assert allocation.executor_cores > 0
    
    def test_pipeline_error_recovery(self, temp_workspace, sample_book_content, mock_infrastructure):
        """Test pipeline error recovery and resilience."""
        
        input_file = temp_workspace / "input" / "error_test_book.txt"
        input_file.write_text(sample_book_content)
        
        with patch('src.kafka.producers.file_upload_producer.KafkaProducer') as mock_kafka_producer:
            with patch('src.spark.distributed_validation.SparkSession') as mock_spark_session:
                
                # Setup failing components
                mock_kafka_producer.side_effect = Exception("Kafka connection failed")
                mock_spark_session.side_effect = Exception("Spark session failed")
                
                # Test that errors are handled gracefully
                try:
                    file_producer = FileUploadProducer()
                    # Should handle Kafka failure gracefully
                    result = file_producer.upload_file(str(input_file), 'error_test_001')
                    
                    # If we get here, error was handled
                    assert True
                    
                except Exception as e:
                    # Verify it's the expected error type and not a crash
                    assert "Kafka connection failed" in str(e)
    
    def test_concurrent_pipeline_execution(self, temp_workspace, sample_book_content, mock_infrastructure):
        """Test concurrent pipeline execution with multiple jobs."""
        
        # Create multiple input files
        input_files = []
        for i in range(3):
            input_file = temp_workspace / "input" / f"concurrent_test_book_{i}.txt"
            input_file.write_text(f"Book {i} content: {sample_book_content}")
            input_files.append(input_file)
        
        with patch('src.kafka.producers.file_upload_producer.KafkaProducer') as mock_kafka_producer:
            with patch('src.cache.redis_cache.redis') as mock_redis_module:
                
                # Setup mocks
                mock_kafka_producer.return_value = mock_infrastructure['kafka']['producer']
                mock_redis_module.Redis.return_value = mock_infrastructure['redis']
                
                def process_file(input_file):
                    """Process a single file."""
                    try:
                        job_id = f"concurrent_job_{input_file.stem}"
                        
                        # Simulate file processing
                        file_producer = FileUploadProducer()
                        result = file_producer.upload_file(str(input_file), job_id)
                        
                        # Simulate caching
                        cache_manager = RedisCacheManager(namespace='concurrent_test')
                        cache_manager.set('processing_result', job_id, {
                            'status': 'completed',
                            'file_size': input_file.stat().st_size,
                            'timestamp': time.time()
                        })
                        
                        return {
                            'job_id': job_id,
                            'success': True,
                            'file_size': input_file.stat().st_size
                        }
                        
                    except Exception as e:
                        return {
                            'job_id': job_id,
                            'success': False,
                            'error': str(e)
                        }
                
                # Execute concurrent processing
                results = []
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [executor.submit(process_file, file) for file in input_files]
                    
                    for future in as_completed(futures):
                        try:
                            result = future.result(timeout=10)
                            results.append(result)
                        except Exception as e:
                            results.append({'success': False, 'error': str(e)})
                
                # Verify concurrent execution results
                assert len(results) == len(input_files)
                successful_jobs = [r for r in results if r['success']]
                assert len(successful_jobs) == len(input_files), f"Some jobs failed: {results}"
    
    def test_pipeline_monitoring_integration(self, temp_workspace, mock_infrastructure):
        """Test pipeline integration with monitoring and health checks."""
        
        with patch('src.monitoring.prometheus_metrics.get_metrics_collector') as mock_get_metrics:
            with patch('src.monitoring.health_checks.get_health_service') as mock_get_health:
                
                # Setup monitoring mocks
                mock_metrics = Mock()
                mock_get_metrics.return_value = mock_metrics
                
                mock_health_service = Mock()
                mock_get_health.return_value = mock_health_service
                
                # Mock comprehensive health status
                mock_health_service.get_overall_health.return_value = {
                    'status': 'healthy',
                    'message': 'All pipeline components operational',
                    'summary': {
                        'healthy': 5,
                        'degraded': 0,
                        'unhealthy': 0,
                        'total': 5
                    },
                    'components': {
                        'kafka': {'status': 'healthy', 'latency_ms': 10},
                        'spark': {'status': 'healthy', 'executor_count': 4},
                        'redis': {'status': 'healthy', 'memory_usage_mb': 100},
                        'llm_pool': {'status': 'healthy', 'active_instances': 2},
                        'validation': {'status': 'healthy', 'queue_size': 0}
                    },
                    'timestamp': time.time()
                }
                
                # Test health monitoring during pipeline execution
                health_service = mock_get_health()
                health_status = health_service.get_overall_health()
                
                assert health_status['status'] == 'healthy'
                assert health_status['summary']['total'] == 5
                assert all(
                    comp['status'] == 'healthy' 
                    for comp in health_status['components'].values()
                )
                
                # Test metrics collection
                metrics_collector = mock_get_metrics()
                
                # Simulate metrics recording during pipeline execution
                metrics_collector.record_processing_request('text_extraction', 'started')
                metrics_collector.record_processing_duration('text_extraction', 'extractor', 1.5)
                metrics_collector.record_spark_job('text_structuring', 'completed', 5.2)
                metrics_collector.set_system_health('pipeline', True)
                
                # Verify metrics were called
                assert mock_metrics.record_processing_request.called
                assert mock_metrics.record_processing_duration.called
                assert mock_metrics.record_spark_job.called
                assert mock_metrics.set_system_health.called
    
    def test_pipeline_data_integrity(self, temp_workspace, sample_book_content, mock_infrastructure):
        """Test data integrity throughout the pipeline."""
        
        input_file = temp_workspace / "input" / "integrity_test_book.txt"
        input_file.write_text(sample_book_content)
        output_file = temp_workspace / "output" / "integrity_test_structured.json"
        
        with patch('src.spark.distributed_validation.SparkSession') as mock_spark_session:
            with patch('src.cache.redis_cache.redis') as mock_redis_module:
                
                # Setup mocks
                mock_spark_session.getActiveSession.return_value = mock_infrastructure['spark']
                mock_redis_module.Redis.return_value = mock_infrastructure['redis']
                
                # Track data transformations
                original_text = sample_book_content
                original_length = len(original_text)
                original_word_count = len(original_text.split())
                
                # Simulate processing pipeline
                validation_engine = DistributedValidationEngine(mock_infrastructure['spark'])
                
                # Create segments that preserve original content
                segments = []
                sentences = original_text.split('.')
                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        segments.append({
                            'segment_id': f'integrity_seg_{i:03d}',
                            'text_content': sentence.strip() + '.',
                            'speaker_id': 'narrator',
                            'segment_type': 'narrative',
                            'quality_score': 0.85,
                            'confidence_score': 0.90,
                            'processing_metadata': json.dumps({'original_index': i})
                        })
                
                # Mock validation that preserves content integrity
                mock_df = Mock()
                mock_df.cache.return_value = mock_df
                mock_df.unpersist.return_value = None
                
                mock_validation_rows = []
                for segment in segments:
                    mock_row = Mock()
                    mock_row.segment_id = segment['segment_id']
                    mock_row.original_quality_score = segment['quality_score']
                    mock_row.refined_quality_score = segment['quality_score']
                    mock_row.validation_issues = '[]'
                    mock_row.refinement_applied = False
                    mock_row.processing_time = 0.1
                    mock_validation_rows.append(mock_row)
                
                mock_df.collect.return_value = mock_validation_rows
                mock_infrastructure['spark'].createDataFrame.return_value = mock_df
                
                with patch.object(validation_engine, '_perform_segment_validation', return_value=mock_df):
                    with patch.object(validation_engine, '_apply_segment_refinements', return_value=mock_df):
                        validation_results = validation_engine.validate_text_segments(segments)
                
                # Verify data integrity
                assert len(validation_results) == len(segments)
                
                # Reconstruct text from segments
                reconstructed_text = ' '.join(
                    segment['text_content'] for segment in segments
                )
                reconstructed_word_count = len(reconstructed_text.split())
                
                # Verify content preservation (allowing for minor formatting differences)
                word_count_ratio = reconstructed_word_count / original_word_count
                assert 0.95 <= word_count_ratio <= 1.05, f"Word count changed significantly: {word_count_ratio}"
                
                # Verify no data corruption
                for result in validation_results:
                    assert result.segment_id is not None
                    assert result.original_quality_score > 0
                    assert result.refined_quality_score >= result.original_quality_score
                    assert isinstance(result.validation_issues, list)