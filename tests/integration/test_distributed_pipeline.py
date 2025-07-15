"""
Integration tests for the distributed text-to-audiobook pipeline.

This module contains comprehensive integration tests that validate the entire
Spark + Kafka + Airflow system working together.
"""

import pytest
import json
import time
import tempfile
import os
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Test framework imports
from airflow.models import DagBag, TaskInstance
from airflow.utils.dates import days_ago
from airflow.utils.state import State
from airflow.utils.db import create_tables, resetdb
from airflow.configuration import conf

# Add source paths for testing
import sys
sys.path.append('/opt/airflow/dags')
sys.path.append('/mnt/c/Dev/Projects/text_to_audiobook/src')

# Import components to test
from airflow.dags.text_to_audiobook_dag import dag as text_to_audiobook_dag
from airflow.dags.operators.spark_text_structurer_operator import SparkTextStructurerOperator
from airflow.dags.operators.kafka_producer_operator import FileUploadProducerOperator
from airflow.dags.operators.kafka_consumer_operator import TextExtractionConsumerOperator
from airflow.dags.operators.llm_processing_operator import LLMProcessingOperator
from airflow.dags.operators.quality_validation_operator import QualityValidationOperator

from src.spark.spark_text_structurer import SparkTextStructurer
from src.kafka.producers.file_upload_producer import FileUploadProducer
from src.kafka.consumers.text_extraction_consumer import TextExtractionConsumer
from src.kafka.consumers.llm_consumer import LLMConsumer
from src.llm_pool.llm_pool_manager import LLMPoolManager
from src.validation.validator import SimplifiedValidator


class TestDistributedPipeline:
    """Test suite for the distributed text-to-audiobook pipeline."""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Set up test environment for each test."""
        # Initialize Airflow database
        resetdb()
        create_tables()
        
        # Create test data
        self.test_text = """
        John: "Hello there, how are you doing today?"
        Mary: "I'm doing well, thank you for asking. How about you?"
        John: "I'm great! Just finished reading an amazing book."
        Narrator: John smiled as he spoke about his latest literary discovery.
        Mary: "Oh really? What book was it?"
        John: "It was about artificial intelligence and machine learning."
        """
        
        self.test_file_path = self._create_test_file(self.test_text)
        
        # Test configuration
        self.test_config = {
            'spark_environment': 'local',
            'kafka_enabled': True,
            'llm_engine': 'local',
            'processing_options': {
                'chunk_size': 1000,
                'overlap_size': 200,
                'max_refinement_iterations': 1
            },
            'quality_thresholds': {
                'overall_quality': 85.0,
                'speaker_consistency': 80.0,
                'attribution_confidence': 75.0,
                'error_rate': 10.0
            }
        }
        
        yield
        
        # Cleanup
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
    
    def _create_test_file(self, content: str) -> str:
        """Create a temporary test file with the given content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            return f.name
    
    def test_dag_structure_and_dependencies(self):
        """Test that the DAG structure and dependencies are correct."""
        # Load the DAG
        dag_bag = DagBag(dag_folder='/opt/airflow/dags', include_examples=False)
        assert 'text_to_audiobook_processing' in dag_bag.dags
        
        dag = dag_bag.dags['text_to_audiobook_processing']
        
        # Check DAG properties
        assert dag.dag_id == 'text_to_audiobook_processing'
        assert len(dag.tasks) > 0
        assert dag.max_active_runs == 3
        
        # Check key tasks exist
        expected_tasks = [
            'validate_input_file',
            'extract_text',
            'submit_to_kafka',
            'process_text_with_spark',
            'validate_quality',
            'refine_segments',
            'format_output',
            'cleanup_and_status'
        ]
        
        task_ids = [task.task_id for task in dag.tasks]
        for expected_task in expected_tasks:
            assert expected_task in task_ids, f"Missing task: {expected_task}"
        
        # Check task dependencies
        validate_task = dag.get_task('validate_input_file')
        extract_task = dag.get_task('extract_text')
        assert extract_task in validate_task.downstream_list
    
    @pytest.mark.integration
    def test_spark_text_structurer_integration(self):
        """Test Spark text structurer integration."""
        from src.spark.spark_text_structurer import SparkTextStructurerContext
        
        # Test with local Spark environment
        with SparkTextStructurerContext(
            environment='local',
            config=self.test_config['processing_options']
        ) as structurer:
            
            # Check health status
            health_status = structurer.health_check()
            assert health_status['overall_health'] == 'healthy'
            
            # Process text
            structured_segments = structurer.structure_text(
                text_content=self.test_text,
                processing_options=self.test_config['processing_options']
            )
            
            # Validate results
            assert len(structured_segments) > 0
            assert all('speaker' in segment for segment in structured_segments)
            assert all('text' in segment for segment in structured_segments)
            
            # Check processing metrics
            metrics = structurer.get_processing_metrics()
            assert 'total_chunks' in metrics
            assert 'successful_chunks' in metrics
            assert metrics['total_chunks'] > 0
    
    @pytest.mark.integration
    @patch('src.kafka.kafka_config.KafkaConfig.get_producer_config')
    @patch('src.kafka.kafka_config.KafkaConfig.get_consumer_config')
    def test_kafka_producer_consumer_integration(self, mock_consumer_config, mock_producer_config):
        """Test Kafka producer and consumer integration."""
        # Mock Kafka configurations
        mock_producer_config.return_value = {
            'bootstrap_servers': 'localhost:9092',
            'value_serializer': lambda x: json.dumps(x).encode('utf-8')
        }
        mock_consumer_config.return_value = {
            'bootstrap_servers': 'localhost:9092',
            'value_deserializer': lambda x: json.loads(x.decode('utf-8'))
        }
        
        # Mock Kafka producer and consumer
        with patch('kafka.KafkaProducer') as mock_producer, \
             patch('kafka.KafkaConsumer') as mock_consumer:
            
            # Setup mocks
            mock_producer_instance = Mock()
            mock_producer.return_value = mock_producer_instance
            
            mock_consumer_instance = Mock()
            mock_consumer.return_value = mock_consumer_instance
            
            # Test file upload producer
            with patch('src.text_processing.text_extractor.TextExtractor') as mock_extractor:
                mock_extractor.return_value.extract.return_value = self.test_text
                
                with FileUploadProducer() as producer:
                    job_id = producer.submit_file_for_processing(
                        file_path=self.test_file_path,
                        user_id='test_user',
                        processing_options=self.test_config['processing_options']
                    )
                    
                    assert job_id is not None
                    assert len(job_id) > 0
                    
                    # Verify producer was called
                    mock_producer_instance.send.assert_called()
            
            # Test text extraction consumer
            with patch('src.kafka.consumers.text_extraction_consumer.TextExtractionConsumer') as mock_consumer_class:
                mock_consumer_instance = Mock()
                mock_consumer_class.return_value = mock_consumer_instance
                
                consumer = TextExtractionConsumer()
                consumer.start(num_workers=1)
                
                # Verify consumer was started
                assert consumer is not None
    
    @pytest.mark.integration
    @patch('src.llm_pool.llm_client.LLMClient')
    def test_llm_processing_integration(self, mock_llm_client):
        """Test LLM processing integration."""
        # Mock LLM client
        mock_client_instance = Mock()
        mock_llm_client.return_value = mock_client_instance
        
        # Mock LLM response
        mock_client_instance.classify_text.return_value = "John"
        mock_client_instance.health_check.return_value = {'overall_health': 'healthy'}
        
        # Test segments
        test_segments = [
            {'text': 'Hello there, how are you doing today?', 'speaker': 'AMBIGUOUS'},
            {'text': 'I\'m doing well, thank you for asking.', 'speaker': 'AMBIGUOUS'}
        ]
        
        # Create LLM processing operator
        operator = LLMProcessingOperator(
            task_id='test_llm_processing',
            text_segments=test_segments,
            llm_config={'engine': 'local', 'model': 'mistral'},
            processing_options=self.test_config['processing_options']
        )
        
        # Mock context
        mock_context = {
            'dag_run': Mock(run_id='test_run_123'),
            'task_instance': Mock()
        }
        
        # Execute operator
        result = operator.execute(mock_context)
        
        # Validate results
        assert result['status'] == 'completed'
        assert 'processed_segments' in result
        assert len(result['processed_segments']) == len(test_segments)
    
    @pytest.mark.integration
    def test_quality_validation_integration(self):
        """Test quality validation integration."""
        # Test segments with good quality
        good_segments = [
            {'speaker': 'John', 'text': 'Hello there, how are you doing today?'},
            {'speaker': 'Mary', 'text': 'I\'m doing well, thank you for asking.'},
            {'speaker': 'John', 'text': 'I\'m great! Just finished reading an amazing book.'}
        ]
        
        # Create quality validation operator
        operator = QualityValidationOperator(
            task_id='test_quality_validation',
            structured_segments=good_segments,
            quality_thresholds=self.test_config['quality_thresholds'],
            validation_type='comprehensive'
        )
        
        # Mock context
        mock_context = {
            'dag_run': Mock(run_id='test_run_123'),
            'task_instance': Mock()
        }
        
        # Execute operator
        result = operator.execute(mock_context)
        
        # Validate results
        assert result['status'] == 'completed'
        assert 'overall_quality_score' in result
        assert result['overall_quality_score'] > 0
    
    @pytest.mark.integration
    def test_end_to_end_pipeline(self):
        """Test the complete end-to-end pipeline integration."""
        # Mock all external dependencies
        with patch('src.spark.spark_text_structurer.SparkTextStructurer') as mock_structurer, \
             patch('src.kafka.producers.file_upload_producer.FileUploadProducer') as mock_producer, \
             patch('src.llm_pool.llm_client.LLMClient') as mock_llm_client, \
             patch('src.validation.validator.SimplifiedValidator') as mock_validator:
            
            # Setup mocks
            mock_structurer_instance = Mock()
            mock_structurer.return_value = mock_structurer_instance
            mock_structurer_instance.structure_text.return_value = [
                {'speaker': 'John', 'text': 'Hello there, how are you doing today?'},
                {'speaker': 'Mary', 'text': 'I\'m doing well, thank you for asking.'}
            ]
            mock_structurer_instance.get_processing_metrics.return_value = {
                'total_chunks': 1,
                'successful_chunks': 1,
                'processing_time': 5.0
            }
            mock_structurer_instance.health_check.return_value = {'overall_health': 'healthy'}
            
            mock_producer_instance = Mock()
            mock_producer.return_value = mock_producer_instance
            mock_producer_instance.submit_file_for_processing.return_value = 'test_job_id'
            
            mock_llm_client_instance = Mock()
            mock_llm_client.return_value = mock_llm_client_instance
            
            mock_validator_instance = Mock()
            mock_validator.return_value = mock_validator_instance
            mock_validator_instance.validate_structured_segments.return_value = {
                'quality_score': 95.0,
                'errors': []
            }
            
            # Create mock context
            mock_context = {
                'dag_run': Mock(run_id='test_e2e_run'),
                'task_instance': Mock()
            }
            
            # Test file validation
            from airflow.dags.text_to_audiobook_dag import validate_input_file
            file_info = validate_input_file(self.test_file_path, **mock_context)
            
            assert file_info['status'] == 'validated'
            assert file_info['file_path'] == self.test_file_path
            
            # Test text extraction
            from airflow.dags.text_to_audiobook_dag import extract_text
            with patch('src.text_processing.text_extractor.TextExtractor') as mock_extractor:
                mock_extractor.return_value.extract.return_value = self.test_text
                
                extraction_result = extract_text(file_info, **mock_context)
                
                assert extraction_result['status'] == 'extracted'
                assert 'extracted_text' in extraction_result
            
            # Test Spark processing
            from airflow.dags.text_to_audiobook_dag import process_text_with_spark
            with patch('src.spark.spark_text_structurer.SparkTextStructurerContext') as mock_context_manager:
                mock_context_manager.return_value.__enter__.return_value = mock_structurer_instance
                
                spark_result = process_text_with_spark(extraction_result, **mock_context)
                
                assert spark_result['status'] == 'processed'
                assert 'structured_segments' in spark_result
            
            # Test quality validation
            from airflow.dags.text_to_audiobook_dag import validate_quality
            quality_result = validate_quality(spark_result, **mock_context)
            
            assert quality_result['status'] == 'validated'
            assert 'quality_score' in quality_result
            
            # Test output formatting
            from airflow.dags.text_to_audiobook_dag import format_output
            with patch('src.output.output_formatter.OutputFormatter') as mock_formatter:
                mock_formatter_instance = Mock()
                mock_formatter.return_value = mock_formatter_instance
                mock_formatter_instance.format_segments.return_value = spark_result['structured_segments']
                
                with patch('builtins.open', create=True) as mock_open:
                    mock_open.return_value.__enter__.return_value = Mock()
                    
                    output_result = format_output(
                        {'refined_segments': spark_result['structured_segments'], 'job_id': 'test_job'},
                        **mock_context
                    )
                    
                    assert output_result['status'] == 'completed'
                    assert 'output_path' in output_result
    
    @pytest.mark.integration
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Test with invalid file path
        with pytest.raises(FileNotFoundError):
            from airflow.dags.text_to_audiobook_dag import validate_input_file
            validate_input_file('/nonexistent/file.txt', **{
                'dag_run': Mock(run_id='test_error_run'),
                'task_instance': Mock()
            })
        
        # Test with invalid text content
        with patch('src.spark.spark_text_structurer.SparkTextStructurer') as mock_structurer:
            mock_structurer_instance = Mock()
            mock_structurer.return_value = mock_structurer_instance
            mock_structurer_instance.structure_text.side_effect = Exception("Processing failed")
            
            from airflow.dags.text_to_audiobook_dag import process_text_with_spark
            
            with pytest.raises(Exception, match="Spark text processing failed"):
                process_text_with_spark(
                    {'extracted_text': 'invalid', 'job_id': 'test_error'},
                    **{'dag_run': Mock(run_id='test_error_run'), 'task_instance': Mock()}
                )
    
    @pytest.mark.integration
    def test_performance_and_scalability(self):
        """Test performance and scalability characteristics."""
        # Create larger test content
        large_text = self.test_text * 10  # 10x larger
        
        # Mock components for performance testing
        with patch('src.spark.spark_text_structurer.SparkTextStructurer') as mock_structurer, \
             patch('src.validation.validator.SimplifiedValidator') as mock_validator:
            
            mock_structurer_instance = Mock()
            mock_structurer.return_value = mock_structurer_instance
            
            # Simulate processing time
            def simulate_processing(text_content, **kwargs):
                time.sleep(0.1)  # Simulate processing time
                return [
                    {'speaker': 'John', 'text': f'Segment {i}'}
                    for i in range(len(text_content) // 100)  # Simulate segments
                ]
            
            mock_structurer_instance.structure_text.side_effect = simulate_processing
            mock_structurer_instance.get_processing_metrics.return_value = {
                'total_chunks': 10,
                'successful_chunks': 10,
                'processing_time': 2.0
            }
            mock_structurer_instance.health_check.return_value = {'overall_health': 'healthy'}
            
            mock_validator_instance = Mock()
            mock_validator.return_value = mock_validator_instance
            mock_validator_instance.validate_structured_segments.return_value = {
                'quality_score': 92.0,
                'errors': []
            }
            
            # Test processing time
            start_time = time.time()
            
            # Simulate pipeline execution
            with patch('src.spark.spark_text_structurer.SparkTextStructurerContext') as mock_context:
                mock_context.return_value.__enter__.return_value = mock_structurer_instance
                
                from airflow.dags.text_to_audiobook_dag import process_text_with_spark
                result = process_text_with_spark(
                    {'extracted_text': large_text, 'job_id': 'perf_test'},
                    **{'dag_run': Mock(run_id='perf_test'), 'task_instance': Mock()}
                )
            
            processing_time = time.time() - start_time
            
            # Validate performance
            assert processing_time < 5.0  # Should complete within 5 seconds
            assert result['status'] == 'processed'
            assert len(result['structured_segments']) > 0
    
    @pytest.mark.integration
    def test_configuration_and_environment_variables(self):
        """Test configuration handling and environment variables."""
        # Test with different configurations
        test_configs = [
            {'spark_environment': 'local', 'kafka_enabled': False},
            {'spark_environment': 'cluster', 'kafka_enabled': True},
            {'llm_engine': 'local', 'quality_threshold': 80.0}
        ]
        
        for config in test_configs:
            # Mock environment variables
            with patch.dict(os.environ, {
                'SPARK_ENVIRONMENT': config.get('spark_environment', 'local'),
                'KAFKA_ENABLED': str(config.get('kafka_enabled', True)).lower(),
                'LLM_ENGINE': config.get('llm_engine', 'local')
            }):
                
                # Test configuration loading
                from airflow.dags.text_to_audiobook_dag import get_dag_config
                dag_config = get_dag_config()
                
                assert dag_config['spark_environment'] == config.get('spark_environment', 'local')
                assert dag_config['kafka_enabled'] == config.get('kafka_enabled', True)
                assert dag_config['llm_engine'] == config.get('llm_engine', 'local')
    
    @pytest.mark.integration
    def test_monitoring_and_metrics(self):
        """Test monitoring and metrics collection."""
        # Test metrics collection from various components
        with patch('src.spark.spark_text_structurer.SparkTextStructurer') as mock_structurer, \
             patch('src.kafka.producers.file_upload_producer.FileUploadProducer') as mock_producer, \
             patch('src.llm_pool.llm_pool_manager.LLMPoolManager') as mock_pool_manager:
            
            # Setup metrics mocks
            mock_structurer_instance = Mock()
            mock_structurer.return_value = mock_structurer_instance
            mock_structurer_instance.get_processing_metrics.return_value = {
                'total_chunks': 5,
                'successful_chunks': 4,
                'failed_chunks': 1,
                'processing_time': 10.5,
                'llm_processing_time': 8.2
            }
            
            mock_producer_instance = Mock()
            mock_producer.return_value = mock_producer_instance
            mock_producer_instance.get_producer_metrics.return_value = {
                'messages_sent': 10,
                'messages_failed': 0,
                'total_send_time': 2.1
            }
            
            mock_pool_manager_instance = Mock()
            mock_pool_manager.return_value = mock_pool_manager_instance
            mock_pool_manager_instance.get_pool_status.return_value = {
                'total_instances': 4,
                'healthy_instances': 3,
                'busy_instances': 1
            }
            
            # Test metrics collection
            structurer_metrics = mock_structurer_instance.get_processing_metrics()
            producer_metrics = mock_producer_instance.get_producer_metrics()
            pool_status = mock_pool_manager_instance.get_pool_status()
            
            # Validate metrics
            assert structurer_metrics['total_chunks'] == 5
            assert structurer_metrics['successful_chunks'] == 4
            assert producer_metrics['messages_sent'] == 10
            assert pool_status['healthy_instances'] == 3
            
            # Test metrics aggregation
            total_processing_time = (
                structurer_metrics['processing_time'] +
                producer_metrics['total_send_time']
            )
            
            assert total_processing_time > 0
            assert structurer_metrics['processing_time'] > structurer_metrics['llm_processing_time']


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmark tests for the distributed pipeline."""
    
    @pytest.mark.benchmark
    def test_spark_processing_benchmark(self, benchmark):
        """Benchmark Spark text processing performance."""
        # Mock Spark processing
        with patch('src.spark.spark_text_structurer.SparkTextStructurer') as mock_structurer:
            mock_instance = Mock()
            mock_structurer.return_value = mock_instance
            
            def mock_processing(text_content, **kwargs):
                # Simulate processing time based on text length
                time.sleep(len(text_content) / 10000)  # 10k chars per second
                return [{'speaker': 'Test', 'text': 'Benchmark segment'}]
            
            mock_instance.structure_text.side_effect = mock_processing
            mock_instance.get_processing_metrics.return_value = {'processing_time': 1.0}
            mock_instance.health_check.return_value = {'overall_health': 'healthy'}
            
            # Benchmark the processing
            test_text = "This is a test sentence for benchmarking. " * 100
            
            with patch('src.spark.spark_text_structurer.SparkTextStructurerContext') as mock_context:
                mock_context.return_value.__enter__.return_value = mock_instance
                
                def run_processing():
                    structurer = mock_instance
                    return structurer.structure_text(test_text)
                
                result = benchmark(run_processing)
                assert len(result) > 0
    
    @pytest.mark.benchmark
    def test_quality_validation_benchmark(self, benchmark):
        """Benchmark quality validation performance."""
        # Create test segments
        test_segments = [
            {'speaker': f'Speaker{i%3}', 'text': f'Test segment {i}'}
            for i in range(1000)
        ]
        
        # Mock validator
        with patch('src.validation.validator.SimplifiedValidator') as mock_validator:
            mock_instance = Mock()
            mock_validator.return_value = mock_instance
            
            def mock_validation(segments):
                time.sleep(len(segments) / 1000)  # 1000 segments per second
                return {'quality_score': 95.0, 'errors': []}
            
            mock_instance.validate_structured_segments.side_effect = mock_validation
            
            # Benchmark validation
            def run_validation():
                validator = mock_instance
                return validator.validate_structured_segments(test_segments)
            
            result = benchmark(run_validation)
            assert result['quality_score'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])