"""
Unit tests for custom Airflow operators.

This module contains unit tests for the custom operators used in the
text-to-audiobook pipeline.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add source paths for testing
import sys
sys.path.append('/opt/airflow/dags')
sys.path.append('/mnt/c/Dev/Projects/text_to_audiobook/src')

from airflow.dags.operators.spark_text_structurer_operator import SparkTextStructurerOperator
from airflow.dags.operators.kafka_producer_operator import KafkaProducerOperator
from airflow.dags.operators.kafka_consumer_operator import KafkaConsumerOperator
from airflow.dags.operators.llm_processing_operator import LLMProcessingOperator
from airflow.dags.operators.quality_validation_operator import QualityValidationOperator


class TestSparkTextStructurerOperator:
    """Test SparkTextStructurerOperator functionality."""
    
    def test_operator_initialization(self):
        """Test operator initialization with various parameters."""
        operator = SparkTextStructurerOperator(
            task_id='test_spark_operator',
            text_content='Test content',
            processing_options={'chunk_size': 1000},
            spark_environment='local'
        )
        
        assert operator.task_id == 'test_spark_operator'
        assert operator.text_content == 'Test content'
        assert operator.processing_options == {'chunk_size': 1000}
        assert operator.spark_environment == 'local'
    
    @patch('src.spark.spark_text_structurer.SparkTextStructurerContext')
    def test_operator_execution(self, mock_context):
        """Test operator execution with mocked SparkTextStructurer."""
        # Setup mocks
        mock_structurer = Mock()
        mock_context.return_value.__enter__.return_value = mock_structurer
        
        mock_structurer.health_check.return_value = {'overall_health': 'healthy'}
        mock_structurer.structure_text.return_value = [
            {'speaker': 'John', 'text': 'Hello'},
            {'speaker': 'Mary', 'text': 'Hi there'}
        ]
        mock_structurer.get_processing_metrics.return_value = {
            'total_chunks': 1,
            'successful_chunks': 1,
            'processing_time': 2.5
        }
        
        # Create operator
        operator = SparkTextStructurerOperator(
            task_id='test_spark_operator',
            text_content='John: Hello\nMary: Hi there',
            processing_options={'chunk_size': 1000},
            spark_environment='local'
        )
        
        # Create mock context
        mock_context_dict = {
            'dag_run': Mock(run_id='test_run_123'),
            'task_instance': Mock()
        }
        
        # Execute operator
        result = operator.execute(mock_context_dict)
        
        # Verify results
        assert result['status'] == 'completed'
        assert result['job_id'] == 'test_run_123'
        assert result['total_segments'] == 2
        assert 'structured_segments' in result
        assert 'processing_metrics' in result
        
        # Verify mocks were called
        mock_structurer.health_check.assert_called_once()
        mock_structurer.structure_text.assert_called_once()
        mock_structurer.get_processing_metrics.assert_called_once()
    
    @patch('src.spark.spark_text_structurer.SparkTextStructurerContext')
    def test_operator_health_check_failure(self, mock_context):
        """Test operator behavior when health check fails."""
        # Setup mocks
        mock_structurer = Mock()
        mock_context.return_value.__enter__.return_value = mock_structurer
        
        mock_structurer.health_check.return_value = {'overall_health': 'unhealthy'}
        
        # Create operator
        operator = SparkTextStructurerOperator(
            task_id='test_spark_operator',
            text_content='Test content',
            spark_environment='local'
        )
        
        # Create mock context
        mock_context_dict = {
            'dag_run': Mock(run_id='test_run_123'),
            'task_instance': Mock()
        }
        
        # Execute operator and expect failure
        with pytest.raises(Exception, match="System health check failed"):
            operator.execute(mock_context_dict)


class TestKafkaProducerOperator:
    """Test KafkaProducerOperator functionality."""
    
    def test_operator_initialization(self):
        """Test operator initialization."""
        operator = KafkaProducerOperator(
            task_id='test_kafka_producer',
            topic='test_topic',
            message_data={'key': 'value'},
            key='test_key',
            message_type='test_message'
        )
        
        assert operator.task_id == 'test_kafka_producer'
        assert operator.topic == 'test_topic'
        assert operator.message_data == {'key': 'value'}
        assert operator.key == 'test_key'
        assert operator.message_type == 'test_message'
    
    @patch('kafka.KafkaProducer')
    @patch('src.kafka.kafka_config.KafkaConfig')
    def test_operator_execution(self, mock_config, mock_producer):
        """Test operator execution with mocked Kafka producer."""
        # Setup mocks
        mock_config.get_producer_config.return_value = {
            'bootstrap_servers': 'localhost:9092'
        }
        
        mock_producer_instance = Mock()
        mock_producer.return_value = mock_producer_instance
        
        mock_future = Mock()
        mock_metadata = Mock()
        mock_metadata.partition = 0
        mock_metadata.offset = 123
        mock_future.get.return_value = mock_metadata
        mock_producer_instance.send.return_value = mock_future
        
        # Create operator
        operator = KafkaProducerOperator(
            task_id='test_kafka_producer',
            topic='test_topic',
            message_data={'key': 'value'},
            key='test_key'
        )
        
        # Create mock context
        mock_context_dict = {
            'dag_run': Mock(run_id='test_run_123'),
            'task_instance': Mock()
        }
        
        # Execute operator
        result = operator.execute(mock_context_dict)
        
        # Verify results
        assert result['status'] == 'sent'
        assert result['success'] is True
        assert result['topic'] == 'test_topic'
        assert result['key'] == 'test_key'
        assert result['job_id'] == 'test_run_123'
        
        # Verify producer was called
        mock_producer_instance.send.assert_called_once()


class TestKafkaConsumerOperator:
    """Test KafkaConsumerOperator functionality."""
    
    def test_operator_initialization(self):
        """Test operator initialization."""
        operator = KafkaConsumerOperator(
            task_id='test_kafka_consumer',
            topic='test_topic',
            consumer_type='text_extraction',
            max_messages=10,
            timeout_seconds=60
        )
        
        assert operator.task_id == 'test_kafka_consumer'
        assert operator.topic == 'test_topic'
        assert operator.consumer_type == 'text_extraction'
        assert operator.max_messages == 10
        assert operator.timeout_seconds == 60
    
    @patch('src.kafka.consumers.text_extraction_consumer.TextExtractionConsumer')
    def test_operator_execution(self, mock_consumer_class):
        """Test operator execution with mocked consumer."""
        # Setup mocks
        mock_consumer = Mock()
        mock_consumer_class.return_value = mock_consumer
        
        mock_consumer.get_metrics.return_value = {
            'messages_processed': 5,
            'messages_failed': 0,
            'processing_time': 10.5
        }
        mock_consumer.health_check.return_value = {'healthy': True}
        
        # Create operator
        operator = KafkaConsumerOperator(
            task_id='test_kafka_consumer',
            topic='test_topic',
            consumer_type='text_extraction',
            max_messages=10,
            timeout_seconds=60
        )
        
        # Create mock context
        mock_context_dict = {
            'dag_run': Mock(run_id='test_run_123'),
            'task_instance': Mock()
        }
        
        # Execute operator
        result = operator.execute(mock_context_dict)
        
        # Verify results
        assert result['status'] == 'completed'
        assert result['topic'] == 'test_topic'
        assert result['consumer_type'] == 'text_extraction'
        assert result['messages_processed'] == 5
        assert result['messages_failed'] == 0
        
        # Verify consumer was started and stopped
        mock_consumer.start.assert_called_once()
        mock_consumer.stop.assert_called_once()


class TestLLMProcessingOperator:
    """Test LLMProcessingOperator functionality."""
    
    def test_operator_initialization(self):
        """Test operator initialization."""
        test_segments = [
            {'text': 'Hello world', 'speaker': 'AMBIGUOUS'},
            {'text': 'Hi there', 'speaker': 'AMBIGUOUS'}
        ]
        
        operator = LLMProcessingOperator(
            task_id='test_llm_processor',
            text_segments=test_segments,
            llm_config={'engine': 'local', 'model': 'mistral'},
            operation_type='speaker_attribution'
        )
        
        assert operator.task_id == 'test_llm_processor'
        assert operator.text_segments == test_segments
        assert operator.llm_config == {'engine': 'local', 'model': 'mistral'}
        assert operator.operation_type == 'speaker_attribution'
    
    @patch('src.attribution.llm.orchestrator.LLMOrchestrator')
    def test_speaker_attribution_execution(self, mock_orchestrator_class):
        """Test speaker attribution execution."""
        # Setup mocks
        mock_orchestrator = Mock()
        mock_orchestrator_class.return_value = mock_orchestrator
        
        mock_orchestrator.process_segment.side_effect = [
            {'text': 'Hello world', 'speaker': 'John'},
            {'text': 'Hi there', 'speaker': 'Mary'}
        ]
        
        # Create operator
        test_segments = [
            {'text': 'Hello world', 'speaker': 'AMBIGUOUS'},
            {'text': 'Hi there', 'speaker': 'AMBIGUOUS'}
        ]
        
        operator = LLMProcessingOperator(
            task_id='test_llm_processor',
            text_segments=test_segments,
            llm_config={'engine': 'local', 'model': 'mistral'},
            operation_type='speaker_attribution'
        )
        
        # Create mock context
        mock_context_dict = {
            'dag_run': Mock(run_id='test_run_123'),
            'task_instance': Mock()
        }
        
        # Execute operator
        result = operator.execute(mock_context_dict)
        
        # Verify results
        assert result['status'] == 'completed'
        assert result['operation_type'] == 'speaker_attribution'
        assert result['total_segments'] == 2
        assert len(result['processed_segments']) == 2
        
        # Verify orchestrator was called
        assert mock_orchestrator.process_segment.call_count == 2
    
    @patch('src.llm_pool.llm_client.LLMClient')
    def test_text_classification_execution(self, mock_client_class):
        """Test text classification execution."""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_client.classify_text.side_effect = ['John', 'Mary']
        
        # Create operator
        test_segments = [
            {'text': 'Hello world', 'id': '1'},
            {'text': 'Hi there', 'id': '2'}
        ]
        
        operator = LLMProcessingOperator(
            task_id='test_llm_processor',
            text_segments=test_segments,
            llm_config={'engine': 'local', 'model': 'mistral'},
            operation_type='text_classification'
        )
        
        # Create mock context
        mock_context_dict = {
            'dag_run': Mock(run_id='test_run_123'),
            'task_instance': Mock()
        }
        
        # Execute operator
        result = operator.execute(mock_context_dict)
        
        # Verify results
        assert result['status'] == 'completed'
        assert result['operation_type'] == 'text_classification'
        assert result['total_segments'] == 2
        assert len(result['classifications']) == 2
        
        # Verify client was called
        assert mock_client.classify_text.call_count == 2


class TestQualityValidationOperator:
    """Test QualityValidationOperator functionality."""
    
    def test_operator_initialization(self):
        """Test operator initialization."""
        test_segments = [
            {'speaker': 'John', 'text': 'Hello'},
            {'speaker': 'Mary', 'text': 'Hi'}
        ]
        
        operator = QualityValidationOperator(
            task_id='test_validator',
            structured_segments=test_segments,
            quality_thresholds={'overall_quality': 90.0},
            validation_type='comprehensive'
        )
        
        assert operator.task_id == 'test_validator'
        assert operator.structured_segments == test_segments
        assert operator.quality_thresholds == {'overall_quality': 90.0}
        assert operator.validation_type == 'comprehensive'
    
    @patch('src.validation.validator.SimplifiedValidator')
    def test_comprehensive_validation_execution(self, mock_validator_class):
        """Test comprehensive validation execution."""
        # Setup mocks
        mock_validator = Mock()
        mock_validator_class.return_value = mock_validator
        
        mock_validator.validate_structured_segments.return_value = {
            'quality_score': 95.0,
            'errors': [],
            'warnings': [],
            'speaker_consistency': 92.0
        }
        
        # Create operator
        test_segments = [
            {'speaker': 'John', 'text': 'Hello'},
            {'speaker': 'Mary', 'text': 'Hi'}
        ]
        
        operator = QualityValidationOperator(
            task_id='test_validator',
            structured_segments=test_segments,
            quality_thresholds={'overall_quality': 90.0},
            validation_type='comprehensive'
        )
        
        # Create mock context
        mock_context_dict = {
            'dag_run': Mock(run_id='test_run_123'),
            'task_instance': Mock()
        }
        
        # Execute operator
        result = operator.execute(mock_context_dict)
        
        # Verify results
        assert result['status'] == 'completed'
        assert result['validation_type'] == 'comprehensive'
        assert result['overall_quality_score'] == 95.0
        assert result['total_segments'] == 2
        
        # Verify validator was called
        mock_validator.validate_structured_segments.assert_called_once()
    
    def test_speaker_consistency_validation(self):
        """Test speaker consistency validation."""
        # Create operator
        test_segments = [
            {'speaker': 'John', 'text': 'Hello'},
            {'speaker': 'Mary', 'text': 'Hi'},
            {'speaker': 'John', 'text': 'How are you?'},
            {'speaker': 'Mary', 'text': 'Good, thanks'}
        ]
        
        operator = QualityValidationOperator(
            task_id='test_validator',
            structured_segments=test_segments,
            validation_type='speaker_consistency'
        )
        
        # Create mock context
        mock_context_dict = {
            'dag_run': Mock(run_id='test_run_123'),
            'task_instance': Mock()
        }
        
        # Execute operator
        result = operator.execute(mock_context_dict)
        
        # Verify results
        assert result['status'] == 'completed'
        assert result['validation_type'] == 'speaker_consistency'
        assert result['unique_speakers'] == 2
        assert result['total_segments'] == 4
        assert 'speaker_stats' in result
        assert 'speaker_transitions' in result
        
        # Check speaker stats
        speaker_stats = result['speaker_stats']
        assert 'John' in speaker_stats
        assert 'Mary' in speaker_stats
        assert speaker_stats['John']['count'] == 2
        assert speaker_stats['Mary']['count'] == 2
    
    def test_attribution_quality_validation(self):
        """Test attribution quality validation."""
        # Create operator with mixed quality segments
        test_segments = [
            {'speaker': 'John', 'text': 'Hello', 'confidence': 0.95},
            {'speaker': 'Mary', 'text': 'Hi', 'confidence': 0.89},
            {'speaker': 'AMBIGUOUS', 'text': 'Unclear', 'confidence': 0.40},
            {'speaker': 'ERROR', 'text': 'Failed', 'confidence': 0.0}
        ]
        
        operator = QualityValidationOperator(
            task_id='test_validator',
            structured_segments=test_segments,
            validation_type='attribution_quality'
        )
        
        # Create mock context
        mock_context_dict = {
            'dag_run': Mock(run_id='test_run_123'),
            'task_instance': Mock()
        }
        
        # Execute operator
        result = operator.execute(mock_context_dict)
        
        # Verify results
        assert result['status'] == 'completed'
        assert result['validation_type'] == 'attribution_quality'
        assert result['total_segments'] == 4
        
        # Check attribution stats
        attribution_stats = result['attribution_stats']
        assert attribution_stats['attributed_segments'] == 2
        assert attribution_stats['ambiguous_segments'] == 1
        assert attribution_stats['error_segments'] == 1
        
        # Check quality metrics
        assert result['attribution_rate'] == 50.0  # 2/4 * 100
        assert result['error_rate'] == 25.0  # 1/4 * 100
        assert result['average_confidence'] > 0
    
    def test_quality_threshold_checking(self):
        """Test quality threshold checking."""
        # Create operator with thresholds
        test_segments = [
            {'speaker': 'John', 'text': 'Hello', 'confidence': 0.95},
            {'speaker': 'Mary', 'text': 'Hi', 'confidence': 0.89}
        ]
        
        operator = QualityValidationOperator(
            task_id='test_validator',
            structured_segments=test_segments,
            quality_thresholds={
                'overall_quality': 90.0,
                'attribution_confidence': 85.0,
                'error_rate': 10.0
            },
            validation_type='attribution_quality',
            fail_on_threshold=False  # Don't fail for testing
        )
        
        # Create mock context
        mock_context_dict = {
            'dag_run': Mock(run_id='test_run_123'),
            'task_instance': Mock()
        }
        
        # Execute operator
        result = operator.execute(mock_context_dict)
        
        # Verify threshold results
        assert 'threshold_results' in result
        threshold_results = result['threshold_results']
        assert 'all_passed' in threshold_results
        assert 'details' in threshold_results
        
        # Check specific thresholds
        for threshold_name, threshold_data in threshold_results['details'].items():
            assert 'actual' in threshold_data
            assert 'threshold' in threshold_data
            assert 'passed' in threshold_data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])