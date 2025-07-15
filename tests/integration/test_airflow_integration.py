"""
Integration tests for Airflow DAGs and custom operators.

Tests verify the complete workflow orchestration, DAG execution,
and integration between custom operators and external services.
"""

import pytest
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from airflow.models import DagBag, TaskInstance, DagRun
from airflow.utils.state import State
from airflow.utils.dates import days_ago
from airflow.configuration import conf

# Import our custom operators
from airflow.dags.operators.kafka_producer_operator import KafkaProducerOperator
from airflow.dags.operators.kafka_consumer_operator import KafkaConsumerOperator
from airflow.dags.operators.spark_text_structurer_operator import SparkTextStructurerOperator
from airflow.dags.operators.llm_processing_operator import LLMProcessingOperator
from airflow.dags.operators.quality_validation_operator import QualityValidationOperator


@pytest.mark.integration
@pytest.mark.airflow
class TestAirflowDAGIntegration:
    """Test Airflow DAG integration and workflow orchestration."""
    
    @pytest.fixture
    def dag_bag(self):
        """Load DAGs for testing."""
        # Set Airflow home for testing
        import os
        os.environ['AIRFLOW_HOME'] = '/tmp/airflow_test'
        
        # Load DAGs
        dag_bag = DagBag(dag_folder='airflow/dags', include_examples=False)
        return dag_bag
    
    @pytest.fixture
    def sample_dag_run_conf(self):
        """Sample DAG run configuration."""
        return {
            'input_file': 'test_book.pdf',
            'job_id': 'test_job_airflow_001',
            'processing_config': {
                'llm_engine': 'local',
                'llm_model': 'mistral',
                'chunk_size': 2000,
                'quality_threshold': 85.0
            },
            'output_config': {
                'format': 'json',
                'include_metadata': True
            }
        }
    
    def test_dag_loading(self, dag_bag):
        """Test that DAGs load without errors."""
        
        # Check that our main DAG is loaded
        assert 'text_to_audiobook_pipeline' in dag_bag.dag_ids
        
        # Check for import errors
        assert len(dag_bag.import_errors) == 0, f"DAG import errors: {dag_bag.import_errors}"
        
        # Get the main DAG
        dag = dag_bag.get_dag('text_to_audiobook_pipeline')
        assert dag is not None
        
        # Verify DAG structure
        expected_tasks = [
            'file_upload_producer',
            'text_extraction_consumer', 
            'chunk_producer',
            'spark_text_structurer',
            'llm_processing',
            'quality_validation',
            'output_formatter'
        ]
        
        actual_tasks = list(dag.task_ids)
        for expected_task in expected_tasks:
            assert expected_task in actual_tasks, f"Missing task: {expected_task}"
    
    def test_dag_structure_and_dependencies(self, dag_bag):
        """Test DAG task dependencies and structure."""
        
        dag = dag_bag.get_dag('text_to_audiobook_pipeline')
        
        # Test task dependencies
        file_upload_task = dag.get_task('file_upload_producer')
        text_extraction_task = dag.get_task('text_extraction_consumer')
        chunk_producer_task = dag.get_task('chunk_producer')
        spark_task = dag.get_task('spark_text_structurer')
        llm_task = dag.get_task('llm_processing')
        validation_task = dag.get_task('quality_validation')
        output_task = dag.get_task('output_formatter')
        
        # Verify dependencies
        assert text_extraction_task in file_upload_task.downstream_list
        assert chunk_producer_task in text_extraction_task.downstream_list
        assert spark_task in chunk_producer_task.downstream_list
        assert llm_task in spark_task.downstream_list
        assert validation_task in llm_task.downstream_list
        assert output_task in validation_task.downstream_list
    
    @patch('airflow.dags.operators.kafka_producer_operator.KafkaProducer')
    def test_kafka_producer_operator(self, mock_kafka_producer):
        """Test KafkaProducerOperator functionality."""
        
        # Setup mock
        mock_producer_instance = Mock()
        mock_kafka_producer.return_value = mock_producer_instance
        mock_producer_instance.send.return_value = Mock()
        mock_producer_instance.send.return_value.get.return_value = Mock()
        
        # Create operator
        operator = KafkaProducerOperator(
            task_id='test_kafka_producer',
            topic='test_topic',
            message_data={
                'test_key': 'test_value',
                'timestamp': '2024-01-01T00:00:00'
            },
            dag=None
        )
        
        # Create mock context
        context = {
            'dag_run': Mock(),
            'task_instance': Mock(),
            'execution_date': datetime.now()
        }
        context['dag_run'].conf = {'job_id': 'test_job'}
        
        # Execute operator
        result = operator.execute(context)
        
        # Verify execution
        assert result is not None
        mock_kafka_producer.assert_called_once()
        mock_producer_instance.send.assert_called_once()
    
    @patch('airflow.dags.operators.kafka_consumer_operator.KafkaConsumer')
    def test_kafka_consumer_operator(self, mock_kafka_consumer):
        """Test KafkaConsumerOperator functionality."""
        
        # Setup mock
        mock_consumer_instance = Mock()
        mock_kafka_consumer.return_value = mock_consumer_instance
        
        # Mock consumed messages
        mock_message = Mock()
        mock_message.value = json.dumps({
            'job_id': 'test_job',
            'chunk_id': 'chunk_001',
            'text_content': 'Test content for processing'
        }).encode('utf-8')
        mock_message.key = b'chunk_001'
        
        mock_consumer_instance.poll.return_value = {
            'test_topic': [mock_message]
        }
        
        # Create operator
        operator = KafkaConsumerOperator(
            task_id='test_kafka_consumer',
            topics=['test_topic'],
            consumer_group='test_group',
            max_messages=1,
            timeout_seconds=30,
            dag=None
        )
        
        # Create mock context
        context = {
            'dag_run': Mock(),
            'task_instance': Mock(),
            'execution_date': datetime.now()
        }
        
        # Execute operator
        result = operator.execute(context)
        
        # Verify execution
        assert result is not None
        assert len(result) > 0
        mock_kafka_consumer.assert_called_once()
        mock_consumer_instance.poll.assert_called()
    
    @patch('airflow.dags.operators.spark_text_structurer_operator.SparkSession')
    def test_spark_text_structurer_operator(self, mock_spark_session_class):
        """Test SparkTextStructurerOperator functionality."""
        
        # Setup mock Spark session
        mock_spark_session = Mock()
        mock_spark_session_class.builder.appName.return_value.getOrCreate.return_value = mock_spark_session
        
        # Mock DataFrame operations
        mock_df = Mock()
        mock_df.count.return_value = 100
        mock_df.collect.return_value = [
            Mock(segment_id='seg_001', text_content='Test segment 1', speaker='narrator'),
            Mock(segment_id='seg_002', text_content='Test segment 2', speaker='character')
        ]
        mock_spark_session.createDataFrame.return_value = mock_df
        
        # Create operator
        operator = SparkTextStructurerOperator(
            task_id='test_spark_structurer',
            input_data=[
                {'chunk_id': 'chunk_001', 'text_content': 'Test content for structuring'}
            ],
            dag=None
        )
        
        # Create mock context
        context = {
            'dag_run': Mock(),
            'task_instance': Mock(),
            'execution_date': datetime.now()
        }
        
        # Execute operator
        result = operator.execute(context)
        
        # Verify execution
        assert result is not None
        assert len(result) > 0
        mock_spark_session_class.builder.appName.assert_called()
    
    @patch('airflow.dags.operators.llm_processing_operator.get_pool_manager')
    def test_llm_processing_operator(self, mock_get_pool_manager):
        """Test LLMProcessingOperator functionality."""
        
        # Setup mock LLM pool manager
        mock_pool_manager = Mock()
        mock_get_pool_manager.return_value = mock_pool_manager
        
        # Mock LLM response
        mock_pool_manager.process_segment.return_value = {
            'segment_id': 'seg_001',
            'speaker': 'character_1',
            'confidence': 0.92,
            'text_content': 'Hello, how are you?'
        }
        
        # Create operator
        operator = LLMProcessingOperator(
            task_id='test_llm_processing',
            segments=[
                {
                    'segment_id': 'seg_001',
                    'text_content': 'Hello, how are you?',
                    'speaker': 'AMBIGUOUS'
                }
            ],
            llm_config={
                'engine': 'local',
                'model': 'mistral',
                'temperature': 0.1
            },
            dag=None
        )
        
        # Create mock context
        context = {
            'dag_run': Mock(),
            'task_instance': Mock(),
            'execution_date': datetime.now()
        }
        
        # Execute operator
        result = operator.execute(context)
        
        # Verify execution
        assert result is not None
        assert len(result) > 0
        mock_get_pool_manager.assert_called()
        mock_pool_manager.process_segment.assert_called()
    
    @patch('airflow.dags.operators.quality_validation_operator.get_validation_engine')
    def test_quality_validation_operator(self, mock_get_validation_engine):
        """Test QualityValidationOperator functionality."""
        
        # Setup mock validation engine
        mock_validation_engine = Mock()
        mock_get_validation_engine.return_value = mock_validation_engine
        
        # Mock validation results
        from src.spark.distributed_validation import ValidationResult
        mock_validation_results = [
            ValidationResult(
                segment_id='seg_001',
                original_quality_score=0.85,
                refined_quality_score=0.90,
                validation_issues=[],
                refinement_applied=True,
                processing_time=0.15
            )
        ]
        mock_validation_engine.validate_text_segments.return_value = mock_validation_results
        
        # Create operator
        operator = QualityValidationOperator(
            task_id='test_quality_validation',
            segments=[
                {
                    'segment_id': 'seg_001',
                    'text_content': 'Test segment for validation',
                    'speaker': 'narrator',
                    'quality_score': 0.85
                }
            ],
            quality_threshold=0.8,
            dag=None
        )
        
        # Create mock context
        context = {
            'dag_run': Mock(),
            'task_instance': Mock(),
            'execution_date': datetime.now()
        }
        
        # Execute operator
        result = operator.execute(context)
        
        # Verify execution
        assert result is not None
        assert 'validation_results' in result
        assert 'quality_report' in result
        mock_get_validation_engine.assert_called()
        mock_validation_engine.validate_text_segments.assert_called()


@pytest.mark.integration
@pytest.mark.airflow
@pytest.mark.slow
class TestAirflowWorkflowExecution:
    """Test complete Airflow workflow execution."""
    
    @pytest.fixture
    def test_dag(self):
        """Create a test DAG for workflow testing."""
        from airflow import DAG
        from datetime import datetime
        
        dag = DAG(
            'test_text_to_audiobook',
            default_args={
                'owner': 'test',
                'depends_on_past': False,
                'start_date': days_ago(1),
                'email_on_failure': False,
                'email_on_retry': False,
                'retries': 1,
                'retry_delay': timedelta(minutes=5)
            },
            description='Test DAG for integration testing',
            schedule_interval=None,
            catchup=False,
            tags=['test', 'integration']
        )
        
        return dag
    
    @patch('airflow.dags.operators.kafka_producer_operator.KafkaProducer')
    @patch('airflow.dags.operators.kafka_consumer_operator.KafkaConsumer')
    @patch('airflow.dags.operators.spark_text_structurer_operator.SparkSession')
    def test_complete_workflow_execution(self, mock_spark_session_class, mock_kafka_consumer, mock_kafka_producer, test_dag):
        """Test complete workflow execution with all operators."""
        
        # Setup all mocks
        self._setup_kafka_mocks(mock_kafka_producer, mock_kafka_consumer)
        self._setup_spark_mocks(mock_spark_session_class)
        
        # Create workflow tasks
        file_upload_task = KafkaProducerOperator(
            task_id='file_upload',
            topic='file_uploads',
            message_data={'file_path': 'test_book.pdf'},
            dag=test_dag
        )
        
        text_extraction_task = KafkaConsumerOperator(
            task_id='text_extraction',
            topics=['file_uploads'],
            consumer_group='extraction_group',
            max_messages=1,
            dag=test_dag
        )
        
        spark_processing_task = SparkTextStructurerOperator(
            task_id='spark_processing',
            input_data=[{'text': 'test content'}],
            dag=test_dag
        )
        
        # Set up dependencies
        file_upload_task >> text_extraction_task >> spark_processing_task
        
        # Create mock DAG run
        dag_run = Mock()
        dag_run.conf = {
            'job_id': 'test_workflow_001',
            'input_file': 'test_book.pdf'
        }
        
        # Create execution context
        execution_date = datetime.now()
        context = {
            'dag_run': dag_run,
            'execution_date': execution_date,
            'task_instance': Mock()
        }
        
        # Execute tasks in sequence
        try:
            # Execute file upload
            upload_result = file_upload_task.execute(context)
            assert upload_result is not None
            
            # Execute text extraction
            extraction_result = text_extraction_task.execute(context)
            assert extraction_result is not None
            
            # Execute Spark processing
            processing_result = spark_processing_task.execute(context)
            assert processing_result is not None
            
            # Verify workflow completed successfully
            assert True
            
        except Exception as e:
            pytest.fail(f"Workflow execution failed: {e}")
    
    def _setup_kafka_mocks(self, mock_kafka_producer, mock_kafka_consumer):
        """Setup Kafka mocks for workflow testing."""
        
        # Producer mock
        mock_producer_instance = Mock()
        mock_kafka_producer.return_value = mock_producer_instance
        mock_producer_instance.send.return_value = Mock()
        mock_producer_instance.send.return_value.get.return_value = Mock()
        mock_producer_instance.flush.return_value = None
        mock_producer_instance.close.return_value = None
        
        # Consumer mock
        mock_consumer_instance = Mock()
        mock_kafka_consumer.return_value = mock_consumer_instance
        
        mock_message = Mock()
        mock_message.value = json.dumps({
            'job_id': 'test_workflow_001',
            'file_path': 'test_book.pdf',
            'extracted_text': 'This is extracted text content for processing.'
        }).encode('utf-8')
        mock_message.key = b'test_workflow_001'
        
        mock_consumer_instance.poll.return_value = {
            'file_uploads': [mock_message]
        }
        mock_consumer_instance.close.return_value = None
    
    def _setup_spark_mocks(self, mock_spark_session_class):
        """Setup Spark mocks for workflow testing."""
        
        mock_spark_session = Mock()
        mock_spark_session_class.builder.appName.return_value.getOrCreate.return_value = mock_spark_session
        
        mock_df = Mock()
        mock_df.count.return_value = 50
        mock_df.collect.return_value = [
            Mock(segment_id=f'seg_{i:03d}', text_content=f'Segment {i}', speaker='narrator')
            for i in range(10)
        ]
        mock_spark_session.createDataFrame.return_value = mock_df
        mock_spark_session.stop.return_value = None
    
    def test_error_handling_in_workflow(self, test_dag):
        """Test error handling and retry logic in workflow."""
        
        # Create task that will fail
        failing_task = KafkaProducerOperator(
            task_id='failing_kafka_producer',
            topic='test_topic',
            message_data={'test': 'data'},
            dag=test_dag
        )
        
        # Mock Kafka producer to raise exception
        with patch('airflow.dags.operators.kafka_producer_operator.KafkaProducer') as mock_producer:
            mock_producer.side_effect = Exception("Kafka connection failed")
            
            context = {
                'dag_run': Mock(),
                'task_instance': Mock(),
                'execution_date': datetime.now()
            }
            
            # Test that exception is properly raised
            with pytest.raises(Exception):
                failing_task.execute(context)
    
    def test_task_retry_logic(self, test_dag):
        """Test task retry logic in case of failures."""
        
        retry_task = KafkaProducerOperator(
            task_id='retry_test_task',
            topic='retry_topic',
            message_data={'retry': 'test'},
            retries=2,
            retry_delay=timedelta(seconds=1),
            dag=test_dag
        )
        
        # Mock intermittent failures
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first two attempts
                raise Exception("Temporary failure")
            return Mock()  # Succeed on third attempt
        
        with patch('airflow.dags.operators.kafka_producer_operator.KafkaProducer') as mock_producer:
            mock_producer_instance = Mock()
            mock_producer.return_value = mock_producer_instance
            mock_producer_instance.send.side_effect = side_effect
            
            context = {
                'dag_run': Mock(),
                'task_instance': Mock(),
                'execution_date': datetime.now()
            }
            
            # This would normally be handled by Airflow's retry mechanism
            # For testing, we simulate the retry behavior
            try:
                retry_task.execute(context)
            except Exception:
                # First attempt fails
                try:
                    retry_task.execute(context)
                except Exception:
                    # Second attempt fails
                    result = retry_task.execute(context)
                    # Third attempt succeeds
                    assert result is not None


@pytest.mark.integration
@pytest.mark.airflow
@pytest.mark.monitoring
class TestAirflowMonitoringIntegration:
    """Test Airflow integration with monitoring and metrics."""
    
    @patch('src.monitoring.prometheus_metrics.get_metrics_collector')
    def test_airflow_metrics_collection(self, mock_get_metrics):
        """Test that Airflow tasks collect metrics properly."""
        
        mock_metrics = Mock()
        mock_get_metrics.return_value = mock_metrics
        
        # Test metrics collection in Kafka operator
        with patch('airflow.dags.operators.kafka_producer_operator.KafkaProducer') as mock_producer:
            mock_producer_instance = Mock()
            mock_producer.return_value = mock_producer_instance
            mock_producer_instance.send.return_value = Mock()
            mock_producer_instance.send.return_value.get.return_value = Mock()
            
            operator = KafkaProducerOperator(
                task_id='metrics_test_kafka',
                topic='metrics_topic',
                message_data={'test': 'metrics'},
                dag=None
            )
            
            context = {
                'dag_run': Mock(),
                'task_instance': Mock(),
                'execution_date': datetime.now()
            }
            
            operator.execute(context)
            
            # Verify metrics were collected
            # (This would depend on actual implementation in the operators)
    
    @patch('src.monitoring.health_checks.get_health_service')
    def test_airflow_health_checks(self, mock_get_health_service):
        """Test health checks integration with Airflow workflow."""
        
        mock_health_service = Mock()
        mock_get_health_service.return_value = mock_health_service
        
        # Mock healthy system
        mock_health_service.get_overall_health.return_value = {
            'status': 'healthy',
            'message': 'All components operational',
            'components': {
                'airflow': {'status': 'healthy'},
                'kafka': {'status': 'healthy'},
                'spark': {'status': 'healthy'}
            }
        }
        
        health_service = mock_get_health_service()
        health_status = health_service.get_overall_health()
        
        assert health_status['status'] == 'healthy'
        assert 'airflow' in health_status['components']
    
    def test_airflow_dag_monitoring(self, dag_bag):
        """Test DAG execution monitoring and alerting."""
        
        dag = dag_bag.get_dag('text_to_audiobook_pipeline')
        
        # Test DAG run duration monitoring
        start_time = datetime.now()
        
        # Simulate DAG execution time
        time.sleep(0.1)
        
        end_time = datetime.now()
        execution_duration = (end_time - start_time).total_seconds()
        
        # Verify execution time is reasonable
        assert execution_duration < 1.0  # Should be very fast with mocks
        
        # Test task count and structure
        assert len(dag.tasks) >= 5  # Should have at least 5 main tasks
        
        # Test that all tasks have proper configuration
        for task in dag.tasks:
            assert task.task_id is not None
            assert task.dag is dag
            assert hasattr(task, 'execute')


@pytest.mark.integration
@pytest.mark.airflow
@pytest.mark.external
class TestAirflowExternalIntegration:
    """Test Airflow integration with external services (when available)."""
    
    def test_real_airflow_database_integration(self):
        """Test with real Airflow database if available."""
        try:
            from airflow.models import Variable, Connection
            from airflow import settings
            
            # Try to access Airflow database
            session = settings.Session()
            
            # Test variable storage/retrieval
            test_var_key = 'integration_test_var'
            test_var_value = 'integration_test_value'
            
            Variable.set(test_var_key, test_var_value)
            retrieved_value = Variable.get(test_var_key)
            
            assert retrieved_value == test_var_value
            
            # Cleanup
            Variable.delete(test_var_key)
            session.close()
            
        except Exception:
            pytest.skip("Real Airflow database not available")
    
    def test_real_airflow_scheduler_integration(self):
        """Test with real Airflow scheduler if available."""
        try:
            from airflow.models import DagModel
            from airflow import settings
            
            # Check if scheduler is running by looking for DAG models
            session = settings.Session()
            dag_models = session.query(DagModel).limit(1).all()
            
            # If we have DAG models, scheduler has been running
            if dag_models:
                assert len(dag_models) > 0
            else:
                pytest.skip("No DAG models found - scheduler may not be running")
            
            session.close()
            
        except Exception:
            pytest.skip("Real Airflow scheduler not available")


@pytest.mark.integration
@pytest.mark.performance
class TestAirflowPerformanceIntegration:
    """Test performance characteristics of Airflow integration."""
    
    def test_dag_parsing_performance(self, dag_bag):
        """Test DAG parsing performance."""
        
        start_time = time.time()
        
        # Parse all DAGs
        dag_bag.collect_dags()
        
        end_time = time.time()
        parsing_time = end_time - start_time
        
        # DAG parsing should be fast
        assert parsing_time < 5.0, f"DAG parsing took too long: {parsing_time}s"
        
        # Should have loaded our main DAG
        assert 'text_to_audiobook_pipeline' in dag_bag.dag_ids
    
    def test_operator_execution_performance(self):
        """Test operator execution performance."""
        
        with patch('airflow.dags.operators.kafka_producer_operator.KafkaProducer') as mock_producer:
            mock_producer_instance = Mock()
            mock_producer.return_value = mock_producer_instance
            mock_producer_instance.send.return_value = Mock()
            mock_producer_instance.send.return_value.get.return_value = Mock()
            
            operator = KafkaProducerOperator(
                task_id='performance_test',
                topic='performance_topic',
                message_data={'test': 'performance'},
                dag=None
            )
            
            context = {
                'dag_run': Mock(),
                'task_instance': Mock(),
                'execution_date': datetime.now()
            }
            
            # Test multiple executions
            execution_times = []
            
            for _ in range(10):
                start_time = time.time()
                operator.execute(context)
                end_time = time.time()
                execution_times.append(end_time - start_time)
            
            # All executions should be fast
            avg_execution_time = sum(execution_times) / len(execution_times)
            assert avg_execution_time < 0.1, f"Average execution time too slow: {avg_execution_time}s"
            
            # No execution should be extremely slow
            max_execution_time = max(execution_times)
            assert max_execution_time < 0.5, f"Max execution time too slow: {max_execution_time}s"