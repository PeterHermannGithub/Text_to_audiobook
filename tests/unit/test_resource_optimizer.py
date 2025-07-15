"""
Unit tests for the Spark resource optimization system.

Tests cover resource allocation strategies, workload analysis, performance monitoring,
and dynamic optimization algorithms.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.spark.resource_optimizer import (
    SparkResourceOptimizer,
    ResourceAllocation,
    WorkloadCharacteristics,
    SystemResources,
    PerformanceMetrics,
    get_resource_optimizer,
    optimize_for_workload
)


class TestResourceAllocation:
    """Test ResourceAllocation dataclass."""
    
    def test_resource_allocation_creation(self):
        """Test ResourceAllocation creation and basic properties."""
        allocation = ResourceAllocation(
            executor_instances=4,
            executor_cores=2,
            executor_memory="4g",
            driver_memory="2g",
            max_result_size="1g",
            broadcast_threshold="200MB",
            sql_shuffle_partitions=16
        )
        
        assert allocation.executor_instances == 4
        assert allocation.executor_cores == 2
        assert allocation.executor_memory == "4g"
        assert allocation.driver_memory == "2g"
        assert allocation.max_result_size == "1g"
        assert allocation.broadcast_threshold == "200MB"
        assert allocation.sql_shuffle_partitions == 16
    
    def test_to_spark_config(self):
        """Test conversion to Spark configuration dictionary."""
        allocation = ResourceAllocation(
            executor_instances=3,
            executor_cores=4,
            executor_memory="8g",
            driver_memory="4g",
            max_result_size="2g",
            broadcast_threshold="512MB",
            sql_shuffle_partitions=24
        )
        
        config = allocation.to_spark_config()
        
        assert isinstance(config, dict)
        assert config['spark.executor.instances'] == '3'
        assert config['spark.executor.cores'] == '4'
        assert config['spark.executor.memory'] == '8g'
        assert config['spark.driver.memory'] == '4g'
        assert config['spark.driver.maxResultSize'] == '2g'
        assert config['spark.sql.autoBroadcastJoinThreshold'] == '512MB'
        assert config['spark.sql.shuffle.partitions'] == '24'


class TestWorkloadCharacteristics:
    """Test WorkloadCharacteristics dataclass."""
    
    def test_workload_characteristics_creation(self):
        """Test WorkloadCharacteristics creation."""
        workload = WorkloadCharacteristics(
            data_size_mb=100.0,
            complexity_score=7.5,
            parallelism_potential=0.8,
            memory_intensive=True,
            io_intensive=False,
            cpu_intensive=True,
            estimated_duration_minutes=20.0
        )
        
        assert workload.data_size_mb == 100.0
        assert workload.complexity_score == 7.5
        assert workload.parallelism_potential == 0.8
        assert workload.memory_intensive is True
        assert workload.io_intensive is False
        assert workload.cpu_intensive is True
        assert workload.estimated_duration_minutes == 20.0
    
    def test_workload_characteristics_validation(self):
        """Test that characteristics are properly validated."""
        # Test complexity score bounds
        workload1 = WorkloadCharacteristics(
            data_size_mb=50.0,
            complexity_score=15.0,  # Above max
            parallelism_potential=0.5,
            memory_intensive=False,
            io_intensive=False,
            cpu_intensive=False,
            estimated_duration_minutes=10.0
        )
        assert workload1.complexity_score == 10  # Clamped to max
        
        workload2 = WorkloadCharacteristics(
            data_size_mb=50.0,
            complexity_score=-5.0,  # Below min
            parallelism_potential=0.5,
            memory_intensive=False,
            io_intensive=False,
            cpu_intensive=False,
            estimated_duration_minutes=10.0
        )
        assert workload2.complexity_score == 1  # Clamped to min
        
        # Test parallelism potential bounds
        workload3 = WorkloadCharacteristics(
            data_size_mb=50.0,
            complexity_score=5.0,
            parallelism_potential=1.5,  # Above max
            memory_intensive=False,
            io_intensive=False,
            cpu_intensive=False,
            estimated_duration_minutes=10.0
        )
        assert workload3.parallelism_potential == 1.0  # Clamped to max


class TestSystemResources:
    """Test SystemResources dataclass."""
    
    def test_system_resources_creation(self):
        """Test SystemResources creation."""
        timestamp = datetime.now()
        resources = SystemResources(
            total_cpu_cores=8,
            available_cpu_cores=6,
            total_memory_gb=32.0,
            available_memory_gb=24.0,
            cpu_usage_percent=25.0,
            memory_usage_percent=75.0,
            disk_io_usage_percent=10.0,
            network_io_mbps=5.0,
            timestamp=timestamp
        )
        
        assert resources.total_cpu_cores == 8
        assert resources.available_cpu_cores == 6
        assert resources.total_memory_gb == 32.0
        assert resources.available_memory_gb == 24.0
        assert resources.cpu_usage_percent == 25.0
        assert resources.memory_usage_percent == 75.0
        assert resources.disk_io_usage_percent == 10.0
        assert resources.network_io_mbps == 5.0
        assert resources.timestamp == timestamp


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""
    
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics creation."""
        metrics = PerformanceMetrics(
            job_duration_seconds=120.0,
            cpu_utilization=0.75,
            memory_utilization=0.65,
            gc_time_percent=8.0,
            shuffle_read_mb=500.0,
            shuffle_write_mb=400.0,
            input_size_mb=1000.0,
            output_size_mb=800.0,
            task_count=100,
            failed_tasks=2
        )
        
        assert metrics.job_duration_seconds == 120.0
        assert metrics.cpu_utilization == 0.75
        assert metrics.memory_utilization == 0.65
        assert metrics.gc_time_percent == 8.0
        assert metrics.shuffle_read_mb == 500.0
        assert metrics.shuffle_write_mb == 400.0
        assert metrics.input_size_mb == 1000.0
        assert metrics.output_size_mb == 800.0
        assert metrics.task_count == 100
        assert metrics.failed_tasks == 2
    
    def test_efficiency_score_calculation(self):
        """Test efficiency score calculation."""
        # Test high efficiency case
        high_efficiency_metrics = PerformanceMetrics(
            job_duration_seconds=60.0,
            cpu_utilization=0.8,    # Near optimal
            memory_utilization=0.7,  # Near optimal
            gc_time_percent=5.0,     # Low GC time
            shuffle_read_mb=100.0,
            shuffle_write_mb=80.0,
            input_size_mb=200.0,
            output_size_mb=160.0,
            task_count=50,
            failed_tasks=0           # No failures
        )
        
        efficiency = high_efficiency_metrics.efficiency_score()
        assert 0.8 <= efficiency <= 1.0  # Should be high efficiency
        
        # Test low efficiency case
        low_efficiency_metrics = PerformanceMetrics(
            job_duration_seconds=300.0,
            cpu_utilization=0.3,     # Low CPU usage
            memory_utilization=0.9,  # High memory usage
            gc_time_percent=25.0,    # High GC time
            shuffle_read_mb=1000.0,
            shuffle_write_mb=800.0,
            input_size_mb=500.0,
            output_size_mb=400.0,
            task_count=100,
            failed_tasks=20          # Many failures
        )
        
        efficiency = low_efficiency_metrics.efficiency_score()
        assert 0.0 <= efficiency <= 0.4  # Should be low efficiency
    
    def test_efficiency_score_edge_cases(self):
        """Test efficiency score with edge cases."""
        # Test perfect metrics
        perfect_metrics = PerformanceMetrics(
            job_duration_seconds=60.0,
            cpu_utilization=0.8,
            memory_utilization=0.7,
            gc_time_percent=0.0,
            shuffle_read_mb=100.0,
            shuffle_write_mb=80.0,
            input_size_mb=200.0,
            output_size_mb=160.0,
            task_count=50,
            failed_tasks=0
        )
        
        efficiency = perfect_metrics.efficiency_score()
        assert efficiency > 0.9
        
        # Test worst case metrics
        worst_metrics = PerformanceMetrics(
            job_duration_seconds=600.0,
            cpu_utilization=0.1,
            memory_utilization=0.1,
            gc_time_percent=50.0,
            shuffle_read_mb=5000.0,
            shuffle_write_mb=4000.0,
            input_size_mb=1000.0,
            output_size_mb=800.0,
            task_count=10,
            failed_tasks=10  # All tasks failed
        )
        
        efficiency = worst_metrics.efficiency_score()
        assert 0.0 <= efficiency <= 0.2


class TestSparkResourceOptimizer:
    """Test SparkResourceOptimizer main functionality."""
    
    @pytest.fixture
    def mock_spark_session(self):
        """Create mock Spark session for testing."""
        mock_session = Mock()
        mock_session.sparkContext = Mock()
        mock_session.sparkContext.applicationId = "test_app_001"
        mock_session.conf.set.return_value = None
        return mock_session
    
    @pytest.fixture
    def resource_optimizer(self, mock_spark_session):
        """Create resource optimizer with mocked dependencies."""
        with patch('src.spark.resource_optimizer.SparkSession') as mock_spark_class:
            mock_spark_class.getActiveSession.return_value = mock_spark_session
            
            with patch('src.spark.resource_optimizer.get_metrics_collector') as mock_metrics:
                mock_metrics.return_value = Mock()
                optimizer = SparkResourceOptimizer(mock_spark_session)
                return optimizer
    
    def test_optimizer_initialization(self, resource_optimizer):
        """Test optimizer initialization."""
        assert resource_optimizer.spark is not None
        assert resource_optimizer.metrics_collector is not None
        assert isinstance(resource_optimizer.performance_history, type(resource_optimizer.performance_history))
        assert isinstance(resource_optimizer.allocation_history, type(resource_optimizer.allocation_history))
        assert isinstance(resource_optimizer.optimization_strategies, dict)
        assert len(resource_optimizer.optimization_strategies) > 0
        assert isinstance(resource_optimizer.config, dict)
    
    def test_optimization_strategies_exist(self, resource_optimizer):
        """Test that all expected optimization strategies exist."""
        expected_strategies = [
            'conservative', 'balanced', 'aggressive', 
            'memory_optimized', 'cpu_optimized'
        ]
        
        for strategy in expected_strategies:
            assert strategy in resource_optimizer.optimization_strategies
            assert callable(resource_optimizer.optimization_strategies[strategy])
    
    def test_config_validation(self, resource_optimizer):
        """Test that configuration values are reasonable."""
        config = resource_optimizer.config
        
        assert config['min_executor_instances'] >= 1
        assert config['max_executor_instances'] > config['min_executor_instances']
        assert config['min_executor_cores'] >= 1
        assert config['max_executor_cores'] > config['min_executor_cores']
        assert config['min_executor_memory_gb'] >= 1
        assert config['max_executor_memory_gb'] > config['min_executor_memory_gb']
        assert 0 < config['optimization_threshold'] < 1
        assert 0 < config['adaptation_rate'] < 1
    
    def test_analyze_workload_general(self, resource_optimizer):
        """Test workload analysis for general processing."""
        workload = resource_optimizer.analyze_workload(
            data_size_mb=100.0,
            processing_type='general',
            estimated_complexity=5.0
        )
        
        assert isinstance(workload, WorkloadCharacteristics)
        assert workload.data_size_mb == 100.0
        assert workload.complexity_score == 5.0
        assert 0 <= workload.parallelism_potential <= 1
        assert workload.estimated_duration_minutes > 0
    
    def test_analyze_workload_specific_types(self, resource_optimizer):
        """Test workload analysis for specific processing types."""
        test_cases = [
            'text_extraction',
            'llm_processing', 
            'audio_generation',
            'validation'
        ]
        
        for processing_type in test_cases:
            workload = resource_optimizer.analyze_workload(
                data_size_mb=50.0,
                processing_type=processing_type,
                estimated_complexity=6.0
            )
            
            assert isinstance(workload, WorkloadCharacteristics)
            assert workload.data_size_mb == 50.0
            assert workload.complexity_score == 6.0
            
            # Verify type-specific characteristics are set
            assert isinstance(workload.memory_intensive, bool)
            assert isinstance(workload.io_intensive, bool)
            assert isinstance(workload.cpu_intensive, bool)
    
    @patch('src.spark.resource_optimizer.psutil')
    def test_get_system_resources(self, mock_psutil, resource_optimizer):
        """Test system resource collection."""
        # Mock psutil responses
        mock_psutil.cpu_count.return_value = 8
        mock_psutil.cpu_percent.return_value = 25.0
        
        mock_memory = Mock()
        mock_memory.total = 32 * 1024**3  # 32GB
        mock_memory.available = 24 * 1024**3  # 24GB
        mock_memory.percent = 75.0
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk_io = Mock()
        mock_disk_io.read_bytes = 1024**2  # 1MB
        mock_disk_io.write_bytes = 512**2  # 0.25MB
        mock_psutil.disk_io_counters.return_value = mock_disk_io
        
        mock_network_io = Mock()
        mock_network_io.bytes_sent = 2 * 1024**2  # 2MB
        mock_network_io.bytes_recv = 3 * 1024**2  # 3MB
        mock_psutil.net_io_counters.return_value = mock_network_io
        
        resources = resource_optimizer.get_system_resources()
        
        assert isinstance(resources, SystemResources)
        assert resources.total_cpu_cores == 8
        assert resources.available_cpu_cores > 0
        assert resources.total_memory_gb == 32.0
        assert resources.available_memory_gb == 24.0
        assert resources.cpu_usage_percent == 25.0
        assert resources.memory_usage_percent == 75.0
        assert resources.disk_io_usage_percent >= 0
        assert resources.network_io_mbps >= 0
    
    def test_conservative_allocation_strategy(self, resource_optimizer):
        """Test conservative allocation strategy."""
        workload = WorkloadCharacteristics(
            data_size_mb=100.0,
            complexity_score=5.0,
            parallelism_potential=0.7,
            memory_intensive=False,
            io_intensive=False,
            cpu_intensive=False,
            estimated_duration_minutes=10.0
        )
        
        system_resources = SystemResources(
            total_cpu_cores=8,
            available_cpu_cores=6,
            total_memory_gb=16.0,
            available_memory_gb=12.0,
            cpu_usage_percent=25.0,
            memory_usage_percent=25.0,
            disk_io_usage_percent=5.0,
            network_io_mbps=2.0
        )
        
        allocation = resource_optimizer._conservative_allocation(workload, system_resources)
        
        assert isinstance(allocation, ResourceAllocation)
        assert 1 <= allocation.executor_instances <= 4  # Conservative limits
        assert allocation.executor_cores <= 2  # Conservative cores
        assert allocation.executor_memory.endswith('g')
        assert allocation.driver_memory.endswith('g')
        assert allocation.sql_shuffle_partitions > 0
    
    def test_balanced_allocation_strategy(self, resource_optimizer):
        """Test balanced allocation strategy."""
        workload = WorkloadCharacteristics(
            data_size_mb=200.0,
            complexity_score=6.0,
            parallelism_potential=0.8,
            memory_intensive=False,
            io_intensive=True,
            cpu_intensive=False,
            estimated_duration_minutes=15.0
        )
        
        system_resources = SystemResources(
            total_cpu_cores=16,
            available_cpu_cores=12,
            total_memory_gb=32.0,
            available_memory_gb=24.0,
            cpu_usage_percent=25.0,
            memory_usage_percent=25.0,
            disk_io_usage_percent=10.0,
            network_io_mbps=5.0
        )
        
        allocation = resource_optimizer._balanced_allocation(workload, system_resources)
        
        assert isinstance(allocation, ResourceAllocation)
        assert allocation.executor_instances >= 1
        assert allocation.executor_instances <= resource_optimizer.config['max_executor_instances']
        assert allocation.executor_cores >= 1
        assert allocation.executor_cores <= resource_optimizer.config['max_executor_cores']
        assert allocation.sql_shuffle_partitions >= allocation.executor_instances * allocation.executor_cores * 2
    
    def test_aggressive_allocation_strategy(self, resource_optimizer):
        """Test aggressive allocation strategy."""
        workload = WorkloadCharacteristics(
            data_size_mb=500.0,
            complexity_score=8.0,
            parallelism_potential=0.9,
            memory_intensive=True,
            io_intensive=False,
            cpu_intensive=True,
            estimated_duration_minutes=30.0
        )
        
        system_resources = SystemResources(
            total_cpu_cores=32,
            available_cpu_cores=28,
            total_memory_gb=64.0,
            available_memory_gb=50.0,
            cpu_usage_percent=12.5,
            memory_usage_percent=22.0,
            disk_io_usage_percent=8.0,
            network_io_mbps=10.0
        )
        
        allocation = resource_optimizer._aggressive_allocation(workload, system_resources)
        
        assert isinstance(allocation, ResourceAllocation)
        # Aggressive should use more resources
        assert allocation.executor_instances >= 2
        assert allocation.executor_cores >= 2
        # Should have higher shuffle partitions for max parallelism
        assert allocation.sql_shuffle_partitions >= allocation.executor_instances * allocation.executor_cores * 4
    
    def test_memory_optimized_allocation_strategy(self, resource_optimizer):
        """Test memory-optimized allocation strategy."""
        workload = WorkloadCharacteristics(
            data_size_mb=300.0,
            complexity_score=7.0,
            parallelism_potential=0.6,
            memory_intensive=True,
            io_intensive=False,
            cpu_intensive=False,
            estimated_duration_minutes=25.0
        )
        
        system_resources = SystemResources(
            total_cpu_cores=16,
            available_cpu_cores=12,
            total_memory_gb=64.0,
            available_memory_gb=48.0,
            cpu_usage_percent=25.0,
            memory_usage_percent=25.0,
            disk_io_usage_percent=5.0,
            network_io_mbps=3.0
        )
        
        allocation = resource_optimizer._memory_optimized_allocation(workload, system_resources)
        
        assert isinstance(allocation, ResourceAllocation)
        # Memory optimized should have fewer executors but more memory each
        assert allocation.executor_instances <= 6
        # Should allocate significant memory per executor
        memory_gb = int(allocation.executor_memory.replace('g', ''))
        assert memory_gb >= 3
    
    def test_cpu_optimized_allocation_strategy(self, resource_optimizer):
        """Test CPU-optimized allocation strategy."""
        workload = WorkloadCharacteristics(
            data_size_mb=150.0,
            complexity_score=6.0,
            parallelism_potential=0.9,
            memory_intensive=False,
            io_intensive=False,
            cpu_intensive=True,
            estimated_duration_minutes=20.0
        )
        
        system_resources = SystemResources(
            total_cpu_cores=24,
            available_cpu_cores=20,
            total_memory_gb=32.0,
            available_memory_gb=24.0,
            cpu_usage_percent=16.7,
            memory_usage_percent=25.0,
            disk_io_usage_percent=3.0,
            network_io_mbps=2.0
        )
        
        allocation = resource_optimizer._cpu_optimized_allocation(workload, system_resources)
        
        assert isinstance(allocation, ResourceAllocation)
        # CPU optimized should have more cores per executor
        assert allocation.executor_cores >= 2
        # Should have higher shuffle partitions for CPU-intensive tasks
        assert allocation.sql_shuffle_partitions >= allocation.executor_instances * allocation.executor_cores * 8
    
    def test_optimize_allocation(self, resource_optimizer):
        """Test the main optimize_allocation method."""
        workload = WorkloadCharacteristics(
            data_size_mb=100.0,
            complexity_score=5.0,
            parallelism_potential=0.7,
            memory_intensive=False,
            io_intensive=False,
            cpu_intensive=False,
            estimated_duration_minutes=15.0
        )
        
        with patch.object(resource_optimizer, 'get_system_resources') as mock_get_resources:
            mock_resources = SystemResources(
                total_cpu_cores=8,
                available_cpu_cores=6,
                total_memory_gb=16.0,
                available_memory_gb=12.0,
                cpu_usage_percent=25.0,
                memory_usage_percent=25.0,
                disk_io_usage_percent=5.0,
                network_io_mbps=2.0
            )
            mock_get_resources.return_value = mock_resources
            
            allocation = resource_optimizer.optimize_allocation(workload, 'balanced')
            
            assert isinstance(allocation, ResourceAllocation)
            assert resource_optimizer.current_allocation is allocation
            assert len(resource_optimizer.allocation_history) > 0
    
    def test_validate_allocation(self, resource_optimizer):
        """Test allocation validation and adjustment."""
        # Test allocation that exceeds system resources
        over_allocation = ResourceAllocation(
            executor_instances=20,  # Too many
            executor_cores=8,       # Too many cores
            executor_memory="32g",  # Too much memory
            driver_memory="16g",
            max_result_size="4g",
            broadcast_threshold="1g",
            sql_shuffle_partitions=160
        )
        
        system_resources = SystemResources(
            total_cpu_cores=8,
            available_cpu_cores=6,
            total_memory_gb=16.0,
            available_memory_gb=12.0,
            cpu_usage_percent=25.0,
            memory_usage_percent=25.0,
            disk_io_usage_percent=5.0,
            network_io_mbps=2.0
        )
        
        validated = resource_optimizer._validate_allocation(over_allocation, system_resources)
        
        assert isinstance(validated, ResourceAllocation)
        # Should be reduced to fit system resources
        assert validated.executor_instances <= system_resources.available_cpu_cores
        assert validated.executor_instances >= resource_optimizer.config['min_executor_instances']
        assert validated.executor_instances <= resource_optimizer.config['max_executor_instances']
        
        # Memory should be adjusted
        executor_memory_gb = int(validated.executor_memory.replace('g', ''))
        driver_memory_gb = int(validated.driver_memory.replace('g', ''))
        total_memory_needed = (validated.executor_instances * executor_memory_gb) + driver_memory_gb
        assert total_memory_needed <= system_resources.available_memory_gb * 0.9
    
    def test_apply_allocation(self, resource_optimizer, mock_spark_session):
        """Test applying allocation to Spark session."""
        allocation = ResourceAllocation(
            executor_instances=4,
            executor_cores=2,
            executor_memory="4g",
            driver_memory="2g",
            max_result_size="1g",
            broadcast_threshold="200MB",
            sql_shuffle_partitions=16
        )
        
        result = resource_optimizer.apply_allocation(allocation)
        
        assert result is True
        # Verify Spark configuration was set
        assert mock_spark_session.conf.set.call_count >= 7  # All config parameters
    
    def test_apply_allocation_no_spark(self, resource_optimizer):
        """Test applying allocation when no Spark session available."""
        resource_optimizer.spark = None
        
        allocation = ResourceAllocation(
            executor_instances=2,
            executor_cores=2,
            executor_memory="2g",
            driver_memory="1g",
            max_result_size="1g",
            broadcast_threshold="100MB",
            sql_shuffle_partitions=8
        )
        
        result = resource_optimizer.apply_allocation(allocation)
        assert result is False
    
    def test_monitoring_lifecycle(self, resource_optimizer):
        """Test monitoring start and stop."""
        assert resource_optimizer.monitoring_enabled is False
        
        resource_optimizer.start_monitoring()
        assert resource_optimizer.monitoring_enabled is True
        assert resource_optimizer.monitoring_thread is not None
        
        resource_optimizer.stop_monitoring()
        assert resource_optimizer.monitoring_enabled is False
    
    def test_generate_optimization_recommendations(self, resource_optimizer):
        """Test optimization recommendation generation."""
        # Add some performance history
        low_efficiency_metrics = PerformanceMetrics(
            job_duration_seconds=300.0,
            cpu_utilization=0.3,
            memory_utilization=0.9,
            gc_time_percent=25.0,
            shuffle_read_mb=1000.0,
            shuffle_write_mb=800.0,
            input_size_mb=500.0,
            output_size_mb=400.0,
            task_count=100,
            failed_tasks=20
        )
        
        # Add to history
        resource_optimizer.performance_history.append({
            'timestamp': datetime.now(),
            'job_id': 'test_job',
            'metrics': low_efficiency_metrics,
            'allocation': None
        })
        
        # Mock current resources with high usage
        resource_optimizer.current_resources = SystemResources(
            total_cpu_cores=8,
            available_cpu_cores=1,  # High CPU usage
            total_memory_gb=16.0,
            available_memory_gb=1.0,  # High memory usage
            cpu_usage_percent=95.0,
            memory_usage_percent=90.0,
            disk_io_usage_percent=5.0,
            network_io_mbps=2.0
        )
        
        recommendations = resource_optimizer._generate_optimization_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should recommend reducing resource usage due to high system load
        cpu_recommendation = any('cpu' in rec.lower() for rec in recommendations)
        memory_recommendation = any('memory' in rec.lower() for rec in recommendations)
        
        assert cpu_recommendation or memory_recommendation
    
    def test_get_optimization_report(self, resource_optimizer):
        """Test optimization report generation."""
        # Add some test data
        test_metrics = PerformanceMetrics(
            job_duration_seconds=120.0,
            cpu_utilization=0.75,
            memory_utilization=0.65,
            gc_time_percent=8.0,
            shuffle_read_mb=200.0,
            shuffle_write_mb=150.0,
            input_size_mb=400.0,
            output_size_mb=320.0,
            task_count=50,
            failed_tasks=1
        )
        
        resource_optimizer.performance_history.append({
            'timestamp': datetime.now(),
            'job_id': 'test_job',
            'metrics': test_metrics,
            'allocation': None
        })
        
        report = resource_optimizer.get_optimization_report()
        
        assert isinstance(report, dict)
        assert 'timestamp' in report
        assert 'optimization_status' in report
        assert 'average_efficiency_score' in report
        assert 'performance_history_size' in report
        assert 'allocation_history_size' in report
        assert 'current_system_resources' in report
        assert 'current_allocation' in report
        assert 'recommendations' in report
        
        # Verify data types
        assert isinstance(report['average_efficiency_score'], (int, float))
        assert isinstance(report['performance_history_size'], int)
        assert isinstance(report['allocation_history_size'], int)
        assert isinstance(report['recommendations'], list)


class TestResourceOptimizerUtilities:
    """Test utility functions for resource optimization."""
    
    @patch('src.spark.resource_optimizer.SparkResourceOptimizer')
    def test_get_resource_optimizer_singleton(self, mock_optimizer_class):
        """Test global optimizer singleton pattern."""
        # Reset global state
        import src.spark.resource_optimizer
        src.spark.resource_optimizer._global_resource_optimizer = None
        
        mock_instance = Mock()
        mock_optimizer_class.return_value = mock_instance
        
        # First call should create instance
        optimizer1 = get_resource_optimizer()
        assert optimizer1 is mock_instance
        
        # Second call should return same instance
        optimizer2 = get_resource_optimizer()
        assert optimizer2 is optimizer1
        
        # Only one instance should be created
        mock_optimizer_class.assert_called_once()
    
    @patch('src.spark.resource_optimizer.get_resource_optimizer')
    def test_optimize_for_workload_convenience(self, mock_get_optimizer):
        """Test convenience function for workload optimization."""
        mock_optimizer = Mock()
        mock_workload = Mock()
        mock_allocation = Mock()
        
        mock_optimizer.analyze_workload.return_value = mock_workload
        mock_optimizer.optimize_allocation.return_value = mock_allocation
        mock_get_optimizer.return_value = mock_optimizer
        
        result = optimize_for_workload(
            data_size_mb=100.0,
            processing_type='text_extraction',
            strategy='balanced'
        )
        
        assert result is mock_allocation
        mock_optimizer.analyze_workload.assert_called_once_with(100.0, 'text_extraction')
        mock_optimizer.optimize_allocation.assert_called_once_with(mock_workload, 'balanced')


class TestResourceOptimizerIntegration:
    """Integration tests for resource optimizer components."""
    
    @pytest.mark.slow
    def test_end_to_end_optimization_flow(self):
        """Test complete optimization flow."""
        with patch('src.spark.resource_optimizer.SparkSession') as mock_spark_class:
            with patch('src.spark.resource_optimizer.get_metrics_collector') as mock_metrics:
                with patch('src.spark.resource_optimizer.psutil') as mock_psutil:
                    
                    # Setup mocks
                    mock_session = Mock()
                    mock_session.sparkContext.applicationId = "test_app"
                    mock_session.conf.set.return_value = None
                    mock_spark_class.getActiveSession.return_value = mock_session
                    
                    mock_metrics.return_value = Mock()
                    
                    # Mock system resources
                    mock_psutil.cpu_count.return_value = 8
                    mock_psutil.cpu_percent.return_value = 25.0
                    mock_memory = Mock()
                    mock_memory.total = 16 * 1024**3
                    mock_memory.available = 12 * 1024**3
                    mock_memory.percent = 25.0
                    mock_psutil.virtual_memory.return_value = mock_memory
                    mock_psutil.disk_io_counters.return_value = Mock(read_bytes=0, write_bytes=0)
                    mock_psutil.net_io_counters.return_value = Mock(bytes_sent=0, bytes_recv=0)
                    
                    # Create optimizer
                    optimizer = SparkResourceOptimizer(mock_session)
                    
                    # Test full workflow
                    workload = optimizer.analyze_workload(200.0, 'llm_processing', 7.0)
                    allocation = optimizer.optimize_allocation(workload, 'balanced')
                    apply_result = optimizer.apply_allocation(allocation)
                    report = optimizer.get_optimization_report()
                    
                    # Verify results
                    assert isinstance(workload, WorkloadCharacteristics)
                    assert isinstance(allocation, ResourceAllocation)
                    assert apply_result is True
                    assert isinstance(report, dict)
                    assert 'recommendations' in report
    
    def test_error_handling_and_recovery(self):
        """Test error handling in optimization components."""
        with patch('src.spark.resource_optimizer.SparkSession') as mock_spark_class:
            with patch('src.spark.resource_optimizer.get_metrics_collector') as mock_metrics:
                
                # Setup failing Spark session
                mock_session = Mock()
                mock_session.conf.set.side_effect = Exception("Spark configuration error")
                mock_spark_class.getActiveSession.return_value = mock_session
                
                mock_metrics.return_value = Mock()
                
                optimizer = SparkResourceOptimizer(mock_session)
                
                # Test that apply_allocation handles errors gracefully
                allocation = ResourceAllocation(
                    executor_instances=2,
                    executor_cores=2,
                    executor_memory="2g",
                    driver_memory="1g",
                    max_result_size="1g",
                    broadcast_threshold="100MB",
                    sql_shuffle_partitions=8
                )
                
                result = optimizer.apply_allocation(allocation)
                assert result is False  # Should fail gracefully
    
    def test_performance_under_load(self):
        """Test optimizer performance with multiple rapid calls."""
        with patch('src.spark.resource_optimizer.SparkSession') as mock_spark_class:
            with patch('src.spark.resource_optimizer.get_metrics_collector') as mock_metrics:
                with patch('src.spark.resource_optimizer.psutil') as mock_psutil:
                    
                    # Setup lightweight mocks
                    mock_session = Mock()
                    mock_spark_class.getActiveSession.return_value = mock_session
                    mock_metrics.return_value = Mock()
                    
                    # Mock minimal psutil responses
                    mock_psutil.cpu_count.return_value = 4
                    mock_psutil.cpu_percent.return_value = 50.0
                    mock_memory = Mock()
                    mock_memory.total = 8 * 1024**3
                    mock_memory.available = 6 * 1024**3
                    mock_memory.percent = 25.0
                    mock_psutil.virtual_memory.return_value = mock_memory
                    mock_psutil.disk_io_counters.return_value = None
                    mock_psutil.net_io_counters.return_value = None
                    
                    optimizer = SparkResourceOptimizer(mock_session)
                    
                    # Test rapid optimization calls
                    start_time = time.time()
                    
                    for i in range(10):
                        workload = optimizer.analyze_workload(50.0 * (i + 1), 'general', 5.0)
                        allocation = optimizer.optimize_allocation(workload, 'balanced')
                        
                        assert isinstance(workload, WorkloadCharacteristics)
                        assert isinstance(allocation, ResourceAllocation)
                    
                    end_time = time.time()
                    elapsed = end_time - start_time
                    
                    # Should complete quickly (less than 2 seconds for 10 optimizations)
                    assert elapsed < 2.0