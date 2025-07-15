"""
Dynamic resource optimization for Spark clusters.

This module provides intelligent resource allocation and optimization for Spark
jobs based on workload characteristics, system resources, and performance metrics.
"""

import logging
import time
import psutil
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark import SparkContext
import json
import threading
from collections import deque, defaultdict

from ..monitoring.prometheus_metrics import get_metrics_collector


@dataclass
class ResourceAllocation:
    """Represents a resource allocation configuration."""
    executor_instances: int
    executor_cores: int
    executor_memory: str
    driver_memory: str
    max_result_size: str
    broadcast_threshold: str
    sql_shuffle_partitions: int
    
    def to_spark_config(self) -> Dict[str, str]:
        """Convert to Spark configuration dictionary."""
        return {
            'spark.executor.instances': str(self.executor_instances),
            'spark.executor.cores': str(self.executor_cores),
            'spark.executor.memory': self.executor_memory,
            'spark.driver.memory': self.driver_memory,
            'spark.driver.maxResultSize': self.max_result_size,
            'spark.sql.autoBroadcastJoinThreshold': self.broadcast_threshold,
            'spark.sql.shuffle.partitions': str(self.sql_shuffle_partitions)
        }


@dataclass
class WorkloadCharacteristics:
    """Characteristics of a processing workload."""
    data_size_mb: float
    complexity_score: float  # 1-10 scale
    parallelism_potential: float  # 0-1 scale
    memory_intensive: bool
    io_intensive: bool
    cpu_intensive: bool
    estimated_duration_minutes: float
    
    def __post_init__(self):
        """Validate characteristics."""
        self.complexity_score = max(1, min(10, self.complexity_score))
        self.parallelism_potential = max(0, min(1, self.parallelism_potential))


@dataclass
class SystemResources:
    """Current system resource availability."""
    total_cpu_cores: int
    available_cpu_cores: int
    total_memory_gb: float
    available_memory_gb: float
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_io_usage_percent: float
    network_io_mbps: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceMetrics:
    """Performance metrics for resource optimization."""
    job_duration_seconds: float
    cpu_utilization: float
    memory_utilization: float
    gc_time_percent: float
    shuffle_read_mb: float
    shuffle_write_mb: float
    input_size_mb: float
    output_size_mb: float
    task_count: int
    failed_tasks: int
    
    def efficiency_score(self) -> float:
        """Calculate efficiency score (0-1)."""
        # Penalize high GC time and failed tasks
        gc_penalty = max(0, 1 - (self.gc_time_percent / 20))
        failure_penalty = max(0, 1 - (self.failed_tasks / max(1, self.task_count)))
        
        # Reward high resource utilization (but not too high)
        cpu_efficiency = 1 - abs(0.8 - self.cpu_utilization)
        memory_efficiency = 1 - abs(0.7 - self.memory_utilization)
        
        return (gc_penalty + failure_penalty + cpu_efficiency + memory_efficiency) / 4


class SparkResourceOptimizer:
    """
    Dynamic resource optimizer for Spark clusters.
    
    Provides intelligent resource allocation based on:
    - Workload characteristics
    - System resource availability
    - Historical performance metrics
    - Real-time monitoring data
    """
    
    def __init__(self, spark_session: Optional[SparkSession] = None):
        """
        Initialize the resource optimizer.
        
        Args:
            spark_session: Existing Spark session or None to use active session
        """
        self.spark = spark_session or SparkSession.getActiveSession()
        self.logger = logging.getLogger(__name__)
        self.metrics_collector = get_metrics_collector()
        
        # Resource optimization history
        self.performance_history = deque(maxlen=100)
        self.allocation_history = deque(maxlen=50)
        
        # Current system state
        self.current_resources = None
        self.current_allocation = None
        
        # Optimization strategies
        self.optimization_strategies = {
            'conservative': self._conservative_allocation,
            'balanced': self._balanced_allocation,
            'aggressive': self._aggressive_allocation,
            'memory_optimized': self._memory_optimized_allocation,
            'cpu_optimized': self._cpu_optimized_allocation
        }
        
        # Resource monitoring
        self.monitoring_enabled = False
        self.monitoring_thread = None
        self.monitoring_interval = 30  # seconds
        
        # Configuration
        self.config = {
            'min_executor_instances': 1,
            'max_executor_instances': 20,
            'min_executor_cores': 1,
            'max_executor_cores': 8,
            'min_executor_memory_gb': 1,
            'max_executor_memory_gb': 16,
            'optimization_threshold': 0.1,  # Minimum improvement to trigger change
            'adaptation_rate': 0.2  # How quickly to adapt to changes
        }
        
        self.logger.info("Spark resource optimizer initialized")
    
    def analyze_workload(self, data_size_mb: float, 
                        processing_type: str = 'general',
                        estimated_complexity: float = 5.0) -> WorkloadCharacteristics:
        """
        Analyze workload characteristics for resource optimization.
        
        Args:
            data_size_mb: Size of input data in MB
            processing_type: Type of processing (text_extraction, llm_processing, etc.)
            estimated_complexity: Complexity estimate (1-10)
            
        Returns:
            WorkloadCharacteristics object
        """
        
        # Determine characteristics based on processing type
        type_characteristics = {
            'text_extraction': {
                'parallelism_potential': 0.8,
                'memory_intensive': False,
                'io_intensive': True,
                'cpu_intensive': False
            },
            'llm_processing': {
                'parallelism_potential': 0.6,
                'memory_intensive': True,
                'io_intensive': False,
                'cpu_intensive': True
            },
            'audio_generation': {
                'parallelism_potential': 0.7,
                'memory_intensive': False,
                'io_intensive': True,
                'cpu_intensive': True
            },
            'validation': {
                'parallelism_potential': 0.9,
                'memory_intensive': False,
                'io_intensive': False,
                'cpu_intensive': True
            },
            'general': {
                'parallelism_potential': 0.7,
                'memory_intensive': False,
                'io_intensive': False,
                'cpu_intensive': True
            }
        }
        
        characteristics = type_characteristics.get(processing_type, type_characteristics['general'])
        
        # Estimate duration based on data size and complexity
        base_duration = (data_size_mb / 100) * estimated_complexity  # Base estimate
        duration_factor = {
            'text_extraction': 0.5,
            'llm_processing': 2.0,
            'audio_generation': 1.5,
            'validation': 0.3,
            'general': 1.0
        }
        estimated_duration = base_duration * duration_factor.get(processing_type, 1.0)
        
        return WorkloadCharacteristics(
            data_size_mb=data_size_mb,
            complexity_score=estimated_complexity,
            parallelism_potential=characteristics['parallelism_potential'],
            memory_intensive=characteristics['memory_intensive'],
            io_intensive=characteristics['io_intensive'],
            cpu_intensive=characteristics['cpu_intensive'],
            estimated_duration_minutes=estimated_duration
        )
    
    def get_system_resources(self) -> SystemResources:
        """Get current system resource availability."""
        
        # CPU information
        cpu_count = psutil.cpu_count(logical=True)
        cpu_usage = psutil.cpu_percent(interval=1)
        available_cpu = max(1, int(cpu_count * (1 - cpu_usage / 100)))
        
        # Memory information
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024**3)
        available_memory_gb = memory.available / (1024**3)
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_usage = 0
        if disk_io:
            # Simplified disk usage calculation
            disk_usage = min(100, (disk_io.read_bytes + disk_io.write_bytes) / (1024**2) / 10)
        
        # Network I/O
        network_io = psutil.net_io_counters()
        network_usage = 0
        if network_io:
            network_usage = (network_io.bytes_sent + network_io.bytes_recv) / (1024**2)
        
        resources = SystemResources(
            total_cpu_cores=cpu_count,
            available_cpu_cores=available_cpu,
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb,
            cpu_usage_percent=cpu_usage,
            memory_usage_percent=memory.percent,
            disk_io_usage_percent=disk_usage,
            network_io_mbps=network_usage
        )
        
        self.current_resources = resources
        
        # Record system metrics
        self.metrics_collector.set_resource_usage(
            'spark_optimizer', 'localhost',
            cpu_usage, int(memory.used)
        )
        
        return resources
    
    def optimize_allocation(self, workload: WorkloadCharacteristics,
                          strategy: str = 'balanced') -> ResourceAllocation:
        """
        Optimize resource allocation for a given workload.
        
        Args:
            workload: Workload characteristics
            strategy: Optimization strategy
            
        Returns:
            Optimized resource allocation
        """
        
        if strategy not in self.optimization_strategies:
            self.logger.warning(f"Unknown strategy {strategy}, using 'balanced'")
            strategy = 'balanced'
        
        # Get current system resources
        system_resources = self.get_system_resources()
        
        # Apply optimization strategy
        allocation = self.optimization_strategies[strategy](workload, system_resources)
        
        # Validate and adjust allocation
        allocation = self._validate_allocation(allocation, system_resources)
        
        # Store allocation history
        self.allocation_history.append({
            'timestamp': datetime.now(),
            'workload': workload,
            'allocation': allocation,
            'strategy': strategy,
            'system_resources': system_resources
        })
        
        self.current_allocation = allocation
        
        self.logger.info(f"Optimized allocation for {strategy} strategy: "
                        f"{allocation.executor_instances} executors, "
                        f"{allocation.executor_cores} cores each, "
                        f"{allocation.executor_memory} memory")
        
        return allocation
    
    def _conservative_allocation(self, workload: WorkloadCharacteristics,
                               system_resources: SystemResources) -> ResourceAllocation:
        """Conservative resource allocation strategy."""
        
        # Use fewer resources to ensure stability
        executor_instances = max(1, min(4, system_resources.available_cpu_cores // 2))
        executor_cores = min(2, system_resources.available_cpu_cores // executor_instances)
        
        # Conservative memory allocation
        memory_per_executor = min(4, system_resources.available_memory_gb / (executor_instances + 1))
        executor_memory = f"{int(memory_per_executor)}g"
        driver_memory = f"{max(1, int(memory_per_executor * 0.5))}g"
        
        return ResourceAllocation(
            executor_instances=executor_instances,
            executor_cores=executor_cores,
            executor_memory=executor_memory,
            driver_memory=driver_memory,
            max_result_size="1g",
            broadcast_threshold="100MB",
            sql_shuffle_partitions=executor_instances * executor_cores * 2
        )
    
    def _balanced_allocation(self, workload: WorkloadCharacteristics,
                           system_resources: SystemResources) -> ResourceAllocation:
        """Balanced resource allocation strategy."""
        
        # Balance between performance and resource usage
        parallelism_factor = workload.parallelism_potential
        
        # Calculate optimal executor count
        max_executors = min(
            self.config['max_executor_instances'],
            int(system_resources.available_cpu_cores * 0.8)
        )
        executor_instances = max(
            self.config['min_executor_instances'],
            int(max_executors * parallelism_factor)
        )
        
        # Determine cores per executor
        total_cores = system_resources.available_cpu_cores
        cores_per_executor = max(1, min(
            self.config['max_executor_cores'],
            total_cores // executor_instances
        ))
        
        # Memory allocation based on workload characteristics
        base_memory = system_resources.available_memory_gb / (executor_instances + 1)
        if workload.memory_intensive:
            memory_multiplier = 1.5
        else:
            memory_multiplier = 1.0
        
        memory_per_executor = min(
            self.config['max_executor_memory_gb'],
            base_memory * memory_multiplier
        )
        
        executor_memory = f"{int(memory_per_executor)}g"
        driver_memory = f"{max(2, int(memory_per_executor * 0.6))}g"
        
        # Adjust shuffle partitions based on data size
        shuffle_partitions = max(
            executor_instances * cores_per_executor * 2,
            int(workload.data_size_mb / 128)  # 128MB per partition
        )
        
        return ResourceAllocation(
            executor_instances=executor_instances,
            executor_cores=cores_per_executor,
            executor_memory=executor_memory,
            driver_memory=driver_memory,
            max_result_size="2g",
            broadcast_threshold="200MB",
            sql_shuffle_partitions=shuffle_partitions
        )
    
    def _aggressive_allocation(self, workload: WorkloadCharacteristics,
                             system_resources: SystemResources) -> ResourceAllocation:
        """Aggressive resource allocation strategy."""
        
        # Use maximum available resources
        executor_instances = min(
            self.config['max_executor_instances'],
            max(2, int(system_resources.available_cpu_cores * 0.9))
        )
        
        cores_per_executor = min(
            self.config['max_executor_cores'],
            max(2, system_resources.available_cpu_cores // executor_instances)
        )
        
        # Aggressive memory allocation
        memory_per_executor = min(
            self.config['max_executor_memory_gb'],
            system_resources.available_memory_gb * 0.8 / executor_instances
        )
        
        executor_memory = f"{int(memory_per_executor)}g"
        driver_memory = f"{max(2, int(memory_per_executor * 0.7))}g"
        
        # High shuffle partitions for maximum parallelism
        shuffle_partitions = executor_instances * cores_per_executor * 4
        
        return ResourceAllocation(
            executor_instances=executor_instances,
            executor_cores=cores_per_executor,
            executor_memory=executor_memory,
            driver_memory=driver_memory,
            max_result_size="4g",
            broadcast_threshold="512MB",
            sql_shuffle_partitions=shuffle_partitions
        )
    
    def _memory_optimized_allocation(self, workload: WorkloadCharacteristics,
                                   system_resources: SystemResources) -> ResourceAllocation:
        """Memory-optimized resource allocation strategy."""
        
        # Fewer executors with more memory each
        executor_instances = max(1, min(6, system_resources.available_cpu_cores // 3))
        cores_per_executor = min(4, system_resources.available_cpu_cores // executor_instances)
        
        # Maximum memory allocation
        memory_per_executor = min(
            self.config['max_executor_memory_gb'],
            system_resources.available_memory_gb * 0.9 / executor_instances
        )
        
        executor_memory = f"{int(memory_per_executor)}g"
        driver_memory = f"{max(3, int(memory_per_executor * 0.8))}g"
        
        return ResourceAllocation(
            executor_instances=executor_instances,
            executor_cores=cores_per_executor,
            executor_memory=executor_memory,
            driver_memory=driver_memory,
            max_result_size="8g",
            broadcast_threshold="1g",
            sql_shuffle_partitions=executor_instances * cores_per_executor * 2
        )
    
    def _cpu_optimized_allocation(self, workload: WorkloadCharacteristics,
                                system_resources: SystemResources) -> ResourceAllocation:
        """CPU-optimized resource allocation strategy."""
        
        # More executors with more cores each
        executor_instances = min(
            self.config['max_executor_instances'],
            system_resources.available_cpu_cores // 2
        )
        
        cores_per_executor = min(
            self.config['max_executor_cores'],
            max(2, system_resources.available_cpu_cores // executor_instances)
        )
        
        # Moderate memory allocation
        memory_per_executor = min(
            6,  # Max 6GB per executor for CPU workloads
            system_resources.available_memory_gb / (executor_instances + 1)
        )
        
        executor_memory = f"{int(memory_per_executor)}g"
        driver_memory = f"{max(1, int(memory_per_executor * 0.5))}g"
        
        # Higher shuffle partitions for CPU-intensive tasks
        shuffle_partitions = executor_instances * cores_per_executor * 8
        
        return ResourceAllocation(
            executor_instances=executor_instances,
            executor_cores=cores_per_executor,
            executor_memory=executor_memory,
            driver_memory=driver_memory,
            max_result_size="2g",
            broadcast_threshold="100MB",
            sql_shuffle_partitions=shuffle_partitions
        )
    
    def _validate_allocation(self, allocation: ResourceAllocation,
                           system_resources: SystemResources) -> ResourceAllocation:
        """Validate and adjust resource allocation."""
        
        # Ensure allocation doesn't exceed system resources
        total_cores_needed = allocation.executor_instances * allocation.executor_cores
        if total_cores_needed > system_resources.available_cpu_cores:
            # Reduce instances or cores
            allocation.executor_instances = min(
                allocation.executor_instances,
                system_resources.available_cpu_cores // allocation.executor_cores
            )
        
        # Parse and validate memory
        executor_memory_gb = int(allocation.executor_memory.replace('g', ''))
        driver_memory_gb = int(allocation.driver_memory.replace('g', ''))
        
        total_memory_needed = (allocation.executor_instances * executor_memory_gb) + driver_memory_gb
        if total_memory_needed > system_resources.available_memory_gb * 0.9:
            # Scale down memory
            scale_factor = (system_resources.available_memory_gb * 0.9) / total_memory_needed
            executor_memory_gb = max(1, int(executor_memory_gb * scale_factor))
            driver_memory_gb = max(1, int(driver_memory_gb * scale_factor))
            
            allocation.executor_memory = f"{executor_memory_gb}g"
            allocation.driver_memory = f"{driver_memory_gb}g"
        
        # Validate against configuration limits
        allocation.executor_instances = max(
            self.config['min_executor_instances'],
            min(self.config['max_executor_instances'], allocation.executor_instances)
        )
        
        allocation.executor_cores = max(
            self.config['min_executor_cores'],
            min(self.config['max_executor_cores'], allocation.executor_cores)
        )
        
        return allocation
    
    def apply_allocation(self, allocation: ResourceAllocation) -> bool:
        """
        Apply resource allocation to Spark session.
        
        Args:
            allocation: Resource allocation to apply
            
        Returns:
            True if successful, False otherwise
        """
        
        if not self.spark:
            self.logger.error("No Spark session available")
            return False
        
        try:
            # Get Spark configuration
            spark_config = allocation.to_spark_config()
            
            # Apply configuration
            for key, value in spark_config.items():
                self.spark.conf.set(key, value)
            
            self.logger.info(f"Applied resource allocation: {spark_config}")
            
            # Record allocation metrics
            self.metrics_collector.set_spark_executors(
                self.spark.sparkContext.applicationId,
                allocation.executor_instances
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply resource allocation: {e}")
            return False
    
    def monitor_performance(self, job_id: str) -> Optional[PerformanceMetrics]:
        """Monitor and collect performance metrics for optimization."""
        
        if not self.spark:
            return None
        
        try:
            sc = self.spark.sparkContext
            status_tracker = sc.statusTracker()
            
            # Get job information
            job_ids = status_tracker.getJobIds()
            if not job_ids:
                return None
            
            # Get metrics for the latest job
            latest_job_id = max(job_ids)
            job_info = status_tracker.getJobInfo(latest_job_id)
            
            if not job_info:
                return None
            
            # Collect performance metrics
            stage_ids = job_info.stageIds
            total_task_count = 0
            total_failed_tasks = 0
            total_shuffle_read = 0
            total_shuffle_write = 0
            
            for stage_id in stage_ids:
                stage_info = status_tracker.getStageInfo(stage_id)
                if stage_info:
                    total_task_count += stage_info.numTasks
                    total_failed_tasks += stage_info.numFailedTasks
                    
                    # Shuffle metrics (if available)
                    if hasattr(stage_info, 'shuffleReadBytes'):
                        total_shuffle_read += stage_info.shuffleReadBytes / (1024**2)  # MB
                    if hasattr(stage_info, 'shuffleWriteBytes'):
                        total_shuffle_write += stage_info.shuffleWriteBytes / (1024**2)  # MB
            
            # Calculate metrics
            job_duration = time.time() - (job_info.submissionTime / 1000) if job_info.submissionTime else 0
            
            # Get executor information
            executor_infos = status_tracker.getExecutorInfos()
            total_cpu_time = sum(info.totalCores for info in executor_infos)
            total_memory = sum(info.maxMemory for info in executor_infos)
            
            # Calculate utilization (simplified)
            cpu_utilization = min(1.0, total_cpu_time / (job_duration * 100)) if job_duration > 0 else 0
            memory_utilization = 0.7  # Placeholder - would need more detailed metrics
            
            metrics = PerformanceMetrics(
                job_duration_seconds=job_duration,
                cpu_utilization=cpu_utilization,
                memory_utilization=memory_utilization,
                gc_time_percent=5.0,  # Placeholder
                shuffle_read_mb=total_shuffle_read,
                shuffle_write_mb=total_shuffle_write,
                input_size_mb=0,  # Would need to be tracked separately
                output_size_mb=0,  # Would need to be tracked separately
                task_count=total_task_count,
                failed_tasks=total_failed_tasks
            )
            
            # Store performance history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'job_id': job_id,
                'metrics': metrics,
                'allocation': self.current_allocation
            })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect performance metrics: {e}")
            return None
    
    def start_monitoring(self):
        """Start continuous resource monitoring."""
        if self.monitoring_enabled:
            return
        
        self.monitoring_enabled = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Started resource monitoring")
    
    def stop_monitoring(self):
        """Stop continuous resource monitoring."""
        self.monitoring_enabled = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        self.logger.info("Stopped resource monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_enabled:
            try:
                # Update system resources
                self.get_system_resources()
                
                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        
        current_time = datetime.now()
        
        # Calculate average performance over recent history
        recent_performance = [
            entry['metrics'] for entry in self.performance_history
            if (current_time - entry['timestamp']).total_seconds() < 3600  # Last hour
        ]
        
        avg_efficiency = 0
        if recent_performance:
            avg_efficiency = sum(metrics.efficiency_score() for metrics in recent_performance) / len(recent_performance)
        
        # System resource summary
        system_summary = {}
        if self.current_resources:
            system_summary = {
                'cpu_usage_percent': self.current_resources.cpu_usage_percent,
                'memory_usage_percent': self.current_resources.memory_usage_percent,
                'available_cores': self.current_resources.available_cpu_cores,
                'available_memory_gb': self.current_resources.available_memory_gb
            }
        
        # Current allocation summary
        allocation_summary = {}
        if self.current_allocation:
            allocation_summary = {
                'executor_instances': self.current_allocation.executor_instances,
                'executor_cores': self.current_allocation.executor_cores,
                'executor_memory': self.current_allocation.executor_memory,
                'total_cores': self.current_allocation.executor_instances * self.current_allocation.executor_cores
            }
        
        return {
            'timestamp': current_time.isoformat(),
            'optimization_status': 'active' if self.monitoring_enabled else 'inactive',
            'average_efficiency_score': avg_efficiency,
            'performance_history_size': len(self.performance_history),
            'allocation_history_size': len(self.allocation_history),
            'current_system_resources': system_summary,
            'current_allocation': allocation_summary,
            'recommendations': self._generate_optimization_recommendations()
        }
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on history."""
        recommendations = []
        
        if not self.performance_history:
            recommendations.append("No performance history available - run more jobs to get recommendations")
            return recommendations
        
        # Analyze recent performance
        recent_metrics = [entry['metrics'] for entry in self.performance_history[-10:]]
        
        if len(recent_metrics) >= 3:
            avg_efficiency = sum(m.efficiency_score() for m in recent_metrics) / len(recent_metrics)
            avg_failed_tasks = sum(m.failed_tasks for m in recent_metrics) / len(recent_metrics)
            avg_gc_time = sum(m.gc_time_percent for m in recent_metrics) / len(recent_metrics)
            
            if avg_efficiency < 0.6:
                recommendations.append("Low efficiency detected - consider adjusting resource allocation")
            
            if avg_failed_tasks > 0.1:
                recommendations.append("High task failure rate - consider reducing parallelism or increasing memory")
            
            if avg_gc_time > 15:
                recommendations.append("High GC time - consider increasing executor memory")
        
        # System resource recommendations
        if self.current_resources:
            if self.current_resources.cpu_usage_percent > 90:
                recommendations.append("High CPU usage - consider reducing executor instances")
            elif self.current_resources.cpu_usage_percent < 30:
                recommendations.append("Low CPU usage - consider increasing parallelism")
            
            if self.current_resources.memory_usage_percent > 85:
                recommendations.append("High memory usage - consider reducing executor memory or instances")
        
        return recommendations


# Global resource optimizer instance
_global_resource_optimizer = None


def get_resource_optimizer() -> SparkResourceOptimizer:
    """Get the global resource optimizer instance."""
    global _global_resource_optimizer
    
    if _global_resource_optimizer is None:
        _global_resource_optimizer = SparkResourceOptimizer()
    
    return _global_resource_optimizer


def optimize_for_workload(data_size_mb: float, processing_type: str = 'general',
                         strategy: str = 'balanced') -> ResourceAllocation:
    """Convenience function to optimize resources for a workload."""
    optimizer = get_resource_optimizer()
    workload = optimizer.analyze_workload(data_size_mb, processing_type)
    return optimizer.optimize_allocation(workload, strategy)