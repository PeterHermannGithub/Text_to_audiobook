"""
Prometheus metrics collector for the distributed text-to-audiobook pipeline.

This module provides comprehensive metrics collection for monitoring system
performance, health, and operational statistics across all components.
"""

import time
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from dataclasses import dataclass, field
from threading import Lock
from collections import defaultdict, deque
import json
import os

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, Info, Enum,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        push_to_gateway, delete_from_gateway
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock classes for when prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def time(self): return self
        def labels(self, *args, **kwargs): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    class Summary:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def time(self): return self
        def labels(self, *args, **kwargs): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
    
    class Enum:
        def __init__(self, *args, **kwargs): pass
        def state(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    CollectorRegistry = None
    CONTENT_TYPE_LATEST = 'text/plain'
    
    def generate_latest(*args, **kwargs): return ""
    def push_to_gateway(*args, **kwargs): pass
    def delete_from_gateway(*args, **kwargs): pass


@dataclass
class MetricValue:
    """Represents a metric value with metadata."""
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    help_text: str = ""


class PrometheusMetricsCollector:
    """
    Comprehensive metrics collector for the distributed text-to-audiobook pipeline.
    
    This class provides a unified interface for collecting, storing, and exposing
    metrics from all components of the system.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None,
                 namespace: str = "text_to_audiobook",
                 enable_push_gateway: bool = False,
                 push_gateway_url: str = "localhost:9091"):
        """
        Initialize the Prometheus metrics collector.
        
        Args:
            registry: Prometheus registry instance
            namespace: Namespace for all metrics
            enable_push_gateway: Whether to enable push gateway
            push_gateway_url: URL of the push gateway
        """
        self.registry = registry or CollectorRegistry()
        self.namespace = namespace
        self.enable_push_gateway = enable_push_gateway
        self.push_gateway_url = push_gateway_url
        self.logger = logging.getLogger(__name__)
        
        # Thread safety
        self._lock = Lock()
        
        # Metric storage
        self._metrics = {}
        self._metric_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Initialize core metrics
        self._initialize_core_metrics()
        
        # System info
        self._initialize_system_info()
        
        self.logger.info(f"Prometheus metrics collector initialized (namespace: {namespace})")
    
    def _initialize_core_metrics(self):
        """Initialize core system metrics."""
        # System health metrics
        self.system_health = Gauge(
            'system_health_status',
            'Overall system health status (1=healthy, 0=unhealthy)',
            ['component'],
            registry=self.registry,
            namespace=self.namespace
        )
        
        # Processing metrics
        self.processing_requests_total = Counter(
            'processing_requests_total',
            'Total number of processing requests',
            ['job_type', 'status'],
            registry=self.registry,
            namespace=self.namespace
        )
        
        self.processing_duration_seconds = Histogram(
            'processing_duration_seconds',
            'Processing duration in seconds',
            ['job_type', 'component'],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0],
            registry=self.registry,
            namespace=self.namespace
        )
        
        self.processing_queue_size = Gauge(
            'processing_queue_size',
            'Current size of processing queues',
            ['queue_type'],
            registry=self.registry,
            namespace=self.namespace
        )
        
        # Spark metrics
        self.spark_jobs_total = Counter(
            'spark_jobs_total',
            'Total number of Spark jobs',
            ['job_type', 'status'],
            registry=self.registry,
            namespace=self.namespace
        )
        
        self.spark_job_duration_seconds = Histogram(
            'spark_job_duration_seconds',
            'Spark job duration in seconds',
            ['job_type'],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0],
            registry=self.registry,
            namespace=self.namespace
        )
        
        self.spark_active_executors = Gauge(
            'spark_active_executors',
            'Number of active Spark executors',
            ['application_id'],
            registry=self.registry,
            namespace=self.namespace
        )
        
        self.spark_memory_usage_bytes = Gauge(
            'spark_memory_usage_bytes',
            'Spark memory usage in bytes',
            ['application_id', 'executor_id'],
            registry=self.registry,
            namespace=self.namespace
        )
        
        # Kafka metrics
        self.kafka_messages_produced_total = Counter(
            'kafka_messages_produced_total',
            'Total number of Kafka messages produced',
            ['topic', 'status'],
            registry=self.registry,
            namespace=self.namespace
        )
        
        self.kafka_messages_consumed_total = Counter(
            'kafka_messages_consumed_total',
            'Total number of Kafka messages consumed',
            ['topic', 'consumer_group'],
            registry=self.registry,
            namespace=self.namespace
        )
        
        self.kafka_consumer_lag = Gauge(
            'kafka_consumer_lag',
            'Kafka consumer lag',
            ['topic', 'partition', 'consumer_group'],
            registry=self.registry,
            namespace=self.namespace
        )
        
        self.kafka_producer_batch_size = Histogram(
            'kafka_producer_batch_size',
            'Kafka producer batch size',
            ['topic'],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500],
            registry=self.registry,
            namespace=self.namespace
        )
        
        # LLM metrics
        self.llm_requests_total = Counter(
            'llm_requests_total',
            'Total number of LLM requests',
            ['engine', 'model', 'status'],
            registry=self.registry,
            namespace=self.namespace
        )
        
        self.llm_response_time_seconds = Histogram(
            'llm_response_time_seconds',
            'LLM response time in seconds',
            ['engine', 'model'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry,
            namespace=self.namespace
        )
        
        self.llm_active_instances = Gauge(
            'llm_active_instances',
            'Number of active LLM instances',
            ['engine', 'model'],
            registry=self.registry,
            namespace=self.namespace
        )
        
        self.llm_pool_utilization = Gauge(
            'llm_pool_utilization',
            'LLM pool utilization percentage',
            ['pool_name'],
            registry=self.registry,
            namespace=self.namespace
        )
        
        # Quality metrics
        self.quality_score = Gauge(
            'quality_score',
            'Quality score for processed segments',
            ['job_id', 'metric_type'],
            registry=self.registry,
            namespace=self.namespace
        )
        
        self.validation_errors_total = Counter(
            'validation_errors_total',
            'Total number of validation errors',
            ['error_type', 'severity'],
            registry=self.registry,
            namespace=self.namespace
        )
        
        # Airflow metrics
        self.airflow_dag_runs_total = Counter(
            'airflow_dag_runs_total',
            'Total number of DAG runs',
            ['dag_id', 'status'],
            registry=self.registry,
            namespace=self.namespace
        )
        
        self.airflow_task_duration_seconds = Histogram(
            'airflow_task_duration_seconds',
            'Airflow task duration in seconds',
            ['dag_id', 'task_id'],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0],
            registry=self.registry,
            namespace=self.namespace
        )
        
        self.airflow_active_tasks = Gauge(
            'airflow_active_tasks',
            'Number of active Airflow tasks',
            ['dag_id'],
            registry=self.registry,
            namespace=self.namespace
        )
        
        # Resource metrics
        self.cpu_usage_percent = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            ['component', 'host'],
            registry=self.registry,
            namespace=self.namespace
        )
        
        self.memory_usage_bytes = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            ['component', 'host'],
            registry=self.registry,
            namespace=self.namespace
        )
        
        self.disk_usage_bytes = Gauge(
            'disk_usage_bytes',
            'Disk usage in bytes',
            ['component', 'host', 'mount_point'],
            registry=self.registry,
            namespace=self.namespace
        )
        
        # Custom business metrics
        self.segments_processed_total = Counter(
            'segments_processed_total',
            'Total number of text segments processed',
            ['job_id', 'segment_type'],
            registry=self.registry,
            namespace=self.namespace
        )
        
        self.speakers_detected_total = Counter(
            'speakers_detected_total',
            'Total number of speakers detected',
            ['job_id'],
            registry=self.registry,
            namespace=self.namespace
        )
        
        self.text_extraction_bytes = Counter(
            'text_extraction_bytes',
            'Total bytes of text extracted',
            ['file_format'],
            registry=self.registry,
            namespace=self.namespace
        )
    
    def _initialize_system_info(self):
        """Initialize system information metrics."""
        self.system_info = Info(
            'system_info',
            'System information',
            registry=self.registry,
            namespace=self.namespace
        )
        
        # Set system information
        info_data = {
            'version': '1.0.0',
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            'platform': os.name,
            'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
            'prometheus_available': str(PROMETHEUS_AVAILABLE)
        }
        
        if PROMETHEUS_AVAILABLE:
            self.system_info.info(info_data)
    
    def record_processing_request(self, job_type: str, status: str):
        """Record a processing request."""
        with self._lock:
            self.processing_requests_total.labels(
                job_type=job_type,
                status=status
            ).inc()
            
            self.logger.debug(f"Recorded processing request: {job_type} - {status}")
    
    def record_processing_duration(self, job_type: str, component: str, duration: float):
        """Record processing duration."""
        with self._lock:
            self.processing_duration_seconds.labels(
                job_type=job_type,
                component=component
            ).observe(duration)
            
            self.logger.debug(f"Recorded processing duration: {component} - {duration:.2f}s")
    
    def set_queue_size(self, queue_type: str, size: int):
        """Set current queue size."""
        with self._lock:
            self.processing_queue_size.labels(queue_type=queue_type).set(size)
    
    def record_spark_job(self, job_type: str, status: str, duration: Optional[float] = None):
        """Record Spark job metrics."""
        with self._lock:
            self.spark_jobs_total.labels(
                job_type=job_type,
                status=status
            ).inc()
            
            if duration is not None:
                self.spark_job_duration_seconds.labels(job_type=job_type).observe(duration)
            
            self.logger.debug(f"Recorded Spark job: {job_type} - {status}")
    
    def set_spark_executors(self, application_id: str, executor_count: int):
        """Set number of active Spark executors."""
        with self._lock:
            self.spark_active_executors.labels(application_id=application_id).set(executor_count)
    
    def set_spark_memory_usage(self, application_id: str, executor_id: str, memory_bytes: int):
        """Set Spark memory usage."""
        with self._lock:
            self.spark_memory_usage_bytes.labels(
                application_id=application_id,
                executor_id=executor_id
            ).set(memory_bytes)
    
    def record_kafka_message(self, topic: str, message_type: str, status: str = "success"):
        """Record Kafka message metrics."""
        with self._lock:
            if message_type == "produced":
                self.kafka_messages_produced_total.labels(
                    topic=topic,
                    status=status
                ).inc()
            elif message_type == "consumed":
                self.kafka_messages_consumed_total.labels(
                    topic=topic,
                    consumer_group="default"
                ).inc()
            
            self.logger.debug(f"Recorded Kafka message: {topic} - {message_type}")
    
    def set_kafka_consumer_lag(self, topic: str, partition: int, consumer_group: str, lag: int):
        """Set Kafka consumer lag."""
        with self._lock:
            self.kafka_consumer_lag.labels(
                topic=topic,
                partition=str(partition),
                consumer_group=consumer_group
            ).set(lag)
    
    def record_kafka_batch_size(self, topic: str, batch_size: int):
        """Record Kafka producer batch size."""
        with self._lock:
            self.kafka_producer_batch_size.labels(topic=topic).observe(batch_size)
    
    def record_llm_request(self, engine: str, model: str, status: str, response_time: Optional[float] = None):
        """Record LLM request metrics."""
        with self._lock:
            self.llm_requests_total.labels(
                engine=engine,
                model=model,
                status=status
            ).inc()
            
            if response_time is not None:
                self.llm_response_time_seconds.labels(
                    engine=engine,
                    model=model
                ).observe(response_time)
            
            self.logger.debug(f"Recorded LLM request: {engine}/{model} - {status}")
    
    def set_llm_active_instances(self, engine: str, model: str, count: int):
        """Set number of active LLM instances."""
        with self._lock:
            self.llm_active_instances.labels(
                engine=engine,
                model=model
            ).set(count)
    
    def set_llm_pool_utilization(self, pool_name: str, utilization_percent: float):
        """Set LLM pool utilization."""
        with self._lock:
            self.llm_pool_utilization.labels(pool_name=pool_name).set(utilization_percent)
    
    def set_quality_score(self, job_id: str, metric_type: str, score: float):
        """Set quality score for a job."""
        with self._lock:
            self.quality_score.labels(
                job_id=job_id,
                metric_type=metric_type
            ).set(score)
    
    def record_validation_error(self, error_type: str, severity: str):
        """Record validation error."""
        with self._lock:
            self.validation_errors_total.labels(
                error_type=error_type,
                severity=severity
            ).inc()
    
    def record_airflow_dag_run(self, dag_id: str, status: str):
        """Record Airflow DAG run."""
        with self._lock:
            self.airflow_dag_runs_total.labels(
                dag_id=dag_id,
                status=status
            ).inc()
    
    def record_airflow_task_duration(self, dag_id: str, task_id: str, duration: float):
        """Record Airflow task duration."""
        with self._lock:
            self.airflow_task_duration_seconds.labels(
                dag_id=dag_id,
                task_id=task_id
            ).observe(duration)
    
    def set_airflow_active_tasks(self, dag_id: str, count: int):
        """Set number of active Airflow tasks."""
        with self._lock:
            self.airflow_active_tasks.labels(dag_id=dag_id).set(count)
    
    def set_resource_usage(self, component: str, host: str, cpu_percent: float, memory_bytes: int):
        """Set resource usage metrics."""
        with self._lock:
            self.cpu_usage_percent.labels(
                component=component,
                host=host
            ).set(cpu_percent)
            
            self.memory_usage_bytes.labels(
                component=component,
                host=host
            ).set(memory_bytes)
    
    def set_disk_usage(self, component: str, host: str, mount_point: str, usage_bytes: int):
        """Set disk usage metrics."""
        with self._lock:
            self.disk_usage_bytes.labels(
                component=component,
                host=host,
                mount_point=mount_point
            ).set(usage_bytes)
    
    def record_segments_processed(self, job_id: str, segment_type: str, count: int = 1):
        """Record processed segments."""
        with self._lock:
            self.segments_processed_total.labels(
                job_id=job_id,
                segment_type=segment_type
            ).inc(count)
    
    def record_speakers_detected(self, job_id: str, count: int = 1):
        """Record speakers detected."""
        with self._lock:
            self.speakers_detected_total.labels(job_id=job_id).inc(count)
    
    def record_text_extraction(self, file_format: str, byte_count: int):
        """Record text extraction metrics."""
        with self._lock:
            self.text_extraction_bytes.labels(file_format=file_format).inc(byte_count)
    
    def set_system_health(self, component: str, healthy: bool):
        """Set system health status."""
        with self._lock:
            self.system_health.labels(component=component).set(1 if healthy else 0)
    
    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format."""
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus client not available\n"
        
        return generate_latest(self.registry)
    
    def push_to_gateway(self, job_name: str, grouping_key: Optional[Dict[str, str]] = None):
        """Push metrics to push gateway."""
        if not PROMETHEUS_AVAILABLE or not self.enable_push_gateway:
            return
        
        try:
            push_to_gateway(
                self.push_gateway_url,
                job=job_name,
                registry=self.registry,
                grouping_key=grouping_key or {}
            )
            self.logger.debug(f"Pushed metrics to gateway: {job_name}")
        except Exception as e:
            self.logger.error(f"Failed to push metrics to gateway: {e}")
    
    def delete_from_gateway(self, job_name: str, grouping_key: Optional[Dict[str, str]] = None):
        """Delete metrics from push gateway."""
        if not PROMETHEUS_AVAILABLE or not self.enable_push_gateway:
            return
        
        try:
            delete_from_gateway(
                self.push_gateway_url,
                job=job_name,
                grouping_key=grouping_key or {}
            )
            self.logger.debug(f"Deleted metrics from gateway: {job_name}")
        except Exception as e:
            self.logger.error(f"Failed to delete metrics from gateway: {e}")
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        return {
            'namespace': self.namespace,
            'prometheus_available': PROMETHEUS_AVAILABLE,
            'push_gateway_enabled': self.enable_push_gateway,
            'push_gateway_url': self.push_gateway_url,
            'registry_collectors': len(self.registry._collector_to_names) if self.registry else 0,
            'metric_history_entries': sum(len(history) for history in self._metric_history.values())
        }
    
    def create_timer(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """Create a timer context manager for measuring duration."""
        return MetricTimer(self, metric_name, labels or {})
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass


class MetricTimer:
    """Context manager for timing operations."""
    
    def __init__(self, collector: PrometheusMetricsCollector, metric_name: str, labels: Dict[str, str]):
        self.collector = collector
        self.metric_name = metric_name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            
            # Record duration based on metric type
            if 'processing' in self.metric_name:
                self.collector.record_processing_duration(
                    self.labels.get('job_type', 'unknown'),
                    self.labels.get('component', 'unknown'),
                    duration
                )
            elif 'spark' in self.metric_name:
                self.collector.record_spark_job(
                    self.labels.get('job_type', 'unknown'),
                    'completed',
                    duration
                )
            elif 'llm' in self.metric_name:
                self.collector.record_llm_request(
                    self.labels.get('engine', 'unknown'),
                    self.labels.get('model', 'unknown'),
                    'completed',
                    duration
                )


# Global metrics collector instance
_global_metrics_collector = None


def get_metrics_collector() -> PrometheusMetricsCollector:
    """Get the global metrics collector instance."""
    global _global_metrics_collector
    
    if _global_metrics_collector is None:
        _global_metrics_collector = PrometheusMetricsCollector()
    
    return _global_metrics_collector


def initialize_metrics_collector(
    namespace: str = "text_to_audiobook",
    enable_push_gateway: bool = False,
    push_gateway_url: str = "localhost:9091"
) -> PrometheusMetricsCollector:
    """Initialize the global metrics collector."""
    global _global_metrics_collector
    
    _global_metrics_collector = PrometheusMetricsCollector(
        namespace=namespace,
        enable_push_gateway=enable_push_gateway,
        push_gateway_url=push_gateway_url
    )
    
    return _global_metrics_collector


# Convenience functions for common metrics
def record_processing_request(job_type: str, status: str):
    """Record a processing request."""
    get_metrics_collector().record_processing_request(job_type, status)


def record_processing_duration(job_type: str, component: str, duration: float):
    """Record processing duration."""
    get_metrics_collector().record_processing_duration(job_type, component, duration)


def record_llm_request(engine: str, model: str, status: str, response_time: Optional[float] = None):
    """Record LLM request."""
    get_metrics_collector().record_llm_request(engine, model, status, response_time)


def set_system_health(component: str, healthy: bool):
    """Set system health status."""
    get_metrics_collector().set_system_health(component, healthy)


def create_timer(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Create a timer context manager."""
    return get_metrics_collector().create_timer(metric_name, labels or {})


# Decorator for timing functions
def timed_metric(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to time function execution."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with create_timer(metric_name, labels or {}):
                return func(*args, **kwargs)
        return wrapper
    return decorator