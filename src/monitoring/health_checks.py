"""
Health check endpoints and monitoring integration.

This module provides HTTP endpoints for health checks and integrates monitoring
with all components of the distributed text-to-audiobook system.
"""

import json
import time
import logging
import threading
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, jsonify, request
from werkzeug.serving import make_server
import psutil
import os

from .prometheus_metrics import get_metrics_collector, PrometheusMetricsCollector


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    response_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'component': self.component,
            'status': self.status,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp,
            'response_time': self.response_time
        }


class HealthChecker:
    """Base class for health checking components."""
    
    def __init__(self, name: str, timeout: float = 30.0):
        self.name = name
        self.timeout = timeout
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def check_health(self) -> HealthCheckResult:
        """Perform health check. Override in subclasses."""
        start_time = time.time()
        
        try:
            # Default implementation - just return healthy
            response_time = time.time() - start_time
            return HealthCheckResult(
                component=self.name,
                status="healthy",
                message="Component is operational",
                response_time=response_time
            )
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheckResult(
                component=self.name,
                status="unhealthy",
                message=str(e),
                response_time=response_time
            )


class SparkHealthChecker(HealthChecker):
    """Health checker for Spark components."""
    
    def __init__(self, timeout: float = 30.0):
        super().__init__("spark", timeout)
    
    def check_health(self) -> HealthCheckResult:
        """Check Spark health."""
        start_time = time.time()
        
        try:
            # Try to import and check Spark
            try:
                from pyspark.sql import SparkSession
                spark = SparkSession.getActiveSession()
                
                if spark is None:
                    return HealthCheckResult(
                        component=self.name,
                        status="degraded",
                        message="No active Spark session",
                        response_time=time.time() - start_time
                    )
                
                # Check if Spark context is active
                if spark.sparkContext._jsc.sc().isStopped():
                    return HealthCheckResult(
                        component=self.name,
                        status="unhealthy",
                        message="Spark context is stopped",
                        response_time=time.time() - start_time
                    )
                
                # Get basic metrics
                details = {
                    'application_id': spark.sparkContext.applicationId,
                    'application_name': spark.sparkContext.appName,
                    'master': spark.sparkContext.master,
                    'version': spark.version
                }
                
                return HealthCheckResult(
                    component=self.name,
                    status="healthy",
                    message="Spark is operational",
                    details=details,
                    response_time=time.time() - start_time
                )
                
            except ImportError:
                return HealthCheckResult(
                    component=self.name,
                    status="unhealthy",
                    message="Spark not available",
                    response_time=time.time() - start_time
                )
                
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                status="unhealthy",
                message=f"Spark health check failed: {str(e)}",
                response_time=time.time() - start_time
            )


class KafkaHealthChecker(HealthChecker):
    """Health checker for Kafka components."""
    
    def __init__(self, timeout: float = 30.0):
        super().__init__("kafka", timeout)
    
    def check_health(self) -> HealthCheckResult:
        """Check Kafka health."""
        start_time = time.time()
        
        try:
            # Try to create a simple producer and consumer
            from kafka import KafkaProducer, KafkaConsumer
            from kafka.errors import KafkaError
            
            try:
                # Quick producer test
                producer = KafkaProducer(
                    bootstrap_servers=['localhost:9092'],
                    value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                    request_timeout_ms=5000
                )
                
                # Quick consumer test
                consumer = KafkaConsumer(
                    bootstrap_servers=['localhost:9092'],
                    consumer_timeout_ms=5000,
                    auto_offset_reset='earliest'
                )
                
                # Get cluster metadata
                cluster_metadata = producer.cluster
                
                details = {
                    'brokers': len(cluster_metadata.brokers),
                    'topics': len(cluster_metadata.topics),
                    'bootstrap_servers': 'localhost:9092'
                }
                
                producer.close()
                consumer.close()
                
                return HealthCheckResult(
                    component=self.name,
                    status="healthy",
                    message="Kafka is operational",
                    details=details,
                    response_time=time.time() - start_time
                )
                
            except KafkaError as e:
                return HealthCheckResult(
                    component=self.name,
                    status="unhealthy",
                    message=f"Kafka error: {str(e)}",
                    response_time=time.time() - start_time
                )
                
        except ImportError:
            return HealthCheckResult(
                component=self.name,
                status="unhealthy",
                message="Kafka client not available",
                response_time=time.time() - start_time
            )
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                status="unhealthy",
                message=f"Kafka health check failed: {str(e)}",
                response_time=time.time() - start_time
            )


class LLMHealthChecker(HealthChecker):
    """Health checker for LLM components."""
    
    def __init__(self, timeout: float = 30.0):
        super().__init__("llm", timeout)
    
    def check_health(self) -> HealthCheckResult:
        """Check LLM health."""
        start_time = time.time()
        
        try:
            # Check LLM pool manager
            from ..llm_pool.llm_pool_manager import get_pool_manager
            
            pool_manager = get_pool_manager()
            pool_status = pool_manager.get_pool_status()
            
            if pool_status.get('healthy_instances', 0) == 0:
                return HealthCheckResult(
                    component=self.name,
                    status="unhealthy",
                    message="No healthy LLM instances available",
                    details=pool_status,
                    response_time=time.time() - start_time
                )
            
            # Check if utilization is too high
            utilization = pool_status.get('utilization_percent', 0)
            if utilization > 95:
                status = "degraded"
                message = f"LLM pool utilization is high ({utilization}%)"
            else:
                status = "healthy"
                message = "LLM pool is operational"
            
            return HealthCheckResult(
                component=self.name,
                status=status,
                message=message,
                details=pool_status,
                response_time=time.time() - start_time
            )
            
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                status="unhealthy",
                message=f"LLM health check failed: {str(e)}",
                response_time=time.time() - start_time
            )


class DatabaseHealthChecker(HealthChecker):
    """Health checker for database components."""
    
    def __init__(self, timeout: float = 30.0):
        super().__init__("database", timeout)
    
    def check_health(self) -> HealthCheckResult:
        """Check database health."""
        start_time = time.time()
        
        try:
            # Check if we can connect to the database
            # This is a placeholder - actual implementation would depend on DB type
            
            # For now, assume database is healthy if we can import required modules
            try:
                import sqlite3
                # Quick connection test
                conn = sqlite3.connect(':memory:')
                conn.execute('SELECT 1')
                conn.close()
                
                return HealthCheckResult(
                    component=self.name,
                    status="healthy",
                    message="Database is operational",
                    details={'type': 'sqlite'},
                    response_time=time.time() - start_time
                )
                
            except Exception as e:
                return HealthCheckResult(
                    component=self.name,
                    status="unhealthy",
                    message=f"Database connection failed: {str(e)}",
                    response_time=time.time() - start_time
                )
                
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                status="unhealthy",
                message=f"Database health check failed: {str(e)}",
                response_time=time.time() - start_time
            )


class SystemResourcesHealthChecker(HealthChecker):
    """Health checker for system resources."""
    
    def __init__(self, timeout: float = 30.0):
        super().__init__("system_resources", timeout)
    
    def check_health(self) -> HealthCheckResult:
        """Check system resources health."""
        start_time = time.time()
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Check thresholds
            status = "healthy"
            issues = []
            
            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent}%")
                status = "degraded"
            
            if memory.percent > 90:
                issues.append(f"High memory usage: {memory.percent}%")
                status = "degraded"
            
            if disk.percent > 90:
                issues.append(f"High disk usage: {disk.percent}%")
                status = "degraded"
            
            if cpu_percent > 95 or memory.percent > 95 or disk.percent > 95:
                status = "unhealthy"
            
            details = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available': memory.available,
                'disk_percent': disk.percent,
                'disk_free': disk.free,
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
            
            message = "System resources are healthy"
            if issues:
                message = f"System resource issues: {', '.join(issues)}"
            
            return HealthCheckResult(
                component=self.name,
                status=status,
                message=message,
                details=details,
                response_time=time.time() - start_time
            )
            
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                status="unhealthy",
                message=f"System resources health check failed: {str(e)}",
                response_time=time.time() - start_time
            )


class HealthCheckService:
    """Service for managing health checks across all components."""
    
    def __init__(self, metrics_collector: Optional[PrometheusMetricsCollector] = None):
        self.metrics_collector = metrics_collector or get_metrics_collector()
        self.logger = logging.getLogger(__name__)
        self.health_checkers: Dict[str, HealthChecker] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
        self.check_interval = 30.0  # seconds
        self.running = False
        self.check_thread = None
        
        # Initialize health checkers
        self._initialize_health_checkers()
    
    def _initialize_health_checkers(self):
        """Initialize all health checkers."""
        self.health_checkers = {
            'spark': SparkHealthChecker(),
            'kafka': KafkaHealthChecker(),
            'llm': LLMHealthChecker(),
            'database': DatabaseHealthChecker(),
            'system_resources': SystemResourcesHealthChecker()
        }
    
    def add_health_checker(self, name: str, checker: HealthChecker):
        """Add a custom health checker."""
        self.health_checkers[name] = checker
    
    def check_all_components(self) -> Dict[str, HealthCheckResult]:
        """Check health of all components."""
        results = {}
        
        # Use thread pool for parallel health checks
        with ThreadPoolExecutor(max_workers=len(self.health_checkers)) as executor:
            future_to_name = {
                executor.submit(checker.check_health): name
                for name, checker in self.health_checkers.items()
            }
            
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result()
                    results[name] = result
                    
                    # Update metrics
                    self.metrics_collector.set_system_health(
                        name, result.status == "healthy"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Health check failed for {name}: {e}")
                    results[name] = HealthCheckResult(
                        component=name,
                        status="unhealthy",
                        message=f"Health check exception: {str(e)}"
                    )
        
        # Store results
        self.last_results = results
        return results
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        results = self.last_results
        
        if not results:
            return {
                'status': 'unknown',
                'message': 'No health check results available',
                'timestamp': time.time()
            }
        
        # Determine overall status
        healthy_count = sum(1 for r in results.values() if r.status == "healthy")
        degraded_count = sum(1 for r in results.values() if r.status == "degraded")
        unhealthy_count = sum(1 for r in results.values() if r.status == "unhealthy")
        
        total_count = len(results)
        
        if unhealthy_count > 0:
            overall_status = "unhealthy"
            message = f"{unhealthy_count} components are unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
            message = f"{degraded_count} components are degraded"
        else:
            overall_status = "healthy"
            message = "All components are healthy"
        
        return {
            'status': overall_status,
            'message': message,
            'summary': {
                'healthy': healthy_count,
                'degraded': degraded_count,
                'unhealthy': unhealthy_count,
                'total': total_count
            },
            'components': {name: result.to_dict() for name, result in results.items()},
            'timestamp': time.time()
        }
    
    def start_continuous_monitoring(self):
        """Start continuous health monitoring."""
        if self.running:
            return
        
        self.running = True
        self.check_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.check_thread.start()
        
        self.logger.info("Started continuous health monitoring")
    
    def stop_continuous_monitoring(self):
        """Stop continuous health monitoring."""
        self.running = False
        if self.check_thread:
            self.check_thread.join(timeout=10.0)
        
        self.logger.info("Stopped continuous health monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self.check_all_components()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Brief pause before retrying


# Flask app for health check endpoints
def create_health_check_app(health_service: HealthCheckService) -> Flask:
    """Create Flask app with health check endpoints."""
    app = Flask(__name__)
    
    @app.route('/health')
    def health_check():
        """Main health check endpoint."""
        return jsonify(health_service.get_overall_health())
    
    @app.route('/health/detailed')
    def detailed_health_check():
        """Detailed health check endpoint."""
        results = health_service.check_all_components()
        return jsonify({
            'timestamp': time.time(),
            'components': {name: result.to_dict() for name, result in results.items()}
        })
    
    @app.route('/health/components/<component>')
    def component_health_check(component):
        """Health check for specific component."""
        if component not in health_service.health_checkers:
            return jsonify({'error': f'Component {component} not found'}), 404
        
        checker = health_service.health_checkers[component]
        result = checker.check_health()
        return jsonify(result.to_dict())
    
    @app.route('/health/metrics')
    def health_metrics():
        """Health metrics endpoint."""
        return jsonify({
            'check_interval': health_service.check_interval,
            'checkers': list(health_service.health_checkers.keys()),
            'last_check': max(
                (r.timestamp for r in health_service.last_results.values()),
                default=0
            ),
            'monitoring_active': health_service.running
        })
    
    @app.route('/metrics')
    def prometheus_metrics():
        """Prometheus metrics endpoint."""
        return health_service.metrics_collector.get_metrics(), 200, {
            'Content-Type': 'text/plain; version=0.0.4; charset=utf-8'
        }
    
    return app


class HealthCheckServer:
    """HTTP server for health check endpoints."""
    
    def __init__(self, health_service: HealthCheckService, 
                 host: str = '0.0.0.0', port: int = 8080):
        self.health_service = health_service
        self.host = host
        self.port = port
        self.app = create_health_check_app(health_service)
        self.server = None
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start the health check server."""
        self.server = make_server(self.host, self.port, self.app, threaded=True)
        
        # Start in separate thread
        server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        server_thread.start()
        
        self.logger.info(f"Health check server started on {self.host}:{self.port}")
    
    def stop(self):
        """Stop the health check server."""
        if self.server:
            self.server.shutdown()
            self.logger.info("Health check server stopped")


# Global health check service instance
_global_health_service = None


def get_health_service() -> HealthCheckService:
    """Get the global health check service instance."""
    global _global_health_service
    
    if _global_health_service is None:
        _global_health_service = HealthCheckService()
    
    return _global_health_service


def initialize_health_monitoring(start_server: bool = True, 
                                start_continuous: bool = True,
                                host: str = '0.0.0.0', 
                                port: int = 8080) -> HealthCheckService:
    """Initialize health monitoring system."""
    global _global_health_service
    
    _global_health_service = HealthCheckService()
    
    if start_continuous:
        _global_health_service.start_continuous_monitoring()
    
    if start_server:
        server = HealthCheckServer(_global_health_service, host, port)
        server.start()
    
    return _global_health_service


if __name__ == "__main__":
    # Example usage
    health_service = initialize_health_monitoring(
        start_server=True,
        start_continuous=True,
        host='0.0.0.0',
        port=8080
    )
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        health_service.stop_continuous_monitoring()
        print("Health monitoring stopped")