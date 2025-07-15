"""
LLM Pool Manager for distributed text processing.

This module provides a comprehensive LLM pool management system that handles
multiple local LLM instances across distributed workers, with load balancing,
health monitoring, and fault tolerance.
"""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import psutil
import random
from queue import Queue, Empty
import json

from config import settings


class LLMInstanceStatus(Enum):
    """Status of an LLM instance."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    BUSY = "busy"
    OFFLINE = "offline"


@dataclass
class LLMInstance:
    """Represents an LLM instance."""
    instance_id: str
    host: str
    port: int
    model_name: str
    status: LLMInstanceStatus
    current_load: int
    max_load: int
    last_health_check: float
    response_time_avg: float
    total_requests: int
    failed_requests: int
    created_at: float
    
    @property
    def url(self) -> str:
        """Get the full URL for the LLM instance."""
        return f"http://{self.host}:{self.port}"
    
    @property
    def health_score(self) -> float:
        """Calculate health score based on performance metrics."""
        if self.total_requests == 0:
            return 1.0
        
        # Factor in success rate, response time, and current load
        success_rate = 1.0 - (self.failed_requests / self.total_requests)
        load_factor = 1.0 - (self.current_load / self.max_load)
        response_factor = min(1.0, 1.0 / max(0.1, self.response_time_avg))
        
        return (success_rate * 0.4) + (load_factor * 0.3) + (response_factor * 0.3)
    
    def is_available(self) -> bool:
        """Check if instance is available for new requests."""
        return (self.status == LLMInstanceStatus.HEALTHY and 
                self.current_load < self.max_load)


@dataclass
class LLMRequest:
    """Represents an LLM request."""
    request_id: str
    prompt: str
    model_config: Dict[str, Any]
    priority: int
    timeout: float
    retry_count: int
    max_retries: int
    created_at: float
    
    def should_retry(self) -> bool:
        """Check if request should be retried."""
        return self.retry_count < self.max_retries


@dataclass
class LLMResponse:
    """Represents an LLM response."""
    request_id: str
    response_text: str
    response_time: float
    instance_id: str
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class LLMPoolManager:
    """Manages a pool of LLM instances for distributed processing."""
    
    def __init__(self, pool_config: Dict[str, Any] = None):
        """Initialize LLM pool manager."""
        self.config = pool_config or self._get_default_config()
        self.instances: Dict[str, LLMInstance] = {}
        self.request_queue = Queue()
        self.response_handlers: Dict[str, asyncio.Future] = {}
        self.running = False
        self.workers = []
        self.health_monitor_thread = None
        self.load_balancer = LLMLoadBalancer(self)
        self.health_monitor = LLMHealthMonitor(self)
        self.metrics_collector = LLMMetricsCollector(self)
        self.logger = logging.getLogger(__name__)
        
        # Thread pool for handling requests
        self.executor = ThreadPoolExecutor(
            max_workers=self.config['max_concurrent_requests']
        )
        
        # Initialize instances
        self._initialize_instances()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'default_instances': [
                {'host': 'localhost', 'port': 11434, 'model': 'deepseek-v2:16b'},
                {'host': 'localhost', 'port': 11435, 'model': 'deepseek-v2:16b'},
                {'host': 'localhost', 'port': 11436, 'model': 'deepseek-v2:16b'},
                {'host': 'localhost', 'port': 11437, 'model': 'deepseek-v2:16b'},
            ],
            'max_concurrent_requests': 10,
            'request_timeout': 30.0,
            'health_check_interval': 30.0,
            'max_retries': 3,
            'retry_delay': 1.0,
            'instance_max_load': 5,
            'auto_scale': True,
            'metrics_enabled': True
        }
    
    def _initialize_instances(self):
        """Initialize LLM instances."""
        for i, instance_config in enumerate(self.config['default_instances']):
            instance_id = f"llm-{i}"
            instance = LLMInstance(
                instance_id=instance_id,
                host=instance_config['host'],
                port=instance_config['port'],
                model_name=instance_config['model'],
                status=LLMInstanceStatus.OFFLINE,
                current_load=0,
                max_load=self.config['instance_max_load'],
                last_health_check=0.0,
                response_time_avg=0.0,
                total_requests=0,
                failed_requests=0,
                created_at=time.time()
            )
            self.instances[instance_id] = instance
    
    def start(self):
        """Start the LLM pool manager."""
        if self.running:
            return
        
        self.running = True
        self.logger.info(f"Starting LLM pool manager with {len(self.instances)} instances")
        
        # Start health monitoring
        self.health_monitor_thread = threading.Thread(
            target=self.health_monitor.run, daemon=True
        )
        self.health_monitor_thread.start()
        
        # Start request processing workers
        for i in range(self.config['max_concurrent_requests']):
            worker = threading.Thread(
                target=self._process_requests, daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        # Initial health check
        self.health_monitor.check_all_instances()
        
        self.logger.info("LLM pool manager started successfully")
    
    def stop(self):
        """Stop the LLM pool manager."""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Stopping LLM pool manager")
        
        # Stop executor
        self.executor.shutdown(wait=True)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        # Stop health monitor
        if self.health_monitor_thread:
            self.health_monitor_thread.join(timeout=5.0)
        
        self.logger.info("LLM pool manager stopped")
    
    def submit_request(self, prompt: str, model_config: Dict[str, Any] = None,
                      priority: int = 0, timeout: float = None) -> str:
        """Submit a request to the LLM pool."""
        request_id = f"req-{int(time.time() * 1000000)}"
        
        request = LLMRequest(
            request_id=request_id,
            prompt=prompt,
            model_config=model_config or {},
            priority=priority,
            timeout=timeout or self.config['request_timeout'],
            retry_count=0,
            max_retries=self.config['max_retries'],
            created_at=time.time()
        )
        
        # Add to queue
        self.request_queue.put(request)
        
        # Create future for response
        future = asyncio.Future()
        self.response_handlers[request_id] = future
        
        return request_id
    
    def get_response(self, request_id: str, timeout: float = None) -> LLMResponse:
        """Get response for a submitted request."""
        if request_id not in self.response_handlers:
            raise ValueError(f"Unknown request ID: {request_id}")
        
        future = self.response_handlers[request_id]
        
        try:
            # Wait for response
            response = future.result(timeout=timeout)
            return response
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request {request_id} timed out")
        finally:
            # Clean up
            self.response_handlers.pop(request_id, None)
    
    def _process_requests(self):
        """Process requests from the queue."""
        while self.running:
            try:
                request = self.request_queue.get(timeout=1.0)
                
                # Process the request
                response = self._handle_request(request)
                
                # Send response back
                if request.request_id in self.response_handlers:
                    future = self.response_handlers[request.request_id]
                    if not future.done():
                        future.set_result(response)
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing request: {e}")
    
    def _handle_request(self, request: LLMRequest) -> LLMResponse:
        """Handle a single request."""
        start_time = time.time()
        
        # Get best instance
        instance = self.load_balancer.get_best_instance()
        if not instance:
            return LLMResponse(
                request_id=request.request_id,
                response_text="",
                response_time=time.time() - start_time,
                instance_id="none",
                success=False,
                error_message="No available instances"
            )
        
        # Increment load
        instance.current_load += 1
        
        try:
            # Make request to LLM instance
            response_text = self._make_llm_request(instance, request)
            
            # Update metrics
            response_time = time.time() - start_time
            instance.total_requests += 1
            instance.response_time_avg = (
                (instance.response_time_avg * (instance.total_requests - 1) + response_time) /
                instance.total_requests
            )
            
            return LLMResponse(
                request_id=request.request_id,
                response_text=response_text,
                response_time=response_time,
                instance_id=instance.instance_id,
                success=True
            )
            
        except Exception as e:
            # Handle error
            instance.failed_requests += 1
            
            # Retry if possible
            if request.should_retry():
                request.retry_count += 1
                time.sleep(self.config['retry_delay'])
                return self._handle_request(request)
            
            return LLMResponse(
                request_id=request.request_id,
                response_text="",
                response_time=time.time() - start_time,
                instance_id=instance.instance_id,
                success=False,
                error_message=str(e)
            )
        finally:
            # Decrement load
            instance.current_load -= 1
    
    def _make_llm_request(self, instance: LLMInstance, request: LLMRequest) -> str:
        """Make actual HTTP request to LLM instance."""
        url = f"{instance.url}/api/generate"
        
        payload = {
            "model": instance.model_name,
            "prompt": request.prompt,
            "stream": False,
            **request.model_config
        }
        
        # Create session with retry strategy
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504]
        )
        session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
        session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
        
        try:
            response = session.post(
                url,
                json=payload,
                timeout=request.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"HTTP request failed: {e}")
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get current pool status."""
        healthy_instances = sum(1 for i in self.instances.values() 
                               if i.status == LLMInstanceStatus.HEALTHY)
        total_load = sum(i.current_load for i in self.instances.values())
        
        return {
            'total_instances': len(self.instances),
            'healthy_instances': healthy_instances,
            'total_load': total_load,
            'queue_size': self.request_queue.qsize(),
            'running': self.running,
            'instances': [
                {
                    'instance_id': i.instance_id,
                    'host': i.host,
                    'port': i.port,
                    'status': i.status.value,
                    'current_load': i.current_load,
                    'max_load': i.max_load,
                    'health_score': i.health_score,
                    'response_time_avg': i.response_time_avg,
                    'total_requests': i.total_requests,
                    'failed_requests': i.failed_requests
                }
                for i in self.instances.values()
            ]
        }
    
    def add_instance(self, host: str, port: int, model_name: str) -> str:
        """Add a new LLM instance to the pool."""
        instance_id = f"llm-{len(self.instances)}"
        
        instance = LLMInstance(
            instance_id=instance_id,
            host=host,
            port=port,
            model_name=model_name,
            status=LLMInstanceStatus.OFFLINE,
            current_load=0,
            max_load=self.config['instance_max_load'],
            last_health_check=0.0,
            response_time_avg=0.0,
            total_requests=0,
            failed_requests=0,
            created_at=time.time()
        )
        
        self.instances[instance_id] = instance
        
        # Health check for new instance
        self.health_monitor.check_instance(instance)
        
        self.logger.info(f"Added new LLM instance: {instance_id}")
        return instance_id
    
    def remove_instance(self, instance_id: str) -> bool:
        """Remove an LLM instance from the pool."""
        if instance_id not in self.instances:
            return False
        
        instance = self.instances[instance_id]
        
        # Wait for current requests to finish
        while instance.current_load > 0:
            time.sleep(0.1)
        
        del self.instances[instance_id]
        
        self.logger.info(f"Removed LLM instance: {instance_id}")
        return True


class LLMLoadBalancer:
    """Load balancer for LLM instances."""
    
    def __init__(self, pool_manager: LLMPoolManager):
        self.pool_manager = pool_manager
    
    def get_best_instance(self) -> Optional[LLMInstance]:
        """Get the best available instance for a request."""
        available_instances = [
            instance for instance in self.pool_manager.instances.values()
            if instance.is_available()
        ]
        
        if not available_instances:
            return None
        
        # Sort by health score (descending) and current load (ascending)
        available_instances.sort(
            key=lambda i: (-i.health_score, i.current_load)
        )
        
        return available_instances[0]


class LLMHealthMonitor:
    """Health monitor for LLM instances."""
    
    def __init__(self, pool_manager: LLMPoolManager):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        """Run health monitoring loop."""
        while self.pool_manager.running:
            self.check_all_instances()
            time.sleep(self.pool_manager.config['health_check_interval'])
    
    def check_all_instances(self):
        """Check health of all instances."""
        for instance in self.pool_manager.instances.values():
            self.check_instance(instance)
    
    def check_instance(self, instance: LLMInstance):
        """Check health of a single instance."""
        try:
            # Make a simple health check request
            url = f"{instance.url}/api/version"
            response = requests.get(url, timeout=5.0)
            
            if response.status_code == 200:
                instance.status = LLMInstanceStatus.HEALTHY
                instance.last_health_check = time.time()
            else:
                instance.status = LLMInstanceStatus.UNHEALTHY
                
        except Exception as e:
            instance.status = LLMInstanceStatus.OFFLINE
            self.logger.warning(f"Health check failed for {instance.instance_id}: {e}")
        
        instance.last_health_check = time.time()


class LLMMetricsCollector:
    """Metrics collector for LLM pool."""
    
    def __init__(self, pool_manager: LLMPoolManager):
        self.pool_manager = pool_manager
        self.metrics_history = []
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current metrics."""
        pool_status = self.pool_manager.get_pool_status()
        
        metrics = {
            'timestamp': time.time(),
            'pool_status': pool_status,
            'system_metrics': {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            }
        }
        
        self.metrics_history.append(metrics)
        
        # Keep only last 1000 metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        return metrics
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 metrics
        
        return {
            'average_cpu': sum(m['system_metrics']['cpu_usage'] for m in recent_metrics) / len(recent_metrics),
            'average_memory': sum(m['system_metrics']['memory_usage'] for m in recent_metrics) / len(recent_metrics),
            'total_requests': sum(i.total_requests for i in self.pool_manager.instances.values()),
            'total_failures': sum(i.failed_requests for i in self.pool_manager.instances.values()),
            'healthy_instances': sum(1 for i in self.pool_manager.instances.values() 
                                   if i.status == LLMInstanceStatus.HEALTHY)
        }


# Singleton instance for global access
_pool_manager_instance = None


def get_pool_manager(config: Dict[str, Any] = None) -> LLMPoolManager:
    """Get the global LLM pool manager instance."""
    global _pool_manager_instance
    
    if _pool_manager_instance is None:
        _pool_manager_instance = LLMPoolManager(config)
    
    return _pool_manager_instance


def initialize_pool_manager(config: Dict[str, Any] = None) -> LLMPoolManager:
    """Initialize and start the LLM pool manager."""
    pool_manager = get_pool_manager(config)
    pool_manager.start()
    return pool_manager


def shutdown_pool_manager():
    """Shutdown the LLM pool manager."""
    global _pool_manager_instance
    
    if _pool_manager_instance:
        _pool_manager_instance.stop()
        _pool_manager_instance = None