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
from .http_pool_manager import HTTPConnectionPoolManager, ConnectionPoolConfig
from .intelligent_router import IntelligentRequestRouter, RoutingDecision


class LLMInstanceStatus(Enum):
    """Status of an LLM instance."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    BUSY = "busy"
    OFFLINE = "offline"


@dataclass
class LLMInstance:
    """Represents an LLM instance with model-aware capabilities."""
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
    # Phase 3.2.2: Model-specific capabilities and metrics
    model_capabilities: Optional[Dict[str, Any]] = None
    engine_type: str = "local"  # "local" or "gcp"
    provider: str = "ollama"   # "ollama" or "google_cloud"
    cost_per_request: float = 0.0
    quality_score: float = 0.0
    speed_score: float = 0.0
    reliability_score: float = 0.95
    optimal_use_cases: List[str] = None
    model_specific_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        """Initialize model capabilities and metrics after creation."""
        if self.optimal_use_cases is None:
            self.optimal_use_cases = []
        if self.model_specific_metrics is None:
            self.model_specific_metrics = {
                "avg_tokens_per_sec": 0.0,
                "avg_input_tokens": 0.0,
                "avg_output_tokens": 0.0,
                "cost_efficiency_score": 0.0,
                "context_utilization": 0.0
            }
        
        # Load model capabilities from settings if available
        if self.model_capabilities is None:
            self._load_model_capabilities()
    
    def _load_model_capabilities(self):
        """Load model capabilities from settings configuration."""
        if hasattr(settings, 'MODEL_CAPABILITIES') and self.model_name in settings.MODEL_CAPABILITIES:
            capabilities = settings.MODEL_CAPABILITIES[self.model_name]
            self.model_capabilities = capabilities.copy()
            
            # Set derived properties from capabilities
            self.engine_type = capabilities.get("engine", "local")
            self.provider = capabilities.get("provider", "ollama")
            self.cost_per_request = capabilities.get("cost_per_1k_tokens", 0.0)
            self.reliability_score = capabilities.get("reliability_score", 0.95)
            self.optimal_use_cases = capabilities.get("optimal_use_cases", [])
            
            # Convert capability tiers to numeric scores (0.0-1.0)
            speed_tier = capabilities.get("speed_tier", "medium")
            quality_tier = capabilities.get("quality_tier", "medium") 
            
            speed_mapping = {"slow": 0.3, "medium": 0.6, "fast": 0.9}
            quality_mapping = {"basic": 0.4, "medium": 0.7, "high": 1.0}
            
            self.speed_score = speed_mapping.get(speed_tier, 0.6)
            self.quality_score = quality_mapping.get(quality_tier, 0.7)
        else:
            # Fallback to default capabilities
            self.model_capabilities = self._get_default_capabilities()
    
    def _get_default_capabilities(self) -> Dict[str, Any]:
        """Get default capabilities for unknown models."""
        return {
            "engine": self.engine_type,
            "provider": self.provider,
            "speed_tier": "medium",
            "quality_tier": "medium", 
            "cost_tier": "free" if self.engine_type == "local" else "medium",
            "max_context_length": 8192,
            "supports_json": True,
            "optimal_use_cases": ["general"],
            "cost_per_1k_tokens": 0.0 if self.engine_type == "local" else 0.002,
            "avg_response_time_ms": 1000,
            "reliability_score": 0.9
        }
    
    @property
    def url(self) -> str:
        """Get the full URL for the LLM instance."""
        if self.engine_type == "gcp":
            return f"gcp://{self.provider}/{self.model_name}"
        return f"http://{self.host}:{self.port}"
    
    @property
    def health_score(self) -> float:
        """Calculate enhanced health score based on performance metrics and model capabilities."""
        if self.total_requests == 0:
            return self.reliability_score
        
        # Factor in success rate, response time, current load, and model-specific metrics
        success_rate = 1.0 - (self.failed_requests / self.total_requests)
        load_factor = 1.0 - (self.current_load / self.max_load)
        response_factor = min(1.0, 1.0 / max(0.1, self.response_time_avg))
        
        # Include model capability scores
        capability_factor = (self.speed_score + self.quality_score + self.reliability_score) / 3.0
        
        # Enhanced scoring with model awareness
        return (success_rate * 0.3) + (load_factor * 0.25) + (response_factor * 0.25) + (capability_factor * 0.2)
    
    @property
    def suitability_score(self) -> float:
        """Calculate overall suitability score for request routing."""
        # Combine health score with model-specific metrics
        health = self.health_score
        cost_efficiency = self.model_specific_metrics.get("cost_efficiency_score", 0.5)
        
        # Weight health more heavily, but include cost efficiency
        return (health * 0.7) + (cost_efficiency * 0.3)
    
    def is_available(self) -> bool:
        """Check if instance is available for new requests."""
        return (self.status == LLMInstanceStatus.HEALTHY and 
                self.current_load < self.max_load)
    
    def is_suitable_for_use_case(self, use_case: str) -> bool:
        """Check if this model is suitable for a specific use case."""
        if not self.optimal_use_cases:
            return True  # Default to suitable if no specific use cases defined
        
        # Check for exact match or partial match
        for optimal_case in self.optimal_use_cases:
            if use_case == optimal_case or use_case in optimal_case or optimal_case in use_case:
                return True
        
        return False
    
    def get_estimated_cost(self, input_tokens: int, output_tokens: int = None) -> float:
        """Calculate estimated cost for a request with given token counts."""
        if self.engine_type == "local":
            return 0.0
        
        total_tokens = input_tokens + (output_tokens or input_tokens * 0.5)  # Estimate output if not provided
        return (total_tokens / 1000.0) * self.cost_per_request
    
    def update_model_metrics(self, request_tokens: int, response_tokens: int, 
                           response_time: float, context_length: int):
        """Update model-specific performance metrics."""
        # Update token-based metrics
        total_requests = self.total_requests + 1
        
        # Rolling average for tokens per second
        tokens_per_sec = (request_tokens + response_tokens) / max(0.1, response_time)
        self.model_specific_metrics["avg_tokens_per_sec"] = (
            (self.model_specific_metrics["avg_tokens_per_sec"] * self.total_requests + tokens_per_sec) / 
            total_requests
        )
        
        # Rolling average for input/output tokens
        self.model_specific_metrics["avg_input_tokens"] = (
            (self.model_specific_metrics["avg_input_tokens"] * self.total_requests + request_tokens) /
            total_requests
        )
        
        self.model_specific_metrics["avg_output_tokens"] = (
            (self.model_specific_metrics["avg_output_tokens"] * self.total_requests + response_tokens) /
            total_requests
        )
        
        # Context utilization
        max_context = self.model_capabilities.get("max_context_length", 8192) if self.model_capabilities else 8192
        utilization = context_length / max_context
        self.model_specific_metrics["context_utilization"] = (
            (self.model_specific_metrics["context_utilization"] * self.total_requests + utilization) /
            total_requests
        )
        
        # Cost efficiency score (tokens per dollar, higher is better)
        if self.cost_per_request > 0:
            cost_efficiency = tokens_per_sec / self.cost_per_request
            self.model_specific_metrics["cost_efficiency_score"] = (
                (self.model_specific_metrics["cost_efficiency_score"] * self.total_requests + cost_efficiency) /
                total_requests
            )
        else:
            self.model_specific_metrics["cost_efficiency_score"] = 1.0  # Free models get max efficiency
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary including model-specific metrics."""
        return {
            "basic_metrics": {
                "health_score": self.health_score,
                "suitability_score": self.suitability_score,
                "response_time_avg": self.response_time_avg,
                "total_requests": self.total_requests,
                "failed_requests": self.failed_requests,
                "success_rate": 1.0 - (self.failed_requests / max(1, self.total_requests))
            },
            "model_capabilities": {
                "model_name": self.model_name,
                "engine_type": self.engine_type,
                "provider": self.provider,
                "speed_score": self.speed_score,
                "quality_score": self.quality_score,
                "reliability_score": self.reliability_score,
                "optimal_use_cases": self.optimal_use_cases
            },
            "model_specific_metrics": self.model_specific_metrics,
            "cost_metrics": {
                "cost_per_request": self.cost_per_request,
                "estimated_hourly_cost": self.cost_per_request * (3600 / max(1, self.response_time_avg))
            }
        }


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
        self.intelligent_router = IntelligentRequestRouter(self)
        self.logger = logging.getLogger(__name__)
        
        # HTTP connection pool manager for optimized requests
        if settings.HTTP_POOL_ENABLED:
            self.http_pool_config = ConnectionPoolConfig.from_settings()
            self.http_pool_manager = HTTPConnectionPoolManager(self.http_pool_config)
            self.logger.info("HTTP connection pooling enabled for LLM requests")
        else:
            self.http_pool_manager = None
            self.logger.info("HTTP connection pooling disabled")
        
        # Thread pool for handling requests
        self.executor = ThreadPoolExecutor(
            max_workers=self.config['max_concurrent_requests']
        )
        
        # Initialize instances
        self._initialize_instances()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration with multi-model support."""
        # Check if multi-model configuration is enabled
        if getattr(settings, 'MULTI_MODEL_ENABLED', False) and hasattr(settings, 'MULTI_MODEL_POOLS'):
            return self._get_multi_model_config()
        
        # Fallback to legacy single-model configuration
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
            'metrics_enabled': True,
            'pool_name': 'default'
        }
    
    def _get_multi_model_config(self) -> Dict[str, Any]:
        """Generate configuration from multi-model pool settings."""
        # Use the primary pool as default, or the first available pool
        pool_configs = settings.MULTI_MODEL_POOLS
        pool_name = self.config.get('pool_name', 'primary') if self.config else 'primary'
        
        if pool_name not in pool_configs:
            pool_name = list(pool_configs.keys())[0]  # Use first available pool
        
        pool_config = pool_configs[pool_name]
        
        # Generate instances for all models in the selected pool
        instances = []
        port_counter = 11434  # Starting port for local models
        
        for model_name in pool_config['models']:
            model_caps = settings.MODEL_CAPABILITIES.get(model_name, {})
            engine_type = model_caps.get('engine', 'local')
            max_instances = pool_config.get('max_instances_per_model', 2)
            
            if engine_type == 'local':
                # Create multiple instances for local models (different ports)
                for i in range(max_instances):
                    instances.append({
                        'host': 'localhost',
                        'port': port_counter,
                        'model': model_name,
                        'engine_type': engine_type,
                        'provider': model_caps.get('provider', 'ollama')
                    })
                    port_counter += 1
            else:
                # For cloud models, create logical instances (same endpoint, different instance IDs)
                for i in range(max_instances):
                    instances.append({
                        'host': 'cloud',
                        'port': 443,  # HTTPS port for cloud services
                        'model': model_name,
                        'engine_type': engine_type,
                        'provider': model_caps.get('provider', 'google_cloud')
                    })
        
        return {
            'default_instances': instances,
            'max_concurrent_requests': len(instances) * 2,  # 2 requests per instance
            'request_timeout': 60.0,
            'health_check_interval': pool_config.get('health_check_interval', 30.0),
            'max_retries': 3,
            'retry_delay': 1.0,
            'instance_max_load': 5,
            'auto_scale': True,
            'metrics_enabled': True,
            'pool_name': pool_name,
            'pool_config': pool_config
        }
    
    def _initialize_instances(self):
        """Initialize LLM instances with model-aware capabilities."""
        for i, instance_config in enumerate(self.config['default_instances']):
            instance_id = f"llm-{instance_config['model'].replace(':', '-').replace('.', '-')}-{i}"
            
            # Determine engine type and provider
            engine_type = instance_config.get('engine_type', 'local')
            provider = instance_config.get('provider', 'ollama')
            
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
                created_at=time.time(),
                engine_type=engine_type,
                provider=provider
            )
            self.instances[instance_id] = instance
            self.logger.info(f"Initialized {engine_type} instance: {instance_id} ({instance.model_name})")
    
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
        
        # Clean up HTTP connection pool manager
        if self.http_pool_manager:
            self.http_pool_manager.close()
            self.logger.info("HTTP connection pool closed")
        
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
    
    def submit_intelligent_request(self, prompt: str, model_config: Dict[str, Any] = None,
                                 priority: int = 0, timeout: float = None,
                                 routing_strategy: str = None) -> Tuple[str, RoutingDecision]:
        """Submit a request with intelligent routing and return routing decision."""
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
        
        # Get intelligent routing decision
        routing_decision = self.intelligent_router.route_request(request, routing_strategy)
        
        # Add routing decision to request metadata
        request.model_config['_routing_decision'] = routing_decision
        
        # Add to queue
        self.request_queue.put(request)
        
        # Create future for response
        future = asyncio.Future()
        self.response_handlers[request_id] = future
        
        return request_id, routing_decision
    
    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get comprehensive routing analytics from intelligent router."""
        return self.intelligent_router.get_routing_analytics()
    
    def get_routing_recommendations(self, prompt: str, model_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get instance recommendations for a specific request."""
        # Create a temporary request for analysis
        temp_request = LLMRequest(
            request_id="temp",
            prompt=prompt,
            model_config=model_config or {},
            priority=0,
            timeout=30.0,
            retry_count=0,
            max_retries=3,
            created_at=time.time()
        )
        return self.intelligent_router.get_instance_recommendations(temp_request)
    
    def optimize_routing_strategy(self) -> Dict[str, Any]:
        """Get routing optimization analysis and recommendations."""
        return self.intelligent_router.optimize_routing_strategy()
    
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
        """Handle a single request with intelligent routing and model-aware processing."""
        start_time = time.time()
        
        # Check if intelligent routing decision is available
        routing_decision = request.model_config.get('_routing_decision')
        
        if routing_decision and routing_decision.selected_instance.is_available():
            # Use intelligent routing decision
            instance = routing_decision.selected_instance
            self.logger.debug(f"Using intelligent routing to {instance.model_name}")
        else:
            # Fallback to load balancer
            instance = self.load_balancer.get_best_instance(request)
            if routing_decision:
                # Try fallback chain if primary instance is not available
                for fallback_id in routing_decision.fallback_chain:
                    fallback_instance = self.instances.get(fallback_id)
                    if fallback_instance and fallback_instance.is_available():
                        instance = fallback_instance
                        self.logger.info(f"Using fallback instance {instance.model_name}")
                        break
        
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
            # Make request to LLM instance (handles different engine types)
            response_text, token_metrics = self._make_llm_request(instance, request)
            
            # Update metrics
            response_time = time.time() - start_time
            instance.total_requests += 1
            instance.response_time_avg = (
                (instance.response_time_avg * (instance.total_requests - 1) + response_time) /
                instance.total_requests
            )
            
            # Update model-specific metrics if token information is available
            if token_metrics:
                instance.update_model_metrics(
                    request_tokens=token_metrics.get('input_tokens', len(request.prompt.split())),
                    response_tokens=token_metrics.get('output_tokens', len(response_text.split())),
                    response_time=response_time,
                    context_length=token_metrics.get('context_length', len(request.prompt))
                )
            
            return LLMResponse(
                request_id=request.request_id,
                response_text=response_text,
                response_time=response_time,
                instance_id=instance.instance_id,
                success=True,
                metadata={
                    'model_name': instance.model_name,
                    'engine_type': instance.engine_type,
                    'provider': instance.provider,
                    'token_metrics': token_metrics
                }
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
                error_message=str(e),
                metadata={
                    'model_name': instance.model_name,
                    'engine_type': instance.engine_type,
                    'provider': instance.provider
                }
            )
        finally:
            # Decrement load
            instance.current_load -= 1
    
    def _make_llm_request(self, instance: LLMInstance, request: LLMRequest) -> Tuple[str, Dict[str, Any]]:
        """Make actual request to LLM instance with engine-aware handling."""
        if instance.engine_type == "local":
            return self._make_local_llm_request(instance, request)
        elif instance.engine_type == "gcp":
            return self._make_gcp_llm_request(instance, request)
        else:
            raise ValueError(f"Unsupported engine type: {instance.engine_type}")
    
    def _make_local_llm_request(self, instance: LLMInstance, request: LLMRequest) -> Tuple[str, Dict[str, Any]]:
        """Make request to local Ollama instance with connection pooling."""
        url = f"http://{instance.host}:{instance.port}/api/generate"
        
        payload = {
            "model": instance.model_name,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 2048
            },
            **request.model_config
        }
        
        try:
            # Use HTTP connection pool manager if available
            if self.http_pool_manager:
                response = self.http_pool_manager.post(
                    url,
                    json_data=payload,
                    timeout=request.timeout,
                    request_complexity='complex'  # LLM generation is complex
                )
            else:
                # Fallback to legacy session creation
                session = requests.Session()
                retry_strategy = Retry(
                    total=3,
                    backoff_factor=0.1,
                    status_forcelist=[500, 502, 503, 504]
                )
                session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
                session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
                
                response = session.post(
                    url,
                    json=payload,
                    timeout=request.timeout
                )
                response.raise_for_status()
            
            result = response.json()
            response_text = result.get("response", "")
            
            # Extract token metrics from Ollama response
            token_metrics = {
                'input_tokens': len(request.prompt.split()),  # Approximate
                'output_tokens': len(response_text.split()),  # Approximate
                'context_length': len(request.prompt),
                'total_duration': result.get('total_duration', 0),
                'load_duration': result.get('load_duration', 0),
                'prompt_eval_count': result.get('prompt_eval_count', 0),
                'eval_count': result.get('eval_count', 0)
            }
            
            return response_text, token_metrics
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Local LLM HTTP request failed: {e}")
        except ConnectionError as e:
            # Circuit breaker is open
            raise Exception(f"Circuit breaker open: {e}")
    
    def _make_gcp_llm_request(self, instance: LLMInstance, request: LLMRequest) -> Tuple[str, Dict[str, Any]]:
        """Make request to Google Cloud LLM instance."""
        try:
            # Import Google Cloud libraries
            import google.generativeai as genai
            
            # Configure the model
            model = genai.GenerativeModel(instance.model_name)
            
            # Prepare generation config
            generation_config = {
                "temperature": request.model_config.get("temperature", 0.1),
                "top_p": request.model_config.get("top_p", 0.9),
                "max_output_tokens": request.model_config.get("max_output_tokens", 2048)
            }
            
            # Make the request
            response = model.generate_content(
                request.prompt,
                generation_config=generation_config
            )
            
            response_text = response.text.strip()
            
            # Extract token metrics from GCP response
            token_metrics = {
                'input_tokens': getattr(response.usage_metadata, 'prompt_token_count', len(request.prompt.split())),
                'output_tokens': getattr(response.usage_metadata, 'candidates_token_count', len(response_text.split())),
                'context_length': len(request.prompt),
                'total_tokens': getattr(response.usage_metadata, 'total_token_count', 0)
            }
            
            return response_text, token_metrics
            
        except ImportError:
            raise Exception("Google Cloud AI libraries not available")
        except Exception as e:
            raise Exception(f"GCP LLM request failed: {e}")
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get current pool status."""
        healthy_instances = sum(1 for i in self.instances.values() 
                               if i.status == LLMInstanceStatus.HEALTHY)
        total_load = sum(i.current_load for i in self.instances.values())
        
        status = {
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
        
        # Add HTTP connection pool statistics
        if self.http_pool_manager:
            status['http_pool_stats'] = self.http_pool_manager.get_stats()
        
        # Add intelligent routing analytics
        status['routing_analytics'] = self.get_routing_analytics()
        
        # Add model capabilities summary
        status['model_capabilities'] = {}
        for instance in self.instances.values():
            status['model_capabilities'][instance.model_name] = {
                'engine_type': instance.engine_type,
                'provider': instance.provider,
                'speed_score': instance.speed_score,
                'quality_score': instance.quality_score,
                'optimal_use_cases': instance.optimal_use_cases,
                'performance_summary': instance.get_performance_summary()
            }
        
        return status
    
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
    """Model-aware load balancer for LLM instances."""
    
    def __init__(self, pool_manager: LLMPoolManager):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
    
    def get_best_instance(self, request: LLMRequest = None) -> Optional[LLMInstance]:
        """Get the best available instance for a request with model-aware selection."""
        available_instances = [
            instance for instance in self.pool_manager.instances.values()
            if instance.is_available()
        ]
        
        if not available_instances:
            return None
        
        # If no specific request context, use legacy behavior
        if not request:
            return self._get_best_instance_legacy(available_instances)
        
        # Model-aware selection based on request characteristics
        return self._get_best_instance_model_aware(available_instances, request)
    
    def _get_best_instance_legacy(self, available_instances: List[LLMInstance]) -> LLMInstance:
        """Legacy instance selection based on health score and load."""
        available_instances.sort(
            key=lambda i: (-i.health_score, i.current_load)
        )
        return available_instances[0]
    
    def _get_best_instance_model_aware(self, available_instances: List[LLMInstance], 
                                     request: LLMRequest) -> Optional[LLMInstance]:
        """Model-aware instance selection based on request characteristics and routing strategy."""
        # Determine request complexity and characteristics
        request_complexity = self._analyze_request_complexity(request)
        routing_strategy = self._get_routing_strategy()
        use_case = request.model_config.get('use_case', 'general')
        
        # Filter instances suitable for the use case
        suitable_instances = [
            instance for instance in available_instances
            if instance.is_suitable_for_use_case(use_case)
        ]
        
        if not suitable_instances:
            # Fallback to all available instances if none are specifically suitable
            suitable_instances = available_instances
            self.logger.warning(f"No instances suitable for use case '{use_case}', using all available")
        
        # Apply routing strategy
        if routing_strategy == "speed_first":
            return self._select_fastest_instance(suitable_instances, request_complexity)
        elif routing_strategy == "quality_first":
            return self._select_highest_quality_instance(suitable_instances, request_complexity)
        elif routing_strategy == "cost_first":
            return self._select_most_cost_effective_instance(suitable_instances, request)
        elif routing_strategy == "balanced":
            return self._select_balanced_instance(suitable_instances, request)
        else:
            # Default to suitability score
            return self._select_by_suitability_score(suitable_instances)
    
    def _analyze_request_complexity(self, request: LLMRequest) -> str:
        """Analyze request complexity based on prompt length and other factors."""
        prompt_length = len(request.prompt)
        
        # Simple heuristics for complexity classification
        if prompt_length < 500:
            return "simple"
        elif prompt_length < 2000:
            return "medium"
        elif prompt_length < 10000:
            return "complex"
        else:
            return "batch"
    
    def _get_routing_strategy(self) -> str:
        """Get the current routing strategy from pool configuration."""
        pool_config = self.pool_manager.config.get('pool_config', {})
        return pool_config.get('routing_strategy', 'balanced')
    
    def _select_fastest_instance(self, instances: List[LLMInstance], 
                               complexity: str) -> LLMInstance:
        """Select the fastest instance for the given complexity."""
        # Weight by speed score and response time
        instances.sort(key=lambda i: (
            -i.speed_score,  # Higher speed score is better
            i.response_time_avg,  # Lower response time is better
            i.current_load  # Lower load is better
        ))
        return instances[0]
    
    def _select_highest_quality_instance(self, instances: List[LLMInstance], 
                                       complexity: str) -> LLMInstance:
        """Select the highest quality instance for the given complexity."""
        # Weight by quality score and reliability
        instances.sort(key=lambda i: (
            -i.quality_score,  # Higher quality score is better
            -i.reliability_score,  # Higher reliability is better
            i.current_load  # Lower load is better
        ))
        return instances[0]
    
    def _select_most_cost_effective_instance(self, instances: List[LLMInstance], 
                                           request: LLMRequest) -> LLMInstance:
        """Select the most cost-effective instance."""
        # Prefer local models (cost_per_request = 0), then by cost efficiency
        instances.sort(key=lambda i: (
            i.cost_per_request,  # Lower cost is better (0 for local models)
            -i.model_specific_metrics.get('cost_efficiency_score', 0.5),  # Higher efficiency is better
            i.current_load  # Lower load is better
        ))
        return instances[0]
    
    def _select_balanced_instance(self, instances: List[LLMInstance], 
                                request: LLMRequest) -> LLMInstance:
        """Select instance using balanced criteria."""
        # Use suitability score which already balances health and cost efficiency
        return self._select_by_suitability_score(instances)
    
    def _select_by_suitability_score(self, instances: List[LLMInstance]) -> LLMInstance:
        """Select instance with highest suitability score."""
        instances.sort(key=lambda i: (
            -i.suitability_score,  # Higher suitability is better
            i.current_load  # Lower load is better as tiebreaker
        ))
        return instances[0]
    
    def get_instance_recommendations(self, request: LLMRequest) -> List[Dict[str, Any]]:
        """Get ranked instance recommendations for a request."""
        available_instances = [
            instance for instance in self.pool_manager.instances.values()
            if instance.is_available()
        ]
        
        if not available_instances:
            return []
        
        request_complexity = self._analyze_request_complexity(request)
        use_case = request.model_config.get('use_case', 'general')
        
        recommendations = []
        for instance in available_instances:
            suitability = instance.is_suitable_for_use_case(use_case)
            estimated_cost = instance.get_estimated_cost(
                input_tokens=len(request.prompt.split()),
                output_tokens=512  # Estimate
            )
            
            recommendations.append({
                'instance_id': instance.instance_id,
                'model_name': instance.model_name,
                'engine_type': instance.engine_type,
                'suitability_score': instance.suitability_score,
                'health_score': instance.health_score,
                'suitable_for_use_case': suitability,
                'estimated_cost': estimated_cost,
                'estimated_response_time': instance.response_time_avg,
                'current_load': instance.current_load,
                'quality_score': instance.quality_score,
                'speed_score': instance.speed_score,
                'recommendation_reason': self._get_recommendation_reason(instance, request_complexity, use_case)
            })
        
        # Sort by suitability score
        recommendations.sort(key=lambda r: -r['suitability_score'])
        return recommendations
    
    def _get_recommendation_reason(self, instance: LLMInstance, complexity: str, use_case: str) -> str:
        """Generate human-readable recommendation reason."""
        reasons = []
        
        if instance.engine_type == "local":
            reasons.append("cost-effective (local)")
        
        if instance.quality_score > 0.8:
            reasons.append("high quality")
        
        if instance.speed_score > 0.8:
            reasons.append("fast response")
        
        if instance.is_suitable_for_use_case(use_case):
            reasons.append(f"optimized for {use_case}")
        
        if instance.current_load == 0:
            reasons.append("no current load")
        
        return ", ".join(reasons) if reasons else "available"


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
            
            # Use HTTP connection pool manager if available
            if self.pool_manager.http_pool_manager:
                response = self.pool_manager.http_pool_manager.get(
                    url, 
                    timeout=5.0,
                    request_complexity='simple'  # Health checks are simple
                )
            else:
                # Fallback to direct requests
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