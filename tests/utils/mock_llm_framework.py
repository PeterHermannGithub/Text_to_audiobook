"""
Mock LLM Framework for Multi-Model Integration Testing.

This module provides comprehensive mock implementations of all LLM models
and instances for testing the multi-model load balancing system without
requiring actual LLM services.
"""

import time
import random
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import uuid

from src.llm_pool.llm_pool_manager import LLMInstance, LLMInstanceStatus, LLMRequest, LLMResponse
from src.llm_pool.intelligent_router import RequestComplexity, RoutingStrategy
from config import settings


class MockFailureType(Enum):
    """Types of failures that can be simulated."""
    NONE = "none"
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    INVALID_RESPONSE = "invalid_response"
    SERVICE_UNAVAILABLE = "service_unavailable"
    RATE_LIMIT = "rate_limit"


@dataclass
class MockModelProfile:
    """Profile defining the characteristics of a mock model."""
    model_name: str
    engine_type: str
    provider: str
    base_response_time: float  # Base response time in seconds
    response_time_variance: float  # Variance in response time (±%)
    cost_per_1k_tokens: float
    failure_rate: float  # Probability of failure (0.0-1.0)
    quality_score: float  # Quality score (0.0-1.0)
    speed_score: float  # Speed score (0.0-1.0)
    reliability_score: float
    max_context_length: int
    supports_json: bool
    optimal_use_cases: List[str]
    
    # Response patterns
    response_templates: List[str] = field(default_factory=list)
    
    # Performance characteristics per complexity
    complexity_modifiers: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class MockTestScenario:
    """Test scenario configuration for mock framework."""
    name: str
    description: str
    duration_seconds: float
    request_rate: float  # Requests per second
    complexity_distribution: Dict[RequestComplexity, float]  # Probability distribution
    failure_injection: Dict[str, float]  # Model -> failure rate override
    load_patterns: List[Dict[str, Any]]  # Time-based load patterns


class MockLLMInstance:
    """Mock implementation of LLMInstance for testing."""
    
    def __init__(self, profile: MockModelProfile, instance_id: str = None):
        """Initialize mock LLM instance."""
        self.profile = profile
        self.instance_id = instance_id or f"mock-{profile.model_name}-{uuid.uuid4().hex[:8]}"
        
        # Mock instance state
        self.status = LLMInstanceStatus.HEALTHY
        self.current_load = 0
        self.max_load = 5
        self.last_health_check = time.time()
        self.response_time_avg = profile.base_response_time
        self.total_requests = 0
        self.failed_requests = 0
        self.created_at = time.time()
        
        # Model-specific properties from profile
        self.model_name = profile.model_name
        self.engine_type = profile.engine_type
        self.provider = profile.provider
        self.cost_per_request = profile.cost_per_1k_tokens
        self.quality_score = profile.quality_score
        self.speed_score = profile.speed_score
        self.reliability_score = profile.reliability_score
        self.optimal_use_cases = profile.optimal_use_cases.copy()
        
        # Model capabilities
        self.model_capabilities = {
            "engine": profile.engine_type,
            "provider": profile.provider,
            "max_context_length": profile.max_context_length,
            "supports_json": profile.supports_json,
            "optimal_use_cases": profile.optimal_use_cases,
            "cost_per_1k_tokens": profile.cost_per_1k_tokens
        }
        
        # Model-specific metrics
        self.model_specific_metrics = {
            "avg_tokens_per_sec": 100.0 / profile.base_response_time,
            "avg_input_tokens": 0.0,
            "avg_output_tokens": 0.0,
            "cost_efficiency_score": profile.quality_score / max(profile.cost_per_1k_tokens, 0.001),
            "context_utilization": 0.0
        }
        
        # Mock state tracking
        self._request_history: List[Dict[str, Any]] = []
        self._forced_failure_type: Optional[MockFailureType] = None
        self._custom_response_time: Optional[float] = None
        
        self.logger = logging.getLogger(__name__)
    
    @property
    def url(self) -> str:
        """Get mock URL for the instance."""
        if self.engine_type == "gcp":
            return f"mock-gcp://{self.provider}/{self.model_name}"
        return f"mock-http://localhost:1143{self.instance_id[-1]}"
    
    @property
    def health_score(self) -> float:
        """Calculate mock health score."""
        if self.total_requests == 0:
            return self.reliability_score
        
        success_rate = 1.0 - (self.failed_requests / self.total_requests)
        load_factor = 1.0 - (self.current_load / self.max_load)
        response_factor = min(1.0, 1.0 / max(0.1, self.response_time_avg))
        capability_factor = (self.speed_score + self.quality_score + self.reliability_score) / 3.0
        
        return (success_rate * 0.3) + (load_factor * 0.25) + (response_factor * 0.25) + (capability_factor * 0.2)
    
    @property
    def suitability_score(self) -> float:
        """Calculate mock suitability score."""
        health = self.health_score
        cost_efficiency = self.model_specific_metrics.get("cost_efficiency_score", 0.5)
        return (health * 0.7) + (cost_efficiency * 0.3)
    
    def is_available(self) -> bool:
        """Check if mock instance is available."""
        return (self.status == LLMInstanceStatus.HEALTHY and 
                self.current_load < self.max_load)
    
    def is_suitable_for_use_case(self, use_case: str) -> bool:
        """Check suitability for use case."""
        if not self.optimal_use_cases:
            return True
        
        for optimal_case in self.optimal_use_cases:
            if use_case == optimal_case or use_case in optimal_case or optimal_case in use_case:
                return True
        return False
    
    def get_estimated_cost(self, input_tokens: int, output_tokens: int = None) -> float:
        """Calculate estimated cost for mock request."""
        if self.engine_type == "local":
            return 0.0
        
        total_tokens = input_tokens + (output_tokens or input_tokens * 0.5)
        return (total_tokens / 1000.0) * self.cost_per_request
    
    def update_model_metrics(self, request_tokens: int, response_tokens: int, 
                           response_time: float, context_length: int):
        """Update mock model metrics."""
        total_requests = self.total_requests + 1
        
        # Update rolling averages
        tokens_per_sec = (request_tokens + response_tokens) / max(0.1, response_time)
        self.model_specific_metrics["avg_tokens_per_sec"] = (
            (self.model_specific_metrics["avg_tokens_per_sec"] * self.total_requests + tokens_per_sec) / 
            total_requests
        )
        
        self.model_specific_metrics["avg_input_tokens"] = (
            (self.model_specific_metrics["avg_input_tokens"] * self.total_requests + request_tokens) /
            total_requests
        )
        
        self.model_specific_metrics["avg_output_tokens"] = (
            (self.model_specific_metrics["avg_output_tokens"] * self.total_requests + response_tokens) /
            total_requests
        )
        
        # Context utilization
        max_context = self.model_capabilities.get("max_context_length", 8192)
        utilization = context_length / max_context
        self.model_specific_metrics["context_utilization"] = (
            (self.model_specific_metrics["context_utilization"] * self.total_requests + utilization) /
            total_requests
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get mock performance summary."""
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
    
    def force_failure(self, failure_type: MockFailureType):
        """Force a specific failure type for testing."""
        self._forced_failure_type = failure_type
    
    def set_custom_response_time(self, response_time: float):
        """Set custom response time for testing."""
        self._custom_response_time = response_time
    
    def reset_mock_state(self):
        """Reset mock state for clean testing."""
        self._forced_failure_type = None
        self._custom_response_time = None
        self._request_history.clear()
        self.current_load = 0
        self.total_requests = 0
        self.failed_requests = 0
        self.status = LLMInstanceStatus.HEALTHY


class MockLLMFramework:
    """
    Comprehensive Mock LLM Framework for Multi-Model Testing.
    
    Provides realistic mock implementations of all models defined in settings.py
    with configurable failure patterns, performance characteristics, and test scenarios.
    """
    
    def __init__(self):
        """Initialize the mock framework."""
        self.logger = logging.getLogger(__name__)
        
        # Create model profiles from settings
        self.model_profiles = self._create_model_profiles()
        
        # Mock instances
        self.mock_instances: Dict[str, MockLLMInstance] = {}
        
        # Test scenario tracking
        self.current_scenario: Optional[MockTestScenario] = None
        self.scenario_start_time: Optional[float] = None
        
        # Global mock controls
        self.global_failure_rate = 0.0
        self.global_latency_multiplier = 1.0
        self.network_simulation_enabled = False
        
        # Request tracking for analytics
        self.request_log: List[Dict[str, Any]] = []
        
        self.logger.info("Mock LLM Framework initialized with profiles for all configured models")
    
    def _create_model_profiles(self) -> Dict[str, MockModelProfile]:
        """Create model profiles from settings configuration."""
        profiles = {}
        
        for model_name, capabilities in settings.MODEL_CAPABILITIES.items():
            # Convert tier strings to numeric scores
            speed_scores = {"slow": 0.3, "medium": 0.6, "fast": 0.9}
            quality_scores = {"basic": 0.4, "medium": 0.7, "high": 1.0}
            
            profile = MockModelProfile(
                model_name=model_name,
                engine_type=capabilities["engine"],
                provider=capabilities["provider"],
                base_response_time=capabilities["avg_response_time_ms"] / 1000.0,  # Convert to seconds
                response_time_variance=0.2,  # ±20% variance
                cost_per_1k_tokens=capabilities["cost_per_1k_tokens"],
                failure_rate=1.0 - capabilities["reliability_score"],  # Convert reliability to failure rate
                quality_score=quality_scores.get(capabilities["quality_tier"], 0.7),
                speed_score=speed_scores.get(capabilities["speed_tier"], 0.6),
                reliability_score=capabilities["reliability_score"],
                max_context_length=capabilities["max_context_length"],
                supports_json=capabilities["supports_json"],
                optimal_use_cases=capabilities["optimal_use_cases"].copy(),
                response_templates=self._get_response_templates(model_name),
                complexity_modifiers=self._get_complexity_modifiers(model_name)
            )
            profiles[model_name] = profile
            
        return profiles
    
    def _get_response_templates(self, model_name: str) -> List[str]:
        """Get response templates for specific model."""
        templates = [
            '{"speaker": "Character", "confidence": 0.95, "reasoning": "Clear dialogue pattern"}',
            '{"speaker": "Narrator", "confidence": 0.90, "reasoning": "Descriptive text"}',
            '{"speaker": "AMBIGUOUS", "confidence": 0.60, "reasoning": "Unclear speaker attribution"}'
        ]
        
        # Model-specific response patterns
        if "gemini" in model_name:
            templates.extend([
                '{"speaker": "Character", "confidence": 0.98, "reasoning": "High-quality analysis"}',
                '{"speaker": "Narrator", "confidence": 0.95, "reasoning": "Sophisticated narrative detection"}'
            ])
        elif "deepseek" in model_name:
            templates.extend([
                '{"speaker": "Character", "confidence": 0.92, "reasoning": "Deep reasoning applied"}',
                '{"speaker": "Narrator", "confidence": 0.88, "reasoning": "Complex pattern analysis"}'
            ])
        
        return templates
    
    def _get_complexity_modifiers(self, model_name: str) -> Dict[str, Dict[str, float]]:
        """Get complexity-based performance modifiers."""
        base_modifiers = {
            "simple": {"time_multiplier": 0.7, "quality_boost": 0.1},
            "medium": {"time_multiplier": 1.0, "quality_boost": 0.0},
            "complex": {"time_multiplier": 1.5, "quality_boost": -0.05},
            "batch": {"time_multiplier": 2.0, "quality_boost": -0.1},
            "heavy": {"time_multiplier": 3.0, "quality_boost": -0.15}
        }
        
        # Model-specific adjustments
        if "fast" in model_name or "flash" in model_name:
            # Fast models handle complexity better
            for complexity in base_modifiers:
                base_modifiers[complexity]["time_multiplier"] *= 0.8
        elif "pro" in model_name or "deepseek" in model_name:
            # High-quality models maintain quality under complexity
            for complexity in base_modifiers:
                base_modifiers[complexity]["quality_boost"] *= 0.5
        
        return base_modifiers
    
    def create_mock_instances(self, models: List[str] = None, instances_per_model: int = 2) -> Dict[str, MockLLMInstance]:
        """Create mock instances for specified models."""
        if models is None:
            models = list(self.model_profiles.keys())
        
        created_instances = {}
        
        for model_name in models:
            if model_name not in self.model_profiles:
                self.logger.warning(f"Unknown model {model_name}, skipping")
                continue
            
            profile = self.model_profiles[model_name]
            
            for i in range(instances_per_model):
                instance_id = f"mock-{model_name.replace(':', '-').replace('.', '-')}-{i}"
                mock_instance = MockLLMInstance(profile, instance_id)
                
                self.mock_instances[instance_id] = mock_instance
                created_instances[instance_id] = mock_instance
                
                self.logger.debug(f"Created mock instance: {instance_id}")
        
        self.logger.info(f"Created {len(created_instances)} mock instances for {len(models)} models")
        return created_instances
    
    def simulate_request(self, instance: MockLLMInstance, request: LLMRequest) -> LLMResponse:
        """Simulate an LLM request with realistic behavior."""
        start_time = time.time()
        
        # Increment load
        instance.current_load += 1
        
        try:
            # Check for forced failures
            if instance._forced_failure_type and instance._forced_failure_type != MockFailureType.NONE:
                return self._generate_failure_response(instance, request, instance._forced_failure_type)
            
            # Check for random failures based on failure rate
            if random.random() < instance.profile.failure_rate * (1.0 + self.global_failure_rate):
                failure_types = [MockFailureType.TIMEOUT, MockFailureType.CONNECTION_ERROR, 
                               MockFailureType.INVALID_RESPONSE, MockFailureType.SERVICE_UNAVAILABLE]
                failure_type = random.choice(failure_types)
                return self._generate_failure_response(instance, request, failure_type)
            
            # Calculate response time
            response_time = self._calculate_response_time(instance, request)
            
            # Simulate processing delay
            time.sleep(min(response_time, 0.1))  # Cap simulation delay at 100ms for testing speed
            
            # Generate realistic response
            response_text = self._generate_response_text(instance, request)
            
            # Calculate costs and tokens
            input_tokens = len(request.prompt.split())
            output_tokens = len(response_text.split())
            estimated_cost = instance.get_estimated_cost(input_tokens, output_tokens)
            
            # Update instance metrics
            instance.total_requests += 1
            instance.response_time_avg = (
                (instance.response_time_avg * (instance.total_requests - 1) + response_time) /
                instance.total_requests
            )
            
            instance.update_model_metrics(
                request_tokens=input_tokens,
                response_tokens=output_tokens,
                response_time=response_time,
                context_length=len(request.prompt)
            )
            
            # Log request for analytics
            self._log_request(instance, request, response_time, estimated_cost, True)
            
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
                    'token_metrics': {
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'context_length': len(request.prompt),
                        'estimated_cost': estimated_cost
                    }
                }
            )
            
        except Exception as e:
            # Handle unexpected errors
            instance.failed_requests += 1
            self._log_request(instance, request, 0.0, 0.0, False, str(e))
            
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
    
    def _calculate_response_time(self, instance: MockLLMInstance, request: LLMRequest) -> float:
        """Calculate realistic response time based on model and request characteristics."""
        if instance._custom_response_time:
            return instance._custom_response_time
        
        base_time = instance.profile.base_response_time
        
        # Apply complexity modifiers
        complexity = self._determine_request_complexity(request)
        complexity_modifier = instance.profile.complexity_modifiers.get(
            complexity.value, {"time_multiplier": 1.0}
        )["time_multiplier"]
        
        # Add variance
        variance = random.uniform(
            1.0 - instance.profile.response_time_variance,
            1.0 + instance.profile.response_time_variance
        )
        
        # Apply global latency multiplier
        response_time = base_time * complexity_modifier * variance * self.global_latency_multiplier
        
        return max(0.1, response_time)  # Minimum 100ms response time
    
    def _determine_request_complexity(self, request: LLMRequest) -> RequestComplexity:
        """Determine request complexity for mock simulation."""
        prompt_length = len(request.prompt)
        word_count = len(request.prompt.split())
        
        if prompt_length > 20000 or word_count > 4000:
            return RequestComplexity.HEAVY
        elif prompt_length > 10000 or word_count > 2000:
            return RequestComplexity.BATCH
        elif prompt_length > 2000 or word_count > 400:
            return RequestComplexity.COMPLEX
        elif prompt_length > 500 or word_count > 100:
            return RequestComplexity.MEDIUM
        else:
            return RequestComplexity.SIMPLE
    
    def _generate_response_text(self, instance: MockLLMInstance, request: LLMRequest) -> str:
        """Generate realistic response text for the model."""
        # Select response template based on model characteristics
        templates = instance.profile.response_templates
        if not templates:
            templates = ['{"speaker": "Character", "confidence": 0.85, "reasoning": "Mock response"}']
        
        # Add model-specific quality variations
        base_response = random.choice(templates)
        
        # Parse and modify based on model quality
        try:
            response_data = json.loads(base_response)
            
            # Adjust confidence based on model quality
            quality_adjustment = (instance.quality_score - 0.7) * 0.2  # ±0.2 adjustment
            response_data["confidence"] = max(0.1, min(1.0, 
                response_data["confidence"] + quality_adjustment
            ))
            
            # Add model-specific reasoning patterns
            if "gemini" in instance.model_name:
                response_data["reasoning"] = f"Advanced analysis: {response_data['reasoning']}"
            elif "deepseek" in instance.model_name:
                response_data["reasoning"] = f"Deep reasoning: {response_data['reasoning']}"
            elif "mistral" in instance.model_name:
                response_data["reasoning"] = f"Efficient classification: {response_data['reasoning']}"
            
            return json.dumps(response_data, indent=2)
            
        except json.JSONDecodeError:
            # Fallback to template string
            return base_response
    
    def _generate_failure_response(self, instance: MockLLMInstance, request: LLMRequest, 
                                 failure_type: MockFailureType) -> LLMResponse:
        """Generate appropriate failure response."""
        instance.failed_requests += 1
        response_time = random.uniform(0.1, 2.0)  # Random failure response time
        
        error_messages = {
            MockFailureType.TIMEOUT: "Request timeout after 30 seconds",
            MockFailureType.CONNECTION_ERROR: "Connection refused to mock LLM service",
            MockFailureType.INVALID_RESPONSE: "Invalid JSON response from mock LLM",
            MockFailureType.SERVICE_UNAVAILABLE: "Mock LLM service temporarily unavailable",
            MockFailureType.RATE_LIMIT: "Rate limit exceeded for mock LLM service"
        }
        
        self._log_request(instance, request, response_time, 0.0, False, error_messages[failure_type])
        
        return LLMResponse(
            request_id=request.request_id,
            response_text="",
            response_time=response_time,
            instance_id=instance.instance_id,
            success=False,
            error_message=error_messages[failure_type],
            metadata={
                'model_name': instance.model_name,
                'engine_type': instance.engine_type,
                'provider': instance.provider,
                'failure_type': failure_type.value
            }
        )
    
    def _log_request(self, instance: MockLLMInstance, request: LLMRequest,
                    response_time: float, cost: float, success: bool, error: str = None):
        """Log request for analytics and debugging."""
        log_entry = {
            'timestamp': time.time(),
            'instance_id': instance.instance_id,
            'model_name': instance.model_name,
            'request_id': request.request_id,
            'prompt_length': len(request.prompt),
            'response_time': response_time,
            'cost': cost,
            'success': success,
            'error': error,
            'complexity': self._determine_request_complexity(request).value
        }
        
        self.request_log.append(log_entry)
        
        # Keep only recent requests (last 1000)
        if len(self.request_log) > 1000:
            self.request_log = self.request_log[-1000:]
    
    def run_test_scenario(self, scenario: MockTestScenario) -> Dict[str, Any]:
        """Run a specific test scenario and return results."""
        self.current_scenario = scenario
        self.scenario_start_time = time.time()
        
        self.logger.info(f"Starting test scenario: {scenario.name}")
        
        # Reset all mock instances
        for instance in self.mock_instances.values():
            instance.reset_mock_state()
        
        # Clear request log
        self.request_log.clear()
        
        # Scenario results will be collected by the test framework
        scenario_results = {
            'scenario_name': scenario.name,
            'start_time': self.scenario_start_time,
            'duration': scenario.duration_seconds,
            'target_request_rate': scenario.request_rate,
            'complexity_distribution': {k.value: v for k, v in scenario.complexity_distribution.items()},
            'requests_generated': 0,
            'requests_completed': 0,
            'average_response_time': 0.0,
            'total_cost': 0.0,
            'error_rate': 0.0
        }
        
        return scenario_results
    
    def set_global_failure_rate(self, rate: float):
        """Set global failure rate multiplier for testing fault tolerance."""
        self.global_failure_rate = rate
        self.logger.info(f"Set global failure rate multiplier to {rate}")
    
    def set_global_latency_multiplier(self, multiplier: float):
        """Set global latency multiplier for testing performance under load."""
        self.global_latency_multiplier = multiplier
        self.logger.info(f"Set global latency multiplier to {multiplier}")
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary from mock framework."""
        if not self.request_log:
            return {"error": "No requests logged"}
        
        total_requests = len(self.request_log)
        successful_requests = sum(1 for req in self.request_log if req['success'])
        total_cost = sum(req['cost'] for req in self.request_log)
        total_response_time = sum(req['response_time'] for req in self.request_log if req['success'])
        
        # Model-wise breakdown
        model_stats = defaultdict(lambda: {'requests': 0, 'successes': 0, 'total_time': 0.0, 'total_cost': 0.0})
        for req in self.request_log:
            model = req['model_name']
            model_stats[model]['requests'] += 1
            if req['success']:
                model_stats[model]['successes'] += 1
                model_stats[model]['total_time'] += req['response_time']
            model_stats[model]['total_cost'] += req['cost']
        
        return {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'error_rate': (total_requests - successful_requests) / total_requests,
            'average_response_time': total_response_time / max(successful_requests, 1),
            'total_cost': total_cost,
            'average_cost_per_request': total_cost / total_requests,
            'model_breakdown': {
                model: {
                    'requests': stats['requests'],
                    'success_rate': stats['successes'] / max(stats['requests'], 1),
                    'avg_response_time': stats['total_time'] / max(stats['successes'], 1),
                    'total_cost': stats['total_cost']
                }
                for model, stats in model_stats.items()
            }
        }
    
    def cleanup(self):
        """Clean up mock framework resources."""
        self.mock_instances.clear()
        self.request_log.clear()
        self.current_scenario = None
        self.scenario_start_time = None
        self.logger.info("Mock LLM Framework cleaned up")


# Pre-defined test scenarios for common testing patterns
COMMON_TEST_SCENARIOS = {
    "balanced_load": MockTestScenario(
        name="Balanced Load Test",
        description="Balanced workload across all complexity levels",
        duration_seconds=60.0,
        request_rate=5.0,
        complexity_distribution={
            RequestComplexity.SIMPLE: 0.4,
            RequestComplexity.MEDIUM: 0.3,
            RequestComplexity.COMPLEX: 0.2,
            RequestComplexity.BATCH: 0.1
        },
        failure_injection={},
        load_patterns=[]
    ),
    
    "high_load_stress": MockTestScenario(
        name="High Load Stress Test",
        description="High request rate to test system limits",
        duration_seconds=30.0,
        request_rate=20.0,
        complexity_distribution={
            RequestComplexity.SIMPLE: 0.6,
            RequestComplexity.MEDIUM: 0.3,
            RequestComplexity.COMPLEX: 0.1
        },
        failure_injection={},
        load_patterns=[]
    ),
    
    "fault_tolerance": MockTestScenario(
        name="Fault Tolerance Test",
        description="Test system behavior under high failure rates",
        duration_seconds=45.0,
        request_rate=3.0,
        complexity_distribution={
            RequestComplexity.SIMPLE: 0.5,
            RequestComplexity.MEDIUM: 0.5
        },
        failure_injection={
            "deepseek-v2:16b": 0.3,  # 30% failure rate
            "gemini-1.0-pro": 0.2   # 20% failure rate
        },
        load_patterns=[]
    ),
    
    "cost_optimization": MockTestScenario(
        name="Cost Optimization Test", 
        description="Test cost-aware routing with budget constraints",
        duration_seconds=90.0,
        request_rate=2.0,
        complexity_distribution={
            RequestComplexity.SIMPLE: 0.3,
            RequestComplexity.MEDIUM: 0.4,
            RequestComplexity.COMPLEX: 0.3
        },
        failure_injection={},
        load_patterns=[]
    )
}


def create_mock_framework() -> MockLLMFramework:
    """Factory function to create and initialize mock framework."""
    return MockLLMFramework()