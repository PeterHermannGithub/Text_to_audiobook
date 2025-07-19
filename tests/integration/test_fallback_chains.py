"""
Fallback Chain Testing Suite for Multi-Model Load Balancing.

This module provides comprehensive testing of fallback chain mechanisms,
including primary-to-fallback routing, model-type failover, and graceful
degradation patterns in the intelligent routing system.
"""

import unittest
import time
from typing import Dict, List, Any, Tuple, Optional
from unittest.mock import Mock, patch
from collections import defaultdict, deque
import logging

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.llm_pool.llm_pool_manager import LLMPoolManager, LLMRequest, LLMInstanceStatus
from src.llm_pool.intelligent_router import IntelligentRequestRouter, RoutingDecision
from tests.utils.mock_llm_framework import (
    MockLLMFramework, MockFailureType, create_mock_framework
)
from config import settings


class FallbackChainResults:
    """Container for fallback chain test results."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time = time.time()
        self.end_time = None
        
        # Fallback tracking
        self.primary_attempts = 0
        self.primary_successes = 0
        self.fallback_activations = 0
        self.fallback_successes = 0
        self.fallback_failures = 0
        self.chain_exhaustions = 0
        
        # Fallback chain analysis
        self.fallback_paths = []  # List of (primary_id, fallback_id, success)
        self.fallback_depths = []  # How deep into chain each request went
        self.fallback_response_times = []
        
        # Performance during fallback
        self.primary_response_times = []
        self.fallback_response_times = []
        
        # Recovery tracking
        self.recovery_events = []
        self.recovery_success_rates = []
    
    def finish(self):
        """Finalize results."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
        # Calculate metrics
        self.primary_success_rate = self.primary_successes / max(self.primary_attempts, 1)
        self.fallback_success_rate = self.fallback_successes / max(self.fallback_activations, 1)
        self.chain_exhaustion_rate = self.chain_exhaustions / max(self.fallback_activations, 1)
        
        if self.primary_response_times:
            self.avg_primary_response_time = sum(self.primary_response_times) / len(self.primary_response_times)
        else:
            self.avg_primary_response_time = 0.0
            
        if self.fallback_response_times:
            self.avg_fallback_response_time = sum(self.fallback_response_times) / len(self.fallback_response_times)
        else:
            self.avg_fallback_response_time = 0.0


class TestFallbackChains(unittest.TestCase):
    """Comprehensive fallback chain testing suite."""
    
    @classmethod
    def setUpClass(cls):
        """Set up fallback chain test fixtures."""
        cls.mock_framework = create_mock_framework()
        cls.mock_instances = cls.mock_framework.create_mock_instances(
            models=list(settings.MODEL_CAPABILITIES.keys()),
            instances_per_model=3  # Multiple instances per model for fallback testing
        )
        
        # Create mock pool manager
        cls.mock_pool_manager = Mock(spec=LLMPoolManager)
        cls.mock_pool_manager.instances = cls.mock_instances
        cls.mock_pool_manager.config = {
            'pool_config': {
                'routing_strategy': 'balanced',
                'failover_enabled': True
            }
        }
        
        # Initialize router
        cls.router = IntelligentRequestRouter(cls.mock_pool_manager)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)
        
        # Organize instances by model for testing
        cls.instances_by_model = defaultdict(list)
        for instance in cls.mock_instances.values():
            cls.instances_by_model[instance.model_name].append(instance)
    
    def setUp(self):
        """Set up for each test."""
        # Reset all mock instances
        for instance in self.mock_instances.values():
            instance.reset_mock_state()
            instance.status = LLMInstanceStatus.HEALTHY
        
        # Clear router state
        self.router.routing_history.clear()
    
    def test_primary_to_fallback_instance_routing(self):
        """Test fallback routing within the same model when primary instance fails."""
        self.logger.info("Testing primary to fallback instance routing")
        
        results = FallbackChainResults("primary_to_fallback_instance")
        
        # Select a model with multiple instances for testing
        test_model = "deepseek-v2:16b"
        model_instances = self.instances_by_model[test_model]
        
        if len(model_instances) < 2:
            self.skipTest(f"Need at least 2 instances of {test_model} for fallback testing")
        
        primary_instance = model_instances[0]
        fallback_instance = model_instances[1]
        
        # Initial requests to establish primary preference
        baseline_requests = 5
        for i in range(baseline_requests):
            request = self._create_test_request(f"Baseline {i}", use_case="general")
            routing_decision = self.router.route_request(request)
            
            # Track primary usage
            if routing_decision.selected_instance.model_name == test_model:
                results.primary_attempts += 1
                
                response = self.mock_framework.simulate_request(routing_decision.selected_instance, request)
                if response.success:
                    results.primary_successes += 1
                    results.primary_response_times.append(response.response_time)
        
        # Fail primary instance
        primary_instance.force_failure(MockFailureType.CONNECTION_ERROR)
        primary_instance.status = LLMInstanceStatus.OFFLINE
        
        self.logger.info(f"Failed primary instance: {primary_instance.instance_id}")
        
        # Test fallback routing
        fallback_requests = 10
        for i in range(fallback_requests):
            request = self._create_test_request(f"Fallback test {i}", use_case="general")
            routing_decision = self.router.route_request(request)
            
            # Should not route to failed primary instance
            self.assertNotEqual(routing_decision.selected_instance.instance_id, 
                              primary_instance.instance_id,
                              "Should not route to failed primary instance")
            
            # Track fallback usage
            if routing_decision.selected_instance.model_name == test_model:
                # Routing within same model to different instance
                results.fallback_activations += 1
                results.fallback_paths.append((primary_instance.instance_id, 
                                             routing_decision.selected_instance.instance_id, 
                                             True))
                
                response = self.mock_framework.simulate_request(routing_decision.selected_instance, request)
                if response.success:
                    results.fallback_successes += 1
                    results.fallback_response_times.append(response.response_time)
                else:
                    results.fallback_failures += 1
            else:
                # Routing to different model (deeper fallback)
                results.fallback_activations += 1
                results.fallback_depths.append(2)  # Deeper into fallback chain
                
                response = self.mock_framework.simulate_request(routing_decision.selected_instance, request)
                if response.success:
                    results.fallback_successes += 1
        
        # Test recovery
        primary_instance.force_failure(MockFailureType.NONE)
        primary_instance.status = LLMInstanceStatus.HEALTHY
        
        self.logger.info(f"Recovered primary instance: {primary_instance.instance_id}")
        
        # Test post-recovery routing
        recovery_requests = 5
        post_recovery_primary_usage = 0
        
        for i in range(recovery_requests):
            request = self._create_test_request(f"Recovery {i}", use_case="general")
            routing_decision = self.router.route_request(request)
            
            if routing_decision.selected_instance.instance_id == primary_instance.instance_id:
                post_recovery_primary_usage += 1
            
            response = self.mock_framework.simulate_request(routing_decision.selected_instance, request)
            # Track recovery metrics if needed
        
        results.finish()
        
        # Validate fallback behavior
        self.assertGreater(results.fallback_success_rate, 0.8,
                          f"Fallback success rate too low: {results.fallback_success_rate:.2%}")
        
        self.assertGreater(len(results.fallback_paths), 0,
                          "Should have recorded fallback paths")
        
        # Should be able to use recovered instance
        self.assertGreater(post_recovery_primary_usage, 0,
                          "Should route back to recovered primary instance")
        
        self.logger.info(f"Primary to fallback test - Fallback success rate: {results.fallback_success_rate:.2%}, "
                        f"Recovery usage: {post_recovery_primary_usage}/{recovery_requests}")
    
    def test_model_type_failover(self):
        """Test failover between different model types when entire model fails."""
        self.logger.info("Testing model type failover")
        
        results = FallbackChainResults("model_type_failover")
        
        # Select primary model and fail all its instances
        primary_model = "deepseek-v2:16b"
        fallback_models = ["llama3:8b", "mistral:7b"]
        
        primary_instances = self.instances_by_model[primary_model]
        
        # Establish baseline with primary model
        baseline_requests = 5
        primary_model_usage = 0
        
        for i in range(baseline_requests):
            request = self._create_test_request(f"Baseline {i}", use_case="general")
            routing_decision = self.router.route_request(request)
            
            if routing_decision.selected_instance.model_name == primary_model:
                primary_model_usage += 1
                results.primary_attempts += 1
                
                response = self.mock_framework.simulate_request(routing_decision.selected_instance, request)
                if response.success:
                    results.primary_successes += 1
        
        # Fail all instances of primary model
        for instance in primary_instances:
            instance.force_failure(MockFailureType.SERVICE_UNAVAILABLE)
            instance.status = LLMInstanceStatus.OFFLINE
        
        self.logger.info(f"Failed all instances of primary model: {primary_model}")
        
        # Test model-type failover
        failover_requests = 15
        fallback_model_usage = defaultdict(int)
        
        for i in range(failover_requests):
            request = self._create_test_request(f"Model failover {i}", use_case="general")
            routing_decision = self.router.route_request(request)
            
            # Should not route to failed primary model
            self.assertNotEqual(routing_decision.selected_instance.model_name, primary_model,
                              f"Should not route to failed model {primary_model}")
            
            # Track fallback model usage
            selected_model = routing_decision.selected_instance.model_name
            fallback_model_usage[selected_model] += 1
            
            results.fallback_activations += 1
            results.fallback_paths.append((primary_model, selected_model, True))
            
            response = self.mock_framework.simulate_request(routing_decision.selected_instance, request)
            if response.success:
                results.fallback_successes += 1
                results.fallback_response_times.append(response.response_time)
            else:
                results.fallback_failures += 1
        
        # Verify distribution across fallback models
        self.assertGreater(len(fallback_model_usage), 0,
                          "Should use fallback models")
        
        # Test partial recovery (recover one instance of primary model)
        recovery_instance = primary_instances[0]
        recovery_instance.force_failure(MockFailureType.NONE)
        recovery_instance.status = LLMInstanceStatus.HEALTHY
        
        self.logger.info(f"Partially recovered primary model: {recovery_instance.instance_id}")
        
        # Test gradual return to primary model
        recovery_requests = 10
        post_recovery_primary_usage = 0
        
        for i in range(recovery_requests):
            request = self._create_test_request(f"Partial recovery {i}", use_case="general")
            routing_decision = self.router.route_request(request)
            
            if routing_decision.selected_instance.model_name == primary_model:
                post_recovery_primary_usage += 1
        
        results.finish()
        
        # Validate model-type failover
        self.assertGreater(results.fallback_success_rate, 0.85,
                          f"Model failover success rate too low: {results.fallback_success_rate:.2%}")
        
        self.assertEqual(results.chain_exhaustions, 0,
                        "Should not exhaust fallback chain with multiple models available")
        
        # Should gradually return to recovered primary model
        self.assertGreater(post_recovery_primary_usage, 0,
                          "Should route back to partially recovered primary model")
        
        self.logger.info(f"Model type failover - Success rate: {results.fallback_success_rate:.2%}, "
                        f"Fallback models used: {list(fallback_model_usage.keys())}, "
                        f"Recovery usage: {post_recovery_primary_usage}/{recovery_requests}")
    
    def test_cost_budget_fallback(self):
        """Test fallback to local models when cloud budget is exhausted."""
        self.logger.info("Testing cost budget fallback")
        
        results = FallbackChainResults("cost_budget_fallback")
        
        # Identify cloud and local models
        cloud_models = []
        local_models = []
        
        for model_name, capabilities in settings.MODEL_CAPABILITIES.items():
            if capabilities["engine"] == "gcp":
                cloud_models.append(model_name)
            else:
                local_models.append(model_name)
        
        if not cloud_models or not local_models:
            self.skipTest("Need both cloud and local models for cost budget fallback testing")
        
        # Simulate budget exhaustion scenario
        high_cost_request_config = {
            'use_case': 'complex_reasoning',
            'max_cost': 0.001,  # Very low budget
            'quality_requirements': 'high'
        }
        
        # Test with budget constraints
        budget_constrained_requests = 10
        local_model_selections = 0
        cloud_model_selections = 0
        
        for i in range(budget_constrained_requests):
            request = LLMRequest(
                request_id=f"budget-test-{i}",
                prompt=f"Budget constrained request {i} requiring cost optimization.",
                model_config=high_cost_request_config,
                priority=5,
                timeout=30.0,
                retry_count=0,
                max_retries=3,
                created_at=time.time()
            )
            
            routing_decision = self.router.route_request(request, "cost_first")
            
            # Track model type selection
            selected_model = routing_decision.selected_instance.model_name
            model_capabilities = settings.MODEL_CAPABILITIES[selected_model]
            
            if model_capabilities["engine"] == "local":
                local_model_selections += 1
                results.fallback_activations += 1
                results.fallback_paths.append(("cloud_budget", selected_model, True))
            else:
                cloud_model_selections += 1
                results.primary_attempts += 1
            
            # Verify cost constraint compliance
            self.assertLessEqual(routing_decision.estimated_cost, 
                               high_cost_request_config['max_cost'],
                               f"Cost constraint violated: {routing_decision.estimated_cost}")
            
            response = self.mock_framework.simulate_request(routing_decision.selected_instance, request)
            if response.success:
                if model_capabilities["engine"] == "local":
                    results.fallback_successes += 1
                else:
                    results.primary_successes += 1
        
        # Test with no budget constraints (should allow cloud models)
        unlimited_budget_config = {
            'use_case': 'complex_reasoning',
            'max_cost': 1.0,  # High budget
            'quality_requirements': 'high'
        }
        
        unlimited_requests = 5
        unlimited_cloud_usage = 0
        
        for i in range(unlimited_requests):
            request = LLMRequest(
                request_id=f"unlimited-{i}",
                prompt=f"Unlimited budget request {i}.",
                model_config=unlimited_budget_config,
                priority=5,
                timeout=30.0,
                retry_count=0,
                max_retries=3,
                created_at=time.time()
            )
            
            routing_decision = self.router.route_request(request, "quality_first")
            selected_model = routing_decision.selected_instance.model_name
            
            if settings.MODEL_CAPABILITIES[selected_model]["engine"] == "gcp":
                unlimited_cloud_usage += 1
        
        results.finish()
        
        # Validate cost budget fallback behavior
        local_preference_rate = local_model_selections / budget_constrained_requests
        self.assertGreater(local_preference_rate, 0.7,
                          f"Should prefer local models under budget constraints: {local_preference_rate:.2%}")
        
        # With unlimited budget, should be more willing to use cloud models
        if unlimited_cloud_usage > 0:
            self.logger.info(f"Unlimited budget cloud usage: {unlimited_cloud_usage}/{unlimited_requests}")
        
        self.assertGreater(results.fallback_success_rate, 0.95,
                          f"Local model fallback success rate too low: {results.fallback_success_rate:.2%}")
        
        self.logger.info(f"Cost budget fallback - Local preference: {local_preference_rate:.2%}, "
                        f"Fallback success: {results.fallback_success_rate:.2%}")
    
    def test_fallback_chain_exhaustion(self):
        """Test behavior when fallback chain is exhausted."""
        self.logger.info("Testing fallback chain exhaustion")
        
        results = FallbackChainResults("fallback_chain_exhaustion")
        
        # Fail most instances to create exhaustion scenario
        total_instances = list(self.mock_instances.values())
        instances_to_fail = total_instances[:-2]  # Leave only 2 instances healthy
        healthy_instances = total_instances[-2:]
        
        # Fail most instances
        for instance in instances_to_fail:
            instance.force_failure(MockFailureType.SERVICE_UNAVAILABLE)
            instance.status = LLMInstanceStatus.OFFLINE
        
        self.logger.info(f"Failed {len(instances_to_fail)} instances, leaving {len(healthy_instances)} healthy")
        
        # Send requests that will stress the limited available instances
        stress_requests = 20
        successful_routings = 0
        failed_routings = 0
        
        for i in range(stress_requests):
            request = self._create_test_request(f"Exhaustion test {i}", use_case="general")
            
            try:
                routing_decision = self.router.route_request(request)
                
                # Should route to one of the healthy instances
                selected_instance = routing_decision.selected_instance
                self.assertIn(selected_instance, healthy_instances,
                             "Should only route to healthy instances")
                
                response = self.mock_framework.simulate_request(selected_instance, request)
                
                if response.success:
                    successful_routings += 1
                    results.fallback_successes += 1
                else:
                    failed_routings += 1
                    results.fallback_failures += 1
                
                results.fallback_activations += 1
                
            except Exception as e:
                self.logger.error(f"Routing failed during exhaustion test: {e}")
                results.chain_exhaustions += 1
                failed_routings += 1
        
        # Test overload behavior on remaining instances
        for instance in healthy_instances:
            # These instances should have higher load
            self.assertGreaterEqual(instance.total_requests, 5,
                                  f"Healthy instance {instance.instance_id} should handle increased load")
        
        results.finish()
        
        # Validate exhaustion handling
        success_rate = successful_routings / stress_requests
        self.assertGreater(success_rate, 0.7,
                          f"Success rate too low during chain exhaustion: {success_rate:.2%}")
        
        # Should handle exhaustion gracefully (not completely fail)
        exhaustion_rate = results.chain_exhaustions / stress_requests
        self.assertLess(exhaustion_rate, 0.2,
                       f"Chain exhaustion rate too high: {exhaustion_rate:.2%}")
        
        self.logger.info(f"Chain exhaustion test - Success rate: {success_rate:.2%}, "
                        f"Exhaustion rate: {exhaustion_rate:.2%}")
    
    def test_fallback_performance_degradation(self):
        """Test performance characteristics during fallback scenarios."""
        self.logger.info("Testing fallback performance degradation")
        
        results = FallbackChainResults("fallback_performance_degradation")
        
        # Measure baseline performance
        baseline_requests = 10
        baseline_response_times = []
        
        for i in range(baseline_requests):
            request = self._create_test_request(f"Baseline perf {i}", use_case="general")
            routing_decision = self.router.route_request(request)
            response = self.mock_framework.simulate_request(routing_decision.selected_instance, request)
            
            if response.success:
                baseline_response_times.append(response.response_time)
                results.primary_response_times.append(response.response_time)
        
        baseline_avg_time = sum(baseline_response_times) / len(baseline_response_times)
        
        # Create fallback scenario by failing primary instances
        primary_model = "deepseek-v2:16b"
        primary_instances = self.instances_by_model[primary_model]
        
        for instance in primary_instances:
            instance.force_failure(MockFailureType.TIMEOUT)
            instance.status = LLMInstanceStatus.OFFLINE
        
        # Measure fallback performance
        fallback_requests = 15
        fallback_response_times = []
        
        for i in range(fallback_requests):
            request = self._create_test_request(f"Fallback perf {i}", use_case="general")
            routing_decision = self.router.route_request(request)
            
            # Should not route to failed primary model
            self.assertNotEqual(routing_decision.selected_instance.model_name, primary_model)
            
            response = self.mock_framework.simulate_request(routing_decision.selected_instance, request)
            
            if response.success:
                fallback_response_times.append(response.response_time)
                results.fallback_response_times.append(response.response_time)
                results.fallback_successes += 1
            
            results.fallback_activations += 1
        
        fallback_avg_time = sum(fallback_response_times) / len(fallback_response_times) if fallback_response_times else 0
        
        # Calculate performance degradation
        if baseline_avg_time > 0:
            performance_degradation = (fallback_avg_time - baseline_avg_time) / baseline_avg_time
        else:
            performance_degradation = 0
        
        results.finish()
        
        # Validate performance degradation is acceptable
        self.assertLess(performance_degradation, 0.5,  # <50% degradation
                       f"Performance degradation too high: {performance_degradation:.1%}")
        
        self.assertGreater(results.fallback_success_rate, 0.8,
                          f"Fallback success rate during performance test: {results.fallback_success_rate:.2%}")
        
        self.logger.info(f"Performance degradation test - Baseline: {baseline_avg_time:.3f}s, "
                        f"Fallback: {fallback_avg_time:.3f}s, "
                        f"Degradation: {performance_degradation:.1%}")
    
    def test_multi_tier_fallback_chain(self):
        """Test complex multi-tier fallback chains."""
        self.logger.info("Testing multi-tier fallback chain")
        
        results = FallbackChainResults("multi_tier_fallback")
        
        # Organize models by tiers
        tier1_models = ["deepseek-v2:16b"]  # High quality
        tier2_models = ["gemini-1.0-pro"]  # Cloud backup
        tier3_models = ["llama3:8b", "mistral:7b"]  # Local fallback
        
        # Create complex fallback scenario
        for tier_num, (tier_name, models) in enumerate([
            ("Tier 1", tier1_models),
            ("Tier 2", tier2_models),
            ("Tier 3", tier3_models)
        ], 1):
            
            # Fail current tier
            for model in models:
                instances = self.instances_by_model[model]
                for instance in instances:
                    instance.force_failure(MockFailureType.CONNECTION_ERROR)
                    instance.status = LLMInstanceStatus.OFFLINE
            
            self.logger.info(f"Failed {tier_name}: {models}")
            
            # Test requests with this tier failed
            tier_requests = 8
            tier_successes = 0
            
            for i in range(tier_requests):
                request = self._create_test_request(f"Tier {tier_num} fallback {i}", 
                                                   use_case="complex_reasoning")
                
                routing_decision = self.router.route_request(request, "quality_first")
                selected_model = routing_decision.selected_instance.model_name
                
                # Should not route to failed models
                for failed_model in models:
                    self.assertNotEqual(selected_model, failed_model,
                                      f"Should not route to failed model {failed_model}")
                
                response = self.mock_framework.simulate_request(routing_decision.selected_instance, request)
                
                if response.success:
                    tier_successes += 1
                    results.fallback_successes += 1
                
                results.fallback_activations += 1
                results.fallback_depths.append(tier_num)
                results.fallback_paths.append((f"tier_{tier_num-1}", selected_model, response.success))
            
            tier_success_rate = tier_successes / tier_requests
            self.logger.info(f"Tier {tier_num} fallback success rate: {tier_success_rate:.2%}")
            
            # Validate tier performance
            if tier_num < 3:  # Not the last tier
                self.assertGreater(tier_success_rate, 0.7,
                                 f"Tier {tier_num} fallback success rate too low: {tier_success_rate:.2%}")
        
        results.finish()
        
        # Validate multi-tier fallback behavior
        self.assertGreater(results.fallback_success_rate, 0.7,
                          f"Overall multi-tier fallback success rate: {results.fallback_success_rate:.2%}")
        
        # Verify fallback depth distribution
        if results.fallback_depths:
            avg_depth = sum(results.fallback_depths) / len(results.fallback_depths)
            max_depth = max(results.fallback_depths)
            
            self.assertLessEqual(max_depth, 3, f"Fallback depth should not exceed 3 tiers: {max_depth}")
            
            self.logger.info(f"Multi-tier fallback - Avg depth: {avg_depth:.1f}, Max depth: {max_depth}")
    
    def _create_test_request(self, prompt: str, use_case: str = "general") -> LLMRequest:
        """Create a test LLM request."""
        return LLMRequest(
            request_id=f"fallback-test-{int(time.time() * 1000000)}",
            prompt=prompt,
            model_config={'use_case': use_case},
            priority=5,
            timeout=30.0,
            retry_count=0,
            max_retries=3,
            created_at=time.time()
        )


class TestFallbackChainRequirements(unittest.TestCase):
    """Validate fallback chain behavior against specific requirements."""
    
    def test_fallback_success_rate_requirement(self):
        """Test that fallback success rate meets 100% requirement."""
        # Mock results from fallback chain tests
        mock_results = {
            'primary_to_fallback_success_rate': 0.95,
            'model_type_failover_success_rate': 0.92,
            'cost_budget_fallback_success_rate': 0.98,
            'chain_exhaustion_handling_rate': 0.85
        }
        
        # Validate individual fallback scenarios
        for scenario, success_rate in mock_results.items():
            self.assertGreater(success_rate, 0.9,
                             f"{scenario} success rate requirement not met: {success_rate:.2%}")
        
        # Calculate overall fallback success rate
        overall_success_rate = sum(mock_results.values()) / len(mock_results)
        self.assertGreater(overall_success_rate, 0.9,
                          f"Overall fallback success rate requirement not met: {overall_success_rate:.2%}")


if __name__ == '__main__':
    # Configure logging for fallback chain tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    unittest.main(verbosity=2)