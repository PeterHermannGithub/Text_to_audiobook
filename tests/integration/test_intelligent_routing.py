"""
Intelligent Routing Strategy Validation Suite.

This module provides comprehensive testing of all routing strategies in the
multi-model load balancing system, validating routing decisions, performance,
and cost optimization behaviors.
"""

import unittest
import time
import statistics
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock, patch
import json
import logging

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.llm_pool.intelligent_router import (
    IntelligentRequestRouter, RoutingStrategy, RequestComplexity,
    RequestCharacteristics, RoutingDecision
)
from src.llm_pool.llm_pool_manager import LLMPoolManager, LLMRequest
from tests.utils.mock_llm_framework import (
    MockLLMFramework, MockLLMInstance, create_mock_framework,
    COMMON_TEST_SCENARIOS
)
from config import settings


class TestIntelligentRouting(unittest.TestCase):
    """Test suite for intelligent request routing functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for all routing tests."""
        cls.mock_framework = create_mock_framework()
        cls.mock_instances = cls.mock_framework.create_mock_instances(
            models=list(settings.MODEL_CAPABILITIES.keys()),
            instances_per_model=2
        )
        
        # Create mock pool manager
        cls.mock_pool_manager = Mock(spec=LLMPoolManager)
        cls.mock_pool_manager.instances = cls.mock_instances
        cls.mock_pool_manager.config = {
            'pool_config': {
                'routing_strategy': 'balanced'
            }
        }
        
        # Initialize intelligent router
        cls.router = IntelligentRequestRouter(cls.mock_pool_manager)
        
        # Set up logging
        logging.basicConfig(level=logging.DEBUG)
        cls.logger = logging.getLogger(__name__)
    
    def setUp(self):
        """Set up for each test."""
        # Reset mock framework state
        for instance in self.mock_instances.values():
            instance.reset_mock_state()
        
        # Clear router history
        self.router.routing_history.clear()
        self.router.model_performance_cache.clear()
        self.router.request_patterns.clear()
    
    def test_speed_first_routing_strategy(self):
        """Test speed_first routing strategy prioritizes fastest models."""
        self.logger.info("Testing speed_first routing strategy")
        
        # Create request with speed requirements
        request = self._create_test_request(
            prompt="Quick classification needed urgently.",
            urgency="critical",
            use_case="classification"
        )
        
        # Route with speed_first strategy
        routing_decision = self.router.route_request(request, "speed_first")
        
        # Validate routing decision
        self.assertIsInstance(routing_decision, RoutingDecision)
        self.assertEqual(routing_decision.routing_strategy_used, "speed_first")
        self.assertGreater(routing_decision.confidence_score, 0.5)
        
        # Verify speed optimization in reasoning
        speed_keywords = ["fast", "speed", "quick", "critical"]
        reasoning_text = " ".join(routing_decision.reasoning).lower()
        self.assertTrue(any(keyword in reasoning_text for keyword in speed_keywords),
                       f"Speed reasoning not found in: {routing_decision.reasoning}")
        
        # Verify selected model has good speed characteristics
        selected_instance = routing_decision.selected_instance
        self.assertGreater(selected_instance.speed_score, 0.6,
                          f"Selected model {selected_instance.model_name} has low speed score: {selected_instance.speed_score}")
        
        # Test routing overhead
        start_time = time.time()
        for _ in range(10):
            self.router.route_request(request, "speed_first")
        avg_routing_time = (time.time() - start_time) / 10
        
        self.assertLess(avg_routing_time, 0.01,  # <10ms routing overhead
                       f"Routing overhead too high: {avg_routing_time*1000:.2f}ms")
        
        self.logger.info(f"Speed_first routing completed in {avg_routing_time*1000:.2f}ms avg")
    
    def test_quality_first_routing_strategy(self):
        """Test quality_first routing strategy prioritizes highest quality models."""
        self.logger.info("Testing quality_first routing strategy")
        
        # Create request requiring high quality
        request = self._create_test_request(
            prompt="Analyze this complex narrative passage for deep character insights...",
            quality_requirements="high",
            use_case="analysis"
        )
        
        # Route with quality_first strategy
        routing_decision = self.router.route_request(request, "quality_first")
        
        # Validate routing decision
        self.assertEqual(routing_decision.routing_strategy_used, "quality_first")
        self.assertGreater(routing_decision.confidence_score, 0.6)
        
        # Verify quality optimization in reasoning
        quality_keywords = ["quality", "high", "analysis", "complex"]
        reasoning_text = " ".join(routing_decision.reasoning).lower()
        self.assertTrue(any(keyword in reasoning_text for keyword in quality_keywords),
                       f"Quality reasoning not found in: {routing_decision.reasoning}")
        
        # Verify selected model has good quality characteristics
        selected_instance = routing_decision.selected_instance
        self.assertGreater(selected_instance.quality_score, 0.7,
                          f"Selected model {selected_instance.model_name} has low quality score: {selected_instance.quality_score}")
        
        # Verify fallback chain includes high-quality alternatives
        for fallback_id in routing_decision.fallback_chain:
            if fallback_id in self.mock_instances:
                fallback_instance = self.mock_instances[fallback_id]
                self.assertGreater(fallback_instance.quality_score, 0.6,
                                 f"Fallback {fallback_instance.model_name} has insufficient quality")
        
        self.logger.info(f"Quality_first selected {selected_instance.model_name} with quality score {selected_instance.quality_score}")
    
    def test_cost_first_routing_strategy(self):
        """Test cost_first routing strategy prioritizes cost optimization."""
        self.logger.info("Testing cost_first routing strategy")
        
        # Create request with cost constraints
        request = self._create_test_request(
            prompt="Simple classification task.",
            cost_constraints=0.001,  # Low cost budget
            use_case="classification"
        )
        
        # Route with cost_first strategy
        routing_decision = self.router.route_request(request, "cost_first")
        
        # Validate routing decision
        self.assertEqual(routing_decision.routing_strategy_used, "cost_first")
        self.assertGreater(routing_decision.confidence_score, 0.5)
        
        # Verify cost optimization
        self.assertLessEqual(routing_decision.estimated_cost, request.model_config['max_cost'],
                           f"Estimated cost {routing_decision.estimated_cost} exceeds budget {request.model_config['max_cost']}")
        
        # Verify preference for local models (zero cost)
        selected_instance = routing_decision.selected_instance
        if selected_instance.engine_type == "local":
            self.assertEqual(routing_decision.estimated_cost, 0.0,
                           "Local model should have zero cost")
        
        # Verify cost-aware reasoning
        cost_keywords = ["cost", "local", "free", "budget", "efficient"]
        reasoning_text = " ".join(routing_decision.reasoning).lower()
        self.assertTrue(any(keyword in reasoning_text for keyword in cost_keywords),
                       f"Cost reasoning not found in: {routing_decision.reasoning}")
        
        self.logger.info(f"Cost_first selected {selected_instance.model_name} with estimated cost ${routing_decision.estimated_cost:.4f}")
    
    def test_balanced_routing_strategy(self):
        """Test balanced routing strategy considers all factors appropriately."""
        self.logger.info("Testing balanced routing strategy")
        
        # Create request with balanced requirements
        request = self._create_test_request(
            prompt="Medium complexity speaker attribution task.",
            use_case="speaker_attribution"
        )
        
        # Route with balanced strategy
        routing_decision = self.router.route_request(request, "balanced")
        
        # Validate routing decision
        self.assertEqual(routing_decision.routing_strategy_used, "balanced")
        self.assertGreater(routing_decision.confidence_score, 0.5)
        
        # Verify balanced considerations in reasoning
        selected_instance = routing_decision.selected_instance
        
        # Should have reasonable scores across all dimensions
        self.assertGreater(selected_instance.health_score, 0.5)
        self.assertGreater(selected_instance.suitability_score, 0.5)
        
        # Verify reasoning mentions multiple factors
        reasoning_text = " ".join(routing_decision.reasoning).lower()
        factor_keywords = ["balanced", "health", "suitability", "performance"]
        matched_factors = sum(1 for keyword in factor_keywords if keyword in reasoning_text)
        self.assertGreater(matched_factors, 1,
                          f"Balanced reasoning should mention multiple factors: {routing_decision.reasoning}")
        
        self.logger.info(f"Balanced routing selected {selected_instance.model_name} with suitability score {selected_instance.suitability_score:.3f}")
    
    def test_adaptive_routing_strategy(self):
        """Test adaptive routing strategy adjusts based on system state."""
        self.logger.info("Testing adaptive routing strategy")
        
        # Create request
        request = self._create_test_request(
            prompt="Adaptive routing test request.",
            use_case="general"
        )
        
        # Test adaptive routing under normal load
        routing_decision_normal = self.router.route_request(request, "adaptive")
        self.assertEqual(routing_decision_normal.routing_strategy_used, "adaptive")
        
        # Simulate high load on all instances
        for instance in self.mock_instances.values():
            instance.current_load = instance.max_load - 1  # Near capacity
        
        # Test adaptive routing under high load
        routing_decision_loaded = self.router.route_request(request, "adaptive")
        
        # Verify adaptive behavior changes under load
        self.assertIsInstance(routing_decision_loaded, RoutingDecision)
        
        # Adaptive should prefer less loaded instances
        selected_instance = routing_decision_loaded.selected_instance
        self.assertLess(selected_instance.current_load, selected_instance.max_load,
                       "Adaptive routing should select available instances")
        
        self.logger.info(f"Adaptive routing adapted to load conditions, selected {selected_instance.model_name}")
    
    def test_routing_strategy_weighting_validation(self):
        """Test that routing strategies apply correct weighting factors."""
        self.logger.info("Testing routing strategy weighting validation")
        
        # Create test request
        request = self._create_test_request(
            prompt="Test weighting validation.",
            use_case="testing"
        )
        
        # Test speed_first weighting (should emphasize speed_score)
        with patch.object(self.router, '_apply_strategy_weighting') as mock_weighting:
            mock_weighting.return_value = 0.85  # Mock return value
            
            self.router.route_request(request, "speed_first")
            
            # Verify weighting was called
            self.assertTrue(mock_weighting.called)
            
            # Get the call arguments
            call_args = mock_weighting.call_args
            scores, strategy = call_args[0]
            
            # Verify correct strategy was passed
            self.assertEqual(strategy, RoutingStrategy.SPEED_FIRST)
            
            # Verify scores structure
            expected_score_keys = ['health_score', 'suitability_score', 'use_case_score', 
                                 'cost_score', 'quality_score', 'speed_score', 
                                 'capacity_score', 'json_score']
            for key in expected_score_keys:
                self.assertIn(key, scores, f"Missing score key: {key}")
        
        # Test actual weighting calculation
        test_scores = {
            'speed_score': 0.9,
            'quality_score': 0.7,
            'cost_score': 0.5,
            'health_score': 0.8,
            'capacity_score': 0.6,
            'use_case_score': 0.7,
            'suitability_score': 0.75,
            'json_score': 1.0
        }
        
        # Calculate speed_first weighting
        speed_first_score = self.router._apply_strategy_weighting(test_scores, RoutingStrategy.SPEED_FIRST)
        
        # Verify speed has high influence (should be heavily weighted)
        self.assertGreater(speed_first_score, 0.7,
                          f"Speed_first score {speed_first_score} should reflect high speed weight")
        
        # Calculate quality_first weighting 
        quality_first_score = self.router._apply_strategy_weighting(test_scores, RoutingStrategy.QUALITY_FIRST)
        
        # Verify quality has high influence
        self.assertGreater(quality_first_score, 0.7,
                          f"Quality_first score {quality_first_score} should reflect high quality weight")
        
        self.logger.info(f"Weighting validation: speed_first={speed_first_score:.3f}, quality_first={quality_first_score:.3f}")
    
    def test_request_complexity_analysis(self):
        """Test request complexity analysis accuracy."""
        self.logger.info("Testing request complexity analysis")
        
        # Test simple request
        simple_request = self._create_test_request(
            prompt="Yes or no?",
            use_case="simple_classification"
        )
        simple_characteristics = self.router._analyze_request_characteristics(simple_request)
        self.assertEqual(simple_characteristics.complexity, RequestComplexity.SIMPLE)
        
        # Test medium request
        medium_request = self._create_test_request(
            prompt="Analyze the speaker attribution for this dialogue: " + "word " * 100,
            use_case="speaker_attribution"
        )
        medium_characteristics = self.router._analyze_request_characteristics(medium_request)
        self.assertIn(medium_characteristics.complexity, [RequestComplexity.MEDIUM, RequestComplexity.COMPLEX])
        
        # Test complex request
        complex_request = self._create_test_request(
            prompt="Perform detailed analysis of narrative structure: " + "complex analysis " * 200,
            use_case="complex_reasoning"
        )
        complex_characteristics = self.router._analyze_request_characteristics(complex_request)
        self.assertIn(complex_characteristics.complexity, [RequestComplexity.COMPLEX, RequestComplexity.BATCH])
        
        # Test batch request
        batch_request = self._create_test_request(
            prompt="Process multiple documents: " + "document content " * 500,
            use_case="batch_processing"
        )
        batch_characteristics = self.router._analyze_request_characteristics(batch_request)
        self.assertIn(batch_characteristics.complexity, [RequestComplexity.BATCH, RequestComplexity.HEAVY])
        
        self.logger.info(f"Complexity analysis: simple={simple_characteristics.complexity.value}, "
                        f"medium={medium_characteristics.complexity.value}, "
                        f"complex={complex_characteristics.complexity.value}, "
                        f"batch={batch_characteristics.complexity.value}")
    
    def test_fallback_chain_generation(self):
        """Test fallback chain generation for routing decisions."""
        self.logger.info("Testing fallback chain generation")
        
        # Create request
        request = self._create_test_request(
            prompt="Test fallback chain generation.",
            use_case="testing"
        )
        
        # Route request
        routing_decision = self.router.route_request(request, "balanced")
        
        # Validate fallback chain
        self.assertIsInstance(routing_decision.fallback_chain, list)
        self.assertLessEqual(len(routing_decision.fallback_chain), 3,
                           "Fallback chain should have at most 3 alternatives")
        
        # Verify fallback instances exist and are different from primary
        primary_id = routing_decision.selected_instance.instance_id
        for fallback_id in routing_decision.fallback_chain:
            self.assertIn(fallback_id, self.mock_instances,
                         f"Fallback instance {fallback_id} not found in pool")
            self.assertNotEqual(fallback_id, primary_id,
                              "Fallback should be different from primary instance")
        
        # Verify fallback instances are available
        for fallback_id in routing_decision.fallback_chain:
            fallback_instance = self.mock_instances[fallback_id]
            self.assertTrue(fallback_instance.is_available(),
                          f"Fallback instance {fallback_id} is not available")
        
        self.logger.info(f"Fallback chain generated: {len(routing_decision.fallback_chain)} alternatives")
    
    def test_routing_performance_overhead(self):
        """Test routing performance and overhead measurements."""
        self.logger.info("Testing routing performance overhead")
        
        # Prepare multiple test requests
        test_requests = [
            self._create_test_request(f"Test request {i}", "general") 
            for i in range(50)
        ]
        
        # Measure routing overhead for each strategy
        strategy_performance = {}
        
        for strategy in ["speed_first", "quality_first", "cost_first", "balanced", "adaptive"]:
            start_time = time.time()
            
            for request in test_requests:
                routing_decision = self.router.route_request(request, strategy)
                self.assertIsInstance(routing_decision, RoutingDecision)
            
            total_time = time.time() - start_time
            avg_time_per_request = total_time / len(test_requests)
            
            strategy_performance[strategy] = {
                'total_time': total_time,
                'avg_time_per_request': avg_time_per_request,
                'requests_per_second': len(test_requests) / total_time
            }
            
            # Validate routing overhead is under 10ms per request
            self.assertLess(avg_time_per_request, 0.01,
                           f"{strategy} routing overhead {avg_time_per_request*1000:.2f}ms exceeds 10ms limit")
        
        # Log performance results
        for strategy, perf in strategy_performance.items():
            self.logger.info(f"{strategy}: {perf['avg_time_per_request']*1000:.2f}ms avg, "
                           f"{perf['requests_per_second']:.1f} req/s")
        
        # Verify consistent performance across strategies
        avg_times = [perf['avg_time_per_request'] for perf in strategy_performance.values()]
        performance_variance = statistics.stdev(avg_times) if len(avg_times) > 1 else 0
        self.assertLess(performance_variance, 0.005,
                       f"High variance in routing performance: {performance_variance*1000:.2f}ms std dev")
    
    def test_routing_analytics_collection(self):
        """Test routing analytics and pattern detection."""
        self.logger.info("Testing routing analytics collection")
        
        # Generate routing history
        test_patterns = [
            ("simple", "classification", "speed_first"),
            ("medium", "speaker_attribution", "balanced"),
            ("complex", "analysis", "quality_first"),
            ("simple", "extraction", "cost_first"),
            ("medium", "general", "adaptive")
        ]
        
        for complexity, use_case, strategy in test_patterns * 5:  # Repeat for history
            request = self._create_test_request(
                prompt=f"Test {complexity} {use_case} request.",
                use_case=use_case
            )
            routing_decision = self.router.route_request(request, strategy)
            self.assertIsInstance(routing_decision, RoutingDecision)
        
        # Get routing analytics
        analytics = self.router.get_routing_analytics()
        
        # Validate analytics structure
        self.assertIsInstance(analytics, dict)
        self.assertIn('total_requests', analytics)
        self.assertIn('model_usage_distribution', analytics)
        self.assertIn('strategy_usage_distribution', analytics)
        self.assertIn('complexity_distribution', analytics)
        self.assertIn('average_confidence_score', analytics)
        self.assertIn('most_common_patterns', analytics)
        
        # Validate data consistency
        self.assertEqual(analytics['total_requests'], len(test_patterns) * 5)
        self.assertGreater(analytics['average_confidence_score'], 0.0)
        self.assertLessEqual(analytics['average_confidence_score'], 1.0)
        
        # Verify model usage distribution
        model_usage = analytics['model_usage_distribution']
        self.assertIsInstance(model_usage, dict)
        self.assertGreater(len(model_usage), 0, "Should have model usage data")
        
        # Verify strategy distribution matches test patterns
        strategy_usage = analytics['strategy_usage_distribution']
        for _, _, strategy in test_patterns:
            self.assertIn(strategy, strategy_usage,
                         f"Strategy {strategy} should appear in usage distribution")
        
        self.logger.info(f"Analytics collected for {analytics['total_requests']} requests, "
                        f"avg confidence: {analytics['average_confidence_score']:.3f}")
    
    def test_routing_optimization_recommendations(self):
        """Test routing optimization analysis and recommendations."""
        self.logger.info("Testing routing optimization recommendations")
        
        # Generate diverse routing history
        for _ in range(20):
            request = self._create_test_request(
                prompt="Optimization test request.",
                use_case="testing"
            )
            routing_decision = self.router.route_request(request, "balanced")
            self.assertIsInstance(routing_decision, RoutingDecision)
        
        # Get optimization recommendations
        optimization = self.router.optimize_routing_strategy()
        
        # Validate optimization structure
        self.assertIsInstance(optimization, dict)
        self.assertIn('optimizations', optimization)
        self.assertIn('current_performance', optimization)
        
        # Validate current performance data
        current_perf = optimization['current_performance']
        self.assertIn('total_requests', current_perf)
        self.assertIn('average_routing_time_ms', current_perf)
        
        # Validate optimization recommendations
        optimizations = optimization['optimizations']
        self.assertIsInstance(optimizations, list)
        
        # If there are recommendations, validate structure
        for opt in optimizations:
            self.assertIsInstance(opt, str)
            self.assertGreater(len(opt), 10, "Optimization recommendation should be descriptive")
        
        self.logger.info(f"Optimization analysis generated {len(optimizations)} recommendations")
    
    def _create_test_request(self, prompt: str, use_case: str = "general", 
                           urgency: str = "medium", quality_requirements: str = "medium",
                           cost_constraints: float = float('inf')) -> LLMRequest:
        """Create a test LLM request with specified characteristics."""
        return LLMRequest(
            request_id=f"test-{int(time.time() * 1000000)}",
            prompt=prompt,
            model_config={
                'use_case': use_case,
                'quality_requirements': quality_requirements,
                'max_cost': cost_constraints,
                'urgency': urgency
            },
            priority=5,
            timeout=30.0,
            retry_count=0,
            max_retries=3,
            created_at=time.time()
        )


class TestRoutingStrategyWeighting(unittest.TestCase):
    """Detailed tests for routing strategy weighting algorithms."""
    
    def setUp(self):
        """Set up weighting tests."""
        self.mock_framework = create_mock_framework()
        self.mock_instances = self.mock_framework.create_mock_instances(
            models=["deepseek-v2:16b", "llama3:8b"],
            instances_per_model=1
        )
        
        mock_pool_manager = Mock(spec=LLMPoolManager)
        mock_pool_manager.instances = self.mock_instances
        mock_pool_manager.config = {'pool_config': {'routing_strategy': 'balanced'}}
        
        self.router = IntelligentRequestRouter(mock_pool_manager)
    
    def test_speed_first_weighting_coefficients(self):
        """Test speed_first strategy uses correct weighting coefficients."""
        test_scores = {
            'speed_score': 0.9,
            'quality_score': 0.7,
            'cost_score': 0.5,
            'health_score': 0.8,
            'capacity_score': 0.6,
            'use_case_score': 0.75,
            'suitability_score': 0.8,
            'json_score': 1.0
        }
        
        # Calculate weighted score
        final_score = self.router._apply_strategy_weighting(test_scores, RoutingStrategy.SPEED_FIRST)
        
        # Expected weights for speed_first: speed_score: 0.4, capacity_score: 0.2, health_score: 0.15
        expected_core_contribution = (0.9 * 0.4) + (0.6 * 0.2) + (0.8 * 0.15)
        
        # Should be dominated by speed score
        self.assertGreater(final_score, 0.7, "Speed_first should yield high score for fast model")
        self.assertLess(final_score, 1.0, "Score should be normalized")
        
        # Verify speed dominance by comparing with low-speed scenario
        low_speed_scores = test_scores.copy()
        low_speed_scores['speed_score'] = 0.3
        
        low_speed_final = self.router._apply_strategy_weighting(low_speed_scores, RoutingStrategy.SPEED_FIRST)
        self.assertLess(low_speed_final, final_score,
                       "Lower speed score should result in lower final score for speed_first")
    
    def test_quality_first_weighting_coefficients(self):
        """Test quality_first strategy uses correct weighting coefficients."""
        test_scores = {
            'speed_score': 0.5,
            'quality_score': 0.95,
            'cost_score': 0.6,
            'health_score': 0.8,
            'capacity_score': 0.7,
            'use_case_score': 0.9,
            'suitability_score': 0.8,
            'json_score': 1.0
        }
        
        # Calculate weighted score
        final_score = self.router._apply_strategy_weighting(test_scores, RoutingStrategy.QUALITY_FIRST)
        
        # Quality should dominate (weight: 0.4)
        self.assertGreater(final_score, 0.8, "Quality_first should yield high score for high-quality model")
        
        # Compare with low-quality scenario
        low_quality_scores = test_scores.copy()
        low_quality_scores['quality_score'] = 0.4
        
        low_quality_final = self.router._apply_strategy_weighting(low_quality_scores, RoutingStrategy.QUALITY_FIRST)
        self.assertLess(low_quality_final, final_score,
                       "Lower quality score should result in lower final score for quality_first")
    
    def test_cost_first_weighting_coefficients(self):
        """Test cost_first strategy uses correct weighting coefficients."""
        test_scores = {
            'speed_score': 0.6,
            'quality_score': 0.7,
            'cost_score': 0.95,  # High cost score (low actual cost)
            'health_score': 0.8,
            'capacity_score': 0.7,
            'use_case_score': 0.75,
            'suitability_score': 0.8,
            'json_score': 1.0
        }
        
        # Calculate weighted score
        final_score = self.router._apply_strategy_weighting(test_scores, RoutingStrategy.COST_FIRST)
        
        # Cost should dominate (weight: 0.5)
        self.assertGreater(final_score, 0.8, "Cost_first should yield high score for cost-effective model")
        
        # Compare with high-cost scenario
        high_cost_scores = test_scores.copy()
        high_cost_scores['cost_score'] = 0.2  # Low cost score (high actual cost)
        
        high_cost_final = self.router._apply_strategy_weighting(high_cost_scores, RoutingStrategy.COST_FIRST)
        self.assertLess(high_cost_final, final_score,
                       "Higher actual cost should result in lower final score for cost_first")
    
    def test_balanced_weighting_coefficients(self):
        """Test balanced strategy uses equal weighting across factors."""
        test_scores = {
            'speed_score': 0.8,
            'quality_score': 0.7,
            'cost_score': 0.6,
            'health_score': 0.9,
            'capacity_score': 0.8,
            'use_case_score': 0.85,
            'suitability_score': 0.75,
            'json_score': 1.0
        }
        
        # Calculate weighted score
        final_score = self.router._apply_strategy_weighting(test_scores, RoutingStrategy.BALANCED)
        
        # Should be close to average of major factors
        major_factors = [test_scores['health_score'], test_scores['suitability_score'], 
                        test_scores['use_case_score'], test_scores['quality_score']]
        avg_major = sum(major_factors) / len(major_factors)
        
        # Should be within reasonable range of average
        self.assertGreater(final_score, avg_major - 0.2)
        self.assertLess(final_score, avg_major + 0.2)


class TestRoutingConsistency(unittest.TestCase):
    """Test routing consistency and deterministic behavior."""
    
    def setUp(self):
        """Set up consistency tests."""
        self.mock_framework = create_mock_framework()
        self.mock_instances = self.mock_framework.create_mock_instances(
            models=list(settings.MODEL_CAPABILITIES.keys()),
            instances_per_model=1
        )
        
        mock_pool_manager = Mock(spec=LLMPoolManager)
        mock_pool_manager.instances = self.mock_instances
        mock_pool_manager.config = {'pool_config': {'routing_strategy': 'balanced'}}
        
        self.router = IntelligentRequestRouter(mock_pool_manager)
    
    def test_routing_determinism(self):
        """Test that identical requests produce consistent routing decisions."""
        # Create identical requests
        request1 = LLMRequest(
            request_id="test-1",
            prompt="Identical test prompt for determinism check.",
            model_config={'use_case': 'testing'},
            priority=5,
            timeout=30.0,
            retry_count=0,
            max_retries=3,
            created_at=time.time()
        )
        
        request2 = LLMRequest(
            request_id="test-2",
            prompt="Identical test prompt for determinism check.",
            model_config={'use_case': 'testing'},
            priority=5,
            timeout=30.0,
            retry_count=0,
            max_retries=3,
            created_at=time.time()
        )
        
        # Route both requests with same strategy
        decision1 = self.router.route_request(request1, "balanced")
        decision2 = self.router.route_request(request2, "balanced")
        
        # Should route to same model (unless load changes)
        self.assertEqual(decision1.selected_instance.model_name, 
                        decision2.selected_instance.model_name,
                        "Identical requests should route to same model")
        
        # Confidence scores should be very similar
        confidence_diff = abs(decision1.confidence_score - decision2.confidence_score)
        self.assertLess(confidence_diff, 0.1,
                       f"Confidence scores should be similar: {decision1.confidence_score} vs {decision2.confidence_score}")
    
    def test_routing_stability_under_load(self):
        """Test routing stability when system load changes."""
        # Create test request
        request = LLMRequest(
            request_id="load-test",
            prompt="Load stability test request.",
            model_config={'use_case': 'testing'},
            priority=5,
            timeout=30.0,
            retry_count=0,
            max_retries=3,
            created_at=time.time()
        )
        
        # Route under normal load
        normal_decision = self.router.route_request(request, "balanced")
        
        # Simulate load on selected instance
        selected_instance = normal_decision.selected_instance
        original_load = selected_instance.current_load
        selected_instance.current_load = selected_instance.max_load - 1
        
        # Route again under load
        loaded_decision = self.router.route_request(request, "balanced")
        
        # Should adapt to load or maintain stability
        self.assertIsInstance(loaded_decision, RoutingDecision)
        
        # If different instance selected, it should be available
        if loaded_decision.selected_instance.instance_id != selected_instance.instance_id:
            self.assertTrue(loaded_decision.selected_instance.is_available(),
                           "Should route to available instance under load")
        
        # Restore original load
        selected_instance.current_load = original_load


if __name__ == '__main__':
    # Configure logging for test runs
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    unittest.main(verbosity=2)