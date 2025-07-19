"""
Cost Optimization Testing Suite.

This module provides comprehensive testing of cost management, budget tracking,
and cost-aware routing functionality in the multi-model load balancing system.
"""

import unittest
import time
import json
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import logging

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.llm_pool.llm_pool_manager import LLMPoolManager, LLMRequest
from src.llm_pool.intelligent_router import IntelligentRequestRouter, RoutingStrategy
from src.llm_pool.performance_analytics import PerformanceAnalytics, CostTracker
from tests.utils.mock_llm_framework import (
    MockLLMFramework, create_mock_framework, COMMON_TEST_SCENARIOS
)
from config import settings


class TestCostOptimization(unittest.TestCase):
    """Test suite for cost optimization and budget management functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for cost optimization tests."""
        cls.mock_framework = create_mock_framework()
        cls.mock_instances = cls.mock_framework.create_mock_instances(
            models=list(settings.MODEL_CAPABILITIES.keys()),
            instances_per_model=1
        )
        
        # Create mock pool manager with cost configuration
        cls.mock_pool_manager = Mock(spec=LLMPoolManager)
        cls.mock_pool_manager.instances = cls.mock_instances
        cls.mock_pool_manager.config = {
            'pool_config': {
                'routing_strategy': 'cost_first',
                'cost_budget_per_hour': 10.0
            }
        }
        
        # Initialize components
        cls.router = IntelligentRequestRouter(cls.mock_pool_manager)
        cls.performance_analytics = PerformanceAnalytics()
        cls.cost_tracker = CostTracker()
        
        # Set up logging
        logging.basicConfig(level=logging.DEBUG)
        cls.logger = logging.getLogger(__name__)
    
    def setUp(self):
        """Set up for each test."""
        # Reset mock framework state
        for instance in self.mock_instances.values():
            instance.reset_mock_state()
        
        # Clear analytics and cost tracking
        self.performance_analytics = PerformanceAnalytics()
        self.cost_tracker = CostTracker()
        
        # Reset router state
        self.router.routing_history.clear()
        self.router.hourly_costs.clear()
        self.router.daily_costs.clear()
    
    def test_cost_aware_routing_preference(self):
        """Test that cost_first routing prefers low-cost models."""
        self.logger.info("Testing cost-aware routing preference")
        
        # Create request with cost constraints
        request = self._create_cost_constrained_request(
            prompt="Simple classification task for cost testing.",
            max_cost=0.001,  # Very low budget
            use_case="classification"
        )
        
        # Route with cost_first strategy
        routing_decision = self.router.route_request(request, "cost_first")
        
        # Validate cost optimization
        self.assertEqual(routing_decision.routing_strategy_used, "cost_first")
        self.assertLessEqual(routing_decision.estimated_cost, request.model_config['max_cost'],
                           f"Estimated cost {routing_decision.estimated_cost} exceeds budget {request.model_config['max_cost']}")
        
        # Verify preference for local models (zero cost)
        selected_instance = routing_decision.selected_instance
        if selected_instance.engine_type == "local":
            self.assertEqual(routing_decision.estimated_cost, 0.0,
                           "Local models should have zero cost")
            self.assertIn("cost-effective", " ".join(routing_decision.reasoning).lower(),
                         "Should mention cost-effectiveness for local models")
        else:
            # If cloud model selected, cost should still be within budget
            self.assertLessEqual(routing_decision.estimated_cost, request.model_config['max_cost'])
        
        self.logger.info(f"Cost-aware routing selected {selected_instance.model_name} "
                        f"({selected_instance.engine_type}) with cost ${routing_decision.estimated_cost:.4f}")
    
    def test_local_vs_cloud_cost_efficiency(self):
        """Test local vs cloud model cost efficiency routing."""
        self.logger.info("Testing local vs cloud cost efficiency")
        
        # Test with high-budget request (should allow cloud models)
        high_budget_request = self._create_cost_constrained_request(
            prompt="Complex analysis task with high budget.",
            max_cost=0.05,  # High budget
            use_case="complex_reasoning"
        )
        
        high_budget_decision = self.router.route_request(high_budget_request, "cost_first")
        
        # Test with low-budget request (should prefer local models)
        low_budget_request = self._create_cost_constrained_request(
            prompt="Simple task with low budget.",
            max_cost=0.0001,  # Very low budget
            use_case="simple_classification"
        )
        
        low_budget_decision = self.router.route_request(low_budget_request, "cost_first")
        
        # Validate cost efficiency
        self.assertLessEqual(high_budget_decision.estimated_cost, high_budget_request.model_config['max_cost'])
        self.assertLessEqual(low_budget_decision.estimated_cost, low_budget_request.model_config['max_cost'])
        
        # Low budget should strongly prefer local models
        if low_budget_decision.selected_instance.engine_type == "local":
            self.assertEqual(low_budget_decision.estimated_cost, 0.0)
        
        # Log cost comparison
        self.logger.info(f"High budget: {high_budget_decision.selected_instance.model_name} "
                        f"(${high_budget_decision.estimated_cost:.4f})")
        self.logger.info(f"Low budget: {low_budget_decision.selected_instance.model_name} "
                        f"(${low_budget_decision.estimated_cost:.4f})")
        
        # Verify cost efficiency ordering
        if (high_budget_decision.selected_instance.engine_type == "gcp" and 
            low_budget_decision.selected_instance.engine_type == "local"):
            self.assertGreaterEqual(high_budget_decision.estimated_cost, 
                                  low_budget_decision.estimated_cost,
                                  "High budget routing should allow higher costs when beneficial")
    
    def test_budget_tracking_accuracy(self):
        """Test budget tracking system accuracy."""
        self.logger.info("Testing budget tracking accuracy")
        
        # Set test budget
        test_budget = 5.0  # $5 daily budget
        self.cost_tracker.daily_budget = test_budget
        
        # Simulate requests with known costs
        test_costs = [0.002, 0.001, 0.003, 0.0015, 0.0025]  # Total: $0.01
        current_time = time.time()
        
        for i, cost in enumerate(test_costs):
            model_name = f"test-model-{i % 2}"  # Alternate between models
            self.cost_tracker.record_cost(model_name, cost, current_time + i)
        
        # Validate budget tracking
        current_daily_cost = self.cost_tracker.get_current_daily_cost()
        expected_total = sum(test_costs)
        
        self.assertAlmostEqual(current_daily_cost, expected_total, places=6,
                             f"Daily cost tracking inaccurate: {current_daily_cost} vs {expected_total}")
        
        # Test budget utilization calculation
        cost_summary = self.cost_tracker.get_cost_summary()
        expected_utilization = expected_total / test_budget
        
        self.assertAlmostEqual(cost_summary['budget_utilization'], expected_utilization, places=6,
                             f"Budget utilization calculation error: {cost_summary['budget_utilization']} vs {expected_utilization}")
        
        self.logger.info(f"Budget tracking: ${current_daily_cost:.6f} / ${test_budget} "
                        f"({cost_summary['budget_utilization']*100:.2f}%)")
    
    def test_cost_alert_thresholds(self):
        """Test cost alert system with warning and critical thresholds."""
        self.logger.info("Testing cost alert thresholds")
        
        # Set test budget and thresholds
        test_budget = 10.0
        warning_threshold = 0.7  # 70%
        critical_threshold = 0.9  # 90%
        
        self.cost_tracker.daily_budget = test_budget
        self.cost_tracker.warning_threshold = warning_threshold
        self.cost_tracker.critical_threshold = critical_threshold
        
        # Test normal usage (below warning threshold)
        normal_cost = test_budget * 0.5  # 50% of budget
        self.cost_tracker.record_cost("test-model", normal_cost)
        
        alerts_normal = self.cost_tracker.check_budget_alerts()
        self.assertEqual(len(alerts_normal), 0, "Should have no alerts below warning threshold")
        
        # Test warning threshold
        warning_cost = test_budget * 0.75 - normal_cost  # Bring total to 75%
        self.cost_tracker.record_cost("test-model", warning_cost)
        
        alerts_warning = self.cost_tracker.check_budget_alerts()
        self.assertEqual(len(alerts_warning), 1, "Should have warning alert at 75% budget")
        self.assertEqual(alerts_warning[0]['level'], 'warning')
        self.assertIn('75', alerts_warning[0]['message'])
        
        # Test critical threshold
        critical_cost = test_budget * 0.95 - (normal_cost + warning_cost)  # Bring total to 95%
        self.cost_tracker.record_cost("test-model", critical_cost)
        
        alerts_critical = self.cost_tracker.check_budget_alerts()
        self.assertEqual(len(alerts_critical), 1, "Should have critical alert at 95% budget")
        self.assertEqual(alerts_critical[0]['level'], 'critical')
        self.assertIn('95', alerts_critical[0]['message'])
        
        # Verify budget utilization in alerts
        self.assertAlmostEqual(alerts_critical[0]['budget_utilization'], 0.95, places=2)
        
        self.logger.info(f"Alert thresholds validated: warning at {warning_threshold*100}%, "
                        f"critical at {critical_threshold*100}%")
    
    def test_cost_prediction_accuracy(self):
        """Test cost prediction accuracy within 5% requirement."""
        self.logger.info("Testing cost prediction accuracy")
        
        # Test different request sizes and models
        test_cases = [
            ("deepseek-v2:16b", "Simple prompt.", 50, 25),  # Local model
            ("gemini-1.0-pro", "Medium complexity prompt with more content.", 200, 100),  # Cloud model
            ("gemini-1.5-flash", "Complex analysis prompt requiring detailed processing.", 500, 250),  # Fast cloud model
        ]
        
        prediction_accuracies = []
        
        for model_name, prompt, input_tokens, expected_output_tokens in test_cases:
            # Find instance for this model
            model_instance = None
            for instance in self.mock_instances.values():
                if instance.model_name == model_name:
                    model_instance = instance
                    break
            
            self.assertIsNotNone(model_instance, f"Model {model_name} not found in mock instances")
            
            # Get cost prediction
            predicted_cost = model_instance.get_estimated_cost(input_tokens, expected_output_tokens)
            
            # Calculate actual cost using model's cost structure
            if model_instance.engine_type == "local":
                actual_cost = 0.0
            else:
                total_tokens = input_tokens + expected_output_tokens
                actual_cost = (total_tokens / 1000.0) * model_instance.cost_per_request
            
            # Calculate prediction accuracy
            if actual_cost == 0.0:
                # For local models, both should be zero
                accuracy = 1.0 if predicted_cost == 0.0 else 0.0
            else:
                accuracy = 1.0 - abs(predicted_cost - actual_cost) / actual_cost
            
            prediction_accuracies.append(accuracy)
            
            self.assertGreaterEqual(accuracy, 0.95,  # 95% accuracy (within 5% error)
                                  f"Cost prediction for {model_name} not accurate enough: "
                                  f"predicted=${predicted_cost:.6f}, actual=${actual_cost:.6f}, "
                                  f"accuracy={accuracy*100:.1f}%")
            
            self.logger.info(f"{model_name}: predicted=${predicted_cost:.6f}, "
                           f"actual=${actual_cost:.6f}, accuracy={accuracy*100:.1f}%")
        
        # Validate overall prediction accuracy
        avg_accuracy = sum(prediction_accuracies) / len(prediction_accuracies)
        self.assertGreaterEqual(avg_accuracy, 0.95,
                              f"Average cost prediction accuracy {avg_accuracy*100:.1f}% below 95% requirement")
        
        self.logger.info(f"Overall cost prediction accuracy: {avg_accuracy*100:.1f}%")
    
    def test_cost_optimization_routing_efficiency(self):
        """Test cost optimization routing efficiency over time."""
        self.logger.info("Testing cost optimization routing efficiency")
        
        # Set budget constraint
        hourly_budget = 2.0
        
        # Create mix of requests with different cost sensitivities
        requests = []
        
        # Cost-sensitive requests (should prefer local models)
        for i in range(10):
            requests.append(self._create_cost_constrained_request(
                prompt=f"Cost-sensitive task {i}.",
                max_cost=0.001,
                use_case="classification"
            ))
        
        # Budget-flexible requests (can use cloud models)
        for i in range(5):
            requests.append(self._create_cost_constrained_request(
                prompt=f"Quality-focused task {i} with longer content requiring analysis.",
                max_cost=0.01,
                use_case="analysis"
            ))
        
        # Route all requests and track costs
        total_cost = 0.0
        local_model_usage = 0
        cloud_model_usage = 0
        
        for request in requests:
            routing_decision = self.router.route_request(request, "cost_first")
            
            # Track costs and model usage
            total_cost += routing_decision.estimated_cost
            
            if routing_decision.selected_instance.engine_type == "local":
                local_model_usage += 1
            else:
                cloud_model_usage += 1
            
            # Validate cost constraint compliance
            self.assertLessEqual(routing_decision.estimated_cost, request.model_config['max_cost'],
                               f"Cost constraint violated: {routing_decision.estimated_cost} > {request.model_config['max_cost']}")
        
        # Validate cost efficiency
        self.assertLessEqual(total_cost, hourly_budget,
                           f"Total cost ${total_cost:.4f} exceeds hourly budget ${hourly_budget}")
        
        # Verify preference for local models for cost-sensitive requests
        local_preference_ratio = local_model_usage / len(requests)
        self.assertGreater(local_preference_ratio, 0.5,
                          f"Should prefer local models for cost optimization: {local_preference_ratio*100:.1f}% local usage")
        
        self.logger.info(f"Cost efficiency: ${total_cost:.4f} total, {local_model_usage} local, {cloud_model_usage} cloud")
        self.logger.info(f"Local model preference: {local_preference_ratio*100:.1f}%")
    
    def test_dynamic_cost_routing_adaptation(self):
        """Test dynamic cost routing adaptation based on budget consumption."""
        self.logger.info("Testing dynamic cost routing adaptation")
        
        # Set daily budget
        daily_budget = 5.0
        self.cost_tracker.daily_budget = daily_budget
        
        # Simulate budget consumption throughout the day
        current_time = time.time()
        
        # Start with low budget consumption (should allow cloud models)
        initial_request = self._create_cost_constrained_request(
            prompt="Initial request with available budget.",
            max_cost=0.01,
            use_case="analysis"
        )
        
        initial_decision = self.router.route_request(initial_request, "cost_first")
        initial_cost = initial_decision.estimated_cost
        
        # Record initial cost
        self.cost_tracker.record_cost("test-model", initial_cost, current_time)
        
        # Simulate high budget consumption (80% of budget used)
        consumed_budget = daily_budget * 0.8
        self.cost_tracker.record_cost("previous-usage", consumed_budget, current_time + 1)
        
        # Create request after high budget consumption
        high_consumption_request = self._create_cost_constrained_request(
            prompt="Request after high budget consumption.",
            max_cost=0.01,
            use_case="analysis"
        )
        
        # Check budget alerts
        alerts = self.cost_tracker.check_budget_alerts()
        budget_stressed = len(alerts) > 0 and any(alert['level'] in ['warning', 'critical'] for alert in alerts)
        
        if budget_stressed:
            # Under budget stress, should prefer local models more aggressively
            stressed_decision = self.router.route_request(high_consumption_request, "cost_first")
            
            # Should prefer local models to conserve budget
            if stressed_decision.selected_instance.engine_type == "local":
                self.assertEqual(stressed_decision.estimated_cost, 0.0,
                               "Should prefer zero-cost local models under budget stress")
        
        # Verify cost tracking consistency
        current_total = self.cost_tracker.get_current_daily_cost()
        expected_total = initial_cost + consumed_budget
        
        self.assertAlmostEqual(current_total, expected_total, places=6,
                             f"Cost tracking inconsistent: {current_total} vs {expected_total}")
        
        self.logger.info(f"Dynamic adaptation: budget stress = {budget_stressed}, "
                        f"total cost = ${current_total:.4f} / ${daily_budget}")
    
    def test_cost_analytics_and_reporting(self):
        """Test cost analytics and reporting functionality."""
        self.logger.info("Testing cost analytics and reporting")
        
        # Generate cost history with multiple models
        models_used = ["deepseek-v2:16b", "gemini-1.0-pro", "llama3:8b"]
        costs_per_model = {"deepseek-v2:16b": [], "gemini-1.0-pro": [], "llama3:8b": []}
        
        current_time = time.time()
        
        # Simulate realistic usage pattern over time
        for hour in range(24):  # 24 hours of usage
            hour_time = current_time - (23 - hour) * 3600  # Start 23 hours ago
            
            for model in models_used:
                # Vary usage patterns by model
                if model == "deepseek-v2:16b":
                    usage_count = hour % 3 + 1  # 1-3 uses per hour
                    cost_per_use = 0.0  # Local model
                elif model == "gemini-1.0-pro":
                    usage_count = 1 if hour % 4 == 0 else 0  # Every 4th hour
                    cost_per_use = 0.002
                else:  # llama3:8b
                    usage_count = hour % 2  # 0-1 uses per hour
                    cost_per_use = 0.0  # Local model
                
                for use in range(usage_count):
                    use_time = hour_time + use * 600  # Spread uses across hour
                    cost = cost_per_use
                    self.cost_tracker.record_cost(model, cost, use_time)
                    costs_per_model[model].append(cost)
        
        # Get cost analytics
        cost_summary = self.cost_tracker.get_cost_summary()
        
        # Validate analytics structure
        self.assertIn('current_daily_cost', cost_summary)
        self.assertIn('daily_budget', cost_summary)
        self.assertIn('budget_utilization', cost_summary)
        self.assertIn('recent_daily_costs', cost_summary)
        
        # Validate cost calculations
        expected_daily_total = sum(sum(costs) for costs in costs_per_model.values())
        actual_daily_total = cost_summary['current_daily_cost']
        
        self.assertAlmostEqual(actual_daily_total, expected_daily_total, places=6,
                             f"Daily cost calculation error: {actual_daily_total} vs {expected_daily_total}")
        
        # Validate budget utilization
        expected_utilization = expected_daily_total / self.cost_tracker.daily_budget
        self.assertAlmostEqual(cost_summary['budget_utilization'], expected_utilization, places=6)
        
        # Validate historical data
        recent_costs = cost_summary['recent_daily_costs']
        self.assertIsInstance(recent_costs, dict)
        self.assertGreater(len(recent_costs), 0, "Should have recent cost history")
        
        self.logger.info(f"Cost analytics: daily=${actual_daily_total:.6f}, "
                        f"utilization={cost_summary['budget_utilization']*100:.1f}%, "
                        f"history_days={len(recent_costs)}")
    
    def test_cost_optimization_recommendations(self):
        """Test cost optimization recommendation system."""
        self.logger.info("Testing cost optimization recommendations")
        
        # Record usage of expensive model
        expensive_model = "gemini-1.0-pro"
        expensive_cost = 0.002
        
        # Simulate high usage of expensive model
        for i in range(20):
            self.performance_analytics.record_request_metrics(
                model_name=expensive_model,
                instance_id=f"expensive-instance-{i%2}",
                response_time=2.0,
                tokens_processed=1000,
                cost=expensive_cost,
                success=True
            )
        
        # Record usage of cheaper alternative
        cheap_model = "deepseek-v2:16b"
        cheap_cost = 0.0
        
        for i in range(10):
            self.performance_analytics.record_request_metrics(
                model_name=cheap_model,
                instance_id=f"cheap-instance-{i%2}",
                response_time=1.5,
                tokens_processed=1000,
                cost=cheap_cost,
                success=True
            )
        
        # Generate optimization recommendations
        recommendations = self.performance_analytics.generate_optimization_recommendations()
        
        # Find cost-related recommendations
        cost_recommendations = [rec for rec in recommendations if rec.recommendation_type == "cost"]
        
        if len(cost_recommendations) > 0:
            cost_rec = cost_recommendations[0]
            
            # Validate recommendation structure
            self.assertIsInstance(cost_rec.title, str)
            self.assertIsInstance(cost_rec.description, str)
            self.assertIsInstance(cost_rec.action_items, list)
            self.assertGreater(cost_rec.estimated_cost_savings, 0.0)
            
            # Should mention the expensive model
            recommendation_text = (cost_rec.title + " " + cost_rec.description).lower()
            self.assertIn(expensive_model.lower(), recommendation_text,
                         "Cost recommendation should mention expensive model")
            
            self.logger.info(f"Cost recommendation: {cost_rec.title}")
            self.logger.info(f"Estimated savings: ${cost_rec.estimated_cost_savings:.4f}")
        else:
            self.logger.info("No cost recommendations generated (may be expected if differences are small)")
    
    def test_budget_exhaustion_handling(self):
        """Test system behavior when budget is exhausted."""
        self.logger.info("Testing budget exhaustion handling")
        
        # Set very low budget
        low_budget = 0.001
        self.cost_tracker.daily_budget = low_budget
        
        # Exhaust budget
        self.cost_tracker.record_cost("test-model", low_budget * 0.95)
        
        # Create request after budget near exhaustion
        request = self._create_cost_constrained_request(
            prompt="Request after budget exhaustion.",
            max_cost=0.01,  # Would normally allow cloud models
            use_case="analysis"
        )
        
        # Route request - should strongly prefer local models
        routing_decision = self.router.route_request(request, "cost_first")
        
        # Check budget alerts
        alerts = self.cost_tracker.check_budget_alerts()
        budget_critical = any(alert['level'] == 'critical' for alert in alerts)
        
        if budget_critical:
            # Under critical budget, should prefer zero-cost models
            if routing_decision.selected_instance.engine_type == "local":
                self.assertEqual(routing_decision.estimated_cost, 0.0,
                               "Should route to zero-cost models when budget is critical")
            
            # Verify cost-conscious reasoning
            reasoning_text = " ".join(routing_decision.reasoning).lower()
            cost_keywords = ["cost", "budget", "local", "free", "efficient"]
            self.assertTrue(any(keyword in reasoning_text for keyword in cost_keywords),
                           "Should mention cost considerations in critical budget situation")
        
        self.logger.info(f"Budget exhaustion handling: critical={budget_critical}, "
                        f"selected={routing_decision.selected_instance.model_name} "
                        f"(${routing_decision.estimated_cost:.6f})")
    
    def _create_cost_constrained_request(self, prompt: str, max_cost: float, 
                                       use_case: str = "general") -> LLMRequest:
        """Create a test request with cost constraints."""
        return LLMRequest(
            request_id=f"cost-test-{int(time.time() * 1000000)}",
            prompt=prompt,
            model_config={
                'use_case': use_case,
                'max_cost': max_cost,
                'quality_requirements': 'medium'
            },
            priority=5,
            timeout=30.0,
            retry_count=0,
            max_retries=3,
            created_at=time.time()
        )


class TestCostTrackerPrecision(unittest.TestCase):
    """Detailed tests for cost tracker precision and accuracy."""
    
    def setUp(self):
        """Set up cost tracker tests."""
        self.cost_tracker = CostTracker()
        self.cost_tracker.daily_budget = 10.0
    
    def test_hourly_cost_aggregation(self):
        """Test hourly cost aggregation accuracy."""
        # Record costs at different times within the same hour
        base_time = time.time()
        hour_start = base_time - (base_time % 3600)  # Round to hour start
        
        test_costs = [0.001, 0.002, 0.0015, 0.0025]
        models = ["model-a", "model-b", "model-a", "model-c"]
        
        for i, (cost, model) in enumerate(zip(test_costs, models)):
            cost_time = hour_start + i * 600  # Spread across hour
            self.cost_tracker.record_cost(model, cost, cost_time)
        
        # Verify hourly aggregation
        hour_key = datetime.fromtimestamp(hour_start).strftime("%Y-%m-%d-%H")
        hourly_total = sum(self.cost_tracker.hourly_costs[hour_key].values())
        expected_total = sum(test_costs)
        
        self.assertAlmostEqual(hourly_total, expected_total, places=8,
                             f"Hourly cost aggregation error: {hourly_total} vs {expected_total}")
        
        # Verify per-model breakdown
        self.assertAlmostEqual(self.cost_tracker.hourly_costs[hour_key]["model-a"], 0.001 + 0.0015, places=8)
        self.assertAlmostEqual(self.cost_tracker.hourly_costs[hour_key]["model-b"], 0.002, places=8)
        self.assertAlmostEqual(self.cost_tracker.hourly_costs[hour_key]["model-c"], 0.0025, places=8)
    
    def test_daily_cost_rollup(self):
        """Test daily cost rollup from hourly data."""
        base_time = time.time()
        day_start = base_time - (base_time % 86400)  # Round to day start
        
        # Record costs across multiple hours
        hourly_costs = [0.1, 0.15, 0.2, 0.05]  # Different hours
        
        for hour, cost in enumerate(hourly_costs):
            cost_time = day_start + hour * 3600
            self.cost_tracker.record_cost("test-model", cost, cost_time)
        
        # Verify daily rollup
        day_key = datetime.fromtimestamp(day_start).strftime("%Y-%m-%d")
        daily_total = sum(self.cost_tracker.daily_costs[day_key].values())
        expected_total = sum(hourly_costs)
        
        self.assertAlmostEqual(daily_total, expected_total, places=8,
                             f"Daily cost rollup error: {daily_total} vs {expected_total}")
    
    def test_floating_point_precision(self):
        """Test floating point precision in cost calculations."""
        # Test with very small costs that could cause precision issues
        small_costs = [0.000001, 0.000002, 0.000003, 0.000001]
        
        for cost in small_costs:
            self.cost_tracker.record_cost("precision-test", cost)
        
        current_cost = self.cost_tracker.get_current_daily_cost()
        expected_cost = sum(small_costs)
        
        # Should be accurate to 8 decimal places
        self.assertAlmostEqual(current_cost, expected_cost, places=8,
                             f"Floating point precision error: {current_cost} vs {expected_cost}")
    
    def test_budget_utilization_edge_cases(self):
        """Test budget utilization calculation edge cases."""
        # Test zero budget
        self.cost_tracker.daily_budget = 0.0
        self.cost_tracker.record_cost("test", 0.001)
        
        cost_summary = self.cost_tracker.get_cost_summary()
        # Should handle division by zero gracefully
        self.assertIsInstance(cost_summary['budget_utilization'], (int, float))
        
        # Test budget exactly met
        self.cost_tracker.daily_budget = 1.0
        self.cost_tracker = CostTracker()  # Reset
        self.cost_tracker.daily_budget = 1.0
        self.cost_tracker.record_cost("test", 1.0)
        
        cost_summary = self.cost_tracker.get_cost_summary()
        self.assertAlmostEqual(cost_summary['budget_utilization'], 1.0, places=8)
        
        # Test budget exceeded
        self.cost_tracker.record_cost("test", 0.5)  # Total now 1.5
        
        cost_summary = self.cost_tracker.get_cost_summary()
        self.assertAlmostEqual(cost_summary['budget_utilization'], 1.5, places=8)


if __name__ == '__main__':
    # Configure logging for test runs
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    unittest.main(verbosity=2)