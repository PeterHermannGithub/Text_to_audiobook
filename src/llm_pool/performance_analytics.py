"""
Performance Analytics & Optimization Module for Multi-Model Load Balancing.

This module provides comprehensive performance tracking, analytics, and optimization
recommendations for the multi-model LLM pool system.
"""

import logging
import time
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json

from config import settings


@dataclass
class PerformanceMetrics:
    """Performance metrics for a specific model or instance."""
    model_name: str
    instance_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    total_tokens_processed: int = 0
    total_cost: float = 0.0
    
    # Rolling window metrics (last N requests)
    recent_response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_costs: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_quality_scores: deque = field(default_factory=lambda: deque(maxlen=50))
    recent_error_types: deque = field(default_factory=lambda: deque(maxlen=50))
    
    # Time-based metrics
    hourly_requests: Dict[str, int] = field(default_factory=dict)
    hourly_costs: Dict[str, float] = field(default_factory=dict)
    daily_requests: Dict[str, int] = field(default_factory=dict)
    daily_costs: Dict[str, float] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_response_time / self.successful_requests
    
    @property
    def average_cost_per_request(self) -> float:
        """Calculate average cost per request."""
        if self.total_requests == 0:
            return 0.0
        return self.total_cost / self.total_requests
    
    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens processed per second."""
        if self.total_response_time == 0:
            return 0.0
        return self.total_tokens_processed / self.total_response_time
    
    @property
    def recent_average_response_time(self) -> float:
        """Calculate recent average response time."""
        if not self.recent_response_times:
            return 0.0
        return statistics.mean(self.recent_response_times)
    
    @property
    def recent_average_cost(self) -> float:
        """Calculate recent average cost."""
        if not self.recent_costs:
            return 0.0
        return statistics.mean(self.recent_costs)
    
    @property
    def response_time_percentiles(self) -> Dict[str, float]:
        """Calculate response time percentiles."""
        if not self.recent_response_times:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        
        sorted_times = sorted(self.recent_response_times)
        length = len(sorted_times)
        
        return {
            "p50": sorted_times[int(length * 0.5)] if length > 0 else 0.0,
            "p95": sorted_times[int(length * 0.95)] if length > 0 else 0.0,
            "p99": sorted_times[int(length * 0.99)] if length > 0 else 0.0
        }


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation."""
    recommendation_type: str  # "scaling", "routing", "cost", "performance"
    priority: str  # "high", "medium", "low"
    title: str
    description: str
    expected_impact: str
    implementation_complexity: str  # "low", "medium", "high"
    estimated_cost_savings: float
    estimated_performance_gain: float
    action_items: List[str]


class PerformanceAnalytics:
    """
    Comprehensive Performance Analytics & Optimization Engine.
    
    This class provides detailed performance tracking, analytics, and optimization
    recommendations for the multi-model LLM pool system.
    """
    
    def __init__(self):
        """Initialize performance analytics engine."""
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics storage
        self.model_metrics: Dict[str, PerformanceMetrics] = {}
        self.instance_metrics: Dict[str, PerformanceMetrics] = {}
        
        # Global system metrics
        self.system_start_time = time.time()
        self.global_request_count = 0
        self.global_success_count = 0
        self.global_error_count = 0
        self.global_cost = 0.0
        
        # Cost tracking and budgeting
        self.cost_tracker = CostTracker()
        self.performance_benchmarks = PerformanceBenchmarks()
        
        # Performance history for trend analysis
        self.performance_history: List[Dict[str, Any]] = []
        
        self.logger.info("Performance Analytics engine initialized")
    
    def record_request_metrics(self, model_name: str, instance_id: str, 
                             response_time: float, tokens_processed: int,
                             cost: float, success: bool, error_type: str = None,
                             quality_score: float = None):
        """Record metrics for a completed request."""
        current_time = time.time()
        hour_key = datetime.fromtimestamp(current_time).strftime("%Y-%m-%d-%H")
        day_key = datetime.fromtimestamp(current_time).strftime("%Y-%m-%d")
        
        # Update model metrics
        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = PerformanceMetrics(model_name, "model_aggregate")
        
        model_metrics = self.model_metrics[model_name]
        model_metrics.total_requests += 1
        if success:
            model_metrics.successful_requests += 1
            model_metrics.total_response_time += response_time
            model_metrics.recent_response_times.append(response_time)
        else:
            model_metrics.failed_requests += 1
            if error_type:
                model_metrics.recent_error_types.append(error_type)
        
        model_metrics.total_tokens_processed += tokens_processed
        model_metrics.total_cost += cost
        model_metrics.recent_costs.append(cost)
        
        if quality_score is not None:
            model_metrics.recent_quality_scores.append(quality_score)
        
        # Update time-based metrics
        model_metrics.hourly_requests[hour_key] = model_metrics.hourly_requests.get(hour_key, 0) + 1
        model_metrics.hourly_costs[hour_key] = model_metrics.hourly_costs.get(hour_key, 0.0) + cost
        model_metrics.daily_requests[day_key] = model_metrics.daily_requests.get(day_key, 0) + 1
        model_metrics.daily_costs[day_key] = model_metrics.daily_costs.get(day_key, 0.0) + cost
        
        # Update instance metrics
        if instance_id not in self.instance_metrics:
            self.instance_metrics[instance_id] = PerformanceMetrics(model_name, instance_id)
        
        instance_metrics = self.instance_metrics[instance_id]
        instance_metrics.total_requests += 1
        if success:
            instance_metrics.successful_requests += 1
            instance_metrics.total_response_time += response_time
            instance_metrics.recent_response_times.append(response_time)
        else:
            instance_metrics.failed_requests += 1
            if error_type:
                instance_metrics.recent_error_types.append(error_type)
        
        instance_metrics.total_tokens_processed += tokens_processed
        instance_metrics.total_cost += cost
        instance_metrics.recent_costs.append(cost)
        
        if quality_score is not None:
            instance_metrics.recent_quality_scores.append(quality_score)
        
        # Update global metrics
        self.global_request_count += 1
        if success:
            self.global_success_count += 1
        else:
            self.global_error_count += 1
        self.global_cost += cost
        
        # Update cost tracker
        self.cost_tracker.record_cost(model_name, cost, current_time)
        
        self.logger.debug(f"Recorded metrics for {model_name}: "
                         f"response_time={response_time:.3f}s, cost=${cost:.4f}, success={success}")
    
    def get_model_performance_summary(self, model_name: str) -> Dict[str, Any]:
        """Get comprehensive performance summary for a specific model."""
        if model_name not in self.model_metrics:
            return {"error": f"No metrics found for model {model_name}"}
        
        metrics = self.model_metrics[model_name]
        
        return {
            "model_name": model_name,
            "total_requests": metrics.total_requests,
            "success_rate": metrics.success_rate,
            "average_response_time": metrics.average_response_time,
            "recent_average_response_time": metrics.recent_average_response_time,
            "response_time_percentiles": metrics.response_time_percentiles,
            "average_cost_per_request": metrics.average_cost_per_request,
            "recent_average_cost": metrics.recent_average_cost,
            "total_cost": metrics.total_cost,
            "tokens_per_second": metrics.tokens_per_second,
            "recent_quality_scores": list(metrics.recent_quality_scores) if metrics.recent_quality_scores else [],
            "recent_error_types": list(metrics.recent_error_types) if metrics.recent_error_types else [],
            "hourly_request_trend": dict(list(metrics.hourly_requests.items())[-24:]),  # Last 24 hours
            "daily_cost_trend": dict(list(metrics.daily_costs.items())[-7:])  # Last 7 days
        }
    
    def get_comparative_analysis(self) -> Dict[str, Any]:
        """Get comparative analysis across all models."""
        if not self.model_metrics:
            return {"error": "No model metrics available"}
        
        analysis = {
            "model_comparison": {},
            "rankings": {
                "fastest_models": [],
                "most_reliable_models": [],
                "most_cost_effective_models": [],
                "highest_quality_models": []
            },
            "optimization_opportunities": []
        }
        
        # Compare models
        for model_name, metrics in self.model_metrics.items():
            if metrics.total_requests == 0:
                continue
                
            analysis["model_comparison"][model_name] = {
                "requests": metrics.total_requests,
                "success_rate": metrics.success_rate,
                "avg_response_time": metrics.average_response_time,
                "avg_cost": metrics.average_cost_per_request,
                "tokens_per_second": metrics.tokens_per_second,
                "recent_quality": statistics.mean(metrics.recent_quality_scores) if metrics.recent_quality_scores else 0.0
            }
        
        # Generate rankings
        model_stats = [(name, stats) for name, stats in analysis["model_comparison"].items()]
        
        # Fastest models (by response time)
        analysis["rankings"]["fastest_models"] = sorted(
            model_stats, key=lambda x: x[1]["avg_response_time"]
        )[:3]
        
        # Most reliable models (by success rate)
        analysis["rankings"]["most_reliable_models"] = sorted(
            model_stats, key=lambda x: x[1]["success_rate"], reverse=True
        )[:3]
        
        # Most cost-effective models (by cost per request)
        analysis["rankings"]["most_cost_effective_models"] = sorted(
            model_stats, key=lambda x: x[1]["avg_cost"]
        )[:3]
        
        # Highest quality models (by quality score)
        quality_models = [(name, stats) for name, stats in model_stats if stats["recent_quality"] > 0]
        analysis["rankings"]["highest_quality_models"] = sorted(
            quality_models, key=lambda x: x[1]["recent_quality"], reverse=True
        )[:3]
        
        return analysis
    
    def generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on performance analysis."""
        recommendations = []
        
        if not self.model_metrics:
            return recommendations
        
        # Analyze cost optimization opportunities
        cost_recommendations = self._analyze_cost_optimization()
        recommendations.extend(cost_recommendations)
        
        # Analyze performance optimization opportunities
        performance_recommendations = self._analyze_performance_optimization()
        recommendations.extend(performance_recommendations)
        
        # Analyze scaling recommendations
        scaling_recommendations = self._analyze_scaling_needs()
        recommendations.extend(scaling_recommendations)
        
        # Analyze routing optimization
        routing_recommendations = self._analyze_routing_optimization()
        recommendations.extend(routing_recommendations)
        
        # Sort by priority and impact
        recommendations.sort(key=lambda r: (
            {"high": 3, "medium": 2, "low": 1}[r.priority],
            r.estimated_cost_savings + r.estimated_performance_gain
        ), reverse=True)
        
        return recommendations
    
    def _analyze_cost_optimization(self) -> List[OptimizationRecommendation]:
        """Analyze cost optimization opportunities."""
        recommendations = []
        
        # Check for high-cost models with low utilization
        for model_name, metrics in self.model_metrics.items():
            if metrics.total_requests < 10:  # Low sample size
                continue
                
            avg_cost = metrics.average_cost_per_request
            if avg_cost > 0.01:  # High cost threshold
                # Check if there are cheaper alternatives
                cheaper_alternatives = [
                    (name, m) for name, m in self.model_metrics.items()
                    if m.average_cost_per_request < avg_cost * 0.5 and m.success_rate >= metrics.success_rate * 0.9
                ]
                
                if cheaper_alternatives:
                    potential_savings = avg_cost * metrics.total_requests * 0.5
                    recommendations.append(OptimizationRecommendation(
                        recommendation_type="cost",
                        priority="high" if potential_savings > 1.0 else "medium",
                        title=f"Cost Optimization for {model_name}",
                        description=f"Model {model_name} has high cost per request (${avg_cost:.4f}). "
                                   f"Consider using cheaper alternatives like {cheaper_alternatives[0][0]} "
                                   f"for similar tasks.",
                        expected_impact=f"Potential savings: ${potential_savings:.2f}",
                        implementation_complexity="low",
                        estimated_cost_savings=potential_savings,
                        estimated_performance_gain=0.0,
                        action_items=[
                            f"Test {cheaper_alternatives[0][0]} for current {model_name} workloads",
                            "Implement cost-aware routing for budget-sensitive requests",
                            "Set up cost alerts for high-spend models"
                        ]
                    ))
        
        return recommendations
    
    def _analyze_performance_optimization(self) -> List[OptimizationRecommendation]:
        """Analyze performance optimization opportunities."""
        recommendations = []
        
        # Check for slow models with high usage
        for model_name, metrics in self.model_metrics.items():
            if metrics.total_requests < 5:
                continue
                
            avg_response_time = metrics.average_response_time
            if avg_response_time > 3.0 and metrics.total_requests > 50:  # Slow and heavily used
                faster_alternatives = [
                    (name, m) for name, m in self.model_metrics.items()
                    if m.average_response_time < avg_response_time * 0.7 and m.success_rate >= metrics.success_rate * 0.9
                ]
                
                if faster_alternatives:
                    time_savings = (avg_response_time - faster_alternatives[0][1].average_response_time) * metrics.total_requests
                    recommendations.append(OptimizationRecommendation(
                        recommendation_type="performance",
                        priority="high" if time_savings > 300 else "medium",  # 5+ minutes saved
                        title=f"Performance Optimization for {model_name}",
                        description=f"Model {model_name} has slow response times ({avg_response_time:.2f}s avg). "
                                   f"Consider using {faster_alternatives[0][0]} for speed-critical requests.",
                        expected_impact=f"Potential time savings: {time_savings:.1f} seconds",
                        implementation_complexity="low",
                        estimated_cost_savings=0.0,
                        estimated_performance_gain=time_savings,
                        action_items=[
                            f"Implement speed-first routing for time-critical requests",
                            f"Test {faster_alternatives[0][0]} as alternative for {model_name}",
                            "Optimize request preprocessing to reduce payload size"
                        ]
                    ))
        
        return recommendations
    
    def _analyze_scaling_needs(self) -> List[OptimizationRecommendation]:
        """Analyze scaling recommendations."""
        recommendations = []
        
        # Check for models with high error rates
        for model_name, metrics in self.model_metrics.items():
            if metrics.total_requests < 10:
                continue
                
            error_rate = 1.0 - metrics.success_rate
            if error_rate > 0.1:  # High error rate
                recommendations.append(OptimizationRecommendation(
                    recommendation_type="scaling",
                    priority="high",
                    title=f"High Error Rate Alert: {model_name}",
                    description=f"Model {model_name} has high error rate ({error_rate*100:.1f}%). "
                               f"This may indicate capacity issues or model problems.",
                    expected_impact=f"Reduce error rate from {error_rate*100:.1f}% to <5%",
                    implementation_complexity="medium",
                    estimated_cost_savings=0.0,
                    estimated_performance_gain=50.0,  # Reliability improvement
                    action_items=[
                        f"Add more instances for {model_name}",
                        "Implement health checks and automatic failover",
                        "Investigate root cause of failures",
                        "Consider alternative models for reliability"
                    ]
                ))
        
        return recommendations
    
    def _analyze_routing_optimization(self) -> List[OptimizationRecommendation]:
        """Analyze routing optimization opportunities."""
        recommendations = []
        
        # Check for underutilized models
        total_requests = sum(m.total_requests for m in self.model_metrics.values())
        if total_requests > 0:
            for model_name, metrics in self.model_metrics.items():
                utilization = metrics.total_requests / total_requests
                if utilization < 0.05 and metrics.success_rate > 0.9:  # Underutilized but reliable
                    recommendations.append(OptimizationRecommendation(
                        recommendation_type="routing",
                        priority="low",
                        title=f"Underutilized Model: {model_name}",
                        description=f"Model {model_name} is underutilized ({utilization*100:.1f}% of requests) "
                                   f"but has good performance. Consider routing more suitable requests to it.",
                        expected_impact="Better load distribution and cost optimization",
                        implementation_complexity="low",
                        estimated_cost_savings=0.0,
                        estimated_performance_gain=10.0,
                        action_items=[
                            f"Review use cases suitable for {model_name}",
                            "Adjust routing weights to increase utilization",
                            "Consider using for backup/fallback scenarios"
                        ]
                    ))
        
        return recommendations
    
    def get_system_health_score(self) -> float:
        """Calculate overall system health score (0.0-1.0)."""
        if not self.model_metrics:
            return 0.0
        
        health_factors = []
        
        # Success rate factor
        overall_success_rate = self.global_success_count / max(self.global_request_count, 1)
        health_factors.append(overall_success_rate)
        
        # Performance factor (based on response times)
        avg_response_times = [
            m.average_response_time for m in self.model_metrics.values()
            if m.total_requests > 0
        ]
        if avg_response_times:
            # Normalize response time to 0-1 scale (assuming 5s is poor, <1s is excellent)
            avg_response_time = statistics.mean(avg_response_times)
            performance_factor = max(0.0, min(1.0, (5.0 - avg_response_time) / 5.0))
            health_factors.append(performance_factor)
        
        # Cost efficiency factor
        if self.global_cost > 0:
            cost_per_request = self.global_cost / self.global_request_count
            # Normalize cost (assuming $0.01 is expensive, $0 is excellent)
            cost_factor = max(0.0, min(1.0, (0.01 - cost_per_request) / 0.01))
            health_factors.append(cost_factor)
        
        return statistics.mean(health_factors) if health_factors else 0.0
    
    def export_analytics_data(self) -> Dict[str, Any]:
        """Export comprehensive analytics data for external analysis."""
        return {
            "timestamp": time.time(),
            "system_uptime": time.time() - self.system_start_time,
            "global_metrics": {
                "total_requests": self.global_request_count,
                "success_count": self.global_success_count,
                "error_count": self.global_error_count,
                "total_cost": self.global_cost,
                "system_health_score": self.get_system_health_score()
            },
            "model_metrics": {
                name: self.get_model_performance_summary(name)
                for name in self.model_metrics.keys()
            },
            "comparative_analysis": self.get_comparative_analysis(),
            "optimization_recommendations": [
                {
                    "type": rec.recommendation_type,
                    "priority": rec.priority,
                    "title": rec.title,
                    "description": rec.description,
                    "expected_impact": rec.expected_impact,
                    "complexity": rec.implementation_complexity,
                    "cost_savings": rec.estimated_cost_savings,
                    "performance_gain": rec.estimated_performance_gain,
                    "actions": rec.action_items
                }
                for rec in self.generate_optimization_recommendations()
            ],
            "cost_tracking": self.cost_tracker.get_cost_summary(),
            "performance_benchmarks": self.performance_benchmarks.get_benchmark_results()
        }


class CostTracker:
    """Cost tracking and budget management."""
    
    def __init__(self):
        self.hourly_costs: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.daily_costs: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.monthly_costs: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Budget settings from configuration
        self.daily_budget = getattr(settings, 'COST_MANAGEMENT', {}).get('daily_budget_usd', 50.0)
        self.warning_threshold = getattr(settings, 'COST_MANAGEMENT', {}).get('cost_alerts', {}).get('warning_threshold', 0.7)
        self.critical_threshold = getattr(settings, 'COST_MANAGEMENT', {}).get('cost_alerts', {}).get('critical_threshold', 0.9)
    
    def record_cost(self, model_name: str, cost: float, timestamp: float = None):
        """Record cost for a model at a specific time."""
        if timestamp is None:
            timestamp = time.time()
        
        dt = datetime.fromtimestamp(timestamp)
        hour_key = dt.strftime("%Y-%m-%d-%H")
        day_key = dt.strftime("%Y-%m-%d")
        month_key = dt.strftime("%Y-%m")
        
        self.hourly_costs[hour_key][model_name] += cost
        self.daily_costs[day_key][model_name] += cost
        self.monthly_costs[month_key][model_name] += cost
    
    def get_current_daily_cost(self) -> float:
        """Get current daily cost."""
        today = datetime.now().strftime("%Y-%m-%d")
        return sum(self.daily_costs[today].values())
    
    def check_budget_alerts(self) -> List[Dict[str, Any]]:
        """Check for budget alert conditions."""
        alerts = []
        current_cost = self.get_current_daily_cost()
        
        if current_cost >= self.daily_budget * self.critical_threshold:
            alerts.append({
                "level": "critical",
                "message": f"Daily cost (${current_cost:.2f}) is {current_cost/self.daily_budget*100:.1f}% of budget",
                "budget_utilization": current_cost / self.daily_budget
            })
        elif current_cost >= self.daily_budget * self.warning_threshold:
            alerts.append({
                "level": "warning", 
                "message": f"Daily cost (${current_cost:.2f}) is {current_cost/self.daily_budget*100:.1f}% of budget",
                "budget_utilization": current_cost / self.daily_budget
            })
        
        return alerts
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get comprehensive cost summary."""
        current_cost = self.get_current_daily_cost()
        
        return {
            "current_daily_cost": current_cost,
            "daily_budget": self.daily_budget,
            "budget_utilization": current_cost / self.daily_budget,
            "budget_alerts": self.check_budget_alerts(),
            "recent_daily_costs": dict(list(self.daily_costs.items())[-7:])  # Last 7 days
        }


class PerformanceBenchmarks:
    """Performance benchmarking suite."""
    
    def __init__(self):
        self.benchmark_results: Dict[str, Any] = {}
    
    def run_model_comparison_benchmark(self, models: List[str], test_prompts: List[str]) -> Dict[str, Any]:
        """Run comparative performance benchmark across models."""
        # This would be implemented to actually test models
        # For now, return placeholder structure
        return {
            "benchmark_type": "model_comparison",
            "models_tested": models,
            "test_prompts_count": len(test_prompts),
            "results": {
                model: {
                    "avg_response_time": 0.0,
                    "success_rate": 0.0,
                    "cost_per_prompt": 0.0,
                    "quality_score": 0.0
                }
                for model in models
            },
            "timestamp": time.time()
        }
    
    def get_benchmark_results(self) -> Dict[str, Any]:
        """Get latest benchmark results."""
        return self.benchmark_results