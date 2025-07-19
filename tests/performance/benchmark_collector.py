"""
Performance Metrics Collection System for Multi-Model Benchmarking.

This module provides comprehensive metrics collection, analysis, and validation
for the multi-model load balancing system performance benchmarks.
"""

import time
import statistics
import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import math

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.llm_pool.performance_analytics import PerformanceAnalytics


@dataclass
class PerformanceMetric:
    """Individual performance metric with statistical data."""
    name: str
    value: float
    unit: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    target_value: Optional[float] = None
    target_comparison: str = "less_than"  # "less_than", "greater_than", "equals", "between"
    target_range: Optional[Tuple[float, float]] = None


@dataclass
class BenchmarkSession:
    """Complete benchmark session with all metrics and analysis."""
    session_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    metrics: List[PerformanceMetric] = field(default_factory=list)
    aggregated_metrics: Dict[str, Any] = field(default_factory=dict)
    performance_requirements: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    baseline_comparison: Optional[Dict[str, Any]] = None
    statistical_analysis: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self):
        """Finalize benchmark session."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time


class PerformanceMetricsCollector:
    """
    Comprehensive Performance Metrics Collection and Analysis System.
    
    Collects, analyzes, and validates performance metrics for multi-model
    load balancing benchmarks against specified requirements.
    """
    
    def __init__(self):
        """Initialize the performance metrics collector."""
        self.logger = logging.getLogger(__name__)
        
        # Performance requirements from strategic plan
        self.performance_requirements = {
            'throughput_improvement_min': 2.0,  # 2x minimum improvement
            'throughput_improvement_target': 3.0,  # 3x target improvement
            'cost_reduction_min': 0.30,  # 30% minimum reduction
            'cost_reduction_target': 0.50,  # 50% target reduction
            'p95_response_time_max': 1.0,  # 1 second maximum
            'p99_response_time_max': 2.0,  # 2 second maximum
            'error_rate_max': 0.01,  # 1% maximum error rate
            'routing_overhead_max': 0.01,  # 10ms maximum routing overhead
            'cost_prediction_accuracy_min': 0.95  # 95% minimum accuracy
        }
        
        # Active benchmark sessions
        self.active_sessions: Dict[str, BenchmarkSession] = {}
        self.completed_sessions: Dict[str, BenchmarkSession] = {}
        
        # Baseline performance data for comparisons
        self.baseline_metrics: Dict[str, float] = {}
        
        self.logger.info("Performance Metrics Collector initialized")
    
    def start_benchmark_session(self, name: str, context: Dict[str, Any] = None) -> str:
        """Start a new benchmark session and return session ID."""
        session_id = f"bench_{int(time.time() * 1000000)}"
        
        session = BenchmarkSession(
            session_id=session_id,
            name=name,
            start_time=time.time(),
            performance_requirements=self.performance_requirements.copy()
        )
        
        if context:
            session.aggregated_metrics['context'] = context
        
        self.active_sessions[session_id] = session
        
        self.logger.info(f"Started benchmark session: {name} ({session_id})")
        return session_id
    
    def record_metric(self, session_id: str, name: str, value: float, unit: str,
                     context: Dict[str, Any] = None, target_value: float = None,
                     target_comparison: str = "less_than") -> None:
        """Record a performance metric for the specified session."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found or not active")
        
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            context=context or {},
            target_value=target_value,
            target_comparison=target_comparison
        )
        
        self.active_sessions[session_id].metrics.append(metric)
        
        self.logger.debug(f"Recorded metric {name}: {value} {unit} (session: {session_id})")
    
    def record_throughput_metrics(self, session_id: str, total_requests: int,
                                successful_requests: int, duration: float,
                                baseline_rps: float = None) -> None:
        """Record throughput-related metrics."""
        if duration <= 0:
            raise ValueError("Duration must be positive")
        
        throughput_rps = successful_requests / duration
        error_rate = (total_requests - successful_requests) / total_requests if total_requests > 0 else 0
        
        # Record basic throughput metrics
        self.record_metric(session_id, "throughput_rps", throughput_rps, "requests/second",
                          target_value=2.0, target_comparison="greater_than")
        
        self.record_metric(session_id, "error_rate", error_rate, "percentage",
                          target_value=self.performance_requirements['error_rate_max'],
                          target_comparison="less_than")
        
        self.record_metric(session_id, "total_requests", total_requests, "count")
        self.record_metric(session_id, "successful_requests", successful_requests, "count")
        self.record_metric(session_id, "duration", duration, "seconds")
        
        # Calculate throughput improvement if baseline provided
        if baseline_rps and baseline_rps > 0:
            improvement = throughput_rps / baseline_rps
            self.record_metric(session_id, "throughput_improvement", improvement, "multiplier",
                              target_value=self.performance_requirements['throughput_improvement_min'],
                              target_comparison="greater_than")
            
            self.logger.info(f"Throughput improvement: {improvement:.2f}x ({baseline_rps:.2f} → {throughput_rps:.2f} RPS)")
    
    def record_response_time_metrics(self, session_id: str, response_times: List[float]) -> None:
        """Record response time statistics and percentiles."""
        if not response_times:
            self.logger.warning("No response times provided")
            return
        
        # Calculate statistics
        avg_time = statistics.mean(response_times)
        median_time = statistics.median(response_times)
        
        # Calculate percentiles
        sorted_times = sorted(response_times)
        p95_time = sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) > 1 else sorted_times[0]
        p99_time = sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) > 1 else sorted_times[0]
        
        # Record metrics
        self.record_metric(session_id, "avg_response_time", avg_time, "seconds")
        self.record_metric(session_id, "median_response_time", median_time, "seconds")
        
        self.record_metric(session_id, "p95_response_time", p95_time, "seconds",
                          target_value=self.performance_requirements['p95_response_time_max'],
                          target_comparison="less_than")
        
        self.record_metric(session_id, "p99_response_time", p99_time, "seconds",
                          target_value=self.performance_requirements['p99_response_time_max'],
                          target_comparison="less_than")
        
        # Calculate variance and consistency metrics
        if len(response_times) > 1:
            std_dev = statistics.stdev(response_times)
            coefficient_of_variation = std_dev / avg_time if avg_time > 0 else 0
            
            self.record_metric(session_id, "response_time_std_dev", std_dev, "seconds")
            self.record_metric(session_id, "response_time_cv", coefficient_of_variation, "ratio",
                              target_value=0.5, target_comparison="less_than")  # Target: CV < 0.5
    
    def record_cost_metrics(self, session_id: str, costs: List[float],
                          baseline_total_cost: float = None) -> None:
        """Record cost-related metrics."""
        if not costs:
            self.logger.warning("No costs provided")
            return
        
        total_cost = sum(costs)
        avg_cost = statistics.mean(costs)
        
        # Record basic cost metrics
        self.record_metric(session_id, "total_cost", total_cost, "usd")
        self.record_metric(session_id, "avg_cost_per_request", avg_cost, "usd")
        
        # Calculate cost reduction if baseline provided
        if baseline_total_cost and baseline_total_cost > 0:
            cost_reduction = (baseline_total_cost - total_cost) / baseline_total_cost
            self.record_metric(session_id, "cost_reduction", cost_reduction, "percentage",
                              target_value=self.performance_requirements['cost_reduction_min'],
                              target_comparison="greater_than")
            
            self.logger.info(f"Cost reduction: {cost_reduction:.1%} (${baseline_total_cost:.4f} → ${total_cost:.4f})")
        
        # Cost distribution analysis
        if len(costs) > 1:
            cost_std = statistics.stdev(costs)
            self.record_metric(session_id, "cost_std_dev", cost_std, "usd")
            
            # Count zero-cost requests (local model usage)
            zero_cost_requests = sum(1 for cost in costs if cost == 0.0)
            local_model_ratio = zero_cost_requests / len(costs)
            self.record_metric(session_id, "local_model_usage_ratio", local_model_ratio, "percentage")
    
    def record_routing_metrics(self, session_id: str, routing_times: List[float],
                             model_usage: Dict[str, int]) -> None:
        """Record routing-related metrics."""
        if routing_times:
            avg_routing_time = statistics.mean(routing_times)
            max_routing_time = max(routing_times)
            
            self.record_metric(session_id, "avg_routing_time", avg_routing_time, "seconds",
                              target_value=self.performance_requirements['routing_overhead_max'],
                              target_comparison="less_than")
            
            self.record_metric(session_id, "max_routing_time", max_routing_time, "seconds",
                              target_value=0.05, target_comparison="less_than")  # 50ms max
        
        # Model usage distribution
        if model_usage:
            total_requests = sum(model_usage.values())
            usage_entropy = self._calculate_entropy(list(model_usage.values()))
            
            self.record_metric(session_id, "model_usage_entropy", usage_entropy, "bits")
            self.record_metric(session_id, "models_used", len(model_usage), "count")
            
            # Calculate load balancing effectiveness
            if len(model_usage) > 1:
                usage_values = list(model_usage.values())
                usage_variance = statistics.variance(usage_values)
                ideal_usage = total_requests / len(model_usage)
                load_balance_score = 1.0 - (usage_variance / (ideal_usage ** 2))
                
                self.record_metric(session_id, "load_balance_score", load_balance_score, "score",
                                  target_value=0.7, target_comparison="greater_than")
    
    def record_quality_metrics(self, session_id: str, quality_scores: List[float],
                             prediction_accuracies: List[float] = None) -> None:
        """Record quality and accuracy metrics."""
        if quality_scores:
            avg_quality = statistics.mean(quality_scores)
            min_quality = min(quality_scores)
            
            self.record_metric(session_id, "avg_quality_score", avg_quality, "score")
            self.record_metric(session_id, "min_quality_score", min_quality, "score",
                              target_value=0.8, target_comparison="greater_than")
        
        if prediction_accuracies:
            avg_accuracy = statistics.mean(prediction_accuracies)
            self.record_metric(session_id, "avg_prediction_accuracy", avg_accuracy, "percentage",
                              target_value=self.performance_requirements['cost_prediction_accuracy_min'],
                              target_comparison="greater_than")
    
    def finish_benchmark_session(self, session_id: str) -> BenchmarkSession:
        """Finish a benchmark session and perform comprehensive analysis."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found or not active")
        
        session = self.active_sessions[session_id]
        session.finish()
        
        # Perform comprehensive analysis
        self._aggregate_session_metrics(session)
        self._validate_performance_requirements(session)
        self._perform_statistical_analysis(session)
        
        # Move to completed sessions
        self.completed_sessions[session_id] = session
        del self.active_sessions[session_id]
        
        self.logger.info(f"Finished benchmark session: {session.name} ({session_id}) - Duration: {session.duration:.2f}s")
        return session
    
    def _aggregate_session_metrics(self, session: BenchmarkSession) -> None:
        """Aggregate metrics by type for easier analysis."""
        metrics_by_name = defaultdict(list)
        
        for metric in session.metrics:
            metrics_by_name[metric.name].append(metric.value)
        
        # Calculate aggregations
        for metric_name, values in metrics_by_name.items():
            if len(values) > 1:
                session.aggregated_metrics[f"{metric_name}_mean"] = statistics.mean(values)
                session.aggregated_metrics[f"{metric_name}_std"] = statistics.stdev(values)
                session.aggregated_metrics[f"{metric_name}_min"] = min(values)
                session.aggregated_metrics[f"{metric_name}_max"] = max(values)
            elif len(values) == 1:
                session.aggregated_metrics[f"{metric_name}_value"] = values[0]
    
    def _validate_performance_requirements(self, session: BenchmarkSession) -> None:
        """Validate session metrics against performance requirements."""
        validation_results = {}
        
        for metric in session.metrics:
            if metric.target_value is not None:
                passed = self._evaluate_target(metric.value, metric.target_value, metric.target_comparison)
                
                validation_results[metric.name] = {
                    'value': metric.value,
                    'target': metric.target_value,
                    'comparison': metric.target_comparison,
                    'passed': passed,
                    'unit': metric.unit
                }
                
                if not passed:
                    self.logger.warning(f"Performance requirement failed: {metric.name} = {metric.value} {metric.unit} "
                                      f"(target: {metric.target_comparison} {metric.target_value})")
        
        session.validation_results = validation_results
        
        # Calculate overall pass rate
        total_requirements = len(validation_results)
        passed_requirements = sum(1 for result in validation_results.values() if result['passed'])
        pass_rate = passed_requirements / total_requirements if total_requirements > 0 else 0
        
        session.aggregated_metrics['requirements_pass_rate'] = pass_rate
        self.logger.info(f"Performance requirements pass rate: {pass_rate:.1%} ({passed_requirements}/{total_requirements})")
    
    def _perform_statistical_analysis(self, session: BenchmarkSession) -> None:
        """Perform statistical analysis on session metrics."""
        analysis = {}
        
        # Group metrics by type for analysis
        throughput_metrics = [m for m in session.metrics if 'throughput' in m.name or 'rps' in m.name]
        time_metrics = [m for m in session.metrics if 'time' in m.name and m.unit == 'seconds']
        cost_metrics = [m for m in session.metrics if 'cost' in m.name and m.unit == 'usd']
        
        # Statistical significance testing (basic implementation)
        if len(throughput_metrics) > 10:
            throughput_values = [m.value for m in throughput_metrics]
            analysis['throughput_confidence_interval'] = self._calculate_confidence_interval(throughput_values)
        
        if len(time_metrics) > 10:
            time_values = [m.value for m in time_metrics]
            analysis['response_time_confidence_interval'] = self._calculate_confidence_interval(time_values)
        
        # Performance trend analysis (if enough data points)
        if len(session.metrics) > 20:
            analysis['performance_trend'] = self._analyze_performance_trend(session.metrics)
        
        session.statistical_analysis = analysis
    
    def compare_with_baseline(self, session_id: str, baseline_session_id: str) -> Dict[str, Any]:
        """Compare session performance with baseline session."""
        if session_id not in self.completed_sessions:
            raise ValueError(f"Session {session_id} not found in completed sessions")
        
        if baseline_session_id not in self.completed_sessions:
            raise ValueError(f"Baseline session {baseline_session_id} not found")
        
        current_session = self.completed_sessions[session_id]
        baseline_session = self.completed_sessions[baseline_session_id]
        
        comparison = {}
        
        # Compare key metrics
        key_metrics = ['throughput_rps', 'p95_response_time', 'total_cost', 'error_rate']
        
        for metric_name in key_metrics:
            current_value = self._get_metric_value(current_session, metric_name)
            baseline_value = self._get_metric_value(baseline_session, metric_name)
            
            if current_value is not None and baseline_value is not None:
                if metric_name in ['throughput_rps']:
                    # Higher is better
                    improvement = current_value / baseline_value
                    comparison[metric_name] = {
                        'current': current_value,
                        'baseline': baseline_value,
                        'improvement': improvement,
                        'better': improvement > 1.0
                    }
                else:
                    # Lower is better
                    reduction = (baseline_value - current_value) / baseline_value
                    comparison[metric_name] = {
                        'current': current_value,
                        'baseline': baseline_value,
                        'reduction': reduction,
                        'better': reduction > 0.0
                    }
        
        current_session.baseline_comparison = comparison
        return comparison
    
    def generate_performance_report(self, session_id: str) -> Dict[str, Any]:
        """Generate comprehensive performance report for a session."""
        if session_id not in self.completed_sessions:
            raise ValueError(f"Session {session_id} not found in completed sessions")
        
        session = self.completed_sessions[session_id]
        
        report = {
            'session_info': {
                'session_id': session_id,
                'name': session.name,
                'duration': session.duration,
                'start_time': datetime.fromtimestamp(session.start_time).isoformat(),
                'end_time': datetime.fromtimestamp(session.end_time).isoformat() if session.end_time else None
            },
            'performance_summary': self._extract_performance_summary(session),
            'requirements_validation': session.validation_results,
            'statistical_analysis': session.statistical_analysis,
            'baseline_comparison': session.baseline_comparison,
            'recommendations': self._generate_recommendations(session)
        }
        
        return report
    
    def export_session_data(self, session_id: str, file_path: str = None) -> str:
        """Export session data to JSON file."""
        if session_id not in self.completed_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.completed_sessions[session_id]
        
        # Convert to serializable format
        data = {
            'session': asdict(session),
            'metrics': [asdict(metric) for metric in session.metrics]
        }
        
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"benchmark_session_{session_id}_{timestamp}.json"
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Exported session data to {file_path}")
        return file_path
    
    def _evaluate_target(self, value: float, target: float, comparison: str) -> bool:
        """Evaluate if a value meets its target based on comparison type."""
        if comparison == "less_than":
            return value < target
        elif comparison == "greater_than":
            return value > target
        elif comparison == "equals":
            return abs(value - target) < 0.01  # Small tolerance for float comparison
        else:
            return False
    
    def _calculate_entropy(self, values: List[int]) -> float:
        """Calculate Shannon entropy for distribution analysis."""
        if not values or sum(values) == 0:
            return 0.0
        
        total = sum(values)
        probabilities = [v / total for v in values if v > 0]
        
        entropy = -sum(p * math.log2(p) for p in probabilities)
        return entropy
    
    def _calculate_confidence_interval(self, values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for a set of values."""
        if len(values) < 2:
            return (0.0, 0.0)
        
        mean = statistics.mean(values)
        std_err = statistics.stdev(values) / math.sqrt(len(values))
        
        # Use t-distribution critical value (approximation for normal)
        t_critical = 1.96 if confidence == 0.95 else 2.58  # 99% confidence
        
        margin_error = t_critical * std_err
        return (mean - margin_error, mean + margin_error)
    
    def _analyze_performance_trend(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        # Group metrics by name and analyze trends
        trends = {}
        
        metrics_by_name = defaultdict(list)
        for metric in metrics:
            metrics_by_name[metric.name].append((metric.timestamp, metric.value))
        
        for metric_name, data_points in metrics_by_name.items():
            if len(data_points) < 5:  # Need enough points for trend analysis
                continue
                
            # Sort by timestamp
            data_points.sort(key=lambda x: x[0])
            
            # Simple linear trend analysis
            times = [dp[0] for dp in data_points]
            values = [dp[1] for dp in data_points]
            
            # Calculate trend slope (simplified)
            if len(times) > 1:
                time_range = times[-1] - times[0]
                value_change = values[-1] - values[0]
                
                if time_range > 0:
                    trend_slope = value_change / time_range
                    trends[metric_name] = {
                        'slope': trend_slope,
                        'direction': 'improving' if trend_slope < 0 else 'degrading' if trend_slope > 0 else 'stable',
                        'data_points': len(data_points)
                    }
        
        return trends
    
    def _get_metric_value(self, session: BenchmarkSession, metric_name: str) -> Optional[float]:
        """Get metric value from session."""
        for metric in session.metrics:
            if metric.name == metric_name:
                return metric.value
        return None
    
    def _extract_performance_summary(self, session: BenchmarkSession) -> Dict[str, Any]:
        """Extract key performance metrics summary."""
        summary = {}
        
        key_metrics = [
            'throughput_rps', 'p95_response_time', 'p99_response_time',
            'total_cost', 'error_rate', 'avg_routing_time'
        ]
        
        for metric_name in key_metrics:
            value = self._get_metric_value(session, metric_name)
            if value is not None:
                summary[metric_name] = value
        
        # Add derived metrics
        if 'throughput_improvement' in session.aggregated_metrics:
            summary['throughput_improvement'] = session.aggregated_metrics['throughput_improvement']
        
        if 'cost_reduction' in session.aggregated_metrics:
            summary['cost_reduction'] = session.aggregated_metrics['cost_reduction']
        
        return summary
    
    def _generate_recommendations(self, session: BenchmarkSession) -> List[str]:
        """Generate performance recommendations based on session results."""
        recommendations = []
        
        # Check for failed requirements
        failed_requirements = [
            name for name, result in session.validation_results.items()
            if not result['passed']
        ]
        
        if 'p95_response_time' in failed_requirements:
            recommendations.append("Consider using speed_first routing strategy for improved response times")
        
        if 'error_rate' in failed_requirements:
            recommendations.append("Investigate error causes and implement additional fault tolerance")
        
        if 'cost_reduction' in failed_requirements:
            recommendations.append("Increase local model usage ratio for better cost optimization")
        
        if 'throughput_improvement' in failed_requirements:
            recommendations.append("Consider increasing concurrent workers or optimizing routing overhead")
        
        # Check for performance trends
        if session.statistical_analysis.get('performance_trend'):
            trends = session.statistical_analysis['performance_trend']
            for metric, trend_data in trends.items():
                if trend_data['direction'] == 'degrading':
                    recommendations.append(f"Monitor {metric} trend - showing degradation over time")
        
        return recommendations


def create_metrics_collector() -> PerformanceMetricsCollector:
    """Factory function to create a performance metrics collector."""
    return PerformanceMetricsCollector()