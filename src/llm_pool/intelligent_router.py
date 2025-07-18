"""
Intelligent Request Routing Engine for Multi-Model Load Balancing.

This module provides sophisticated request routing capabilities that analyze
request characteristics and intelligently route to the most suitable LLM model
based on multiple criteria including complexity, cost, speed, and quality requirements.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import re
import json
from collections import defaultdict

from config import settings
from .llm_pool_manager import LLMInstance, LLMRequest, LLMPoolManager


class RequestComplexity(Enum):
    """Request complexity levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    BATCH = "batch"
    HEAVY = "heavy"


class RoutingStrategy(Enum):
    """Available routing strategies."""
    SPEED_FIRST = "speed_first"
    QUALITY_FIRST = "quality_first"
    COST_FIRST = "cost_first"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"


@dataclass
class RequestCharacteristics:
    """Analyzed characteristics of a request."""
    complexity: RequestComplexity
    estimated_tokens: int
    use_case: str
    priority: int
    requires_json: bool
    context_length: int
    language: str
    content_type: str  # text, code, creative, analytical, etc.
    urgency: str  # low, medium, high, critical
    quality_requirements: str  # basic, medium, high
    cost_constraints: float  # maximum acceptable cost


@dataclass
class RoutingDecision:
    """Result of routing decision with detailed rationale."""
    selected_instance: LLMInstance
    confidence_score: float
    reasoning: List[str]
    alternatives: List[Dict[str, Any]]
    estimated_cost: float
    estimated_response_time: float
    fallback_chain: List[str]
    routing_strategy_used: str


class IntelligentRequestRouter:
    """
    Intelligent Request Routing Engine for Multi-Model Load Balancing.
    
    This class implements sophisticated request analysis and routing algorithms
    to optimally distribute requests across multiple LLM models based on:
    - Request complexity and characteristics
    - Model capabilities and performance
    - Cost optimization strategies
    - Quality requirements
    - Real-time load balancing
    """
    
    def __init__(self, pool_manager: LLMPoolManager):
        """Initialize the intelligent request router."""
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Routing statistics and learning
        self.routing_history: List[Dict[str, Any]] = []
        self.model_performance_cache: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.request_patterns: Dict[str, int] = defaultdict(int)
        
        # Cost tracking
        self.hourly_costs: Dict[str, float] = defaultdict(float)
        self.daily_costs: Dict[str, float] = defaultdict(float)
        
        self.logger.info("Intelligent Request Router initialized")
    
    def route_request(self, request: LLMRequest, 
                     routing_strategy: str = None) -> RoutingDecision:
        """
        Route a request to the most suitable LLM instance.
        
        Args:
            request: The LLM request to route
            routing_strategy: Override default routing strategy
            
        Returns:
            RoutingDecision with selected instance and reasoning
        """
        start_time = time.time()
        
        # Analyze request characteristics
        characteristics = self._analyze_request_characteristics(request)
        
        # Get available instances
        available_instances = self._get_available_instances()
        if not available_instances:
            raise Exception("No available LLM instances")
        
        # Determine routing strategy
        strategy = self._determine_routing_strategy(routing_strategy, characteristics)
        
        # Score and rank instances
        scored_instances = self._score_instances(available_instances, characteristics, strategy)
        
        # Select best instance with fallback chain
        decision = self._make_routing_decision(scored_instances, characteristics, strategy)
        
        # Update routing history and statistics
        self._update_routing_history(decision, characteristics, time.time() - start_time)
        
        self.logger.info(f"Routed request to {decision.selected_instance.model_name} "
                        f"(strategy: {strategy}, confidence: {decision.confidence_score:.2f})")
        
        return decision
    
    def _analyze_request_characteristics(self, request: LLMRequest) -> RequestCharacteristics:
        """Analyze request to determine its characteristics and requirements."""
        prompt = request.prompt
        model_config = request.model_config
        
        # Estimate complexity based on multiple factors
        complexity = self._estimate_complexity(prompt, model_config)
        
        # Estimate token count
        estimated_tokens = self._estimate_token_count(prompt)
        
        # Detect use case and content type
        use_case = self._detect_use_case(prompt, model_config)
        content_type = self._detect_content_type(prompt)
        
        # Determine quality requirements
        quality_requirements = model_config.get('quality_requirements', 'medium')
        
        # Determine urgency
        urgency = self._determine_urgency(request, model_config)
        
        # Check if JSON output is required
        requires_json = self._requires_json_output(prompt, model_config)
        
        # Detect language
        language = self._detect_language(prompt)
        
        return RequestCharacteristics(
            complexity=complexity,
            estimated_tokens=estimated_tokens,
            use_case=use_case,
            priority=request.priority,
            requires_json=requires_json,
            context_length=len(prompt),
            language=language,
            content_type=content_type,
            urgency=urgency,
            quality_requirements=quality_requirements,
            cost_constraints=model_config.get('max_cost', float('inf'))
        )
    
    def _estimate_complexity(self, prompt: str, model_config: Dict[str, Any]) -> RequestComplexity:
        """Estimate request complexity using multiple heuristics."""
        prompt_length = len(prompt)
        word_count = len(prompt.split())
        
        # Check for complexity indicators
        complexity_indicators = {
            'batch': ['analyze', 'process', 'multiple', 'list', 'batch', 'all'],
            'complex': ['reason', 'explain', 'analyze', 'compare', 'evaluate', 'synthesize'],
            'medium': ['classify', 'categorize', 'identify', 'extract', 'summarize'],
            'simple': ['yes', 'no', 'true', 'false', 'short', 'brief']
        }
        
        prompt_lower = prompt.lower()
        complexity_scores = {}
        
        for complexity_level, indicators in complexity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in prompt_lower)
            complexity_scores[complexity_level] = score
        
        # Determine complexity based on length and indicators
        if prompt_length > 20000 or word_count > 4000:
            return RequestComplexity.HEAVY
        elif prompt_length > 10000 or word_count > 2000 or complexity_scores.get('batch', 0) > 1:
            return RequestComplexity.BATCH
        elif prompt_length > 2000 or word_count > 400 or complexity_scores.get('complex', 0) > 0:
            return RequestComplexity.COMPLEX
        elif prompt_length > 500 or word_count > 100 or complexity_scores.get('medium', 0) > 0:
            return RequestComplexity.MEDIUM
        else:
            return RequestComplexity.SIMPLE
    
    def _estimate_token_count(self, prompt: str) -> int:
        """Estimate token count for the prompt."""
        # Simple approximation: ~4 characters per token for English text
        return max(len(prompt) // 4, len(prompt.split()))
    
    def _detect_use_case(self, prompt: str, model_config: Dict[str, Any]) -> str:
        """Detect the use case for the request."""
        # Check explicit use case in config
        if 'use_case' in model_config:
            return model_config['use_case']
        
        # Detect from prompt content
        use_case_patterns = {
            'classification': ['classify', 'categorize', 'label', 'tag'],
            'extraction': ['extract', 'find', 'identify', 'locate'],
            'analysis': ['analyze', 'examine', 'evaluate', 'assess'],
            'generation': ['generate', 'create', 'write', 'compose'],
            'translation': ['translate', 'convert', 'transform'],
            'summarization': ['summarize', 'condense', 'abstract'],
            'question_answering': ['what', 'why', 'how', 'when', 'where'],
            'coding': ['code', 'function', 'class', 'method', 'algorithm'],
            'creative_writing': ['story', 'poem', 'creative', 'fiction']
        }
        
        prompt_lower = prompt.lower()
        for use_case, patterns in use_case_patterns.items():
            if any(pattern in prompt_lower for pattern in patterns):
                return use_case
        
        return 'general'
    
    def _detect_content_type(self, prompt: str) -> str:
        """Detect the type of content in the prompt."""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['code', 'function', 'class', 'def ', 'import']):
            return 'code'
        elif any(word in prompt_lower for word in ['story', 'poem', 'creative', 'fiction']):
            return 'creative'
        elif any(word in prompt_lower for word in ['analyze', 'evaluate', 'research']):
            return 'analytical'
        elif any(word in prompt_lower for word in ['data', 'table', 'csv', 'json']):
            return 'data'
        else:
            return 'text'
    
    def _determine_urgency(self, request: LLMRequest, model_config: Dict[str, Any]) -> str:
        """Determine request urgency based on timeout and priority."""
        timeout = request.timeout
        priority = request.priority
        
        if priority >= 8 or (timeout and timeout < 30):
            return 'critical'
        elif priority >= 5 or (timeout and timeout < 60):
            return 'high'
        elif priority >= 2 or (timeout and timeout < 120):
            return 'medium'
        else:
            return 'low'
    
    def _requires_json_output(self, prompt: str, model_config: Dict[str, Any]) -> bool:
        """Check if the request requires JSON output."""
        if model_config.get('format') == 'json':
            return True
        
        json_indicators = ['json', 'javascript object notation', '{"', '[{']
        prompt_lower = prompt.lower()
        return any(indicator in prompt_lower for indicator in json_indicators)
    
    def _detect_language(self, prompt: str) -> str:
        """Detect the primary language of the prompt."""
        # Simple heuristic - could be enhanced with language detection library
        english_indicators = len(re.findall(r'\b(?:the|and|is|in|to|of|a|that|it|with|for|as|was|on|are|you)\b', prompt.lower()))
        total_words = len(prompt.split())
        
        if total_words > 0 and english_indicators / total_words > 0.1:
            return 'english'
        else:
            return 'unknown'
    
    def _get_available_instances(self) -> List[LLMInstance]:
        """Get available instances from pool manager."""
        return [
            instance for instance in self.pool_manager.instances.values()
            if instance.is_available()
        ]
    
    def _determine_routing_strategy(self, override_strategy: str, 
                                 characteristics: RequestCharacteristics) -> RoutingStrategy:
        """Determine the routing strategy to use."""
        if override_strategy:
            try:
                return RoutingStrategy(override_strategy)
            except ValueError:
                self.logger.warning(f"Invalid routing strategy: {override_strategy}, using default")
        
        # Check pool configuration
        pool_config = self.pool_manager.config.get('pool_config', {})
        config_strategy = pool_config.get('routing_strategy', 'balanced')
        
        try:
            return RoutingStrategy(config_strategy)
        except ValueError:
            return RoutingStrategy.BALANCED
    
    def _score_instances(self, instances: List[LLMInstance], 
                        characteristics: RequestCharacteristics,
                        strategy: RoutingStrategy) -> List[Tuple[LLMInstance, float, Dict[str, float]]]:
        """Score instances based on their suitability for the request."""
        scored_instances = []
        
        for instance in instances:
            score, score_breakdown = self._calculate_instance_score(instance, characteristics, strategy)
            scored_instances.append((instance, score, score_breakdown))
        
        # Sort by score (descending)
        scored_instances.sort(key=lambda x: x[1], reverse=True)
        return scored_instances
    
    def _calculate_instance_score(self, instance: LLMInstance, 
                                characteristics: RequestCharacteristics,
                                strategy: RoutingStrategy) -> Tuple[float, Dict[str, float]]:
        """Calculate a comprehensive score for an instance."""
        score_breakdown = {}
        
        # Base scores from instance
        health_score = instance.health_score
        suitability_score = instance.suitability_score
        
        # Use case compatibility
        use_case_score = 1.0 if instance.is_suitable_for_use_case(characteristics.use_case) else 0.5
        
        # Cost score (lower cost = higher score)
        estimated_cost = instance.get_estimated_cost(characteristics.estimated_tokens)
        if estimated_cost <= characteristics.cost_constraints:
            cost_score = 1.0 - min(estimated_cost / max(characteristics.cost_constraints, 0.01), 1.0)
        else:
            cost_score = 0.0  # Exceeds budget
        
        # Quality score based on requirements
        quality_score = self._calculate_quality_score(instance, characteristics)
        
        # Speed score based on urgency
        speed_score = self._calculate_speed_score(instance, characteristics)
        
        # Capacity score based on current load
        capacity_score = 1.0 - (instance.current_load / instance.max_load)
        
        # JSON support score
        json_score = 1.0 if not characteristics.requires_json or instance.model_capabilities.get('supports_json', True) else 0.3
        
        # Store individual scores
        score_breakdown = {
            'health_score': health_score,
            'suitability_score': suitability_score,
            'use_case_score': use_case_score,
            'cost_score': cost_score,
            'quality_score': quality_score,
            'speed_score': speed_score,
            'capacity_score': capacity_score,
            'json_score': json_score
        }
        
        # Apply strategy-specific weighting
        final_score = self._apply_strategy_weighting(score_breakdown, strategy)
        
        return final_score, score_breakdown
    
    def _calculate_quality_score(self, instance: LLMInstance, 
                               characteristics: RequestCharacteristics) -> float:
        """Calculate quality score based on requirements."""
        base_quality = instance.quality_score
        
        quality_requirements = characteristics.quality_requirements
        if quality_requirements == 'basic':
            return min(base_quality * 1.2, 1.0)  # Bonus for basic requirements
        elif quality_requirements == 'high':
            return base_quality * 0.8 if base_quality < 0.8 else base_quality  # Penalty for low quality
        else:  # medium
            return base_quality
    
    def _calculate_speed_score(self, instance: LLMInstance, 
                             characteristics: RequestCharacteristics) -> float:
        """Calculate speed score based on urgency."""
        base_speed = instance.speed_score
        
        urgency = characteristics.urgency
        if urgency == 'critical':
            return base_speed * 1.5  # Strong preference for fast models
        elif urgency == 'high':
            return base_speed * 1.2
        elif urgency == 'low':
            return min(base_speed * 0.8, 1.0)  # Less emphasis on speed
        else:  # medium
            return base_speed
    
    def _apply_strategy_weighting(self, scores: Dict[str, float], 
                                strategy: RoutingStrategy) -> float:
        """Apply strategy-specific weighting to scores."""
        if strategy == RoutingStrategy.SPEED_FIRST:
            weights = {
                'speed_score': 0.4, 'capacity_score': 0.2, 'health_score': 0.15,
                'quality_score': 0.1, 'cost_score': 0.05, 'use_case_score': 0.05,
                'suitability_score': 0.025, 'json_score': 0.025
            }
        elif strategy == RoutingStrategy.QUALITY_FIRST:
            weights = {
                'quality_score': 0.4, 'health_score': 0.2, 'use_case_score': 0.15,
                'suitability_score': 0.1, 'speed_score': 0.05, 'capacity_score': 0.05,
                'cost_score': 0.025, 'json_score': 0.025
            }
        elif strategy == RoutingStrategy.COST_FIRST:
            weights = {
                'cost_score': 0.5, 'health_score': 0.2, 'capacity_score': 0.1,
                'speed_score': 0.1, 'quality_score': 0.05, 'use_case_score': 0.025,
                'suitability_score': 0.02, 'json_score': 0.005
            }
        elif strategy == RoutingStrategy.ADAPTIVE:
            # Adaptive strategy based on request characteristics
            weights = self._get_adaptive_weights(scores)
        else:  # BALANCED
            weights = {
                'health_score': 0.2, 'suitability_score': 0.15, 'use_case_score': 0.15,
                'quality_score': 0.15, 'speed_score': 0.1, 'cost_score': 0.1,
                'capacity_score': 0.1, 'json_score': 0.05
            }
        
        # Calculate weighted score
        final_score = sum(scores[key] * weights.get(key, 0.0) for key in scores)
        return min(final_score, 1.0)
    
    def _get_adaptive_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Get adaptive weights based on current system state and scores."""
        # Simple adaptive strategy - could be enhanced with ML
        base_weights = {
            'health_score': 0.2, 'suitability_score': 0.15, 'use_case_score': 0.15,
            'quality_score': 0.15, 'speed_score': 0.1, 'cost_score': 0.1,
            'capacity_score': 0.1, 'json_score': 0.05
        }
        
        # Boost capacity score if system is under high load
        total_load = sum(i.current_load for i in self.pool_manager.instances.values())
        total_capacity = sum(i.max_load for i in self.pool_manager.instances.values())
        load_ratio = total_load / max(total_capacity, 1)
        
        if load_ratio > 0.8:
            base_weights['capacity_score'] *= 2.0
            base_weights['health_score'] *= 1.5
        
        return base_weights
    
    def _make_routing_decision(self, scored_instances: List[Tuple[LLMInstance, float, Dict[str, float]]], 
                             characteristics: RequestCharacteristics,
                             strategy: RoutingStrategy) -> RoutingDecision:
        """Make the final routing decision with fallback chain."""
        if not scored_instances:
            raise Exception("No suitable instances available")
        
        # Select best instance
        best_instance, best_score, best_breakdown = scored_instances[0]
        
        # Create fallback chain (top 3 alternatives)
        fallback_chain = [inst[0].instance_id for inst in scored_instances[1:4]]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(best_instance, best_breakdown, characteristics, strategy)
        
        # Create alternatives list
        alternatives = []
        for instance, score, breakdown in scored_instances[1:4]:
            alternatives.append({
                'instance_id': instance.instance_id,
                'model_name': instance.model_name,
                'score': score,
                'reasoning': f"Alternative with score {score:.2f}"
            })
        
        return RoutingDecision(
            selected_instance=best_instance,
            confidence_score=best_score,
            reasoning=reasoning,
            alternatives=alternatives,
            estimated_cost=best_instance.get_estimated_cost(characteristics.estimated_tokens),
            estimated_response_time=best_instance.response_time_avg,
            fallback_chain=fallback_chain,
            routing_strategy_used=strategy.value
        )
    
    def _generate_reasoning(self, instance: LLMInstance, score_breakdown: Dict[str, float],
                          characteristics: RequestCharacteristics, strategy: RoutingStrategy) -> List[str]:
        """Generate human-readable reasoning for the routing decision."""
        reasoning = []
        
        # Strategy-based reasoning
        reasoning.append(f"Using {strategy.value} routing strategy")
        
        # Top scoring factors
        top_factors = sorted(score_breakdown.items(), key=lambda x: x[1], reverse=True)[:3]
        for factor, score in top_factors:
            if score > 0.7:
                reasoning.append(f"High {factor.replace('_', ' ')}: {score:.2f}")
        
        # Model-specific factors
        if instance.engine_type == "local":
            reasoning.append("Cost-effective local model")
        
        if instance.is_suitable_for_use_case(characteristics.use_case):
            reasoning.append(f"Optimized for {characteristics.use_case}")
        
        if characteristics.urgency == "critical" and instance.speed_score > 0.8:
            reasoning.append("Fast response for critical request")
        
        return reasoning
    
    def _update_routing_history(self, decision: RoutingDecision, 
                              characteristics: RequestCharacteristics, routing_time: float):
        """Update routing history for analytics and learning."""
        history_entry = {
            'timestamp': time.time(),
            'model_selected': decision.selected_instance.model_name,
            'instance_id': decision.selected_instance.instance_id,
            'confidence_score': decision.confidence_score,
            'complexity': characteristics.complexity.value,
            'use_case': characteristics.use_case,
            'estimated_cost': decision.estimated_cost,
            'routing_time_ms': routing_time * 1000,
            'routing_strategy': decision.routing_strategy_used
        }
        
        self.routing_history.append(history_entry)
        
        # Keep only recent history (last 1000 entries)
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]
        
        # Update request patterns
        pattern_key = f"{characteristics.complexity.value}_{characteristics.use_case}"
        self.request_patterns[pattern_key] += 1
    
    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get comprehensive routing analytics."""
        if not self.routing_history:
            return {"status": "No routing history available"}
        
        recent_history = self.routing_history[-100:]  # Last 100 requests
        
        # Model usage distribution
        model_usage = defaultdict(int)
        strategy_usage = defaultdict(int)
        complexity_distribution = defaultdict(int)
        
        total_cost = 0.0
        total_routing_time = 0.0
        
        for entry in recent_history:
            model_usage[entry['model_selected']] += 1
            strategy_usage[entry['routing_strategy']] += 1
            complexity_distribution[entry['complexity']] += 1
            total_cost += entry['estimated_cost']
            total_routing_time += entry['routing_time_ms']
        
        return {
            'total_requests': len(recent_history),
            'model_usage_distribution': dict(model_usage),
            'strategy_usage_distribution': dict(strategy_usage),
            'complexity_distribution': dict(complexity_distribution),
            'average_confidence_score': sum(e['confidence_score'] for e in recent_history) / len(recent_history),
            'total_estimated_cost': total_cost,
            'average_routing_time_ms': total_routing_time / len(recent_history),
            'most_common_patterns': dict(sorted(self.request_patterns.items(), 
                                              key=lambda x: x[1], reverse=True)[:10])
        }
    
    def optimize_routing_strategy(self) -> Dict[str, Any]:
        """Analyze routing performance and suggest optimizations."""
        analytics = self.get_routing_analytics()
        optimizations = []
        
        if analytics.get('total_requests', 0) < 10:
            return {"status": "Insufficient data for optimization analysis"}
        
        # Analyze model utilization
        model_usage = analytics.get('model_usage_distribution', {})
        if model_usage:
            max_usage = max(model_usage.values())
            min_usage = min(model_usage.values())
            
            if max_usage > min_usage * 3:
                optimizations.append("Consider load balancing - some models are overutilized")
        
        # Analyze cost efficiency
        avg_cost = analytics.get('total_estimated_cost', 0) / analytics.get('total_requests', 1)
        if avg_cost > 0.01:  # High cost threshold
            optimizations.append("Consider cost optimization - high average request cost")
        
        # Analyze routing time
        avg_routing_time = analytics.get('average_routing_time_ms', 0)
        if avg_routing_time > 50:  # High routing overhead
            optimizations.append("Consider routing optimization - high routing overhead")
        
        return {
            'optimizations': optimizations,
            'current_performance': analytics,
            'recommendations': self._generate_optimization_recommendations(analytics)
        }
    
    def _generate_optimization_recommendations(self, analytics: Dict[str, Any]) -> List[str]:
        """Generate specific optimization recommendations."""
        recommendations = []
        
        # Cost optimization
        if analytics.get('total_estimated_cost', 0) > 1.0:
            recommendations.append("Enable cost_first routing strategy for budget-sensitive requests")
        
        # Performance optimization  
        if analytics.get('average_routing_time_ms', 0) > 30:
            recommendations.append("Consider caching routing decisions for similar request patterns")
        
        # Load balancing
        model_usage = analytics.get('model_usage_distribution', {})
        if model_usage and len(model_usage) > 1:
            usage_variance = max(model_usage.values()) - min(model_usage.values())
            if usage_variance > len(analytics.get('total_requests', 0)) * 0.3:
                recommendations.append("Improve load distribution across available models")
        
        return recommendations