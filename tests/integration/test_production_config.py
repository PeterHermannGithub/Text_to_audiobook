"""
Production Configuration Validation Suite for Multi-Model Load Balancing.

This module provides comprehensive validation of production configuration
settings, ensuring all model capabilities, pool configurations, and system
settings are correctly loaded and applied.
"""

import unittest
import json
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch
import logging

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.llm_pool.llm_pool_manager import LLMPoolManager, LLMInstance
from src.llm_pool.intelligent_router import IntelligentRequestRouter, RoutingStrategy
from src.llm_pool.performance_analytics import PerformanceAnalytics, CostTracker
from tests.utils.mock_llm_framework import create_mock_framework
from config import settings


class ConfigurationValidationResults:
    """Container for configuration validation results."""
    
    def __init__(self):
        self.total_validations = 0
        self.passed_validations = 0
        self.failed_validations = 0
        self.validation_details = []
        self.critical_failures = []
        self.warnings = []
    
    def add_validation(self, name: str, passed: bool, details: str = "", critical: bool = False):
        """Add a validation result."""
        self.total_validations += 1
        
        if passed:
            self.passed_validations += 1
        else:
            self.failed_validations += 1
            if critical:
                self.critical_failures.append(f"{name}: {details}")
            else:
                self.warnings.append(f"{name}: {details}")
        
        self.validation_details.append({
            'name': name,
            'passed': passed,
            'details': details,
            'critical': critical
        })
    
    @property
    def pass_rate(self) -> float:
        """Calculate validation pass rate."""
        return self.passed_validations / max(self.total_validations, 1)
    
    @property
    def has_critical_failures(self) -> bool:
        """Check if there are critical failures."""
        return len(self.critical_failures) > 0


class TestProductionConfiguration(unittest.TestCase):
    """Production configuration validation test suite."""
    
    @classmethod
    def setUpClass(cls):
        """Set up production configuration test fixtures."""
        cls.mock_framework = create_mock_framework()
        cls.validation_results = ConfigurationValidationResults()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)
    
    def setUp(self):
        """Set up for each test."""
        # Reset validation results for each test category
        pass
    
    def test_model_capabilities_configuration(self):
        """Test that all model capabilities are properly configured."""
        self.logger.info("Validating model capabilities configuration")
        
        results = ConfigurationValidationResults()
        
        # Validate MODEL_CAPABILITIES exists and is properly structured
        results.add_validation(
            "MODEL_CAPABILITIES_EXISTS",
            hasattr(settings, 'MODEL_CAPABILITIES'),
            "MODEL_CAPABILITIES must be defined in settings",
            critical=True
        )
        
        if not hasattr(settings, 'MODEL_CAPABILITIES'):
            self.fail("MODEL_CAPABILITIES not found in settings")
        
        model_capabilities = settings.MODEL_CAPABILITIES
        
        # Validate each model configuration
        required_capability_fields = [
            'engine', 'provider', 'speed_tier', 'quality_tier', 'cost_tier',
            'max_context_length', 'supports_json', 'optimal_use_cases',
            'cost_per_1k_tokens', 'avg_response_time_ms', 'reliability_score'
        ]
        
        for model_name, capabilities in model_capabilities.items():
            # Check all required fields are present
            for field in required_capability_fields:
                field_present = field in capabilities
                results.add_validation(
                    f"MODEL_{model_name.replace(':', '_').replace('.', '_').upper()}_{field.upper()}",
                    field_present,
                    f"Field '{field}' missing for model {model_name}",
                    critical=True
                )
            
            # Validate field values
            if 'engine' in capabilities:
                valid_engines = ['local', 'gcp']
                engine_valid = capabilities['engine'] in valid_engines
                results.add_validation(
                    f"MODEL_{model_name.replace(':', '_').replace('.', '_').upper()}_ENGINE_VALID",
                    engine_valid,
                    f"Invalid engine '{capabilities['engine']}' for {model_name}, must be one of {valid_engines}",
                    critical=True
                )
            
            if 'speed_tier' in capabilities:
                valid_speed_tiers = ['slow', 'medium', 'fast']
                speed_valid = capabilities['speed_tier'] in valid_speed_tiers
                results.add_validation(
                    f"MODEL_{model_name.replace(':', '_').replace('.', '_').upper()}_SPEED_TIER_VALID",
                    speed_valid,
                    f"Invalid speed_tier '{capabilities['speed_tier']}' for {model_name}",
                    critical=False
                )
            
            if 'quality_tier' in capabilities:
                valid_quality_tiers = ['basic', 'medium', 'high']
                quality_valid = capabilities['quality_tier'] in valid_quality_tiers
                results.add_validation(
                    f"MODEL_{model_name.replace(':', '_').replace('.', '_').upper()}_QUALITY_TIER_VALID",
                    quality_valid,
                    f"Invalid quality_tier '{capabilities['quality_tier']}' for {model_name}",
                    critical=False
                )
            
            if 'cost_tier' in capabilities:
                valid_cost_tiers = ['free', 'low', 'medium', 'high']
                cost_valid = capabilities['cost_tier'] in valid_cost_tiers
                results.add_validation(
                    f"MODEL_{model_name.replace(':', '_').replace('.', '_').upper()}_COST_TIER_VALID",
                    cost_valid,
                    f"Invalid cost_tier '{capabilities['cost_tier']}' for {model_name}",
                    critical=False
                )
            
            # Validate numeric fields
            if 'reliability_score' in capabilities:
                reliability = capabilities['reliability_score']
                reliability_valid = isinstance(reliability, (int, float)) and 0.0 <= reliability <= 1.0
                results.add_validation(
                    f"MODEL_{model_name.replace(':', '_').replace('.', '_').upper()}_RELIABILITY_VALID",
                    reliability_valid,
                    f"Invalid reliability_score {reliability} for {model_name}, must be 0.0-1.0",
                    critical=True
                )
            
            if 'max_context_length' in capabilities:
                context_length = capabilities['max_context_length']
                context_valid = isinstance(context_length, int) and context_length > 0
                results.add_validation(
                    f"MODEL_{model_name.replace(':', '_').replace('.', '_').upper()}_CONTEXT_LENGTH_VALID",
                    context_valid,
                    f"Invalid max_context_length {context_length} for {model_name}",
                    critical=True
                )
        
        # Validate minimum number of models
        min_models = 3
        enough_models = len(model_capabilities) >= min_models
        results.add_validation(
            "SUFFICIENT_MODELS_CONFIGURED",
            enough_models,
            f"Need at least {min_models} models, found {len(model_capabilities)}",
            critical=True
        )
        
        # Validate mix of local and cloud models
        local_models = [m for m, c in model_capabilities.items() if c.get('engine') == 'local']
        cloud_models = [m for m, c in model_capabilities.items() if c.get('engine') == 'gcp']
        
        has_local = len(local_models) > 0
        has_cloud = len(cloud_models) > 0
        
        results.add_validation(
            "HAS_LOCAL_MODELS",
            has_local,
            f"No local models found. Local models: {local_models}",
            critical=False
        )
        
        results.add_validation(
            "HAS_CLOUD_MODELS", 
            has_cloud,
            f"No cloud models found. Cloud models: {cloud_models}",
            critical=False
        )
        
        # Validate configuration completeness
        self.assertFalse(results.has_critical_failures,
                        f"Critical model capability configuration failures: {results.critical_failures}")
        
        self.assertGreater(results.pass_rate, 0.9,
                          f"Model capabilities validation pass rate too low: {results.pass_rate:.2%}")
        
        self.logger.info(f"Model capabilities validation - Pass rate: {results.pass_rate:.2%}, "
                        f"Models: {len(model_capabilities)}, "
                        f"Local: {len(local_models)}, Cloud: {len(cloud_models)}")
    
    def test_pool_configuration_validation(self):
        """Test that pool configurations are properly structured."""
        self.logger.info("Validating pool configuration")
        
        results = ConfigurationValidationResults()
        
        # Validate MULTI_MODEL_POOLS exists
        results.add_validation(
            "MULTI_MODEL_POOLS_EXISTS",
            hasattr(settings, 'MULTI_MODEL_POOLS'),
            "MULTI_MODEL_POOLS must be defined in settings",
            critical=True
        )
        
        if not hasattr(settings, 'MULTI_MODEL_POOLS'):
            self.fail("MULTI_MODEL_POOLS not found in settings")
        
        pools = settings.MULTI_MODEL_POOLS
        
        # Validate pool structure
        required_pool_fields = [
            'models', 'routing_strategy', 'failover_enabled',
            'max_instances_per_model', 'health_check_interval'
        ]
        
        for pool_name, pool_config in pools.items():
            for field in required_pool_fields:
                field_present = field in pool_config
                results.add_validation(
                    f"POOL_{pool_name.upper()}_{field.upper()}_EXISTS",
                    field_present,
                    f"Field '{field}' missing in pool '{pool_name}'",
                    critical=True
                )
            
            # Validate routing strategy
            if 'routing_strategy' in pool_config:
                valid_strategies = ['speed_first', 'quality_first', 'cost_first', 'balanced', 'adaptive']
                strategy_valid = pool_config['routing_strategy'] in valid_strategies
                results.add_validation(
                    f"POOL_{pool_name.upper()}_STRATEGY_VALID",
                    strategy_valid,
                    f"Invalid routing strategy '{pool_config['routing_strategy']}' in pool '{pool_name}'",
                    critical=True
                )
            
            # Validate models list
            if 'models' in pool_config:
                models = pool_config['models']
                models_valid = isinstance(models, list) and len(models) > 0
                results.add_validation(
                    f"POOL_{pool_name.upper()}_MODELS_VALID",
                    models_valid,
                    f"Pool '{pool_name}' must have non-empty models list",
                    critical=True
                )
                
                # Validate models exist in MODEL_CAPABILITIES
                if models_valid:
                    for model in models:
                        model_exists = model in settings.MODEL_CAPABILITIES
                        results.add_validation(
                            f"POOL_{pool_name.upper()}_MODEL_{model.replace(':', '_').replace('.', '_').upper()}_EXISTS",
                            model_exists,
                            f"Model '{model}' in pool '{pool_name}' not found in MODEL_CAPABILITIES",
                            critical=True
                        )
        
        # Validate minimum pools
        min_pools = 2
        enough_pools = len(pools) >= min_pools
        results.add_validation(
            "SUFFICIENT_POOLS_CONFIGURED",
            enough_pools,
            f"Need at least {min_pools} pools for redundancy, found {len(pools)}",
            critical=False
        )
        
        # Validate configuration completeness
        self.assertFalse(results.has_critical_failures,
                        f"Critical pool configuration failures: {results.critical_failures}")
        
        self.logger.info(f"Pool configuration validation - Pass rate: {results.pass_rate:.2%}, "
                        f"Pools: {list(pools.keys())}")
    
    def test_cost_management_configuration(self):
        """Test cost management configuration settings."""
        self.logger.info("Validating cost management configuration")
        
        results = ConfigurationValidationResults()
        
        # Validate COST_MANAGEMENT exists
        results.add_validation(
            "COST_MANAGEMENT_EXISTS",
            hasattr(settings, 'COST_MANAGEMENT'),
            "COST_MANAGEMENT must be defined in settings",
            critical=True
        )
        
        if hasattr(settings, 'COST_MANAGEMENT'):
            cost_config = settings.COST_MANAGEMENT
            
            # Validate required fields
            required_cost_fields = ['enabled', 'daily_budget_usd', 'cost_alerts']
            for field in required_cost_fields:
                field_present = field in cost_config
                results.add_validation(
                    f"COST_MANAGEMENT_{field.upper()}_EXISTS",
                    field_present,
                    f"Field '{field}' missing in COST_MANAGEMENT",
                    critical=True
                )
            
            # Validate budget value
            if 'daily_budget_usd' in cost_config:
                budget = cost_config['daily_budget_usd']
                budget_valid = isinstance(budget, (int, float)) and budget > 0
                results.add_validation(
                    "COST_MANAGEMENT_BUDGET_VALID",
                    budget_valid,
                    f"Invalid daily_budget_usd: {budget}",
                    critical=True
                )
            
            # Validate cost alerts configuration
            if 'cost_alerts' in cost_config:
                alerts = cost_config['cost_alerts']
                alerts_valid = isinstance(alerts, dict)
                results.add_validation(
                    "COST_ALERTS_STRUCTURE_VALID",
                    alerts_valid,
                    "cost_alerts must be a dictionary",
                    critical=True
                )
                
                if alerts_valid:
                    # Validate alert thresholds
                    required_alert_fields = ['warning_threshold', 'critical_threshold']
                    for field in required_alert_fields:
                        field_present = field in alerts
                        results.add_validation(
                            f"COST_ALERTS_{field.upper()}_EXISTS",
                            field_present,
                            f"Field '{field}' missing in cost_alerts",
                            critical=True
                        )
                        
                        if field_present:
                            threshold = alerts[field]
                            threshold_valid = isinstance(threshold, (int, float)) and 0.0 < threshold < 1.0
                            results.add_validation(
                                f"COST_ALERTS_{field.upper()}_VALID",
                                threshold_valid,
                                f"Invalid {field}: {threshold}, must be between 0.0 and 1.0",
                                critical=True
                            )
                    
                    # Validate threshold ordering
                    if 'warning_threshold' in alerts and 'critical_threshold' in alerts:
                        warning = alerts['warning_threshold']
                        critical = alerts['critical_threshold']
                        
                        if isinstance(warning, (int, float)) and isinstance(critical, (int, float)):
                            threshold_order_valid = warning < critical
                            results.add_validation(
                                "COST_ALERTS_THRESHOLD_ORDER",
                                threshold_order_valid,
                                f"Warning threshold ({warning}) must be less than critical threshold ({critical})",
                                critical=True
                            )
        
        # Test cost tracker functionality
        try:
            cost_tracker = CostTracker()
            tracker_initialized = True
        except Exception as e:
            tracker_initialized = False
            error_msg = str(e)
        
        results.add_validation(
            "COST_TRACKER_INITIALIZATION",
            tracker_initialized,
            f"CostTracker failed to initialize: {error_msg if not tracker_initialized else ''}",
            critical=True
        )
        
        self.assertFalse(results.has_critical_failures,
                        f"Critical cost management configuration failures: {results.critical_failures}")
        
        self.logger.info(f"Cost management validation - Pass rate: {results.pass_rate:.2%}")
    
    def test_http_pool_configuration(self):
        """Test HTTP connection pool configuration settings."""
        self.logger.info("Validating HTTP connection pool configuration")
        
        results = ConfigurationValidationResults()
        
        # Validate HTTP pool settings exist
        http_pool_settings = [
            'HTTP_POOL_ENABLED', 'HTTP_POOL_MAX_CONNECTIONS', 'HTTP_POOL_SIZE',
            'HTTP_CONNECTION_TIMEOUT', 'HTTP_READ_TIMEOUT',
            'HTTP_CIRCUIT_BREAKER_ENABLED', 'HTTP_CIRCUIT_BREAKER_FAILURE_THRESHOLD'
        ]
        
        for setting in http_pool_settings:
            setting_exists = hasattr(settings, setting)
            results.add_validation(
                f"{setting}_EXISTS",
                setting_exists,
                f"HTTP pool setting '{setting}' not found",
                critical=False  # HTTP pool is optional but recommended
            )
            
            if setting_exists:
                value = getattr(settings, setting)
                
                # Validate numeric settings
                if setting in ['HTTP_POOL_MAX_CONNECTIONS', 'HTTP_POOL_SIZE', 'HTTP_CIRCUIT_BREAKER_FAILURE_THRESHOLD']:
                    value_valid = isinstance(value, int) and value > 0
                    results.add_validation(
                        f"{setting}_VALUE_VALID",
                        value_valid,
                        f"Invalid value for {setting}: {value}",
                        critical=False
                    )
                
                # Validate timeout settings
                elif setting in ['HTTP_CONNECTION_TIMEOUT', 'HTTP_READ_TIMEOUT']:
                    value_valid = isinstance(value, (int, float)) and value > 0
                    results.add_validation(
                        f"{setting}_VALUE_VALID",
                        value_valid,
                        f"Invalid timeout value for {setting}: {value}",
                        critical=False
                    )
                
                # Validate boolean settings
                elif setting in ['HTTP_POOL_ENABLED', 'HTTP_CIRCUIT_BREAKER_ENABLED']:
                    value_valid = isinstance(value, bool)
                    results.add_validation(
                        f"{setting}_VALUE_VALID",
                        value_valid,
                        f"Invalid boolean value for {setting}: {value}",
                        critical=False
                    )
        
        self.logger.info(f"HTTP pool configuration validation - Pass rate: {results.pass_rate:.2%}")
    
    def test_instance_initialization_configuration(self):
        """Test that LLM instances can be properly initialized from configuration."""
        self.logger.info("Validating LLM instance initialization")
        
        results = ConfigurationValidationResults()
        
        try:
            # Test instance creation with mock pool manager
            mock_instances = self.mock_framework.create_mock_instances(
                models=list(settings.MODEL_CAPABILITIES.keys()),
                instances_per_model=1
            )
            
            instance_creation_success = len(mock_instances) > 0
            results.add_validation(
                "INSTANCE_CREATION_SUCCESS",
                instance_creation_success,
                f"Failed to create instances from configuration",
                critical=True
            )
            
            # Validate each instance has proper capabilities
            for instance_id, instance in mock_instances.items():
                # Check model capabilities are loaded
                capabilities_loaded = instance.model_capabilities is not None
                results.add_validation(
                    f"INSTANCE_{instance_id.replace('-', '_').upper()}_CAPABILITIES_LOADED",
                    capabilities_loaded,
                    f"Model capabilities not loaded for instance {instance_id}",
                    critical=True
                )
                
                # Check required properties
                required_properties = ['speed_score', 'quality_score', 'reliability_score', 'engine_type']
                for prop in required_properties:
                    prop_exists = hasattr(instance, prop) and getattr(instance, prop) is not None
                    results.add_validation(
                        f"INSTANCE_{instance_id.replace('-', '_').upper()}_{prop.upper()}_EXISTS",
                        prop_exists,
                        f"Property '{prop}' missing for instance {instance_id}",
                        critical=True
                    )
                
                # Validate property values
                if hasattr(instance, 'speed_score'):
                    speed_valid = 0.0 <= instance.speed_score <= 1.0
                    results.add_validation(
                        f"INSTANCE_{instance_id.replace('-', '_').upper()}_SPEED_SCORE_VALID",
                        speed_valid,
                        f"Invalid speed_score {instance.speed_score} for instance {instance_id}",
                        critical=True
                    )
                
                if hasattr(instance, 'quality_score'):
                    quality_valid = 0.0 <= instance.quality_score <= 1.0
                    results.add_validation(
                        f"INSTANCE_{instance_id.replace('-', '_').upper()}_QUALITY_SCORE_VALID",
                        quality_valid,
                        f"Invalid quality_score {instance.quality_score} for instance {instance_id}",
                        critical=True
                    )
        
        except Exception as e:
            results.add_validation(
                "INSTANCE_INITIALIZATION_ERROR",
                False,
                f"Exception during instance initialization: {e}",
                critical=True
            )
        
        self.assertFalse(results.has_critical_failures,
                        f"Critical instance initialization failures: {results.critical_failures}")
        
        self.logger.info(f"Instance initialization validation - Pass rate: {results.pass_rate:.2%}")
    
    def test_routing_strategy_configuration(self):
        """Test routing strategy configuration validation."""
        self.logger.info("Validating routing strategy configuration")
        
        results = ConfigurationValidationResults()
        
        try:
            # Test router initialization
            mock_instances = self.mock_framework.create_mock_instances(
                models=list(settings.MODEL_CAPABILITIES.keys()),
                instances_per_model=1
            )
            
            mock_pool_manager = Mock(spec=LLMPoolManager)
            mock_pool_manager.instances = mock_instances
            mock_pool_manager.config = {'pool_config': {'routing_strategy': 'balanced'}}
            
            router = IntelligentRequestRouter(mock_pool_manager)
            router_initialized = True
            
        except Exception as e:
            router_initialized = False
            error_msg = str(e)
        
        results.add_validation(
            "ROUTER_INITIALIZATION_SUCCESS",
            router_initialized,
            f"Router initialization failed: {error_msg if not router_initialized else ''}",
            critical=True
        )
        
        if router_initialized:
            # Test all routing strategies
            valid_strategies = ['speed_first', 'quality_first', 'cost_first', 'balanced', 'adaptive']
            
            for strategy in valid_strategies:
                try:
                    # Test strategy enum conversion
                    strategy_enum = RoutingStrategy(strategy)
                    strategy_valid = True
                except ValueError:
                    strategy_valid = False
                
                results.add_validation(
                    f"ROUTING_STRATEGY_{strategy.upper()}_VALID",
                    strategy_valid,
                    f"Routing strategy '{strategy}' not recognized",
                    critical=True
                )
            
            # Test routing strategy weighting
            test_scores = {
                'speed_score': 0.8,
                'quality_score': 0.7,
                'cost_score': 0.6,
                'health_score': 0.9,
                'capacity_score': 0.7,
                'use_case_score': 0.8,
                'suitability_score': 0.75,
                'json_score': 1.0
            }
            
            for strategy in valid_strategies:
                try:
                    strategy_enum = RoutingStrategy(strategy)
                    weighted_score = router._apply_strategy_weighting(test_scores, strategy_enum)
                    weighting_valid = 0.0 <= weighted_score <= 1.0
                except Exception:
                    weighting_valid = False
                
                results.add_validation(
                    f"ROUTING_STRATEGY_{strategy.upper()}_WEIGHTING_VALID",
                    weighting_valid,
                    f"Routing strategy '{strategy}' weighting calculation failed",
                    critical=True
                )
        
        self.assertFalse(results.has_critical_failures,
                        f"Critical routing strategy configuration failures: {results.critical_failures}")
        
        self.logger.info(f"Routing strategy validation - Pass rate: {results.pass_rate:.2%}")
    
    def test_configuration_edge_cases(self):
        """Test configuration edge cases and error handling."""
        self.logger.info("Testing configuration edge cases")
        
        results = ConfigurationValidationResults()
        
        # Test with missing configuration sections
        with patch.object(settings, 'MODEL_CAPABILITIES', {}):
            try:
                empty_instances = self.mock_framework.create_mock_instances(models=[])
                empty_config_handled = len(empty_instances) == 0
            except Exception:
                empty_config_handled = True  # Should handle gracefully
        
        results.add_validation(
            "EMPTY_MODEL_CONFIG_HANDLED",
            empty_config_handled,
            "Empty model configuration should be handled gracefully",
            critical=False
        )
        
        # Test with invalid model configuration
        invalid_config = {
            'invalid-model': {
                'engine': 'invalid_engine',
                'speed_tier': 'invalid_speed',
                'reliability_score': 2.0  # Invalid range
            }
        }
        
        with patch.object(settings, 'MODEL_CAPABILITIES', invalid_config):
            try:
                invalid_instances = self.mock_framework.create_mock_instances(models=['invalid-model'])
                # Should either handle gracefully or create with defaults
                invalid_handled = True
            except Exception as e:
                # Should have proper error handling
                invalid_handled = "Invalid" in str(e) or "Error" in str(e)
        
        results.add_validation(
            "INVALID_MODEL_CONFIG_HANDLED",
            invalid_handled,
            "Invalid model configuration should be handled with proper error messages",
            critical=False
        )
        
        # Test cost tracker with invalid budget
        try:
            cost_tracker = CostTracker()
            cost_tracker.daily_budget = -1.0  # Invalid budget
            
            alerts = cost_tracker.check_budget_alerts()
            negative_budget_handled = isinstance(alerts, list)  # Should return empty list or handle gracefully
            
        except Exception:
            negative_budget_handled = False
        
        results.add_validation(
            "NEGATIVE_BUDGET_HANDLED",
            negative_budget_handled,
            "Negative budget values should be handled gracefully",
            critical=False
        )
        
        self.logger.info(f"Configuration edge cases validation - Pass rate: {results.pass_rate:.2%}")
    
    def test_performance_analytics_configuration(self):
        """Test performance analytics configuration."""
        self.logger.info("Validating performance analytics configuration")
        
        results = ConfigurationValidationResults()
        
        try:
            # Test analytics initialization
            analytics = PerformanceAnalytics()
            analytics_initialized = True
        except Exception as e:
            analytics_initialized = False
            error_msg = str(e)
        
        results.add_validation(
            "ANALYTICS_INITIALIZATION_SUCCESS",
            analytics_initialized,
            f"Performance analytics initialization failed: {error_msg if not analytics_initialized else ''}",
            critical=True
        )
        
        if analytics_initialized:
            # Test analytics functionality
            try:
                # Record sample metrics
                analytics.record_request_metrics(
                    model_name="test-model",
                    instance_id="test-instance",
                    response_time=1.0,
                    tokens_processed=100,
                    cost=0.001,
                    success=True
                )
                
                metrics_recording_works = True
            except Exception:
                metrics_recording_works = False
            
            results.add_validation(
                "ANALYTICS_METRICS_RECORDING",
                metrics_recording_works,
                "Analytics metrics recording failed",
                critical=True
            )
            
            # Test analytics reporting
            try:
                summary = analytics.get_model_performance_summary("test-model")
                reporting_works = isinstance(summary, dict) and 'model_name' in summary
            except Exception:
                reporting_works = False
            
            results.add_validation(
                "ANALYTICS_REPORTING",
                reporting_works,
                "Analytics reporting failed",
                critical=True
            )
        
        self.assertFalse(results.has_critical_failures,
                        f"Critical analytics configuration failures: {results.critical_failures}")
        
        self.logger.info(f"Performance analytics validation - Pass rate: {results.pass_rate:.2%}")


class TestConfigurationIntegration(unittest.TestCase):
    """Test integration of all configuration components."""
    
    def test_full_system_configuration_integration(self):
        """Test full system integration with current configuration."""
        logging.info("Testing full system configuration integration")
        
        results = ConfigurationValidationResults()
        
        try:
            # Create full system components
            mock_framework = create_mock_framework()
            mock_instances = mock_framework.create_mock_instances(
                models=list(settings.MODEL_CAPABILITIES.keys()),
                instances_per_model=2
            )
            
            mock_pool_manager = Mock(spec=LLMPoolManager)
            mock_pool_manager.instances = mock_instances
            mock_pool_manager.config = {
                'pool_config': settings.MULTI_MODEL_POOLS.get('primary', {})
            }
            
            router = IntelligentRequestRouter(mock_pool_manager)
            analytics = PerformanceAnalytics()
            cost_tracker = CostTracker()
            
            integration_success = True
            
        except Exception as e:
            integration_success = False
            error_msg = str(e)
        
        results.add_validation(
            "FULL_SYSTEM_INTEGRATION",
            integration_success,
            f"Full system integration failed: {error_msg if not integration_success else ''}",
            critical=True
        )
        
        if integration_success:
            # Test end-to-end request processing
            try:
                from src.llm_pool.llm_pool_manager import LLMRequest
                
                test_request = LLMRequest(
                    request_id="integration-test",
                    prompt="Test integration request",
                    model_config={'use_case': 'general'},
                    priority=5,
                    timeout=30.0,
                    retry_count=0,
                    max_retries=3,
                    created_at=time.time()
                )
                
                routing_decision = router.route_request(test_request)
                e2e_success = routing_decision is not None
                
            except Exception:
                e2e_success = False
            
            results.add_validation(
                "END_TO_END_REQUEST_PROCESSING",
                e2e_success,
                "End-to-end request processing failed",
                critical=True
            )
        
        # Validate overall integration
        assert not results.has_critical_failures, \
            f"Critical integration failures: {results.critical_failures}"
        
        assert results.pass_rate > 0.95, \
            f"Integration validation pass rate too low: {results.pass_rate:.2%}"


if __name__ == '__main__':
    # Configure logging for configuration tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    unittest.main(verbosity=2)