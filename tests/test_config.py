"""
Test Configuration for Text_to_audiobook Quality Validation

This file contains configuration settings for the testing infrastructure,
including quality thresholds, test data paths, and validation criteria.
"""

import os

# Test execution configuration
TEST_CONFIG = {
    # Quality thresholds for regression testing
    'quality_thresholds': {
        'minimum_overall_quality': 85.0,      # Overall system quality threshold
        'simple_dialogue_quality': 95.0,      # Simple dialogue attribution
        'script_format_quality': 98.0,        # Script format processing
        'mixed_content_quality': 85.0,        # Mixed content handling
        'character_detection_quality': 90.0,  # Character name detection
        'unfixable_recovery_quality': 70.0,   # UNFIXABLE segment recovery
        'unicode_formatting_quality': 95.0,   # Unicode and formatting cleanup
        'complex_dialogue_quality': 90.0      # Complex multi-speaker dialogue
    },
    
    # Performance benchmarks (in seconds)
    'performance_thresholds': {
        'basic_operations_max_time': 1.0,     # Basic component operations
        'small_text_processing': 5.0,         # Small text (< 1000 chars)
        'medium_text_processing': 30.0,       # Medium text (1000-5000 chars)
        'large_text_processing': 120.0        # Large text (> 5000 chars)
    },
    
    # Error tolerance levels
    'error_tolerance': {
        'max_errors_simple': 0,               # Simple test cases
        'max_errors_complex': 2,              # Complex test cases
        'max_unfixable_segments': 5,          # Percentage of UNFIXABLE segments
        'max_ambiguous_segments': 10          # Percentage of AMBIGUOUS segments
    },
    
    # Test data configuration
    'test_data': {
        'use_external_files': False,          # Whether to load test files from disk
        'test_files_directory': 'tests/data', # Directory for test files
        'generate_reports': True,             # Generate detailed test reports
        'save_test_outputs': False            # Save test outputs for manual review
    },
    
    # Component testing configuration
    'component_tests': {
        'test_validator': True,               # Test SimplifiedValidator
        'test_attributor': True,              # Test RuleBasedAttributor
        'test_segmenter': True,               # Test DeterministicSegmenter
        'test_recovery': True,                # Test UnfixableRecoverySystem
        'test_formatter': True,               # Test OutputFormatter
        'test_preprocessor': True,            # Test TextPreprocessor
        'test_integration': False             # Test full pipeline (requires LLM)
    },
    
    # Regression testing configuration
    'regression_tests': {
        'compare_with_baseline': False,       # Compare with saved baseline results
        'baseline_file': 'tests/baseline_results.json',
        'quality_degradation_threshold': 5.0, # Max allowed quality degradation (%)
        'performance_degradation_threshold': 50.0  # Max allowed performance degradation (%)
    },
    
    # Reporting configuration
    'reporting': {
        'generate_html_report': False,        # Generate HTML test report
        'generate_json_report': True,         # Generate JSON test report
        'report_directory': 'tests/reports',  # Directory for test reports
        'include_performance_graphs': False,  # Include performance visualizations
        'include_error_details': True        # Include detailed error information
    }
}

# Test environment configuration
ENVIRONMENT_CONFIG = {
    'mock_llm_responses': True,              # Use mock LLM responses for testing
    'skip_llm_dependent_tests': True,        # Skip tests requiring actual LLM
    'use_test_spacy_model': False,           # Use lightweight spaCy model for tests
    'disable_logging_during_tests': True,   # Disable verbose logging during tests
    'cleanup_temp_files': True              # Clean up temporary files after tests
}

# Mock LLM responses for testing (when LLM is not available)
MOCK_LLM_RESPONSES = {
    'simple_dialogue_classification': ['John', 'Mary'],
    'script_format_classification': ['John', 'Mary', 'narrator'],
    'mixed_content_classification': ['John', 'narrator'],
    'complex_dialogue_classification': ['Sarah', 'Tom', 'narrator', 'Sarah'],
    'fallback_classification': ['AMBIGUOUS']
}

# Test data validation schemas
VALIDATION_SCHEMAS = {
    'segment_schema': {
        'required_fields': ['speaker', 'text'],
        'optional_fields': ['attribution_confidence', 'attribution_method', 'errors'],
        'speaker_validation': {
            'allowed_special_speakers': ['narrator', 'AMBIGUOUS', 'UNFIXABLE'],
            'max_speaker_name_length': 50,
            'min_speaker_name_length': 1
        },
        'text_validation': {
            'min_text_length': 1,
            'max_text_length': 2000,
            'forbidden_characters': ['\x00', '\x01', '\x02']  # Control characters
        }
    },
    
    'quality_report_schema': {
        'required_fields': ['quality_score', 'error_count', 'attribution_metrics'],
        'quality_score_range': [0.0, 100.0],
        'error_analysis_fields': ['total_errors', 'error_breakdown']
    }
}

def get_test_config(component: str = None) -> dict:
    """
    Get test configuration for a specific component or overall configuration.
    
    Args:
        component: Optional component name ('validator', 'attributor', etc.)
        
    Returns:
        Dictionary containing relevant test configuration
    """
    if component:
        return TEST_CONFIG.get(f'{component}_config', {})
    return TEST_CONFIG

def get_quality_threshold(test_type: str) -> float:
    """
    Get quality threshold for a specific test type.
    
    Args:
        test_type: Type of test (e.g., 'simple_dialogue_quality')
        
    Returns:
        Quality threshold value
    """
    return TEST_CONFIG['quality_thresholds'].get(test_type, 85.0)

def get_performance_threshold(operation_type: str) -> float:
    """
    Get performance threshold for a specific operation type.
    
    Args:
        operation_type: Type of operation (e.g., 'basic_operations_max_time')
        
    Returns:
        Performance threshold in seconds
    """
    return TEST_CONFIG['performance_thresholds'].get(operation_type, 30.0)

def should_skip_test(test_name: str) -> tuple[bool, str]:
    """
    Determine if a test should be skipped based on configuration.
    
    Args:
        test_name: Name of the test
        
    Returns:
        Tuple of (should_skip, reason)
    """
    if 'integration' in test_name.lower() and not TEST_CONFIG['component_tests']['test_integration']:
        return True, "Integration tests disabled in configuration"
    
    if 'llm' in test_name.lower() and ENVIRONMENT_CONFIG['skip_llm_dependent_tests']:
        return True, "LLM-dependent tests disabled"
        
    return False, ""

def get_mock_llm_response(test_case: str) -> list:
    """
    Get mock LLM response for testing when actual LLM is not available.
    
    Args:
        test_case: Name of the test case
        
    Returns:
        List of mock speaker classifications
    """
    return MOCK_LLM_RESPONSES.get(test_case, ['AMBIGUOUS'])

# Test result validation functions
def validate_segment(segment: dict) -> tuple[bool, list]:
    """
    Validate a segment against the expected schema.
    
    Args:
        segment: Segment dictionary to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    schema = VALIDATION_SCHEMAS['segment_schema']
    
    # Check required fields
    for field in schema['required_fields']:
        if field not in segment:
            errors.append(f"Missing required field: {field}")
    
    # Validate speaker
    if 'speaker' in segment:
        speaker = segment['speaker']
        if not isinstance(speaker, str):
            errors.append("Speaker must be a string")
        elif len(speaker) < schema['speaker_validation']['min_speaker_name_length']:
            errors.append("Speaker name too short")
        elif len(speaker) > schema['speaker_validation']['max_speaker_name_length']:
            errors.append("Speaker name too long")
    
    # Validate text
    if 'text' in segment:
        text = segment['text']
        if not isinstance(text, str):
            errors.append("Text must be a string")
        elif len(text) < schema['text_validation']['min_text_length']:
            errors.append("Text too short")
        elif len(text) > schema['text_validation']['max_text_length']:
            errors.append("Text too long")
        elif any(char in text for char in schema['text_validation']['forbidden_characters']):
            errors.append("Text contains forbidden control characters")
    
    return len(errors) == 0, errors

def validate_quality_report(report: dict) -> tuple[bool, list]:
    """
    Validate a quality report against the expected schema.
    
    Args:
        report: Quality report dictionary to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    schema = VALIDATION_SCHEMAS['quality_report_schema']
    
    # Check required fields
    for field in schema['required_fields']:
        if field not in report:
            errors.append(f"Missing required field: {field}")
    
    # Validate quality score range
    if 'quality_score' in report:
        score = report['quality_score']
        min_score, max_score = schema['quality_score_range']
        if not isinstance(score, (int, float)):
            errors.append("Quality score must be numeric")
        elif score < min_score or score > max_score:
            errors.append(f"Quality score {score} outside valid range [{min_score}, {max_score}]")
    
    return len(errors) == 0, errors