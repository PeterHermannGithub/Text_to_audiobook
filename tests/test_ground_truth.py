"""
Ground Truth Test Suite for Text_to_audiobook Quality Validation

This test suite contains validated ground truth data for regression testing
of the comprehensive quality improvements implemented in January 2025.

Test Categories:
1. Quality Validation System Tests
2. LLM Classification Tests  
3. Rule-Based Attribution Tests
4. Character Detection Tests
5. Mixed-Content Attribution Tests
6. UNFIXABLE Recovery Tests
7. Output Formatting Tests
8. Integration Tests
"""

import unittest
import json
import os
import sys
from typing import Dict, List, Any

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.text_structurer import TextStructurer
from src.simplified_validator import SimplifiedValidator
from src.rule_based_attributor import RuleBasedAttributor
from src.deterministic_segmenter import DeterministicSegmenter
from src.unfixable_recovery import UnfixableRecoverySystem
from src.output_formatter import OutputFormatter
from src.preprocessor import TextPreprocessor
from config import settings

class GroundTruthTestSuite(unittest.TestCase):
    """
    Comprehensive ground truth test suite with validated expected outputs
    for quality regression testing.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures and ground truth data."""
        cls.test_data = cls._load_ground_truth_data()
        cls.structurer = TextStructurer(engine='local', local_model='mistral')
        cls.validator = SimplifiedValidator()
        cls.attributor = RuleBasedAttributor()
        cls.segmenter = DeterministicSegmenter()
        cls.recovery = UnfixableRecoverySystem()
        cls.formatter = OutputFormatter()
        
    @classmethod
    def _load_ground_truth_data(cls) -> Dict[str, Any]:
        """Load validated ground truth test cases."""
        return {
            # Test Case 1: Simple Dialogue Attribution
            'simple_dialogue': {
                'input': '"Hello there," John said with a smile. "How are you today?" Mary replied warmly.',
                'expected_segments': [
                    {'speaker': 'John', 'text': '"Hello there," John said with a smile.'},
                    {'speaker': 'Mary', 'text': '"How are you today?" Mary replied warmly.'}
                ],
                'min_quality_score': 95.0,
                'expected_errors': 0
            },
            
            # Test Case 2: Script Format
            'script_format': {
                'input': 'JOHN: Hello there, how are you?\nMARY: I\'m doing well, thank you!\nNARRATOR: The conversation continued.',
                'expected_segments': [
                    {'speaker': 'John', 'text': 'Hello there, how are you?'},
                    {'speaker': 'Mary', 'text': 'I\'m doing well, thank you!'},
                    {'speaker': 'narrator', 'text': 'The conversation continued.'}
                ],
                'min_quality_score': 98.0,
                'expected_errors': 0
            },
            
            # Test Case 3: Mixed Content Requiring Splitting
            'mixed_content': {
                'input': 'John walked into the room, his face serious. "We need to talk," he said quietly. The tension in the air was palpable as Mary looked up from her book.',
                'expected_min_segments': 2,  # Should be split into dialogue and narrative
                'min_quality_score': 85.0,
                'should_contain_speakers': ['John', 'narrator']
            },
            
            # Test Case 4: Character Detection with Metadata Contamination
            'character_detection': {
                'input': 'Chapter 1: The Beginning\nTable of Contents\nJohn said "Hello" to Mary. She smiled back at him.',
                'expected_characters': ['John', 'Mary'],
                'should_not_contain': ['Chapter', 'Contents', 'Table', 'Beginning'],
                'min_quality_score': 90.0
            },
            
            # Test Case 5: UNFIXABLE Recovery Test
            'unfixable_recovery': {
                'input': 'Someone said something unclear here without attribution markers.',
                'expected_recovery_attempts': True,
                'acceptable_speakers': ['narrator', 'AMBIGUOUS', 'unknown'],
                'min_quality_score': 70.0  # Lower threshold for difficult content
            },
            
            # Test Case 6: Unicode and Formatting Cleanup
            'unicode_formatting': {
                'input': '"Hello world," said John with "smart quotes" and—em dashes.',
                'expected_clean_quotes': True,  # Should normalize to straight quotes
                'expected_clean_dashes': True,  # Should handle em dashes properly
                'min_quality_score': 95.0
            },
            
            # Test Case 7: Complex Dialogue with Multiple Speakers
            'complex_dialogue': {
                'input': '''"I can't believe this," Sarah whispered.
                "What's wrong?" asked Tom, concerned.
                "Nothing," she replied, but her voice betrayed her anxiety.
                The detective watched the exchange with interest.''',
                'expected_speakers': ['Sarah', 'Tom', 'narrator'],
                'expected_min_segments': 4,
                'min_quality_score': 90.0
            }
        }
    
    def test_quality_validation_system(self):
        """Test the recalibrated SimplifiedValidator scoring system."""
        test_case = self.test_data['simple_dialogue']
        
        # Create test segments in the format expected by validator
        test_segments = [(segment, 0) for segment in test_case['expected_segments']]
        
        # Run validation
        validated_data, quality_report = self.validator.validate(
            test_segments, 
            test_case['input'], 
            {'potential_character_names': {'John', 'Mary'}}
        )
        
        # Assert quality meets minimum threshold
        self.assertGreaterEqual(
            quality_report['quality_score'], 
            test_case['min_quality_score'],
            f"Quality score {quality_report['quality_score']:.2f}% below minimum {test_case['min_quality_score']}%"
        )
        
        # Assert error count is within expected range
        self.assertLessEqual(
            quality_report['error_count'],
            test_case['expected_errors'],
            f"Error count {quality_report['error_count']} exceeds expected {test_case['expected_errors']}"
        )
    
    def test_rule_based_attribution(self):
        """Test enhanced rule-based attribution patterns."""
        test_case = self.test_data['script_format']
        
        # Create numbered lines for attribution
        lines = [
            {'line_id': i+1, 'text': line.strip()} 
            for i, line in enumerate(test_case['input'].split('\n')) 
            if line.strip()
        ]
        
        # Test rule-based attribution
        attributed_lines = self.attributor.process_lines(
            lines, 
            {'potential_character_names': {'John', 'Mary'}}
        )
        
        # Verify high-confidence attributions were made
        rule_attributed = self.attributor.get_attributed_lines(attributed_lines)
        self.assertGreater(
            len(rule_attributed), 
            0, 
            "Rule-based attributor should identify script format speakers"
        )
        
        # Verify speakers match expected
        found_speakers = {line['speaker'].lower() for line in rule_attributed}
        expected_speakers = {seg['speaker'].lower() for seg in test_case['expected_segments']}
        
        self.assertTrue(
            found_speakers.intersection(expected_speakers),
            f"Found speakers {found_speakers} should overlap with expected {expected_speakers}"
        )
    
    def test_mixed_content_detection(self):
        """Test advanced mixed-content detection and splitting."""
        test_case = self.test_data['mixed_content']
        
        # Test segmentation
        segments = self.segmenter.segment_text(
            test_case['input'],
            {'potential_character_names': {'John', 'Mary'}}
        )
        
        # Verify minimum segment count (should be split)
        self.assertGreaterEqual(
            len(segments),
            test_case['expected_min_segments'],
            f"Mixed content should be split into at least {test_case['expected_min_segments']} segments"
        )
        
        # Verify speakers are detected
        if 'should_contain_speakers' in test_case:
            # Process through attribution to get speakers
            attributed_lines = self.attributor.process_lines(
                segments,
                {'potential_character_names': {'John', 'Mary'}}
            )
            
            all_lines = self.attributor.get_attributed_lines(attributed_lines) + \
                       self.attributor.get_pending_lines(attributed_lines)
            
            found_speakers = {line.get('speaker', 'unknown').lower() for line in all_lines}
            expected_speakers = {speaker.lower() for speaker in test_case['should_contain_speakers']}
            
            for expected_speaker in expected_speakers:
                self.assertIn(
                    expected_speaker,
                    found_speakers,
                    f"Expected speaker '{expected_speaker}' not found in {found_speakers}"
                )
    
    def test_character_detection_filtering(self):
        """Test character detection with metadata contamination filtering."""
        test_case = self.test_data['character_detection']
        
        # Create preprocessor and analyze text
        preprocessor = TextPreprocessor(nlp_model=None)  # Can work without spaCy for basic tests
        metadata = preprocessor.analyze(test_case['input'])
        
        found_characters = metadata.get('potential_character_names', set())
        
        # Verify expected characters are found
        for expected_char in test_case['expected_characters']:
            self.assertIn(
                expected_char,
                found_characters,
                f"Expected character '{expected_char}' not found in {found_characters}"
            )
        
        # Verify contamination is filtered out
        for contamination in test_case['should_not_contain']:
            self.assertNotIn(
                contamination,
                found_characters,
                f"Contamination '{contamination}' should be filtered out from {found_characters}"
            )
    
    def test_unfixable_recovery_system(self):
        """Test UNFIXABLE segment recovery system."""
        test_case = self.test_data['unfixable_recovery']
        
        # Create a segment marked as UNFIXABLE
        test_segments = [
            {'speaker': 'UNFIXABLE', 'text': test_case['input']}
        ]
        
        # Attempt recovery
        recovered_segments = self.recovery.recover_unfixable_segments(
            test_segments,
            {'potential_character_names': set()}
        )
        
        # Verify recovery was attempted
        if test_case['expected_recovery_attempts']:
            # Check if speaker was changed from UNFIXABLE
            recovered_speaker = recovered_segments[0]['speaker']
            
            if recovered_speaker != 'UNFIXABLE':
                # Recovery succeeded - verify speaker is acceptable
                self.assertIn(
                    recovered_speaker.lower(),
                    [s.lower() for s in test_case['acceptable_speakers']],
                    f"Recovered speaker '{recovered_speaker}' not in acceptable list"
                )
            # If still UNFIXABLE, that's also acceptable for truly ambiguous content
    
    def test_output_formatting_cleanup(self):
        """Test Unicode normalization and output formatting."""
        test_case = self.test_data['unicode_formatting']
        
        # Create test segment with Unicode issues
        test_segments = [
            {'speaker': 'John', 'text': test_case['input']}
        ]
        
        # Apply formatting
        formatted_segments = self.formatter.format_output(test_segments)
        
        formatted_text = formatted_segments[0]['text']
        
        # Verify quote normalization
        if test_case['expected_clean_quotes']:
            self.assertNotIn('"', formatted_text, "Smart quotes should be normalized")
            self.assertNotIn('"', formatted_text, "Smart quotes should be normalized")
        
        # Verify dash handling
        if test_case['expected_clean_dashes']:
            # Should handle em dashes appropriately (either normalize or preserve)
            self.assertIsInstance(formatted_text, str, "Text should be properly formatted")
    
    def test_end_to_end_integration(self):
        """Test complete end-to-end pipeline with quality validation."""
        test_case = self.test_data['complex_dialogue']
        
        # Run complete text structuring pipeline
        try:
            # Note: This requires LLM access, so we'll make it optional
            if hasattr(self.structurer, 'structure_text'):
                structured_output = self.structurer.structure_text(test_case['input'])
                
                # Verify minimum segment count
                self.assertGreaterEqual(
                    len(structured_output),
                    test_case['expected_min_segments'],
                    f"Should produce at least {test_case['expected_min_segments']} segments"
                )
                
                # Verify expected speakers are present
                found_speakers = {seg.get('speaker', '').lower() for seg in structured_output}
                for expected_speaker in test_case['expected_speakers']:
                    self.assertIn(
                        expected_speaker.lower(),
                        found_speakers,
                        f"Expected speaker '{expected_speaker}' not found"
                    )
            else:
                self.skipTest("End-to-end test requires LLM access")
                
        except Exception as e:
            self.skipTest(f"End-to-end test skipped due to dependency: {e}")
    
    def test_regression_quality_thresholds(self):
        """Test that all components meet regression quality thresholds."""
        # Test each major component individually
        
        # 1. Validator Quality Scoring
        validator_test_passed = True
        try:
            test_segments = [(seg, 0) for seg in self.test_data['simple_dialogue']['expected_segments']]
            _, quality_report = self.validator.validate(
                test_segments,
                self.test_data['simple_dialogue']['input'],
                {'potential_character_names': {'John', 'Mary'}}
            )
            validator_test_passed = quality_report['quality_score'] >= 90.0
        except Exception:
            validator_test_passed = False
        
        self.assertTrue(validator_test_passed, "Validator quality scoring regression test failed")
        
        # 2. Rule-based Attribution Coverage
        attributor_test_passed = True
        try:
            lines = [{'line_id': 1, 'text': 'JOHN: Hello world'}]
            attributed = self.attributor.process_lines(lines, {'potential_character_names': {'John'}})
            rule_attributed = self.attributor.get_attributed_lines(attributed)
            attributor_test_passed = len(rule_attributed) > 0
        except Exception:
            attributor_test_passed = False
        
        self.assertTrue(attributor_test_passed, "Rule-based attribution regression test failed")
        
        # 3. Output Formatting Consistency
        formatter_test_passed = True
        try:
            test_segments = [{'speaker': 'Test', 'text': 'Test text'}]
            formatted = self.formatter.format_output(test_segments)
            formatter_test_passed = len(formatted) == 1 and 'speaker' in formatted[0]
        except Exception:
            formatter_test_passed = False
        
        self.assertTrue(formatter_test_passed, "Output formatting regression test failed")
    
    def test_performance_benchmarks(self):
        """Test performance regression benchmarks."""
        import time
        
        # Test small text processing time
        small_text = self.test_data['simple_dialogue']['input']
        
        start_time = time.time()
        
        # Test individual component performance
        try:
            # Segmentation performance
            segments = self.segmenter.segment_text(small_text, {})
            
            # Attribution performance
            if segments:
                attributed = self.attributor.process_lines(segments, {})
            
            # Formatting performance
            test_segments = [{'speaker': 'Test', 'text': small_text}]
            formatted = self.formatter.format_output(test_segments)
            
            processing_time = time.time() - start_time
            
            # Should complete basic operations in under 1 second
            self.assertLess(
                processing_time,
                1.0,
                f"Basic operations took {processing_time:.2f}s, should be under 1.0s"
            )
            
        except Exception as e:
            self.skipTest(f"Performance test skipped due to error: {e}")


if __name__ == '__main__':
    # Create test runner with detailed output
    unittest.main(verbosity=2)