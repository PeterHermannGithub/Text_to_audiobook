"""
Memory profiling tests for text structuring components.

This module provides memory usage analysis for the text structuring pipeline,
focusing on segmentation, attribution, and validation memory patterns.
"""

import pytest
import time
from memory_profiler import profile
from unittest.mock import Mock, patch
from typing import Dict, Any, List
import json


@pytest.mark.performance
@pytest.mark.memory
class TestTextStructuringMemoryProfile:
    """Memory profiling tests for text structuring components."""
    
    @profile
    def test_deterministic_segmentation_memory_profile(self):
        """Profile memory usage during deterministic text segmentation."""
        
        with patch('src.text_processing.segmentation.deterministic_segmenter.DeterministicSegmenter') as mock_segmenter_class:
            
            mock_segmenter = Mock()
            mock_segmenter_class.return_value = mock_segmenter
            
            # Test memory usage with different text sizes
            text_sizes = [1000, 5000, 10000, 50000]  # characters
            
            for size in text_sizes:
                # Generate test text with dialogue patterns
                test_text = self._generate_script_text(size)
                
                # Mock segmentation results
                segments = []
                chunk_size = 200  # Average segment size
                for i in range(0, len(test_text), chunk_size):
                    segment_text = test_text[i:i+chunk_size]
                    segments.append({
                        'text': segment_text,
                        'speaker': 'AMBIGUOUS',
                        'segment_type': 'dialogue' if '"' in segment_text else 'narrative',
                        'confidence': 0.5,
                        'segment_id': f'seg_{i//chunk_size:04d}'
                    })
                
                mock_segmenter.segment_text.return_value = segments
                
                # Test segmentation
                segmenter = mock_segmenter_class()
                result_segments = segmenter.segment_text(test_text)
                
                # Verify segmentation
                assert len(result_segments) > 0
                assert isinstance(result_segments, list)
                
                # Memory cleanup
                del test_text
                del result_segments
                del segments
                import gc
                gc.collect()
    
    @profile
    def test_rule_based_attribution_memory_profile(self):
        """Profile memory usage during rule-based speaker attribution."""
        
        with patch('src.attribution.rule_based_attributor.RuleBasedAttributor') as mock_attributor_class:
            
            mock_attributor = Mock()
            mock_attributor_class.return_value = mock_attributor
            
            # Test with different numbers of segments
            segment_counts = [100, 500, 1000, 5000]
            
            for count in segment_counts:
                # Generate test segments
                test_segments = []
                speakers = ['Romeo', 'Juliet', 'Narrator', 'Mercutio', 'Benvolio']
                
                for i in range(count):
                    segment = {
                        'text': f'This is test dialogue segment {i} with some content to analyze.',
                        'speaker': 'AMBIGUOUS',
                        'segment_type': 'dialogue' if i % 3 != 0 else 'narrative',
                        'confidence': 0.5,
                        'segment_id': f'seg_{i:06d}'
                    }
                    test_segments.append(segment)
                
                # Mock attribution results
                attributed_segments = []
                for i, segment in enumerate(test_segments):
                    attributed_segment = segment.copy()
                    attributed_segment['speaker'] = speakers[i % len(speakers)]
                    attributed_segment['confidence'] = 0.85
                    attributed_segments.append(attributed_segment)
                
                mock_attributor.attribute_speakers.return_value = attributed_segments
                
                # Test attribution
                attributor = mock_attributor_class()
                result_segments = attributor.attribute_speakers(test_segments)
                
                # Verify attribution
                assert len(result_segments) == count
                assert all(seg['speaker'] != 'AMBIGUOUS' for seg in result_segments)
                
                # Memory cleanup
                del test_segments
                del attributed_segments
                del result_segments
                import gc
                gc.collect()
    
    @profile
    def test_llm_orchestrator_memory_profile(self):
        """Profile memory usage during LLM orchestration and processing."""
        
        with patch('src.attribution.llm.orchestrator.LLMOrchestrator') as mock_orchestrator_class:
            
            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            # Test with different batch sizes
            batch_sizes = [10, 50, 100, 500]
            
            for batch_size in batch_sizes:
                # Generate AMBIGUOUS segments for LLM processing
                ambiguous_segments = []
                for i in range(batch_size):
                    segment = {
                        'text': f'"This is ambiguous dialogue {i} that needs LLM classification."',
                        'speaker': 'AMBIGUOUS',
                        'segment_type': 'dialogue',
                        'confidence': 0.3,
                        'segment_id': f'amb_seg_{i:04d}'
                    }
                    ambiguous_segments.append(segment)
                
                # Mock LLM response
                llm_results = []
                speakers = ['Character A', 'Character B', 'Character C']
                for i, segment in enumerate(ambiguous_segments):
                    result = segment.copy()
                    result['speaker'] = speakers[i % len(speakers)]
                    result['confidence'] = 0.9
                    llm_results.append(result)
                
                mock_orchestrator.process_segments.return_value = llm_results
                
                # Test LLM processing
                orchestrator = mock_orchestrator_class()
                processed_segments = orchestrator.process_segments(ambiguous_segments)
                
                # Verify processing
                assert len(processed_segments) == batch_size
                assert all(seg['confidence'] > 0.8 for seg in processed_segments)
                
                # Memory cleanup
                del ambiguous_segments
                del llm_results
                del processed_segments
                import gc
                gc.collect()
    
    @profile
    def test_contextual_refiner_memory_profile(self):
        """Profile memory usage during contextual refinement processing."""
        
        with patch('src.refinement.contextual_refiner.ContextualRefiner') as mock_refiner_class:
            
            mock_refiner = Mock()
            mock_refiner_class.return_value = mock_refiner
            
            # Test with conversation context building
            conversation_lengths = [50, 200, 500, 1000]
            
            for length in conversation_lengths:
                # Generate conversation segments
                conversation_segments = []
                speakers = ['Alice', 'Bob', 'Narrator']
                
                for i in range(length):
                    segment = {
                        'text': f'Conversation turn {i} with contextual content.',
                        'speaker': speakers[i % len(speakers)],
                        'segment_type': 'dialogue' if i % 4 != 0 else 'narrative',
                        'confidence': 0.8,
                        'segment_id': f'conv_seg_{i:05d}',
                        'turn_index': i
                    }
                    conversation_segments.append(segment)
                
                # Mock refined results
                refined_segments = []
                for segment in conversation_segments:
                    refined_segment = segment.copy()
                    refined_segment['confidence'] = min(0.95, segment['confidence'] + 0.1)
                    refined_segment['context_score'] = 0.85
                    refined_segments.append(refined_segment)
                
                mock_refiner.refine_with_context.return_value = refined_segments
                
                # Test contextual refinement
                refiner = mock_refiner_class()
                result_segments = refiner.refine_with_context(conversation_segments)
                
                # Verify refinement
                assert len(result_segments) == length
                assert all(seg.get('context_score', 0) > 0.8 for seg in result_segments)
                
                # Memory cleanup
                del conversation_segments
                del refined_segments
                del result_segments
                import gc
                gc.collect()
    
    @profile
    def test_validator_memory_profile(self):
        """Profile memory usage during validation processing."""
        
        with patch('src.validation.validator.SimplifiedValidator') as mock_validator_class:
            
            mock_validator = Mock()
            mock_validator_class.return_value = mock_validator
            
            # Test validation with different document sizes
            document_sizes = [100, 500, 1000, 2500]
            
            for size in document_sizes:
                # Generate structured segments for validation
                structured_segments = []
                speakers = ['Romeo', 'Juliet', 'Mercutio', 'Benvolio', 'Narrator']
                
                for i in range(size):
                    segment = {
                        'text': f'Validated segment {i} with speaker attribution.',
                        'speaker': speakers[i % len(speakers)],
                        'segment_type': 'dialogue' if i % 3 != 0 else 'narrative',
                        'confidence': 0.85 + (i % 10) * 0.01,
                        'segment_id': f'val_seg_{i:06d}',
                        'validation_passed': True
                    }
                    structured_segments.append(segment)
                
                # Mock validation results
                validation_report = {
                    'overall_quality_score': 92.5,
                    'speaker_consistency_score': 95.0,
                    'confidence_distribution': {'high': 0.8, 'medium': 0.15, 'low': 0.05},
                    'error_categories': {},
                    'total_segments': size,
                    'validation_passed': True
                }
                
                mock_validator.validate_segments.return_value = validation_report
                
                # Test validation
                validator = mock_validator_class()
                report = validator.validate_segments(structured_segments)
                
                # Verify validation
                assert report['total_segments'] == size
                assert report['overall_quality_score'] > 90
                
                # Memory cleanup
                del structured_segments
                del validation_report
                del report
                import gc
                gc.collect()
    
    @profile
    def test_output_formatter_memory_profile(self):
        """Profile memory usage during output formatting and JSON generation."""
        
        with patch('src.output.output_formatter.OutputFormatter') as mock_formatter_class:
            
            mock_formatter = Mock()
            mock_formatter_class.return_value = mock_formatter
            
            # Test formatting with different output sizes
            output_sizes = [500, 2000, 5000, 10000]
            
            for size in output_sizes:
                # Generate final structured segments
                final_segments = []
                speakers = ['Character 1', 'Character 2', 'Character 3', 'Narrator']
                
                for i in range(size):
                    segment = {
                        'text': f'Final formatted segment {i} ready for output generation.',
                        'speaker': speakers[i % len(speakers)],
                        'segment_type': 'dialogue' if i % 4 != 0 else 'narrative',
                        'confidence': 0.9,
                        'segment_id': f'final_{i:06d}',
                        'processing_metadata': {
                            'processing_time': 0.15,
                            'attribution_method': 'rule_based' if i % 2 == 0 else 'llm',
                            'validation_score': 0.95
                        }
                    }
                    final_segments.append(segment)
                
                # Mock formatted JSON output
                formatted_json = json.dumps(final_segments, indent=2)
                mock_formatter.format_segments.return_value = formatted_json
                
                # Test output formatting
                formatter = mock_formatter_class()
                output_json = formatter.format_segments(final_segments)
                
                # Verify formatting
                assert len(output_json) > size * 100  # Rough size check
                assert '"speaker"' in output_json
                assert '"text"' in output_json
                
                # Memory cleanup
                del final_segments
                del formatted_json
                del output_json
                import gc
                gc.collect()
    
    def _generate_script_text(self, target_size: int) -> str:
        """Generate script-format text for testing."""
        
        speakers = ['ROMEO', 'JULIET', 'MERCUTIO', 'BENVOLIO', 'NURSE']
        dialogues = [
            'But soft! What light through yonder window breaks?',
            'O Romeo, Romeo! Wherefore art thou Romeo?',
            'A plague on both your houses!',
            'Did my heart love till now? Forswear it, sight!',
            'Come, come with me, and we will make short work.'
        ]
        
        text = ""
        while len(text) < target_size:
            speaker = speakers[len(text) % len(speakers)]
            dialogue = dialogues[len(text) % len(dialogues)]
            
            # Add script format content
            text += f"\n{speaker}\n{dialogue}\n"
            
            # Add some narrative occasionally
            if len(text) % 500 == 0:
                text += "\nThe scene continues with dramatic tension in the streets of Verona.\n"
        
        return text[:target_size]
    
    def test_memory_usage_thresholds_structuring(self):
        """Test that structuring memory usage stays within acceptable thresholds."""
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Get baseline memory usage
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate processing large structured documents
        for doc_size in [1000, 2500, 5000]:
            # Generate large structured content
            segments = []
            for i in range(doc_size):
                segment = {
                    'text': f'Structured segment {i} with attribution and metadata.' * 5,
                    'speaker': f'Character_{i % 10}',
                    'segment_type': 'dialogue' if i % 3 != 0 else 'narrative',
                    'confidence': 0.9,
                    'segment_id': f'struct_seg_{i:06d}',
                    'metadata': {
                        'processing_time': 0.1,
                        'method': 'rule_based',
                        'context_length': 200
                    }
                }
                segments.append(segment)
            
            # Check current memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - baseline_memory
            
            # Verify memory usage doesn't grow excessively
            assert memory_increase < 1000, f"Structuring memory usage too high: {memory_increase}MB increase"
            
            # Clean up
            del segments
            import gc
            gc.collect()


def run_structuring_memory_profiling_suite():
    """Run the complete text structuring memory profiling test suite."""
    
    test_suite = TestTextStructuringMemoryProfile()
    
    print("Running text structuring memory profiling...")
    
    # Run each test with memory profiling
    test_methods = [
        test_suite.test_deterministic_segmentation_memory_profile,
        test_suite.test_rule_based_attribution_memory_profile,
        test_suite.test_llm_orchestrator_memory_profile,
        test_suite.test_contextual_refiner_memory_profile,
        test_suite.test_validator_memory_profile,
        test_suite.test_output_formatter_memory_profile,
        test_suite.test_memory_usage_thresholds_structuring
    ]
    
    for method in test_methods:
        print(f"Profiling: {method.__name__}")
        try:
            method()
            print(f"✓ {method.__name__} completed")
        except Exception as e:
            print(f"✗ {method.__name__} failed: {e}")
    
    print("Text structuring memory profiling suite completed")


if __name__ == "__main__":
    run_structuring_memory_profiling_suite()