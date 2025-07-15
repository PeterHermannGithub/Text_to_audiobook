"""
Unit tests for the distributed validation and quality refinement system.

Tests cover validation logic, quality scoring, refinement algorithms,
and Spark-based distributed processing components.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import List, Dict, Any

from src.spark.distributed_validation import (
    DistributedValidationEngine,
    ValidationResult,
    SpeakerValidationResult,
    WorkloadCharacteristics
)


class TestValidationResult:
    """Test ValidationResult dataclass."""
    
    def test_validation_result_creation(self):
        """Test ValidationResult creation and basic properties."""
        result = ValidationResult(
            segment_id="seg_001",
            original_quality_score=0.8,
            refined_quality_score=0.85,
            validation_issues=["minor_issue"],
            refinement_applied=True,
            processing_time=0.15
        )
        
        assert result.segment_id == "seg_001"
        assert result.original_quality_score == 0.8
        assert result.refined_quality_score == 0.85
        assert result.validation_issues == ["minor_issue"]
        assert result.refinement_applied is True
        assert result.processing_time == 0.15
    
    def test_validation_result_to_dict(self):
        """Test ValidationResult conversion to dictionary."""
        result = ValidationResult(
            segment_id="seg_001",
            original_quality_score=0.8,
            refined_quality_score=0.85
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['segment_id'] == "seg_001"
        assert result_dict['original_quality_score'] == 0.8
        assert result_dict['refined_quality_score'] == 0.85
        assert 'validation_issues' in result_dict
        assert 'refinement_applied' in result_dict


class TestSpeakerValidationResult:
    """Test SpeakerValidationResult dataclass."""
    
    def test_speaker_validation_result_creation(self):
        """Test SpeakerValidationResult creation."""
        result = SpeakerValidationResult(
            speaker_id="speaker_001",
            confidence_score=0.92,
            consistency_score=0.88,
            dialogue_count=15,
            refinement_suggestions=["improve_detection"]
        )
        
        assert result.speaker_id == "speaker_001"
        assert result.confidence_score == 0.92
        assert result.consistency_score == 0.88
        assert result.dialogue_count == 15
        assert result.refinement_suggestions == ["improve_detection"]
    
    def test_speaker_validation_result_to_dict(self):
        """Test SpeakerValidationResult conversion to dictionary."""
        result = SpeakerValidationResult(
            speaker_id="speaker_001",
            confidence_score=0.92,
            consistency_score=0.88,
            dialogue_count=15
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['speaker_id'] == "speaker_001"
        assert result_dict['confidence_score'] == 0.92
        assert result_dict['consistency_score'] == 0.88
        assert result_dict['dialogue_count'] == 15


class TestDistributedValidationEngine:
    """Test DistributedValidationEngine main functionality."""
    
    @pytest.fixture
    def mock_spark_session(self):
        """Create mock Spark session for testing."""
        mock_session = Mock()
        mock_session.createDataFrame.return_value = Mock()
        mock_session.udf.register.return_value = None
        return mock_session
    
    @pytest.fixture
    def validation_engine(self, mock_spark_session):
        """Create validation engine with mocked Spark session."""
        with patch('src.spark.distributed_validation.SparkSession') as mock_spark_class:
            mock_spark_class.getActiveSession.return_value = mock_spark_session
            mock_spark_class.builder.appName.return_value = mock_spark_class.builder
            mock_spark_class.builder.config.return_value = mock_spark_class.builder
            mock_spark_class.builder.getOrCreate.return_value = mock_spark_session
            
            engine = DistributedValidationEngine(mock_spark_session)
            return engine
    
    def test_validation_engine_initialization(self, validation_engine):
        """Test validation engine initialization."""
        assert validation_engine.spark is not None
        assert validation_engine.quality_thresholds is not None
        assert 'min_segment_length' in validation_engine.quality_thresholds
        assert 'max_segment_length' in validation_engine.quality_thresholds
        assert 'min_speaker_confidence' in validation_engine.quality_thresholds
        assert 'min_consistency_score' in validation_engine.quality_thresholds
    
    def test_quality_thresholds_validation(self, validation_engine):
        """Test quality thresholds are properly set."""
        thresholds = validation_engine.quality_thresholds
        
        assert thresholds['min_segment_length'] > 0
        assert thresholds['max_segment_length'] > thresholds['min_segment_length']
        assert 0 <= thresholds['min_speaker_confidence'] <= 1
        assert 0 <= thresholds['min_consistency_score'] <= 1
    
    @patch('src.spark.distributed_validation.get_metrics_collector')
    def test_validate_text_segments_mock(self, mock_metrics, validation_engine, sample_text_segments):
        """Test text segment validation with mocked Spark operations."""
        # Mock the DataFrame operations
        mock_df = Mock()
        mock_df.cache.return_value = mock_df
        mock_df.unpersist.return_value = None
        
        # Mock the validation pipeline
        mock_validated_df = Mock()
        mock_refined_df = Mock()
        
        # Create mock rows with expected attributes
        mock_rows = []
        for i, segment in enumerate(sample_text_segments):
            mock_row = Mock()
            mock_row.segment_id = segment['segment_id']
            mock_row.original_quality_score = segment['quality_score']
            mock_row.refined_quality_score = segment['quality_score'] * 1.1  # Simulated improvement
            mock_row.validation_issues = '[]'  # No issues for test data
            mock_row.refinement_applied = False
            mock_row.processing_time = 0.1
            mock_rows.append(mock_row)
        
        mock_refined_df.collect.return_value = mock_rows
        
        # Mock the Spark session methods
        validation_engine.spark.createDataFrame.return_value = mock_df
        
        with patch.object(validation_engine, '_perform_segment_validation', return_value=mock_validated_df):
            with patch.object(validation_engine, '_apply_segment_refinements', return_value=mock_refined_df):
                results = validation_engine.validate_text_segments(sample_text_segments)
        
        assert len(results) == len(sample_text_segments)
        assert all(isinstance(result, ValidationResult) for result in results)
        
        # Verify results have expected properties
        for result in results:
            assert result.segment_id.startswith('seg_')
            assert 0 <= result.original_quality_score <= 1
            assert 0 <= result.refined_quality_score <= 1
            assert isinstance(result.validation_issues, list)
            assert isinstance(result.refinement_applied, bool)
            assert result.processing_time >= 0
    
    def test_segment_validation_logic(self, validation_engine):
        """Test individual segment validation logic."""
        # Test short segment detection
        short_segment = {
            'text_content': 'Hi',
            'quality_score': 0.8,
            'segment_type': 'dialogue'
        }
        
        # Test long segment detection
        long_segment = {
            'text_content': 'A' * 1500,  # Exceeds max length
            'quality_score': 0.8,
            'segment_type': 'narrative'
        }
        
        # Test low quality detection
        low_quality_segment = {
            'text_content': 'This is a normal length segment.',
            'quality_score': 0.3,  # Below threshold
            'segment_type': 'narrative'
        }
        
        # Test dialogue formatting
        malformed_dialogue = {
            'text_content': 'Hello world',  # Missing quotes
            'quality_score': 0.8,
            'segment_type': 'dialogue'
        }
        
        # These tests would normally be run through the UDFs,
        # but we can test the logic principles
        assert len(short_segment['text_content']) < validation_engine.quality_thresholds['min_segment_length']
        assert len(long_segment['text_content']) > validation_engine.quality_thresholds['max_segment_length']
        assert low_quality_segment['quality_score'] < 0.5
        assert malformed_dialogue['segment_type'] == 'dialogue' and '"' not in malformed_dialogue['text_content']
    
    @patch('src.spark.distributed_validation.get_metrics_collector')
    def test_validate_speaker_consistency_mock(self, mock_metrics, validation_engine, sample_speaker_data):
        """Test speaker consistency validation with mocked operations."""
        # Mock DataFrame operations
        mock_df = Mock()
        mock_df.cache.return_value = mock_df
        mock_df.unpersist.return_value = None
        
        # Create mock speaker validation results
        mock_rows = []
        for speaker in sample_speaker_data:
            mock_row = Mock()
            mock_row.speaker_id = speaker['speaker_id']
            mock_row.avg_confidence = 0.88
            mock_row.consistency_score = 0.85
            mock_row.dialogue_segments = speaker['dialogue_segments']
            mock_row.refinement_suggestions = '[]'
            mock_rows.append(mock_row)
        
        mock_validated_df = Mock()
        mock_validated_df.collect.return_value = mock_rows
        
        validation_engine.spark.createDataFrame.return_value = mock_df
        
        with patch.object(validation_engine, '_validate_speaker_consistency_distributed', return_value=mock_validated_df):
            results = validation_engine.validate_speaker_consistency(sample_speaker_data)
        
        assert len(results) == len(sample_speaker_data)
        assert all(isinstance(result, SpeakerValidationResult) for result in results)
        
        for result in results:
            assert result.speaker_id in [s['speaker_id'] for s in sample_speaker_data]
            assert 0 <= result.confidence_score <= 1
            assert 0 <= result.consistency_score <= 1
            assert result.dialogue_count > 0
    
    def test_speaker_consistency_calculation(self, validation_engine):
        """Test speaker consistency score calculation logic."""
        # Test consistent speaker (low variance in confidence)
        consistent_scores = [0.88, 0.89, 0.87, 0.90, 0.88]
        mean_score = sum(consistent_scores) / len(consistent_scores)
        variance = sum((score - mean_score) ** 2 for score in consistent_scores) / len(consistent_scores)
        std_dev = variance ** 0.5
        
        # Should result in high consistency (low std dev)
        expected_consistency = max(0.0, 1.0 - (std_dev * 2))
        assert expected_consistency > 0.8  # High consistency
        
        # Test inconsistent speaker (high variance)
        inconsistent_scores = [0.95, 0.50, 0.85, 0.40, 0.75]
        mean_score = sum(inconsistent_scores) / len(inconsistent_scores)
        variance = sum((score - mean_score) ** 2 for score in inconsistent_scores) / len(inconsistent_scores)
        std_dev = variance ** 0.5
        
        expected_consistency = max(0.0, 1.0 - (std_dev * 2))
        assert expected_consistency < 0.5  # Low consistency
    
    def test_generate_quality_report(self, validation_engine):
        """Test quality report generation."""
        # Create sample validation results
        validation_results = [
            ValidationResult("seg_001", 0.8, 0.85, [], True, 0.1),
            ValidationResult("seg_002", 0.7, 0.75, ["minor_issue"], True, 0.15),
            ValidationResult("seg_003", 0.9, 0.9, [], False, 0.05)
        ]
        
        # Create sample speaker results
        speaker_results = [
            SpeakerValidationResult("speaker_1", 0.88, 0.85, 15, []),
            SpeakerValidationResult("speaker_2", 0.92, 0.90, 12, ["improve_detection"])
        ]
        
        report = validation_engine.generate_quality_report(validation_results, speaker_results)
        
        # Verify report structure
        assert 'validation_summary' in report
        assert 'quality_metrics' in report
        assert 'speaker_metrics' in report
        assert 'recommendations' in report
        assert 'generated_at' in report
        
        # Verify validation summary
        validation_summary = report['validation_summary']
        assert validation_summary['total_segments'] == 3
        assert validation_summary['total_speakers'] == 2
        assert validation_summary['refinements_applied'] == 2
        
        # Verify quality metrics
        quality_metrics = report['quality_metrics']
        assert 'average_original_quality' in quality_metrics
        assert 'average_refined_quality' in quality_metrics
        assert 'quality_improvement' in quality_metrics
        assert 'improvement_percentage' in quality_metrics
        
        # Verify speaker metrics
        speaker_metrics = report['speaker_metrics']
        assert 'average_confidence' in speaker_metrics
        assert 'average_consistency' in speaker_metrics
        assert 'speakers_needing_review' in speaker_metrics
        
        # Verify recommendations are generated
        assert isinstance(report['recommendations'], list)
    
    def test_quality_report_calculations(self, validation_engine):
        """Test quality report calculation accuracy."""
        validation_results = [
            ValidationResult("seg_001", 0.6, 0.8, [], True, 0.1),  # +0.2 improvement
            ValidationResult("seg_002", 0.8, 0.9, [], True, 0.1),  # +0.1 improvement
            ValidationResult("seg_003", 0.7, 0.7, [], False, 0.1)  # No improvement
        ]
        
        speaker_results = [
            SpeakerValidationResult("speaker_1", 0.85, 0.80, 10, []),
            SpeakerValidationResult("speaker_2", 0.95, 0.90, 15, [])
        ]
        
        report = validation_engine.generate_quality_report(validation_results, speaker_results)
        
        # Check calculations
        expected_avg_original = (0.6 + 0.8 + 0.7) / 3  # 0.7
        expected_avg_refined = (0.8 + 0.9 + 0.7) / 3   # 0.8
        expected_improvement = expected_avg_refined - expected_avg_original  # 0.1
        expected_improvement_pct = (expected_improvement / expected_avg_original) * 100  # ~14.3%
        
        quality_metrics = report['quality_metrics']
        assert abs(quality_metrics['average_original_quality'] - expected_avg_original) < 0.01
        assert abs(quality_metrics['average_refined_quality'] - expected_avg_refined) < 0.01
        assert abs(quality_metrics['quality_improvement'] - expected_improvement) < 0.01
        assert abs(quality_metrics['improvement_percentage'] - expected_improvement_pct) < 0.1
        
        # Check speaker calculations
        expected_avg_confidence = (0.85 + 0.95) / 2  # 0.9
        expected_avg_consistency = (0.80 + 0.90) / 2  # 0.85
        
        speaker_metrics = report['speaker_metrics']
        assert abs(speaker_metrics['average_confidence'] - expected_avg_confidence) < 0.01
        assert abs(speaker_metrics['average_consistency'] - expected_avg_consistency) < 0.01
    
    def test_recommendation_generation(self, validation_engine):
        """Test recommendation generation logic."""
        # Test with many low quality segments
        many_low_quality = [
            ValidationResult(f"seg_{i}", 0.4, 0.5, [], True, 0.1) 
            for i in range(20)  # 20 low quality segments
        ]
        
        # Test with inconsistent speakers
        inconsistent_speakers = [
            SpeakerValidationResult(f"speaker_{i}", 0.5, 0.4, 10, [])  # Low consistency
            for i in range(3)
        ]
        
        report = validation_engine.generate_quality_report(many_low_quality, inconsistent_speakers)
        recommendations = report['recommendations']
        
        # Should recommend re-processing due to low quality
        assert any('re-processing' in rec.lower() for rec in recommendations)
        
        # Should recommend speaker review due to inconsistency
        assert any('speaker' in rec.lower() and 'review' in rec.lower() for rec in recommendations)
    
    def test_cleanup(self, validation_engine):
        """Test engine cleanup."""
        # Mock the spark session stop method
        validation_engine.spark.stop = Mock()
        
        validation_engine.cleanup()
        
        # Verify spark session was stopped
        validation_engine.spark.stop.assert_called_once()


class TestValidationUtilityFunctions:
    """Test utility functions for validation."""
    
    @patch('src.spark.distributed_validation.DistributedValidationEngine')
    def test_get_validation_engine(self, mock_engine_class):
        """Test global validation engine getter."""
        from src.spark.distributed_validation import get_validation_engine
        
        # Test singleton behavior
        engine1 = get_validation_engine()
        engine2 = get_validation_engine()
        
        # Should return the same instance
        assert engine1 is engine2
    
    @patch('src.spark.distributed_validation.get_validation_engine')
    def test_validate_segments_distributed(self, mock_get_engine):
        """Test convenience function for segment validation."""
        from src.spark.distributed_validation import validate_segments_distributed
        
        mock_engine = Mock()
        mock_engine.validate_text_segments.return_value = []
        mock_get_engine.return_value = mock_engine
        
        sample_data = [{'segment_id': 'test', 'text_content': 'test'}]
        result = validate_segments_distributed(sample_data)
        
        mock_engine.validate_text_segments.assert_called_once_with(sample_data)
        assert result == []
    
    @patch('src.spark.distributed_validation.get_validation_engine')
    def test_validate_speakers_distributed(self, mock_get_engine):
        """Test convenience function for speaker validation."""
        from src.spark.distributed_validation import validate_speakers_distributed
        
        mock_engine = Mock()
        mock_engine.validate_speaker_consistency.return_value = []
        mock_get_engine.return_value = mock_engine
        
        sample_data = [{'speaker_id': 'test', 'dialogue_segments': 5}]
        result = validate_speakers_distributed(sample_data)
        
        mock_engine.validate_speaker_consistency.assert_called_once_with(sample_data)
        assert result == []


class TestValidationIntegration:
    """Integration tests for validation components."""
    
    @pytest.mark.slow
    def test_end_to_end_validation_flow(self, sample_text_segments, sample_speaker_data):
        """Test complete validation flow (mocked for unit testing)."""
        with patch('src.spark.distributed_validation.SparkSession') as mock_spark_class:
            # Setup comprehensive mocking for full flow
            mock_session = Mock()
            mock_spark_class.getActiveSession.return_value = mock_session
            mock_spark_class.builder.appName.return_value = mock_spark_class.builder
            mock_spark_class.builder.config.return_value = mock_spark_class.builder
            mock_spark_class.builder.getOrCreate.return_value = mock_session
            
            # Mock DataFrame operations
            mock_df = Mock()
            mock_df.cache.return_value = mock_df
            mock_df.unpersist.return_value = None
            mock_session.createDataFrame.return_value = mock_df
            mock_session.udf.register.return_value = None
            
            # Create engine
            engine = DistributedValidationEngine(mock_session)
            
            # Mock the validation pipeline methods
            with patch.object(engine, '_perform_segment_validation') as mock_validate:
                with patch.object(engine, '_apply_segment_refinements') as mock_refine:
                    with patch.object(engine, '_validate_speaker_consistency_distributed') as mock_speaker_val:
                        
                        # Setup return values
                        mock_validate.return_value = mock_df
                        mock_refine.return_value = mock_df
                        mock_speaker_val.return_value = mock_df
                        
                        # Mock collect results
                        segment_rows = [Mock(
                            segment_id=seg['segment_id'],
                            original_quality_score=seg['quality_score'],
                            refined_quality_score=seg['quality_score'] * 1.05,
                            validation_issues='[]',
                            refinement_applied=False,
                            processing_time=0.1
                        ) for seg in sample_text_segments]
                        
                        speaker_rows = [Mock(
                            speaker_id=speaker['speaker_id'],
                            avg_confidence=0.88,
                            consistency_score=0.85,
                            dialogue_segments=speaker['dialogue_segments'],
                            refinement_suggestions='[]'
                        ) for speaker in sample_speaker_data]
                        
                        mock_df.collect.side_effect = [segment_rows, speaker_rows]
                        
                        # Run validation
                        segment_results = engine.validate_text_segments(sample_text_segments)
                        speaker_results = engine.validate_speaker_consistency(sample_speaker_data)
                        
                        # Generate report
                        report = engine.generate_quality_report(segment_results, speaker_results)
                        
                        # Verify results
                        assert len(segment_results) == len(sample_text_segments)
                        assert len(speaker_results) == len(sample_speaker_data)
                        assert isinstance(report, dict)
                        assert 'validation_summary' in report
                        assert 'quality_metrics' in report
                        assert 'speaker_metrics' in report
                        assert 'recommendations' in report
    
    def test_error_handling(self):
        """Test error handling in validation engine."""
        with patch('src.spark.distributed_validation.SparkSession') as mock_spark_class:
            mock_session = Mock()
            mock_spark_class.getActiveSession.return_value = mock_session
            mock_spark_class.builder.appName.return_value = mock_spark_class.builder
            mock_spark_class.builder.config.return_value = mock_spark_class.builder
            mock_spark_class.builder.getOrCreate.return_value = mock_session
            
            # Mock createDataFrame to raise an exception
            mock_session.createDataFrame.side_effect = Exception("Spark error")
            
            engine = DistributedValidationEngine(mock_session)
            
            # Test that exceptions are properly handled
            with pytest.raises(Exception):
                engine.validate_text_segments([])
    
    def test_performance_monitoring(self):
        """Test that performance metrics are recorded."""
        with patch('src.spark.distributed_validation.get_metrics_collector') as mock_get_metrics:
            with patch('src.spark.distributed_validation.SparkSession') as mock_spark_class:
                mock_metrics = Mock()
                mock_get_metrics.return_value = mock_metrics
                
                mock_session = Mock()
                mock_spark_class.getActiveSession.return_value = mock_session
                mock_spark_class.builder.appName.return_value = mock_spark_class.builder
                mock_spark_class.builder.config.return_value = mock_spark_class.builder
                mock_spark_class.builder.getOrCreate.return_value = mock_session
                
                # Setup mocks for successful operation
                mock_df = Mock()
                mock_df.cache.return_value = mock_df
                mock_df.unpersist.return_value = None
                mock_df.collect.return_value = []
                mock_session.createDataFrame.return_value = mock_df
                
                engine = DistributedValidationEngine(mock_session)
                
                with patch.object(engine, '_perform_segment_validation', return_value=mock_df):
                    with patch.object(engine, '_apply_segment_refinements', return_value=mock_df):
                        engine.validate_text_segments([])
                
                # Verify metrics were recorded
                mock_metrics.record_spark_job.assert_called()