"""
Distributed validation and quality refinement using Spark.

This module provides distributed validation of processed text segments,
speaker detection refinement, and quality assurance checks across the pipeline.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, udf, when, regexp_replace, length, split, count, avg
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, BooleanType
import json
import time
from datetime import datetime

from ..monitoring.prometheus_metrics import get_metrics_collector


@dataclass
class ValidationResult:
    """Result of distributed validation process."""
    segment_id: str
    original_quality_score: float
    refined_quality_score: float
    validation_issues: List[str] = field(default_factory=list)
    refinement_applied: bool = False
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'segment_id': self.segment_id,
            'original_quality_score': self.original_quality_score,
            'refined_quality_score': self.refined_quality_score,
            'validation_issues': self.validation_issues,
            'refinement_applied': self.refinement_applied,
            'processing_time': self.processing_time
        }


@dataclass
class SpeakerValidationResult:
    """Result of speaker detection validation."""
    speaker_id: str
    confidence_score: float
    consistency_score: float
    dialogue_count: int
    refinement_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'speaker_id': self.speaker_id,
            'confidence_score': self.confidence_score,
            'consistency_score': self.consistency_score,
            'dialogue_count': self.dialogue_count,
            'refinement_suggestions': self.refinement_suggestions
        }


class DistributedValidationEngine:
    """
    Distributed validation engine using Spark for quality assurance and refinement.
    
    This class provides comprehensive validation and refinement capabilities:
    - Text segment quality validation
    - Speaker detection consistency checking
    - Dialogue attribution refinement
    - Quality score computation and improvement
    """
    
    def __init__(self, spark_session: Optional[SparkSession] = None):
        """
        Initialize the distributed validation engine.
        
        Args:
            spark_session: Existing Spark session or None to create new one
        """
        self.spark = spark_session or self._create_spark_session()
        self.logger = logging.getLogger(__name__)
        self.metrics_collector = get_metrics_collector()
        
        # Validation thresholds
        self.quality_thresholds = {
            'min_segment_length': 10,
            'max_segment_length': 1000,
            'min_speaker_confidence': 0.7,
            'min_consistency_score': 0.8
        }
        
        # Initialize validation schemas
        self._initialize_schemas()
        
        self.logger.info("Distributed validation engine initialized")
    
    def _create_spark_session(self) -> SparkSession:
        """Create Spark session optimized for validation tasks."""
        return SparkSession.builder \
            .appName("TextToAudiobook_DistributedValidation") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
    
    def _initialize_schemas(self):
        """Initialize Spark SQL schemas for validation data."""
        self.segment_schema = StructType([
            StructField("segment_id", StringType(), True),
            StructField("text_content", StringType(), True),
            StructField("speaker_id", StringType(), True),
            StructField("quality_score", FloatType(), True),
            StructField("confidence_score", FloatType(), True),
            StructField("segment_type", StringType(), True),
            StructField("processing_metadata", StringType(), True)
        ])
        
        self.speaker_schema = StructType([
            StructField("speaker_id", StringType(), True),
            StructField("name", StringType(), True),
            StructField("voice_characteristics", StringType(), True),
            StructField("dialogue_segments", IntegerType(), True),
            StructField("confidence_scores", StringType(), True)  # JSON array
        ])
    
    def validate_text_segments(self, segments_data: List[Dict[str, Any]]) -> List[ValidationResult]:
        """
        Validate and refine text segments using distributed processing.
        
        Args:
            segments_data: List of segment dictionaries
            
        Returns:
            List of validation results with refinement suggestions
        """
        start_time = time.time()
        
        try:
            # Create DataFrame from segments data
            segments_df = self.spark.createDataFrame(segments_data, self.segment_schema)
            
            # Cache for multiple operations
            segments_df.cache()
            
            # Perform distributed validation
            validated_df = self._perform_segment_validation(segments_df)
            
            # Apply refinements
            refined_df = self._apply_segment_refinements(validated_df)
            
            # Collect results
            results = []
            for row in refined_df.collect():
                result = ValidationResult(
                    segment_id=row.segment_id,
                    original_quality_score=row.original_quality_score,
                    refined_quality_score=row.refined_quality_score,
                    validation_issues=json.loads(row.validation_issues or "[]"),
                    refinement_applied=row.refinement_applied,
                    processing_time=row.processing_time
                )
                results.append(result)
            
            # Record metrics
            processing_time = time.time() - start_time
            self.metrics_collector.record_spark_job(
                "segment_validation", "completed", processing_time
            )
            
            self.logger.info(f"Validated {len(results)} segments in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Segment validation failed: {e}")
            self.metrics_collector.record_spark_job("segment_validation", "failed")
            raise
        finally:
            segments_df.unpersist()
    
    def _perform_segment_validation(self, segments_df: DataFrame) -> DataFrame:
        """Perform distributed validation checks on segments."""
        
        # Define validation UDFs
        @udf(returnType=StringType())
        def validate_segment_quality(text_content, quality_score, segment_type):
            """Validate individual segment quality."""
            issues = []
            
            if not text_content or len(text_content.strip()) < self.quality_thresholds['min_segment_length']:
                issues.append("segment_too_short")
            
            if len(text_content) > self.quality_thresholds['max_segment_length']:
                issues.append("segment_too_long")
            
            if quality_score < 0.5:
                issues.append("low_quality_score")
            
            # Check for dialogue formatting issues
            if segment_type == "dialogue" and not any(char in text_content for char in ['"', "'", '«', '»']):
                issues.append("missing_dialogue_markers")
            
            # Check for narrative consistency
            if segment_type == "narrative" and any(char in text_content for char in ['"', "'"]):
                issues.append("unexpected_dialogue_in_narrative")
            
            return json.dumps(issues)
        
        @udf(returnType=FloatType())
        def calculate_refined_quality(text_content, original_score, validation_issues):
            """Calculate refined quality score based on validation."""
            issues = json.loads(validation_issues or "[]")
            refined_score = original_score
            
            # Apply quality adjustments
            if "segment_too_short" in issues:
                refined_score *= 0.7
            if "segment_too_long" in issues:
                refined_score *= 0.8
            if "low_quality_score" in issues:
                refined_score = max(refined_score * 0.5, 0.1)
            if "missing_dialogue_markers" in issues:
                refined_score *= 0.6
            
            # Positive adjustments for good segments
            if not issues and len(text_content) > 50:
                refined_score = min(refined_score * 1.1, 1.0)
            
            return float(refined_score)
        
        # Register UDFs
        self.spark.udf.register("validate_segment_quality", validate_segment_quality)
        self.spark.udf.register("calculate_refined_quality", calculate_refined_quality)
        
        # Apply validation
        validated_df = segments_df.withColumn(
            "validation_issues",
            validate_segment_quality(col("text_content"), col("quality_score"), col("segment_type"))
        ).withColumn(
            "original_quality_score", col("quality_score")
        ).withColumn(
            "refined_quality_score",
            calculate_refined_quality(col("text_content"), col("quality_score"), col("validation_issues"))
        ).withColumn(
            "processing_time", 
            when(col("validation_issues").isNotNull(), 0.1).otherwise(0.05)
        )
        
        return validated_df
    
    def _apply_segment_refinements(self, validated_df: DataFrame) -> DataFrame:
        """Apply refinements to validated segments."""
        
        @udf(returnType=StringType())
        def refine_text_content(text_content, validation_issues):
            """Apply text refinements based on validation issues."""
            issues = json.loads(validation_issues or "[]")
            refined_text = text_content
            
            # Apply specific refinements
            if "missing_dialogue_markers" in issues:
                # Add basic dialogue markers if missing
                if not any(char in refined_text for char in ['"', "'", '«', '»']):
                    refined_text = f'"{refined_text}"'
            
            # Clean up excessive whitespace
            refined_text = regexp_replace(refined_text, r'\s+', ' ')
            
            # Remove leading/trailing whitespace
            refined_text = refined_text.strip()
            
            return refined_text
        
        @udf(returnType=BooleanType())
        def should_apply_refinement(validation_issues):
            """Determine if refinement should be applied."""
            issues = json.loads(validation_issues or "[]")
            refinement_issues = ["missing_dialogue_markers", "excessive_whitespace"]
            return any(issue in issues for issue in refinement_issues)
        
        # Register refinement UDFs
        self.spark.udf.register("refine_text_content", refine_text_content)
        self.spark.udf.register("should_apply_refinement", should_apply_refinement)
        
        # Apply refinements
        refined_df = validated_df.withColumn(
            "refined_text_content",
            refine_text_content(col("text_content"), col("validation_issues"))
        ).withColumn(
            "refinement_applied",
            should_apply_refinement(col("validation_issues"))
        )
        
        return refined_df
    
    def validate_speaker_consistency(self, speaker_data: List[Dict[str, Any]]) -> List[SpeakerValidationResult]:
        """
        Validate speaker detection consistency across segments.
        
        Args:
            speaker_data: List of speaker dictionaries with dialogue information
            
        Returns:
            List of speaker validation results
        """
        start_time = time.time()
        
        try:
            # Create DataFrame from speaker data
            speakers_df = self.spark.createDataFrame(speaker_data, self.speaker_schema)
            speakers_df.cache()
            
            # Perform speaker consistency validation
            validated_speakers_df = self._validate_speaker_consistency_distributed(speakers_df)
            
            # Collect results
            results = []
            for row in validated_speakers_df.collect():
                result = SpeakerValidationResult(
                    speaker_id=row.speaker_id,
                    confidence_score=row.avg_confidence,
                    consistency_score=row.consistency_score,
                    dialogue_count=row.dialogue_segments,
                    refinement_suggestions=json.loads(row.refinement_suggestions or "[]")
                )
                results.append(result)
            
            # Record metrics
            processing_time = time.time() - start_time
            self.metrics_collector.record_spark_job(
                "speaker_validation", "completed", processing_time
            )
            
            self.logger.info(f"Validated {len(results)} speakers in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Speaker validation failed: {e}")
            self.metrics_collector.record_spark_job("speaker_validation", "failed")
            raise
        finally:
            speakers_df.unpersist()
    
    def _validate_speaker_consistency_distributed(self, speakers_df: DataFrame) -> DataFrame:
        """Perform distributed speaker consistency validation."""
        
        @udf(returnType=FloatType())
        def calculate_consistency_score(confidence_scores_json, dialogue_count):
            """Calculate speaker consistency score."""
            try:
                scores = json.loads(confidence_scores_json)
                if not scores or len(scores) < 2:
                    return 0.5
                
                # Calculate standard deviation of confidence scores
                mean_score = sum(scores) / len(scores)
                variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
                std_dev = variance ** 0.5
                
                # Consistency score is inverse of std deviation, normalized
                consistency = max(0.0, 1.0 - (std_dev * 2))
                
                # Boost consistency for speakers with more dialogue
                if dialogue_count > 10:
                    consistency = min(consistency * 1.1, 1.0)
                
                return float(consistency)
            except:
                return 0.5
        
        @udf(returnType=StringType())
        def generate_refinement_suggestions(avg_confidence, consistency_score, dialogue_count):
            """Generate refinement suggestions for speakers."""
            suggestions = []
            
            if avg_confidence < self.quality_thresholds['min_speaker_confidence']:
                suggestions.append("improve_speaker_detection")
            
            if consistency_score < self.quality_thresholds['min_consistency_score']:
                suggestions.append("review_speaker_attribution")
            
            if dialogue_count < 3:
                suggestions.append("verify_speaker_presence")
            
            if dialogue_count > 50:
                suggestions.append("consider_speaker_splitting")
            
            return json.dumps(suggestions)
        
        # Register UDFs
        self.spark.udf.register("calculate_consistency_score", calculate_consistency_score)
        self.spark.udf.register("generate_refinement_suggestions", generate_refinement_suggestions)
        
        # Calculate average confidence scores
        @udf(returnType=FloatType())
        def calculate_avg_confidence(confidence_scores_json):
            """Calculate average confidence score."""
            try:
                scores = json.loads(confidence_scores_json)
                return float(sum(scores) / len(scores)) if scores else 0.0
            except:
                return 0.0
        
        self.spark.udf.register("calculate_avg_confidence", calculate_avg_confidence)
        
        # Apply speaker validation
        validated_speakers_df = speakers_df.withColumn(
            "avg_confidence",
            calculate_avg_confidence(col("confidence_scores"))
        ).withColumn(
            "consistency_score",
            calculate_consistency_score(col("confidence_scores"), col("dialogue_segments"))
        ).withColumn(
            "refinement_suggestions",
            generate_refinement_suggestions(
                col("avg_confidence"),
                col("consistency_score"),
                col("dialogue_segments")
            )
        )
        
        return validated_speakers_df
    
    def generate_quality_report(self, validation_results: List[ValidationResult], 
                              speaker_results: List[SpeakerValidationResult]) -> Dict[str, Any]:
        """
        Generate comprehensive quality report from validation results.
        
        Args:
            validation_results: List of segment validation results
            speaker_results: List of speaker validation results
            
        Returns:
            Comprehensive quality report
        """
        
        # Aggregate segment metrics
        total_segments = len(validation_results)
        avg_original_quality = sum(r.original_quality_score for r in validation_results) / total_segments
        avg_refined_quality = sum(r.refined_quality_score for r in validation_results) / total_segments
        refinements_applied = sum(1 for r in validation_results if r.refinement_applied)
        
        # Aggregate speaker metrics
        total_speakers = len(speaker_results)
        avg_speaker_confidence = sum(r.confidence_score for r in speaker_results) / total_speakers if total_speakers > 0 else 0
        avg_consistency = sum(r.consistency_score for r in speaker_results) / total_speakers if total_speakers > 0 else 0
        
        # Quality improvement metrics
        quality_improvement = avg_refined_quality - avg_original_quality
        improvement_percentage = (quality_improvement / avg_original_quality) * 100 if avg_original_quality > 0 else 0
        
        report = {
            'validation_summary': {
                'total_segments': total_segments,
                'total_speakers': total_speakers,
                'refinements_applied': refinements_applied,
                'refinement_rate': refinements_applied / total_segments if total_segments > 0 else 0
            },
            'quality_metrics': {
                'average_original_quality': avg_original_quality,
                'average_refined_quality': avg_refined_quality,
                'quality_improvement': quality_improvement,
                'improvement_percentage': improvement_percentage
            },
            'speaker_metrics': {
                'average_confidence': avg_speaker_confidence,
                'average_consistency': avg_consistency,
                'speakers_needing_review': sum(1 for r in speaker_results 
                                             if r.confidence_score < self.quality_thresholds['min_speaker_confidence'])
            },
            'recommendations': self._generate_recommendations(validation_results, speaker_results),
            'generated_at': datetime.now().isoformat()
        }
        
        # Record quality metrics
        self.metrics_collector.set_quality_score("overall", "validation", avg_refined_quality * 100)
        self.metrics_collector.set_quality_score("speakers", "consistency", avg_consistency * 100)
        
        return report
    
    def _generate_recommendations(self, validation_results: List[ValidationResult], 
                                speaker_results: List[SpeakerValidationResult]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Segment recommendations
        low_quality_segments = sum(1 for r in validation_results if r.refined_quality_score < 0.6)
        if low_quality_segments > len(validation_results) * 0.1:
            recommendations.append("Consider re-processing segments with low quality scores")
        
        # Speaker recommendations
        inconsistent_speakers = sum(1 for r in speaker_results 
                                  if r.consistency_score < self.quality_thresholds['min_consistency_score'])
        if inconsistent_speakers > 0:
            recommendations.append("Review speaker attribution for inconsistent speakers")
        
        # Performance recommendations
        total_refinements = sum(1 for r in validation_results if r.refinement_applied)
        if total_refinements > len(validation_results) * 0.3:
            recommendations.append("Consider improving initial processing to reduce refinement needs")
        
        return recommendations
    
    def cleanup(self):
        """Clean up Spark session and resources."""
        if self.spark:
            self.spark.stop()
        self.logger.info("Distributed validation engine cleaned up")


# Global validation engine instance
_global_validation_engine = None


def get_validation_engine() -> DistributedValidationEngine:
    """Get the global validation engine instance."""
    global _global_validation_engine
    
    if _global_validation_engine is None:
        _global_validation_engine = DistributedValidationEngine()
    
    return _global_validation_engine


def validate_segments_distributed(segments_data: List[Dict[str, Any]]) -> List[ValidationResult]:
    """Convenience function for distributed segment validation."""
    return get_validation_engine().validate_text_segments(segments_data)


def validate_speakers_distributed(speaker_data: List[Dict[str, Any]]) -> List[SpeakerValidationResult]:
    """Convenience function for distributed speaker validation."""
    return get_validation_engine().validate_speaker_consistency(speaker_data)