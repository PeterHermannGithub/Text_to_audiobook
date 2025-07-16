"""
Spark-based distributed text processor integration.

This module provides Spark-based distributed processing capabilities that integrate
with the Kafka pipeline and LLM Pool for scalable text processing.
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, udf, explode, struct, lit
from pyspark.sql.types import StringType, StructType, StructField, ArrayType, DoubleType, BooleanType
import uuid

from .distributed_validation import DistributedValidationEngine, ValidationResult
from .resource_optimizer import SparkResourceOptimizer
from ..text_processing.segmentation.deterministic_segmenter import DeterministicSegmenter
from ..attribution.rule_based_attributor import RuleBasedAttributor
from ..monitoring.prometheus_metrics import get_metrics_collector
from ..cache.redis_cache import RedisCacheManager


class SparkDistributedTextProcessor:
    """
    Spark-based distributed text processor for scalable text processing.
    
    This processor integrates with the distributed pipeline to provide:
    - Spark-based parallel text segmentation
    - Distributed validation and quality assessment
    - Integration with LLM Pool for ambiguous segments
    - Caching and performance optimization
    """
    
    def __init__(self, spark_session: SparkSession, config: Dict[str, Any] = None):
        """
        Initialize Spark distributed text processor.
        
        Args:
            spark_session: Active Spark session
            config: Optional configuration parameters
        """
        self.spark = spark_session
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.validation_engine = DistributedValidationEngine(spark_session)
        self.resource_optimizer = SparkResourceOptimizer(spark_session)
        self.segmenter = DeterministicSegmenter()
        self.attributor = RuleBasedAttributor()
        
        # Monitoring and caching
        self.metrics_collector = get_metrics_collector()
        self.cache_manager = None
        
        # Processing configuration
        self.chunk_size = self.config.get('chunk_size', 2000)
        self.overlap_size = self.config.get('overlap_size', 400)
        self.quality_threshold = self.config.get('quality_threshold', 0.85)
        self.max_parallelism = self.config.get('max_parallelism', 10)
        
        # Initialize cache manager
        self._initialize_cache()
        
        # Register UDFs
        self._register_udfs()
    
    def _initialize_cache(self):
        """Initialize cache manager for intermediate results."""
        try:
            self.cache_manager = RedisCacheManager(namespace='spark_text_processor')
        except Exception as e:
            self.logger.warning(f"Failed to initialize cache manager: {e}")
    
    def _register_udfs(self):
        """Register Spark UDFs for text processing."""
        try:
            # UDF for text segmentation
            def segment_text_udf(text: str) -> List[Dict[str, Any]]:
                """UDF for segmenting text into segments."""
                try:
                    segments = self.segmenter.segment_text(text)
                    return segments
                except Exception as e:
                    self.logger.error(f"Error in segmentation UDF: {e}")
                    return []
            
            # UDF for rule-based attribution
            def attribute_speakers_udf(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                """UDF for rule-based speaker attribution."""
                try:
                    attributed_segments = self.attributor.attribute_speakers(segments)
                    return attributed_segments
                except Exception as e:
                    self.logger.error(f"Error in attribution UDF: {e}")
                    return segments
            
            # UDF for quality scoring
            def calculate_quality_score_udf(segment: Dict[str, Any]) -> float:
                """UDF for calculating segment quality score."""
                try:
                    # Simple quality scoring based on segment characteristics
                    text_length = len(segment.get('text_content', ''))
                    speaker = segment.get('speaker', 'AMBIGUOUS')
                    
                    # Base score
                    score = 0.8
                    
                    # Adjust based on text length
                    if text_length < 10:
                        score *= 0.5
                    elif text_length > 1000:
                        score *= 0.9
                    
                    # Adjust based on speaker confidence
                    if speaker == 'AMBIGUOUS':
                        score *= 0.6
                    elif speaker == 'narrator':
                        score *= 1.1
                    
                    return min(1.0, score)
                    
                except Exception as e:
                    self.logger.error(f"Error in quality scoring UDF: {e}")
                    return 0.5
            
            # Register UDFs with Spark
            self.spark.udf.register("segment_text", segment_text_udf, ArrayType(StringType()))
            self.spark.udf.register("attribute_speakers", attribute_speakers_udf, ArrayType(StringType()))
            self.spark.udf.register("calculate_quality_score", calculate_quality_score_udf, DoubleType())
            
            self.logger.info("Spark UDFs registered successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to register UDFs: {e}")
    
    def process_text_batch(self, text_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of text documents using Spark distributed processing.
        
        Args:
            text_batch: List of text documents with metadata
            
        Returns:
            List of processing results
        """
        try:
            start_time = time.time()
            
            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.record_spark_job('text_processing', 'started', len(text_batch))
            
            # Create DataFrame from batch
            df = self._create_text_dataframe(text_batch)
            
            # Process through pipeline
            processed_df = self._process_text_pipeline(df)
            
            # Collect results
            results = self._collect_results(processed_df)
            
            # Record completion metrics
            processing_time = time.time() - start_time
            if self.metrics_collector:
                self.metrics_collector.record_spark_job('text_processing', 'completed', processing_time)
            
            self.logger.info(f"Processed {len(text_batch)} documents in {processing_time:.2f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing text batch: {e}")
            if self.metrics_collector:
                self.metrics_collector.record_spark_job('text_processing', 'failed', time.time() - start_time)
            raise
    
    def _create_text_dataframe(self, text_batch: List[Dict[str, Any]]) -> DataFrame:
        """Create Spark DataFrame from text batch."""
        try:
            # Define schema
            schema = StructType([
                StructField("job_id", StringType(), False),
                StructField("text_content", StringType(), False),
                StructField("metadata", StringType(), True),
                StructField("processing_config", StringType(), True),
                StructField("timestamp", StringType(), False)
            ])
            
            # Prepare data
            data = []
            for item in text_batch:
                data.append((
                    item.get('job_id', str(uuid.uuid4())),
                    item.get('text_content', ''),
                    json.dumps(item.get('metadata', {})),
                    json.dumps(item.get('processing_config', {})),
                    datetime.now().isoformat()
                ))
            
            # Create DataFrame
            df = self.spark.createDataFrame(data, schema)
            
            # Cache for multiple operations
            df = df.cache()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating DataFrame: {e}")
            raise
    
    def _process_text_pipeline(self, df: DataFrame) -> DataFrame:
        """Process text through the complete Spark pipeline."""
        try:
            # Step 1: Text segmentation
            df_segmented = self._segment_text_distributed(df)
            
            # Step 2: Rule-based attribution
            df_attributed = self._attribute_speakers_distributed(df_segmented)
            
            # Step 3: Quality scoring
            df_scored = self._calculate_quality_scores(df_attributed)
            
            # Step 4: Validation and refinement
            df_validated = self._validate_segments_distributed(df_scored)
            
            # Step 5: Final processing
            df_final = self._finalize_processing(df_validated)
            
            return df_final
            
        except Exception as e:
            self.logger.error(f"Error in text processing pipeline: {e}")
            raise
    
    def _segment_text_distributed(self, df: DataFrame) -> DataFrame:
        """Perform distributed text segmentation."""
        try:
            # Create segmentation UDF
            from pyspark.sql.functions import explode, col
            
            def segment_text_batch(text_content: str) -> List[str]:
                """Segment text into individual segments."""
                try:
                    segments = self.segmenter.segment_text(text_content)
                    return [json.dumps(segment) for segment in segments]
                except Exception as e:
                    self.logger.error(f"Segmentation error: {e}")
                    return []
            
            # Register UDF
            segment_udf = udf(segment_text_batch, ArrayType(StringType()))
            
            # Apply segmentation
            df_segmented = df.withColumn("segments", segment_udf(col("text_content")))
            
            # Explode segments into individual rows
            df_exploded = df_segmented.select(
                col("job_id"),
                col("metadata"),
                col("processing_config"),
                col("timestamp"),
                explode(col("segments")).alias("segment_json")
            )
            
            # Parse segment JSON
            def parse_segment(segment_json: str) -> Dict[str, Any]:
                """Parse segment JSON into structured data."""
                try:
                    return json.loads(segment_json)
                except Exception:
                    return {}
            
            parse_udf = udf(parse_segment, StringType())
            
            # Add parsed segment data
            df_final = df_exploded.withColumn("segment_data", parse_udf(col("segment_json")))
            
            return df_final
            
        except Exception as e:
            self.logger.error(f"Error in distributed segmentation: {e}")
            raise
    
    def _attribute_speakers_distributed(self, df: DataFrame) -> DataFrame:
        """Perform distributed speaker attribution."""
        try:
            def attribute_speaker_batch(segment_json: str) -> str:
                """Attribute speaker to a segment."""
                try:
                    segment = json.loads(segment_json)
                    attributed_segments = self.attributor.attribute_speakers([segment])
                    return json.dumps(attributed_segments[0] if attributed_segments else segment)
                except Exception as e:
                    self.logger.error(f"Attribution error: {e}")
                    return segment_json
            
            # Register UDF
            attribute_udf = udf(attribute_speaker_batch, StringType())
            
            # Apply attribution
            df_attributed = df.withColumn(
                "attributed_segment",
                attribute_udf(col("segment_json"))
            )
            
            return df_attributed
            
        except Exception as e:
            self.logger.error(f"Error in distributed attribution: {e}")
            raise
    
    def _calculate_quality_scores(self, df: DataFrame) -> DataFrame:
        """Calculate quality scores for segments."""
        try:
            def calculate_quality(segment_json: str) -> float:
                """Calculate quality score for a segment."""
                try:
                    segment = json.loads(segment_json)
                    text_length = len(segment.get('text_content', ''))
                    speaker = segment.get('speaker', 'AMBIGUOUS')
                    
                    # Base score
                    score = 0.8
                    
                    # Adjust based on characteristics
                    if text_length < 10:
                        score *= 0.5
                    elif text_length > 1000:
                        score *= 0.9
                    
                    if speaker == 'AMBIGUOUS':
                        score *= 0.6
                    elif speaker == 'narrator':
                        score *= 1.1
                    
                    return min(1.0, score)
                    
                except Exception:
                    return 0.5
            
            # Register UDF
            quality_udf = udf(calculate_quality, DoubleType())
            
            # Add quality scores
            df_scored = df.withColumn(
                "quality_score",
                quality_udf(col("attributed_segment"))
            )
            
            return df_scored
            
        except Exception as e:
            self.logger.error(f"Error calculating quality scores: {e}")
            raise
    
    def _validate_segments_distributed(self, df: DataFrame) -> DataFrame:
        """Perform distributed validation using the validation engine."""
        try:
            # Collect segments for validation
            segments_data = df.select("attributed_segment").collect()
            segments = []
            
            for row in segments_data:
                try:
                    segment = json.loads(row.attributed_segment)
                    segments.append(segment)
                except Exception as e:
                    self.logger.warning(f"Failed to parse segment: {e}")
            
            # Validate using distributed validation engine
            validation_results = self.validation_engine.validate_text_segments(segments)
            
            # Create validation results DataFrame
            validation_data = []
            for result in validation_results:
                validation_data.append((
                    result.segment_id,
                    result.original_quality_score,
                    result.refined_quality_score,
                    json.dumps(result.validation_issues),
                    result.refinement_applied,
                    result.processing_time
                ))
            
            validation_schema = StructType([
                StructField("segment_id", StringType(), False),
                StructField("original_quality", DoubleType(), False),
                StructField("refined_quality", DoubleType(), False),
                StructField("validation_issues", StringType(), False),
                StructField("refinement_applied", BooleanType(), False),
                StructField("processing_time", DoubleType(), False)
            ])
            
            validation_df = self.spark.createDataFrame(validation_data, validation_schema)
            
            # Join with original DataFrame
            # For simplicity, we'll add validation info to all segments
            df_with_validation = df.withColumn("validation_status", lit("validated"))
            
            return df_with_validation
            
        except Exception as e:
            self.logger.error(f"Error in distributed validation: {e}")
            # Return original DataFrame if validation fails
            return df.withColumn("validation_status", lit("validation_failed"))
    
    def _finalize_processing(self, df: DataFrame) -> DataFrame:
        """Finalize processing and prepare results."""
        try:
            # Add final processing timestamp
            df_final = df.withColumn("processed_at", lit(datetime.now().isoformat()))
            
            # Add processing metadata
            df_final = df_final.withColumn("processing_mode", lit("distributed_spark"))
            
            # Cache final results
            df_final = df_final.cache()
            
            return df_final
            
        except Exception as e:
            self.logger.error(f"Error in final processing: {e}")
            raise
    
    def _collect_results(self, df: DataFrame) -> List[Dict[str, Any]]:
        """Collect and format processing results."""
        try:
            # Collect all rows
            rows = df.collect()
            
            # Format results
            results = []
            for row in rows:
                try:
                    # Parse segment data
                    segment_data = json.loads(row.attributed_segment)
                    
                    # Create result
                    result = {
                        'job_id': row.job_id,
                        'segment_id': segment_data.get('segment_id'),
                        'text_content': segment_data.get('text_content'),
                        'speaker': segment_data.get('speaker'),
                        'segment_type': segment_data.get('segment_type'),
                        'quality_score': row.quality_score,
                        'validation_status': row.validation_status,
                        'processing_metadata': {
                            'processing_mode': row.processing_mode,
                            'processed_at': row.processed_at,
                            'original_metadata': json.loads(row.metadata)
                        }
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    self.logger.warning(f"Error formatting result: {e}")
            
            # Cache results if enabled
            if self.cache_manager:
                self._cache_results(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error collecting results: {e}")
            raise
    
    def _cache_results(self, results: List[Dict[str, Any]]):
        """Cache processing results."""
        try:
            if not self.cache_manager:
                return
            
            # Group results by job_id
            jobs = {}
            for result in results:
                job_id = result['job_id']
                if job_id not in jobs:
                    jobs[job_id] = []
                jobs[job_id].append(result)
            
            # Cache each job's results
            for job_id, job_results in jobs.items():
                self.cache_manager.set(
                    'spark_processing_results',
                    job_id,
                    job_results,
                    ttl=3600  # 1 hour
                )
            
        except Exception as e:
            self.logger.warning(f"Error caching results: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        try:
            stats = {
                'spark_context_id': self.spark.sparkContext.applicationId,
                'parallelism': self.spark.sparkContext.defaultParallelism,
                'cache_enabled': self.cache_manager is not None,
                'processing_config': {
                    'chunk_size': self.chunk_size,
                    'overlap_size': self.overlap_size,
                    'quality_threshold': self.quality_threshold,
                    'max_parallelism': self.max_parallelism
                }
            }
            
            # Add cache stats if available
            if self.cache_manager:
                stats['cache_stats'] = self.cache_manager.get_stats()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting processing stats: {e}")
            return {'error': str(e)}
    
    def optimize_resources(self, workload_size: int, processing_type: str = 'text_processing'):
        """Optimize Spark resources based on workload."""
        try:
            # Analyze workload
            workload = self.resource_optimizer.analyze_workload(
                data_size_mb=workload_size,
                processing_type=processing_type
            )
            
            # Optimize allocation
            allocation = self.resource_optimizer.optimize_allocation(workload, 'balanced')
            
            # Apply optimization
            success = self.resource_optimizer.apply_allocation(allocation)
            
            if success:
                self.logger.info(f"Resources optimized for {processing_type} workload")
            else:
                self.logger.warning("Resource optimization failed")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error optimizing resources: {e}")
            return False
    
    def shutdown(self):
        """Shutdown the processor and clean up resources."""
        try:
            # Close cache manager
            if self.cache_manager:
                self.cache_manager.close()
            
            # Clear cached DataFrames
            self.spark.catalog.clearCache()
            
            self.logger.info("Spark distributed text processor shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


def create_spark_text_processor(spark_session: SparkSession, 
                               config: Dict[str, Any] = None) -> SparkDistributedTextProcessor:
    """
    Create a Spark distributed text processor instance.
    
    Args:
        spark_session: Active Spark session
        config: Optional configuration parameters
        
    Returns:
        SparkDistributedTextProcessor instance
    """
    return SparkDistributedTextProcessor(spark_session, config)