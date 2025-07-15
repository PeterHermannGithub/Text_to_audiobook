"""
Distributed text structurer using Apache Spark.

This module provides a distributed version of the TextStructurer that can
process large documents across multiple workers with parallel LLM inference
and sophisticated error handling.
"""

import logging
import time
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import Future, ThreadPoolExecutor
import traceback

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, udf, explode, collect_list, struct, lit, when, coalesce
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType, ArrayType, MapType, DoubleType
from pyspark.broadcast import Broadcast

from ..text_processing.text_extractor import TextExtractor
from ..text_processing.preprocessor import TextPreprocessor
from ..text_processing.segmentation.deterministic_segmenter import DeterministicSegmenter
from ..text_processing.segmentation.chunking import ChunkManager
from ..attribution.rule_based_attributor import RuleBasedAttributor
from ..validation.validator import SimplifiedValidator
from ..refinement.contextual_refiner import ContextualRefiner
from ..output.output_formatter import OutputFormatter
from ..llm_pool.llm_client import SparkLLMClient
from ..kafka.producers.chunk_producer import ChunkProducer
from ..kafka.consumers.llm_consumer import LLMConsumer
from ..kafka.schemas.message_schemas import MessageFactory, ProcessingStatus

from .spark_config import SparkConfig
from config import settings


@dataclass
class ProcessingChunk:
    """Represents a chunk of text for processing."""
    chunk_id: str
    job_id: str
    chunk_index: int
    total_chunks: int
    text_lines: List[str]
    context_lines: List[str]
    metadata: Dict[str, Any]
    context_hint: Dict[str, Any]
    created_at: float
    status: str = "pending"
    retry_count: int = 0


@dataclass
class ProcessingResult:
    """Represents the result of processing a chunk."""
    chunk_id: str
    job_id: str
    chunk_index: int
    classifications: List[str]
    processing_time: float
    worker_id: str
    success: bool
    error_message: Optional[str] = None
    validation_score: float = 0.0
    retry_count: int = 0


class SparkTextStructurer:
    """
    Distributed text structurer using Apache Spark for parallel processing.
    
    This class provides a distributed version of the original TextStructurer
    that can scale across multiple workers with parallel LLM inference.
    """
    
    def __init__(self, spark_session: SparkSession = None, config: Dict[str, Any] = None):
        """Initialize the Spark text structurer."""
        self.spark = spark_session or SparkConfig.create_spark_session()
        self.config = config or SparkConfig.get_distributed_config()
        self.job_id = str(uuid.uuid4())
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.text_extractor = TextExtractor()
        self.preprocessor = TextPreprocessor()
        self.segmenter = DeterministicSegmenter()
        self.chunk_manager = ChunkManager()
        self.rule_based_attributor = RuleBasedAttributor()
        self.validator = SimplifiedValidator()
        self.contextual_refiner = ContextualRefiner()
        self.output_formatter = OutputFormatter()
        
        # Kafka integration
        self.chunk_producer = ChunkProducer() if self._kafka_enabled() else None
        self.llm_consumer = LLMConsumer() if self._kafka_enabled() else None
        
        # Metrics
        self.processing_metrics = {
            'total_chunks': 0,
            'successful_chunks': 0,
            'failed_chunks': 0,
            'retry_count': 0,
            'total_processing_time': 0.0,
            'llm_processing_time': 0.0,
            'validation_time': 0.0,
            'start_time': 0.0,
            'end_time': 0.0
        }
        
        # Create DataFrame schemas
        self.chunk_schema = self._create_chunk_schema()
        self.result_schema = self._create_result_schema()
        
        self.logger.info(f"SparkTextStructurer initialized with job_id: {self.job_id}")
    
    def _kafka_enabled(self) -> bool:
        """Check if Kafka integration is enabled."""
        return getattr(settings, 'KAFKA_ENABLED', False)
    
    def _create_chunk_schema(self) -> StructType:
        """Create schema for processing chunks."""
        return StructType([
            StructField("chunk_id", StringType(), False),
            StructField("job_id", StringType(), False),
            StructField("chunk_index", IntegerType(), False),
            StructField("total_chunks", IntegerType(), False),
            StructField("text_lines", ArrayType(StringType()), False),
            StructField("context_lines", ArrayType(StringType()), False),
            StructField("metadata", MapType(StringType(), StringType()), False),
            StructField("context_hint", MapType(StringType(), StringType()), False),
            StructField("created_at", DoubleType(), False),
            StructField("status", StringType(), False),
            StructField("retry_count", IntegerType(), False)
        ])
    
    def _create_result_schema(self) -> StructType:
        """Create schema for processing results."""
        return StructType([
            StructField("chunk_id", StringType(), False),
            StructField("job_id", StringType(), False),
            StructField("chunk_index", IntegerType(), False),
            StructField("classifications", ArrayType(StringType()), False),
            StructField("processing_time", DoubleType(), False),
            StructField("worker_id", StringType(), False),
            StructField("success", BooleanType(), False),
            StructField("error_message", StringType(), True),
            StructField("validation_score", DoubleType(), False),
            StructField("retry_count", IntegerType(), False)
        ])
    
    def structure_text(self, text_content: str, processing_options: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Structure text using distributed processing.
        
        Args:
            text_content: The text to structure
            processing_options: Optional processing configuration
            
        Returns:
            List of structured segments with speaker attribution
        """
        start_time = time.time()
        self.processing_metrics['start_time'] = start_time
        
        try:
            self.logger.info(f"Starting distributed text structuring for job {self.job_id}")
            
            # Phase 1: Preprocessing and metadata extraction
            self.logger.info("Phase 1: Preprocessing and metadata extraction")
            text_metadata = self._extract_metadata(text_content)
            
            # Phase 2: Create processing chunks
            self.logger.info("Phase 2: Creating processing chunks")
            chunks = self._create_processing_chunks(text_content, text_metadata)
            
            if not chunks:
                self.logger.warning("No chunks created for processing")
                return []
            
            # Phase 3: Distributed processing
            self.logger.info(f"Phase 3: Distributed processing of {len(chunks)} chunks")
            results = self._process_chunks_distributed(chunks, text_metadata)
            
            # Phase 4: Merge and validate results
            self.logger.info("Phase 4: Merging and validating results")
            structured_segments = self._merge_results(results, text_metadata)
            
            # Phase 5: Contextual refinement
            self.logger.info("Phase 5: Contextual refinement")
            refined_segments = self._refine_segments(structured_segments, text_metadata)
            
            # Phase 6: Final formatting
            self.logger.info("Phase 6: Final formatting")
            final_segments = self._format_output(refined_segments)
            
            # Update metrics
            self.processing_metrics['end_time'] = time.time()
            self.processing_metrics['total_processing_time'] = (
                self.processing_metrics['end_time'] - self.processing_metrics['start_time']
            )
            
            self.logger.info(f"Distributed text structuring completed in {self.processing_metrics['total_processing_time']:.2f}s")
            self._log_processing_metrics()
            
            return final_segments
            
        except Exception as e:
            self.logger.error(f"Error in distributed text structuring: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _extract_metadata(self, text_content: str) -> Dict[str, Any]:
        """Extract metadata from text content."""
        try:
            # Use existing preprocessor for metadata extraction
            nlp_model = self.preprocessor.nlp_model
            metadata = self.preprocessor.analyze(text_content)
            
            self.logger.info(f"Metadata extracted: {len(metadata.get('potential_character_names', set()))} characters found")
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {e}")
            return {}
    
    def _create_processing_chunks(self, text_content: str, metadata: Dict[str, Any]) -> List[ProcessingChunk]:
        """Create processing chunks from text content."""
        try:
            # Create sliding windows using existing chunk manager
            scene_breaks = metadata.get('scene_breaks', [])
            
            if settings.SLIDING_WINDOW_ENABLED:
                windows = self.chunk_manager.create_sliding_windows(text_content, scene_breaks, metadata)
            else:
                chunks = self.chunk_manager.create_chunks(text_content, scene_breaks)
                windows = self.chunk_manager._convert_chunks_to_windows(chunks)
            
            # Convert to ProcessingChunk objects
            processing_chunks = []
            
            for i, window in enumerate(windows):
                chunk_id = f"{self.job_id}-chunk-{i}"
                
                chunk = ProcessingChunk(
                    chunk_id=chunk_id,
                    job_id=self.job_id,
                    chunk_index=i,
                    total_chunks=len(windows),
                    text_lines=window['task_lines'],
                    context_lines=window['context_lines'],
                    metadata=metadata,
                    context_hint=self.chunk_manager.create_context_hint_for_chunk(i, {}),
                    created_at=time.time(),
                    status="pending"
                )
                
                processing_chunks.append(chunk)
            
            self.processing_metrics['total_chunks'] = len(processing_chunks)
            self.logger.info(f"Created {len(processing_chunks)} processing chunks")
            
            return processing_chunks
            
        except Exception as e:
            self.logger.error(f"Error creating processing chunks: {e}")
            return []
    
    def _process_chunks_distributed(self, chunks: List[ProcessingChunk], metadata: Dict[str, Any]) -> List[ProcessingResult]:
        """Process chunks using distributed Spark processing."""
        try:
            # Convert chunks to Spark DataFrame
            chunk_data = [asdict(chunk) for chunk in chunks]
            chunks_df = self.spark.createDataFrame(chunk_data, self.chunk_schema)
            
            # Cache the DataFrame for better performance
            chunks_df.cache()
            
            # Broadcast metadata for efficient access across workers
            metadata_broadcast = self.spark.sparkContext.broadcast(metadata)
            
            # Create UDF for processing chunks
            process_chunk_udf = udf(self._create_process_chunk_function(metadata_broadcast), self.result_schema)
            
            # Process chunks in parallel
            results_df = chunks_df.select(
                process_chunk_udf(struct([col(c) for c in chunks_df.columns])).alias("result")
            ).select("result.*")
            
            # Collect results
            results_data = results_df.collect()
            
            # Convert back to ProcessingResult objects
            results = []
            for row in results_data:
                result = ProcessingResult(
                    chunk_id=row['chunk_id'],
                    job_id=row['job_id'],
                    chunk_index=row['chunk_index'],
                    classifications=row['classifications'],
                    processing_time=row['processing_time'],
                    worker_id=row['worker_id'],
                    success=row['success'],
                    error_message=row['error_message'],
                    validation_score=row['validation_score'],
                    retry_count=row['retry_count']
                )
                results.append(result)
            
            # Update metrics
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]
            
            self.processing_metrics['successful_chunks'] = len(successful_results)
            self.processing_metrics['failed_chunks'] = len(failed_results)
            self.processing_metrics['llm_processing_time'] = sum(r.processing_time for r in results)
            
            self.logger.info(f"Distributed processing completed: {len(successful_results)} successful, {len(failed_results)} failed")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in distributed chunk processing: {e}")
            return []
    
    def _create_process_chunk_function(self, metadata_broadcast: Broadcast):
        """Create a function for processing chunks in Spark workers."""
        def process_chunk(chunk_struct):
            """Process a single chunk in a Spark worker."""
            try:
                # Extract chunk data
                chunk_data = chunk_struct.asDict()
                
                # Create LLM client for this worker
                worker_id = f"spark-worker-{time.time()}"
                llm_client = SparkLLMClient(worker_id)
                
                # Convert to numbered lines format
                numbered_lines = [{"line_id": i+1, "text": line} 
                                for i, line in enumerate(chunk_data['text_lines'])]
                
                if not numbered_lines:
                    return {
                        'chunk_id': chunk_data['chunk_id'],
                        'job_id': chunk_data['job_id'],
                        'chunk_index': chunk_data['chunk_index'],
                        'classifications': [],
                        'processing_time': 0.0,
                        'worker_id': worker_id,
                        'success': False,
                        'error_message': "No text lines to process",
                        'validation_score': 0.0,
                        'retry_count': 0
                    }
                
                start_time = time.time()
                
                # Phase 1: Rule-based attribution
                rule_attributor = RuleBasedAttributor()
                attributed_lines = rule_attributor.process_lines(numbered_lines, metadata_broadcast.value)
                
                # Separate lines that need AI processing
                pending_ai_lines = rule_attributor.get_pending_lines(attributed_lines)
                rule_attributed_lines = rule_attributor.get_attributed_lines(attributed_lines)
                
                # Phase 2: LLM processing for remaining lines
                classifications = []
                if pending_ai_lines:
                    # Process with LLM
                    llm_chunk_data = {
                        'chunk_id': chunk_data['chunk_id'],
                        'text_lines': [line['text'] for line in pending_ai_lines],
                        'metadata': metadata_broadcast.value,
                        'context_hint': chunk_data['context_hint']
                    }
                    
                    result = llm_client.process_chunk(llm_chunk_data)
                    
                    if result['success']:
                        llm_classifications = result['classifications']
                    else:
                        llm_classifications = ["AMBIGUOUS"] * len(pending_ai_lines)
                    
                    # Merge rule-based and LLM classifications
                    classifications = rule_attributor.merge_classifications(
                        rule_attributed_lines, llm_classifications
                    )
                else:
                    # Only rule-based classifications
                    classifications = [line.get('speaker', 'AMBIGUOUS') for line in rule_attributed_lines]
                
                processing_time = time.time() - start_time
                
                # Basic validation
                validation_score = 0.9 if all(c != "AMBIGUOUS" for c in classifications) else 0.7
                
                return {
                    'chunk_id': chunk_data['chunk_id'],
                    'job_id': chunk_data['job_id'],
                    'chunk_index': chunk_data['chunk_index'],
                    'classifications': classifications,
                    'processing_time': processing_time,
                    'worker_id': worker_id,
                    'success': True,
                    'error_message': None,
                    'validation_score': validation_score,
                    'retry_count': 0
                }
                
            except Exception as e:
                return {
                    'chunk_id': chunk_data.get('chunk_id', 'unknown'),
                    'job_id': chunk_data.get('job_id', 'unknown'),
                    'chunk_index': chunk_data.get('chunk_index', 0),
                    'classifications': [],
                    'processing_time': 0.0,
                    'worker_id': f"spark-worker-{time.time()}",
                    'success': False,
                    'error_message': str(e),
                    'validation_score': 0.0,
                    'retry_count': 0
                }
        
        return process_chunk
    
    def _merge_results(self, results: List[ProcessingResult], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Merge processing results into structured segments."""
        try:
            # Sort results by chunk index
            results.sort(key=lambda r: r.chunk_index)
            
            # Merge classifications into structured segments
            all_segments = []
            
            for result in results:
                if result.success and result.classifications:
                    # Create segments for this chunk
                    chunk_segments = []
                    
                    for i, classification in enumerate(result.classifications):
                        segment = {
                            'speaker': classification,
                            'text': '',  # Will be populated later
                            'line_id': len(all_segments) + i + 1,
                            'chunk_id': result.chunk_id,
                            'chunk_index': result.chunk_index,
                            'worker_id': result.worker_id,
                            'processing_time': result.processing_time,
                            'validation_score': result.validation_score
                        }
                        chunk_segments.append(segment)
                    
                    all_segments.extend(chunk_segments)
                else:
                    self.logger.warning(f"Skipping failed chunk result: {result.chunk_id}")
            
            self.logger.info(f"Merged results into {len(all_segments)} segments")
            
            return all_segments
            
        except Exception as e:
            self.logger.error(f"Error merging results: {e}")
            return []
    
    def _refine_segments(self, segments: List[Dict[str, Any]], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Refine segments using contextual refinement."""
        try:
            # Use existing contextual refiner
            refined_segments = self.contextual_refiner.refine_segments(segments, metadata)
            
            self.logger.info(f"Contextual refinement completed on {len(refined_segments)} segments")
            
            return refined_segments
            
        except Exception as e:
            self.logger.error(f"Error in contextual refinement: {e}")
            return segments
    
    def _format_output(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format final output."""
        try:
            # Use existing output formatter
            formatted_segments = self.output_formatter.format_segments(segments)
            
            self.logger.info(f"Output formatting completed on {len(formatted_segments)} segments")
            
            return formatted_segments
            
        except Exception as e:
            self.logger.error(f"Error in output formatting: {e}")
            return segments
    
    def _log_processing_metrics(self):
        """Log processing metrics."""
        metrics = self.processing_metrics
        
        self.logger.info("=== DISTRIBUTED PROCESSING METRICS ===")
        self.logger.info(f"Job ID: {self.job_id}")
        self.logger.info(f"Total chunks: {metrics['total_chunks']}")
        self.logger.info(f"Successful chunks: {metrics['successful_chunks']}")
        self.logger.info(f"Failed chunks: {metrics['failed_chunks']}")
        self.logger.info(f"Success rate: {metrics['successful_chunks'] / max(1, metrics['total_chunks']) * 100:.1f}%")
        self.logger.info(f"Total processing time: {metrics['total_processing_time']:.2f}s")
        self.logger.info(f"LLM processing time: {metrics['llm_processing_time']:.2f}s")
        self.logger.info(f"Average time per chunk: {metrics['llm_processing_time'] / max(1, metrics['total_chunks']):.2f}s")
        
        if metrics['failed_chunks'] > 0:
            self.logger.warning(f"Failed chunks detected: {metrics['failed_chunks']}")
    
    def get_processing_metrics(self) -> Dict[str, Any]:
        """Get current processing metrics."""
        return self.processing_metrics.copy()
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of distributed processing system."""
        try:
            # Check Spark session
            spark_healthy = self.spark is not None and not self.spark.sparkContext._jsc.sc().isStopped()
            
            # Check LLM pool (if available)
            llm_healthy = True
            try:
                from ..llm_pool.llm_pool_manager import get_pool_manager
                pool_manager = get_pool_manager()
                pool_status = pool_manager.get_pool_status()
                llm_healthy = pool_status['healthy_instances'] > 0
            except:
                llm_healthy = False
            
            # Check Kafka (if enabled)
            kafka_healthy = True
            if self._kafka_enabled():
                try:
                    from ..kafka.kafka_config import KafkaHealthCheck
                    kafka_status = KafkaHealthCheck.check_cluster_health()
                    kafka_healthy = kafka_status['status'] == 'healthy'
                except:
                    kafka_healthy = False
            
            overall_healthy = spark_healthy and llm_healthy and kafka_healthy
            
            return {
                'overall_health': 'healthy' if overall_healthy else 'unhealthy',
                'spark_health': 'healthy' if spark_healthy else 'unhealthy',
                'llm_pool_health': 'healthy' if llm_healthy else 'unhealthy',
                'kafka_health': 'healthy' if kafka_healthy else 'unhealthy',
                'job_id': self.job_id,
                'metrics': self.processing_metrics
            }
            
        except Exception as e:
            return {
                'overall_health': 'unhealthy',
                'error': str(e),
                'job_id': self.job_id
            }
    
    def stop(self):
        """Stop the distributed processing system."""
        try:
            if self.spark:
                self.spark.stop()
            
            if self.chunk_producer:
                self.chunk_producer.close()
            
            if self.llm_consumer:
                self.llm_consumer.close()
            
            self.logger.info("Distributed processing system stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping distributed processing system: {e}")


class SparkTextStructurerFactory:
    """Factory for creating SparkTextStructurer instances."""
    
    @staticmethod
    def create_local_instance(config: Dict[str, Any] = None) -> SparkTextStructurer:
        """Create a local Spark instance for development."""
        from .spark_config import SparkEnvironments
        
        spark_conf = SparkEnvironments.local_config()
        spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
        
        return SparkTextStructurer(spark, config)
    
    @staticmethod
    def create_cluster_instance(config: Dict[str, Any] = None) -> SparkTextStructurer:
        """Create a cluster Spark instance for production."""
        from .spark_config import SparkEnvironments
        
        spark_conf = SparkEnvironments.cluster_config()
        spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
        
        return SparkTextStructurer(spark, config)
    
    @staticmethod
    def create_kubernetes_instance(config: Dict[str, Any] = None) -> SparkTextStructurer:
        """Create a Kubernetes Spark instance."""
        from .spark_config import SparkEnvironments
        
        spark_conf = SparkEnvironments.kubernetes_config()
        spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
        
        return SparkTextStructurer(spark, config)


# Context manager for SparkTextStructurer
class SparkTextStructurerContext:
    """Context manager for SparkTextStructurer lifecycle."""
    
    def __init__(self, environment: str = "local", config: Dict[str, Any] = None):
        self.environment = environment
        self.config = config
        self.structurer = None
    
    def __enter__(self) -> SparkTextStructurer:
        if self.environment == "local":
            self.structurer = SparkTextStructurerFactory.create_local_instance(self.config)
        elif self.environment == "cluster":
            self.structurer = SparkTextStructurerFactory.create_cluster_instance(self.config)
        elif self.environment == "kubernetes":
            self.structurer = SparkTextStructurerFactory.create_kubernetes_instance(self.config)
        else:
            raise ValueError(f"Unknown environment: {self.environment}")
        
        return self.structurer
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.structurer:
            self.structurer.stop()


# Utility functions
def create_spark_text_structurer(environment: str = "local", config: Dict[str, Any] = None) -> SparkTextStructurer:
    """Create a SparkTextStructurer instance."""
    if environment == "local":
        return SparkTextStructurerFactory.create_local_instance(config)
    elif environment == "cluster":
        return SparkTextStructurerFactory.create_cluster_instance(config)
    elif environment == "kubernetes":
        return SparkTextStructurerFactory.create_kubernetes_instance(config)
    else:
        raise ValueError(f"Unknown environment: {environment}")


def process_text_distributed(text_content: str, environment: str = "local", 
                            config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Process text using distributed processing."""
    with SparkTextStructurerContext(environment, config) as structurer:
        return structurer.structure_text(text_content)