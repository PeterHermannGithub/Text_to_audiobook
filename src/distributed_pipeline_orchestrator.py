"""
Distributed Pipeline Orchestrator for the text-to-audiobook system.

This orchestrator replaces the monolithic TextStructurer with a distributed
architecture that integrates Kafka, Spark, LLM Pool, Redis caching, and
comprehensive monitoring for enterprise-scale text processing.
"""

import time
import logging
import json
import threading
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import uuid

# Distributed processing imports
from .kafka.producers.file_upload_producer import FileUploadProducer
from .kafka.producers.chunk_producer import ChunkProducer
from .kafka.consumers.text_extraction_consumer import TextExtractionConsumer
from .kafka.consumers.llm_consumer import LLMConsumer
from .spark.distributed_validation import DistributedValidationEngine
from .spark.resource_optimizer import SparkResourceOptimizer
from .spark.spark_text_structurer import SparkTextStructurer
from .llm_pool.llm_pool_manager import get_pool_manager
from .cache.redis_cache import RedisCacheManager
from .monitoring.prometheus_metrics import get_metrics_collector
from .monitoring.health_checks import HealthCheckService

# Legacy processing imports (for fallback)
from .text_processing.preprocessor import TextPreprocessor
from .text_processing.segmentation.deterministic_segmenter import DeterministicSegmenter
from .text_processing.segmentation.chunking import ChunkManager
from .attribution.rule_based_attributor import RuleBasedAttributor
from .attribution.llm.orchestrator import LLMOrchestrator
from .validation.validator import SimplifiedValidator
from .refinement.contextual_refiner import ContextualRefiner
from .output.output_formatter import OutputFormatter

from config import settings


@dataclass
class DistributedProcessingConfig:
    """Configuration for distributed processing pipeline."""
    
    # Processing modes
    processing_mode: str = "distributed"  # "local", "distributed", "hybrid"
    enable_kafka: bool = True
    enable_spark: bool = True
    enable_caching: bool = True
    enable_monitoring: bool = True
    
    # Kafka configuration
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_consumer_group: str = "text-to-audiobook-processors"
    kafka_batch_size: int = 100
    kafka_timeout_ms: int = 30000
    
    # Spark configuration
    spark_master: str = "local[*]"
    spark_app_name: str = "text-to-audiobook-processor"
    spark_executor_memory: str = "2g"
    spark_executor_cores: int = 2
    spark_executor_instances: int = 2
    
    # LLM Pool configuration
    llm_pool_size: int = 3
    llm_pool_max_workers: int = 10
    llm_pool_timeout: int = 300
    
    # Cache configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    cache_ttl: int = 3600
    
    # Performance configuration
    chunk_size: int = 2000
    overlap_size: int = 400
    max_concurrent_chunks: int = 10
    quality_threshold: float = 0.85
    
    # Monitoring configuration
    metrics_enabled: bool = True
    health_check_interval: int = 30
    performance_monitoring: bool = True


@dataclass
class ProcessingResult:
    """Result of distributed processing operation."""
    
    job_id: str
    success: bool
    processed_segments: List[Dict[str, Any]]
    processing_time: float
    performance_metrics: Dict[str, Any]
    error_details: Optional[str] = None
    quality_report: Optional[Dict[str, Any]] = None
    cache_stats: Optional[Dict[str, Any]] = None


class DistributedPipelineOrchestrator:
    """
    Distributed pipeline orchestrator that coordinates all distributed components.
    
    This orchestrator replaces the monolithic TextStructurer and provides:
    - Kafka-based event-driven processing
    - Spark distributed validation and structuring
    - LLM pool for scalable inference
    - Redis caching for performance optimization
    - Comprehensive monitoring and health checks
    """
    
    def __init__(self, config: DistributedProcessingConfig = None, 
                 engine: str = None, local_model: str = None):
        """
        Initialize the distributed pipeline orchestrator.
        
        Args:
            config: Distributed processing configuration
            engine: LLM engine for fallback processing
            local_model: Local model for fallback processing
        """
        self.config = config or DistributedProcessingConfig()
        self.engine = engine or settings.DEFAULT_LLM_ENGINE
        self.local_model = local_model or settings.DEFAULT_LOCAL_MODEL
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.kafka_producer = None
        self.kafka_consumer = None
        self.spark_session = None
        self.validation_engine = None
        self.resource_optimizer = None
        self.llm_pool = None
        self.cache_manager = None
        self.metrics_collector = None
        self.health_service = None
        
        # Legacy components for fallback
        self.legacy_orchestrator = None
        self.legacy_segmenter = None
        self.legacy_attributor = None
        self.legacy_validator = None
        self.legacy_refiner = None
        
        # Processing state
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.processing_stats: Dict[str, Any] = {
            'total_jobs': 0,
            'successful_jobs': 0,
            'failed_jobs': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0
        }
        
        # Initialize the pipeline
        self._initialize_pipeline()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, settings.LOG_LEVEL),
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler(Path(settings.LOG_DIR) / 'distributed_pipeline.log'),
                logging.StreamHandler()
            ]
        )
    
    def _initialize_pipeline(self):
        """Initialize all distributed pipeline components."""
        try:
            self.logger.info("Initializing distributed pipeline orchestrator...")
            
            # Initialize monitoring first
            if self.config.enable_monitoring:
                self._initialize_monitoring()
            
            # Initialize caching
            if self.config.enable_caching:
                self._initialize_caching()
            
            # Initialize Kafka components
            if self.config.enable_kafka:
                self._initialize_kafka()
            
            # Initialize Spark components
            if self.config.enable_spark:
                self._initialize_spark()
            
            # Initialize LLM pool
            self._initialize_llm_pool()
            
            # Initialize legacy components for fallback
            self._initialize_legacy_components()
            
            # Perform health checks
            self._perform_initial_health_checks()
            
            self.logger.info("Distributed pipeline orchestrator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize distributed pipeline: {e}", exc_info=True)
            self.logger.info("Falling back to legacy processing mode")
            self.config.processing_mode = "local"
            self._initialize_legacy_components()
    
    def _initialize_monitoring(self):
        """Initialize monitoring and metrics collection."""
        try:
            self.metrics_collector = get_metrics_collector()
            self.health_service = HealthCheckService()
            
            if self.config.performance_monitoring:
                self.health_service.start_monitoring()
            
            self.logger.info("Monitoring and metrics collection initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize monitoring: {e}")
            self.config.enable_monitoring = False
    
    def _initialize_caching(self):
        """Initialize Redis caching layer."""
        try:
            self.cache_manager = RedisCacheManager(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                namespace='distributed_pipeline'
            )
            
            # Test cache connection
            if not self.cache_manager.ping():
                raise Exception("Cache connection test failed")
            
            self.logger.info("Redis caching layer initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize caching: {e}")
            self.config.enable_caching = False
            self.cache_manager = None
    
    def _initialize_kafka(self):
        """Initialize Kafka producers and consumers."""
        try:
            # Initialize producers
            self.file_upload_producer = FileUploadProducer(
                bootstrap_servers=self.config.kafka_bootstrap_servers
            )
            self.chunk_producer = ChunkProducer(
                bootstrap_servers=self.config.kafka_bootstrap_servers
            )
            
            # Initialize consumers
            self.text_extraction_consumer = TextExtractionConsumer(
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                consumer_group=self.config.kafka_consumer_group
            )
            self.llm_consumer = LLMConsumer(
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                consumer_group=f"{self.config.kafka_consumer_group}_llm"
            )
            
            self.logger.info("Kafka producers and consumers initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize Kafka: {e}")
            self.config.enable_kafka = False
    
    def _initialize_spark(self):
        """Initialize Spark components."""
        try:
            # Initialize Spark session through the text structurer
            self.spark_structurer = SparkTextStructurer(
                master=self.config.spark_master,
                app_name=self.config.spark_app_name
            )
            
            # Get the Spark session
            self.spark_session = self.spark_structurer.spark_session
            
            # Initialize validation engine
            self.validation_engine = DistributedValidationEngine(self.spark_session)
            
            # Initialize resource optimizer
            self.resource_optimizer = SparkResourceOptimizer(self.spark_session)
            
            # Optimize resources based on configuration
            self._optimize_spark_resources()
            
            self.logger.info("Spark components initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize Spark: {e}")
            self.config.enable_spark = False
    
    def _initialize_llm_pool(self):
        """Initialize LLM pool for distributed inference."""
        try:
            self.llm_pool = get_pool_manager(
                pool_size=self.config.llm_pool_size,
                max_workers=self.config.llm_pool_max_workers,
                timeout=self.config.llm_pool_timeout
            )
            
            self.logger.info("LLM pool initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM pool: {e}")
            self.llm_pool = None
    
    def _initialize_legacy_components(self):
        """Initialize legacy components for fallback processing."""
        try:
            # Initialize legacy LLM orchestrator
            self.legacy_orchestrator = LLMOrchestrator({
                'engine': self.engine,
                'local_model': self.local_model
            })
            
            # Initialize legacy processing components
            self.legacy_segmenter = DeterministicSegmenter()
            self.legacy_attributor = RuleBasedAttributor()
            self.legacy_validator = SimplifiedValidator()
            self.legacy_refiner = ContextualRefiner(self.legacy_orchestrator)
            
            self.logger.info("Legacy components initialized for fallback")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize legacy components: {e}")
            raise
    
    def _optimize_spark_resources(self):
        """Optimize Spark resources based on workload."""
        if self.resource_optimizer:
            try:
                # Analyze default workload
                workload = self.resource_optimizer.analyze_workload(
                    data_size_mb=100,  # Default assumption
                    processing_type='text_structuring'
                )
                
                # Optimize allocation
                allocation = self.resource_optimizer.optimize_allocation(
                    workload, strategy='balanced'
                )
                
                # Apply optimization
                self.resource_optimizer.apply_allocation(allocation)
                
                self.logger.info("Spark resources optimized")
                
            except Exception as e:
                self.logger.warning(f"Failed to optimize Spark resources: {e}")
    
    def _perform_initial_health_checks(self):
        """Perform initial health checks on all components."""
        if self.health_service:
            try:
                health_status = self.health_service.get_overall_health()
                
                if health_status['status'] == 'healthy':
                    self.logger.info("All components passed initial health checks")
                else:
                    self.logger.warning(f"Some components failed health checks: {health_status}")
                
            except Exception as e:
                self.logger.warning(f"Failed to perform health checks: {e}")
    
    def process_text(self, text: str, job_id: str = None) -> ProcessingResult:
        """
        Process text through the distributed pipeline.
        
        Args:
            text: Raw text to process
            job_id: Optional job identifier
            
        Returns:
            ProcessingResult with processed segments and metadata
        """
        job_id = job_id or f"job_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting distributed processing for job {job_id}")
            
            # Record job start
            self.active_jobs[job_id] = {
                'start_time': start_time,
                'status': 'processing',
                'text_length': len(text)
            }
            
            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.record_processing_request('text_processing', 'started')
            
            # Choose processing mode
            if self.config.processing_mode == "distributed":
                result = self._process_distributed(text, job_id)
            elif self.config.processing_mode == "hybrid":
                result = self._process_hybrid(text, job_id)
            else:
                result = self._process_local(text, job_id)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            # Update statistics
            self._update_processing_stats(result, processing_time)
            
            # Record completion metrics
            if self.metrics_collector:
                self.metrics_collector.record_processing_duration(
                    'text_processing', 'orchestrator', processing_time
                )
                
                if result.success:
                    self.metrics_collector.record_processing_request('text_processing', 'completed')
                else:
                    self.metrics_collector.record_processing_request('text_processing', 'failed')
            
            # Clean up job tracking
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            
            self.logger.info(f"Completed processing for job {job_id} in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process text for job {job_id}: {e}", exc_info=True)
            
            # Clean up job tracking
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            
            return ProcessingResult(
                job_id=job_id,
                success=False,
                processed_segments=[],
                processing_time=time.time() - start_time,
                performance_metrics={},
                error_details=str(e)
            )
    
    def _process_distributed(self, text: str, job_id: str) -> ProcessingResult:
        """Process text using full distributed pipeline."""
        try:
            # Step 1: Send to Kafka for distributed processing
            if self.config.enable_kafka:
                self._send_to_kafka(text, job_id)
            
            # Step 2: Process chunks through Spark
            if self.config.enable_spark:
                segments = self._process_with_spark(text, job_id)
            else:
                segments = self._process_with_legacy_segmentation(text)
            
            # Step 3: LLM processing for ambiguous segments
            if self.llm_pool:
                segments = self._process_with_llm_pool(segments, job_id)
            else:
                segments = self._process_with_legacy_llm(segments)
            
            # Step 4: Validation and quality checks
            if self.validation_engine:
                # Transform segments to Spark validation format
                validation_segments = self._transform_segments_for_spark_validation(segments)
                validation_results = self.validation_engine.validate_text_segments(validation_segments)
                
                # Extract speaker data for speaker validation
                speaker_data = self._extract_speaker_data_for_validation(segments)
                speaker_results = self.validation_engine.validate_speaker_consistency(speaker_data)
                
                # Generate comprehensive quality report
                quality_report = self.validation_engine.generate_quality_report(
                    validation_results, speaker_results
                )
                
                # Apply validation improvements back to segments
                segments = self._apply_validation_improvements(segments, validation_results)
                
                self.logger.info(f"Spark validation completed: {len(validation_results)} segments validated")
            else:
                quality_report = self._generate_legacy_quality_report(segments)
            
            # Step 5: Cache results if enabled
            cache_stats = None
            if self.cache_manager:
                cache_stats = self._cache_results(job_id, segments)
            
            return ProcessingResult(
                job_id=job_id,
                success=True,
                processed_segments=segments,
                processing_time=0.0,  # Will be set by caller
                performance_metrics=self._gather_performance_metrics(job_id),
                quality_report=quality_report,
                cache_stats=cache_stats
            )
            
        except Exception as e:
            self.logger.error(f"Distributed processing failed for job {job_id}: {e}")
            # Fallback to local processing
            return self._process_local(text, job_id)
    
    def _process_hybrid(self, text: str, job_id: str) -> ProcessingResult:
        """Process text using hybrid approach (some distributed, some local)."""
        try:
            # Use distributed validation but local segmentation
            segments = self._process_with_legacy_segmentation(text)
            
            # Use Spark for validation if available
            if self.validation_engine:
                validation_results = self.validation_engine.validate_text_segments(segments)
                quality_report = self.validation_engine.generate_quality_report(
                    validation_results, []
                )
            else:
                quality_report = self._generate_legacy_quality_report(segments)
            
            # Use LLM pool for inference if available
            if self.llm_pool:
                segments = self._process_with_llm_pool(segments, job_id)
            else:
                segments = self._process_with_legacy_llm(segments)
            
            return ProcessingResult(
                job_id=job_id,
                success=True,
                processed_segments=segments,
                processing_time=0.0,
                performance_metrics=self._gather_performance_metrics(job_id),
                quality_report=quality_report
            )
            
        except Exception as e:
            self.logger.error(f"Hybrid processing failed for job {job_id}: {e}")
            return self._process_local(text, job_id)
    
    def _process_local(self, text: str, job_id: str) -> ProcessingResult:
        """Process text using legacy local processing."""
        try:
            # Use legacy segmentation
            segments = self._process_with_legacy_segmentation(text)
            
            # Use legacy LLM processing
            segments = self._process_with_legacy_llm(segments)
            
            # Generate basic quality report
            quality_report = self._generate_legacy_quality_report(segments)
            
            return ProcessingResult(
                job_id=job_id,
                success=True,
                processed_segments=segments,
                processing_time=0.0,
                performance_metrics={'mode': 'local'},
                quality_report=quality_report
            )
            
        except Exception as e:
            self.logger.error(f"Local processing failed for job {job_id}: {e}")
            raise
    
    def _send_to_kafka(self, text: str, job_id: str):
        """Send text to Kafka for distributed processing."""
        if self.file_upload_producer:
            message = {
                'job_id': job_id,
                'text_content': text,
                'timestamp': datetime.now().isoformat(),
                'processing_config': {
                    'chunk_size': self.config.chunk_size,
                    'overlap_size': self.config.overlap_size,
                    'quality_threshold': self.config.quality_threshold
                }
            }
            
            self.file_upload_producer.send_file_upload(message)
            
            if self.metrics_collector:
                self.metrics_collector.record_kafka_message('file_upload', 'sent')
    
    def _process_with_spark(self, text: str, job_id: str) -> List[Dict[str, Any]]:
        """Process text using Spark distributed processing."""
        # Check cache for existing Spark processing results
        cache_key = f"spark_processing_{job_id}"
        cached_result = self._get_cached_result(cache_key)
        
        if cached_result:
            self.logger.info(f"Using cached Spark processing result for job {job_id}")
            return cached_result
        
        # Process with Spark
        if self.spark_structurer:
            segments = self.spark_structurer.structure_text(text)
        else:
            segments = self._process_with_legacy_segmentation(text)
        
        # Cache the result
        self._cache_intermediate_result(cache_key, segments, ttl=1800)  # 30 minutes
        
        return segments
    
    def _process_with_legacy_segmentation(self, text: str) -> List[Dict[str, Any]]:
        """Process text using legacy segmentation approach."""
        # Check cache for existing segmentation results
        text_hash = hash(text)
        cache_key = f"legacy_segmentation_{text_hash}"
        cached_result = self._get_cached_result(cache_key)
        
        if cached_result:
            self.logger.info(f"Using cached legacy segmentation result")
            return cached_result
        
        # Preprocess text
        preprocessor = TextPreprocessor()
        processed_text = preprocessor.preprocess_text(text)
        
        # Segment text
        segments = self.legacy_segmenter.segment_text(processed_text)
        
        # Apply rule-based attribution
        segments = self.legacy_attributor.attribute_speakers(segments)
        
        # Cache the result
        self._cache_intermediate_result(cache_key, segments, ttl=3600)  # 1 hour
        
        return segments
    
    def _process_with_llm_pool(self, segments: List[Dict[str, Any]], job_id: str) -> List[Dict[str, Any]]:
        """Process segments using LLM pool for ambiguous speakers."""
        if not self.llm_pool:
            return segments
        
        # Find ambiguous segments
        ambiguous_segments = [
            seg for seg in segments 
            if seg.get('speaker') == 'AMBIGUOUS' or seg.get('confidence', 1.0) < 0.8
        ]
        
        if not ambiguous_segments:
            return segments
        
        # Process ambiguous segments with LLM pool using caching
        processed_segments = []
        for segment in ambiguous_segments:
            try:
                # Check cache for this specific segment
                segment_text = segment.get('text_content', '')
                segment_hash = hash(segment_text)
                cache_key = f"llm_pool_segment_{segment_hash}"
                cached_result = self._get_cached_result(cache_key)
                
                if cached_result:
                    self.logger.debug(f"Using cached LLM pool result for segment {segment.get('segment_id')}")
                    segment.update(cached_result)
                    processed_segments.append(segment)
                else:
                    # Process with LLM pool
                    result = self.llm_pool.process_segment(segment)
                    segment.update(result)
                    processed_segments.append(segment)
                    
                    # Cache the LLM result
                    self._cache_intermediate_result(cache_key, result, ttl=7200)  # 2 hours
                    
                    if self.metrics_collector:
                        self.metrics_collector.record_llm_request('speaker_attribution', 'completed')
                    
            except Exception as e:
                self.logger.warning(f"LLM pool processing failed for segment {segment.get('segment_id')}: {e}")
                processed_segments.append(segment)  # Keep original
        
        # Update segments list
        segment_dict = {seg['segment_id']: seg for seg in processed_segments}
        
        return [
            segment_dict.get(seg['segment_id'], seg)
            for seg in segments
        ]
    
    def _process_with_legacy_llm(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process segments using legacy LLM orchestrator."""
        if not self.legacy_orchestrator:
            return segments
        
        # Check cache for legacy LLM processing results
        segments_hash = hash(json.dumps(segments, sort_keys=True))
        cache_key = f"legacy_llm_processing_{segments_hash}"
        cached_result = self._get_cached_result(cache_key)
        
        if cached_result:
            self.logger.info(f"Using cached legacy LLM processing result")
            return cached_result
        
        # Use contextual refiner for ambiguous segments
        processed_segments = self.legacy_refiner.refine_segments(segments)
        
        # Cache the result
        self._cache_intermediate_result(cache_key, processed_segments, ttl=3600)  # 1 hour
        
        return processed_segments
    
    def _generate_legacy_quality_report(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a basic quality report using legacy validation."""
        if self.legacy_validator:
            return self.legacy_validator.validate_segments(segments)
        else:
            return {
                'total_segments': len(segments),
                'quality_score': 0.85,
                'issues': []
            }
    
    def _cache_results(self, job_id: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cache final processing results."""
        if not self.cache_manager:
            return {}
        
        try:
            # Cache the complete result using the new helper method
            self._cache_intermediate_result(
                f"processing_results_{job_id}", 
                segments, 
                ttl=self.config.cache_ttl
            )
            
            # Cache individual segments for potential reuse
            for segment in segments:
                segment_key = f"segment_{segment.get('segment_id')}"
                self._cache_intermediate_result(
                    f"segments_{segment_key}",
                    segment,
                    ttl=self.config.cache_ttl
                )
            
            # Cache processing metadata
            metadata = {
                'job_id': job_id,
                'segment_count': len(segments),
                'cached_at': datetime.now().isoformat(),
                'processing_mode': self.config.processing_mode
            }
            
            self._cache_intermediate_result(
                f"metadata_{job_id}",
                metadata,
                ttl=self.config.cache_ttl
            )
            
            return self._get_cache_statistics()
            
        except Exception as e:
            self.logger.warning(f"Failed to cache results for job {job_id}: {e}")
            return {}
    
    def _gather_performance_metrics(self, job_id: str) -> Dict[str, Any]:
        """Gather performance metrics for the job."""
        metrics = {
            'job_id': job_id,
            'processing_mode': self.config.processing_mode,
            'components_enabled': {
                'kafka': self.config.enable_kafka,
                'spark': self.config.enable_spark,
                'caching': self.config.enable_caching,
                'monitoring': self.config.enable_monitoring
            }
        }
        
        # Add enhanced cache statistics
        metrics['cache_stats'] = self._get_cache_statistics()
        
        # Add LLM pool stats if available
        if self.llm_pool:
            metrics['llm_pool_stats'] = self.llm_pool.get_pool_stats()
        
        # Add distributed processing specific metrics
        metrics['distributed_processing'] = {
            'spark_enabled': self.config.enable_spark,
            'kafka_enabled': self.config.enable_kafka,
            'monitoring_enabled': self.config.enable_monitoring,
            'processing_mode': self.config.processing_mode,
            'cache_ttl_config': {
                'spark_results': 1800,  # 30 minutes
                'llm_results': 7200,    # 2 hours
                'segmentation_results': 3600,  # 1 hour
                'legacy_llm_results': 3600     # 1 hour
            }
        }
        
        return metrics
    
    def _transform_segments_for_spark_validation(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform segments to the format expected by Spark validation engine."""
        try:
            validation_segments = []
            
            for i, segment in enumerate(segments):
                # Extract segment data
                segment_id = segment.get('segment_id', f'segment_{i}')
                text_content = segment.get('text_content', '')
                speaker_id = segment.get('speaker', 'unknown')
                
                # Calculate quality score if not present
                quality_score = segment.get('quality_score', self._calculate_segment_quality_score(segment))
                
                # Calculate confidence score
                confidence_score = segment.get('confidence', self._calculate_segment_confidence(segment))
                
                # Determine segment type
                segment_type = self._determine_segment_type(segment)
                
                # Create processing metadata
                processing_metadata = json.dumps({
                    'processing_mode': self.config.processing_mode,
                    'timestamp': datetime.now().isoformat(),
                    'original_data': {
                        'has_dialogue_markers': any(char in text_content for char in ['"', "'", '«', '»']),
                        'word_count': len(text_content.split()),
                        'character_count': len(text_content)
                    }
                })
                
                validation_segment = {
                    'segment_id': segment_id,
                    'text_content': text_content,
                    'speaker_id': speaker_id,
                    'quality_score': float(quality_score),
                    'confidence_score': float(confidence_score),
                    'segment_type': segment_type,
                    'processing_metadata': processing_metadata
                }
                
                validation_segments.append(validation_segment)
            
            self.logger.debug(f"Transformed {len(segments)} segments for Spark validation")
            return validation_segments
            
        except Exception as e:
            self.logger.error(f"Error transforming segments for Spark validation: {e}")
            # Return minimal format on error
            return [{
                'segment_id': f'segment_{i}',
                'text_content': seg.get('text_content', ''),
                'speaker_id': seg.get('speaker', 'unknown'),
                'quality_score': 0.5,
                'confidence_score': 0.5,
                'segment_type': 'unknown',
                'processing_metadata': '{}'
            } for i, seg in enumerate(segments)]
    
    def _extract_speaker_data_for_validation(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract speaker data from segments for speaker validation."""
        try:
            # Group segments by speaker
            speaker_groups = {}
            for segment in segments:
                speaker_id = segment.get('speaker', 'unknown')
                if speaker_id not in speaker_groups:
                    speaker_groups[speaker_id] = []
                speaker_groups[speaker_id].append(segment)
            
            # Create speaker data for validation
            speaker_data = []
            for speaker_id, speaker_segments in speaker_groups.items():
                if speaker_id in ['unknown', 'AMBIGUOUS', 'narrator']:
                    continue  # Skip meta-speakers
                
                # Extract confidence scores
                confidence_scores = []
                dialogue_count = 0
                
                for segment in speaker_segments:
                    # Count dialogue segments
                    if self._is_dialogue_segment(segment):
                        dialogue_count += 1
                    
                    # Collect confidence scores
                    confidence = segment.get('confidence', 0.5)
                    confidence_scores.append(confidence)
                
                # Create speaker data entry
                speaker_entry = {
                    'speaker_id': speaker_id,
                    'name': speaker_id,
                    'voice_characteristics': json.dumps({
                        'segments_count': len(speaker_segments),
                        'avg_segment_length': sum(len(seg.get('text_content', '')) for seg in speaker_segments) / len(speaker_segments),
                        'total_words': sum(len(seg.get('text_content', '').split()) for seg in speaker_segments)
                    }),
                    'dialogue_segments': dialogue_count,
                    'confidence_scores': json.dumps(confidence_scores)
                }
                
                speaker_data.append(speaker_entry)
            
            self.logger.debug(f"Extracted data for {len(speaker_data)} speakers for validation")
            return speaker_data
            
        except Exception as e:
            self.logger.error(f"Error extracting speaker data for validation: {e}")
            return []
    
    def _apply_validation_improvements(self, segments: List[Dict[str, Any]], 
                                     validation_results: List) -> List[Dict[str, Any]]:
        """Apply validation improvements back to the original segments."""
        try:
            # Create a mapping of segment_id to validation result
            validation_map = {}
            for result in validation_results:
                validation_map[result.segment_id] = result
            
            # Apply improvements to segments
            improved_segments = []
            for i, segment in enumerate(segments):
                segment_id = segment.get('segment_id', f'segment_{i}')
                validation_result = validation_map.get(segment_id)
                
                if validation_result:
                    # Apply quality improvements
                    segment['quality_score'] = validation_result.refined_quality_score
                    segment['validation_applied'] = validation_result.refinement_applied
                    segment['validation_issues'] = validation_result.validation_issues
                    segment['validation_processing_time'] = validation_result.processing_time
                    
                    # Update segment metadata
                    segment['processing_metadata'] = segment.get('processing_metadata', {})
                    segment['processing_metadata']['spark_validation'] = {
                        'original_quality': validation_result.original_quality_score,
                        'refined_quality': validation_result.refined_quality_score,
                        'improvement': validation_result.refined_quality_score - validation_result.original_quality_score,
                        'issues_found': len(validation_result.validation_issues),
                        'refinement_applied': validation_result.refinement_applied
                    }
                    
                    self.logger.debug(f"Applied validation improvements to segment {segment_id}")
                
                improved_segments.append(segment)
            
            self.logger.info(f"Applied validation improvements to {len(improved_segments)} segments")
            return improved_segments
            
        except Exception as e:
            self.logger.error(f"Error applying validation improvements: {e}")
            return segments  # Return original segments on error
    
    def _calculate_segment_quality_score(self, segment: Dict[str, Any]) -> float:
        """Calculate a basic quality score for a segment."""
        try:
            text_content = segment.get('text_content', '')
            speaker = segment.get('speaker', 'unknown')
            
            # Base quality score
            quality = 0.7
            
            # Adjust for text length
            text_length = len(text_content.strip())
            if text_length < 10:
                quality *= 0.5
            elif text_length > 500:
                quality *= 0.9
            elif text_length > 100:
                quality *= 1.1
            
            # Adjust for speaker confidence
            if speaker == 'AMBIGUOUS':
                quality *= 0.6
            elif speaker == 'narrator':
                quality *= 1.0
            elif speaker != 'unknown':
                quality *= 1.1
            
            # Adjust for dialogue markers
            if any(char in text_content for char in ['"', "'", '«', '»']):
                quality *= 1.05
            
            return min(1.0, quality)
            
        except Exception:
            return 0.5
    
    def _calculate_segment_confidence(self, segment: Dict[str, Any]) -> float:
        """Calculate confidence score for a segment."""
        return segment.get('confidence', 0.8)
    
    def _determine_segment_type(self, segment: Dict[str, Any]) -> str:
        """Determine the type of a segment (dialogue, narrative, etc.)."""
        try:
            text_content = segment.get('text_content', '')
            speaker = segment.get('speaker', 'unknown')
            
            # Check if it's narrative
            if speaker == 'narrator':
                return 'narrative'
            
            # Check if it has dialogue markers
            if any(char in text_content for char in ['"', "'", '«', '»']):
                return 'dialogue'
            
            # Check if it's ambiguous
            if speaker == 'AMBIGUOUS':
                return 'ambiguous'
            
            # Default to mixed for other cases
            return 'mixed'
            
        except Exception:
            return 'unknown'
    
    def _is_dialogue_segment(self, segment: Dict[str, Any]) -> bool:
        """Check if a segment is a dialogue segment."""
        try:
            text_content = segment.get('text_content', '')
            speaker = segment.get('speaker', 'unknown')
            
            # Check if it has dialogue markers or is attributed to a character
            return (
                any(char in text_content for char in ['"', "'", '«', '»']) or
                (speaker != 'narrator' and speaker != 'AMBIGUOUS' and speaker != 'unknown')
            )
            
        except Exception:
            return False
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result from Redis cache."""
        if not self.cache_manager:
            return None
        
        try:
            cached_data = self.cache_manager.get('intermediate_results', cache_key)
            if cached_data:
                if self.metrics_collector:
                    self.metrics_collector.record_cache_hit('intermediate_results')
                return cached_data
            else:
                if self.metrics_collector:
                    self.metrics_collector.record_cache_miss('intermediate_results')
                return None
                
        except Exception as e:
            self.logger.warning(f"Error retrieving cached result for key {cache_key}: {e}")
            return None
    
    def _cache_intermediate_result(self, cache_key: str, result: Any, ttl: int = 3600):
        """Cache intermediate result in Redis."""
        if not self.cache_manager:
            return
        
        try:
            success = self.cache_manager.set(
                'intermediate_results', cache_key, result, ttl=ttl
            )
            
            if success:
                if self.metrics_collector:
                    self.metrics_collector.record_cache_set('intermediate_results')
                self.logger.debug(f"Cached intermediate result with key {cache_key}")
            else:
                self.logger.warning(f"Failed to cache intermediate result with key {cache_key}")
                
        except Exception as e:
            self.logger.warning(f"Error caching intermediate result for key {cache_key}: {e}")
    
    def _invalidate_cache_for_job(self, job_id: str):
        """Invalidate all cache entries related to a specific job."""
        if not self.cache_manager:
            return
        
        try:
            # Define cache key patterns to invalidate
            cache_patterns = [
                f"spark_processing_{job_id}",
                f"llm_pool_job_{job_id}",
                f"validation_results_{job_id}",
                f"processing_results_{job_id}"
            ]
            
            for pattern in cache_patterns:
                self.cache_manager.delete('intermediate_results', pattern)
                
            self.logger.debug(f"Invalidated cache entries for job {job_id}")
            
        except Exception as e:
            self.logger.warning(f"Error invalidating cache for job {job_id}: {e}")
    
    def _get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        if not self.cache_manager:
            return {'cache_enabled': False}
        
        try:
            cache_stats = self.cache_manager.get_stats()
            
            # Add distributed pipeline specific stats
            pipeline_stats = {
                'cache_enabled': True,
                'cache_namespace': 'distributed_pipeline',
                'intermediate_results_cached': cache_stats.get('keys_count', 0),
                'cache_memory_usage': cache_stats.get('memory_usage', 0),
                'cache_hit_rate': cache_stats.get('hit_rate', 0.0),
                'cache_performance': {
                    'avg_get_time': cache_stats.get('avg_get_time', 0.0),
                    'avg_set_time': cache_stats.get('avg_set_time', 0.0),
                    'total_operations': cache_stats.get('total_operations', 0)
                }
            }
            
            return pipeline_stats
            
        except Exception as e:
            self.logger.warning(f"Error getting cache statistics: {e}")
            return {'cache_enabled': True, 'error': str(e)}
    
    def _update_processing_stats(self, result: ProcessingResult, processing_time: float):
        """Update overall processing statistics."""
        self.processing_stats['total_jobs'] += 1
        
        if result.success:
            self.processing_stats['successful_jobs'] += 1
        else:
            self.processing_stats['failed_jobs'] += 1
        
        self.processing_stats['total_processing_time'] += processing_time
        self.processing_stats['avg_processing_time'] = (
            self.processing_stats['total_processing_time'] / 
            self.processing_stats['total_jobs']
        )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return self.processing_stats.copy()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of all components."""
        if self.health_service:
            return self.health_service.get_overall_health()
        else:
            return {'status': 'unknown', 'message': 'Health service not available'}
    
    def shutdown(self):
        """Gracefully shutdown the distributed pipeline."""
        try:
            self.logger.info("Shutting down distributed pipeline orchestrator...")
            
            # Stop monitoring
            if self.health_service:
                self.health_service.stop_monitoring()
            
            # Close Kafka connections
            if self.file_upload_producer:
                self.file_upload_producer.close()
            if self.text_extraction_consumer:
                self.text_extraction_consumer.close()
            
            # Stop Spark session
            if self.spark_session:
                self.spark_session.stop()
            
            # Shutdown LLM pool
            if self.llm_pool:
                self.llm_pool.shutdown()
            
            # Close cache connections
            if self.cache_manager:
                self.cache_manager.close()
            
            self.logger.info("Distributed pipeline orchestrator shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")