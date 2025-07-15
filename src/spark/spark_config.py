"""
Spark configuration for distributed text processing.

This module provides comprehensive Spark configuration for the text-to-audiobook
distributed processing system, including settings for performance optimization,
resource management, and LLM pool integration.
"""

import os
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from config import settings

class SparkConfig:
    """Centralized Spark configuration management."""
    
    # Core Spark Configuration
    SPARK_APP_NAME = "TextToAudiobookProcessor"
    SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")  # Default to local mode
    
    # Memory and Resource Configuration
    SPARK_EXECUTOR_MEMORY = "4g"
    SPARK_EXECUTOR_CORES = 2
    SPARK_DRIVER_MEMORY = "2g"
    SPARK_DRIVER_MAX_RESULT_SIZE = "2g"
    
    # Performance Optimization
    SPARK_ADAPTIVE_ENABLED = True
    SPARK_ADAPTIVE_COALESCE_PARTITIONS = True
    SPARK_ADAPTIVE_SKEW_JOIN_ENABLED = True
    SPARK_DYNAMIC_ALLOCATION_ENABLED = True
    
    # LLM Pool Configuration
    LLM_POOL_SIZE = 4  # Number of LLM instances per worker
    LLM_REQUEST_TIMEOUT = 30
    LLM_RETRY_ATTEMPTS = 3
    LLM_BATCH_SIZE = 10  # Number of texts to process in batch
    
    # Distributed Processing Settings
    DEFAULT_PARALLELISM = 8
    MAX_PARTITION_BYTES = "134217728"  # 128MB
    BROADCAST_THRESHOLD = "10485760"   # 10MB
    
    # Checkpoint and Persistence
    CHECKPOINT_DIR = "/tmp/spark-checkpoints"
    KRYO_SERIALIZER_ENABLED = True
    
    @classmethod
    def create_spark_conf(cls) -> SparkConf:
        """Create optimized Spark configuration."""
        conf = SparkConf()
        
        # Basic configuration
        conf.setAppName(cls.SPARK_APP_NAME)
        conf.setMaster(cls.SPARK_MASTER)
        
        # Memory configuration
        conf.set("spark.executor.memory", cls.SPARK_EXECUTOR_MEMORY)
        conf.set("spark.executor.cores", str(cls.SPARK_EXECUTOR_CORES))
        conf.set("spark.driver.memory", cls.SPARK_DRIVER_MEMORY)
        conf.set("spark.driver.maxResultSize", cls.SPARK_DRIVER_MAX_RESULT_SIZE)
        
        # Performance optimization
        conf.set("spark.sql.adaptive.enabled", str(cls.SPARK_ADAPTIVE_ENABLED))
        conf.set("spark.sql.adaptive.coalescePartitions.enabled", str(cls.SPARK_ADAPTIVE_COALESCE_PARTITIONS))
        conf.set("spark.sql.adaptive.skewJoin.enabled", str(cls.SPARK_ADAPTIVE_SKEW_JOIN_ENABLED))
        conf.set("spark.dynamicAllocation.enabled", str(cls.SPARK_DYNAMIC_ALLOCATION_ENABLED))
        
        # Parallelism settings
        conf.set("spark.default.parallelism", str(cls.DEFAULT_PARALLELISM))
        conf.set("spark.sql.files.maxPartitionBytes", cls.MAX_PARTITION_BYTES)
        conf.set("spark.sql.autoBroadcastJoinThreshold", cls.BROADCAST_THRESHOLD)
        
        # Serialization
        if cls.KRYO_SERIALIZER_ENABLED:
            conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            conf.set("spark.kryo.registrationRequired", "false")
        
        # Checkpoint directory
        conf.set("spark.sql.streaming.checkpointLocation", cls.CHECKPOINT_DIR)
        
        # Custom configurations for text processing
        conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB")
        
        return conf
    
    @classmethod
    def create_spark_session(cls) -> SparkSession:
        """Create optimized Spark session for text processing."""
        conf = cls.create_spark_conf()
        
        # Create Spark session with configuration
        spark = SparkSession.builder \
            .config(conf=conf) \
            .getOrCreate()
        
        # Set log level to reduce verbosity
        spark.sparkContext.setLogLevel("WARN")
        
        # Configure checkpoint directory
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        spark.sparkContext.setCheckpointDir(cls.CHECKPOINT_DIR)
        
        return spark
    
    @classmethod
    def get_distributed_config(cls) -> dict:
        """Get configuration for distributed processing."""
        return {
            'llm_pool_size': cls.LLM_POOL_SIZE,
            'llm_request_timeout': cls.LLM_REQUEST_TIMEOUT,
            'llm_retry_attempts': cls.LLM_RETRY_ATTEMPTS,
            'llm_batch_size': cls.LLM_BATCH_SIZE,
            'parallelism': cls.DEFAULT_PARALLELISM,
            'chunk_size': settings.CHUNK_SIZE,
            'overlap_size': settings.OVERLAP_SIZE
        }
    
    @classmethod
    def get_performance_config(cls) -> dict:
        """Get performance-related configuration."""
        return {
            'adaptive_enabled': cls.SPARK_ADAPTIVE_ENABLED,
            'coalesce_partitions': cls.SPARK_ADAPTIVE_COALESCE_PARTITIONS,
            'skew_join_enabled': cls.SPARK_ADAPTIVE_SKEW_JOIN_ENABLED,
            'dynamic_allocation': cls.SPARK_DYNAMIC_ALLOCATION_ENABLED,
            'max_partition_bytes': cls.MAX_PARTITION_BYTES,
            'broadcast_threshold': cls.BROADCAST_THRESHOLD
        }


# Environment-specific configurations
class SparkEnvironments:
    """Different Spark configurations for different environments."""
    
    @staticmethod
    def local_config() -> SparkConf:
        """Configuration for local development."""
        conf = SparkConfig.create_spark_conf()
        conf.setMaster("local[*]")
        conf.set("spark.executor.memory", "2g")
        conf.set("spark.driver.memory", "1g")
        return conf
    
    @staticmethod
    def cluster_config() -> SparkConf:
        """Configuration for cluster deployment."""
        conf = SparkConfig.create_spark_conf()
        conf.setMaster("yarn")  # or "k8s://https://kubernetes.default.svc:443"
        conf.set("spark.executor.instances", "4")
        conf.set("spark.executor.memory", "4g")
        conf.set("spark.executor.cores", "2")
        conf.set("spark.driver.memory", "2g")
        return conf
    
    @staticmethod
    def kubernetes_config() -> SparkConf:
        """Configuration for Kubernetes deployment."""
        conf = SparkConfig.create_spark_conf()
        conf.setMaster("k8s://https://kubernetes.default.svc:443")
        conf.set("spark.kubernetes.container.image", "text-to-audiobook-spark:latest")
        conf.set("spark.kubernetes.authenticate.driver.serviceAccountName", "spark-service-account")
        conf.set("spark.executor.instances", "4")
        conf.set("spark.kubernetes.executor.request.cores", "1")
        conf.set("spark.kubernetes.executor.limit.cores", "2")
        return conf


# Utility functions for Spark session management
def get_spark_session(environment: str = "local") -> SparkSession:
    """Get Spark session based on environment."""
    if environment == "local":
        conf = SparkEnvironments.local_config()
    elif environment == "cluster":
        conf = SparkEnvironments.cluster_config()
    elif environment == "kubernetes":
        conf = SparkEnvironments.kubernetes_config()
    else:
        conf = SparkConfig.create_spark_conf()
    
    return SparkSession.builder.config(conf=conf).getOrCreate()


def stop_spark_session(spark: SparkSession = None):
    """Stop Spark session safely."""
    if spark is None:
        spark = SparkSession.getActiveSession()
    
    if spark:
        spark.stop()