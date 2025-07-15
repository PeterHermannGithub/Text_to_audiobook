"""
Custom Airflow operators for text-to-audiobook processing.

This package contains custom operators for the distributed text-to-audiobook
processing pipeline including Spark, Kafka, LLM, and quality validation operations.
"""

from .spark_text_structurer_operator import (
    SparkTextStructurerOperator,
    SparkTextStructurerSubmitOperator,
    DistributedTextProcessingOperator,
    create_spark_text_structurer_operator,
    create_distributed_processing_operator
)

from .kafka_producer_operator import (
    KafkaProducerOperator,
    FileUploadProducerOperator,
    ChunkProducerOperator,
    BatchFileUploadOperator,
    create_kafka_producer_operator,
    create_file_upload_producer_operator
)

from .kafka_consumer_operator import (
    KafkaConsumerOperator,
    TextExtractionConsumerOperator,
    LLMConsumerOperator,
    KafkaMessageSensor,
    create_kafka_consumer_operator,
    create_text_extraction_consumer_operator,
    create_llm_consumer_operator,
    create_kafka_message_sensor
)

from .llm_processing_operator import (
    LLMProcessingOperator,
    BatchLLMProcessingOperator,
    LLMHealthCheckOperator,
    create_llm_processing_operator,
    create_batch_llm_processing_operator,
    create_llm_health_check_operator
)

from .quality_validation_operator import (
    QualityValidationOperator,
    BatchQualityValidationOperator,
    create_quality_validation_operator,
    create_batch_quality_validation_operator
)

__all__ = [
    # Spark operators
    'SparkTextStructurerOperator',
    'SparkTextStructurerSubmitOperator',
    'DistributedTextProcessingOperator',
    'create_spark_text_structurer_operator',
    'create_distributed_processing_operator',
    
    # Kafka producer operators
    'KafkaProducerOperator',
    'FileUploadProducerOperator',
    'ChunkProducerOperator',
    'BatchFileUploadOperator',
    'create_kafka_producer_operator',
    'create_file_upload_producer_operator',
    
    # Kafka consumer operators
    'KafkaConsumerOperator',
    'TextExtractionConsumerOperator',
    'LLMConsumerOperator',
    'KafkaMessageSensor',
    'create_kafka_consumer_operator',
    'create_text_extraction_consumer_operator',
    'create_llm_consumer_operator',
    'create_kafka_message_sensor',
    
    # LLM processing operators
    'LLMProcessingOperator',
    'BatchLLMProcessingOperator',
    'LLMHealthCheckOperator',
    'create_llm_processing_operator',
    'create_batch_llm_processing_operator',
    'create_llm_health_check_operator',
    
    # Quality validation operators
    'QualityValidationOperator',
    'BatchQualityValidationOperator',
    'create_quality_validation_operator',
    'create_batch_quality_validation_operator'
]

# Version information
__version__ = '1.0.0'
__author__ = 'Text-to-audiobook Team'
__description__ = 'Custom Airflow operators for distributed text processing pipeline'