"""
Kafka configuration for event-driven text processing.

This module provides comprehensive Kafka configuration for the text-to-audiobook
event-driven processing system, including topic definitions, producer/consumer
configurations, and message serialization settings.
"""

import os
import json
from typing import Dict, List, Any
from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
import logging

logger = logging.getLogger(__name__)

class KafkaConfig:
    """Centralized Kafka configuration management."""
    
    # Kafka Cluster Configuration
    KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(",")
    KAFKA_SECURITY_PROTOCOL = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
    KAFKA_SASL_MECHANISM = os.getenv("KAFKA_SASL_MECHANISM", "PLAIN")
    
    # Topic Configuration
    TOPIC_REPLICATION_FACTOR = int(os.getenv("KAFKA_REPLICATION_FACTOR", "1"))
    TOPIC_NUM_PARTITIONS = int(os.getenv("KAFKA_NUM_PARTITIONS", "4"))
    TOPIC_RETENTION_MS = int(os.getenv("KAFKA_RETENTION_MS", "604800000"))  # 7 days
    
    # Producer Configuration
    PRODUCER_ACKS = os.getenv("KAFKA_PRODUCER_ACKS", "all")
    PRODUCER_RETRIES = int(os.getenv("KAFKA_PRODUCER_RETRIES", "3"))
    PRODUCER_BATCH_SIZE = int(os.getenv("KAFKA_PRODUCER_BATCH_SIZE", "16384"))
    PRODUCER_LINGER_MS = int(os.getenv("KAFKA_PRODUCER_LINGER_MS", "10"))
    PRODUCER_COMPRESSION_TYPE = os.getenv("KAFKA_PRODUCER_COMPRESSION", "snappy")
    
    # Consumer Configuration
    CONSUMER_AUTO_OFFSET_RESET = os.getenv("KAFKA_CONSUMER_AUTO_OFFSET_RESET", "earliest")
    CONSUMER_ENABLE_AUTO_COMMIT = os.getenv("KAFKA_CONSUMER_AUTO_COMMIT", "true").lower() == "true"
    CONSUMER_AUTO_COMMIT_INTERVAL_MS = int(os.getenv("KAFKA_CONSUMER_AUTO_COMMIT_INTERVAL", "5000"))
    CONSUMER_MAX_POLL_RECORDS = int(os.getenv("KAFKA_CONSUMER_MAX_POLL_RECORDS", "500"))
    CONSUMER_SESSION_TIMEOUT_MS = int(os.getenv("KAFKA_CONSUMER_SESSION_TIMEOUT", "30000"))
    
    # Topic Definitions
    TOPICS = {
        # Core Processing Topics
        "text-extraction-requests": {
            "description": "File upload and text extraction requests",
            "partitions": TOPIC_NUM_PARTITIONS,
            "replication_factor": TOPIC_REPLICATION_FACTOR,
            "config": {
                "retention.ms": str(TOPIC_RETENTION_MS),
                "compression.type": "snappy"
            }
        },
        "text-extraction-results": {
            "description": "Extracted text with metadata",
            "partitions": TOPIC_NUM_PARTITIONS,
            "replication_factor": TOPIC_REPLICATION_FACTOR,
            "config": {
                "retention.ms": str(TOPIC_RETENTION_MS),
                "compression.type": "snappy"
            }
        },
        "processing-requests": {
            "description": "Text processing job requests",
            "partitions": TOPIC_NUM_PARTITIONS,
            "replication_factor": TOPIC_REPLICATION_FACTOR,
            "config": {
                "retention.ms": str(TOPIC_RETENTION_MS),
                "compression.type": "snappy"
            }
        },
        "chunk-processing": {
            "description": "Individual chunk processing messages",
            "partitions": TOPIC_NUM_PARTITIONS * 2,  # Higher parallelism for chunks
            "replication_factor": TOPIC_REPLICATION_FACTOR,
            "config": {
                "retention.ms": str(TOPIC_RETENTION_MS),
                "compression.type": "snappy"
            }
        },
        "llm-classification": {
            "description": "LLM classification requests and responses",
            "partitions": TOPIC_NUM_PARTITIONS,
            "replication_factor": TOPIC_REPLICATION_FACTOR,
            "config": {
                "retention.ms": str(TOPIC_RETENTION_MS),
                "compression.type": "snappy"
            }
        },
        "validation-results": {
            "description": "Quality validation results",
            "partitions": TOPIC_NUM_PARTITIONS,
            "replication_factor": TOPIC_REPLICATION_FACTOR,
            "config": {
                "retention.ms": str(TOPIC_RETENTION_MS),
                "compression.type": "snappy"
            }
        },
        "processing-complete": {
            "description": "Processing completion notifications",
            "partitions": TOPIC_NUM_PARTITIONS,
            "replication_factor": TOPIC_REPLICATION_FACTOR,
            "config": {
                "retention.ms": str(TOPIC_RETENTION_MS),
                "compression.type": "snappy"
            }
        },
        "error-handling": {
            "description": "Error events and dead letter messages",
            "partitions": TOPIC_NUM_PARTITIONS,
            "replication_factor": TOPIC_REPLICATION_FACTOR,
            "config": {
                "retention.ms": str(TOPIC_RETENTION_MS * 2),  # Keep errors longer
                "compression.type": "snappy"
            }
        },
        # Status and Monitoring Topics
        "processing-status": {
            "description": "Processing status updates",
            "partitions": TOPIC_NUM_PARTITIONS,
            "replication_factor": TOPIC_REPLICATION_FACTOR,
            "config": {
                "retention.ms": str(TOPIC_RETENTION_MS),
                "compression.type": "snappy"
            }
        },
        "metrics": {
            "description": "System metrics and monitoring data",
            "partitions": TOPIC_NUM_PARTITIONS,
            "replication_factor": TOPIC_REPLICATION_FACTOR,
            "config": {
                "retention.ms": str(TOPIC_RETENTION_MS),
                "compression.type": "snappy"
            }
        }
    }
    
    @classmethod
    def get_producer_config(cls) -> Dict[str, Any]:
        """Get producer configuration."""
        return {
            'bootstrap_servers': cls.KAFKA_BOOTSTRAP_SERVERS,
            'security_protocol': cls.KAFKA_SECURITY_PROTOCOL,
            'acks': cls.PRODUCER_ACKS,
            'retries': cls.PRODUCER_RETRIES,
            'batch_size': cls.PRODUCER_BATCH_SIZE,
            'linger_ms': cls.PRODUCER_LINGER_MS,
            'compression_type': cls.PRODUCER_COMPRESSION_TYPE,
            'value_serializer': lambda v: json.dumps(v).encode('utf-8'),
            'key_serializer': lambda k: k.encode('utf-8') if k else None
        }
    
    @classmethod
    def get_consumer_config(cls, group_id: str) -> Dict[str, Any]:
        """Get consumer configuration."""
        return {
            'bootstrap_servers': cls.KAFKA_BOOTSTRAP_SERVERS,
            'security_protocol': cls.KAFKA_SECURITY_PROTOCOL,
            'group_id': group_id,
            'auto_offset_reset': cls.CONSUMER_AUTO_OFFSET_RESET,
            'enable_auto_commit': cls.CONSUMER_ENABLE_AUTO_COMMIT,
            'auto_commit_interval_ms': cls.CONSUMER_AUTO_COMMIT_INTERVAL_MS,
            'max_poll_records': cls.CONSUMER_MAX_POLL_RECORDS,
            'session_timeout_ms': cls.CONSUMER_SESSION_TIMEOUT_MS,
            'value_deserializer': lambda m: json.loads(m.decode('utf-8')),
            'key_deserializer': lambda k: k.decode('utf-8') if k else None
        }
    
    @classmethod
    def get_admin_config(cls) -> Dict[str, Any]:
        """Get admin client configuration."""
        return {
            'bootstrap_servers': cls.KAFKA_BOOTSTRAP_SERVERS,
            'security_protocol': cls.KAFKA_SECURITY_PROTOCOL
        }
    
    @classmethod
    def create_producer(cls) -> KafkaProducer:
        """Create Kafka producer with optimized configuration."""
        config = cls.get_producer_config()
        return KafkaProducer(**config)
    
    @classmethod
    def create_consumer(cls, group_id: str, topics: List[str]) -> KafkaConsumer:
        """Create Kafka consumer with optimized configuration."""
        config = cls.get_consumer_config(group_id)
        return KafkaConsumer(*topics, **config)
    
    @classmethod
    def create_admin_client(cls) -> KafkaAdminClient:
        """Create Kafka admin client."""
        config = cls.get_admin_config()
        return KafkaAdminClient(**config)


class KafkaTopicManager:
    """Manages Kafka topic creation and configuration."""
    
    def __init__(self):
        self.admin_client = KafkaConfig.create_admin_client()
        self.logger = logging.getLogger(__name__)
    
    def create_topics(self) -> None:
        """Create all required topics."""
        topics_to_create = []
        
        for topic_name, topic_config in KafkaConfig.TOPICS.items():
            new_topic = NewTopic(
                name=topic_name,
                num_partitions=topic_config['partitions'],
                replication_factor=topic_config['replication_factor'],
                topic_configs=topic_config.get('config', {})
            )
            topics_to_create.append(new_topic)
        
        try:
            fs = self.admin_client.create_topics(topics_to_create, validate_only=False)
            for topic, f in fs.items():
                try:
                    f.result()  # The result itself is None
                    self.logger.info(f"Topic {topic} created successfully")
                except TopicAlreadyExistsError:
                    self.logger.info(f"Topic {topic} already exists")
                except Exception as e:
                    self.logger.error(f"Failed to create topic {topic}: {e}")
        except Exception as e:
            self.logger.error(f"Failed to create topics: {e}")
    
    def delete_topics(self, topics: List[str]) -> None:
        """Delete specified topics."""
        try:
            fs = self.admin_client.delete_topics(topics)
            for topic, f in fs.items():
                try:
                    f.result()
                    self.logger.info(f"Topic {topic} deleted successfully")
                except Exception as e:
                    self.logger.error(f"Failed to delete topic {topic}: {e}")
        except Exception as e:
            self.logger.error(f"Failed to delete topics: {e}")
    
    def list_topics(self) -> List[str]:
        """List all topics."""
        try:
            metadata = self.admin_client.list_topics()
            return list(metadata.topics.keys())
        except Exception as e:
            self.logger.error(f"Failed to list topics: {e}")
            return []
    
    def describe_topic(self, topic_name: str) -> Dict[str, Any]:
        """Describe a specific topic."""
        try:
            metadata = self.admin_client.describe_topics([topic_name])
            return metadata[topic_name]
        except Exception as e:
            self.logger.error(f"Failed to describe topic {topic_name}: {e}")
            return {}


# Consumer Group Definitions
class ConsumerGroups:
    """Predefined consumer groups for different processing stages."""
    
    TEXT_EXTRACTION_GROUP = "text-extraction-consumer-group"
    CHUNK_PROCESSING_GROUP = "chunk-processing-consumer-group"
    LLM_CLASSIFICATION_GROUP = "llm-classification-consumer-group"
    VALIDATION_GROUP = "validation-consumer-group"
    RESULTS_AGGREGATION_GROUP = "results-aggregation-consumer-group"
    ERROR_HANDLING_GROUP = "error-handling-consumer-group"
    MONITORING_GROUP = "monitoring-consumer-group"


# Message Routing Configuration
class MessageRouting:
    """Configuration for message routing and partitioning."""
    
    @staticmethod
    def get_partition_key(message: Dict[str, Any]) -> str:
        """Get partition key for consistent routing."""
        return message.get('job_id', 'default')
    
    @staticmethod
    def get_routing_config() -> Dict[str, str]:
        """Get routing configuration for different message types."""
        return {
            'file_upload': 'text-extraction-requests',
            'text_extracted': 'text-extraction-results',
            'processing_request': 'processing-requests',
            'chunk_processing': 'chunk-processing',
            'llm_request': 'llm-classification',
            'validation_result': 'validation-results',
            'processing_complete': 'processing-complete',
            'error': 'error-handling',
            'status_update': 'processing-status',
            'metrics': 'metrics'
        }


# Health Check Configuration
class KafkaHealthCheck:
    """Kafka cluster health check utilities."""
    
    @staticmethod
    def check_cluster_health() -> Dict[str, Any]:
        """Check Kafka cluster health."""
        try:
            admin_client = KafkaConfig.create_admin_client()
            metadata = admin_client.list_topics()
            
            return {
                'status': 'healthy',
                'topics_count': len(metadata.topics),
                'brokers_count': len(metadata.brokers),
                'topics': list(metadata.topics.keys())
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    @staticmethod
    def check_topic_health(topic_name: str) -> Dict[str, Any]:
        """Check specific topic health."""
        try:
            admin_client = KafkaConfig.create_admin_client()
            metadata = admin_client.describe_topics([topic_name])
            
            topic_metadata = metadata[topic_name]
            return {
                'status': 'healthy',
                'partitions': len(topic_metadata.partitions),
                'replication_factor': len(topic_metadata.partitions[0].replicas) if topic_metadata.partitions else 0
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }


# Utility functions for easy setup
def setup_kafka_infrastructure():
    """Set up Kafka infrastructure (topics, etc.)."""
    logger.info("Setting up Kafka infrastructure...")
    
    # Create topic manager
    topic_manager = KafkaTopicManager()
    
    # Create all required topics
    topic_manager.create_topics()
    
    # Verify topics were created
    existing_topics = topic_manager.list_topics()
    required_topics = set(KafkaConfig.TOPICS.keys())
    missing_topics = required_topics - set(existing_topics)
    
    if missing_topics:
        logger.error(f"Missing topics: {missing_topics}")
        return False
    
    logger.info("Kafka infrastructure setup completed successfully")
    return True


def cleanup_kafka_infrastructure():
    """Clean up Kafka infrastructure."""
    logger.info("Cleaning up Kafka infrastructure...")
    
    topic_manager = KafkaTopicManager()
    topics_to_delete = list(KafkaConfig.TOPICS.keys())
    topic_manager.delete_topics(topics_to_delete)
    
    logger.info("Kafka infrastructure cleanup completed")