"""
Prometheus alert rules for the distributed text-to-audiobook pipeline.

This module provides comprehensive alerting rules for monitoring system health,
performance, and operational conditions across all components.
"""

import yaml
from typing import Dict, Any, List, Optional
from datetime import timedelta
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class AlertRule:
    """Represents a Prometheus alert rule."""
    alert: str
    expr: str
    for_duration: str
    labels: Dict[str, str]
    annotations: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert rule to dictionary format."""
        return {
            'alert': self.alert,
            'expr': self.expr,
            'for': self.for_duration,
            'labels': self.labels,
            'annotations': self.annotations
        }


class AlertRuleGroup:
    """Represents a group of related alert rules."""
    
    def __init__(self, name: str, interval: str = "30s"):
        self.name = name
        self.interval = interval
        self.rules = []
    
    def add_rule(self, rule: AlertRule) -> 'AlertRuleGroup':
        """Add a rule to this group."""
        self.rules.append(rule)
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert rule group to dictionary format."""
        return {
            'name': self.name,
            'interval': self.interval,
            'rules': [rule.to_dict() for rule in self.rules]
        }


class AlertRulesManager:
    """Manager class for creating and organizing alert rules."""
    
    def __init__(self, namespace: str = "text_to_audiobook"):
        self.namespace = namespace
        self.rule_groups = []
    
    def add_rule_group(self, group: AlertRuleGroup) -> 'AlertRulesManager':
        """Add a rule group."""
        self.rule_groups.append(group)
        return self
    
    def create_system_health_rules(self) -> AlertRuleGroup:
        """Create system health monitoring rules."""
        group = AlertRuleGroup("system_health", "30s")
        
        # System health alert
        group.add_rule(AlertRule(
            alert="SystemUnhealthy",
            expr=f'{self.namespace}_system_health_status == 0',
            for_duration="2m",
            labels={
                "severity": "critical",
                "service": "text_to_audiobook",
                "category": "system"
            },
            annotations={
                "summary": "System component {{ $labels.component }} is unhealthy",
                "description": "The {{ $labels.component }} component has been reporting unhealthy status for more than 2 minutes.",
                "runbook_url": "https://docs.example.com/runbooks/system-health"
            }
        ))
        
        # High error rate alert
        group.add_rule(AlertRule(
            alert="HighErrorRate",
            expr=f'rate({self.namespace}_processing_requests_total{{status="failed"}}[5m]) / rate({self.namespace}_processing_requests_total[5m]) > 0.1',
            for_duration="5m",
            labels={
                "severity": "warning",
                "service": "text_to_audiobook",
                "category": "processing"
            },
            annotations={
                "summary": "High error rate detected in processing requests",
                "description": "Error rate is {{ $value | humanizePercentage }} for job type {{ $labels.job_type }}",
                "runbook_url": "https://docs.example.com/runbooks/high-error-rate"
            }
        ))
        
        # Processing queue buildup
        group.add_rule(AlertRule(
            alert="ProcessingQueueBuildup",
            expr=f'{self.namespace}_processing_queue_size > 100',
            for_duration="10m",
            labels={
                "severity": "warning",
                "service": "text_to_audiobook",
                "category": "processing"
            },
            annotations={
                "summary": "Processing queue {{ $labels.queue_type }} is building up",
                "description": "Queue size is {{ $value }} items, which is above the threshold of 100",
                "runbook_url": "https://docs.example.com/runbooks/queue-buildup"
            }
        ))
        
        # Service down alert
        group.add_rule(AlertRule(
            alert="ServiceDown",
            expr=f'up{{job=~".*{self.namespace}.*"}} == 0',
            for_duration="1m",
            labels={
                "severity": "critical",
                "service": "text_to_audiobook",
                "category": "availability"
            },
            annotations={
                "summary": "Service {{ $labels.job }} is down",
                "description": "Service {{ $labels.job }} has been down for more than 1 minute",
                "runbook_url": "https://docs.example.com/runbooks/service-down"
            }
        ))
        
        return group
    
    def create_spark_rules(self) -> AlertRuleGroup:
        """Create Spark-specific alert rules."""
        group = AlertRuleGroup("spark_processing", "30s")
        
        # Spark job failure rate
        group.add_rule(AlertRule(
            alert="SparkJobFailureRate",
            expr=f'rate({self.namespace}_spark_jobs_total{{status="failed"}}[10m]) / rate({self.namespace}_spark_jobs_total[10m]) > 0.2',
            for_duration="5m",
            labels={
                "severity": "warning",
                "service": "text_to_audiobook",
                "category": "spark"
            },
            annotations={
                "summary": "High Spark job failure rate detected",
                "description": "Spark job failure rate is {{ $value | humanizePercentage }} for job type {{ $labels.job_type }}",
                "runbook_url": "https://docs.example.com/runbooks/spark-failures"
            }
        ))
        
        # Long running Spark job
        group.add_rule(AlertRule(
            alert="SparkJobTooLong",
            expr=f'histogram_quantile(0.95, rate({self.namespace}_spark_job_duration_seconds_bucket[10m])) > 1800',
            for_duration="0s",
            labels={
                "severity": "warning",
                "service": "text_to_audiobook",
                "category": "spark"
            },
            annotations={
                "summary": "Spark job taking too long to complete",
                "description": "95th percentile of Spark job duration is {{ $value | humanizeDuration }} for job type {{ $labels.job_type }}",
                "runbook_url": "https://docs.example.com/runbooks/spark-performance"
            }
        ))
        
        # Spark executor failure
        group.add_rule(AlertRule(
            alert="SparkExecutorFailure",
            expr=f'rate({self.namespace}_spark_active_executors[5m]) < -0.1',
            for_duration="2m",
            labels={
                "severity": "warning",
                "service": "text_to_audiobook",
                "category": "spark"
            },
            annotations={
                "summary": "Spark executors are failing",
                "description": "Spark executor count is decreasing rapidly for application {{ $labels.application_id }}",
                "runbook_url": "https://docs.example.com/runbooks/spark-executors"
            }
        ))
        
        # Spark memory usage
        group.add_rule(AlertRule(
            alert="SparkHighMemoryUsage",
            expr=f'{self.namespace}_spark_memory_usage_bytes > 8 * 1024 * 1024 * 1024',
            for_duration="5m",
            labels={
                "severity": "warning",
                "service": "text_to_audiobook",
                "category": "spark"
            },
            annotations={
                "summary": "Spark executor using high memory",
                "description": "Executor {{ $labels.executor_id }} is using {{ $value | humanizeBytes }} of memory",
                "runbook_url": "https://docs.example.com/runbooks/spark-memory"
            }
        ))
        
        return group
    
    def create_kafka_rules(self) -> AlertRuleGroup:
        """Create Kafka-specific alert rules."""
        group = AlertRuleGroup("kafka_messaging", "30s")
        
        # Kafka consumer lag
        group.add_rule(AlertRule(
            alert="KafkaConsumerLag",
            expr=f'{self.namespace}_kafka_consumer_lag > 1000',
            for_duration="5m",
            labels={
                "severity": "warning",
                "service": "text_to_audiobook",
                "category": "kafka"
            },
            annotations={
                "summary": "Kafka consumer lag is high",
                "description": "Consumer lag is {{ $value }} messages for topic {{ $labels.topic }} partition {{ $labels.partition }}",
                "runbook_url": "https://docs.example.com/runbooks/kafka-lag"
            }
        ))
        
        # Kafka producer failures
        group.add_rule(AlertRule(
            alert="KafkaProducerFailures",
            expr=f'rate({self.namespace}_kafka_messages_produced_total{{status="failed"}}[5m]) > 0.01',
            for_duration="2m",
            labels={
                "severity": "warning",
                "service": "text_to_audiobook",
                "category": "kafka"
            },
            annotations={
                "summary": "Kafka producer failures detected",
                "description": "Producer failure rate is {{ $value | humanize }} messages/sec for topic {{ $labels.topic }}",
                "runbook_url": "https://docs.example.com/runbooks/kafka-producer"
            }
        ))
        
        # Kafka message rate drop
        group.add_rule(AlertRule(
            alert="KafkaMessageRateDrop",
            expr=f'rate({self.namespace}_kafka_messages_produced_total[5m]) < 0.1',
            for_duration="10m",
            labels={
                "severity": "warning",
                "service": "text_to_audiobook",
                "category": "kafka"
            },
            annotations={
                "summary": "Kafka message production rate is very low",
                "description": "Message production rate is {{ $value | humanize }} messages/sec for topic {{ $labels.topic }}",
                "runbook_url": "https://docs.example.com/runbooks/kafka-low-throughput"
            }
        ))
        
        # Kafka partition imbalance
        group.add_rule(AlertRule(
            alert="KafkaPartitionImbalance",
            expr=f'stddev by (topic) ({self.namespace}_kafka_consumer_lag) > 500',
            for_duration="15m",
            labels={
                "severity": "info",
                "service": "text_to_audiobook",
                "category": "kafka"
            },
            annotations={
                "summary": "Kafka partition processing imbalance detected",
                "description": "Partition lag standard deviation is {{ $value | humanize }} for topic {{ $labels.topic }}",
                "runbook_url": "https://docs.example.com/runbooks/kafka-rebalancing"
            }
        ))
        
        return group
    
    def create_llm_rules(self) -> AlertRuleGroup:
        """Create LLM-specific alert rules."""
        group = AlertRuleGroup("llm_processing", "30s")
        
        # LLM request failure rate
        group.add_rule(AlertRule(
            alert="LLMHighFailureRate",
            expr=f'rate({self.namespace}_llm_requests_total{{status!="success"}}[5m]) / rate({self.namespace}_llm_requests_total[5m]) > 0.1',
            for_duration="3m",
            labels={
                "severity": "warning",
                "service": "text_to_audiobook",
                "category": "llm"
            },
            annotations={
                "summary": "LLM request failure rate is high",
                "description": "LLM failure rate is {{ $value | humanizePercentage }} for engine {{ $labels.engine }} model {{ $labels.model }}",
                "runbook_url": "https://docs.example.com/runbooks/llm-failures"
            }
        ))
        
        # LLM response time
        group.add_rule(AlertRule(
            alert="LLMSlowResponse",
            expr=f'histogram_quantile(0.95, rate({self.namespace}_llm_response_time_seconds_bucket[5m])) > 30',
            for_duration="5m",
            labels={
                "severity": "warning",
                "service": "text_to_audiobook",
                "category": "llm"
            },
            annotations={
                "summary": "LLM response time is slow",
                "description": "95th percentile response time is {{ $value | humanizeDuration }} for engine {{ $labels.engine }}",
                "runbook_url": "https://docs.example.com/runbooks/llm-performance"
            }
        ))
        
        # LLM pool exhaustion
        group.add_rule(AlertRule(
            alert="LLMPoolExhaustion",
            expr=f'{self.namespace}_llm_pool_utilization > 90',
            for_duration="5m",
            labels={
                "severity": "warning",
                "service": "text_to_audiobook",
                "category": "llm"
            },
            annotations={
                "summary": "LLM pool utilization is very high",
                "description": "Pool {{ $labels.pool_name }} utilization is {{ $value }}%",
                "runbook_url": "https://docs.example.com/runbooks/llm-pool"
            }
        ))
        
        # LLM instance down
        group.add_rule(AlertRule(
            alert="LLMInstanceDown",
            expr=f'{self.namespace}_llm_active_instances == 0',
            for_duration="1m",
            labels={
                "severity": "critical",
                "service": "text_to_audiobook",
                "category": "llm"
            },
            annotations={
                "summary": "No active LLM instances available",
                "description": "No active instances for engine {{ $labels.engine }} model {{ $labels.model }}",
                "runbook_url": "https://docs.example.com/runbooks/llm-instances"
            }
        ))
        
        return group
    
    def create_quality_rules(self) -> AlertRuleGroup:
        """Create quality monitoring alert rules."""
        group = AlertRuleGroup("quality_monitoring", "60s")
        
        # Low quality score
        group.add_rule(AlertRule(
            alert="LowQualityScore",
            expr=f'avg({self.namespace}_quality_score) < 80',
            for_duration="10m",
            labels={
                "severity": "warning",
                "service": "text_to_audiobook",
                "category": "quality"
            },
            annotations={
                "summary": "Quality score is below threshold",
                "description": "Average quality score is {{ $value | humanize }} for metric {{ $labels.metric_type }}",
                "runbook_url": "https://docs.example.com/runbooks/quality-score"
            }
        ))
        
        # High validation error rate
        group.add_rule(AlertRule(
            alert="HighValidationErrorRate",
            expr=f'rate({self.namespace}_validation_errors_total[10m]) > 0.05',
            for_duration="5m",
            labels={
                "severity": "warning",
                "service": "text_to_audiobook",
                "category": "quality"
            },
            annotations={
                "summary": "High validation error rate detected",
                "description": "Validation error rate is {{ $value | humanize }} errors/sec for error type {{ $labels.error_type }}",
                "runbook_url": "https://docs.example.com/runbooks/validation-errors"
            }
        ))
        
        # Processing accuracy drop
        group.add_rule(AlertRule(
            alert="ProcessingAccuracyDrop",
            expr=f'rate({self.namespace}_segments_processed_total{{segment_type="ambiguous"}}[1h]) / rate({self.namespace}_segments_processed_total[1h]) > 0.3',
            for_duration="30m",
            labels={
                "severity": "info",
                "service": "text_to_audiobook",
                "category": "quality"
            },
            annotations={
                "summary": "Processing accuracy has dropped",
                "description": "Ambiguous segment rate is {{ $value | humanizePercentage }} for job {{ $labels.job_id }}",
                "runbook_url": "https://docs.example.com/runbooks/processing-accuracy"
            }
        ))
        
        return group
    
    def create_airflow_rules(self) -> AlertRuleGroup:
        """Create Airflow-specific alert rules."""
        group = AlertRuleGroup("airflow_workflow", "60s")
        
        # DAG run failures
        group.add_rule(AlertRule(
            alert="AirflowDAGFailures",
            expr=f'rate({self.namespace}_airflow_dag_runs_total{{status="failed"}}[1h]) > 0.01',
            for_duration="5m",
            labels={
                "severity": "warning",
                "service": "text_to_audiobook",
                "category": "airflow"
            },
            annotations={
                "summary": "Airflow DAG failures detected",
                "description": "DAG {{ $labels.dag_id }} failure rate is {{ $value | humanize }} failures/sec",
                "runbook_url": "https://docs.example.com/runbooks/airflow-failures"
            }
        ))
        
        # Long running tasks
        group.add_rule(AlertRule(
            alert="AirflowLongRunningTask",
            expr=f'histogram_quantile(0.95, rate({self.namespace}_airflow_task_duration_seconds_bucket[10m])) > 3600',
            for_duration="0s",
            labels={
                "severity": "warning",
                "service": "text_to_audiobook",
                "category": "airflow"
            },
            annotations={
                "summary": "Airflow task taking too long",
                "description": "95th percentile task duration is {{ $value | humanizeDuration }} for DAG {{ $labels.dag_id }} task {{ $labels.task_id }}",
                "runbook_url": "https://docs.example.com/runbooks/airflow-performance"
            }
        ))
        
        # Task queue buildup
        group.add_rule(AlertRule(
            alert="AirflowTaskQueueBuildup",
            expr=f'{self.namespace}_airflow_active_tasks > 50',
            for_duration="15m",
            labels={
                "severity": "warning",
                "service": "text_to_audiobook",
                "category": "airflow"
            },
            annotations={
                "summary": "Airflow task queue building up",
                "description": "Active task count is {{ $value }} for DAG {{ $labels.dag_id }}",
                "runbook_url": "https://docs.example.com/runbooks/airflow-queue"
            }
        ))
        
        return group
    
    def create_resource_rules(self) -> AlertRuleGroup:
        """Create resource monitoring alert rules."""
        group = AlertRuleGroup("resource_monitoring", "60s")
        
        # High CPU usage
        group.add_rule(AlertRule(
            alert="HighCPUUsage",
            expr=f'{self.namespace}_cpu_usage_percent > 80',
            for_duration="10m",
            labels={
                "severity": "warning",
                "service": "text_to_audiobook",
                "category": "resources"
            },
            annotations={
                "summary": "High CPU usage detected",
                "description": "CPU usage is {{ $value }}% for component {{ $labels.component }} on host {{ $labels.host }}",
                "runbook_url": "https://docs.example.com/runbooks/high-cpu"
            }
        ))
        
        # High memory usage
        group.add_rule(AlertRule(
            alert="HighMemoryUsage",
            expr=f'{self.namespace}_memory_usage_bytes > 16 * 1024 * 1024 * 1024',
            for_duration="10m",
            labels={
                "severity": "warning",
                "service": "text_to_audiobook",
                "category": "resources"
            },
            annotations={
                "summary": "High memory usage detected",
                "description": "Memory usage is {{ $value | humanizeBytes }} for component {{ $labels.component }} on host {{ $labels.host }}",
                "runbook_url": "https://docs.example.com/runbooks/high-memory"
            }
        ))
        
        # Disk space warning
        group.add_rule(AlertRule(
            alert="DiskSpaceWarning",
            expr=f'{self.namespace}_disk_usage_bytes > 50 * 1024 * 1024 * 1024',
            for_duration="30m",
            labels={
                "severity": "warning",
                "service": "text_to_audiobook",
                "category": "resources"
            },
            annotations={
                "summary": "Disk space usage is high",
                "description": "Disk usage is {{ $value | humanizeBytes }} for mount {{ $labels.mount_point }} on host {{ $labels.host }}",
                "runbook_url": "https://docs.example.com/runbooks/disk-space"
            }
        ))
        
        # Disk space critical
        group.add_rule(AlertRule(
            alert="DiskSpaceCritical",
            expr=f'{self.namespace}_disk_usage_bytes > 80 * 1024 * 1024 * 1024',
            for_duration="5m",
            labels={
                "severity": "critical",
                "service": "text_to_audiobook",
                "category": "resources"
            },
            annotations={
                "summary": "Disk space is critically low",
                "description": "Disk usage is {{ $value | humanizeBytes }} for mount {{ $labels.mount_point }} on host {{ $labels.host }}",
                "runbook_url": "https://docs.example.com/runbooks/disk-space-critical"
            }
        ))
        
        return group
    
    def generate_all_rules(self) -> 'AlertRulesManager':
        """Generate all alert rule groups."""
        self.add_rule_group(self.create_system_health_rules())
        self.add_rule_group(self.create_spark_rules())
        self.add_rule_group(self.create_kafka_rules())
        self.add_rule_group(self.create_llm_rules())
        self.add_rule_group(self.create_quality_rules())
        self.add_rule_group(self.create_airflow_rules())
        self.add_rule_group(self.create_resource_rules())
        return self
    
    def to_yaml(self) -> str:
        """Convert all rules to YAML format."""
        rules_dict = {
            'groups': [group.to_dict() for group in self.rule_groups]
        }
        return yaml.dump(rules_dict, default_flow_style=False)
    
    def save_to_file(self, filename: str, directory: str = "monitoring/alerts"):
        """Save alert rules to file."""
        alert_dir = Path(directory)
        alert_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = alert_dir / f"{filename}.yml"
        with open(filepath, 'w') as f:
            f.write(self.to_yaml())
        
        print(f"Alert rules saved to: {filepath}")


def generate_alert_rules(namespace: str = "text_to_audiobook", 
                        output_file: str = "alert_rules.yml",
                        output_dir: str = "monitoring/alerts"):
    """Generate comprehensive alert rules."""
    manager = AlertRulesManager(namespace)
    manager.generate_all_rules()
    manager.save_to_file(output_file, output_dir)
    
    print(f"Generated {len(manager.rule_groups)} alert rule groups")


if __name__ == "__main__":
    generate_alert_rules()