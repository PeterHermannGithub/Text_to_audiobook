"""
Grafana dashboard configurations for the distributed text-to-audiobook pipeline.

This module provides pre-built Grafana dashboard configurations for monitoring
system performance, health, and operational metrics.
"""

import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class GrafanaPanel:
    """Represents a Grafana panel configuration."""
    id: int
    title: str
    type: str
    targets: List[Dict[str, Any]]
    gridPos: Dict[str, int]
    options: Dict[str, Any] = None
    fieldConfig: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert panel to dictionary format."""
        panel_dict = {
            'id': self.id,
            'title': self.title,
            'type': self.type,
            'targets': self.targets,
            'gridPos': self.gridPos
        }
        
        if self.options:
            panel_dict['options'] = self.options
        
        if self.fieldConfig:
            panel_dict['fieldConfig'] = self.fieldConfig
        
        return panel_dict


class GrafanaDashboardBuilder:
    """Builder class for creating Grafana dashboards."""
    
    def __init__(self, title: str, datasource: str = "prometheus"):
        self.title = title
        self.datasource = datasource
        self.panels = []
        self.next_panel_id = 1
        self.current_row = 0
        self.current_col = 0
    
    def add_panel(self, panel: GrafanaPanel) -> 'GrafanaDashboardBuilder':
        """Add a panel to the dashboard."""
        self.panels.append(panel)
        return self
    
    def create_graph_panel(self, title: str, query: str, unit: str = "short", 
                          width: int = 12, height: int = 8) -> GrafanaPanel:
        """Create a graph panel."""
        panel = GrafanaPanel(
            id=self.next_panel_id,
            title=title,
            type="graph",
            targets=[{
                "expr": query,
                "format": "time_series",
                "legendFormat": "",
                "refId": "A"
            }],
            gridPos={
                "h": height,
                "w": width,
                "x": self.current_col,
                "y": self.current_row
            },
            options={
                "legend": {
                    "displayMode": "table",
                    "placement": "bottom"
                },
                "tooltip": {
                    "mode": "single"
                }
            },
            fieldConfig={
                "defaults": {
                    "unit": unit,
                    "min": 0
                }
            }
        )
        
        self.next_panel_id += 1
        self._update_position(width, height)
        return panel
    
    def create_stat_panel(self, title: str, query: str, unit: str = "short",
                         width: int = 6, height: int = 4) -> GrafanaPanel:
        """Create a stat panel."""
        panel = GrafanaPanel(
            id=self.next_panel_id,
            title=title,
            type="stat",
            targets=[{
                "expr": query,
                "format": "time_series",
                "legendFormat": "",
                "refId": "A"
            }],
            gridPos={
                "h": height,
                "w": width,
                "x": self.current_col,
                "y": self.current_row
            },
            options={
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"],
                    "fields": ""
                },
                "orientation": "auto",
                "textMode": "auto",
                "colorMode": "value",
                "graphMode": "area",
                "justifyMode": "auto"
            },
            fieldConfig={
                "defaults": {
                    "unit": unit,
                    "min": 0
                }
            }
        )
        
        self.next_panel_id += 1
        self._update_position(width, height)
        return panel
    
    def create_heatmap_panel(self, title: str, query: str, 
                           width: int = 12, height: int = 8) -> GrafanaPanel:
        """Create a heatmap panel."""
        panel = GrafanaPanel(
            id=self.next_panel_id,
            title=title,
            type="heatmap",
            targets=[{
                "expr": query,
                "format": "heatmap",
                "legendFormat": "",
                "refId": "A"
            }],
            gridPos={
                "h": height,
                "w": width,
                "x": self.current_col,
                "y": self.current_row
            },
            options={
                "tooltip": {
                    "show": True,
                    "showHistogram": True
                },
                "legend": {
                    "show": True
                }
            }
        )
        
        self.next_panel_id += 1
        self._update_position(width, height)
        return panel
    
    def create_table_panel(self, title: str, query: str,
                          width: int = 12, height: int = 8) -> GrafanaPanel:
        """Create a table panel."""
        panel = GrafanaPanel(
            id=self.next_panel_id,
            title=title,
            type="table",
            targets=[{
                "expr": query,
                "format": "table",
                "instant": True,
                "legendFormat": "",
                "refId": "A"
            }],
            gridPos={
                "h": height,
                "w": width,
                "x": self.current_col,
                "y": self.current_row
            },
            options={
                "showHeader": True,
                "sortBy": [{"displayName": "Time", "desc": True}]
            }
        )
        
        self.next_panel_id += 1
        self._update_position(width, height)
        return panel
    
    def new_row(self):
        """Start a new row."""
        self.current_row += 8
        self.current_col = 0
    
    def _update_position(self, width: int, height: int):
        """Update current position for next panel."""
        self.current_col += width
        if self.current_col >= 24:
            self.current_col = 0
            self.current_row += height
    
    def build(self) -> Dict[str, Any]:
        """Build the complete dashboard configuration."""
        return {
            "dashboard": {
                "id": None,
                "title": self.title,
                "tags": ["text-to-audiobook", "distributed-processing"],
                "timezone": "browser",
                "panels": [panel.to_dict() for panel in self.panels],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "timepicker": {
                    "refresh_intervals": ["5s", "10s", "30s", "1m", "5m", "15m", "30m", "1h", "2h", "1d"],
                    "time_options": ["5m", "15m", "1h", "6h", "12h", "24h", "2d", "7d", "30d"]
                },
                "refresh": "30s",
                "schemaVersion": 27,
                "version": 1,
                "links": []
            },
            "folderId": 0,
            "overwrite": True
        }


class DashboardConfigurations:
    """Pre-built dashboard configurations for different aspects of the system."""
    
    @staticmethod
    def create_system_overview_dashboard() -> Dict[str, Any]:
        """Create system overview dashboard."""
        builder = GrafanaDashboardBuilder("Text-to-Audiobook System Overview")
        
        # System Health Row
        builder.add_panel(builder.create_stat_panel(
            "System Health",
            'sum(text_to_audiobook_system_health_status)',
            "short", 6, 4
        ))
        
        builder.add_panel(builder.create_stat_panel(
            "Active Processing Jobs",
            'sum(text_to_audiobook_processing_requests_total{status="in_progress"})',
            "short", 6, 4
        ))
        
        builder.add_panel(builder.create_stat_panel(
            "Total Jobs Today",
            'increase(text_to_audiobook_processing_requests_total[24h])',
            "short", 6, 4
        ))
        
        builder.add_panel(builder.create_stat_panel(
            "Success Rate",
            'rate(text_to_audiobook_processing_requests_total{status="completed"}[1h]) / rate(text_to_audiobook_processing_requests_total[1h]) * 100',
            "percent", 6, 4
        ))
        
        builder.new_row()
        
        # Processing Performance Row
        builder.add_panel(builder.create_graph_panel(
            "Processing Requests Rate",
            'rate(text_to_audiobook_processing_requests_total[5m])',
            "reqps", 12, 8
        ))
        
        builder.add_panel(builder.create_graph_panel(
            "Average Processing Duration",
            'rate(text_to_audiobook_processing_duration_seconds_sum[5m]) / rate(text_to_audiobook_processing_duration_seconds_count[5m])',
            "s", 12, 8
        ))
        
        builder.new_row()
        
        # Queue Status Row
        builder.add_panel(builder.create_graph_panel(
            "Queue Sizes",
            'text_to_audiobook_processing_queue_size',
            "short", 12, 8
        ))
        
        builder.add_panel(builder.create_graph_panel(
            "Component Health Status",
            'text_to_audiobook_system_health_status',
            "short", 12, 8
        ))
        
        return builder.build()
    
    @staticmethod
    def create_spark_dashboard() -> Dict[str, Any]:
        """Create Spark monitoring dashboard."""
        builder = GrafanaDashboardBuilder("Spark Processing Dashboard")
        
        # Spark Job Metrics
        builder.add_panel(builder.create_stat_panel(
            "Active Spark Jobs",
            'text_to_audiobook_spark_jobs_total{status="running"}',
            "short", 6, 4
        ))
        
        builder.add_panel(builder.create_stat_panel(
            "Completed Jobs Today",
            'increase(text_to_audiobook_spark_jobs_total{status="completed"}[24h])',
            "short", 6, 4
        ))
        
        builder.add_panel(builder.create_stat_panel(
            "Failed Jobs",
            'increase(text_to_audiobook_spark_jobs_total{status="failed"}[24h])',
            "short", 6, 4
        ))
        
        builder.add_panel(builder.create_stat_panel(
            "Active Executors",
            'sum(text_to_audiobook_spark_active_executors)',
            "short", 6, 4
        ))
        
        builder.new_row()
        
        # Performance Metrics
        builder.add_panel(builder.create_graph_panel(
            "Spark Job Duration",
            'histogram_quantile(0.95, rate(text_to_audiobook_spark_job_duration_seconds_bucket[5m]))',
            "s", 12, 8
        ))
        
        builder.add_panel(builder.create_graph_panel(
            "Spark Memory Usage",
            'text_to_audiobook_spark_memory_usage_bytes',
            "bytes", 12, 8
        ))
        
        builder.new_row()
        
        # Executor Status
        builder.add_panel(builder.create_heatmap_panel(
            "Job Duration Heatmap",
            'rate(text_to_audiobook_spark_job_duration_seconds_bucket[5m])',
            24, 8
        ))
        
        return builder.build()
    
    @staticmethod
    def create_kafka_dashboard() -> Dict[str, Any]:
        """Create Kafka monitoring dashboard."""
        builder = GrafanaDashboardBuilder("Kafka Messaging Dashboard")
        
        # Message Throughput
        builder.add_panel(builder.create_stat_panel(
            "Messages Produced/sec",
            'rate(text_to_audiobook_kafka_messages_produced_total[1m])',
            "short", 6, 4
        ))
        
        builder.add_panel(builder.create_stat_panel(
            "Messages Consumed/sec",
            'rate(text_to_audiobook_kafka_messages_consumed_total[1m])',
            "short", 6, 4
        ))
        
        builder.add_panel(builder.create_stat_panel(
            "Consumer Lag",
            'sum(text_to_audiobook_kafka_consumer_lag)',
            "short", 6, 4
        ))
        
        builder.add_panel(builder.create_stat_panel(
            "Failed Messages",
            'increase(text_to_audiobook_kafka_messages_produced_total{status="failed"}[1h])',
            "short", 6, 4
        ))
        
        builder.new_row()
        
        # Message Flow
        builder.add_panel(builder.create_graph_panel(
            "Message Production Rate by Topic",
            'rate(text_to_audiobook_kafka_messages_produced_total[5m])',
            "msgps", 12, 8
        ))
        
        builder.add_panel(builder.create_graph_panel(
            "Message Consumption Rate by Topic",
            'rate(text_to_audiobook_kafka_messages_consumed_total[5m])',
            "msgps", 12, 8
        ))
        
        builder.new_row()
        
        # Producer Performance
        builder.add_panel(builder.create_graph_panel(
            "Producer Batch Size",
            'histogram_quantile(0.95, rate(text_to_audiobook_kafka_producer_batch_size_bucket[5m]))',
            "short", 12, 8
        ))
        
        builder.add_panel(builder.create_graph_panel(
            "Consumer Lag by Topic",
            'text_to_audiobook_kafka_consumer_lag',
            "short", 12, 8
        ))
        
        return builder.build()
    
    @staticmethod
    def create_llm_dashboard() -> Dict[str, Any]:
        """Create LLM monitoring dashboard."""
        builder = GrafanaDashboardBuilder("LLM Processing Dashboard")
        
        # LLM Request Metrics
        builder.add_panel(builder.create_stat_panel(
            "Requests/sec",
            'rate(text_to_audiobook_llm_requests_total[1m])',
            "reqps", 6, 4
        ))
        
        builder.add_panel(builder.create_stat_panel(
            "Success Rate",
            'rate(text_to_audiobook_llm_requests_total{status="success"}[5m]) / rate(text_to_audiobook_llm_requests_total[5m]) * 100',
            "percent", 6, 4
        ))
        
        builder.add_panel(builder.create_stat_panel(
            "Active Instances",
            'sum(text_to_audiobook_llm_active_instances)',
            "short", 6, 4
        ))
        
        builder.add_panel(builder.create_stat_panel(
            "Pool Utilization",
            'avg(text_to_audiobook_llm_pool_utilization)',
            "percent", 6, 4
        ))
        
        builder.new_row()
        
        # Performance Metrics
        builder.add_panel(builder.create_graph_panel(
            "Response Time by Engine",
            'histogram_quantile(0.95, rate(text_to_audiobook_llm_response_time_seconds_bucket[5m]))',
            "s", 12, 8
        ))
        
        builder.add_panel(builder.create_graph_panel(
            "Request Volume by Model",
            'rate(text_to_audiobook_llm_requests_total[5m])',
            "reqps", 12, 8
        ))
        
        builder.new_row()
        
        # Pool Status
        builder.add_panel(builder.create_graph_panel(
            "Pool Utilization Over Time",
            'text_to_audiobook_llm_pool_utilization',
            "percent", 12, 8
        ))
        
        builder.add_panel(builder.create_heatmap_panel(
            "Response Time Heatmap",
            'rate(text_to_audiobook_llm_response_time_seconds_bucket[5m])',
            12, 8
        ))
        
        return builder.build()
    
    @staticmethod
    def create_quality_dashboard() -> Dict[str, Any]:
        """Create quality monitoring dashboard."""
        builder = GrafanaDashboardBuilder("Quality Metrics Dashboard")
        
        # Quality Scores
        builder.add_panel(builder.create_stat_panel(
            "Average Quality Score",
            'avg(text_to_audiobook_quality_score)',
            "short", 6, 4
        ))
        
        builder.add_panel(builder.create_stat_panel(
            "Validation Errors",
            'increase(text_to_audiobook_validation_errors_total[1h])',
            "short", 6, 4
        ))
        
        builder.add_panel(builder.create_stat_panel(
            "Segments Processed",
            'increase(text_to_audiobook_segments_processed_total[1h])',
            "short", 6, 4
        ))
        
        builder.add_panel(builder.create_stat_panel(
            "Speakers Detected",
            'increase(text_to_audiobook_speakers_detected_total[1h])',
            "short", 6, 4
        ))
        
        builder.new_row()
        
        # Quality Trends
        builder.add_panel(builder.create_graph_panel(
            "Quality Score Over Time",
            'text_to_audiobook_quality_score',
            "short", 12, 8
        ))
        
        builder.add_panel(builder.create_graph_panel(
            "Validation Error Rate",
            'rate(text_to_audiobook_validation_errors_total[5m])',
            "short", 12, 8
        ))
        
        builder.new_row()
        
        # Processing Volume
        builder.add_panel(builder.create_graph_panel(
            "Text Processing Volume",
            'rate(text_to_audiobook_text_extraction_bytes[5m])',
            "binBps", 12, 8
        ))
        
        builder.add_panel(builder.create_table_panel(
            "Recent Quality Scores by Job",
            'text_to_audiobook_quality_score',
            12, 8
        ))
        
        return builder.build()
    
    @staticmethod
    def create_airflow_dashboard() -> Dict[str, Any]:
        """Create Airflow monitoring dashboard."""
        builder = GrafanaDashboardBuilder("Airflow Workflow Dashboard")
        
        # DAG Run Metrics
        builder.add_panel(builder.create_stat_panel(
            "Active DAG Runs",
            'sum(text_to_audiobook_airflow_dag_runs_total{status="running"})',
            "short", 6, 4
        ))
        
        builder.add_panel(builder.create_stat_panel(
            "Success Rate",
            'rate(text_to_audiobook_airflow_dag_runs_total{status="success"}[1h]) / rate(text_to_audiobook_airflow_dag_runs_total[1h]) * 100',
            "percent", 6, 4
        ))
        
        builder.add_panel(builder.create_stat_panel(
            "Active Tasks",
            'sum(text_to_audiobook_airflow_active_tasks)',
            "short", 6, 4
        ))
        
        builder.add_panel(builder.create_stat_panel(
            "Failed Runs Today",
            'increase(text_to_audiobook_airflow_dag_runs_total{status="failed"}[24h])',
            "short", 6, 4
        ))
        
        builder.new_row()
        
        # Task Performance
        builder.add_panel(builder.create_graph_panel(
            "Task Duration by Type",
            'histogram_quantile(0.95, rate(text_to_audiobook_airflow_task_duration_seconds_bucket[5m]))',
            "s", 12, 8
        ))
        
        builder.add_panel(builder.create_graph_panel(
            "DAG Run Success Rate",
            'rate(text_to_audiobook_airflow_dag_runs_total{status="success"}[5m])',
            "short", 12, 8
        ))
        
        builder.new_row()
        
        # Workflow Status
        builder.add_panel(builder.create_heatmap_panel(
            "Task Duration Heatmap",
            'rate(text_to_audiobook_airflow_task_duration_seconds_bucket[5m])',
            24, 8
        ))
        
        return builder.build()
    
    @staticmethod
    def create_resource_dashboard() -> Dict[str, Any]:
        """Create resource monitoring dashboard."""
        builder = GrafanaDashboardBuilder("Resource Usage Dashboard")
        
        # Resource Utilization
        builder.add_panel(builder.create_stat_panel(
            "CPU Usage",
            'avg(text_to_audiobook_cpu_usage_percent)',
            "percent", 6, 4
        ))
        
        builder.add_panel(builder.create_stat_panel(
            "Memory Usage",
            'avg(text_to_audiobook_memory_usage_bytes)',
            "bytes", 6, 4
        ))
        
        builder.add_panel(builder.create_stat_panel(
            "Disk Usage",
            'avg(text_to_audiobook_disk_usage_bytes)',
            "bytes", 6, 4
        ))
        
        builder.add_panel(builder.create_stat_panel(
            "Network I/O",
            'rate(text_to_audiobook_kafka_messages_produced_total[1m]) + rate(text_to_audiobook_kafka_messages_consumed_total[1m])',
            "short", 6, 4
        ))
        
        builder.new_row()
        
        # Resource Trends
        builder.add_panel(builder.create_graph_panel(
            "CPU Usage by Component",
            'text_to_audiobook_cpu_usage_percent',
            "percent", 12, 8
        ))
        
        builder.add_panel(builder.create_graph_panel(
            "Memory Usage by Component",
            'text_to_audiobook_memory_usage_bytes',
            "bytes", 12, 8
        ))
        
        builder.new_row()
        
        # Disk Usage
        builder.add_panel(builder.create_graph_panel(
            "Disk Usage by Mount Point",
            'text_to_audiobook_disk_usage_bytes',
            "bytes", 24, 8
        ))
        
        return builder.build()


def save_dashboard_to_file(dashboard: Dict[str, Any], filename: str, 
                          directory: str = "monitoring/dashboards"):
    """Save dashboard configuration to file."""
    dashboard_dir = Path(directory)
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = dashboard_dir / f"{filename}.json"
    with open(filepath, 'w') as f:
        json.dump(dashboard, f, indent=2)
    
    print(f"Dashboard saved to: {filepath}")


def generate_all_dashboards(output_dir: str = "monitoring/dashboards"):
    """Generate all dashboard configurations."""
    dashboards = {
        "system_overview": DashboardConfigurations.create_system_overview_dashboard(),
        "spark_processing": DashboardConfigurations.create_spark_dashboard(),
        "kafka_messaging": DashboardConfigurations.create_kafka_dashboard(),
        "llm_processing": DashboardConfigurations.create_llm_dashboard(),
        "quality_metrics": DashboardConfigurations.create_quality_dashboard(),
        "airflow_workflow": DashboardConfigurations.create_airflow_dashboard(),
        "resource_usage": DashboardConfigurations.create_resource_dashboard()
    }
    
    for name, dashboard in dashboards.items():
        save_dashboard_to_file(dashboard, name, output_dir)
    
    print(f"Generated {len(dashboards)} dashboard configurations in {output_dir}")


if __name__ == "__main__":
    generate_all_dashboards()