#!/usr/bin/env python3
"""
Service Integration Validation Script for Text-to-Audiobook System

This script validates that all services (Docker, Grafana, Prometheus, Kafka, Spark, Airflow)
are running correctly and can communicate with each other.
"""

import sys
import json
import time
import requests
import subprocess
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ServiceStatus:
    """Represents the status of a service check."""
    name: str
    status: str  # "pass", "fail", "warn"
    message: str
    details: Optional[Dict] = None


class ServiceValidator:
    """Main class for validating all services."""
    
    def __init__(self):
        self.results: List[ServiceStatus] = []
        self.services = {
            'app': 'http://localhost:8000',
            'grafana': 'http://localhost:3000',
            'prometheus': 'http://localhost:9090',
            'kafka-ui': 'http://localhost:8082',
            'spark-master': 'http://localhost:8080',
            'spark-worker': 'http://localhost:8081',
            'airflow': 'http://localhost:8090',  # Updated port
            'redis': 'localhost:6379'
        }
    
    def run_all_checks(self) -> bool:
        """Run all validation checks."""
        print("🚀 Starting Text-to-Audiobook Service Integration Validation")
        print("=" * 70)
        
        # Run checks
        self.check_docker_services()
        self.check_service_endpoints()
        self.check_prometheus_targets()
        self.check_grafana_dashboards()
        self.check_kafka_topics()
        self.check_spark_cluster()
        self.check_airflow_health()
        self.check_service_integration()
        
        # Summary
        self.print_summary()
        
        # Return overall status
        failed_checks = [r for r in self.results if r.status == "fail"]
        return len(failed_checks) == 0
    
    def check_docker_services(self):
        """Check Docker services status."""
        print("\n📦 Checking Docker Services...")
        
        try:
            # Get running containers
            result = subprocess.run(
                ['docker', 'ps', '--format', 'json'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                self.results.append(ServiceStatus(
                    "docker", "fail", 
                    "Docker command failed", 
                    {"error": result.stderr}
                ))
                return
            
            # Parse container info
            containers = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    try:
                        containers.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            
            # Check expected services
            expected_services = [
                'text_to_audiobook_app',
                'text_to_audiobook_redis', 
                'text_to_audiobook_kafka',
                'text_to_audiobook_zookeeper',
                'text_to_audiobook_spark_master',
                'text_to_audiobook_prometheus',
                'text_to_audiobook_grafana'
            ]
            
            running_services = [c['Names'] for c in containers]
            
            for service in expected_services:
                if any(service in name for name in running_services):
                    self.results.append(ServiceStatus(
                        f"docker-{service}", "pass",
                        f"Service {service} is running"
                    ))
                else:
                    self.results.append(ServiceStatus(
                        f"docker-{service}", "fail",
                        f"Service {service} is not running"
                    ))
            
        except subprocess.TimeoutExpired:
            self.results.append(ServiceStatus(
                "docker", "fail", "Docker command timed out"
            ))
        except Exception as e:
            self.results.append(ServiceStatus(
                "docker", "fail", f"Docker check failed: {str(e)}"
            ))
    
    def check_service_endpoints(self):
        """Check if service endpoints are accessible."""
        print("\n🌐 Checking Service Endpoints...")
        
        for service_name, url in self.services.items():
            if service_name == 'redis':
                self.check_redis_connection()
                continue
                
            try:
                # Determine appropriate endpoint
                if service_name == 'prometheus':
                    endpoint = f"{url}/-/healthy"
                elif service_name == 'grafana':
                    endpoint = f"{url}/api/health"
                elif service_name == 'spark-master':
                    endpoint = f"{url}"  # Spark UI homepage
                elif service_name == 'airflow':
                    endpoint = f"{url}/health"
                else:
                    endpoint = url
                
                response = requests.get(endpoint, timeout=10)
                
                if response.status_code == 200:
                    self.results.append(ServiceStatus(
                        f"endpoint-{service_name}", "pass",
                        f"{service_name} endpoint is accessible",
                        {"url": endpoint, "status_code": response.status_code}
                    ))
                else:
                    self.results.append(ServiceStatus(
                        f"endpoint-{service_name}", "warn",
                        f"{service_name} returned status {response.status_code}",
                        {"url": endpoint, "status_code": response.status_code}
                    ))
                    
            except requests.exceptions.ConnectionError:
                self.results.append(ServiceStatus(
                    f"endpoint-{service_name}", "fail",
                    f"{service_name} connection refused",
                    {"url": url}
                ))
            except requests.exceptions.Timeout:
                self.results.append(ServiceStatus(
                    f"endpoint-{service_name}", "fail",
                    f"{service_name} request timed out",
                    {"url": url}
                ))
            except Exception as e:
                self.results.append(ServiceStatus(
                    f"endpoint-{service_name}", "fail",
                    f"{service_name} check failed: {str(e)}",
                    {"url": url}
                ))
    
    def check_redis_connection(self):
        """Check Redis connection."""
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            r.ping()
            self.results.append(ServiceStatus(
                "endpoint-redis", "pass",
                "Redis connection successful"
            ))
        except ImportError:
            self.results.append(ServiceStatus(
                "endpoint-redis", "warn",
                "Redis package not available for testing"
            ))
        except Exception as e:
            self.results.append(ServiceStatus(
                "endpoint-redis", "fail",
                f"Redis connection failed: {str(e)}"
            ))
    
    def check_prometheus_targets(self):
        """Check Prometheus targets status."""
        print("\n📊 Checking Prometheus Targets...")
        
        try:
            response = requests.get(
                "http://localhost:9090/api/v1/targets",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                targets = data.get('data', {}).get('activeTargets', [])
                
                healthy_targets = [t for t in targets if t['health'] == 'up']
                total_targets = len(targets)
                
                if total_targets == 0:
                    self.results.append(ServiceStatus(
                        "prometheus-targets", "warn",
                        "No Prometheus targets configured"
                    ))
                elif len(healthy_targets) == total_targets:
                    self.results.append(ServiceStatus(
                        "prometheus-targets", "pass",
                        f"All {total_targets} Prometheus targets are healthy",
                        {"healthy": len(healthy_targets), "total": total_targets}
                    ))
                else:
                    self.results.append(ServiceStatus(
                        "prometheus-targets", "warn",
                        f"{len(healthy_targets)}/{total_targets} Prometheus targets are healthy",
                        {"healthy": len(healthy_targets), "total": total_targets}
                    ))
            else:
                self.results.append(ServiceStatus(
                    "prometheus-targets", "fail",
                    f"Prometheus API returned status {response.status_code}"
                ))
                
        except Exception as e:
            self.results.append(ServiceStatus(
                "prometheus-targets", "fail",
                f"Prometheus targets check failed: {str(e)}"
            ))
    
    def check_grafana_dashboards(self):
        """Check Grafana dashboards."""
        print("\n📈 Checking Grafana Dashboards...")
        
        try:
            # Check if dashboards are accessible
            response = requests.get(
                "http://localhost:3000/api/search",
                timeout=10,
                auth=('admin', 'admin')
            )
            
            if response.status_code == 200:
                dashboards = response.json()
                
                expected_dashboards = [
                    'system-overview',
                    'kafka-monitoring',
                    'spark-monitoring', 
                    'llm-monitoring',
                    'airflow-monitoring'
                ]
                
                found_dashboards = [d.get('uid', '') for d in dashboards]
                
                missing = [d for d in expected_dashboards if d not in found_dashboards]
                
                if not missing:
                    self.results.append(ServiceStatus(
                        "grafana-dashboards", "pass",
                        f"All {len(expected_dashboards)} expected dashboards found",
                        {"dashboards": len(dashboards)}
                    ))
                else:
                    self.results.append(ServiceStatus(
                        "grafana-dashboards", "warn",
                        f"Missing dashboards: {', '.join(missing)}",
                        {"missing": missing, "found": len(dashboards)}
                    ))
            else:
                self.results.append(ServiceStatus(
                    "grafana-dashboards", "fail",
                    f"Grafana API returned status {response.status_code}"
                ))
                
        except Exception as e:
            self.results.append(ServiceStatus(
                "grafana-dashboards", "fail",
                f"Grafana dashboards check failed: {str(e)}"
            ))
    
    def check_kafka_topics(self):
        """Check Kafka topics."""
        print("\n📨 Checking Kafka Topics...")
        
        try:
            # Try to get topics via Kafka UI API (if available)
            response = requests.get(
                "http://localhost:8082/api/clusters/local/topics",
                timeout=10
            )
            
            if response.status_code == 200:
                topics = response.json()
                
                expected_topics = [
                    'text-extraction-requests',
                    'text-extraction-results', 
                    'processing-requests',
                    'chunk-processing',
                    'llm-classification'
                ]
                
                if isinstance(topics, list):
                    topic_names = [t.get('name', '') for t in topics]
                    
                    found_topics = [t for t in expected_topics if t in topic_names]
                    
                    if found_topics:
                        self.results.append(ServiceStatus(
                            "kafka-topics", "pass",
                            f"Found {len(found_topics)} expected Kafka topics",
                            {"topics": found_topics}
                        ))
                    else:
                        self.results.append(ServiceStatus(
                            "kafka-topics", "warn",
                            "No expected Kafka topics found (may be auto-created on first use)",
                            {"total_topics": len(topic_names)}
                        ))
                else:
                    self.results.append(ServiceStatus(
                        "kafka-topics", "warn",
                        "Kafka topics response format unexpected"
                    ))
            else:
                self.results.append(ServiceStatus(
                    "kafka-topics", "warn",
                    f"Kafka UI API returned status {response.status_code} (topics may be auto-created)"
                ))
                
        except Exception as e:
            self.results.append(ServiceStatus(
                "kafka-topics", "warn",
                f"Kafka topics check failed: {str(e)} (topics may be auto-created on first use)"
            ))
    
    def check_spark_cluster(self):
        """Check Spark cluster status."""
        print("\n⚡ Checking Spark Cluster...")
        
        try:
            # Check Spark master
            response = requests.get("http://localhost:8080", timeout=10)
            
            if response.status_code == 200:
                # Parse the HTML to get worker info (basic check)
                if "Spark Master" in response.text:
                    self.results.append(ServiceStatus(
                        "spark-master", "pass",
                        "Spark Master UI is accessible"
                    ))
                    
                    # Check for workers
                    if "Workers (" in response.text:
                        self.results.append(ServiceStatus(
                            "spark-workers", "pass",
                            "Spark workers are visible in master UI"
                        ))
                    else:
                        self.results.append(ServiceStatus(
                            "spark-workers", "warn",
                            "No Spark workers visible in master UI"
                        ))
                else:
                    self.results.append(ServiceStatus(
                        "spark-master", "warn",
                        "Spark Master UI format unexpected"
                    ))
            else:
                self.results.append(ServiceStatus(
                    "spark-master", "fail",
                    f"Spark Master UI returned status {response.status_code}"
                ))
                
        except Exception as e:
            self.results.append(ServiceStatus(
                "spark-cluster", "fail",
                f"Spark cluster check failed: {str(e)}"
            ))
    
    def check_airflow_health(self):
        """Check Airflow health."""
        print("\n🌊 Checking Airflow Health...")
        
        try:
            # Check Airflow health endpoint
            response = requests.get("http://localhost:8090/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                
                # Check metadatabase
                metadb_status = health_data.get('metadatabase', {}).get('status')
                scheduler_status = health_data.get('scheduler', {}).get('status')
                
                if metadb_status == 'healthy':
                    self.results.append(ServiceStatus(
                        "airflow-metadb", "pass",
                        "Airflow metadatabase is healthy"
                    ))
                else:
                    self.results.append(ServiceStatus(
                        "airflow-metadb", "warn",
                        f"Airflow metadatabase status: {metadb_status}"
                    ))
                
                if scheduler_status == 'healthy':
                    self.results.append(ServiceStatus(
                        "airflow-scheduler", "pass",
                        "Airflow scheduler is healthy"
                    ))
                else:
                    self.results.append(ServiceStatus(
                        "airflow-scheduler", "warn",
                        f"Airflow scheduler status: {scheduler_status}"
                    ))
                    
            else:
                self.results.append(ServiceStatus(
                    "airflow-health", "fail",
                    f"Airflow health endpoint returned status {response.status_code}"
                ))
                
        except Exception as e:
            self.results.append(ServiceStatus(
                "airflow-health", "fail",
                f"Airflow health check failed: {str(e)}"
            ))
    
    def check_service_integration(self):
        """Check service-to-service integration."""
        print("\n🔗 Checking Service Integration...")
        
        # Check if Grafana can reach Prometheus
        try:
            response = requests.get(
                "http://localhost:3000/api/datasources",
                timeout=10,
                auth=('admin', 'admin')
            )
            
            if response.status_code == 200:
                datasources = response.json()
                prometheus_ds = [ds for ds in datasources if ds.get('type') == 'prometheus']
                
                if prometheus_ds:
                    self.results.append(ServiceStatus(
                        "grafana-prometheus-integration", "pass",
                        "Grafana has Prometheus datasource configured"
                    ))
                else:
                    self.results.append(ServiceStatus(
                        "grafana-prometheus-integration", "warn",
                        "No Prometheus datasource found in Grafana"
                    ))
            else:
                self.results.append(ServiceStatus(
                    "grafana-prometheus-integration", "fail",
                    f"Grafana datasources API returned status {response.status_code}"
                ))
                
        except Exception as e:
            self.results.append(ServiceStatus(
                "grafana-prometheus-integration", "fail",
                f"Grafana-Prometheus integration check failed: {str(e)}"
            ))
    
    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 70)
        print("📋 VALIDATION SUMMARY")
        print("=" * 70)
        
        passed = [r for r in self.results if r.status == "pass"]
        warned = [r for r in self.results if r.status == "warn"]
        failed = [r for r in self.results if r.status == "fail"]
        
        print(f"✅ PASSED: {len(passed)}")
        print(f"⚠️  WARNINGS: {len(warned)}")
        print(f"❌ FAILED: {len(failed)}")
        print(f"📊 TOTAL CHECKS: {len(self.results)}")
        
        if failed:
            print("\n❌ FAILED CHECKS:")
            for result in failed:
                print(f"   • {result.name}: {result.message}")
        
        if warned:
            print("\n⚠️  WARNINGS:")
            for result in warned:
                print(f"   • {result.name}: {result.message}")
        
        # Service URLs
        print("\n🌐 SERVICE URLS:")
        print(f"   • Application: http://localhost:8000")
        print(f"   • Grafana: http://localhost:3000 (admin/admin)")
        print(f"   • Prometheus: http://localhost:9090")
        print(f"   • Kafka UI: http://localhost:8082")
        print(f"   • Spark Master: http://localhost:8080")
        print(f"   • Spark Worker: http://localhost:8081")
        print(f"   • Airflow: http://localhost:8090")
        
        overall_status = "✅ ALL SYSTEMS OPERATIONAL" if not failed else "❌ SYSTEM ISSUES DETECTED"
        print(f"\n🚀 OVERALL STATUS: {overall_status}")


def main():
    """Main function."""
    validator = ServiceValidator()
    success = validator.run_all_checks()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()