# System Testing Guide - Text-to-Audiobook Infrastructure

This guide provides step-by-step instructions for testing the complete Text-to-Audiobook system with all 6 technologies: Docker, Grafana, Prometheus, Kafka, Spark, and Airflow.

## 🚀 Quick Start

### 1. Start All Services
```bash
# Using the service manager (recommended)
./service_manager.sh start

# Or using docker-compose directly
docker-compose up -d
```

### 2. Validate System Health
```bash
# Run comprehensive validation
./service_manager.sh validate

# Or run validation script directly
python3 validate_services.py
```

### 3. Access Service UIs
- **Application**: http://localhost:8000
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Kafka UI**: http://localhost:8082
- **Spark Master**: http://localhost:8080
- **Spark Worker**: http://localhost:8081
- **Airflow**: http://localhost:8090

## 📋 Detailed Testing Steps

### Step 1: Infrastructure Validation

#### Docker Services Check
```bash
# Check all containers are running
docker ps

# Expected containers:
# - text_to_audiobook_app
# - text_to_audiobook_redis
# - text_to_audiobook_kafka
# - text_to_audiobook_zookeeper
# - text_to_audiobook_spark_master
# - text_to_audiobook_spark_worker
# - text_to_audiobook_prometheus
# - text_to_audiobook_grafana
# - text_to_audiobook_kafka_ui
```

#### Service Health Checks
```bash
# Check service health (uses health checks we added)
docker-compose ps

# All services should show "healthy" status
```

### Step 2: Monitoring Stack Validation

#### Prometheus Targets
1. Go to http://localhost:9090/targets
2. Verify all targets are "UP":
   - prometheus (self-monitoring)
   - text-to-audiobook-app
   - redis
   - kafka
   - spark-master
   - spark-worker

#### Grafana Dashboards
1. Go to http://localhost:3000 (admin/admin)
2. Navigate to Dashboards
3. Verify all 5 dashboards are available:
   - **System Overview** (uid: system-overview)
   - **Kafka Monitoring** (uid: kafka-monitoring)
   - **Spark Monitoring** (uid: spark-monitoring)
   - **LLM Monitoring** (uid: llm-monitoring)
   - **Airflow & Pipeline Monitoring** (uid: airflow-monitoring)

#### Grafana-Prometheus Integration
1. Go to Configuration > Data Sources in Grafana
2. Verify Prometheus datasource is configured and working
3. Test connection should be successful

### Step 3: Message Queue Validation

#### Kafka Cluster Health
1. Go to http://localhost:8082 (Kafka UI)
2. Verify cluster "local" is connected
3. Check brokers are online
4. Topics will be auto-created on first use

#### Topic Creation Test
```bash
# Test topic creation via kafka container
docker exec text_to_audiobook_kafka kafka-topics --create \
  --topic test-topic \
  --bootstrap-server localhost:9092 \
  --partitions 1 \
  --replication-factor 1

# Verify in Kafka UI
# Topic should appear in http://localhost:8082
```

### Step 4: Distributed Processing Validation

#### Spark Cluster Status
1. Go to http://localhost:8080 (Spark Master UI)
2. Verify:
   - Status: ALIVE
   - Workers: 1 (or more)
   - Memory: Available
   - Cores: Available

#### Spark Worker Status
1. Go to http://localhost:8081 (Spark Worker UI)
2. Verify:
   - Master: Connected
   - Status: ALIVE
   - Executors: Available

#### Spark-Kafka Integration Test
```bash
# This will be tested during application processing
# Spark jobs should appear in the Spark UI when processing starts
```

### Step 5: Workflow Orchestration Validation

#### Airflow Web UI
1. Go to http://localhost:8090
2. Login with default credentials
3. Verify DAGs are loaded:
   - text_to_audiobook_processing

#### Airflow Health Check
```bash
# Check Airflow health endpoint
curl http://localhost:8090/health

# Expected response: {"metadatabase": {"status": "healthy"}, ...}
```

#### DAG Validation
1. In Airflow UI, check the text_to_audiobook_processing DAG
2. Verify all tasks are visible:
   - validate_input_file
   - extract_text
   - process_text_with_spark
   - validate_quality
   - refine_segments
   - format_output

### Step 6: End-to-End Integration Test

#### Application Processing Test
```bash
# Place a test file in input directory
echo "Sample text for processing" > input/test.txt

# Run the application
docker exec text_to_audiobook_app python app.py input/test.txt

# Monitor in various UIs:
# - Spark UI: Should show jobs
# - Kafka UI: Should show messages
# - Prometheus: Should show metrics
# - Grafana: Should show activity
```

#### Monitoring During Processing
1. **Grafana**: Watch System Overview dashboard for activity
2. **Prometheus**: Check metrics are being collected
3. **Spark UI**: Monitor job execution
4. **Kafka UI**: Watch message flow
5. **Airflow**: Check DAG runs (if triggered)

## 🔧 Troubleshooting

### Common Issues

#### Services Not Starting
```bash
# Check logs
./service_manager.sh logs

# Check specific service
./service_manager.sh logs kafka
```

#### Port Conflicts
- **Fixed**: Airflow now uses port 8090 (was 8080)
- **Spark Master UI**: 8080
- **Kafka UI**: 8082
- **Redis Commander** (dev): 8083

#### Memory Issues
```bash
# Check resource usage
docker stats

# Reduce memory if needed by editing docker-compose.yml
```

#### Network Connectivity
```bash
# Test network connectivity between services
docker exec text_to_audiobook_app ping redis
docker exec text_to_audiobook_app ping kafka
docker exec text_to_audiobook_app ping spark
```

### Validation Script Failures

If `validate_services.py` reports failures:

1. **Docker Services**: Check if all containers are running
2. **Endpoints**: Verify services are responding on expected ports
3. **Prometheus Targets**: Check Prometheus configuration
4. **Grafana Dashboards**: Verify dashboard files are loaded
5. **Kafka Topics**: May be auto-created on first use
6. **Spark Cluster**: Check master-worker connectivity

## 📊 Success Criteria

### System is considered working when:

✅ **Infrastructure**
- All Docker containers running and healthy
- No port conflicts
- Health checks passing

✅ **Monitoring**
- Prometheus collecting metrics from all targets
- Grafana dashboards accessible with data
- Service integration between Grafana and Prometheus

✅ **Message Queue**
- Kafka cluster operational
- Zookeeper coordinating properly
- Topics can be created and accessed

✅ **Distributed Processing**
- Spark master and workers connected
- Cluster resources available
- Jobs can be submitted

✅ **Workflow Orchestration**
- Airflow webserver responsive
- DAGs loaded and parseable
- Database connectivity working

✅ **Integration**
- Services can communicate with each other
- End-to-end processing possible
- Monitoring captures all activity

## 🎯 Performance Expectations

- **Startup Time**: ~30-60 seconds for all services
- **Memory Usage**: ~4-6 GB total for all services
- **Health Check Response**: <5 seconds
- **Service Response Time**: <2 seconds for UI access
- **Processing Latency**: Varies by workload

## 📈 Monitoring Best Practices

1. **Regular Validation**: Run validation script weekly
2. **Resource Monitoring**: Monitor memory and CPU usage
3. **Log Monitoring**: Check logs for errors regularly
4. **Performance Trending**: Use Grafana for performance analysis
5. **Backup Strategy**: Regular backup of Grafana dashboards and Prometheus data

## 🚀 Production Considerations

For production deployment:

1. **Use docker/docker-compose.prod.yml** for production configuration
2. **Configure proper secrets** (not admin/admin for Grafana)
3. **Set up SSL/TLS** for external access
4. **Configure backup strategies** for data persistence
5. **Set up log aggregation** (ELK stack is included in prod config)
6. **Monitor resource usage** and scale as needed
7. **Configure alerts** in Grafana for critical issues

---

This comprehensive testing ensures all 6 technologies (Docker, Grafana, Prometheus, Kafka, Spark, Airflow) work together seamlessly in the Text-to-Audiobook system.