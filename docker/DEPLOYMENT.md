# Text-to-Audiobook Docker Deployment Guide

This guide provides comprehensive instructions for deploying the text-to-audiobook system using Docker containers in development and production environments.

## 📁 **Docker Configuration Files**

```
docker/
├── Dockerfile                 # Production application container
├── Dockerfile.dev             # Development container with debugging tools
├── docker-compose.yml         # Full distributed system deployment
├── docker-compose.dev.yml     # Development environment
├── docker-compose.prod.yml    # Production environment with optimization
├── prometheus.yml             # Prometheus monitoring configuration
└── DEPLOYMENT.md              # This file
```

## 🚀 **Quick Start**

### **Development Environment**

```bash
# Start development environment
docker-compose -f docker/docker-compose.dev.yml up -d

# Process a document
docker-compose -f docker/docker-compose.dev.yml exec app python app.py input/book.pdf

# Run tests
docker-compose -f docker/docker-compose.dev.yml exec app pytest tests/

# Access Jupyter notebook (http://localhost:8888)
docker-compose -f docker/docker-compose.dev.yml exec jupyter bash

# View logs
docker-compose -f docker/docker-compose.dev.yml logs -f app
```

### **Production Environment**

```bash
# Start production environment
docker-compose -f docker/docker-compose.prod.yml up -d

# Process a document
docker-compose -f docker/docker-compose.prod.yml exec app python app.py input/book.pdf --distributed

# Scale workers
docker-compose -f docker/docker-compose.prod.yml up -d --scale spark-worker=3

# Monitor system health
curl http://localhost:8000/health
```

## 🏗️ **Architecture Overview**

### **Services Included**

| Service | Purpose | Port | Environment |
|---------|---------|------|-------------|
| **app** | Main application | 8000 | All |
| **nginx** | Load balancer | 80, 443 | Production |
| **kafka** | Event streaming | 9092 | All |
| **zookeeper** | Kafka coordination | 2181 | All |
| **redis** | Caching | 6379 | All |
| **spark** | Distributed processing | 7077, 8080 | All |
| **spark-worker** | Spark workers | 8081 | All |
| **prometheus** | Metrics collection | 9090 | All |
| **grafana** | Monitoring dashboards | 3000 | All |
| **jupyter** | Development notebooks | 8888 | Development |
| **elasticsearch** | Log storage | 9200 | Production |
| **kibana** | Log visualization | 5601 | Production |

### **Network Architecture**

```
Internet → nginx (80/443) → app (8000)
                              ↓
    kafka (9092) ← → redis (6379) ← → spark (7077)
         ↓                              ↓
   zookeeper (2181)              spark-worker (8081)
         ↓                              ↓
   prometheus (9090) → grafana (3000)
```

## 🛠️ **Deployment Options**

### **1. Development Deployment**

**Use Case**: Local development, testing, debugging

**Command**:
```bash
docker-compose -f docker/docker-compose.dev.yml up -d
```

**Features**:
- Hot reload with volume mounting
- Debugging tools included
- Jupyter notebook access
- Reduced resource limits
- Development-friendly logging

**Services**: app, worker, kafka, zookeeper, redis, spark, prometheus, grafana, jupyter

### **2. Standard Deployment**

**Use Case**: Testing distributed features, demo environment

**Command**:
```bash
docker-compose -f docker/docker-compose.yml up -d
```

**Features**:
- Full distributed system
- All production services
- Balanced resource allocation
- Comprehensive monitoring

**Services**: app, kafka, zookeeper, redis, spark, prometheus, grafana, airflow

### **3. Production Deployment**

**Use Case**: Production environment, high availability

**Command**:
```bash
docker-compose -f docker/docker-compose.prod.yml up -d
```

**Features**:
- Optimized resource limits
- Load balancing with nginx
- ELK stack for logging
- Enhanced security
- High availability configuration

**Services**: app, nginx, kafka, zookeeper, redis, spark, prometheus, grafana, elasticsearch, kibana

## ⚙️ **Configuration**

### **Environment Variables**

Create a `.env` file in the project root:

```bash
# Application Configuration
LOG_LEVEL=INFO
PYTHONPATH=/app

# Google Cloud (optional)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
GCP_PROJECT_ID=your-project-id

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_PREFIX=text_to_audiobook

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your_secure_password

# Spark Configuration
SPARK_MASTER=local[*]
SPARK_EXECUTOR_MEMORY=2g
SPARK_DRIVER_MEMORY=1g

# Monitoring Configuration
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_PASSWORD=your_secure_password

# Security (Production)
NGINX_SSL_CERT_PATH=/path/to/cert.pem
NGINX_SSL_KEY_PATH=/path/to/key.pem
```

### **Volume Mounting**

```yaml
volumes:
  - ./input:/app/input          # Input documents
  - ./output:/app/output        # Generated outputs
  - ./logs:/app/logs            # Application logs
  - ./config:/app/config        # Configuration files
```

## 🔧 **Advanced Usage**

### **Custom Resource Limits**

```yaml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2'
    reservations:
      memory: 2G
      cpus: '1'
```

### **Horizontal Scaling**

```bash
# Scale Spark workers
docker-compose -f docker/docker-compose.prod.yml up -d --scale spark-worker=5

# Scale application instances
docker-compose -f docker/docker-compose.prod.yml up -d --scale app=3
```

### **Health Checks**

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

## 📊 **Monitoring and Observability**

### **Prometheus Metrics**

Access Prometheus at `http://localhost:9090`

**Key Metrics**:
- `textapp_documents_processed_total`
- `textapp_processing_duration_seconds`
- `textapp_llm_requests_total`
- `textapp_cache_hits_total`
- `textapp_error_rate`

### **Grafana Dashboards**

Access Grafana at `http://localhost:3000` (admin/admin)

**Pre-configured Dashboards**:
- Application Performance
- System Resources
- Kafka Metrics
- Redis Performance
- Spark Cluster Status

### **Log Aggregation**

**Development**: View logs with `docker-compose logs -f <service>`

**Production**: Access Kibana at `http://localhost:5601` for centralized log analysis

## 🔒 **Security Considerations**

### **Production Security**

1. **Environment Variables**: Use secrets management (Docker Secrets, Kubernetes Secrets)
2. **Network Security**: Implement proper firewall rules
3. **SSL/TLS**: Configure HTTPS with valid certificates
4. **Authentication**: Enable authentication for all services
5. **Resource Limits**: Set appropriate memory and CPU limits

### **Secure Configuration Example**

```yaml
# Use secrets instead of environment variables
secrets:
  redis_password:
    file: ./secrets/redis_password.txt
  grafana_password:
    file: ./secrets/grafana_password.txt

services:
  redis:
    secrets:
      - redis_password
    command: redis-server --requirepass_file /run/secrets/redis_password
```

## 🚨 **Troubleshooting**

### **Common Issues**

**Container Won't Start**:
```bash
# Check logs
docker-compose logs <service_name>

# Check resource usage
docker stats

# Verify network connectivity
docker-compose exec app ping redis
```

**Performance Issues**:
```bash
# Monitor resource usage
docker-compose exec app htop

# Check Kafka lag
docker-compose exec kafka kafka-consumer-groups --bootstrap-server localhost:9092 --describe --all-groups

# Monitor Spark jobs
# Access Spark UI at http://localhost:8080
```

**Memory Issues**:
```bash
# Increase memory limits in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 8G
```

### **Debugging Commands**

```bash
# Enter container shell
docker-compose exec app bash

# View application logs
docker-compose logs -f app

# Check service health
docker-compose ps

# Restart specific service
docker-compose restart <service_name>

# Clean up resources
docker-compose down -v
docker system prune -a
```

## 🔄 **Deployment Workflow**

### **Development to Production**

1. **Test locally**: `docker-compose -f docker/docker-compose.dev.yml up`
2. **Run tests**: `docker-compose exec app pytest tests/`
3. **Build production image**: `docker-compose -f docker/docker-compose.prod.yml build`
4. **Deploy to staging**: `docker-compose -f docker/docker-compose.prod.yml up -d`
5. **Monitor metrics**: Check Grafana dashboards
6. **Deploy to production**: Use CI/CD pipeline

### **Backup and Recovery**

```bash
# Backup volumes
docker run --rm -v text_to_audiobook_redis-data:/data -v $(pwd):/backup alpine tar czf /backup/redis-backup.tar.gz /data

# Restore volumes
docker run --rm -v text_to_audiobook_redis-data:/data -v $(pwd):/backup alpine tar xzf /backup/redis-backup.tar.gz -C /
```

## 📈 **Performance Optimization**

### **Resource Tuning**

```yaml
# Kafka optimization
environment:
  KAFKA_HEAP_OPTS: "-Xmx2G -Xms2G"
  KAFKA_NUM_PARTITIONS: 8
  KAFKA_DEFAULT_REPLICATION_FACTOR: 1

# Spark optimization
environment:
  SPARK_WORKER_MEMORY: 4G
  SPARK_EXECUTOR_MEMORY: 2G
  SPARK_DRIVER_MEMORY: 1G
```

### **Caching Strategy**

```yaml
# Redis configuration
command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
```

## 🎯 **Best Practices**

1. **Use multi-stage builds** for smaller production images
2. **Implement health checks** for all services
3. **Set resource limits** to prevent resource exhaustion
4. **Use secrets management** for sensitive configuration
5. **Monitor application metrics** continuously
6. **Implement proper logging** with structured formats
7. **Use volume mounts** for persistent data
8. **Regular backups** of stateful services
9. **Security scanning** of container images
10. **Graceful shutdown** handling

---

**Note**: This deployment guide provides comprehensive Docker configurations for the text-to-audiobook system. For additional support, refer to the main [README.md](../README.md) and [CLAUDE.md](../CLAUDE.md) documentation.