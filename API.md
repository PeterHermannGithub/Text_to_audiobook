# Text-to-Audiobook API Documentation

Comprehensive API documentation for the text-to-audiobook system, covering CLI interface, health endpoints, monitoring APIs, and distributed processing interfaces.

## 📋 **Table of Contents**

1. [CLI Interface](#cli-interface)
2. [Health Check Endpoints](#health-check-endpoints)
3. [Monitoring APIs](#monitoring-apis)
4. [Distributed Processing APIs](#distributed-processing-apis)
5. [Configuration APIs](#configuration-apis)
6. [Kafka Message Schemas](#kafka-message-schemas)
7. [Error Handling](#error-handling)
8. [Authentication](#authentication)

---

## 🖥️ **CLI Interface**

### **Main Command**

```bash
python app.py [input_file] [OPTIONS]
```

### **Required Arguments**

| Argument | Type | Description |
|----------|------|-------------|
| `input_file` | `str` | Path to input document (.txt, .md, .pdf, .docx, .epub, .mobi) |

### **Optional Arguments**

#### **Input/Output Options**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--structured-input-file` | `str` | None | Path to pre-structured JSON file |
| `--output-filename` | `str` | None | Desired name for final output MP3 file |

#### **Audio Processing Options**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--skip-voice-casting` | `bool` | False | Skip voice casting phase |
| `--add-emotions` | `bool` | False | Add emotional annotations |
| `--voice-quality` | `str` | "premium" | Voice quality: "standard" or "premium" |

#### **LLM Engine Options**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--engine` | `str` | "local" | AI engine: "local" or "gcp" |
| `--model` | `str` | "deepseek-v2:16b" | Local model name |
| `--project_id` | `str` | None | Google Cloud project ID |
| `--location` | `str` | "us-central1" | Google Cloud location |

#### **Distributed Processing Options**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--distributed` | `bool` | False | Enable distributed processing |
| `--processing-mode` | `str` | "local" | Mode: "local", "distributed", "hybrid" |
| `--enable-kafka` | `bool` | False | Enable Kafka event processing |
| `--enable-spark` | `bool` | False | Enable Spark distributed validation |
| `--enable-caching` | `bool` | False | Enable Redis caching |
| `--enable-monitoring` | `bool` | False | Enable Prometheus metrics |

#### **Performance Options**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--workers` | `int` | 3 | Number of worker threads/processes |
| `--chunk-size` | `int` | 2000 | Size of text chunks for processing |
| `--quality-threshold` | `float` | 0.85 | Quality threshold for processing |

#### **Debugging Options**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--debug-llm` | `bool` | False | Enable detailed LLM logging |
| `--debug-distributed` | `bool` | False | Enable distributed processing debug |
| `--performance-monitoring` | `bool` | False | Enable performance metrics |

### **Usage Examples**

```bash
# Basic usage
python app.py input/book.pdf

# Full audiobook generation
python app.py input/book.epub --project_id "my-project" --output-filename "audiobook.mp3"

# Distributed processing
python app.py input/book.pdf --distributed --workers 4

# Debug mode
python app.py input/book.pdf --debug-llm --debug-distributed

# Custom LLM configuration
python app.py input/book.docx --engine gcp --model gemini-1.0-pro --project_id "project"

# Emotion annotation
python app.py input/book.pdf --add-emotions --voice-quality premium
```

### **Exit Codes**

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | File not found |
| 4 | Permission denied |
| 5 | LLM connection error |
| 6 | Distributed processing error |

---

## 🏥 **Health Check Endpoints**

### **Application Health Check**

```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-16T10:30:00Z",
  "version": "1.0.0",
  "components": {
    "database": "healthy",
    "llm_pool": "healthy",
    "kafka": "healthy",
    "redis": "healthy",
    "spark": "healthy"
  },
  "metrics": {
    "uptime": 3600,
    "requests_processed": 150,
    "error_rate": 0.02
  }
}
```

### **Readiness Check**

```http
GET /ready
```

**Response**:
```json
{
  "ready": true,
  "timestamp": "2025-01-16T10:30:00Z",
  "dependencies": {
    "kafka": "ready",
    "redis": "ready",
    "spark": "ready",
    "ollama": "ready"
  }
}
```

### **Liveness Check**

```http
GET /alive
```

**Response**:
```json
{
  "alive": true,
  "timestamp": "2025-01-16T10:30:00Z"
}
```

---

## 📊 **Monitoring APIs**

### **Prometheus Metrics**

```http
GET /metrics
```

**Response Format**: Prometheus text format

**Key Metrics**:

```prometheus
# Document processing metrics
textapp_documents_processed_total{status="success"} 1250
textapp_documents_processed_total{status="error"} 23
textapp_processing_duration_seconds_bucket{le="10"} 980
textapp_processing_duration_seconds_bucket{le="30"} 1200
textapp_processing_duration_seconds_bucket{le="60"} 1250

# LLM metrics
textapp_llm_requests_total{engine="local",model="deepseek-v2:16b"} 5600
textapp_llm_response_time_seconds_bucket{le="1"} 4200
textapp_llm_response_time_seconds_bucket{le="5"} 5500
textapp_llm_errors_total{engine="local",error_type="timeout"} 12

# Cache metrics
textapp_cache_hits_total{cache="redis"} 8900
textapp_cache_misses_total{cache="redis"} 2100
textapp_cache_hit_ratio 0.809

# Distributed processing metrics
textapp_kafka_messages_produced_total{topic="text_chunks"} 15600
textapp_kafka_messages_consumed_total{topic="text_chunks"} 15580
textapp_spark_jobs_completed_total{status="success"} 456
textapp_spark_jobs_completed_total{status="failed"} 8

# System metrics
textapp_memory_usage_bytes 2147483648
textapp_cpu_usage_percent 45.2
textapp_disk_usage_bytes 10737418240
```

### **Application Stats**

```http
GET /stats
```

**Response**:
```json
{
  "processing_stats": {
    "total_documents": 1250,
    "successful_documents": 1227,
    "failed_documents": 23,
    "average_processing_time": 25.3,
    "total_processing_time": 31625.0
  },
  "llm_stats": {
    "total_requests": 5600,
    "successful_requests": 5588,
    "failed_requests": 12,
    "average_response_time": 1.2,
    "engines": {
      "local": 5200,
      "gcp": 400
    }
  },
  "cache_stats": {
    "hits": 8900,
    "misses": 2100,
    "hit_ratio": 0.809,
    "evictions": 45
  },
  "distributed_stats": {
    "kafka_messages": 15600,
    "spark_jobs": 456,
    "active_workers": 4,
    "queue_size": 12
  }
}
```

### **System Information**

```http
GET /info
```

**Response**:
```json
{
  "application": {
    "name": "text-to-audiobook",
    "version": "1.0.0",
    "build": "2025-01-16T10:00:00Z",
    "environment": "production"
  },
  "system": {
    "python_version": "3.11.5",
    "platform": "linux",
    "cpu_count": 8,
    "memory_total": 17179869184,
    "memory_available": 8589934592
  },
  "dependencies": {
    "kafka": "7.4.0",
    "redis": "7.0.0",
    "spark": "3.4.0",
    "spacy": "3.7.2"
  },
  "configuration": {
    "chunk_size": 2500,
    "overlap_size": 500,
    "quality_threshold": 98.0,
    "max_refinement_iterations": 2
  }
}
```

---

## 🔄 **Distributed Processing APIs**

### **Job Submission**

```http
POST /api/v1/jobs
```

**Request Body**:
```json
{
  "input_file": "input/book.pdf",
  "processing_mode": "distributed",
  "options": {
    "chunk_size": 2000,
    "quality_threshold": 0.9,
    "workers": 4,
    "enable_caching": true
  }
}
```

**Response**:
```json
{
  "job_id": "job_12345",
  "status": "submitted",
  "created_at": "2025-01-16T10:30:00Z",
  "estimated_completion": "2025-01-16T10:32:00Z"
}
```

### **Job Status**

```http
GET /api/v1/jobs/{job_id}
```

**Response**:
```json
{
  "job_id": "job_12345",
  "status": "processing",
  "progress": 45.2,
  "created_at": "2025-01-16T10:30:00Z",
  "started_at": "2025-01-16T10:30:15Z",
  "estimated_completion": "2025-01-16T10:32:00Z",
  "stages": {
    "text_extraction": "completed",
    "text_structuring": "in_progress",
    "voice_casting": "pending",
    "audio_generation": "pending"
  },
  "metrics": {
    "segments_processed": 450,
    "total_segments": 1000,
    "processing_rate": 18.5,
    "quality_score": 96.8
  }
}
```

### **Job Results**

```http
GET /api/v1/jobs/{job_id}/results
```

**Response**:
```json
{
  "job_id": "job_12345",
  "status": "completed",
  "results": {
    "structured_text": "/output/book_structured.json",
    "voice_profiles": "/output/book_voice_profiles.json",
    "audiobook": "/output/book.mp3"
  },
  "quality_report": {
    "overall_score": 96.8,
    "segments_total": 1000,
    "segments_high_quality": 968,
    "segments_medium_quality": 32,
    "segments_low_quality": 0
  },
  "processing_stats": {
    "total_time": 120.5,
    "extraction_time": 15.2,
    "structuring_time": 90.3,
    "voice_casting_time": 8.5,
    "audio_generation_time": 6.5
  }
}
```

### **Job Cancellation**

```http
DELETE /api/v1/jobs/{job_id}
```

**Response**:
```json
{
  "job_id": "job_12345",
  "status": "cancelled",
  "cancelled_at": "2025-01-16T10:31:30Z",
  "reason": "user_requested"
}
```

---

## ⚙️ **Configuration APIs**

### **Get Configuration**

```http
GET /api/v1/config
```

**Response**:
```json
{
  "llm": {
    "default_engine": "local",
    "default_model": "deepseek-v2:16b",
    "ollama_url": "http://localhost:11434/api/generate",
    "gcp_model": "gemini-1.0-pro",
    "gcp_location": "us-central1"
  },
  "processing": {
    "chunk_size": 2500,
    "overlap_size": 500,
    "quality_threshold": 98.0,
    "max_refinement_iterations": 2
  },
  "distributed": {
    "kafka_bootstrap_servers": "localhost:9092",
    "redis_url": "redis://localhost:6379/0",
    "spark_master": "local[*]",
    "default_workers": 3
  },
  "monitoring": {
    "prometheus_port": 8000,
    "metrics_enabled": true,
    "health_check_interval": 30
  }
}
```

### **Update Configuration**

```http
PUT /api/v1/config
```

**Request Body**:
```json
{
  "processing": {
    "chunk_size": 3000,
    "quality_threshold": 95.0
  },
  "distributed": {
    "default_workers": 4
  }
}
```

**Response**:
```json
{
  "status": "updated",
  "timestamp": "2025-01-16T10:30:00Z",
  "changes": [
    "processing.chunk_size: 2500 -> 3000",
    "processing.quality_threshold: 98.0 -> 95.0",
    "distributed.default_workers: 3 -> 4"
  ]
}
```

---

## 📨 **Kafka Message Schemas**

### **Text Chunk Message**

**Topic**: `text_chunks`

```json
{
  "job_id": "job_12345",
  "chunk_id": "chunk_001",
  "chunk_data": {
    "text": "Chapter 1: The Beginning...",
    "metadata": {
      "page": 1,
      "chapter": 1,
      "word_count": 250
    }
  },
  "processing_options": {
    "chunk_size": 2000,
    "quality_threshold": 0.9
  },
  "timestamp": "2025-01-16T10:30:00Z"
}
```

### **LLM Request Message**

**Topic**: `llm_requests`

```json
{
  "request_id": "req_67890",
  "job_id": "job_12345",
  "chunk_id": "chunk_001",
  "engine": "local",
  "model": "deepseek-v2:16b",
  "prompt": "Classify the following text...",
  "parameters": {
    "temperature": 0.1,
    "max_tokens": 1000
  },
  "timestamp": "2025-01-16T10:30:00Z"
}
```

### **LLM Response Message**

**Topic**: `llm_responses`

```json
{
  "request_id": "req_67890",
  "job_id": "job_12345",
  "chunk_id": "chunk_001",
  "response": {
    "segments": [
      {
        "text": "Chapter 1: The Beginning",
        "speaker": "narrator",
        "confidence": 0.95
      }
    ]
  },
  "metadata": {
    "processing_time": 1.2,
    "tokens_used": 150,
    "quality_score": 96.8
  },
  "timestamp": "2025-01-16T10:30:01Z"
}
```

### **Quality Report Message**

**Topic**: `quality_reports`

```json
{
  "job_id": "job_12345",
  "chunk_id": "chunk_001",
  "quality_metrics": {
    "overall_score": 96.8,
    "speaker_consistency": 98.2,
    "text_preservation": 99.1,
    "dialogue_detection": 95.5
  },
  "issues": [
    {
      "type": "ambiguous_speaker",
      "segment": "Well, I suppose...",
      "confidence": 0.6
    }
  ],
  "timestamp": "2025-01-16T10:30:02Z"
}
```

---

## ❌ **Error Handling**

### **Standard Error Response**

```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Input file not found: input/book.pdf",
    "details": {
      "file_path": "input/book.pdf",
      "expected_formats": [".txt", ".md", ".pdf", ".docx", ".epub", ".mobi"]
    },
    "timestamp": "2025-01-16T10:30:00Z",
    "request_id": "req_12345"
  }
}
```

### **Error Codes**

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_INPUT` | 400 | Invalid input parameters |
| `FILE_NOT_FOUND` | 404 | Input file not found |
| `UNSUPPORTED_FORMAT` | 400 | Unsupported file format |
| `LLM_CONNECTION_ERROR` | 502 | LLM service unavailable |
| `PROCESSING_TIMEOUT` | 408 | Processing timeout |
| `INSUFFICIENT_RESOURCES` | 503 | Insufficient system resources |
| `QUALITY_THRESHOLD_NOT_MET` | 422 | Quality threshold not met |
| `DISTRIBUTED_PROCESSING_ERROR` | 500 | Distributed processing failure |
| `AUTHENTICATION_ERROR` | 401 | Authentication failed |
| `AUTHORIZATION_ERROR` | 403 | Authorization failed |

---

## 🔐 **Authentication**

### **API Key Authentication**

```http
GET /api/v1/jobs
Authorization: Bearer your-api-key-here
```

### **Health Check Endpoints**

Health check endpoints (`/health`, `/ready`, `/alive`) are accessible without authentication.

### **Monitoring Endpoints**

Monitoring endpoints (`/metrics`, `/stats`, `/info`) require authentication in production environments.

---

## 🔧 **SDK and Integration**

### **Python SDK Example**

```python
from texttoaudiobook import TextToAudiobookClient

# Initialize client
client = TextToAudiobookClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Submit job
job = client.submit_job(
    input_file="input/book.pdf",
    processing_mode="distributed",
    options={
        "chunk_size": 2000,
        "quality_threshold": 0.9
    }
)

# Monitor progress
while job.status != "completed":
    job.refresh()
    print(f"Progress: {job.progress}%")
    time.sleep(10)

# Get results
results = job.get_results()
print(f"Audiobook: {results.audiobook}")
```

### **cURL Examples**

```bash
# Submit job
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "input_file": "input/book.pdf",
    "processing_mode": "distributed",
    "options": {
      "chunk_size": 2000,
      "quality_threshold": 0.9
    }
  }'

# Check job status
curl -X GET http://localhost:8000/api/v1/jobs/job_12345 \
  -H "Authorization: Bearer your-api-key"

# Get health status
curl -X GET http://localhost:8000/health
```

---

**Note**: This API documentation covers all interfaces and endpoints for the text-to-audiobook system. For implementation details, refer to [CLAUDE.md](CLAUDE.md) and [README.md](README.md).