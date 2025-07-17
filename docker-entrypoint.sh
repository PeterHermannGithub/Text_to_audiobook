#!/bin/bash
# Docker entrypoint script for text-to-audiobook application
# Handles service initialization and graceful startup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Function to wait for a service to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=30
    local attempt=1

    log "Waiting for $service_name to be ready at $host:$port..."

    while [ $attempt -le $max_attempts ]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            log "$service_name is ready!"
            return 0
        fi
        
        log "Attempt $attempt/$max_attempts: $service_name not ready, waiting 5 seconds..."
        sleep 5
        ((attempt++))
    done

    warn "$service_name is not ready after $max_attempts attempts, continuing anyway..."
    return 1
}

# Function to check if a URL is accessible
check_url() {
    local url=$1
    local service_name=$2
    local max_attempts=10
    local attempt=1

    log "Checking $service_name at $url..."

    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            log "$service_name is accessible!"
            return 0
        fi
        
        log "Attempt $attempt/$max_attempts: $service_name not accessible, waiting 3 seconds..."
        sleep 3
        ((attempt++))
    done

    warn "$service_name is not accessible after $max_attempts attempts, continuing anyway..."
    return 1
}

# Create necessary directories
log "Creating necessary directories..."
mkdir -p /app/input /app/output /app/logs /app/config

# Set proper permissions
log "Setting directory permissions..."
chmod -R 755 /app/input /app/output /app/logs /app/config

# Initialize application
log "Initializing text-to-audiobook application..."

# Wait for core services if in distributed mode
if [ "${PROCESSING_MODE:-local}" = "distributed" ]; then
    log "Distributed processing mode enabled, waiting for services..."
    
    # Wait for Redis
    if [ -n "${REDIS_HOST}" ]; then
        wait_for_service "${REDIS_HOST}" "${REDIS_PORT:-6379}" "Redis"
    fi
    
    # Wait for Kafka
    if [ -n "${KAFKA_BOOTSTRAP_SERVERS}" ]; then
        kafka_host=$(echo "$KAFKA_BOOTSTRAP_SERVERS" | cut -d: -f1)
        kafka_port=$(echo "$KAFKA_BOOTSTRAP_SERVERS" | cut -d: -f2)
        wait_for_service "$kafka_host" "$kafka_port" "Kafka"
    fi
    
    # Wait for Spark Master
    if [ -n "${SPARK_MASTER}" ]; then
        spark_host=$(echo "$SPARK_MASTER" | sed 's/spark:\/\///' | cut -d: -f1)
        spark_port=$(echo "$SPARK_MASTER" | sed 's/spark:\/\///' | cut -d: -f2)
        wait_for_service "$spark_host" "$spark_port" "Spark Master"
    fi
    
    log "All distributed services are ready!"
else
    log "Local processing mode enabled, skipping service checks..."
fi

# Check external LLM service (Ollama)
if [ -n "${OLLAMA_URL}" ]; then
    check_url "${OLLAMA_URL}" "Ollama LLM Service"
fi

# Download spaCy model if not present
log "Checking spaCy model..."
if ! python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
    log "Downloading spaCy model..."
    python -m spacy download en_core_web_sm
fi

# Create default configuration if not exists
log "Setting up default configuration..."
if [ ! -f /app/config/settings.py ]; then
    log "Creating default settings.py..."
    cat > /app/config/settings.py << EOF
# Auto-generated settings for Docker environment
import os

# LLM Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434/api/generate")
DEFAULT_LOCAL_MODEL = os.getenv("DEFAULT_LOCAL_MODEL", "deepseek-v2:16b")
DEFAULT_LLM_ENGINE = os.getenv("DEFAULT_LLM_ENGINE", "local")

# Distributed Processing Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")

# Application Configuration
OUTPUT_DIR = "/app/output"
LOG_DIR = "/app/logs"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
PROCESSING_MODE = os.getenv("PROCESSING_MODE", "local")
EOF
fi

# Set Python path
export PYTHONPATH=/app:$PYTHONPATH

# Health check function
health_check() {
    log "Running health check..."
    python -c "
import sys
sys.path.insert(0, '/app')
try:
    from src.text_processing.text_extractor import TextExtractor
    from src.text_structurer import TextStructurer
    print('✓ Core modules imported successfully')
    
    # Check if services are accessible
    if '${PROCESSING_MODE:-local}' == 'distributed':
        print('✓ Distributed mode configured')
    else:
        print('✓ Local mode configured')
    
    print('✓ Health check passed')
except Exception as e:
    print(f'✗ Health check failed: {e}')
    sys.exit(1)
"
}

# Run health check
health_check

# Display startup information
log "=== Text-to-Audiobook Application Started ==="
log "Processing Mode: ${PROCESSING_MODE:-local}"
log "Python Path: $PYTHONPATH"
log "Working Directory: $(pwd)"
log "Available Input Files:"
if [ -d "/app/input" ]; then
    ls -la /app/input/ 2>/dev/null || echo "  No input files found"
else
    echo "  Input directory not found"
fi
log "================================================"

# Examples of how to use the application
if [ "${PROCESSING_MODE:-local}" = "distributed" ]; then
    log "Example Usage (Distributed Mode):"
    log "  python app.py input/sample.txt --distributed"
    log "  python app.py input/PrideandPrejudice.txt --distributed --debug-distributed"
else
    log "Example Usage (Local Mode):"
    log "  python app.py input/sample.txt"
    log "  python app.py input/PrideandPrejudice.txt --engine local"
fi

log "Application is ready! 🚀"

# Execute the main command
exec "$@"