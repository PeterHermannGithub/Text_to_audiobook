# config/settings.py
import os

# LLM Configuration
# Use environment variable for Docker compatibility, fallback to localhost
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_URL = f"http://{OLLAMA_HOST}:11434/api/generate"
DEFAULT_LOCAL_MODEL = "deepseek-v2:16b"
GCP_LLM_MODEL = "gemini-1.0-pro"
DEFAULT_LLM_ENGINE = "local"
GCP_LOCATION = "us-central1"

# Multi-Model Load Balancing Configuration (Phase 3.2: Multi-Model Load Balancing)
MULTI_MODEL_ENABLED = True  # Enable multi-model load balancing system
MULTI_MODEL_FALLBACK_ENABLED = True  # Enable fallback between models on failure

# Model Capability Definitions - Used for intelligent routing decisions
MODEL_CAPABILITIES = {
    # Local Models (Ollama)
    "deepseek-v2:16b": {
        "engine": "local",
        "provider": "ollama",
        "speed_tier": "fast",        # fast, medium, slow
        "quality_tier": "high",      # high, medium, basic
        "cost_tier": "free",         # free, low, medium, high
        "max_context_length": 32768,
        "supports_json": True,
        "optimal_use_cases": ["complex_reasoning", "coding", "analysis"],
        "cost_per_1k_tokens": 0.0,   # Free for local models
        "avg_response_time_ms": 1500,
        "reliability_score": 0.95
    },
    "llama3:8b": {
        "engine": "local",
        "provider": "ollama", 
        "speed_tier": "fast",
        "quality_tier": "medium",
        "cost_tier": "free",
        "max_context_length": 8192,
        "supports_json": True,
        "optimal_use_cases": ["general_chat", "simple_reasoning"],
        "cost_per_1k_tokens": 0.0,
        "avg_response_time_ms": 800,
        "reliability_score": 0.92
    },
    "mistral:7b": {
        "engine": "local",
        "provider": "ollama",
        "speed_tier": "medium", 
        "quality_tier": "medium",
        "cost_tier": "free",
        "max_context_length": 32768,
        "supports_json": True,
        "optimal_use_cases": ["classification", "extraction"],
        "cost_per_1k_tokens": 0.0,
        "avg_response_time_ms": 1200,
        "reliability_score": 0.90
    },
    # Cloud Models (Google Cloud)
    "gemini-1.0-pro": {
        "engine": "gcp",
        "provider": "google_cloud",
        "speed_tier": "medium",
        "quality_tier": "high", 
        "cost_tier": "medium",
        "max_context_length": 30720,
        "supports_json": True,
        "optimal_use_cases": ["complex_reasoning", "analysis", "creative_writing"],
        "cost_per_1k_tokens": 0.002,
        "avg_response_time_ms": 2000,
        "reliability_score": 0.98
    },
    "gemini-1.5-flash": {
        "engine": "gcp",
        "provider": "google_cloud",
        "speed_tier": "fast",
        "quality_tier": "medium",
        "cost_tier": "low",
        "max_context_length": 1000000,
        "supports_json": True,
        "optimal_use_cases": ["simple_classification", "batch_processing"],
        "cost_per_1k_tokens": 0.0005,
        "avg_response_time_ms": 800,
        "reliability_score": 0.96
    }
}

# Multi-Model Pool Configurations - Define different pools for different use cases
MULTI_MODEL_POOLS = {
    # Primary pool for high-quality reasoning tasks
    "primary": {
        "models": ["deepseek-v2:16b", "gemini-1.0-pro"],
        "routing_strategy": "quality_first",     # quality_first, speed_first, cost_first, balanced
        "failover_enabled": True,
        "max_instances_per_model": 3,
        "health_check_interval": 30,
        "load_balancing_algorithm": "weighted_round_robin",  # round_robin, weighted_round_robin, least_connections
        "cost_budget_per_hour": 10.0,  # USD budget limit per hour for cloud models
        "preferred_local_ratio": 0.8   # Prefer local models 80% of the time for cost optimization
    },
    
    # Fast pool for simple classification tasks  
    "fast": {
        "models": ["llama3:8b", "mistral:7b", "gemini-1.5-flash"],
        "routing_strategy": "speed_first",
        "failover_enabled": True,
        "max_instances_per_model": 2,
        "health_check_interval": 60,
        "load_balancing_algorithm": "round_robin", 
        "cost_budget_per_hour": 5.0,
        "preferred_local_ratio": 0.9   # Prefer local models 90% of the time for speed/cost
    },
    
    # Fallback pool for reliability
    "fallback": {
        "models": ["mistral:7b", "gemini-1.0-pro"],
        "routing_strategy": "balanced",
        "failover_enabled": False,  # This is the fallback pool
        "max_instances_per_model": 1,
        "health_check_interval": 120,
        "load_balancing_algorithm": "least_connections",
        "cost_budget_per_hour": 15.0,
        "preferred_local_ratio": 0.5   # Balanced local/cloud for reliability
    }
}

# Request Complexity Classification - Used for intelligent pool selection
REQUEST_COMPLEXITY_ROUTING = {
    "simple": {
        "preferred_pools": ["fast", "primary"],
        "max_context_length": 2000,
        "timeout_seconds": 30,
        "examples": ["single_line_classification", "yes_no_questions", "simple_extraction"]
    },
    "medium": {
        "preferred_pools": ["primary", "fast"], 
        "max_context_length": 8000,
        "timeout_seconds": 60,
        "examples": ["multi_line_classification", "speaker_attribution", "content_analysis"]
    },
    "complex": {
        "preferred_pools": ["primary", "fallback"],
        "max_context_length": 20000, 
        "timeout_seconds": 120,
        "examples": ["deep_reasoning", "complex_analysis", "creative_tasks"]
    },
    "batch": {
        "preferred_pools": ["fast", "primary"],
        "max_context_length": 50000,
        "timeout_seconds": 300,
        "examples": ["batch_classification", "bulk_processing", "document_analysis"]
    }
}

# Model-Specific Performance Tuning
MODEL_PERFORMANCE_CONFIG = {
    # Timeout configurations per model type
    "timeouts": {
        "local_fast": {"connection": 10, "read": 60, "total": 120},
        "local_medium": {"connection": 15, "read": 90, "total": 180}, 
        "cloud_fast": {"connection": 20, "read": 120, "total": 240},
        "cloud_medium": {"connection": 30, "read": 180, "total": 300}
    },
    
    # Retry configurations per model type
    "retries": {
        "local": {"max_attempts": 3, "backoff_factor": 1.0, "max_delay": 10},
        "cloud": {"max_attempts": 5, "backoff_factor": 2.0, "max_delay": 30}
    },
    
    # Load thresholds for different model tiers
    "load_thresholds": {
        "fast": {"max_concurrent": 10, "queue_limit": 20},
        "medium": {"max_concurrent": 6, "queue_limit": 12}, 
        "slow": {"max_concurrent": 3, "queue_limit": 6}
    }
}

# Cost Management Configuration
COST_MANAGEMENT = {
    "enabled": True,
    "daily_budget_usd": 50.0,       # Maximum daily spend on cloud models
    "cost_alerts": {
        "warning_threshold": 0.7,    # Alert at 70% of budget
        "critical_threshold": 0.9    # Critical alert at 90% of budget
    },
    "cost_optimization": {
        "prefer_local_under_load": True,   # Switch to local models when cloud usage is high
        "dynamic_routing_enabled": True,   # Enable cost-aware routing decisions
        "budget_tracking_window": "daily"  # daily, hourly, weekly
    }
}

# Chunking Parameters
CHUNK_SIZE = 2500
OVERLAP_SIZE = 500

# Parallel Processing Configuration
PARALLEL_PROCESSING_ENABLED = True  # Enable parallel window processing
MAX_PARALLEL_WORKERS = 4  # Maximum number of parallel workers for window processing
PARALLEL_BATCH_SIZE = 8  # Number of windows to process in each batch

# Batch LLM Processing Configuration
BATCH_LLM_PROCESSING_ENABLED = True  # Enable batch LLM processing for multiple segments
MAX_BATCH_SIZE = 3  # Maximum number of segments to process in a single LLM request
MIN_BATCH_SIZE = 2  # Minimum number of segments required for batch processing
BATCH_MAX_TOTAL_LINES = 20  # Maximum total lines across all segments in a batch

# Async Processing Configuration
ASYNC_PROCESSING_ENABLED = True  # Enable async/await processing pipeline
ASYNC_SEMAPHORE_LIMIT = 8  # Maximum concurrent async operations
ASYNC_TIMEOUT_SECONDS = 300  # Timeout for async operations (5 minutes)
ASYNC_RETRY_ATTEMPTS = 3  # Number of retry attempts for failed async operations

# LLM Response Caching Configuration
LLM_CACHE_ENABLED = True  # Enable LLM response caching with hash-based keys
LLM_CACHE_MAX_SIZE = 10000  # Maximum number of cached responses
LLM_CACHE_TTL_SECONDS = 86400  # Cache time-to-live (24 hours)
LLM_CACHE_HASH_LENGTH = 16  # Length of hash keys (hex characters)
LLM_CACHE_INCLUDE_METADATA = True  # Include metadata in cache key generation
LLM_CACHE_COMPRESS_RESPONSES = True  # Compress cached responses to save memory

# Rule-based Attribution Caching Configuration
RULE_CACHE_ENABLED = True  # Enable rule-based attribution caching
RULE_CACHE_MAX_SIZE = 50000  # Maximum number of cached rule-based results
RULE_CACHE_TTL_SECONDS = 86400  # Cache time-to-live (24 hours)
RULE_CACHE_LINE_CACHING = True  # Enable individual line result caching
RULE_CACHE_BATCH_CACHING = True  # Enable batch processing result caching
RULE_CACHE_FUZZY_CACHING = True  # Enable fuzzy matching result caching
RULE_CACHE_PATTERN_CACHING = True  # Enable pattern matching result caching

# Preprocessing and spaCy NLP Caching Configuration
PREPROCESSING_CACHE_ENABLED = True  # Enable preprocessing result caching
PREPROCESSING_CACHE_MAX_SIZE = 1000  # Maximum number of cached preprocessing results
PREPROCESSING_CACHE_TTL_SECONDS = 86400  # Cache time-to-live (24 hours)
SPACY_CACHE_ENABLED = True  # Enable spaCy document caching
SPACY_CACHE_SERIALIZE_DOCS = True  # Serialize spaCy doc objects for caching
SPACY_CACHE_COMPRESS_DOCS = True  # Compress serialized spaCy docs for memory efficiency
CHARACTER_PROFILE_CACHE_ENABLED = True  # Enable character profile caching
POV_ANALYSIS_CACHE_ENABLED = True  # Enable POV analysis caching
SCENE_BREAK_CACHE_ENABLED = True  # Enable scene break detection caching
DOCUMENT_STRUCTURE_CACHE_ENABLED = True  # Enable document structure analysis caching

# Output Paths
OUTPUT_DIR = "output"
LOG_DIR = "logs"

# Prompt Templates (can be expanded later)
# PROMPT_STRUCTURING = "..."

# Validation Thresholds
REFINEMENT_QUALITY_THRESHOLD = 98.0
SIMILARITY_THRESHOLD = 95

# Refinement Iterations
MAX_REFINEMENT_ITERATIONS = 2

# Other settings
SPACY_MODEL = "en_core_web_sm"

# Logging Configuration
LOG_LEVEL = "INFO"  # Changed from DEBUG to reduce terminal noise
CONSOLE_LOG_LEVEL = "INFO"  # Level for console output
FILE_LOG_LEVEL = "DEBUG"    # Level for file output (more detailed)

# LLM Debug Logging Configuration
LLM_DEBUG_LOGGING = False   # Enable detailed LLM interaction logging
LLM_DEBUG_LOG_FILE = "llm_debug.log"  # Separate debug log file for LLM interactions
LLM_DEBUG_TRUNCATE_LENGTH = 0  # Maximum characters to log for large prompts/responses (0 = no limit)
LLM_DEBUG_INCLUDE_CONTEXT = True  # Include context information (engine, model, timing) in debug logs
LLM_DEBUG_LOG_PROCESSING_STEPS = True  # Log intermediate processing steps (sanitization, parsing)

# POV Analysis Configuration (Ultrathink Architecture)
POV_SAMPLE_SIZE = 2000  # Number of words to analyze for POV detection
POV_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for POV classification (0.0-1.0)
POV_ENABLE_NARRATOR_DISCOVERY = True  # Enable automatic narrator name discovery in first-person texts
POV_FALLBACK_NARRATOR_ID = "MainCharacter"  # Fallback identifier when narrator name cannot be determined
POV_PRONOUN_WEIGHT_THRESHOLD = 1.5  # Ratio threshold for pronoun-based POV classification

# Sliding Window Configuration (Ultrathink Architecture)
SLIDING_WINDOW_ENABLED = True  # Enable sliding window processing (False = legacy chunking)
CONTEXT_WINDOW_SIZE = 50  # Number of context lines to provide to LLM
TASK_WINDOW_SIZE = 15     # Number of lines to classify per iteration
WINDOW_OVERLAP_RATIO = 0.3  # Overlap ratio between sliding windows (0.0-1.0)
ADAPTIVE_WINDOW_SIZING = True  # Enable adaptive window sizing based on content complexity

# HTTP Connection Pool Configuration (Phase 3.1: Connection Pooling)
HTTP_POOL_ENABLED = True  # Enable HTTP connection pooling for LLM requests
HTTP_POOL_MAX_CONNECTIONS = 100  # Maximum connections per pool
HTTP_POOL_SIZE = 20  # Connection pool size per host
HTTP_POOL_BLOCK = False  # Block when pool is full (False = create new connection)

# HTTP Timeout Configuration
HTTP_CONNECTION_TIMEOUT = 15.0  # Connection timeout in seconds
HTTP_READ_TIMEOUT = 120.0  # Read timeout in seconds (2 minutes)
HTTP_TOTAL_TIMEOUT = 300.0  # Total timeout in seconds (5 minutes)

# HTTP Retry Configuration
HTTP_MAX_RETRIES = 3  # Maximum number of retry attempts
HTTP_RETRY_BACKOFF_FACTOR = 0.5  # Backoff factor for exponential retry
HTTP_RETRY_STATUS_CODES = [500, 502, 503, 504]  # HTTP status codes to retry

# HTTP Keep-Alive Configuration
HTTP_KEEP_ALIVE = True  # Enable HTTP keep-alive connections
HTTP_KEEP_ALIVE_TIMEOUT = 60.0  # Keep-alive timeout in seconds

# HTTP Circuit Breaker Configuration
HTTP_CIRCUIT_BREAKER_ENABLED = True  # Enable circuit breaker for fault tolerance
HTTP_CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5  # Number of failures before opening circuit
HTTP_CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 60.0  # Time before attempting recovery (seconds)
HTTP_CIRCUIT_BREAKER_TEST_REQUESTS = 3  # Number of test requests in half-open state

# HTTP Performance Configuration
HTTP_ENABLE_COMPRESSION = True  # Enable HTTP compression (gzip, deflate)
HTTP_MAX_REDIRECTS = 5  # Maximum number of redirects to follow
HTTP_SSL_VERIFY = True  # Enable SSL certificate verification

# HTTP Pool Monitoring Configuration
HTTP_POOL_METRICS_ENABLED = True  # Enable HTTP pool metrics collection
HTTP_POOL_METRICS_RETENTION_SECONDS = 3600  # Metrics retention time (1 hour)
HTTP_POOL_STATS_LOGGING_ENABLED = True  # Enable periodic pool stats logging
HTTP_POOL_STATS_LOGGING_INTERVAL = 300  # Stats logging interval (5 minutes)
