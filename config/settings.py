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

# Chunking Parameters
CHUNK_SIZE = 2500
OVERLAP_SIZE = 500

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
