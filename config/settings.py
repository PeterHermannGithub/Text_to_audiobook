# config/settings.py

# LLM Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_LOCAL_MODEL = "mistral"
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
