# Text-to-Audiobook API Documentation

This document provides comprehensive API documentation for the text-to-audiobook system's internal components and interfaces.

## Overview

The text-to-audiobook system is built with a modular architecture that exposes various internal APIs for text processing, speaker attribution, and audiobook generation. This document covers the key interfaces and how to use them.

## Core APIs

### Text Processing API

#### TextExtractor Class

**Location**: `src/text_processing/text_extractor.py`

```python
class TextExtractor:
    """Extracts text from various document formats with intelligent content filtering."""
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a document file.
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            str: Extracted and filtered text content
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            UnsupportedFormatError: If the file format is not supported
        """
        
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List[str]: File extensions (e.g., ['.txt', '.pdf', '.docx'])
        """
        
    def get_story_content_score(self, text: str) -> float:
        """
        Assess the quality/confidence of story content vs metadata.
        
        Args:
            text (str): Text content to analyze
            
        Returns:
            float: Confidence score (0.0-1.0)
        """
```

#### DeterministicSegmenter Class

**Location**: `src/text_processing/segmentation/deterministic_segmenter.py`

```python
class DeterministicSegmenter:
    """Rule-based text segmentation with intelligent script format conversion."""
    
    def segment_text(self, text: str, format_type: str = "auto") -> List[Dict[str, Any]]:
        """
        Segment text into logical dialogue and narrative units.
        
        Args:
            text (str): Raw text to segment
            format_type (str): Format hint ("script", "narrative", "mixed", "auto")
            
        Returns:
            List[Dict[str, Any]]: List of text segments with metadata
        """
        
    def detect_format(self, text: str) -> str:
        """
        Automatically detect the text format.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            str: Detected format ("script", "narrative", "mixed")
        """
```

### Attribution API

#### RuleBasedAttributor Class

**Location**: `src/attribution/rule_based_attributor.py`

```python
class RuleBasedAttributor:
    """High-confidence speaker attribution using deterministic rules."""
    
    def attribute_speakers(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Attribute speakers to text segments using rule-based patterns.
        
        Args:
            segments (List[Dict[str, Any]]): Text segments to process
            
        Returns:
            List[Dict[str, Any]]: Segments with speaker attribution
        """
        
    def get_confidence_threshold(self) -> float:
        """
        Get the minimum confidence threshold for rule-based attribution.
        
        Returns:
            float: Confidence threshold (typically 0.8)
        """
```

#### LLMOrchestrator Class

**Location**: `src/attribution/llm/orchestrator.py`

```python
class LLMOrchestrator:
    """Orchestrates LLM-based speaker classification for ambiguous segments."""
    
    def process_segments(
        self, 
        segments: List[Dict[str, Any]], 
        engine: str = "local",
        model: str = None
    ) -> List[Dict[str, Any]]:
        """
        Process ambiguous segments using LLM classification.
        
        Args:
            segments (List[Dict[str, Any]]): Segments needing classification
            engine (str): LLM engine ("local", "gcp")
            model (str): Specific model to use
            
        Returns:
            List[Dict[str, Any]]: Segments with LLM-assigned speakers
        """
        
    def get_supported_engines(self) -> List[str]:
        """
        Get list of supported LLM engines.
        
        Returns:
            List[str]: Available engines
        """
```

### Validation API

#### SimplifiedValidator Class

**Location**: `src/validation/validator.py`

```python
class SimplifiedValidator:
    """Quality validation for speaker attribution without text modification."""
    
    def validate_segments(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate the quality of speaker attribution.
        
        Args:
            segments (List[Dict[str, Any]]): Segments to validate
            
        Returns:
            Dict[str, Any]: Validation report with quality metrics
        """
        
    def get_quality_thresholds(self) -> Dict[str, float]:
        """
        Get quality thresholds for different content types.
        
        Returns:
            Dict[str, float]: Quality thresholds by type
        """
```

### HTTP Connection Pooling API

#### HTTPConnectionPoolManager Class

**Location**: `src/llm_pool/http_pool_manager.py`

```python
class HTTPConnectionPoolManager:
    """Enterprise-grade HTTP connection pooling for LLM requests."""
    
    def get_session(self, url: str) -> requests.Session:
        """
        Get a pooled HTTP session for the given URL.
        
        Args:
            url (str): Target URL
            
        Returns:
            requests.Session: Configured session with connection pooling
        """
        
    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics.
        
        Returns:
            Dict[str, Any]: Pool metrics and statistics
        """
        
    def clear_pool(self, url: str = None) -> None:
        """
        Clear connection pool for specific URL or all pools.
        
        Args:
            url (str, optional): Specific URL to clear, None for all
        """
```

## Configuration API

### Settings Module

**Location**: `config/settings.py`

```python
# Core LLM Configuration
DEFAULT_LLM_ENGINE: str = "local"
DEFAULT_LOCAL_MODEL: str = "deepseek-v2:16b"
OLLAMA_URL: str = "http://localhost:11434/api/generate"

# Processing Parameters
CHUNK_SIZE: int = 2500
OVERLAP_SIZE: int = 500
REFINEMENT_QUALITY_THRESHOLD: float = 98.0

# HTTP Connection Pooling
HTTP_POOL_ENABLED: bool = True
HTTP_POOL_MAX_CONNECTIONS: int = 100
HTTP_CIRCUIT_BREAKER_ENABLED: bool = True
```

## Data Structures

### Segment Schema

```python
{
    "segment_id": str,           # Unique identifier
    "text": str,                 # Segment text content
    "speaker": str,              # Attributed speaker
    "segment_type": str,         # "dialogue", "narrative", "stage_direction"
    "confidence": float,         # Attribution confidence (0.0-1.0)
    "processing_metadata": {
        "attribution_method": str,  # "rule_based", "llm", "contextual"
        "processing_time": float,   # Processing time in seconds
        "quality_score": float      # Content quality score
    }
}
```

### Validation Report Schema

```python
{
    "overall_quality_score": float,      # Overall quality (0-100)
    "speaker_consistency_score": float,  # Speaker consistency (0-100)
    "confidence_distribution": {
        "high": float,    # Percentage with confidence > 0.8
        "medium": float,  # Percentage with confidence 0.5-0.8
        "low": float      # Percentage with confidence < 0.5
    },
    "error_categories": Dict[str, int],  # Error counts by category
    "total_segments": int,
    "validation_passed": bool
}
```

## Error Handling

### Exception Classes

```python
class TextExtractionError(Exception):
    """Raised when text extraction fails."""
    pass

class UnsupportedFormatError(Exception):
    """Raised when file format is not supported."""
    pass

class AttributionError(Exception):
    """Raised when speaker attribution fails."""
    pass

class ValidationError(Exception):
    """Raised when validation fails."""
    pass

class LLMConnectionError(Exception):
    """Raised when LLM connection fails."""
    pass
```

## Usage Examples

### Basic Text Processing

```python
from src.text_processing.text_extractor import TextExtractor
from src.text_structurer import TextStructurer

# Extract text from a document
extractor = TextExtractor()
text = extractor.extract_text("input/romeo_and_juliet.pdf")

# Structure the text
structurer = TextStructurer()
structured_segments = structurer.structure_text(text)

# Access results
for segment in structured_segments:
    print(f"{segment['speaker']}: {segment['text'][:50]}...")
```

### Advanced Attribution

```python
from src.attribution.rule_based_attributor import RuleBasedAttributor
from src.attribution.llm.orchestrator import LLMOrchestrator

# Rule-based attribution first
rule_attributor = RuleBasedAttributor()
attributed_segments = rule_attributor.attribute_speakers(segments)

# LLM processing for ambiguous segments
ambiguous_segments = [s for s in attributed_segments if s['speaker'] == 'AMBIGUOUS']
if ambiguous_segments:
    llm_orchestrator = LLMOrchestrator()
    llm_results = llm_orchestrator.process_segments(ambiguous_segments)
```

### Performance Monitoring

```python
from src.llm_pool.http_pool_manager import HTTPConnectionPoolManager

# Get pool statistics
pool_manager = HTTPConnectionPoolManager()
stats = pool_manager.get_pool_stats()

print(f"Active connections: {stats['active_connections']}")
print(f"Pool efficiency: {stats['pool_efficiency']}%")
print(f"Circuit breaker status: {stats['circuit_breaker_status']}")
```

## Performance Characteristics

### Processing Times (Typical)

- **Text Extraction**: 1-5 seconds per MB of document
- **Rule-based Attribution**: <1 second per 1000 segments
- **LLM Attribution**: 2-10 seconds per 100 segments (depending on model)
- **Validation**: <1 second per 1000 segments

### Memory Usage

- **Text Extraction**: ~2x document size in memory
- **Segmentation**: ~1.5x text size in memory
- **Attribution**: ~1.2x segment data in memory
- **HTTP Pooling**: <1MB RAM overhead with 17,000x memory efficiency

### Quality Metrics

- **Rule-based Attribution**: 85-98% accuracy (high confidence segments)
- **LLM Attribution**: 90-95% accuracy (ambiguous segments)
- **Overall Quality Score**: Target >95% for script format, >85% for mixed content

## Rate Limiting and Quotas

### LLM API Limits

- **Local (Ollama)**: No rate limits, hardware dependent
- **Google Cloud**: 60 requests/minute default
- **HTTP Pool**: 100 concurrent connections max

### Recommended Usage

- Process documents in chunks of 2500 characters
- Use rule-based attribution first to minimize LLM calls
- Enable HTTP connection pooling for production deployments
- Monitor memory usage for large documents (>10MB)

## Security Considerations

- **Input Validation**: All file inputs are validated before processing
- **LLM Communication**: HTTP requests use connection pooling with circuit breakers
- **Data Privacy**: No text content is logged by default (configurable)
- **Memory Management**: Automatic cleanup prevents memory leaks

For implementation details and examples, refer to the individual module documentation and the main [README.md](README.md) file.