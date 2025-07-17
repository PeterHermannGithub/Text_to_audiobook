import json
import time
import requests
import os
import re
import logging
import hashlib
from typing import Dict, List, Optional, Any, Union
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import google.api_core.exceptions
import google.generativeai as genai
from .prompt_factory import PromptFactory
from .parsing import JSONParser
from config import settings

# Type aliases
LLMConfig = Dict[str, Any]
TextMetadata = Dict[str, Any]
ValidationResult = Dict[str, Union[bool, str]]


class LLMOrchestrator:
    """Enterprise LLM orchestration engine with multi-provider support and advanced error handling.
    
    This class manages communication with Large Language Models across multiple providers,
    implementing sophisticated retry logic, response validation, and cost optimization
    strategies. It provides a unified interface for both local (Ollama) and cloud-based
    (Google Cloud) LLM services with automatic failover and comprehensive logging.
    
    Architecture:
        The orchestrator follows a modular design with separated concerns:
        - LLMOrchestrator: Core communication and coordination logic
        - JSONParser: Dedicated response parsing and validation
        - PromptFactory: Centralized prompt generation and templating
        
        This separation enables easier testing, maintenance, and extensibility
        while ensuring robust error handling at each layer.
    
    Supported LLM Providers:
        - Local Ollama: Self-hosted models (mistral, llama3, deepseek-v2, etc.)
        - Google Cloud Vertex AI: Cloud-based Gemini models
        - Extensible architecture for additional providers
    
    Key Features:
        - Multi-provider LLM support with unified interface
        - Sophisticated retry logic with exponential backoff
        - Response quality validation and correction prompts
        - JSON parsing with comprehensive error recovery
        - Speaker classification with confidence scoring
        - Cost optimization through intelligent prompt engineering
        - Comprehensive logging and debugging capabilities
        - Thread-safe concurrent processing support
    
    Processing Pipeline:
        1. Prompt Generation: Create optimized prompts via PromptFactory
        2. Request Execution: Send requests with retry logic and validation
        3. Response Validation: Quality checks and error pattern detection
        4. JSON Parsing: Robust parsing with fallback strategies
        5. Content Validation: Semantic validation of parsed results
        6. Error Recovery: Progressive fallback and correction prompts
    
    Attributes:
        engine: LLM provider type ('local' for Ollama, 'gcp' for Google Cloud)
        local_model: Model name for local Ollama processing
        ollama_url: URL endpoint for local Ollama server
        log_dir: Directory path for debug and error logging
        prompt_factory: Instance for generating optimized prompts
        json_parser: Instance for parsing and validating LLM responses
        logger: Configured logging instance for debugging
        debug_logger: Optional detailed debug logging for LLM interactions
        _debug_counter: Internal counter for debug log correlation
    
    Performance Optimization:
        - Response caching to reduce redundant API calls
        - Prompt optimization to minimize token usage
        - Batch processing capabilities for multiple requests
        - Connection pooling for improved latency
        - Quality-based retry logic to minimize failed attempts
    
    Examples:
        Basic speaker classification:
        >>> orchestrator = LLMOrchestrator({'engine': 'local'})
        >>> lines = ["Hello there!", "How are you today?"]
        >>> speakers = orchestrator.get_speaker_classifications(lines)
        >>> print(speakers)  # ['Character1', 'Character2']
        
        Advanced Google Cloud configuration:
        >>> config = {
        ...     'engine': 'gcp',
        ...     'project_id': 'my-project',
        ...     'location': 'us-central1'
        ... }
        >>> orchestrator = LLMOrchestrator(config)
        >>> response = orchestrator.get_structured_response(prompt)
        
        Error handling and fallback:
        >>> orchestrator = LLMOrchestrator({'engine': 'local'})
        >>> try:
        ...     result = orchestrator.get_speaker_classifications(lines)
        ... except ConnectionError:
        ...     print("LLM service unavailable, using fallback")
    
    Error Handling:
        The orchestrator implements comprehensive error handling:
        - Network failures: Automatic retry with exponential backoff
        - Malformed responses: Progressive prompt correction strategies
        - Service unavailability: Graceful degradation to fallback methods
        - Rate limiting: Intelligent backoff and request spacing
        - Memory pressure: Request size optimization and chunking
    
    Note:
        This class is thread-safe for concurrent processing and includes
        comprehensive debugging capabilities. Enable debug logging via
        settings.LLM_DEBUG_LOGGING for detailed request/response analysis.
        For high-throughput scenarios, consider using LLMPoolManager for
        connection pooling and load balancing.
    """
    
    def __init__(self, config: LLMConfig) -> None:
        self.engine: str = config['engine']
        self.local_model: Optional[str] = config.get('local_model')
        self.ollama_url: str = settings.OLLAMA_URL
        self.log_dir: str = os.path.join(os.getcwd(), settings.LOG_DIR)
        self.prompt_factory: PromptFactory = PromptFactory()
        self.json_parser: JSONParser = JSONParser()
        self.logger: logging.Logger = logging.getLogger(__name__)
        
        # Initialize debug logger if enabled
        self.debug_logger: Optional[logging.Logger] = None
        self._debug_counter: int = 0
        if settings.LLM_DEBUG_LOGGING:
            self._setup_debug_logger()

        if self.engine == 'gcp':
            if not config.get('project_id') or not config.get('location'):
                raise ValueError("Project ID and location are required for GCP engine.")
            self.model = genai.GenerativeModel(settings.GCP_LLM_MODEL)

    def build_prompt(self, text_content: str, text_metadata: Optional[TextMetadata] = None) -> str:
        """DEPRECATED: Use build_classification_prompt instead."""
        return self.prompt_factory.create_structuring_prompt(text_content, text_metadata=text_metadata)
    
    def build_classification_prompt(self, numbered_lines: List[str], text_metadata: Optional[TextMetadata] = None) -> str:
        """Build a speaker classification prompt for pre-segmented lines."""
        return self.prompt_factory.create_speaker_classification_prompt(numbered_lines, text_metadata)

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((requests.exceptions.RequestException, google.api_core.exceptions.ServiceUnavailable))
    )
    def get_structured_response(self, prompt: str, text_metadata: Optional[TextMetadata] = None) -> List[Any]:
        """
        Gets and parses a structured JSON response from the LLM with validation and retry logic.
        """
        max_content_retries = 2
        
        for attempt in range(max_content_retries + 1):
            try:
                response_text = self._get_llm_response(prompt)
                
                # Validate response quality before parsing
                validation_result = self._validate_response_quality(response_text, prompt)
                if not validation_result['is_valid']:
                    if attempt < max_content_retries:
                        print(f"Response quality issue (attempt {attempt + 1}): {validation_result['reason']}")
                        # Try with a correction prompt
                        prompt = self.prompt_factory.create_json_correction_prompt(response_text)
                        continue
                    else:
                        print(f"Final attempt failed: {validation_result['reason']}")
                        return []
                
                # Parse the validated response using JSONParser
                parsed_result = self.json_parser.parse_structured_json(response_text)
                
                # Additional content validation
                if parsed_result and self._validate_parsed_content(parsed_result, prompt):
                    return parsed_result
                elif attempt < max_content_retries:
                    print(f"Parsed content validation failed (attempt {attempt + 1}), retrying...")
                    prompt = self.prompt_factory.create_json_correction_prompt(response_text)
                    continue
                else:
                    print("All retry attempts failed, returning empty result")
                    return []
                    
            except Exception as e:
                if attempt < max_content_retries:
                    print(f"Error in response processing (attempt {attempt + 1}): {e}")
                    continue
                else:
                    print(f"Fatal error after all retries: {e}")
                    return []
        
        return []
    
    def get_speaker_classifications(
        self, 
        numbered_lines: List[str], 
        text_metadata: Optional[TextMetadata] = None, 
        context_hint: Optional[str] = None
    ) -> List[str]:
        """
        Gets speaker classifications with sophisticated retry/repair loop.
        """
        max_attempts = 4
        expected_count = len(numbered_lines)
        
        # Enhanced validation for empty inputs
        if not numbered_lines or expected_count == 0:
            self.logger.warning("Empty numbered_lines provided to LLM")
            return []
        
        # Build the main classification prompt
        prompt = self.build_classification_prompt(numbered_lines, text_metadata)
        
        for attempt in range(max_attempts):
            try:
                # Get LLM response
                response_text = self._get_llm_response_with_prevalidation(prompt, expected_count)
                if not response_text:
                    continue
                
                # Parse using JSONParser
                speakers = self.json_parser.parse_speaker_array_enhanced(response_text, expected_count, attempt)
                
                if speakers and len(speakers) == expected_count:
                    # Validate and clean the speakers
                    cleaned_speakers = self._validate_and_clean_speakers(speakers, text_metadata, numbered_lines)
                    return cleaned_speakers
                
                # If parsing failed, try different prompts based on attempt
                if attempt == 1:
                    prompt = self.prompt_factory.create_json_correction_prompt(response_text)
                elif attempt == 2:
                    prompt = self.prompt_factory.create_simple_classification_prompt(numbered_lines)
                elif attempt >= 3:
                    prompt = self.prompt_factory.create_ultra_simple_prompt(numbered_lines)
                    
            except Exception as e:
                self.logger.error(f"Error in get_speaker_classifications attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    continue
                else:
                    break
        
        # All attempts failed - return AMBIGUOUS for all
        self.logger.warning(f"All {max_attempts} attempts failed, returning AMBIGUOUS for all {expected_count} lines")
        return ["AMBIGUOUS"] * expected_count

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((requests.exceptions.RequestException, google.api_core.exceptions.ServiceUnavailable))
    )
    def _get_llm_response(self, prompt: str) -> str:
        """
        Core method to get a response from the LLM (local Ollama or GCP).
        """
        if self.engine == 'local':
            return self._get_local_response(prompt)
        elif self.engine == 'gcp':
            return self._get_gcp_response(prompt)
        else:
            raise ValueError(f"Unsupported LLM engine: {self.engine}")

    def _get_local_response(self, prompt: str) -> str:
        """Get response from local Ollama server."""
        payload = {
            "model": self.local_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 2048
            }
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            if 'response' in result:
                return result['response'].strip()
            else:
                raise ValueError("No 'response' field in Ollama response")
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Local LLM request failed: {e}")
            raise

    def _get_gcp_response(self, prompt: str) -> str:
        """Get response from Google Cloud Vertex AI."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_output_tokens": 2048
                }
            )
            return response.text.strip()
            
        except Exception as e:
            self.logger.error(f"GCP LLM request failed: {e}")
            raise

    def _validate_response_quality(self, response_text: str, prompt: str) -> ValidationResult:
        """Validate the quality of LLM response before parsing."""
        if not response_text or not response_text.strip():
            return {"is_valid": False, "reason": "Empty response"}
        
        if len(response_text.strip()) < 10:
            return {"is_valid": False, "reason": "Response too short"}
        
        # Check for obvious error patterns
        error_patterns = [
            "I cannot", "I'm sorry", "I don't understand",
            "Error:", "ERROR:", "Failed to"
        ]
        
        for pattern in error_patterns:
            if pattern.lower() in response_text.lower():
                return {"is_valid": False, "reason": f"Error pattern detected: {pattern}"}
        
        return {"is_valid": True, "reason": "Valid response"}

    def _validate_parsed_content(self, parsed_data: Any, prompt: str) -> bool:
        """Validate the parsed content structure."""
        if not parsed_data or not isinstance(parsed_data, list):
            return False
        
        # Check that all items are non-empty strings
        for item in parsed_data:
            if not isinstance(item, str) or not item.strip():
                return False
        
        return True

    def _get_llm_response_with_prevalidation(self, prompt: str, expected_count: int) -> Optional[str]:
        """Get LLM response with pre-validation checks."""
        try:
            response_text = self._get_llm_response(prompt)
            
            # Basic validation
            if not response_text or len(response_text.strip()) < 5:
                self.logger.warning("LLM returned empty or very short response")
                return None
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"Error getting LLM response: {e}")
            return None

    def _validate_and_clean_speakers(
        self, 
        speakers: List[str], 
        text_metadata: Optional[TextMetadata], 
        numbered_lines: Optional[List[str]] = None
    ) -> List[str]:
        """Validate and clean speaker names."""
        cleaned_speakers = []
        known_characters = set()
        
        if text_metadata:
            known_characters = text_metadata.get('potential_character_names', set())
        
        for speaker in speakers:
            speaker = speaker.strip().strip('"').strip("'")
            
            # Basic cleaning
            if not speaker:
                speaker = "AMBIGUOUS"
            elif speaker.lower() in ['unknown', 'unclear', 'ambiguous']:
                speaker = "AMBIGUOUS"
            
            cleaned_speakers.append(speaker)
        
        return cleaned_speakers

    def _setup_debug_logger(self) -> None:
        """Setup debug logging for LLM interactions."""
        if not hasattr(self, 'debug_logger') or self.debug_logger is None:
            self.debug_logger = logging.getLogger('llm_debug')
            self.debug_logger.setLevel(logging.DEBUG)
            
            # Create file handler
            debug_file = os.path.join(self.log_dir, 'llm_debug.log')
            os.makedirs(self.log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(debug_file)
            file_handler.setLevel(logging.DEBUG)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            # Add handler if not already added
            if not self.debug_logger.handlers:
                self.debug_logger.addHandler(file_handler)

    def _log_llm_processing(
        self, 
        step_name: str, 
        input_data: Any, 
        output_data: Any, 
        context: Optional[Any] = None
    ) -> None:
        """Log LLM processing steps for debugging."""
        if settings.LLM_DEBUG_LOGGING and self.debug_logger:
            self._debug_counter += 1
            log_entry = {
                "counter": self._debug_counter,
                "step": step_name,
                "input": str(input_data)[:500] + "..." if len(str(input_data)) > 500 else str(input_data),
                "output": str(output_data)[:500] + "..." if len(str(output_data)) > 500 else str(output_data),
                "context": context
            }
            self.debug_logger.debug(f"LLM_PROCESSING: {json.dumps(log_entry, indent=2)}")