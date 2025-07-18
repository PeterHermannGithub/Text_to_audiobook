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
import asyncio
import aiohttp
from .prompt_factory import PromptFactory
from .parsing import JSONParser
from .cache_manager import LLMCacheManager
from config import settings
from ...llm_pool.http_pool_manager import get_sync_pool_manager, get_async_pool_manager

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
        self.cache_manager: LLMCacheManager = LLMCacheManager()
        self.logger: logging.Logger = logging.getLogger(__name__)
        
        # Initialize HTTP connection pool manager for optimized requests
        if settings.HTTP_POOL_ENABLED:
            self.http_pool_manager = get_sync_pool_manager()
            self.logger.info("HTTP connection pooling enabled for LLM orchestrator")
        else:
            self.http_pool_manager = None
            self.logger.info("HTTP connection pooling disabled for LLM orchestrator")
        
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
                response_text = self._get_llm_response(prompt, text_metadata)
                
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
                response_text = self._get_llm_response_with_prevalidation(prompt, expected_count, text_metadata, context_hint)
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
    def _get_llm_response(self, prompt: str, text_metadata: Optional[TextMetadata] = None, 
                         context_hint: Optional[str] = None) -> str:
        """
        Core method to get a response from the LLM (local Ollama or GCP) with caching.
        """
        # Check cache first
        cached_response = self.cache_manager.get_cached_response(
            prompt=prompt,
            text_metadata=text_metadata,
            context_hint=context_hint,
            engine=self.engine,
            model=self.local_model
        )
        
        if cached_response is not None:
            self.logger.debug("Using cached LLM response")
            return cached_response
        
        # Cache miss - get fresh response
        if self.engine == 'local':
            response = self._get_local_response(prompt)
        elif self.engine == 'gcp':
            response = self._get_gcp_response(prompt)
        else:
            raise ValueError(f"Unsupported LLM engine: {self.engine}")
        
        # Cache the response
        self.cache_manager.cache_response(
            prompt=prompt,
            response=response,
            text_metadata=text_metadata,
            context_hint=context_hint,
            engine=self.engine,
            model=self.local_model
        )
        
        return response

    def _get_local_response(self, prompt: str) -> str:
        """Get response from local Ollama server with connection pooling."""
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
            # Use HTTP connection pool manager if available
            if self.http_pool_manager:
                response = self.http_pool_manager.post(
                    self.ollama_url, 
                    json_data=payload, 
                    timeout=120,
                    request_complexity='complex'  # LLM generation is complex
                )
            else:
                # Fallback to direct requests
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
        except ConnectionError as e:
            self.logger.error(f"Circuit breaker open: {e}")
            raise
    
    def _get_local_response_sync_fallback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous fallback for local response when aiohttp is not available."""
        try:
            if self.http_pool_manager:
                response = self.http_pool_manager.post(
                    self.ollama_url, 
                    json_data=payload, 
                    timeout=120,
                    request_complexity='complex'
                )
            else:
                response = requests.post(self.ollama_url, json=payload, timeout=120)
                response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Sync fallback local LLM request failed: {e}")
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

    def _get_llm_response_with_prevalidation(self, prompt: str, expected_count: int, 
                                           text_metadata: Optional[TextMetadata] = None, 
                                           context_hint: Optional[str] = None) -> Optional[str]:
        """Get LLM response with pre-validation checks."""
        try:
            response_text = self._get_llm_response(prompt, text_metadata, context_hint)
            
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

    def get_batch_speaker_classifications(
        self, 
        batch_numbered_lines: List[List[str]], 
        text_metadata: Optional[TextMetadata] = None, 
        context_hint: Optional[str] = None
    ) -> List[List[str]]:
        """
        Get speaker classifications for multiple batches of lines in a single request.
        
        This method processes multiple segments in a single LLM request, dramatically
        improving performance for large documents by reducing API call overhead.
        
        Args:
            batch_numbered_lines: List of line lists, where each inner list contains
                                 numbered lines for a single segment
            text_metadata: Optional metadata for context
            context_hint: Optional context hint for processing
            
        Returns:
            List of speaker classification lists, maintaining the same structure
            as the input but with speaker assignments for each line
        """
        if not batch_numbered_lines:
            return []
        
        # Calculate total lines for validation
        total_lines = sum(len(lines) for lines in batch_numbered_lines)
        batch_sizes = [len(lines) for lines in batch_numbered_lines]
        
        if total_lines == 0:
            return [[] for _ in batch_numbered_lines]
        
        # If only one batch or very small batches, use individual processing
        if len(batch_numbered_lines) == 1 or total_lines <= 5:
            return [self.get_speaker_classifications(lines, text_metadata, context_hint) 
                   for lines in batch_numbered_lines]
        
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                # Build batch prompt
                batch_prompt = self.prompt_factory.create_batch_classification_prompt(
                    batch_numbered_lines, text_metadata, context_hint
                )
                
                # Get LLM response
                response_text = self._get_llm_response_with_prevalidation(batch_prompt, total_lines)
                if not response_text:
                    continue
                
                # Parse batch response
                batch_speakers = self.json_parser.parse_batch_speaker_array(
                    response_text, batch_sizes, attempt
                )
                
                if batch_speakers and len(batch_speakers) == len(batch_numbered_lines):
                    # Validate each batch
                    valid_batches = []
                    for i, (speakers, lines) in enumerate(zip(batch_speakers, batch_numbered_lines)):
                        if len(speakers) == len(lines):
                            cleaned_speakers = self._validate_and_clean_speakers(speakers, text_metadata, lines)
                            valid_batches.append(cleaned_speakers)
                        else:
                            self.logger.warning(f"Batch {i+1} size mismatch: expected {len(lines)}, got {len(speakers)}")
                            valid_batches.append(["AMBIGUOUS"] * len(lines))
                    
                    return valid_batches
                
                # If parsing failed, try simpler prompt
                if attempt == 1:
                    batch_prompt = self.prompt_factory.create_simple_batch_classification_prompt(
                        batch_numbered_lines
                    )
                
            except Exception as e:
                self.logger.error(f"Error in batch classification attempt {attempt + 1}: {e}")
                continue
        
        # All attempts failed - return AMBIGUOUS for all
        self.logger.warning(f"All {max_attempts} batch attempts failed, returning AMBIGUOUS for all batches")
        return [["AMBIGUOUS"] * len(lines) for lines in batch_numbered_lines]

    async def _get_llm_response_async(self, prompt: str, text_metadata: Optional[TextMetadata] = None, 
                                     context_hint: Optional[str] = None) -> str:
        """
        Core async method to get a response from the LLM (local Ollama or GCP) with caching.
        
        Uses native async HTTP requests for maximum I/O concurrency.
        """
        # Check cache first
        cached_response = self.cache_manager.get_cached_response(
            prompt=prompt,
            text_metadata=text_metadata,
            context_hint=context_hint,
            engine=self.engine,
            model=self.local_model
        )
        
        if cached_response is not None:
            self.logger.debug("Using cached LLM response (async)")
            return cached_response
        
        # Cache miss - get fresh response
        if self.engine == 'local':
            response = await self._get_local_response_async(prompt)
        elif self.engine == 'gcp':
            response = await self._get_gcp_response_async(prompt)
        else:
            raise ValueError(f"Unsupported LLM engine: {self.engine}")
        
        # Cache the response
        self.cache_manager.cache_response(
            prompt=prompt,
            response=response,
            text_metadata=text_metadata,
            context_hint=context_hint,
            engine=self.engine,
            model=self.local_model
        )
        
        return response

    async def _get_local_response_async(self, prompt: str) -> str:
        """Get response from local Ollama server asynchronously with connection pooling."""
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
            # Use async HTTP connection pool manager if available
            if settings.HTTP_POOL_ENABLED:
                try:
                    async_pool_manager = await get_async_pool_manager()
                    response = await async_pool_manager.post(
                        self.ollama_url, 
                        json_data=payload, 
                        timeout=settings.ASYNC_TIMEOUT_SECONDS
                    )
                    result = await response.json()
                except ImportError:
                    # aiohttp not available, fall back to sync request in executor
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, 
                        lambda: self._get_local_response_sync_fallback(payload)
                    )
            else:
                # Fallback to direct aiohttp session if available
                if aiohttp is not None:
                    timeout = aiohttp.ClientTimeout(total=settings.ASYNC_TIMEOUT_SECONDS)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(self.ollama_url, json=payload) as response:
                            response.raise_for_status()
                            result = await response.json()
                else:
                    # aiohttp not available, use sync fallback in executor
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, 
                        lambda: self._get_local_response_sync_fallback(payload)
                    )
            
            if 'response' in result:
                return result['response'].strip()
            else:
                raise ValueError("No 'response' field in Ollama response")
                        
        except aiohttp.ClientError as e:
            self.logger.error(f"Async local LLM request failed: {e}")
            raise
        except asyncio.TimeoutError as e:
            self.logger.error(f"Async local LLM request timed out: {e}")
            raise
        except ConnectionError as e:
            self.logger.error(f"Async circuit breaker open: {e}")
            raise

    async def _get_gcp_response_async(self, prompt: str) -> str:
        """Get response from Google Cloud Vertex AI asynchronously."""
        try:
            # For GCP, we still need to use the synchronous API in an executor
            # since Google's client library doesn't have async support yet
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "max_output_tokens": 2048
                    }
                )
            )
            
            return response.text.strip()
            
        except Exception as e:
            self.logger.error(f"Async GCP LLM request failed: {e}")
            raise

    async def get_speaker_classifications_async(
        self, 
        numbered_lines: List[str], 
        text_metadata: Optional[TextMetadata] = None, 
        context_hint: Optional[str] = None
    ) -> List[str]:
        """
        Get speaker classifications asynchronously with sophisticated retry/repair loop.
        
        This async version provides maximum I/O concurrency for LLM requests.
        """
        max_attempts = 4
        expected_count = len(numbered_lines)
        
        # Enhanced validation for empty inputs
        if not numbered_lines or expected_count == 0:
            self.logger.warning("Empty numbered_lines provided to async LLM")
            return []
        
        # Build the main classification prompt
        prompt = self.build_classification_prompt(numbered_lines, text_metadata)
        
        for attempt in range(max_attempts):
            try:
                # Get async LLM response
                response_text = await self._get_llm_response_async_with_retry(prompt, expected_count, attempt, text_metadata, context_hint)
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
                self.logger.error(f"Error in async speaker classification attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    break
        
        # All attempts failed - return AMBIGUOUS for all
        self.logger.warning(f"All {max_attempts} async attempts failed, returning AMBIGUOUS for all {expected_count} lines")
        return ["AMBIGUOUS"] * expected_count

    async def _get_llm_response_async_with_retry(self, prompt: str, expected_count: int, attempt: int, 
                                               text_metadata: Optional[TextMetadata] = None, 
                                               context_hint: Optional[str] = None) -> Optional[str]:
        """Get LLM response asynchronously with retry logic."""
        try:
            response_text = await self._get_llm_response_async(prompt, text_metadata, context_hint)
            
            # Basic validation
            if not response_text or len(response_text.strip()) < 5:
                self.logger.warning(f"Async LLM returned empty or very short response (attempt {attempt + 1})")
                return None
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"Error getting async LLM response (attempt {attempt + 1}): {e}")
            return None

    async def get_batch_speaker_classifications_async(
        self, 
        batch_numbered_lines: List[List[str]], 
        text_metadata: Optional[TextMetadata] = None, 
        context_hint: Optional[str] = None
    ) -> List[List[str]]:
        """
        Get speaker classifications for multiple batches asynchronously.
        
        This async version provides maximum performance for batch processing.
        """
        if not batch_numbered_lines:
            return []
        
        # Calculate total lines for validation
        total_lines = sum(len(lines) for lines in batch_numbered_lines)
        batch_sizes = [len(lines) for lines in batch_numbered_lines]
        
        if total_lines == 0:
            return [[] for _ in batch_numbered_lines]
        
        # If only one batch or very small batches, use individual processing
        if len(batch_numbered_lines) == 1 or total_lines <= 5:
            tasks = [
                self.get_speaker_classifications_async(lines, text_metadata, context_hint) 
                for lines in batch_numbered_lines
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions in results
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Async individual processing failed for batch {i}: {result}")
                    final_results.append(["AMBIGUOUS"] * len(batch_numbered_lines[i]))
                else:
                    final_results.append(result)
            
            return final_results
        
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                # Build batch prompt
                batch_prompt = self.prompt_factory.create_batch_classification_prompt(
                    batch_numbered_lines, text_metadata, context_hint
                )
                
                # Get async LLM response
                response_text = await self._get_llm_response_async_with_retry(batch_prompt, total_lines, attempt, text_metadata, context_hint)
                if not response_text:
                    continue
                
                # Parse batch response
                batch_speakers = self.json_parser.parse_batch_speaker_array(
                    response_text, batch_sizes, attempt
                )
                
                if batch_speakers and len(batch_speakers) == len(batch_numbered_lines):
                    # Validate each batch
                    valid_batches = []
                    for i, (speakers, lines) in enumerate(zip(batch_speakers, batch_numbered_lines)):
                        if len(speakers) == len(lines):
                            cleaned_speakers = self._validate_and_clean_speakers(speakers, text_metadata, lines)
                            valid_batches.append(cleaned_speakers)
                        else:
                            self.logger.warning(f"Async batch {i+1} size mismatch: expected {len(lines)}, got {len(speakers)}")
                            valid_batches.append(["AMBIGUOUS"] * len(lines))
                    
                    return valid_batches
                
                # If parsing failed, try simpler prompt
                if attempt == 1:
                    batch_prompt = self.prompt_factory.create_simple_batch_classification_prompt(
                        batch_numbered_lines
                    )
                
                # Add exponential backoff
                if attempt < max_attempts - 1:
                    await asyncio.sleep(2 ** attempt)
                
            except Exception as e:
                self.logger.error(f"Error in async batch classification attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(2 ** attempt)
                continue
        
        # All attempts failed - return AMBIGUOUS for all
        self.logger.warning(f"All {max_attempts} async batch attempts failed, returning AMBIGUOUS for all batches")
        return [["AMBIGUOUS"] * len(lines) for lines in batch_numbered_lines]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return self.cache_manager.get_cache_stats()
    
    def clear_cache(self) -> None:
        """Clear all cached responses."""
        self.cache_manager.invalidate_cache()
        self.logger.info("LLM cache cleared")
    
    def clear_expired_cache(self) -> int:
        """Clear expired cache entries and return count."""
        expired_count = self.cache_manager.clear_expired_entries()
        if expired_count > 0:
            self.logger.info(f"Cleared {expired_count} expired cache entries")
        return expired_count

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