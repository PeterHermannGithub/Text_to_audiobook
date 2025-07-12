import json
import time
import requests
import os
import re
import logging
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import google.api_core.exceptions
import google.generativeai as genai
from .prompt_factory import PromptFactory
from config import settings

class LLMOrchestrator:
    def __init__(self, config):
        self.engine = config['engine']
        self.local_model = config.get('local_model')
        self.ollama_url = settings.OLLAMA_URL
        self.log_dir = os.path.join(os.getcwd(), settings.LOG_DIR)
        self.prompt_factory = PromptFactory()
        self.logger = logging.getLogger(__name__)

        if self.engine == 'gcp':
            if not config.get('project_id') or not config.get('location'):
                raise ValueError("Project ID and location are required for GCP engine.")
            self.model = genai.GenerativeModel(settings.GCP_LLM_MODEL)

    def build_prompt(self, text_content, text_metadata=None):
        """DEPRECATED: Use build_classification_prompt instead."""
        return self.prompt_factory.create_structuring_prompt(text_content, text_metadata=text_metadata)
    
    def build_classification_prompt(self, numbered_lines, text_metadata=None):
        """Build a speaker classification prompt for pre-segmented lines."""
        return self.prompt_factory.create_speaker_classification_prompt(numbered_lines, text_metadata)

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((requests.exceptions.RequestException, google.api_core.exceptions.ServiceUnavailable))
    )
    def get_structured_response(self, prompt, text_metadata=None):
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
                
                # Parse the validated response
                parsed_result = self._parse_structured_json(response_text, prompt)
                
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
    
    def get_speaker_classifications(self, numbered_lines, text_metadata=None):
        """
        Gets speaker classifications for pre-segmented lines.
        
        Args:
            numbered_lines: List of strings representing pre-segmented text lines
            text_metadata: Metadata containing character names and format info
            
        Returns:
            List of speaker names (strings) matching the input lines
        """
        if not numbered_lines:
            return []
            
        max_retries = 2
        expected_count = len(numbered_lines)
        
        for attempt in range(max_retries + 1):
            try:
                # Build classification prompt
                prompt = self.build_classification_prompt(numbered_lines, text_metadata)
                
                # Get LLM response
                response_text = self._get_llm_response(prompt)
                
                # Parse speaker array
                speakers = self._parse_speaker_array(response_text, expected_count)
                
                if speakers and len(speakers) == expected_count:
                    self.logger.debug(f"Successfully classified {len(speakers)} speakers")
                    return speakers
                elif attempt < max_retries:
                    self.logger.warning(f"Classification attempt {attempt + 1} failed: expected {expected_count}, got {len(speakers) if speakers else 0}")
                    continue
                else:
                    self.logger.error(f"All classification attempts failed, using fallback")
                    return self._fallback_classification(numbered_lines)
                    
            except Exception as e:
                if attempt < max_retries:
                    self.logger.warning(f"Classification attempt {attempt + 1} error: {e}")
                    continue
                else:
                    self.logger.error(f"Classification failed after all attempts: {e}")
                    return self._fallback_classification(numbered_lines)
        
        return self._fallback_classification(numbered_lines)
    
    def _parse_speaker_array(self, response_text, expected_count):
        """
        Parses LLM response expecting a simple JSON array of speaker names.
        
        Args:
            response_text: Raw LLM response
            expected_count: Expected number of speakers
            
        Returns:
            List of speaker names or None if parsing fails
        """
        if not response_text or not response_text.strip():
            return None
            
        # Clean and extract JSON array
        cleaned_text = self._extract_json_from_text(response_text)
        if not cleaned_text:
            self.logger.warning("No JSON array found in classification response")
            return None
            
        try:
            speakers = json.loads(cleaned_text)
            
            # Validate it's an array of strings
            if not isinstance(speakers, list):
                self.logger.warning(f"Expected array, got {type(speakers)}")
                return None
                
            # Validate all items are strings
            string_speakers = []
            for item in speakers:
                if isinstance(item, str):
                    string_speakers.append(item.strip())
                else:
                    self.logger.warning(f"Non-string speaker found: {item}")
                    string_speakers.append("AMBIGUOUS")
                    
            if len(string_speakers) != expected_count:
                self.logger.warning(f"Speaker count mismatch: expected {expected_count}, got {len(string_speakers)}")
                # Pad or truncate to match expected count
                if len(string_speakers) < expected_count:
                    string_speakers.extend(["AMBIGUOUS"] * (expected_count - len(string_speakers)))
                else:
                    string_speakers = string_speakers[:expected_count]
                    
            return string_speakers
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parsing failed for speaker classification: {e}")
            return None
    
    def _fallback_classification(self, numbered_lines):
        """
        Fallback speaker classification when LLM fails.
        
        Args:
            numbered_lines: List of text lines
            
        Returns:
            List of fallback speaker classifications
        """
        fallback_speakers = []
        
        for line in numbered_lines:
            line = line.strip()
            
            # Simple heuristics for fallback classification
            if not line:
                fallback_speakers.append("narrator")
            elif any(marker in line for marker in ['"', '"', '"', "'", '—']):
                # Likely dialogue
                fallback_speakers.append("AMBIGUOUS")
            elif ':' in line and line.split(':')[0].strip().isupper():
                # Likely script format
                script_name = line.split(':')[0].strip()
                fallback_speakers.append(script_name)
            else:
                # Likely narrative
                fallback_speakers.append("narrator")
                
        self.logger.info(f"Applied fallback classification to {len(fallback_speakers)} lines")
        return fallback_speakers

    def _get_llm_response(self, prompt):
        if self.engine == 'gcp':
            response = self.model.generate_content(prompt)
            return response.text
        else: # local
            payload = {
                "model": self.local_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0}
            }
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            return response.json().get('response', '')

    def _parse_structured_json(self, response_text, original_prompt):
        """
        Parses the LLM's output, expecting a JSON array of strings (paragraphs).
        """
        cleaned_text = self._extract_json_from_text(response_text)
        if not cleaned_text:
            self._log_error("No JSON array found in the response.", original_prompt, response_text)
            return []

        try:
            data = json.loads(cleaned_text)
            # Updated validation: expecting array of strings, not objects
            if isinstance(data, list):
                # Filter out empty strings and validate remaining items
                filtered_data = [item for item in data if isinstance(item, str) and item.strip()]
                if filtered_data:  # Ensure we have at least some valid content
                    return filtered_data
                else:
                    self._log_error("JSON array contains no valid non-empty strings.", original_prompt, response_text)
                    return []
            else:
                self._log_error("JSON data is not an array.", original_prompt, response_text)
                return []
        except json.JSONDecodeError as e:
            self._log_error(f"JSON parsing failed: {e}", original_prompt, response_text)
            return []

    def _extract_json_from_text(self, text):
        """
        Extracts and cleans the JSON array string from LLM response, handling common formatting issues.
        """
        # Remove markdown code block fences (```json, ```, etc.)
        text = re.sub(r'^```(?:json)?\s*|```\s*$', '', text, flags=re.MULTILINE).strip()
        
        # Find the first '[' and the last ']' to isolate the JSON array
        start_index = text.find('[')
        end_index = text.rfind(']')

        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_string = text[start_index : end_index + 1]
            
            # Clean up common JSON formatting issues
            json_string = self._clean_json_string(json_string)
            return json_string
        
        return None

    def _clean_json_string(self, json_string):
        """
        Cleans common JSON formatting issues from LLM responses.
        """
        # Clean up trailing commas before ] or }
        json_string = re.sub(r',\s*\]', ']', json_string)
        json_string = re.sub(r',\s*\}', '}', json_string)
        
        # Handle HTML entities that appear as unicode escapes
        json_string = json_string.replace('\\u003cbr\\u003e', '')  # Remove <br> tags
        json_string = json_string.replace('\\u003c', '<').replace('\\u003e', '>')  # Fix other HTML tags
        
        # Fix common quote escaping issues by parsing and re-encoding properly
        try:
            # Try to parse as JSON first to identify quote issues
            json.loads(json_string)
            return json_string
        except json.JSONDecodeError as e:
            # If parsing fails, try to fix common quote issues
            if 'Invalid control character' in str(e):
                # Remove or escape invalid control characters
                json_string = re.sub(r'[^\x20-\x7E\n\r\t]', '', json_string)
            
            # Try to fix unescaped quotes in strings
            # This is a heuristic approach - look for patterns like "text with "quotes" inside"
            lines = json_string.split('\n')
            fixed_lines = []
            
            for line in lines:
                if '"' in line and not line.strip().startswith('[') and not line.strip().endswith(']'):
                    # Count quotes to detect unescaped quotes
                    quote_count = line.count('"')
                    if quote_count > 2:  # More than opening and closing quotes
                        # Simple fix: escape internal quotes
                        # Find content between first and last quote
                        first_quote = line.find('"')
                        last_quote = line.rfind('"')
                        if first_quote != last_quote and first_quote != -1:
                            prefix = line[:first_quote+1]
                            suffix = line[last_quote:]
                            middle = line[first_quote+1:last_quote]
                            # Escape quotes in the middle part
                            middle = middle.replace('"', '\\"')
                            line = prefix + middle + suffix
                
                fixed_lines.append(line)
            
            json_string = '\n'.join(fixed_lines)
        
        return json_string

    def _validate_response_quality(self, response_text, prompt):
        """
        Validates the quality of an LLM response before attempting to parse it.
        Returns dict with 'is_valid' boolean and 'reason' string.
        """
        if not response_text or not response_text.strip():
            return {'is_valid': False, 'reason': 'Empty response'}
        
        # Check minimum length - should be reasonable for text processing
        if len(response_text.strip()) < 10:
            return {'is_valid': False, 'reason': 'Response too short'}
        
        # Check for obvious error indicators
        error_indicators = [
            'sorry', 'cannot', 'unable', 'error', 'failed',
            'i cannot', 'i am unable', 'i apologize'
        ]
        response_lower = response_text.lower()
        if any(indicator in response_lower for indicator in error_indicators):
            return {'is_valid': False, 'reason': 'Response contains error indicators'}
        
        # Check for presence of JSON structure
        if '[' not in response_text or ']' not in response_text:
            return {'is_valid': False, 'reason': 'No JSON array structure found'}
        
        # Check for suspiciously long response (might indicate hallucination)
        original_text_estimate = len(prompt) * 0.3  # Rough estimate of input text size
        if len(response_text) > original_text_estimate * 3:
            return {'is_valid': False, 'reason': 'Response suspiciously long (possible hallucination)'}
        
        return {'is_valid': True, 'reason': 'Response passed quality checks'}

    def _validate_parsed_content(self, parsed_data, prompt):
        """
        Validates the parsed JSON content for logical consistency.
        """
        if not parsed_data:
            return False
        
        # Check reasonable number of segments
        if len(parsed_data) < 1:
            return False
        
        if len(parsed_data) > 1000:  # Suspiciously high segmentation
            return False
        
        # Check for reasonable segment lengths
        total_chars = sum(len(segment) for segment in parsed_data)
        if total_chars < 50:  # Too little content
            return False
        
        # Check for signs of repetition or corruption
        unique_segments = set(parsed_data)
        if len(unique_segments) < len(parsed_data) * 0.8:  # Too much repetition
            return False
        
        # Check average segment length (should be reasonable for paragraphs)
        avg_length = total_chars / len(parsed_data)
        if avg_length < 10 or avg_length > 2000:
            return False
        
        return True

    def _log_error(self, error_message, prompt, response_text):
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_filename = os.path.join(self.log_dir, f"llm_parsing_error_{timestamp}.log")
        with open(log_filename, "w", encoding="utf-8") as f:
            f.write(f"--- ERROR ---\n{error_message}\n\n--- PROMPT ---\n{prompt}\n\n--- RAW RESPONSE ---\n{response_text}")
        print(f"Error processing LLM response. Details logged to {log_filename}")