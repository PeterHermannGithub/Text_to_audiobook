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
        Gets speaker classifications for pre-segmented lines with enhanced error handling.
        
        Args:
            numbered_lines: List of strings representing pre-segmented text lines
            text_metadata: Metadata containing character names and format info
            
        Returns:
            List of speaker names (strings) matching the input lines
        """
        if not numbered_lines:
            return []
            
        max_retries = 3  # Increased from 2
        expected_count = len(numbered_lines)
        
        # Pre-validate inputs - Use much smaller chunks for better accuracy
        if expected_count > 8:  # Reduced from 50 to 8 for better LLM performance
            self.logger.info(f"Input chunk has {expected_count} lines, splitting into smaller chunks for better accuracy")
            return self._process_contextual_chunks(numbered_lines, text_metadata)
        
        previous_response = None
        for attempt in range(max_retries + 1):
            try:
                # Build classification prompt with attempt-specific optimizations
                prompt = self._build_optimized_prompt(numbered_lines, text_metadata, attempt, previous_response)
                
                # Get LLM response with enhanced validation
                response_text = self._get_llm_response_with_validation(prompt)
                
                # Store response for potential correction in next attempt
                if response_text:
                    previous_response = response_text
                
                if not response_text or not response_text.strip():
                    self.logger.warning(f"Attempt {attempt + 1}: Empty response from LLM")
                    if attempt < max_retries:
                        continue
                    else:
                        return self._fallback_classification(numbered_lines)
                
                # Parse speaker array with enhanced error recovery
                speakers = self._parse_speaker_array_enhanced(response_text, expected_count, attempt)
                
                if speakers and len(speakers) == expected_count:
                    # Validate speaker quality and fix character name association errors
                    speakers = self._validate_and_clean_speakers(speakers, text_metadata, numbered_lines)
                    self.logger.debug(f"Successfully classified {len(speakers)} speakers on attempt {attempt + 1}")
                    return speakers
                elif attempt < max_retries:
                    self.logger.warning(f"Attempt {attempt + 1} failed: expected {expected_count}, got {len(speakers) if speakers else 0}")
                    continue
                else:
                    # Last attempt - try partial recovery
                    if speakers and len(speakers) > 0:
                        self.logger.warning(f"Partial recovery: got {len(speakers)} speakers, padding to {expected_count}")
                        return self._pad_speaker_array(speakers, expected_count)
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
    
    def _process_large_chunk(self, numbered_lines, text_metadata):
        """
        Legacy large chunk processing - replaced by contextual chunks.
        """
        return self._process_contextual_chunks(numbered_lines, text_metadata)
    
    def _process_contextual_chunks(self, numbered_lines, text_metadata):
        """
        Process lines in small contextual chunks with speaker memory for better accuracy.
        """
        chunk_size = 6  # Reduced from 25 to 6 for much better LLM performance
        all_speakers = []
        speaker_context = []  # Track recent speakers for context
        
        self.logger.info(f"Processing {len(numbered_lines)} lines in chunks of {chunk_size}")
        
        for i in range(0, len(numbered_lines), chunk_size):
            chunk = numbered_lines[i:i + chunk_size]
            
            # Add context from previous chunks for continuity
            enhanced_metadata = self._add_speaker_context(text_metadata, speaker_context)
            
            # Process the small chunk
            chunk_speakers = self.get_speaker_classifications(chunk, enhanced_metadata)
            
            if chunk_speakers:
                all_speakers.extend(chunk_speakers)
                
                # Update speaker context for next chunk
                speaker_context = self._update_speaker_context(speaker_context, chunk_speakers)
            else:
                # Fallback if chunk processing fails
                self.logger.warning(f"Chunk {i//chunk_size + 1} failed, using fallback")
                fallback_speakers = ["AMBIGUOUS"] * len(chunk)
                all_speakers.extend(fallback_speakers)
        
        return all_speakers
    
    def _add_speaker_context(self, text_metadata, speaker_context):
        """
        Add recent speaker context to metadata for better continuity.
        """
        enhanced_metadata = text_metadata.copy() if text_metadata else {}
        
        if speaker_context:
            # Add recent speakers to context
            enhanced_metadata['recent_speakers'] = speaker_context[-6:]  # Last 6 speakers
            enhanced_metadata['speaker_context'] = f"Recent speakers in conversation: {', '.join(speaker_context[-6:])}"
        
        return enhanced_metadata
    
    def _update_speaker_context(self, speaker_context, new_speakers):
        """
        Update speaker context with newly classified speakers.
        """
        # Add non-generic speakers to context
        valid_speakers = [s for s in new_speakers if s not in ['narrator', 'AMBIGUOUS', 'UNFIXABLE']]
        speaker_context.extend(valid_speakers)
        
        # Keep only last 10 speakers to prevent context bloat
        return speaker_context[-10:]
    
    def _build_optimized_prompt(self, numbered_lines, text_metadata, attempt, previous_response=None):
        """
        Build optimized prompt based on attempt number and previous failures.
        """
        if attempt == 0:
            # First attempt - use enhanced bulletproof prompt
            return self.build_classification_prompt(numbered_lines, text_metadata)
        elif attempt == 1:
            # Second attempt - simplified prompt with fewer examples
            return self._build_simplified_prompt(numbered_lines, text_metadata)
        elif attempt == 2 and previous_response:
            # Third attempt - send back malformed JSON for correction
            return self.prompt_factory.create_json_correction_prompt(previous_response[:500])  # Limit length
        else:
            # Final attempt - ultra-simple format
            return self._build_minimal_prompt(numbered_lines, text_metadata)
    
    def _build_simplified_prompt(self, numbered_lines, text_metadata):
        """
        Build a simplified prompt for retry attempts.
        """
        character_names = []
        if text_metadata:
            character_names = list(text_metadata.get('potential_character_names', set()))[:5]  # Limit to 5
        
        character_context = f"\\nCharacters: {', '.join(character_names)}" if character_names else ""
        
        numbered_display = ""
        for i, line in enumerate(numbered_lines, 1):
            numbered_display += f"{i}. {line}\\n"
        
        return f"""Classify speakers for each line. Return JSON array with exactly {len(numbered_lines)} names.
Use: "narrator" for description, character names for dialogue, "AMBIGUOUS" if unclear.{character_context}

{numbered_display.strip()}

JSON:"""
    
    def _build_minimal_prompt(self, numbered_lines, text_metadata):
        """
        Build a minimal prompt for final retry attempts.
        """
        numbered_display = ""
        for i, line in enumerate(numbered_lines, 1):
            numbered_display += f"{i}. {line}\\n"
        
        return f"""Return JSON array with {len(numbered_lines)} speaker names:

{numbered_display.strip()}

JSON:"""
    
    def _get_llm_response_with_validation(self, prompt):
        """
        Get LLM response with basic validation.
        """
        response = self._get_llm_response(prompt)
        
        # Basic validation
        if not response or len(response.strip()) < 3:
            return None
            
        # Check for obvious error messages
        error_indicators = ['error', 'cannot', 'unable', 'sorry']
        if any(indicator in response.lower() for indicator in error_indicators):
            return None
            
        return response
    
    def _parse_speaker_array_enhanced(self, response_text, expected_count, attempt):
        """
        Bulletproof speaker array parsing with multi-stage validation and progressive fallback.
        """
        if not response_text or not response_text.strip():
            self.logger.warning("Empty LLM response")
            return None
        
        # Stage 1: Multi-step JSON validation and parsing
        speakers = self._bulletproof_json_parse(response_text, expected_count, attempt)
        if speakers and len(speakers) == expected_count:
            return speakers
        
        # Stage 2: Alternative parsing strategies based on attempt number
        if attempt == 1:
            speakers = self._parse_speaker_array_alternative(response_text, expected_count)
        elif attempt == 2:
            speakers = self._parse_fix_json_format(response_text, expected_count)
        elif attempt >= 3:
            speakers = self._parse_ultra_simple_format(response_text, expected_count)
        
        if speakers and len(speakers) == expected_count:
            return speakers
        
        # Stage 3: Final fallback with padding/truncation
        if speakers and len(speakers) > 0:
            self.logger.warning(f"Partial parse success: got {len(speakers)}, expected {expected_count}")
            return self._smart_padding_truncation(speakers, expected_count)
        
        return None
    
    def _parse_speaker_array_alternative(self, response_text, expected_count):
        """
        Alternative parsing method for malformed responses.
        """
        try:
            # Try to extract lines that look like speaker names
            lines = response_text.strip().split('\\n')
            potential_speakers = []
            
            for line in lines:
                line = line.strip()
                # Skip empty lines and obvious non-speakers
                if not line or line.startswith('#') or line.startswith('//'):
                    continue
                    
                # Try to extract speaker from various formats
                if line.startswith('"') and line.endswith('"'):
                    potential_speakers.append(line[1:-1])
                elif ': ' in line:
                    potential_speakers.append(line.split(': ')[0])
                elif line.replace(',', '').replace('.', '').isalpha():
                    potential_speakers.append(line.replace(',', '').replace('.', ''))
            
            # If we found the right number, return them
            if len(potential_speakers) == expected_count:
                return potential_speakers
            
        except Exception as e:
            self.logger.debug(f"Alternative parsing failed: {e}")
        
        return None
    
    def _bulletproof_json_parse(self, response_text, expected_count, attempt):
        """
        Multi-stage JSON validation with comprehensive error handling.
        """
        # Step 1: Basic JSON validation
        try:
            # Try to parse the entire response as JSON
            data = json.loads(response_text.strip())
            if isinstance(data, list) and len(data) == expected_count:
                if all(isinstance(item, str) for item in data):
                    self.logger.debug(f"Direct JSON parse successful (attempt {attempt + 1})")
                    return data
        except json.JSONDecodeError:
            pass
        
        # Step 2: Extract JSON array pattern
        json_candidates = self._extract_json_candidates(response_text)
        for candidate in json_candidates:
            try:
                data = json.loads(candidate)
                if isinstance(data, list) and all(isinstance(item, str) for item in data):
                    if len(data) == expected_count:
                        self.logger.debug(f"Pattern extraction parse successful (attempt {attempt + 1})")
                        return data
                    elif len(data) > 0:
                        # Store partial result for potential use
                        self.logger.debug(f"Partial pattern match: {len(data)} items")
                        return data
            except json.JSONDecodeError:
                continue
        
        # Step 3: Clean and retry JSON parsing
        cleaned_response = self._clean_json_response(response_text)
        if cleaned_response != response_text:
            try:
                data = json.loads(cleaned_response)
                if isinstance(data, list) and all(isinstance(item, str) for item in data):
                    self.logger.debug(f"Cleaned JSON parse successful (attempt {attempt + 1})")
                    return data
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _extract_json_candidates(self, text):
        """
        Extract potential JSON arrays from text using multiple patterns.
        """
        candidates = []
        
        # Pattern 1: Look for [.*] array patterns
        array_patterns = [
            r'\[(?:[^[\]]*(?:"[^"]*"[^[\]]*)*)*\]',  # Basic array pattern
            r'\[(?:\s*"[^"]*"\s*,?\s*)*\]',          # String array pattern
            r'\[(?:\s*"[^"]*"\s*(?:,\s*"[^"]*"\s*)*)\]'  # Proper comma-separated strings
        ]
        
        for pattern in array_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            candidates.extend(matches)
        
        # Pattern 2: Look for JSON-like structures in code blocks
        code_block_patterns = [
            r'```(?:json)?\s*(\[.*?\])\s*```',
            r'`(\[.*?\])`',
            r'JSON ARRAY:\s*(\[.*?\])',
            r'OUTPUT:\s*(\[.*?\])',
        ]
        
        for pattern in code_block_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            candidates.extend(matches)
        
        # Pattern 3: Line-by-line extraction for simple lists
        lines = text.split('\n')
        potential_arrays = []
        current_array = []
        
        for line in lines:
            line = line.strip()
            # Look for quoted strings that might be array elements
            if re.match(r'^"[^"]*"$', line):
                current_array.append(line)
            elif line == '[' and not current_array:
                current_array = ['[']
            elif line == ']' and current_array and current_array[0] == '[':
                current_array.append(']')
                array_str = ''.join(current_array) if len(current_array) == 2 else '[' + ', '.join(current_array[1:-1]) + ']'
                candidates.append(array_str)
                current_array = []
            elif current_array and current_array[0] == '[':
                current_array.append(line.rstrip(','))
        
        # Remove duplicates and return
        return list(set(candidates))
    
    def _clean_json_response(self, response_text):
        """
        Clean common JSON formatting issues in LLM responses.
        """
        # Remove common prefixes/suffixes
        text = response_text.strip()
        
        # Remove explanatory text before/after JSON
        prefixes_to_remove = [
            r'^.*?(?=\[)',  # Remove everything before first [
            r'^[^[]*',      # Alternative: remove non-bracket content at start
        ]
        
        for pattern in prefixes_to_remove:
            text = re.sub(pattern, '', text, flags=re.DOTALL)
            if text.startswith('['):
                break
        
        # Remove explanatory text after JSON
        if ']' in text:
            bracket_pos = text.rfind(']')
            text = text[:bracket_pos + 1]
        
        # Fix common JSON issues
        fixes = [
            # Fix trailing commas
            (r',\s*]', ']'),
            (r',\s*}', '}'),
            # Fix missing quotes around strings
            (r'\[\s*([^"\[\]]+?)\s*\]', r'["\1"]'),
            # Fix unquoted strings in arrays
            (r'(?<=[\[,\s])([a-zA-Z_][a-zA-Z0-9_\s]*?)(?=[\],\s])', r'"\1"'),
            # Fix single quotes to double quotes
            (r"'([^']*)'", r'"\1"'),
            # Fix extra spaces
            (r'\s+', ' '),
            # Fix missing commas between array elements
            (r'"\s*"', '", "'),
            # Fix bracket spacing
            (r'\[\s+', '['),
            (r'\s+\]', ']'),
        ]
        
        for pattern, replacement in fixes:
            text = re.sub(pattern, replacement, text)
        
        return text.strip()
    
    def _parse_fix_json_format(self, response_text, expected_count):
        """
        Attempt to fix malformed JSON and re-parse.
        """
        # This will be used for attempt 2 when we send back malformed JSON to LLM
        # For now, use enhanced cleaning
        cleaned = self._clean_json_response(response_text)
        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                return [str(item) for item in data]  # Convert all to strings
        except json.JSONDecodeError:
            pass
        return None
    
    def _parse_ultra_simple_format(self, response_text, expected_count):
        """
        Parse ultra-simple format: one speaker per line.
        """
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        
        # Filter out obvious non-speakers
        speakers = []
        for line in lines:
            # Remove numbering if present
            line = re.sub(r'^\d+\.\s*', '', line)
            line = re.sub(r'^[-*]\s*', '', line)
            
            # Remove quotes if present
            line = line.strip('"\'')
            
            if line and not line.lower().startswith(('output', 'result', 'answer')):
                speakers.append(line)
        
        return speakers[:expected_count] if speakers else None
    
    def _smart_padding_truncation(self, speakers, expected_count):
        """
        Intelligently pad or truncate speaker array to match expected count.
        """
        if len(speakers) == expected_count:
            return speakers
        
        if len(speakers) < expected_count:
            # Pad with AMBIGUOUS
            padded = speakers + ["AMBIGUOUS"] * (expected_count - len(speakers))
            self.logger.info(f"Padded speaker array from {len(speakers)} to {expected_count}")
            return padded
        else:
            # Truncate to expected count
            truncated = speakers[:expected_count]
            self.logger.warning(f"Truncated speaker array from {len(speakers)} to {expected_count}")
            return truncated
    
    def _validate_and_clean_speakers(self, speakers, text_metadata, numbered_lines=None):
        """
        Validate and clean speaker names, including fixing character name association errors.
        """
        cleaned_speakers = []
        known_characters = set()
        
        if text_metadata:
            known_characters = text_metadata.get('potential_character_names', set())
        
        for i, speaker in enumerate(speakers):
            speaker = speaker.strip().strip('"').strip("'")
            
            # Apply character name normalization
            normalized_speaker = self._normalize_speaker_name(speaker)
            
            # Filter out obvious metadata
            if self._is_metadata_speaker(normalized_speaker):
                cleaned_speakers.append("narrator")
            elif normalized_speaker.lower() in ['ambiguous', 'unclear', 'unknown']:
                cleaned_speakers.append("AMBIGUOUS")
            else:
                # Check for character name association errors
                if numbered_lines and i < len(numbered_lines):
                    corrected_speaker = self._fix_character_association_error(
                        normalized_speaker, numbered_lines[i], known_characters
                    )
                    cleaned_speakers.append(corrected_speaker)
                else:
                    cleaned_speakers.append(normalized_speaker)
        
        return cleaned_speakers
    
    def _fix_character_association_error(self, assigned_speaker: str, text_line: str, known_characters: set) -> str:
        """
        Detect and fix cases where LLM incorrectly attributed narrative to character based on name presence.
        
        Args:
            assigned_speaker: Speaker assigned by LLM
            text_line: The actual text content
            known_characters: Set of known character names
            
        Returns:
            Corrected speaker (may be same as input if no error detected)
        """
        if assigned_speaker.lower() in ['narrator', 'ambiguous', 'unfixable']:
            return assigned_speaker
            
        # Only check if assigned speaker is a known character
        if assigned_speaker not in known_characters:
            return assigned_speaker
            
        text_line = text_line.strip()
        if not text_line:
            return assigned_speaker
            
        # Extract text content from numbered format if present
        if re.match(r'^\d+\.\s*', text_line):
            text_content = re.sub(r'^\d+\.\s*', '', text_line).strip()
        else:
            text_content = text_line
        
        # Check for narrative indicators that suggest this should be narrator, not character
        narrative_patterns = [
            # Action descriptions: "Character + action verb"
            rf'\\b{re.escape(assigned_speaker)}\\b\\s+(?:walked|ran|moved|stepped|turned|looked|watched|saw|heard|felt|thought|knew|realized|understood|remembered|noticed|observed)',
            rf'\\b{re.escape(assigned_speaker)}\\b\\s+(?:was|were|had|would|could|should|might|must)\\s+\\w+',
            
            # State descriptions: "Character + possessive + state"
            rf'\\b{re.escape(assigned_speaker)}\'s\\s+(?:blood|heart|mind|eyes|face|body|voice|hand|head)',
            rf'\\b{re.escape(assigned_speaker)}\'s\\s+(?:expression|feeling|emotion|pain|fear|anger|joy)',
            
            # Physical descriptions: "The X affecting Character"
            rf'(?:the|a|an)\\s+\\w+\\s+(?:was|were)\\s+(?:affecting|hurting|healing|changing|melting|burning)\\s+\\b{re.escape(assigned_speaker)}\\b',
            
            # Third person descriptions: "Character shot forward", "Character's blood"
            rf'\\b{re.escape(assigned_speaker)}\\b\\s+(?:shot|flew|charged|rushed|jumped|fell|collapsed|stood|sat|lay)',
            rf'\\b(?:the|his|her|their)\\s+\\w+\\s+(?:was|were)\\s+\\w+ing\\s+\\b{re.escape(assigned_speaker)}\\b',
        ]
        
        for pattern in narrative_patterns:
            if re.search(pattern, text_content, re.IGNORECASE):
                self.logger.debug(f"Fixed character association error: '{assigned_speaker}' -> 'narrator' for: {text_content[:50]}...")
                return "narrator"
        
        # Check for dialogue indicators that confirm the character is actually speaking
        dialogue_patterns = [
            rf'\\b{re.escape(assigned_speaker)}\\b\\s+(?:said|asked|replied|whispered|shouted|muttered|cried|exclaimed|declared|announced)',
            rf'"[^"]*"[^"]*\\b{re.escape(assigned_speaker)}\\b\\s+(?:said|asked|replied)',
            rf'\\b{re.escape(assigned_speaker)}\\b\\s*:\\s*["\']',  # Script format
        ]
        
        for pattern in dialogue_patterns:
            if re.search(pattern, text_content, re.IGNORECASE):
                # This confirms character is speaking, keep original assignment
                return assigned_speaker
        
        # Check if text contains quoted speech without attribution
        if '"' in text_content or '"' in text_content or '"' in text_content:
            # Has dialogue markers but no clear attribution, could be character speaking
            return assigned_speaker
        
        # If character name appears but no clear dialogue indicators, likely narrative
        if assigned_speaker.lower() in text_content.lower():
            self.logger.debug(f"Probable narrative about character: '{assigned_speaker}' -> 'narrator' for: {text_content[:50]}...")
            return "narrator"
        
        return assigned_speaker
    
    def _pad_speaker_array(self, speakers, expected_count):
        """
        Pad speaker array to expected count.
        """
        while len(speakers) < expected_count:
            speakers.append("AMBIGUOUS")
        
        # Truncate if too long
        return speakers[:expected_count]
    
    def _normalize_speaker_name(self, speaker_name: str) -> str:
        """
        Normalize speaker names returned by LLM by removing artifacts and standardizing format.
        
        Args:
            speaker_name: Raw speaker name from LLM
            
        Returns:
            Normalized speaker name
        """
        if not speaker_name:
            return ""
            
        name = speaker_name.strip()
        
        # Remove quotes and common artifacts
        name = name.strip('"').strip("'").strip()
        
        # Remove newlines and excessive whitespace
        name = re.sub(r'[\n\r\t]+', ' ', name)
        name = re.sub(r'\s+', ' ', name)
        
        # Handle special LLM responses
        if name.lower() in ['narrator', 'ambiguous', 'unfixable']:
            return name.lower()
        
        # Remove common artifacts in LLM responses
        artifacts_to_remove = [
            r'\s*\([^)]*\)\s*',     # Parenthetical additions
            r'\s*\[[^\]]*\]\s*',    # Bracketed additions  
            r'\s*\{[^}]*\}\s*',     # Curly brace additions
            r'^\s*speaker\s*[:=]\s*',  # "Speaker: Name" format
            r'^\s*character\s*[:=]\s*',  # "Character: Name" format
        ]
        
        for pattern in artifacts_to_remove:
            name = re.sub(pattern, '', name, flags=re.IGNORECASE)
        
        # Clean up punctuation but preserve apostrophes
        name = re.sub(r'^[^\w\']+|[^\w\']+$', '', name)
        
        # Proper capitalization for character names
        if name and name not in ['narrator', 'ambiguous', 'unfixable']:
            words = []
            for word in name.split():
                if word:
                    # Handle names with apostrophes properly
                    if "'" in word:
                        parts = word.split("'")
                        capitalized_parts = []
                        for i, part in enumerate(parts):
                            if part:
                                if i == 0 or len(part) > 1:
                                    capitalized_parts.append(part.capitalize())
                                else:
                                    capitalized_parts.append(part.lower())
                            else:
                                capitalized_parts.append(part)
                        words.append("'".join(capitalized_parts))
                    else:
                        words.append(word.capitalize())
            name = ' '.join(words)
        
        return name.strip()
    
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
                
            # Validate all items are strings and filter metadata speakers
            string_speakers = []
            for item in speakers:
                if isinstance(item, str):
                    cleaned_speaker = item.strip()
                    # Validate speaker is not metadata
                    if self._is_metadata_speaker(cleaned_speaker):
                        self.logger.debug(f"Filtered metadata speaker: {cleaned_speaker}")
                        string_speakers.append("narrator")  # Convert to appropriate fallback
                    else:
                        string_speakers.append(cleaned_speaker)
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
    
    def _is_metadata_speaker(self, speaker_name: str) -> bool:
        """
        Check if a speaker name is likely metadata rather than an actual character.
        
        Returns True if the speaker should be filtered out as metadata.
        """
        if not speaker_name:
            return True
            
        speaker_lower = speaker_name.lower().strip()
        
        # Enhanced metadata speakers to filter out (synchronized with RuleBasedAttributor)
        metadata_speakers = {
            # Book structure
            'chapter', 'prologue', 'epilogue', 'part', 'section', 'book', 'volume', 
            'page', 'appendix', 'index', 'glossary', 'bibliography', 'contents',
            'table of contents', 'toc', 'preface', 'foreword', 'introduction',
            'conclusion', 'afterword', 'postscript', 'dedication', 'acknowledgments',
            
            # Author/editorial
            'author', 'writer', 'editor', 'publisher', 'translator', 'narrator',
            'reader', 'storyteller', "author's note", "author's words", 
            'author note', 'author words', 'editorial', "editor's note",
            "publisher's note", "translator's note", 'note', 'notes',
            
            # Status/quality markers
            'unfixable', 'ambiguous', 'unknown', 'unclear', 'missing', 'error',
            'corrupted', 'incomplete', 'damaged', 'illegible',
            
            # Common non-character words
            'it', 'this', 'that', 'he', 'she', 'they', 'we', 'you', 'i',
            'here', 'there', 'now', 'then', 'when', 'where', 'what', 'who',
            'how', 'why', 'the', 'and', 'or', 'but', 'if', 'so',
            
            # Document artifacts
            'copyright', 'isbn', 'publication', 'edition', 'version', 'draft',
            'manuscript', 'document', 'file', 'text', 'content',
            
            # Formatting artifacts
            'bold', 'italic', 'underline', 'highlight', 'quote', 'citation',
            'footnote', 'endnote', 'reference', 'link', 'url', 'http', 'www'
        }
        
        if speaker_lower in metadata_speakers:
            return True
            
        # Enhanced metadata patterns (synchronized with RuleBasedAttributor)
        metadata_patterns = [
                # Chapter patterns
                r'^chapter\s+\d+', r'^chapter\s+[ivx]+', r'^ch\.\s*\d+', r'^chap\.\s*\d+',
                r'^chapter\s+\w+', r'chapter\s+\d+:', r'chapter\s+[ivx]+:',
                
                # Book structure patterns
                r'^epilogue', r'^prologue', r'^part\s+\d+', r'^book\s+\d+',
                r'^volume\s+\d+', r'^section\s+\d+', r'^appendix', r'^index',
                r'^preface', r'^foreword', r'^introduction', r'^conclusion',
                r'^afterword', r'^postscript', r'^dedication', r'^acknowledgments',
                
                # Author/editorial patterns
                r'^author:', r'^writer:', r'^editor:', r'^publisher:',
                r'^translator:', r'^note:', r'^notes:', r'^editorial:',
                r"author'?s?\s+note", r"author'?s?\s+words", r"editor'?s?\s+note",
                r"publisher'?s?\s+note", r"translator'?s?\s+note",
                
                # Numbering patterns
                r'^\d+\.\s*$', r'^[ivx]+\.\s*$', r'^\d+\s*$', r'^[ivx]+\s*$',
                r'^\(\d+\)$', r'^\([ivx]+\)$', r'^\[\d+\]$', r'^\[[ivx]+\]$',
                
                # Copyright and publication patterns
                r'^copyright', r'^©', r'^isbn', r'^publication', r'^edition',
                r'^version', r'^draft', r'^manuscript', r'^document',
                
                # Common metadata indicators
                r'^table\s+of\s+contents', r'^toc\s*:', r'^contents\s*:',
                r'^bibliography', r'^glossary', r'^references',
                
                # Page/location patterns
                r'^page\s+\d+', r'^p\.\s*\d+', r'^\d+\s*-\s*\d+$'
        ]
        
        for pattern in metadata_patterns:
            if re.match(pattern, speaker_lower):
                return True
        
        # Check for chapter-like patterns with numbers
        if re.match(r'^chapter\s+\w+', speaker_lower):
            return True
            
        # Check for single letters or very short names that are likely artifacts
        if len(speaker_name.strip()) <= 2 and speaker_name.strip().upper() in ['I', 'II', 'III', 'IV', 'V', 'A', 'B']:
            return True
            
        # Check for numeric patterns
        if re.match(r'^\d+\.?\s*$', speaker_name.strip()):
            return True
            
        return False

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