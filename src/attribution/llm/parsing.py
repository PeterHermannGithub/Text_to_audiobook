import json
import re
import logging
from typing import List, Optional, Any


class JSONParser:
    """
    Handles all JSON parsing, cleaning, and extraction for LLM responses.
    Extracted from llm_orchestrator.py for better separation of concerns.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def bulletproof_json_parse(self, response_text: str, expected_count: int, attempt: int) -> Optional[List[str]]:
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
        json_candidates = self.extract_json_candidates(response_text)
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
        cleaned_response = self.clean_json_response(response_text)
        if cleaned_response != response_text:
            try:
                data = json.loads(cleaned_response)
                if isinstance(data, list) and all(isinstance(item, str) for item in data):
                    self.logger.debug(f"Cleaned JSON parse successful (attempt {attempt + 1})")
                    return data
            except json.JSONDecodeError:
                pass
        
        return None
    
    def extract_json_candidates(self, text: str) -> List[str]:
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
                current_array.append(line)
        
        return candidates
    
    def clean_json_response(self, response_text: str) -> str:
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
    
    def parse_structured_json(self, response_text: str) -> List[str]:
        """
        Parses the LLM's output, expecting a JSON array of strings (paragraphs).
        """
        cleaned_text = self.extract_json_from_text(response_text)
        if not cleaned_text:
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
                    return []
            else:
                return []
        except json.JSONDecodeError:
            return []

    def extract_json_from_text(self, text: str) -> str:
        """
        Extracts and cleans the JSON array string from LLM response, handling common formatting issues.
        """
        # Remove markdown code block fences (```json, ```, etc.)
        text = re.sub(r'^```(?:json)?\s*|```\s*$', '', text, flags=re.MULTILINE).strip()
        
        # Find the first '[' and the last ']' to isolate the JSON array
        start_index = text.find('[')
        end_index = text.rfind(']')

        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_string = text[start_index:end_index + 1]
            return self.clean_json_string(json_string)
        
        return ""

    def clean_json_string(self, json_string: str) -> str:
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
                    # Try to fix quote issues in this line
                    # Count quotes to see if they're balanced
                    quote_count = line.count('"')
                    if quote_count > 2 and quote_count % 2 == 0:
                        # Even number of quotes > 2, likely has unescaped internal quotes
                        # Simple fix: escape quotes that aren't at start/end of strings
                        fixed_line = self._fix_internal_quotes(line)
                        fixed_lines.append(fixed_line)
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            
            return '\n'.join(fixed_lines)
    
    def parse_speaker_array_enhanced(self, response_text: str, expected_count: int, attempt: int) -> Optional[List[str]]:
        """
        Bulletproof speaker array parsing with multi-stage validation and progressive fallback.
        """
        if not response_text or not response_text.strip():
            self.logger.warning("Empty LLM response")
            return None
        
        # Stage 1: Multi-step JSON validation and parsing
        speakers = self.bulletproof_json_parse(response_text, expected_count, attempt)
        if speakers and len(speakers) == expected_count:
            return speakers
        
        # Stage 2: Alternative parsing strategies based on attempt number
        if attempt == 1:
            speakers = self.parse_speaker_array_alternative(response_text, expected_count)
        elif attempt == 2:
            speakers = self.parse_fix_json_format(response_text, expected_count)
        elif attempt >= 3:
            speakers = self.parse_ultra_simple_format(response_text, expected_count)
        
        if speakers and len(speakers) == expected_count:
            return speakers
        
        # Stage 3: No padding - return None if wrong length
        if speakers and len(speakers) > 0:
            self.logger.warning(f"Partial parse success: got {len(speakers)}, expected {expected_count} - FAILING instead of padding")
            return None
        
        return None
    
    def parse_speaker_array_alternative(self, response_text: str, expected_count: int) -> Optional[List[str]]:
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
    
    def parse_fix_json_format(self, response_text: str, expected_count: int) -> Optional[List[str]]:
        """
        Attempt to fix malformed JSON and re-parse.
        """
        # This will be used for attempt 2 when we send back malformed JSON to LLM
        # For now, use enhanced cleaning
        cleaned = self.clean_json_response(response_text)
        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                return [str(item) for item in data]  # Convert all to strings
        except json.JSONDecodeError:
            pass
        return None
    
    def parse_ultra_simple_format(self, response_text: str, expected_count: int) -> Optional[List[str]]:
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
    
    def _fix_internal_quotes(self, line: str) -> str:
        """
        Helper method to fix unescaped quotes within JSON strings.
        """
        # This is a heuristic approach - not perfect but handles common cases
        # Look for pattern: "text with "internal quotes" more text"
        pattern = r'"([^"]*)"([^"]*)"([^"]*)"'
        match = re.search(pattern, line)
        
        if match:
            # Escape the internal quotes
            fixed = f'"{match.group(1)}\\"{{match.group(2)}}\\"{match.group(3)}"'
            return line.replace(match.group(0), fixed)
        
        return line