import re
import logging
from typing import List, Dict, Any, Tuple
from fuzzywuzzy import fuzz, process

class RuleBasedAttributor:
    """
    Handles high-confidence speaker attribution using deterministic rules.
    
    This class processes numbered lines from the DeterministicSegmenter and attempts
    to attribute speakers using pattern matching before any LLM processing.
    
    Lines are tagged as either:
    - ATTRIBUTED: Speaker successfully determined by rules
    - PENDING_AI: Requires LLM classification
    """
    
    # Attribution status constants
    ATTRIBUTED = "ATTRIBUTED"
    PENDING_AI = "PENDING_AI"
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common dialogue tags for speaker attribution
        self.dialogue_tags = [
            'said', 'replied', 'asked', 'whispered', 'shouted', 'muttered', 
            'cried', 'exclaimed', 'sighed', 'laughed', 'nodded', 'smiled', 
            'thought', 'continued', 'added', 'answered', 'responded',
            'declared', 'announced', 'stated', 'mentioned', 'noted'
        ]
        
        # High-confidence patterns for script format
        self.script_patterns = [
            r'^(?:–|\s|-)?\s*([A-Z][a-zA-Z0-9_\s\-\'\.]+):\s*(.*)',  # NAME: dialogue
            r'^([A-Z][A-Z0-9_\s]+):\s*(.*)',  # ALL_CAPS_NAME: dialogue
        ]
        
        # Dialogue attribution patterns
        self.dialogue_attribution_patterns = [
            # "dialogue," speaker said/asked/etc
            r'"([^"]*),"\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(' + '|'.join(self.dialogue_tags) + r')',
            # "dialogue." speaker said/asked/etc  
            r'"([^"]*)\.\s*"\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(' + '|'.join(self.dialogue_tags) + r')',
            # "dialogue!" speaker said/asked/etc
            r'"([^"]*)[!?]\s*"\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(' + '|'.join(self.dialogue_tags) + r')',
            # speaker said/asked, "dialogue"
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(' + '|'.join(self.dialogue_tags) + r'),\s*"([^"]*)"',
        ]
    
    def process_lines(self, numbered_lines: List[Dict[str, Any]], text_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process numbered lines and attempt rule-based speaker attribution.
        
        Args:
            numbered_lines: List of line objects with 'line_id' and 'text' keys
            text_metadata: Metadata containing character names and format info
            
        Returns:
            List of line objects with added 'attribution_status' and 'speaker' fields
        """
        known_character_names = text_metadata.get('potential_character_names', set())
        is_script_like = text_metadata.get('is_script_like', False)
        dialogue_markers = text_metadata.get('dialogue_markers', set())
        
        self.logger.info(f"Processing {len(numbered_lines)} lines with rule-based attribution")
        self.logger.debug(f"Known characters: {list(known_character_names)}")
        self.logger.debug(f"Script format: {is_script_like}, Dialogue markers: {dialogue_markers}")
        
        attributed_lines = []
        attribution_stats = {self.ATTRIBUTED: 0, self.PENDING_AI: 0}
        
        for line in numbered_lines:
            line_id = line['line_id']
            text = line['text']
            
            # Try rule-based attribution
            speaker, confidence = self._attribute_speaker(text, known_character_names, is_script_like)
            
            if speaker and confidence >= 0.8:  # High confidence threshold
                attributed_lines.append({
                    'line_id': line_id,
                    'text': text,
                    'speaker': speaker,
                    'attribution_status': self.ATTRIBUTED,
                    'attribution_confidence': confidence,
                    'attribution_method': 'rule_based'
                })
                attribution_stats[self.ATTRIBUTED] += 1
                self.logger.debug(f"Line {line_id}: Attributed to '{speaker}' (confidence: {confidence:.2f})")
            else:
                attributed_lines.append({
                    'line_id': line_id,
                    'text': text,
                    'speaker': None,
                    'attribution_status': self.PENDING_AI,
                    'attribution_confidence': confidence if speaker else 0.0,
                    'attribution_method': None
                })
                attribution_stats[self.PENDING_AI] += 1
                self.logger.debug(f"Line {line_id}: Requires AI processing (confidence: {confidence if speaker else 0.0:.2f})")
        
        self.logger.info(f"Rule-based attribution results: {attribution_stats[self.ATTRIBUTED]} attributed, {attribution_stats[self.PENDING_AI]} require AI")
        return attributed_lines
    
    def _attribute_speaker(self, text: str, known_character_names: set, is_script_like: bool) -> Tuple[str, float]:
        """
        Attempt to attribute a speaker to a text line using various rule-based methods.
        
        Returns:
            Tuple of (speaker_name, confidence_score) where confidence is 0.0-1.0
        """
        text = text.strip()
        if not text:
            return None, 0.0
        
        # Method 1: Script format detection (highest priority)
        if is_script_like or self._looks_like_script_line(text):
            speaker, confidence = self._attribute_script_format(text, known_character_names)
            if speaker:
                return speaker, min(confidence + 0.2, 1.0)  # Boost confidence for script format
        
        # Method 2: Dialogue attribution patterns
        speaker, confidence = self._attribute_dialogue_tags(text, known_character_names)
        if speaker:
            return speaker, confidence
            
        # Method 3: Direct character name presence in dialogue
        speaker, confidence = self._attribute_character_presence(text, known_character_names)
        if speaker:
            return speaker, confidence
            
        # Method 4: Narrative vs dialogue classification
        if self._is_likely_dialogue(text):
            return "AMBIGUOUS", 0.3  # Mark as dialogue but uncertain speaker
        else:
            return "narrator", 0.6  # Likely narrative text
    
    def _looks_like_script_line(self, text: str) -> bool:
        """Check if a line looks like script format (NAME: dialogue)."""
        for pattern in self.script_patterns:
            if re.match(pattern, text):
                return True
        return False
    
    def _attribute_script_format(self, text: str, known_character_names: set) -> Tuple[str, float]:
        """Attribute speaker using script format patterns."""
        for pattern in self.script_patterns:
            match = re.match(pattern, text)
            if match:
                potential_speaker = match.group(1).strip()
                
                # Clean up the speaker name
                potential_speaker = potential_speaker.rstrip(':').strip()
                
                # Check if it matches a known character
                if potential_speaker in known_character_names:
                    return potential_speaker, 0.95
                elif known_character_names:
                    # Try fuzzy matching
                    best_match = process.extractOne(potential_speaker, list(known_character_names), scorer=fuzz.token_set_ratio)
                    if best_match and best_match[1] > 85:
                        return best_match[0], 0.85
                
                # If it looks like a proper name, accept it even if not in known list
                if self._is_likely_character_name(potential_speaker):
                    return potential_speaker, 0.7
                    
        return None, 0.0
    
    def _attribute_dialogue_tags(self, text: str, known_character_names: set) -> Tuple[str, float]:
        """Attribute speaker using dialogue tag patterns like 'said John'."""
        for pattern in self.dialogue_attribution_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Extract speaker name from the matched groups
                speaker_name = None
                for group in match.groups():
                    if group and any(tag in group.lower() for tag in self.dialogue_tags):
                        continue  # Skip the dialogue tag itself
                    if group and self._is_likely_character_name(group):
                        speaker_name = group.strip()
                        break
                
                if speaker_name:
                    # Check against known characters
                    if speaker_name in known_character_names:
                        return speaker_name, 0.9
                    elif known_character_names:
                        best_match = process.extractOne(speaker_name, list(known_character_names), scorer=fuzz.token_set_ratio)
                        if best_match and best_match[1] > 80:
                            return best_match[0], 0.8
                    
                    # Accept if it looks like a proper name
                    return speaker_name, 0.7
                    
        return None, 0.0
    
    def _attribute_character_presence(self, text: str, known_character_names: set) -> Tuple[str, float]:
        """Attribute speaker based on character name presence in dialogue."""
        if not self._is_likely_dialogue(text) or not known_character_names:
            return None, 0.0
            
        best_match_score = 0
        best_match_name = None
        
        for char_name in known_character_names:
            # Use token_set_ratio for partial matches within the text
            score = fuzz.token_set_ratio(char_name.lower(), text.lower())
            if score > best_match_score and score > 70:  # Threshold for direct name presence
                best_match_score = score
                best_match_name = char_name
        
        if best_match_name:
            confidence = min(best_match_score / 100.0, 0.6)  # Cap confidence for this method
            return best_match_name, confidence
            
        return None, 0.0
    
    def _is_likely_dialogue(self, text: str) -> bool:
        """Check if text is likely dialogue based on markers."""
        dialogue_indicators = ['"', '"', '"', "'", '—', '–']
        return any(indicator in text for indicator in dialogue_indicators)
    
    def _is_likely_character_name(self, name: str) -> bool:
        """Check if a string looks like a character name."""
        name = name.strip()
        
        # Basic checks
        if len(name) < 2 or len(name) > 50:
            return False
            
        # Must start with capital letter
        if not name[0].isupper():
            return False
            
        # Check for common non-name words
        non_names = {
            'the', 'and', 'but', 'for', 'with', 'from', 'into', 'then', 
            'here', 'there', 'this', 'that', 'what', 'where', 'when', 
            'why', 'how', 'chapter', 'section', 'part', 'book', 'page'
        }
        
        if name.lower() in non_names:
            return False
            
        # Must contain only letters, spaces, apostrophes, hyphens, and periods
        if not re.match(r"^[A-Za-z\s\'\-\.]+$", name):
            return False
            
        # If it has multiple words, each should be capitalized (except common particles)
        words = name.split()
        particles = {'de', 'la', 'le', 'du', 'von', 'van', 'da', 'di', 'del'}
        
        for i, word in enumerate(words):
            if i > 0 and word.lower() in particles:
                continue  # Allow lowercase particles
            if not word[0].isupper():
                return False
                
        return True
    
    def get_pending_lines(self, attributed_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract lines that require AI processing."""
        return [line for line in attributed_lines if line['attribution_status'] == self.PENDING_AI]
    
    def get_attributed_lines(self, attributed_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract lines that were successfully attributed by rules."""
        return [line for line in attributed_lines if line['attribution_status'] == self.ATTRIBUTED]