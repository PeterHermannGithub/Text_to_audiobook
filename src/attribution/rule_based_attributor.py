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
        
        # Expanded dialogue tags for speaker attribution
        self.dialogue_tags = [
            # Speaking verbs
            'said', 'replied', 'asked', 'whispered', 'shouted', 'muttered', 
            'cried', 'exclaimed', 'sighed', 'laughed', 'continued', 'added', 
            'answered', 'responded', 'declared', 'announced', 'stated', 
            'mentioned', 'noted', 'called', 'yelled', 'screamed', 'gasped',
            'breathed', 'hissed', 'growled', 'barked', 'snapped', 'drawled',
            'chuckled', 'giggled', 'sobbed', 'wailed', 'moaned', 'groaned',
            
            # Action verbs that often accompany dialogue
            'nodded', 'smiled', 'grinned', 'frowned', 'shrugged', 'gestured',
            'pointed', 'waved', 'turned', 'looked', 'glanced', 'stared',
            
            # Thinking/internal verbs
            'thought', 'wondered', 'realized', 'remembered', 'considered',
            'decided', 'concluded', 'assumed', 'believed', 'knew', 'felt'
        ]
        
        # High-confidence patterns for script format
        self.script_patterns = [
            r'^(?:–|\s|-)?\s*([A-Z][a-zA-Z0-9_\s\-\'\.]+):\s*(.*)',  # NAME: dialogue
            r'^([A-Z][A-Z0-9_\s]+):\s*(.*)',  # ALL_CAPS_NAME: dialogue
        ]
        
        # Comprehensive dialogue attribution patterns
        self.dialogue_attribution_patterns = [
            # "dialogue," speaker said/asked/etc
            r'"([^"]*),"\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(' + '|'.join(self.dialogue_tags) + r')',
            # "dialogue." speaker said/asked/etc  
            r'"([^"]*)\.\s*"\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(' + '|'.join(self.dialogue_tags) + r')',
            # "dialogue!" speaker said/asked/etc
            r'"([^"]*)[!?]\s*"\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(' + '|'.join(self.dialogue_tags) + r')',
            # speaker said/asked, "dialogue"
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(' + '|'.join(self.dialogue_tags) + r'),\s*"([^"]*)"',
            
            # Enhanced patterns for better coverage
            # "dialogue," said speaker
            r'"([^"]*),"\s+(' + '|'.join(self.dialogue_tags) + r')\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)',
            # speaker's dialogue patterns
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s*[\':s]+\s*"([^"]*)"',
            # Character action followed by dialogue
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(' + '|'.join(self.dialogue_tags[:10]) + r')[^.]*\.\s*"([^"]*)"',
            # "dialogue" - speaker pattern
            r'"([^"]*)",\s*[-–—]\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)',
            # em-dash dialogue: —Dialogue, speaker said
            r'[–—]\s*"?([^"]*?)"?,?\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(' + '|'.join(self.dialogue_tags) + r')',
            # Curly quote patterns
            r'([“"\'‘])(.*?)([”"\'’]),\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(' + '|'.join(self.dialogue_tags) + r')'
        ]
        
        # Enhanced metadata speakers to filter out (should never be actual character speakers)
        self.metadata_speakers = {
            # Book structure
            'chapter', 'prologue', 'epilogue', 'part', 'section', 'book', 'volume', 
            'page', 'appendix', 'index', 'glossary', 'bibliography', 'contents',
            'table of contents', 'toc', 'preface', 'foreword', 'introduction',
            'conclusion', 'afterword', 'postscript', 'dedication', 'acknowledgments',
            
            # Author/editorial
            'author', 'writer', 'editor', 'publisher', 'translator', 'narrator',
            'reader', 'storyteller', 'author\'s note', 'author\'s words', 
            'author note', 'author words', 'editorial', 'editor\'s note',
            'publisher\'s note', 'translator\'s note', 'note', 'notes',
            
            # Status/quality markers
            'unfixable', 'ambiguous', 'unknown', 'unclear', 'missing', 'eror',
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
        
        # Enhanced metadata patterns to detect and filter
        self.metadata_patterns = [
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
            r'author\'?s?\s+note', r'author\'?s?\s+words', r'editor\'?s?\s+note',
            r'publisher\'?s?\s+note', r'translator\'?s?\s+note',
            
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
            
        # Method 4: Action-based dialogue attribution
        speaker, confidence = self._attribute_action_based_dialogue(text, known_character_names)
        if speaker:
            return speaker, confidence
        
        # Method 5: Internal thought attribution  
        speaker, confidence = self._attribute_internal_thoughts(text, known_character_names)
        if speaker:
            return speaker, confidence
            
        # Method 6: Possessive dialogue attribution
        speaker, confidence = self._attribute_possessive_dialogue(text, known_character_names)
        if speaker:
            return speaker, confidence
            
        # Method 7: Enhanced character name proximity
        speaker, confidence = self._attribute_enhanced_proximity(text, known_character_names)
        if speaker:
            return speaker, confidence
            
        # Method 8: Narrative vs dialogue classification
        if self._is_likely_dialogue(text):
            return "AMBIGUOUS", 0.3  # Mark as dialogue but uncertain speaker
        elif self._is_obvious_narrative(text):
            return "narrator", 0.85  # High confidence for obvious narrative text
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
                
                # Check if it's a metadata speaker (should be filtered out)
                if self._is_metadata_speaker(potential_speaker):
                    return None, 0.0
                    
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
                    # Check if it's a metadata speaker (should be filtered out)
                    if self._is_metadata_speaker(speaker_name):
                        continue
                        
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
    
    def _attribute_action_based_dialogue(self, text: str, known_character_names: set) -> Tuple[str, float]:
        """
        Attribute speaker based on action descriptions followed by dialogue.
        Pattern: "Character did something. 'Dialogue.'"
        """
        # Look for character name followed by action, then dialogue
        action_dialogue_patterns = [
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(' + '|'.join(self.dialogue_tags[:15]) + r')[^.]*?\.\s*[\"'""']([^\"'""']*)[\"'""']',
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:turned|looked|glanced|stepped|moved|walked)[^.]*?\.\s*[\"'""']([^\"'""']*)[\"'""']',
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:paused|hesitated|waited)[^.]*?\.\s*[\"'""']([^\"'""']*)[\"'""']'
        ]
        
        for pattern in action_dialogue_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                potential_speaker = match.group(1).strip()
                
                # Validate against known characters
                if potential_speaker in known_character_names:
                    return potential_speaker, 0.85
                elif known_character_names:
                    best_match = process.extractOne(potential_speaker, list(known_character_names), scorer=fuzz.token_set_ratio)
                    if best_match and best_match[1] > 80:
                        return best_match[0], 0.75
                
                # Accept if it looks like a proper name
                if self._is_likely_character_name(potential_speaker):
                    return potential_speaker, 0.65
        
        return None, 0.0
    
    def _attribute_internal_thoughts(self, text: str, known_character_names: set) -> Tuple[str, float]:
        """
        Attribute speaker based on internal thought patterns.
        Pattern: "Character thought/wondered/realized..."
        """
        thought_patterns = [
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:thought|wondered|realized|remembered|considered|decided|knew|felt|believed)',
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:was thinking|had thought|could think|would think)',
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\'s\s+(?:mind|thoughts|brain)'
        ]
        
        for pattern in thought_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                potential_speaker = match.group(1).strip()
                
                # Validate against known characters
                if potential_speaker in known_character_names:
                    return potential_speaker, 0.80
                elif known_character_names:
                    best_match = process.extractOne(potential_speaker, list(known_character_names), scorer=fuzz.token_set_ratio)
                    if best_match and best_match[1] > 75:
                        return best_match[0], 0.70
                
                # Accept if it looks like a proper name
                if self._is_likely_character_name(potential_speaker):
                    return potential_speaker, 0.60
        
        return None, 0.0
    
    def _attribute_possessive_dialogue(self, text: str, known_character_names: set) -> Tuple[str, float]:
        """
        Attribute speaker based on possessive dialogue patterns.
        Pattern: "Character's voice", "John's words", etc.
        """
        possessive_patterns = [
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\'s\s+(?:voice|words|response|reply|answer)',
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\'s\s+(?:question|statement|comment|remark)'
        ]
        
        for pattern in possessive_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                potential_speaker = match.group(1).strip()
                
                # Only accept if it's a known character
                if potential_speaker in known_character_names:
                    return potential_speaker, 0.75
                elif known_character_names:
                    best_match = process.extractOne(potential_speaker, list(known_character_names), scorer=fuzz.token_set_ratio)
                    if best_match and best_match[1] > 80:
                        return best_match[0], 0.70
        
        return None, 0.0
    
    def _attribute_enhanced_proximity(self, text: str, known_character_names: set) -> Tuple[str, float]:
        """
        Enhanced character name proximity detection with better scoring.
        """
        if not self._is_likely_dialogue(text) or not known_character_names:
            return None, 0.0
        
        best_matches = []
        
        for char_name in known_character_names:
            # Check exact name matches
            if char_name.lower() in text.lower():
                best_matches.append((char_name, 0.85))
                continue
            
            # Check partial matches (first name, last name)
            name_parts = char_name.split()
            for part in name_parts:
                if len(part) > 2 and part.lower() in text.lower():
                    best_matches.append((char_name, 0.70))
                    break
            
            # Fuzzy matching for typos/variations
            similarity = fuzz.partial_ratio(char_name.lower(), text.lower())
            if similarity > 85:
                best_matches.append((char_name, similarity / 100.0 * 0.6))
        
        if best_matches:
            # Return the best match
            best_match = max(best_matches, key=lambda x: x[1])
            return best_match[0], best_match[1]
        
        return None, 0.0
    
    def _is_likely_dialogue(self, text: str) -> bool:
        """Check if text is likely dialogue based on markers."""
        dialogue_indicators = ['"', '"', '"', "'", '—', '–']
        return any(indicator in text for indicator in dialogue_indicators)
    
    def _is_obvious_narrative(self, text: str) -> bool:
        """
        Check if text is obviously narrative and should have high confidence.
        
        Returns True for simple descriptive text that clearly isn't dialogue
        and doesn't require LLM processing.
        """
        text = text.strip()
        
        # Must not contain any dialogue indicators
        if self._is_likely_dialogue(text):
            return False
        
        # Must be reasonably short and simple
        if len(text) > 200:
            return False
        
        # Check for dialogue attribution patterns (indicates character interaction)
        for pattern in self.dialogue_attribution_patterns[:5]:  # Check first 5 patterns
            if re.search(pattern, text, re.IGNORECASE):
                return False
        
        # Check for character interaction indicators
        interaction_indicators = [
            'said', 'replied', 'asked', 'whispered', 'shouted', 'told',
            'responded', 'answered', 'continued', 'added', 'called',
            'yelled', 'muttered', 'exclaimed'
        ]
        
        # If it contains multiple interaction indicators, it's likely complex dialogue
        interaction_count = sum(1 for indicator in interaction_indicators if indicator in text.lower())
        if interaction_count >= 2:
            return False
        
        # Check for possessive patterns indicating character dialogue
        possessive_patterns = [
            r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)*'s\s+(?:voice|words|response|question)",
            r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\s+(?:said|asked|replied|told)"
        ]
        
        for pattern in possessive_patterns:
            if re.search(pattern, text):
                return False
        
        # Check for script-like patterns
        if self._looks_like_script_line(text):
            return False
        
        # If it's simple, descriptive text without dialogue markers or character interactions
        # Examples: "This is the first paragraph.", "The room was quiet.", "It was a dark night."
        return True
    
    def _is_likely_character_name(self, name: str) -> bool:
        """Check if a string looks like a character name."""
        name = name.strip()
        
        # Basic checks
        if len(name) < 2 or len(name) > 50:
            return False
            
        # Must start with capital letter
        if not name[0].isupper():
            return False
            
        # Check if it's a metadata speaker first
        if self._is_metadata_speaker(name):
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
    
    def _is_metadata_speaker(self, speaker_name: str) -> bool:
        """
        Check if a speaker name is likely metadata rather than an actual character.
        
        Returns True if the speaker should be filtered out as metadata.
        """
        if not speaker_name:
            return True
            
        speaker_lower = speaker_name.lower().strip()
        
        # Check against known metadata speakers
        if speaker_lower in self.metadata_speakers:
            return True
            
        # Check against metadata patterns
        for pattern in self.metadata_patterns:
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
    
    def get_pending_lines(self, attributed_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract lines that require AI processing."""
        return [line for line in attributed_lines if line['attribution_status'] == self.PENDING_AI]
    
    def get_attributed_lines(self, attributed_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract lines that were successfully attributed by rules."""
        return [line for line in attributed_lines if line['attribution_status'] == self.ATTRIBUTED]