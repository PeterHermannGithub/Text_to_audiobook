import re
import spacy
from fuzzywuzzy import fuzz, process
from config import settings
from typing import Dict, List, Set, Any
from collections import defaultdict

class CharacterProfile:
    """
    Rich character profile containing name, pronouns, aliases, and titles.
    """
    def __init__(self, name: str):
        self.name = name
        self.pronouns = set()  # e.g., {"he", "his", "him"}
        self.aliases = set()   # e.g., {"The Fool", "Dokja-ssi"}
        self.titles = set()    # e.g., {"Mr.", "Sir", "Captain"}
        self.confidence = 0.0  # Overall confidence in this being a character
        
    def add_pronoun(self, pronoun: str, confidence: float = 1.0):
        """Add a pronoun with confidence weighting."""
        self.pronouns.add(pronoun.lower())
        self.confidence = max(self.confidence, confidence)
        
    def add_alias(self, alias: str, confidence: float = 1.0):
        """Add an alias with confidence weighting."""
        self.aliases.add(alias)
        self.confidence = max(self.confidence, confidence)
        
    def add_title(self, title: str, confidence: float = 1.0):
        """Add a title with confidence weighting."""
        self.titles.add(title)
        self.confidence = max(self.confidence, confidence)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for metadata."""
        return {
            "name": self.name,
            "pronouns": list(self.pronouns),
            "aliases": list(self.aliases),
            "titles": list(self.titles),
            "confidence": self.confidence
        }

class TextPreprocessor:
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        
        # Pronoun mappings for gender detection
        self.pronoun_groups = {
            'male': {'he', 'him', 'his', 'himself'},
            'female': {'she', 'her', 'hers', 'herself'},
            'neutral': {'they', 'them', 'their', 'theirs', 'themselves', 'it', 'its', 'itself'}
        }
        
        # Common titles that indicate characters
        self.title_patterns = [
            r'\b(Mr|Mrs|Ms|Dr|Sir|Captain|Lord|Lady|Professor|President|King|Queen|Prince|Princess|Duke|Duchess)\.',
            r'\b(Master|Mister|Miss|Madam|Dame)\b'
        ]
        
        # Dialogue attribution tags
        self.dialogue_tags = {
            'said', 'replied', 'asked', 'whispered', 'shouted', 'muttered', 
            'cried', 'exclaimed', 'sighed', 'laughed', 'nodded', 'smiled', 
            'thought', 'continued', 'added', 'answered', 'responded'
        }

    def analyze(self, text):
        """
        Analyzes the raw text to extract structural hints and rich character profiles.
        """
        metadata = {
            "dialogue_markers": set(),
            "scene_breaks": [], # List of character indices where scene breaks occur
            "character_profiles": [],  # Rich character profiles (NEW)
            "potential_character_names": set(),  # Maintained for backward compatibility
            "is_script_like": False
        }

        # 1. Normalization
        # Convert smart quotes to straight quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        # Standardize ellipses
        text = re.sub(r'\.{2,}', '...', text)

        # Process the entire text with spaCy once
        doc = None
        if self.nlp:
            doc = self.nlp(text)

        # 2. Dialogue Marker Detection
        if '"' in text: # Basic check for double quotes
            metadata['dialogue_markers'].add('""')
        if '—' in text: # Basic check for em-dashes
            metadata['dialogue_markers'].add('—')
        if "'" in text: # Basic check for single quotes
            metadata['dialogue_markers'].add("'")
        # Check for script/chat format (e.g., - Name: or Name:)
        if re.search(r'^(?:–|\s|-)?\s*[A-Z][a-zA-Z0-9_\s]*:\s*', text, re.MULTILINE):
            metadata['dialogue_markers'].add('script_format')

        # 3. Enhanced Scene Break Detection
        scene_breaks = self._detect_scene_breaks(text)
        metadata['scene_breaks'] = scene_breaks
        
        # 3.5. Document Structure Analysis
        document_structure = self._analyze_document_structure(text)
        metadata['document_structure'] = document_structure

        # 4. Enhanced Character Profiling
        character_profiles = self._extract_character_profiles(text, doc)
        metadata['character_profiles'] = [profile.to_dict() for profile in character_profiles.values()]
        
        # Maintain backward compatibility
        metadata['potential_character_names'] = {profile.name for profile in character_profiles.values()}

        # 5. Basic Script-like Format Detection
        # Check for lines starting with capitalized words followed by a colon
        script_line_count = 0
        total_lines = 0
        for line in text.split('\n'):
            line = line.strip()
            if line:
                total_lines += 1
                if re.match(r'^[A-Z][A-Z0-9_\s]*:\s*', line):
                    script_line_count += 1
        
        if total_lines > 10 and script_line_count / total_lines > 0.1: # More than 10% of lines look like script cues
            metadata['is_script_like'] = True

        return metadata
    
    def _extract_character_profiles(self, text: str, doc) -> Dict[str, CharacterProfile]:
        """
        Extract rich character profiles including names, pronouns, aliases, and titles.
        
        Returns:
            Dictionary mapping canonical names to CharacterProfile objects
        """
        profiles = {}
        
        # Step 1: Extract character names from various patterns
        name_candidates = self._extract_character_names(text, doc)
        
        # Step 2: Create initial profiles
        for name in name_candidates:
            if name not in profiles:
                profiles[name] = CharacterProfile(name)
                profiles[name].confidence = 0.7  # Base confidence for name detection
        
        # Step 3: Enhance profiles with pronouns, aliases, and titles
        self._detect_pronouns(text, doc, profiles)
        self._detect_aliases(text, doc, profiles)
        self._detect_titles(text, doc, profiles)
        
        # Step 4: Filter out low-confidence profiles
        filtered_profiles = {
            name: profile for name, profile in profiles.items()
            if profile.confidence >= 0.5  # Minimum confidence threshold
        }
        
        return filtered_profiles
    
    def _extract_character_names(self, text: str, doc) -> Set[str]:
        """Extract character names using various pattern matching techniques."""
        names = set()
        
        # Pattern 1: Names with dialogue tags
        dialogue_pattern = r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(' + '|'.join(self.dialogue_tags) + r')'
        for match in re.finditer(dialogue_pattern, text):
            name = match.group(1).strip()
            if self._is_valid_character_name(name, doc, match.start(), match.end()):
                names.add(name)
        
        # Pattern 2: Possessives (e.g., John's)
        possessive_pattern = r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\'s\s'
        for match in re.finditer(possessive_pattern, text):
            name = match.group(1).strip()
            if self._is_valid_character_name(name, doc, match.start(), match.end()):
                names.add(name)
        
        # Pattern 3: Names with titles
        for title_pattern in self.title_patterns:
            pattern = title_pattern + r'\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)'
            for match in re.finditer(pattern, text):
                name = match.group(2).strip() if match.lastindex >= 2 else match.group(1).strip()
                if self._is_valid_character_name(name, doc, match.start(), match.end()):
                    names.add(name)
        
        # Pattern 4: Script format names (CHARACTER:)
        script_pattern = r'^(?:–|\s|-)?\s*([A-Z][a-zA-Z0-9_\s]+):\s*'
        for match in re.finditer(script_pattern, text, re.MULTILINE):
            name = match.group(1).strip()
            if self._is_valid_character_name(name, doc, match.start(), match.end()):
                names.add(name)
        
        # Pattern 5: spaCy NER entities (if available)
        if doc:
            for ent in doc.ents:
                if ent.label_ == 'PERSON' and len(ent.text.strip()) > 2:
                    names.add(ent.text.strip())
        
        return names
    
    def _detect_pronouns(self, text: str, doc, profiles: Dict[str, CharacterProfile]):
        """Detect pronouns associated with character names."""
        if not doc:
            return
            
        # Look for patterns like "John said. He was tired."
        for name, profile in profiles.items():
            # Find sentences containing the character name
            sentences_with_name = []
            for sent in doc.sents:
                if name.lower() in sent.text.lower():
                    sentences_with_name.append(sent)
            
            # Look for pronouns in nearby sentences
            for sent in sentences_with_name:
                # Check next few sentences for pronouns
                sent_idx = list(doc.sents).index(sent)
                sentences_to_check = list(doc.sents)[sent_idx:sent_idx + 3]  # Current + next 2
                
                for check_sent in sentences_to_check:
                    for token in check_sent:
                        if token.pos_ == 'PRON' and token.text.lower() in [p for group in self.pronoun_groups.values() for p in group]:
                            # Determine gender group
                            pronoun = token.text.lower()
                            for gender, pronouns in self.pronoun_groups.items():
                                if pronoun in pronouns:
                                    profile.add_pronoun(pronoun, confidence=0.8)
                                    break
    
    def _detect_aliases(self, text: str, doc, profiles: Dict[str, CharacterProfile]):
        """Detect aliases and alternative names for characters."""
        # Look for patterns like "Kim Dokja, also known as The Fool"
        alias_patterns = [
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*),?\s+(?:also\s+)?(?:known\s+as|called)\s+([A-Z][^,.!?]*)',
            r'([A-Z][^,.!?]*),?\s+(?:also\s+)?(?:known\s+as|called)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)',
            r'"([^"]+)"\s*[,.]?\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:said|thought|whispered)',  # "The Fool," John said
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:said|thought|whispered)[^.]*[,.]?\s*"([^"]+)"'   # John said, "The Fool"
        ]
        
        for pattern in alias_patterns:
            for match in re.finditer(pattern, text):
                name1, name2 = match.group(1).strip(), match.group(2).strip()
                
                # Check which one is already in profiles
                if name1 in profiles and name2 not in profiles:
                    profiles[name1].add_alias(name2, confidence=0.7)
                elif name2 in profiles and name1 not in profiles:
                    profiles[name2].add_alias(name1, confidence=0.7)
    
    def _detect_titles(self, text: str, doc, profiles: Dict[str, CharacterProfile]):
        """Detect titles associated with character names."""
        for title_pattern in self.title_patterns:
            pattern = title_pattern + r'\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)'
            for match in re.finditer(pattern, text):
                title = match.group(1)
                name = match.group(2) if match.lastindex >= 2 else match.group(1)
                
                # Find matching profile by name
                for profile_name, profile in profiles.items():
                    if name == profile_name or name in profile.aliases:
                        profile.add_title(title, confidence=0.9)
                        break
    
    def _is_valid_character_name(self, name: str, doc, match_start: int = 0, match_end: int = 0) -> bool:
        """Enhanced validation for character names."""
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
            'why', 'how', 'chapter', 'section', 'part', 'book', 'page',
            'said', 'asked', 'replied', 'thought', 'whispered', 'shouted'
        }
        
        if name.lower() in non_names:
            return False
            
        # Use spaCy validation if available
        if doc and match_start != match_end:
            return self._is_proper_noun(name, doc, match_start, match_end)
        
        # Fallback validation
        return True

    def _is_proper_noun(self, name, doc, match_start, match_end):
        """
        Uses spaCy to verify if a matched name is a proper noun or a PERSON entity.
        Falls back to a basic check if spaCy is not available or match is ambiguous.
        """
        if not doc:
            # Fallback for when spaCy model is not downloaded or doc is None
            return name.lower() not in ["the", "and", "but", "for", "with", "from", "into", "then", "here", "there"]

        # Get the spaCy Span object for the matched name
        span = doc.char_span(match_start, match_end)

        if span is None:
            # If char_span returns None, it means the match doesn't align with token boundaries.
            # Fallback to a basic check for this specific case.
            return name.lower() not in ["the", "and", "but", "for", "with", "from", "into", "then", "here", "there"]

        # Primary Check: Named Entity Recognition (NER) for PERSON
        for ent in span.ents:
            if ent.label_ == 'PERSON':
                return True

        # Secondary Check: Part-of-Speech (POS) Tagging for Proper Noun
        for token in span:
            if token.pos_ == 'PROPN':
                return True
        
        return False
    
    def _detect_scene_breaks(self, text: str) -> List[int]:
        """
        Detect scene breaks and chapter boundaries in the text.
        
        Returns:
            List of character indices where scene breaks occur
        """
        scene_breaks = []
        
        # Scene break patterns to detect
        scene_break_patterns = [
            # Chapter markers
            r'^\s*(?:CHAPTER|Chapter|chapter)\s+(?:\d+|[IVXLCDM]+|[A-Z]+)\s*$',
            r'^\s*(?:\d+|[IVXLCDM]+)\.\s*$',
            
            # Section breaks
            r'^\s*\*\s*\*\s*\*\s*$',
            r'^\s*-{3,}\s*$',
            r'^\s*={3,}\s*$',
            r'^\s*~{3,}\s*$',
            
            # Time/location transitions
            r'^\s*(?:Later|Meanwhile|Elsewhere|The next day|Hours later|Days later|Weeks later)\s*[.:]?\s*$',
            r'^\s*(?:At|In)\s+(?:the|a)\s+\w+[,.]?\s*$',
            
            # Perspective changes
            r'^\s*(?:From|In)\s+\w+\'s\s+(?:perspective|point of view|POV)\s*[.:]?\s*$',
            
            # Dialog scene markers
            r'^\s*\"\s*\*\s*\*\s*\*\s*\"\s*$',
            
            # Multiple blank lines (paragraph breaks that might indicate scenes)
            r'\n\s*\n\s*\n',
        ]
        
        # Find scene breaks
        for pattern in scene_break_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
                scene_breaks.append(match.start())
        
        # Detect implicit scene breaks based on context changes
        lines = text.split('\n')
        current_pos = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                current_pos += len(line) + 1  # +1 for newline
                continue
            
            # Check for major context shifts that suggest scene breaks
            context_shift_indicators = [
                # Time jumps
                r'\b(?:suddenly|meanwhile|later|after|before|when|then)\b',
                # Location changes  
                r'\b(?:outside|inside|upstairs|downstairs|nearby|elsewhere|back|away)\b',
                # Character introductions
                r'\b(?:entered|appeared|arrived|left|departed|returned)\b',
                # Emotional/tonal shifts
                r'\b(?:however|but|yet|still|though|although|nevertheless)\b'
            ]
            
            # Look for dramatic paragraph changes (length, style)
            if i > 0 and i < len(lines) - 1:
                prev_line = lines[i-1].strip()
                next_line = lines[i+1].strip() if i+1 < len(lines) else ""
                
                # Detect sudden style changes (very short line between longer paragraphs)
                if (len(line_stripped) < 30 and 
                    len(prev_line) > 100 and 
                    len(next_line) > 100):
                    scene_breaks.append(current_pos)
            
            current_pos += len(line) + 1
        
        # Remove duplicates and sort
        scene_breaks = sorted(list(set(scene_breaks)))
        
        # Filter out scene breaks that are too close together (within 100 characters)
        filtered_breaks = []
        for break_pos in scene_breaks:
            if not filtered_breaks or break_pos - filtered_breaks[-1] > 100:
                filtered_breaks.append(break_pos)
        
        return filtered_breaks
    
    def _analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """
        Analyze the overall document structure for better processing.
        
        Returns:
            Dictionary containing document structure analysis
        """
        structure = {
            'total_length': len(text),
            'line_count': len(text.split('\n')),
            'paragraph_count': 0,
            'dialogue_density': 0.0,
            'narrative_density': 0.0,
            'average_paragraph_length': 0,
            'has_chapters': False,
            'chapter_markers': [],
            'dialogue_style': 'mixed',  # 'quoted', 'dashed', 'mixed'
            'narrative_perspective': 'unknown',  # 'first', 'third', 'mixed'
            'estimated_genre': 'unknown',  # 'fiction', 'nonfiction', 'script', 'chat'
            'complexity_score': 0.0  # 0-1 scale for processing complexity
        }
        
        # Split into paragraphs (non-empty lines or double newlines)
        paragraphs = []
        current_paragraph = []
        
        for line in text.split('\n'):
            line_stripped = line.strip()
            if line_stripped:
                current_paragraph.append(line_stripped)
            else:
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
        
        # Add final paragraph if exists
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        structure['paragraph_count'] = len(paragraphs)
        
        if paragraphs:
            structure['average_paragraph_length'] = sum(len(p) for p in paragraphs) / len(paragraphs)
        
        # Analyze dialogue vs narrative content
        dialogue_chars = 0
        narrative_chars = 0
        total_chars = len(text)
        
        # Count characters in dialogue vs narrative
        dialogue_patterns = [r'"[^"]*"', r'"[^"]*"', r'"[^"]*"', r'—[^—\n]*']
        
        for pattern in dialogue_patterns:
            for match in re.finditer(pattern, text):
                dialogue_chars += len(match.group())
        
        narrative_chars = total_chars - dialogue_chars
        
        if total_chars > 0:
            structure['dialogue_density'] = dialogue_chars / total_chars
            structure['narrative_density'] = narrative_chars / total_chars
        
        # Detect chapter markers
        chapter_patterns = [
            r'(?:CHAPTER|Chapter|chapter)\s+(?:\d+|[IVXLCDM]+|[A-Z]+)',
            r'^\s*(?:\d+|[IVXLCDM]+)\.\s*(?:[A-Z][^.!?]*)?$'
        ]
        
        for pattern in chapter_patterns:
            matches = list(re.finditer(pattern, text, re.MULTILINE))
            if matches:
                structure['has_chapters'] = True
                structure['chapter_markers'].extend([m.group() for m in matches])
        
        # Determine dialogue style
        quote_count = text.count('"') + text.count('"') + text.count('"')
        dash_count = text.count('—') + text.count('–')
        
        if quote_count > dash_count * 2:
            structure['dialogue_style'] = 'quoted'
        elif dash_count > quote_count * 2:
            structure['dialogue_style'] = 'dashed'
        else:
            structure['dialogue_style'] = 'mixed'
        
        # Detect narrative perspective
        first_person_indicators = len(re.findall(r'\b(?:I|my|me|myself|we|us|our|ourselves)\b', text, re.IGNORECASE))
        third_person_indicators = len(re.findall(r'\b(?:he|she|they|him|her|them|his|hers|their)\b', text, re.IGNORECASE))
        
        if first_person_indicators > third_person_indicators * 1.5:
            structure['narrative_perspective'] = 'first'
        elif third_person_indicators > first_person_indicators * 1.5:
            structure['narrative_perspective'] = 'third'
        else:
            structure['narrative_perspective'] = 'mixed'
        
        # Estimate genre based on patterns
        script_indicators = len(re.findall(r'^\s*[A-Z][A-Z\s]+:\s*', text, re.MULTILINE))
        chat_indicators = len(re.findall(r'^\s*\[\d{1,2}:\d{2}\]|\<[^>]+\>', text, re.MULTILINE))
        
        if script_indicators > structure['line_count'] * 0.1:
            structure['estimated_genre'] = 'script'
        elif chat_indicators > 10:
            structure['estimated_genre'] = 'chat'
        elif structure['dialogue_density'] > 0.4:
            structure['estimated_genre'] = 'fiction'
        else:
            structure['estimated_genre'] = 'nonfiction'
        
        # Calculate complexity score (0-1 scale)
        complexity_factors = [
            structure['dialogue_density'],  # More dialogue = more complex
            min(1.0, len(structure['chapter_markers']) / 10),  # Chapters add complexity
            min(1.0, structure['paragraph_count'] / 100),  # More paragraphs = more complex
            1.0 if structure['dialogue_style'] == 'mixed' else 0.5,  # Mixed styles = complex
            1.0 if structure['narrative_perspective'] == 'mixed' else 0.3,  # Mixed perspective = complex
        ]
        
        structure['complexity_score'] = sum(complexity_factors) / len(complexity_factors)
        
        return structure
