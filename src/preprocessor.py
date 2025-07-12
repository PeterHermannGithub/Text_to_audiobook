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

        # 3. Scene Break Detection
        # Multiple consecutive newlines (e.g., 3 or more)
        for match in re.finditer(r'\n{3,}', text):
            metadata['scene_breaks'].append(match.start())
        # Specific patterns like *** or ###
        for match in re.finditer(r'\*{3,}|#{3,}', text):
            metadata['scene_breaks'].append(match.start())

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
