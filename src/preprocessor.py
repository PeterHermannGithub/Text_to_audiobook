import re
import spacy
from fuzzywuzzy import fuzz, process

class TextPreprocessor:
    def __init__(self, nlp_model):
        self.nlp = nlp_model

    def analyze(self, text):
        """
        Analyzes the raw text to extract structural hints and metadata.
        """
        metadata = {
            "dialogue_markers": set(),
            "scene_breaks": [], # List of character indices where scene breaks occur
            "potential_character_names": set(),
            "is_script_like": False
        }

        # 1. Normalization
        # Convert smart quotes to straight quotes
        text = text.replace('“', '"').replace('”', '"')
        text = text.replace('‘', "'").replace('’', "'")
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
        # Add more sophisticated regex for other markers if needed

        # 3. Scene Break Detection
        # Multiple consecutive newlines (e.g., 3 or more)
        for match in re.finditer(r'\n{3,}', text):
            metadata['scene_breaks'].append(match.start())
        # Specific patterns like *** or ###
        for match in re.finditer(r'\*{3,}|#{3,}', text):
            metadata['scene_breaks'].append(match.start())

        # 4. Character Name Heuristics
        # Find names followed by dialogue tags (e.g., "John said", "Mary replied")
        # Also consider possessives and titles
        name_patterns = [
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(said|replied|asked|whispered|shouted|muttered|cried|exclaimed|sighed|laughed|nodded|smiled|thought)', # Name + dialogue tag
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\'s\s', # Possessives (e.g., John's)
            r'(Mr\.|Mrs\.|Ms\.|Dr\.|Sir|Captain|Lord|Lady|Professor)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)', # Titles
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)(?:\s*[,.!?])?\s*["—]' # Name followed by punctuation and dialogue start
        ]
        for pattern in name_patterns:
            for match in re.finditer(pattern, text):
                # For titles, the name is in group 2
                if match.lastindex == 2 and match.group(1) in ["Mr.", "Mrs.", "Ms.", "Dr.", "Sir", "Captain", "Lord", "Lady", "Professor"]:
                    name = match.group(2).strip()
                else:
                    name = match.group(1).strip()
                
                if len(name) > 2 and self._is_proper_noun(name, doc, match.start(), match.end()):
                    metadata['potential_character_names'].add(name)

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
