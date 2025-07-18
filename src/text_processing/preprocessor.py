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
        self.aliases = set()   # e.g., {"The Scholar", "Alex"}
        self.titles = set()    # e.g., {"Mr.", "Sir", "Captain"}
        self.confidence = 0.0  # Overall confidence in this being a character
        self.appearance_count = 0  # Number of times this character name appears
        
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
        
    def add_appearance(self, confidence_boost: float = 0.0):
        """Increment appearance count and optionally boost confidence."""
        self.appearance_count += 1
        if confidence_boost > 0:
            self.confidence = min(1.0, self.confidence + confidence_boost)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for metadata."""
        return {
            "name": self.name,
            "pronouns": list(self.pronouns),
            "aliases": list(self.aliases),
            "titles": list(self.titles),
            "confidence": self.confidence,
            "appearance_count": self.appearance_count
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
        Analyzes the raw text to extract structural hints, character profiles, and POV analysis.
        Enhanced with Ultrathink architecture's POV analysis.
        """
        metadata = {
            "dialogue_markers": set(),
            "scene_breaks": [], # List of character indices where scene breaks occur
            "character_profiles": [],  # Rich character profiles (NEW)
            "potential_character_names": set(),  # Maintained for backward compatibility
            "is_script_like": False,
            "pov_analysis": {}  # NEW: Point of view analysis (Ultrathink)
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

        # 3.7. POV Analysis (Ultrathink Architecture - Phase 1)
        pov_analysis = self.analyze_pov_profile(text)
        metadata['pov_analysis'] = pov_analysis

        # 4. Enhanced Character Profiling
        character_profiles = self._extract_character_profiles(text, doc)
        metadata['character_profiles'] = [profile.to_dict() for profile in character_profiles.values()]
        
        # Maintain backward compatibility
        metadata['potential_character_names'] = {profile.name for profile in character_profiles.values()}

        # 5. Enhanced Script-like Format Detection
        # Check for lines starting with character names followed by a colon
        script_line_count = 0
        stage_direction_count = 0
        total_lines = 0
        
        # Enhanced patterns for script detection
        script_patterns = [
            r'^[A-Z][A-Z0-9_\s]{1,50}:\s*',  # All caps character names
            r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+)*:\s*',  # Title case multi-word names
            r'^(?:First|Second|Third|Fourth|Fifth)\s+[A-Z][a-z]+:\s*',  # Numbered characters
        ]
        
        stage_direction_patterns = [
            r'^Enter\s+', r'^Exit\s+', r'^Exeunt\s*', 
            r'^(?:ACT|SCENE)\s+[IVX\d]+', r'^PROLOGUE\s*$', r'^EPILOGUE\s*$'
        ]
        
        for line in text.split('\n'):
            line = line.strip()
            if line:
                total_lines += 1
                
                # Check for script patterns
                if any(re.match(pattern, line) for pattern in script_patterns):
                    script_line_count += 1
                
                # Check for stage directions
                elif any(re.match(pattern, line, re.IGNORECASE) for pattern in stage_direction_patterns):
                    stage_direction_count += 1
        
        # More sophisticated script detection
        if total_lines > 10:
            script_ratio = script_line_count / total_lines
            stage_ratio = stage_direction_count / total_lines
            
            # Consider it script-like if:
            # - More than 15% of lines are script format, OR
            # - More than 8% script + stage directions combined, OR  
            # - Significant number of both script and stage direction patterns
            if (script_ratio > 0.15 or 
                (script_ratio + stage_ratio) > 0.08 or
                (script_line_count >= 5 and stage_direction_count >= 2)):
                metadata['is_script_like'] = True

        return metadata
    
    def _extract_character_profiles(self, text: str, doc) -> Dict[str, CharacterProfile]:
        """
        Extract rich character profiles including names, pronouns, aliases, and titles.
        
        Returns:
            Dictionary mapping canonical names to CharacterProfile objects
        """
        profiles = {}
        
        # Step 1: Extract character names with confidence scores from various patterns
        name_candidates_with_confidence = self._extract_character_names_with_confidence(text, doc)
        
        # Step 2: Create initial profiles with method-based confidence and appearance counts
        appearance_counts = getattr(self, '_last_appearance_counts', {})
        for name, confidence in name_candidates_with_confidence.items():
            if name not in profiles:
                profiles[name] = CharacterProfile(name)
                profiles[name].confidence = confidence  # Use extraction-method-based confidence
                profiles[name].appearance_count = appearance_counts.get(name, 1)  # Set appearance count
        
        # Step 3: Enhance profiles with pronouns, aliases, and titles
        self._detect_pronouns(text, doc, profiles)
        self._detect_aliases(text, doc, profiles)
        self._detect_titles(text, doc, profiles)
        
        # Step 4: Filter out low-confidence profiles and rare appearances
        filtered_profiles = {}
        for name, profile in profiles.items():
            # Confidence threshold
            meets_confidence = profile.confidence >= 0.5
            
            # Appearance threshold (more lenient for high-confidence extractions)
            min_appearances = 1 if profile.confidence >= 0.8 else 2
            meets_appearances = profile.appearance_count >= min_appearances
            
            if meets_confidence and meets_appearances:
                filtered_profiles[name] = profile
        
        # Step 5: Consolidate similar character names using fuzzy matching
        consolidated_profiles = self._consolidate_similar_profiles(filtered_profiles)
        
        return consolidated_profiles
    
    def _extract_character_names_with_confidence(self, text: str, doc) -> Dict[str, float]:
        """Extract character names with confidence scores and appearance counts based on extraction method."""
        name_confidence = {}
        name_appearances = {}  # Track appearances for each name
        
        # Pattern 1: Names with dialogue tags (HIGH confidence - 0.9)
        dialogue_pattern = r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(' + '|'.join(self.dialogue_tags) + r')'
        for match in re.finditer(dialogue_pattern, text):
            name = match.group(1).strip()
            normalized_name = self._normalize_character_name(name)
            if normalized_name and self._is_valid_character_name(normalized_name, doc, match.start(), match.end()):
                # Dialogue attribution is very reliable for character names
                name_confidence[normalized_name] = max(name_confidence.get(normalized_name, 0), 0.9)
                name_appearances[normalized_name] = name_appearances.get(normalized_name, 0) + 1
        
        # Pattern 2: Script format names (VERY HIGH confidence - 0.95)
        script_pattern = r'^(?:–|\s|-)?\s*([A-Z][a-zA-Z0-9_\s]+):\s*'
        for match in re.finditer(script_pattern, text, re.MULTILINE):
            name = match.group(1).strip()
            normalized_name = self._normalize_character_name(name)
            if normalized_name and self._is_valid_character_name(normalized_name, doc, match.start(), match.end()):
                # Script format is extremely reliable for character names
                name_confidence[normalized_name] = max(name_confidence.get(normalized_name, 0), 0.95)
                name_appearances[normalized_name] = name_appearances.get(normalized_name, 0) + 1
        
        # Pattern 3: Names with titles (HIGH confidence - 0.85)
        for title_pattern in self.title_patterns:
            pattern = title_pattern + r'\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)'
            for match in re.finditer(pattern, text):
                name = match.group(2).strip() if match.lastindex >= 2 else match.group(1).strip()
                normalized_name = self._normalize_character_name(name)
                if normalized_name and self._is_valid_character_name(normalized_name, doc, match.start(), match.end()):
                    # Titles indicate formal character references
                    name_confidence[normalized_name] = max(name_confidence.get(normalized_name, 0), 0.85)
                    name_appearances[normalized_name] = name_appearances.get(normalized_name, 0) + 1
        
        # Pattern 4: Possessives (MEDIUM confidence - 0.7)
        possessive_pattern = r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\'s\s'
        for match in re.finditer(possessive_pattern, text):
            name = match.group(1).strip()
            normalized_name = self._normalize_character_name(name)
            if normalized_name and self._is_valid_character_name(normalized_name, doc, match.start(), match.end()):
                # Possessives are somewhat reliable but can be false positives
                name_confidence[normalized_name] = max(name_confidence.get(normalized_name, 0), 0.7)
                name_appearances[normalized_name] = name_appearances.get(normalized_name, 0) + 1
        
        # Pattern 5: spaCy NER entities (MEDIUM-LOW confidence - 0.6)
        if doc:
            for ent in doc.ents:
                if ent.label_ == 'PERSON' and len(ent.text.strip()) > 2:
                    normalized_name = self._normalize_character_name(ent.text.strip())
                    if normalized_name and self._is_valid_character_name(normalized_name, doc, ent.start_char, ent.end_char):
                        # spaCy can have false positives, especially with author names
                        name_confidence[normalized_name] = max(name_confidence.get(normalized_name, 0), 0.6)
                        name_appearances[normalized_name] = name_appearances.get(normalized_name, 0) + 1
        
        # Store appearance counts in a way that the calling method can access them
        self._last_appearance_counts = name_appearances
        return name_confidence
    
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
        # Look for patterns like "Alex Johnson, also known as The Scholar"
        alias_patterns = [
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*),?\s+(?:also\s+)?(?:known\s+as|called)\s+([A-Z][^,.!?]*)',
            r'([A-Z][^,.!?]*),?\s+(?:also\s+)?(?:known\s+as|called)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)',
            r'"([^"]+)"\s*[,.]?\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:said|thought|whispered)',  # "The Scholar," John said
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:said|thought|whispered)[^.]*[,.]?\s*"([^"]+)"'   # John said, "The Scholar"
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
            
        # Enhanced blacklists for better character filtering
        
        # Common non-name words (expanded)
        non_names = {
            'the', 'and', 'but', 'for', 'with', 'from', 'into', 'then', 
            'here', 'there', 'this', 'that', 'what', 'where', 'when', 
            'why', 'how', 'chapter', 'section', 'part', 'book', 'page',
            'said', 'asked', 'replied', 'thought', 'whispered', 'shouted',
            # Pronouns that often get capitalized
            'it', 'he', 'she', 'they', 'we', 'you', 'i', 'me', 'him', 'her',
            'them', 'us', 'my', 'your', 'his', 'its', 'our', 'their',
            # Common demonstratives and articles
            'this', 'that', 'these', 'those', 'such', 'some', 'any', 'many', 'few',
            # Time/location words often capitalized
            'now', 'then', 'today', 'tomorrow', 'yesterday', 'here', 'there',
            # Project Gutenberg metadata terms
            'title', 'release date', 'language', 'credits', 'produced by',
            'most recently updated', 'file', 'encoding', 'utf', 'project gutenberg',
            'gutenberg', 'ebook', 'start', 'end', 'unfixable', 'ambiguous'
        }
        
        # Project Gutenberg specific metadata entities  
        pg_metadata_entities = {
            'george saintsbury', 'saintsbury', 'george allen', 'allen',
            'charing cross road', 'ruskin house', 'london', 'publisher',
            'first published', 'originally published', 'printed in',
            'copyright', 'all rights reserved', 'george allen and unwin',
            'title', 'release date', 'author', 'language', 'credits',
            'most recently updated', 'produced by', 'file produced',
            'project gutenberg', 'gutenberg ebook', 'utf-8', 'encoding'
        }
        
        # Famous authors that might appear in literary texts
        famous_authors = {
            # Korean authors (for webtoons/novels)
            'sing shinsong', 'sing shangshong', 'sing shang shong',
            # Japanese authors
            'murakami haruki', 'haruki murakami', 'yukio mishima', 'kazuo ishiguro',
            'banana yoshimoto', 'kobo abe', 'junichiro tanizaki', 'yasunari kawabata',
            # Western literary authors
            'raymond carver', 'ernest hemingway', 'james joyce', 'virginia woolf',
            'william faulkner', 'john steinbeck', 'f scott fitzgerald', 'mark twain',
            'charles dickens', 'jane austen', 'miss austen', 'george orwell', 'aldous huxley',
            'margaret atwood', 'toni morrison', 'maya angelou', 'sylvia plath',
            # Literary critics and scholars (especially for Jane Austen texts)
            'george saintsbury', 'saintsbury', 'literary critic', 'critic', 'scholar',
            # Korean authors (expanded)
            'han kang', 'kim young ha', 'cho nam joo', 'bae suah', 'park min gyu',
            'shin kyung sook', 'park wan suh', 'yi mun yol', 'gong ji young',
            # Contemporary authors
            'brandon sanderson', 'george r r martin', 'j k rowling', 'stephen king',
            'neil gaiman', 'terry pratchett', 'ursula k le guin', 'isaac asimov'
        }
        
        # Literary/publishing terms that aren't characters
        literary_terms = {
            'author', 'writer', 'novelist', 'poet', 'editor', 'publisher', 'translator',
            'protagonist', 'antagonist', 'narrator', 'reader', 'character',
            'chapter', 'prologue', 'epilogue', 'preface', 'foreword', 'afterword',
            'volume', 'edition', 'version', 'draft', 'manuscript', 'publication',
            'copyright', 'isbn', 'bibliography', 'glossary', 'index', 'appendix',
            'novel', 'story', 'tale', 'book', 'text', 'work', 'literature',
            'fiction', 'nonfiction', 'fantasy', 'romance', 'mystery', 'thriller',
            'science fiction', 'historical fiction', 'dystopian', 'utopian'
        }
        
        # Apply all blacklist filters
        name_lower = name.lower()
        
        if name_lower in non_names:
            return False
        
        # Check Project Gutenberg metadata entities
        if name_lower in pg_metadata_entities:
            return False
        
        # Check for partial matches with PG metadata (for variations)
        for entity in pg_metadata_entities:
            if entity in name_lower or name_lower in entity:
                # Be more strict with PG metadata - any partial match is rejected
                return False
        
        if name_lower in famous_authors:
            return False
            
        if name_lower in literary_terms:
            return False
            
        # Check for partial matches with famous authors (to catch variations)
        for author in famous_authors:
            if name_lower in author or author in name_lower:
                # Allow if the name is significantly longer (might be a character with similar name)
                if len(name_lower) > len(author) + 5:
                    continue
                return False
        
        # Enhanced pronoun detection (context-aware)
        if self._is_likely_pronoun_context(name, doc, match_start, match_end):
            return False
        
        # Use spaCy validation if available
        if doc and match_start != match_end:
            return self._is_proper_noun(name, doc, match_start, match_end)
        
        # Fallback validation
        return True

    def _is_likely_pronoun_context(self, name: str, doc, match_start: int, match_end: int) -> bool:
        """
        Enhanced pronoun detection using context analysis.
        
        Returns True if the name is likely a pronoun being misidentified as a character name.
        """
        name_lower = name.lower()
        
        # Direct pronoun check
        common_pronouns = {'it', 'he', 'she', 'they', 'this', 'that', 'these', 'those'}
        if name_lower in common_pronouns:
            return True
        
        # If no spaCy context available, use basic check
        if not doc or match_start == match_end:
            return name_lower in common_pronouns
            
        try:
            # Get the token for contextual analysis
            span = doc.char_span(match_start, match_end)
            if span is None:
                return name_lower in common_pronouns
                
            # Check if spaCy identifies this as a pronoun
            for token in span:
                if token.pos_ == 'PRON':
                    return True
                    
                # Additional check: if it's at sentence start but is a pronoun elsewhere
                if token.i > 0:  # Not the first token in document
                    prev_token = token.doc[token.i - 1]
                    # If previous token is sentence-ending punctuation, this might be 
                    # a pronoun that got capitalized due to sentence start
                    if prev_token.text in '.!?' and name_lower in common_pronouns:
                        return True
                        
        except Exception:
            # Fallback to basic check if spaCy processing fails
            return name_lower in common_pronouns
            
        return False

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
    
    def analyze_pov_profile(self, text: str) -> Dict[str, Any]:
        """
        Analyze the narrative point of view (POV) of the text using sophisticated pronoun analysis.
        
        This method implements the Ultrathink architecture's Phase 1: Dynamic POV Analysis.
        It analyzes a sample of the text to determine narrative perspective and identify narrators.
        
        Args:
            text: Full text to analyze
            
        Returns:
            Dictionary containing POV analysis results
        """
        # Extract sample text for analysis
        words = text.split()
        sample_size = min(settings.POV_SAMPLE_SIZE, len(words))
        sample_text = ' '.join(words[:sample_size])
        
        # Initialize POV analysis result
        pov_analysis = {
            'type': 'UNKNOWN',
            'confidence': 0.0,
            'narrator_identifier': None,
            'sample_stats': {
                'words_analyzed': sample_size,
                'first_person_count': 0,
                'third_person_count': 0,
                'first_person_ratio': 0.0,
                'third_person_ratio': 0.0
            },
            'perspective_shifts': [],
            'narrator_discovery': {
                'attempted': False,
                'success': False,
                'candidate_names': [],
                'selected_name': None
            }
        }
        
        # Count first-person and third-person pronouns
        first_person_pronouns = {'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'}
        third_person_pronouns = {'he', 'she', 'they', 'him', 'her', 'them', 'his', 'hers', 'their', 'theirs', 'himself', 'herself', 'themselves'}
        
        # Use word boundaries for accurate counting
        first_person_pattern = r'\b(?:' + '|'.join(first_person_pronouns) + r')\b'
        third_person_pattern = r'\b(?:' + '|'.join(third_person_pronouns) + r')\b'
        
        first_person_matches = re.findall(first_person_pattern, sample_text, re.IGNORECASE)
        third_person_matches = re.findall(third_person_pattern, sample_text, re.IGNORECASE)
        
        first_person_count = len(first_person_matches)
        third_person_count = len(third_person_matches)
        total_pronouns = first_person_count + third_person_count
        
        # Update sample stats
        pov_analysis['sample_stats'].update({
            'first_person_count': first_person_count,
            'third_person_count': third_person_count,
            'first_person_ratio': first_person_count / max(total_pronouns, 1),
            'third_person_ratio': third_person_count / max(total_pronouns, 1)
        })
        
        # Determine POV type using configurable threshold
        threshold = settings.POV_PRONOUN_WEIGHT_THRESHOLD
        
        if first_person_count > third_person_count * threshold:
            pov_analysis['type'] = 'FIRST_PERSON'
            pov_analysis['confidence'] = min(0.95, first_person_count / max(third_person_count, 1) / threshold)
            
            # Attempt narrator discovery for first-person texts
            if settings.POV_ENABLE_NARRATOR_DISCOVERY:
                narrator_info = self._discover_first_person_narrator(sample_text)
                pov_analysis['narrator_discovery'] = narrator_info
                pov_analysis['narrator_identifier'] = narrator_info.get('selected_name', settings.POV_FALLBACK_NARRATOR_ID)
            else:
                pov_analysis['narrator_identifier'] = settings.POV_FALLBACK_NARRATOR_ID
                
        elif third_person_count > first_person_count * threshold:
            pov_analysis['type'] = 'THIRD_PERSON'
            pov_analysis['confidence'] = min(0.95, third_person_count / max(first_person_count, 1) / threshold)
            pov_analysis['narrator_identifier'] = 'narrator'  # Standard third-person narrator
            
        else:
            pov_analysis['type'] = 'MIXED'
            pov_analysis['confidence'] = 0.5  # Mixed POV has moderate confidence
            pov_analysis['narrator_identifier'] = 'narrator'  # Default to standard narrator for mixed
            
            # Detect perspective shifts for mixed POV
            shifts = self._detect_perspective_shifts(sample_text)
            pov_analysis['perspective_shifts'] = shifts
        
        # Ensure minimum confidence threshold
        if pov_analysis['confidence'] < settings.POV_CONFIDENCE_THRESHOLD:
            pov_analysis['type'] = 'MIXED'
            pov_analysis['confidence'] = max(0.3, pov_analysis['confidence'])
        
        return pov_analysis
    
    def _discover_first_person_narrator(self, text: str) -> Dict[str, Any]:
        """
        Attempt to discover the narrator's name in first-person texts.
        
        Args:
            text: Sample text to analyze
            
        Returns:
            Dictionary containing narrator discovery results
        """
        discovery = {
            'attempted': True,
            'success': False,
            'candidate_names': [],
            'selected_name': None,
            'confidence': 0.0
        }
        
        # Pattern 1: "My name is X" or "I am X"
        name_introduction_patterns = [
            r'(?:my name is|i am|i\'m called|call me|i go by)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)[,.]?\s+(?:that\'s me|that was me|here)',
            r'i[,.]?\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)[,.]?\s+(?:was|am|have|had)'
        ]
        
        for pattern in name_introduction_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                candidate_name = match.strip()
                if self._is_valid_narrator_name(candidate_name):
                    discovery['candidate_names'].append({
                        'name': candidate_name,
                        'pattern': 'introduction',
                        'confidence': 0.9
                    })
        
        # Pattern 2: Dialogue attribution to first-person narrator
        # Look for patterns like: "Text," I said. or I said, "Text"
        dialogue_attribution_patterns = [
            r'"[^"]*[,.]?"\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:said|replied|asked|thought)',
            r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:said|replied|asked|thought)[,.]?\s+"[^"]*"'
        ]
        
        for pattern in dialogue_attribution_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                candidate_name = match.strip()
                if self._is_valid_narrator_name(candidate_name):
                    discovery['candidate_names'].append({
                        'name': candidate_name,
                        'pattern': 'dialogue_attribution',
                        'confidence': 0.7
                    })
        
        # Pattern 3: Others addressing the narrator
        # Look for patterns like: "Hello, X" or addressing patterns
        addressing_patterns = [
            r'"[^"]*[,.]?\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)[,.]?[^"]*"',
            r'"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)[,.]?\s+[^"]*"'
        ]
        
        for pattern in addressing_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                candidate_name = match.strip()
                if self._is_valid_narrator_name(candidate_name):
                    discovery['candidate_names'].append({
                        'name': candidate_name,
                        'pattern': 'addressing',
                        'confidence': 0.6
                    })
        
        # Select the best candidate name
        if discovery['candidate_names']:
            # Score candidates by confidence and frequency
            name_scores = {}
            for candidate in discovery['candidate_names']:
                name = candidate['name']
                confidence = candidate['confidence']
                if name in name_scores:
                    name_scores[name] = max(name_scores[name], confidence)
                else:
                    name_scores[name] = confidence
            
            # Select highest scoring name
            best_name = max(name_scores.items(), key=lambda x: x[1])
            discovery['selected_name'] = best_name[0]
            discovery['confidence'] = best_name[1]
            discovery['success'] = True
        
        return discovery
    
    def _detect_perspective_shifts(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect perspective shifts in mixed POV texts.
        
        Args:
            text: Sample text to analyze
            
        Returns:
            List of detected perspective shifts
        """
        shifts = []
        sentences = re.split(r'[.!?]+', text)
        
        current_pov = None
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Determine POV of this sentence
            first_person_count = len(re.findall(r'\b(?:i|me|my|myself)\b', sentence, re.IGNORECASE))
            third_person_count = len(re.findall(r'\b(?:he|she|they|him|her|them)\b', sentence, re.IGNORECASE))
            
            sentence_pov = None
            if first_person_count > third_person_count:
                sentence_pov = 'FIRST_PERSON'
            elif third_person_count > first_person_count:
                sentence_pov = 'THIRD_PERSON'
            
            # Detect shift
            if sentence_pov and current_pov and sentence_pov != current_pov:
                shifts.append({
                    'position': i,
                    'from_pov': current_pov,
                    'to_pov': sentence_pov,
                    'sentence_preview': sentence[:50] + '...' if len(sentence) > 50 else sentence
                })
            
            if sentence_pov:
                current_pov = sentence_pov
        
        return shifts
    
    def _is_valid_narrator_name(self, name: str) -> bool:
        """
        Validate if a candidate name is suitable as a narrator identifier.
        
        Args:
            name: Candidate narrator name
            
        Returns:
            True if the name is valid for use as narrator identifier
        """
        if not name or len(name) < 2 or len(name) > 30:
            return False
        
        # Must start with capital letter
        if not name[0].isupper():
            return False
        
        # Check for invalid patterns
        invalid_patterns = [
            r'^\d+$',  # Pure numbers
            r'^[^\w\s]+$',  # Pure punctuation
            r'chapter|section|page|book',  # Document structure words
            r'said|asked|replied|thought',  # Dialogue tags
            r'the|and|but|for|with',  # Common non-name words
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, name, re.IGNORECASE):
                return False
        
        return True
    
    def _normalize_character_name(self, raw_name: str) -> str:
        """
        Normalize character names by removing artifacts, formatting issues, and honorifics.
        
        Args:
            raw_name: Raw character name that may contain artifacts
            
        Returns:
            Cleaned and normalized character name, or empty string if invalid
        """
        if not raw_name:
            return ""
        
        name = raw_name
        
        # Step 1: Remove newlines, tabs, and excessive whitespace
        name = re.sub(r'[\n\r\t]+', ' ', name)
        name = re.sub(r'\s+', ' ', name)
        name = name.strip()
        
        # Step 2: Remove superscripts and subscripts (common in ebooks)
        # Remove Unicode superscript/subscript characters
        superscripts = '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᴬᴮᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᴿᵀᵁⱽᵂᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ'
        subscripts = '₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ'
        for char in superscripts + subscripts:
            name = name.replace(char, '')
        
        # Step 3: Remove common honorifics and titles (but preserve formal titles)
        honorifics_to_remove = [
            r'\b(?:Jr\.?|Sr\.?|III?|IV|V|VI|VII|VIII|IX|X)\b',  # Generational suffixes
            r'\b(?:-san|-kun|-chan|-sama|-senpai|-sensei)\b',    # Japanese honorifics
            r'\b(?:-nim|-ssi)\b',                               # Korean honorifics
            r'\s*\([^)]*\)\s*',                                 # Parenthetical additions
            r'\s*\[[^\]]*\]\s*',                               # Bracketed additions
        ]
        
        for pattern in honorifics_to_remove:
            name = re.sub(pattern, '', name, flags=re.IGNORECASE)
        
        # Step 4: Remove excessive punctuation but preserve apostrophes in names
        # Remove leading/trailing punctuation except apostrophes
        name = re.sub(r'^[^\w\']+|[^\w\']+$', '', name)
        
        # Remove multiple consecutive punctuation marks
        name = re.sub(r'[^\w\s\']{2,}', ' ', name)
        
        # Step 5: Handle special formatting artifacts
        # Remove zero-width characters and other invisible Unicode
        invisible_chars = '\u200b\u200c\u200d\u2060\ufeff'
        for char in invisible_chars:
            name = name.replace(char, '')
        
        # Remove HTML entities if present
        name = re.sub(r'&[a-zA-Z]+;', '', name)
        
        # Step 6: Fix capitalization issues
        # Split by spaces and capitalize each word properly
        words = []
        for word in name.split():
            if word:
                # Handle names with apostrophes (e.g., O'Connor, D'Artagnan)
                if "'" in word:
                    parts = word.split("'")
                    capitalized_parts = []
                    for i, part in enumerate(parts):
                        if part:
                            if i == 0 or len(part) > 1:  # Full capitalization for first part or long parts
                                capitalized_parts.append(part.capitalize())
                            else:  # Keep short parts after apostrophe lowercase (e.g., 'd)
                                capitalized_parts.append(part.lower())
                        else:
                            capitalized_parts.append(part)
                    words.append("'".join(capitalized_parts))
                else:
                    words.append(word.capitalize())
        
        name = ' '.join(words)
        
        # Step 7: Final validation and cleanup
        # Remove if the result is too short, too long, or contains invalid patterns
        if len(name) < 2 or len(name) > 50:
            return ""
        
        # Check for invalid patterns that suggest this isn't a real name
        invalid_patterns = [
            r'^\d+$',           # Pure numbers
            r'^[^\w\s]+$',      # Pure punctuation
            r'chapter|section|page|book|volume',  # Document structure words
            r'http[s]?://',     # URLs
            r'@\w+',            # Email addresses or handles
            r'^[A-Z]{3,}$',     # All caps abbreviations (unless exactly 2 chars)
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, name, re.IGNORECASE):
                return ""
        
        # Step 8: Handle special cases for compound names
        # Ensure compound names are properly formatted
        name = re.sub(r'\s+', ' ', name)  # Final whitespace cleanup
        
        return name.strip()

    def _consolidate_similar_profiles(self, profiles: Dict[str, CharacterProfile]) -> Dict[str, CharacterProfile]:
        """
        Consolidate similar character profiles using fuzzy string matching.
        
        This resolves issues like "Alex Johnson" vs "Alex John" being treated as separate characters.
        
        Args:
            profiles: Dictionary mapping names to CharacterProfile objects
            
        Returns:
            Consolidated dictionary with merged similar profiles
        """
        if len(profiles) <= 1:
            return profiles
            
        consolidated = {}
        processed_names = set()
        
        # Convert to list for easier processing
        profile_items = list(profiles.items())
        
        for i, (name1, profile1) in enumerate(profile_items):
            if name1 in processed_names:
                continue
                
            # Start with the current profile as the canonical one
            canonical_profile = profile1
            canonical_name = name1
            similar_profiles = [profile1]
            
            # Compare with all remaining profiles
            for j, (name2, profile2) in enumerate(profile_items[i+1:], i+1):
                if name2 in processed_names:
                    continue
                    
                # Calculate similarity between names
                similarity = fuzz.ratio(name1.lower(), name2.lower())
                
                # Also check if one name is contained in the other (for partial matches)
                name1_lower = name1.lower()
                name2_lower = name2.lower()
                is_substring = (name1_lower in name2_lower) or (name2_lower in name1_lower)
                
                # Consider them similar if:
                # 1. High fuzzy similarity (85%+), OR
                # 2. One is a substring of the other with reasonable length difference
                should_merge = (
                    similarity >= 85 or
                    (is_substring and abs(len(name1) - len(name2)) <= 5)
                )
                
                if should_merge:
                    # Choose the better name (prefer longer, higher confidence)
                    if (len(name2) > len(canonical_name) or 
                        (len(name2) == len(canonical_name) and profile2.confidence > canonical_profile.confidence)):
                        canonical_name = name2
                        canonical_profile = profile2
                    
                    similar_profiles.append(profile2)
                    processed_names.add(name2)
                    
            # Create consolidated profile
            if len(similar_profiles) > 1:
                # Merge all similar profiles into the canonical one
                for profile in similar_profiles:
                    if profile != canonical_profile:
                        # Merge pronouns, aliases, and titles
                        canonical_profile.pronouns.update(profile.pronouns)
                        canonical_profile.aliases.update(profile.aliases)
                        canonical_profile.titles.update(profile.titles)
                        
                        # Use highest confidence
                        canonical_profile.confidence = max(canonical_profile.confidence, profile.confidence)
                        
                        # Sum appearance counts for better character frequency assessment
                        canonical_profile.appearance_count += profile.appearance_count
                        
                        # Add the original names as aliases if they're different
                        if profile.name != canonical_profile.name:
                            canonical_profile.aliases.add(profile.name)
            
            consolidated[canonical_name] = canonical_profile
            processed_names.add(canonical_name)
            
        return consolidated
