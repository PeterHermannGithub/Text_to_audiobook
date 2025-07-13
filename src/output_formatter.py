import re
import logging
import unicodedata
from typing import List, Dict, Any, Optional

class OutputFormatter:
    """
    Post-processing formatter for cleaning up and standardizing output format.
    
    Handles:
    - Unicode character normalization and cleanup
    - Speaker name standardization and formatting
    - Text content cleanup (newlines, whitespace, artifacts)
    - Special character normalization
    - Consistent formatting standards
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Unicode character mappings for common problematic characters
        self.unicode_replacements = {
            # Em dashes and hyphens
            '\u2013': '–',  # en dash
            '\u2014': '—',  # em dash
            '\u2015': '―',  # horizontal bar
            '\u2212': '-',  # minus sign
            
            # Quotation marks
            '\u201c': '"',  # left double quotation mark
            '\u201d': '"',  # right double quotation mark
            '\u2018': "'",  # left single quotation mark
            '\u2019': "'",  # right single quotation mark
            '\u00ab': '"',  # left-pointing double angle quotation mark
            '\u00bb': '"',  # right-pointing double angle quotation mark
            
            # Special brackets and symbols
            '\u300c': '"',  # left corner bracket
            '\u300d': '"',  # right corner bracket
            '\u3008': '<',  # left angle bracket
            '\u3009': '>',  # right angle bracket
            '\u2026': '...',  # horizontal ellipsis
            '\u00a0': ' ',   # non-breaking space
            '\u2009': ' ',   # thin space
            '\u200b': '',    # zero-width space
            '\ufeff': '',    # zero-width no-break space (BOM)
            
            # Mathematical and currency symbols
            '\u00d7': 'x',   # multiplication sign
            '\u00f7': '/',   # division sign
            '\u2022': '•',   # bullet
            '\u25cf': '•',   # black circle
        }
        
        # Speaker name standardization patterns
        self.speaker_patterns = {
            # Remove common prefixes/suffixes
            'prefixes': ['mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sir', 'lady'],
            'suffixes': ['jr.', 'sr.', 'ii', 'iii', 'iv'],
            # Common speaker name variations to standardize
            'variations': {
                'narrator': ['narration', 'narrative', 'description', 'desc'],
                'unknown': ['unknown speaker', 'unidentified', 'unclear'],
                'ambiguous': ['uncertain', 'unclear speaker', 'unknown'],
            }
        }
        
        # Text cleanup patterns
        self.cleanup_patterns = [
            # Multiple whitespace
            (r'\s+', ' '),
            # Multiple newlines
            (r'\n\s*\n+', '\n\n'),
            # Leading/trailing whitespace on lines
            (r'(?m)^\s+|\s+$', ''),
            # Inconsistent quotation marks
            (r'[""]([^"""]*?)[""]', r'"\1"'),
            # Multiple periods/ellipsis
            (r'\.{4,}', '...'),
            # Spaces before punctuation
            (r'\s+([,.!?;:])', r'\1'),
            # Missing spaces after punctuation
            (r'([,.!?;:])([A-Za-z])', r'\1 \2'),
            # Em dash spacing
            (r'\s*—\s*', ' — '),
            # Parentheses spacing
            (r'\s*\(\s*', ' ('),
            (r'\s*\)\s*', ') '),
        ]
        
        # Reserved speaker names that should not be modified
        self.reserved_speakers = {
            'narrator', 'AMBIGUOUS', 'UNFIXABLE', 'unknown'
        }
    
    def format_output(self, structured_segments: List[Dict[str, Any]], 
                     preserve_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Main entry point for output formatting.
        
        Args:
            structured_segments: List of segments with speaker/text
            preserve_metadata: Whether to preserve error and metadata fields
            
        Returns:
            List of cleaned and formatted segments
        """
        self.logger.info(f"Starting output formatting on {len(structured_segments)} segments")
        
        formatted_segments = []
        formatting_stats = {
            'unicode_fixes': 0,
            'speaker_normalizations': 0,
            'text_cleanups': 0,
            'empty_removed': 0
        }
        
        for i, segment in enumerate(structured_segments):
            try:
                formatted_segment = self._format_single_segment(
                    segment.copy(), preserve_metadata, formatting_stats
                )
                
                # Skip empty segments after formatting
                if not formatted_segment.get('text', '').strip():
                    formatting_stats['empty_removed'] += 1
                    self.logger.debug(f"Removed empty segment {i} after formatting")
                    continue
                
                formatted_segments.append(formatted_segment)
                
            except Exception as e:
                self.logger.warning(f"Error formatting segment {i}: {e}")
                # Keep original segment if formatting fails
                formatted_segments.append(segment)
        
        # Log formatting statistics
        self.logger.info(f"Formatting completed: {len(formatted_segments)} segments output")
        if any(formatting_stats.values()):
            stats_summary = ', '.join(f"{key}: {value}" for key, value in formatting_stats.items() if value > 0)
            self.logger.info(f"Formatting fixes applied: {stats_summary}")
        
        return formatted_segments
    
    def _format_single_segment(self, segment: Dict[str, Any], preserve_metadata: bool, 
                              stats: Dict[str, int]) -> Dict[str, Any]:
        """
        Format a single segment with comprehensive cleanup.
        """
        # Clean and normalize speaker name
        if 'speaker' in segment:
            original_speaker = segment['speaker']
            cleaned_speaker = self._normalize_speaker_name(original_speaker)
            if cleaned_speaker != original_speaker:
                segment['speaker'] = cleaned_speaker
                stats['speaker_normalizations'] += 1
                self.logger.debug(f"Speaker normalized: '{original_speaker}' -> '{cleaned_speaker}'")
        
        # Clean and normalize text content
        if 'text' in segment:
            original_text = segment['text']
            cleaned_text = self._clean_text_content(original_text)
            if cleaned_text != original_text:
                segment['text'] = cleaned_text
                stats['text_cleanups'] += 1
                
                # Count unicode fixes
                if self._contains_problematic_unicode(original_text):
                    stats['unicode_fixes'] += 1
        
        # Clean up metadata fields if preserving
        if preserve_metadata:
            segment = self._clean_metadata_fields(segment)
        else:
            # Remove non-essential fields
            essential_fields = {'speaker', 'text'}
            segment = {key: value for key, value in segment.items() if key in essential_fields}
        
        return segment
    
    def _normalize_speaker_name(self, speaker: str) -> str:
        """
        Normalize and standardize speaker names.
        """
        if not speaker or not isinstance(speaker, str):
            return speaker
        
        # Don't modify reserved speakers
        if speaker.lower() in self.reserved_speakers:
            return speaker
        
        # Basic cleanup
        normalized = speaker.strip()
        
        # Unicode normalization
        normalized = self._normalize_unicode(normalized)
        
        # Handle speaker variations
        normalized_lower = normalized.lower()
        for standard, variations in self.speaker_patterns['variations'].items():
            if normalized_lower in variations:
                return standard
        
        # Title case for character names (but preserve special cases)
        if normalized.lower() not in self.reserved_speakers:
            # Split into words and title case each (except particles)
            words = normalized.split()
            particles = {'de', 'la', 'le', 'du', 'von', 'van', 'da', 'di', 'del', 'of', 'the'}
            
            formatted_words = []
            for i, word in enumerate(words):
                if i > 0 and word.lower() in particles:
                    formatted_words.append(word.lower())
                else:
                    formatted_words.append(word.capitalize())
            
            normalized = ' '.join(formatted_words)
        
        # Remove honorifics and titles if desired
        words = normalized.split()
        if len(words) > 1:
            # Remove common prefixes
            if words[0].lower().rstrip('.') in self.speaker_patterns['prefixes']:
                words = words[1:]
            # Remove common suffixes
            if words and words[-1].lower().rstrip('.') in self.speaker_patterns['suffixes']:
                words = words[:-1]
            
            if words:
                normalized = ' '.join(words)
        
        return normalized
    
    def _clean_text_content(self, text: str) -> str:
        """
        Comprehensive text content cleanup.
        """
        if not text or not isinstance(text, str):
            return text
        
        # Unicode normalization and replacement
        cleaned = self._normalize_unicode(text)
        
        # Apply cleanup patterns
        for pattern, replacement in self.cleanup_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        
        # Final whitespace cleanup
        cleaned = cleaned.strip()
        
        # Ensure proper sentence endings
        if cleaned and not cleaned[-1] in '.!?':
            # Don't add period if it ends with dialogue
            if not (cleaned.endswith('"') or cleaned.endswith("'") or cleaned.endswith('"')):
                # Don't add period if it looks incomplete (ends with comma, conjunction, etc.)
                if not cleaned.endswith((',', 'and', 'or', 'but', 'so', 'yet', 'for', 'nor')):
                    cleaned += '.'
        
        return cleaned
    
    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters and replace problematic ones.
        """
        if not text:
            return text
        
        # Apply specific character replacements
        for unicode_char, replacement in self.unicode_replacements.items():
            text = text.replace(unicode_char, replacement)
        
        # Normalize Unicode composition
        text = unicodedata.normalize('NFKC', text)
        
        # Remove or replace control characters (except common ones)
        cleaned_chars = []
        for char in text:
            # Keep printable characters and common whitespace
            if (char.isprintable() or char in '\n\r\t'):
                cleaned_chars.append(char)
            elif unicodedata.category(char).startswith('C'):
                # Control character - replace with space or remove
                if unicodedata.category(char) in ['Cc', 'Cf']:
                    cleaned_chars.append(' ')  # Replace with space
                # Otherwise skip (remove)
            else:
                cleaned_chars.append(char)
        
        return ''.join(cleaned_chars)
    
    def _contains_problematic_unicode(self, text: str) -> bool:
        """
        Check if text contains Unicode characters that need replacement.
        """
        return any(char in text for char in self.unicode_replacements.keys())
    
    def _clean_metadata_fields(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean up metadata fields while preserving them.
        """
        # Clean error lists
        if 'errors' in segment and isinstance(segment['errors'], list):
            # Remove duplicates and empty entries
            cleaned_errors = list(set(error for error in segment['errors'] if error))
            if cleaned_errors:
                segment['errors'] = cleaned_errors
            else:
                del segment['errors']
        
        # Clean confidence scores
        if 'attribution_confidence' in segment:
            try:
                confidence = float(segment['attribution_confidence'])
                segment['attribution_confidence'] = round(confidence, 3)
            except (ValueError, TypeError):
                del segment['attribution_confidence']
        
        # Clean boolean flags
        boolean_fields = ['refined', 'recovered']
        for field in boolean_fields:
            if field in segment and not isinstance(segment[field], bool):
                try:
                    segment[field] = bool(segment[field])
                except (ValueError, TypeError):
                    del segment[field]
        
        # Clean string fields
        string_fields = ['attribution_method', 'recovery_method', 'refinement_method']
        for field in string_fields:
            if field in segment and isinstance(segment[field], str):
                cleaned_value = segment[field].strip()
                if cleaned_value:
                    segment[field] = cleaned_value
                else:
                    del segment[field]
        
        return segment
    
    def create_clean_output(self, structured_segments: List[Dict[str, Any]], 
                           minimal: bool = False) -> List[Dict[str, Any]]:
        """
        Create a clean output version with optional minimal formatting.
        
        Args:
            structured_segments: Input segments
            minimal: If True, only include speaker and text fields
            
        Returns:
            Cleaned segments ready for production use
        """
        # First, apply comprehensive formatting
        formatted_segments = self.format_output(structured_segments, preserve_metadata=not minimal)
        
        if minimal:
            # Create minimal version with only essential fields
            minimal_segments = []
            for segment in formatted_segments:
                minimal_segment = {
                    'speaker': segment.get('speaker', 'unknown'),
                    'text': segment.get('text', '')
                }
                if minimal_segment['text'].strip():  # Only include non-empty segments
                    minimal_segments.append(minimal_segment)
            return minimal_segments
        
        return formatted_segments