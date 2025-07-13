import re
import logging
from typing import List, Dict, Any

class DeterministicSegmenter:
    """
    Handles rule-based text segmentation into numbered line objects.
    Replaces LLM text modification with deterministic paragraph/dialogue boundary detection.
    
    This ensures that text content is never modified, only split into logical segments.
    The output is a list of numbered line objects that preserve exact original content.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Metadata patterns to filter out at segmentation level
        self.metadata_patterns = [
            r'^chapter\s+\d+', r'^chapter\s+[ivx]+', r'^ch\.\s*\d+',
            r'^epilogue', r'^prologue', r'^part\s+\d+', r'^book\s+\d+',
            r'^volume\s+\d+', r'^section\s+\d+', r'^author:', r'^writer:',
            r'^\d+\.\s*$', r'^[ivx]+\.\s*$'
        ]
        
        # Content quality thresholds
        self.min_segment_length = 10  # Minimum meaningful segment length
        self.max_segment_length = 400  # Split large segments more aggressively
        self.dialogue_split_threshold = 200  # Split mixed dialogue/narrative aggressively
    
    def segment_text(self, text_content: str, text_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Segments text into numbered line objects using deterministic rules.
        
        Args:
            text_content: Raw text to segment
            text_metadata: Metadata containing dialogue markers, script format info, etc.
            
        Returns:
            List of dictionaries with 'line_id' and 'text' keys
        """
        if not text_content or not text_content.strip():
            return []
            
        # Extract metadata for context-aware segmentation
        is_script_like = text_metadata.get('is_script_like', False) if text_metadata else False
        dialogue_markers = text_metadata.get('dialogue_markers', set()) if text_metadata else set()
        
        self.logger.debug(f"Segmenting text with script_like={is_script_like}, dialogue_markers={dialogue_markers}")
        
        # Choose segmentation strategy based on text format
        if is_script_like:
            segments = self._segment_script_format(text_content)
        elif 'script_format' in dialogue_markers:
            segments = self._segment_mixed_script_format(text_content)
        else:
            segments = self._segment_narrative_format(text_content)
        
        # Filter and process segments
        filtered_segments = []
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
                
            # Filter out metadata segments
            if self._is_metadata_segment(segment):
                self.logger.debug(f"Filtered out metadata segment: {repr(segment[:100])}")
                continue
                
            # Split overly large segments or mixed content
            if (len(segment) > self.max_segment_length or 
                self._contains_mixed_content(segment)):
                sub_segments = self._split_large_segment(segment)
                filtered_segments.extend(sub_segments)
            else:
                filtered_segments.append(segment)
        
        # Convert to numbered line objects
        numbered_lines = []
        for i, segment in enumerate(filtered_segments):
            if len(segment.strip()) >= self.min_segment_length:
                numbered_lines.append({
                    'line_id': i + 1,
                    'text': segment.strip()
                })
        
        self.logger.info(f"Segmented text into {len(numbered_lines)} lines")
        return numbered_lines
    
    def _segment_script_format(self, text: str) -> List[str]:
        """
        Segments text that's primarily in script format (CHARACTER: dialogue).
        
        Each line starting with a character name pattern becomes its own segment.
        Narrative lines between script lines are kept as separate segments.
        """
        segments = []
        current_segment = ""
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line starts with a script format pattern
            script_match = re.match(r'^(?:–|\s|-)?\s*([A-Z][a-zA-Z0-9_\s]*):\s*(.*)', line)
            
            if script_match:
                # Save any accumulated narrative content
                if current_segment.strip():
                    segments.append(current_segment.strip())
                    current_segment = ""
                
                # Script line becomes its own segment
                segments.append(line)
            else:
                # Accumulate narrative content
                if current_segment:
                    current_segment += " " + line
                else:
                    current_segment = line
        
        # Add any remaining content
        if current_segment.strip():
            segments.append(current_segment.strip())
            
        return segments
    
    def _segment_mixed_script_format(self, text: str) -> List[str]:
        """
        Segments text that has mixed script and narrative format.
        
        More aggressive about splitting when script patterns are detected,
        but handles narrative sections with paragraph-based segmentation.
        """
        segments = []
        current_segment = ""
        
        # Split by major paragraph breaks first
        paragraphs = re.split(r'\n\s*\n', text)
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Check if this paragraph contains script patterns
            lines = paragraph.split('\n')
            has_script_lines = any(re.match(r'^(?:–|\s|-)?\s*([A-Z][a-zA-Z0-9_\s]*):\s*', line.strip()) for line in lines)
            
            if has_script_lines:
                # Process line by line for script content
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    script_match = re.match(r'^(?:–|\s|-)?\s*([A-Z][a-zA-Z0-9_\s]*):\s*(.*)', line)
                    if script_match:
                        # Save any accumulated content
                        if current_segment.strip():
                            segments.append(current_segment.strip())
                            current_segment = ""
                        
                        # Script line becomes its own segment
                        segments.append(line)
                    else:
                        # Non-script line in script paragraph
                        if current_segment:
                            current_segment += " " + line
                        else:
                            current_segment = line
            else:
                # Pure narrative paragraph - segment by sentences or natural breaks
                narrative_segments = self._segment_narrative_paragraph(paragraph)
                
                # Save any accumulated content first
                if current_segment.strip():
                    segments.append(current_segment.strip())
                    current_segment = ""
                
                segments.extend(narrative_segments)
        
        # Add any remaining content
        if current_segment.strip():
            segments.append(current_segment.strip())
            
        return segments
    
    def _segment_narrative_format(self, text: str) -> List[str]:
        """
        Segments narrative text based on paragraph breaks and dialogue boundaries.
        
        Prioritizes natural paragraph breaks, dialogue boundaries, and scene transitions.
        """
        segments = []
        
        # Split by major paragraph breaks first
        paragraphs = re.split(r'\n\s*\n', text)
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Further segment each paragraph if it contains dialogue
            paragraph_segments = self._segment_narrative_paragraph(paragraph)
            segments.extend(paragraph_segments)
            
        return segments
    
    def _segment_narrative_paragraph(self, paragraph: str) -> List[str]:
        """
        Segments a narrative paragraph based on dialogue boundaries and sentence structure.
        
        Args:
            paragraph: A single paragraph of narrative text
            
        Returns:
            List of text segments preserving dialogue boundaries
        """
        if not paragraph.strip():
            return []
            
        # If paragraph has no dialogue markers, treat as single segment
        if not any(marker in paragraph for marker in ['"', '"', '"', '—', "'"]):
            return [paragraph.strip()]
        
        segments = []
        current_segment = ""
        
        # Split by sentences while preserving dialogue boundaries
        sentences = self._split_sentences_preserve_dialogue(paragraph)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if this sentence contains dialogue
            has_dialogue = any(marker in sentence for marker in ['"', '"', '"', '—', "'"])
            
            if has_dialogue:
                # Save any accumulated narrative content
                if current_segment.strip():
                    segments.append(current_segment.strip())
                    current_segment = ""
                
                # Dialogue sentence becomes its own segment (may include attribution)
                segments.append(sentence)
            else:
                # Accumulate narrative content
                if current_segment:
                    current_segment += " " + sentence
                else:
                    current_segment = sentence
        
        # Add any remaining content
        if current_segment.strip():
            segments.append(current_segment.strip())
            
        return segments
    
    def _split_sentences_preserve_dialogue(self, text: str) -> List[str]:
        """
        Splits text into sentences while preserving dialogue integrity.
        
        Ensures that dialogue quotes and their attribution stay together.
        """
        # This is a simplified approach - more sophisticated dialogue detection could be added
        sentences = []
        current_sentence = ""
        in_dialogue = False
        dialogue_char = None
        
        i = 0
        while i < len(text):
            char = text[i]
            current_sentence += char
            
            # Track dialogue state
            if char in ['"', '"', '"'] and not in_dialogue:
                in_dialogue = True
                dialogue_char = char
            elif char == dialogue_char and in_dialogue:
                in_dialogue = False
                dialogue_char = None
            
            # Check for sentence boundaries (only when not in dialogue)
            if not in_dialogue and char in ['.', '!', '?']:
                # Look ahead to see if this is really a sentence end
                if i + 1 < len(text) and text[i + 1].isspace():
                    # Check if next non-space character is uppercase (likely new sentence)
                    j = i + 1
                    while j < len(text) and text[j].isspace():
                        j += 1
                    
                    if j < len(text) and (text[j].isupper() or text[j] in ['"', '"']):
                        sentences.append(current_sentence.strip())
                        current_sentence = ""
            
            i += 1
        
        # Add any remaining content
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
            
        return sentences
    
    def _is_metadata_segment(self, segment: str) -> bool:
        """
        Check if a segment is likely metadata that should be filtered out.
        
        Returns True if the segment should be excluded from processing.
        """
        if not segment or len(segment.strip()) < 3:
            return True
            
        segment_lower = segment.strip().lower()
        
        # Check against metadata patterns
        for pattern in self.metadata_patterns:
            if re.match(pattern, segment_lower):
                return True
        
        # Check for chapter-like patterns
        if re.match(r'^chapter\s+', segment_lower):
            return True
            
        # Check if it's just a chapter title or header
        if (len(segment) < 100 and 
            ('chapter' in segment_lower or 
             'epilogue' in segment_lower or 
             'prologue' in segment_lower or
             re.match(r'^\d+\.\s', segment) or
             segment.count(':') == 1 and segment.endswith(':'))):
            return True
            
        # Check for author/metadata lines
        if (segment_lower.startswith('author:') or 
            segment_lower.startswith('writer:') or
            segment_lower.startswith('–author:') or
            segment_lower == 'author' or
            segment_lower == 'writer'):
            return True
            
        return False
    
    def _split_large_segment(self, segment: str) -> List[str]:
        """
        Split overly large segments into smaller, more manageable pieces.
        
        Preserves dialogue boundaries and natural breaks.
        """
        if len(segment) <= self.max_segment_length:
            return [segment]
            
        self.logger.debug(f"Splitting large segment of {len(segment)} characters")
        
        # Try to split by sentence boundaries first
        sentences = self._split_sentences_preserve_dialogue(segment)
        
        sub_segments = []
        current_sub_segment = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed max length, start new segment
            if (current_sub_segment and 
                len(current_sub_segment) + len(sentence) > self.max_segment_length):
                sub_segments.append(current_sub_segment.strip())
                current_sub_segment = sentence
            else:
                if current_sub_segment:
                    current_sub_segment += " " + sentence
                else:
                    current_sub_segment = sentence
        
        # Add any remaining content
        if current_sub_segment.strip():
            sub_segments.append(current_sub_segment.strip())
        
        # If we still have segments that are too large, split by paragraph
        final_segments = []
        for sub_segment in sub_segments:
            if len(sub_segment) > self.max_segment_length:
                # Split by paragraphs
                paragraphs = re.split(r'\n\s*\n', sub_segment)
                for para in paragraphs:
                    if para.strip():
                        final_segments.append(para.strip())
            else:
                final_segments.append(sub_segment)
        
        self.logger.debug(f"Split into {len(final_segments)} smaller segments")
        return final_segments
    
    def _contains_mixed_content(self, segment: str) -> bool:
        """
        Check if a segment contains mixed dialogue and narrative content.
        
        Returns True if the segment should be split due to mixed content types.
        """
        if len(segment) < self.dialogue_split_threshold:
            return False
            
        # Count dialogue vs narrative content
        dialogue_chars = 0
        narrative_chars = 0
        
        # Split into sentences for analysis
        sentences = self._split_sentences_preserve_dialogue(segment)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if sentence contains dialogue markers
            has_dialogue = any(marker in sentence for marker in ['"', '"', '"', "'", '—', '–'])
            
            if has_dialogue:
                dialogue_chars += len(sentence)
            else:
                narrative_chars += len(sentence)
        
        total_chars = dialogue_chars + narrative_chars
        if total_chars == 0:
            return False
            
        # Calculate content ratios
        dialogue_ratio = dialogue_chars / total_chars
        narrative_ratio = narrative_chars / total_chars
        
        # Mixed content criteria:
        # 1. Both dialogue and narrative present (each >20% of content)
        # 2. Segment is longer than dialogue split threshold
        # 3. Content is reasonably balanced (neither dominates completely)
        
        has_mixed_content = (
            dialogue_ratio > 0.2 and 
            narrative_ratio > 0.2 and 
            len(segment) > self.dialogue_split_threshold and
            min(dialogue_ratio, narrative_ratio) > 0.15  # Neither type dominates completely
        )
        
        if has_mixed_content:
            self.logger.debug(f"Mixed content detected: {dialogue_ratio:.2f} dialogue, {narrative_ratio:.2f} narrative, {len(segment)} chars")
        
        return has_mixed_content