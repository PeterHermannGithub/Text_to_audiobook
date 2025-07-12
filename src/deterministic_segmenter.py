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
        
        # Convert to numbered line objects
        numbered_lines = []
        for i, segment in enumerate(segments):
            if segment.strip():  # Only include non-empty segments
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