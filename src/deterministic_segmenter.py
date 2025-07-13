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
        
        # Enhanced content quality thresholds
        self.min_segment_length = 10  # Minimum meaningful segment length
        self.max_segment_length = 350  # Reduced from 400 for better handling
        self.dialogue_split_threshold = 150  # Reduced from 200 for more aggressive splitting
        self.mixed_content_threshold = 0.15  # Minimum ratio for mixed content detection
        self.max_dialogue_attribution_length = 250  # Max length for dialogue + attribution
    
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
                
            # Enhanced mixed-content and large segment handling
            if (len(segment) > self.max_segment_length or 
                self._contains_mixed_content(segment) or
                self._requires_advanced_splitting(segment)):
                sub_segments = self._advanced_segment_splitting(segment)
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
            
        # Enhanced dialogue marker detection including brackets and system messages
        dialogue_markers = ['"', '"', '"', '—', "'", '[', ']']
        
        # If paragraph has no dialogue markers, treat as single segment
        if not any(marker in paragraph for marker in dialogue_markers):
            return [paragraph.strip()]
        
        segments = []
        current_segment = ""
        
        # Split by sentences while preserving dialogue boundaries
        sentences = self._split_sentences_preserve_dialogue(paragraph)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if this sentence contains dialogue (including brackets for system messages)
            has_dialogue = any(marker in sentence for marker in ['"', '"', '"', '—', "'", '[', ']'])
            
            # Additional check for bracket-enclosed system messages/thoughts
            has_bracket_dialogue = bool(re.search(r'\[([^\]]+)\]', sentence))
            
            if has_dialogue or has_bracket_dialogue:
                # Save any accumulated narrative content
                if current_segment.strip():
                    segments.append(current_segment.strip())
                    current_segment = ""
                
                # Check if this sentence has embedded dialogue that should be split further
                embedded_segments = self._split_embedded_dialogue(sentence)
                
                if len(embedded_segments) > 1:
                    # Multiple segments found - add each separately
                    segments.extend(embedded_segments)
                else:
                    # Single dialogue segment
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
    
    def _split_embedded_dialogue(self, sentence: str) -> List[str]:
        """
        Split sentences that contain embedded dialogue within narrative.
        
        Handles cases like:
        - "Narrative text. [System message] More narrative."
        - "Narrative text. \"Dialogue\" More narrative."
        - "Before text [skill activated] after text."
        
        Args:
            sentence: Sentence that may contain embedded dialogue
            
        Returns:
            List of segments split at dialogue boundaries
        """
        segments = []
        current_pos = 0
        
        # Pattern for bracket-enclosed dialogue (system messages, skills, thoughts)
        bracket_pattern = r'\[([^\]]+)\]'
        bracket_matches = list(re.finditer(bracket_pattern, sentence))
        
        # Pattern for quoted dialogue
        quote_patterns = [
            r'"([^"]*)"',  # Standard quotes
            r'"([^"]*)"',  # Smart quotes left
            r'"([^"]*)"',  # Smart quotes right
        ]
        
        all_dialogue_matches = []
        
        # Collect all bracket matches
        for match in bracket_matches:
            all_dialogue_matches.append({
                'start': match.start(),
                'end': match.end(),
                'text': match.group(0),
                'type': 'bracket'
            })
        
        # Collect all quote matches
        for pattern in quote_patterns:
            for match in re.finditer(pattern, sentence):
                all_dialogue_matches.append({
                    'start': match.start(),
                    'end': match.end(),
                    'text': match.group(0),
                    'type': 'quote'
                })
        
        # Sort matches by position
        all_dialogue_matches.sort(key=lambda x: x['start'])
        
        # If no dialogue found, return original sentence
        if not all_dialogue_matches:
            return [sentence.strip()]
        
        # Split around dialogue matches
        for match in all_dialogue_matches:
            # Add narrative before dialogue
            before_text = sentence[current_pos:match['start']].strip()
            if before_text:
                segments.append(before_text)
            
            # Add the dialogue as its own segment
            dialogue_text = match['text'].strip()
            if dialogue_text:
                segments.append(dialogue_text)
            
            current_pos = match['end']
        
        # Add any remaining narrative after the last dialogue
        after_text = sentence[current_pos:].strip()
        if after_text:
            segments.append(after_text)
        
        # Filter out empty segments and return
        return [seg for seg in segments if seg.strip()]
    
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
    
    def _requires_advanced_splitting(self, segment: str) -> bool:
        """
        Check if a segment requires advanced splitting beyond basic mixed-content detection.
        
        Identifies complex patterns that need specialized handling:
        - Multiple speaker transitions within a segment
        - Complex dialogue with embedded narrative
        - Action sequences with interspersed dialogue
        """
        if len(segment) < 100:  # Too short to need advanced splitting
            return False
        
        # Count potential speaker transition points
        speaker_transitions = 0
        dialogue_blocks = 0
        narrative_blocks = 0
        
        # Look for dialogue attribution patterns that suggest speaker changes
        attribution_patterns = [
            r'"[^"]*",?\s+[A-Z][a-z]+\s+(?:said|asked|replied|whispered|shouted)',
            r'[A-Z][a-z]+\s+(?:said|asked|replied|whispered|shouted)[^.]*\.\s*"',
            r'"[^"]*"\s*[-–—]\s*[A-Z][a-z]+',
        ]
        
        for pattern in attribution_patterns:
            matches = re.findall(pattern, segment)
            speaker_transitions += len(matches)
        
        # Count dialogue and narrative blocks
        sentences = self._split_sentences_preserve_dialogue(segment)
        for sentence in sentences:
            if any(marker in sentence for marker in ['"', '"', '"', "'", '—']):
                dialogue_blocks += 1
            else:
                narrative_blocks += 1
        
        # Advanced splitting criteria
        needs_advanced_splitting = (
            # Multiple speaker transitions (likely conversation)
            speaker_transitions >= 2 or
            # High ratio of dialogue to narrative blocks (complex conversation)
            (dialogue_blocks > 0 and narrative_blocks > 0 and 
             min(dialogue_blocks, narrative_blocks) >= 2) or
            # Very long segment with any dialogue (likely to have attribution issues)
            (len(segment) > 300 and dialogue_blocks > 0)
        )
        
        if needs_advanced_splitting:
            self.logger.debug(f"Advanced splitting needed: {speaker_transitions} transitions, "
                             f"{dialogue_blocks} dialogue blocks, {narrative_blocks} narrative blocks")
        
        return needs_advanced_splitting
    
    def _advanced_segment_splitting(self, segment: str) -> List[str]:
        """
        Advanced segment splitting using multiple strategies for complex mixed content.
        
        Employs a progressive approach:
        1. Dialogue boundary splitting
        2. Speaker transition splitting  
        3. Narrative/action splitting
        4. Fallback to sentence-based splitting
        """
        self.logger.debug(f"Applying advanced splitting to segment of {len(segment)} characters")
        
        # Strategy 1: Try dialogue boundary splitting first
        dialogue_splits = self._split_by_dialogue_boundaries(segment)
        if self._is_good_split(dialogue_splits, segment):
            self.logger.debug(f"Dialogue boundary split successful: {len(dialogue_splits)} segments")
            return dialogue_splits
        
        # Strategy 2: Try speaker transition splitting
        speaker_splits = self._split_by_speaker_transitions(segment)
        if self._is_good_split(speaker_splits, segment):
            self.logger.debug(f"Speaker transition split successful: {len(speaker_splits)} segments")
            return speaker_splits
        
        # Strategy 3: Try narrative/action boundary splitting
        action_splits = self._split_by_action_boundaries(segment)
        if self._is_good_split(action_splits, segment):
            self.logger.debug(f"Action boundary split successful: {len(action_splits)} segments")
            return action_splits
        
        # Strategy 4: Fallback to enhanced sentence-based splitting
        sentence_splits = self._enhanced_sentence_splitting(segment)
        self.logger.debug(f"Fallback to sentence splitting: {len(sentence_splits)} segments")
        return sentence_splits
    
    def _split_by_dialogue_boundaries(self, segment: str) -> List[str]:
        """
        Split segment at clear dialogue boundaries.
        
        Identifies transitions between dialogue and narrative, and between different speakers.
        """
        splits = []
        current_split = ""
        
        sentences = self._split_sentences_preserve_dialogue(segment)
        current_type = None  # 'dialogue' or 'narrative'
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Determine sentence type
            is_dialogue = any(marker in sentence for marker in ['"', '"', '"', "'", '—'])
            sentence_type = 'dialogue' if is_dialogue else 'narrative'
            
            # Check for type transition
            if current_type is None:
                current_type = sentence_type
                current_split = sentence
            elif current_type == sentence_type:
                # Same type, accumulate
                current_split += " " + sentence
            else:
                # Type transition, start new split
                if current_split.strip():
                    splits.append(current_split.strip())
                current_split = sentence
                current_type = sentence_type
        
        # Add remaining content
        if current_split.strip():
            splits.append(current_split.strip())
        
        return splits
    
    def _split_by_speaker_transitions(self, segment: str) -> List[str]:
        """
        Split segment at points where speaker changes are detected.
        
        Uses dialogue attribution patterns to identify speaker boundaries.
        """
        splits = []
        current_split = ""
        
        # Enhanced attribution patterns
        attribution_patterns = [
            # "dialogue," Speaker said
            (r'(".*?"),\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:said|asked|replied|whispered|shouted|muttered|cried|exclaimed)', 'post_dialogue'),
            # Speaker said, "dialogue"  
            (r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:said|asked|replied|whispered|shouted|muttered|cried|exclaimed)[^.]*\.\s*(".*?")', 'pre_dialogue'),
            # "dialogue" - Speaker
            (r'(".*?")\s*[-–—]\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)', 'dash_attribution'),
            # Speaker: "dialogue"
            (r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s*:\s*(".*?")', 'script_format'),
        ]
        
        # Find all attribution points
        attribution_points = []
        for pattern, pattern_type in attribution_patterns:
            for match in re.finditer(pattern, segment):
                attribution_points.append((match.start(), match.end(), pattern_type))
        
        # Sort by position
        attribution_points.sort(key=lambda x: x[0])
        
        if not attribution_points:
            return [segment]  # No speaker transitions found
        
        # Split at attribution points
        last_end = 0
        for start, end, pattern_type in attribution_points:
            # Add content before attribution
            if start > last_end:
                before_content = segment[last_end:start].strip()
                if before_content:
                    if current_split:
                        current_split += " " + before_content
                    else:
                        current_split = before_content
            
            # Add attribution content and start new split
            attribution_content = segment[start:end].strip()
            if current_split:
                current_split += " " + attribution_content
                splits.append(current_split.strip())
                current_split = ""
            else:
                current_split = attribution_content
            
            last_end = end
        
        # Add remaining content
        if last_end < len(segment):
            remaining_content = segment[last_end:].strip()
            if remaining_content:
                if current_split:
                    current_split += " " + remaining_content
                else:
                    current_split = remaining_content
        
        if current_split.strip():
            splits.append(current_split.strip())
        
        return splits
    
    def _split_by_action_boundaries(self, segment: str) -> List[str]:
        """
        Split segment at action/description boundaries.
        
        Identifies transitions between action descriptions and dialogue/thoughts.
        """
        splits = []
        current_split = ""
        
        # Look for action boundary patterns
        action_patterns = [
            r'\.\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:turned|looked|walked|stepped|moved|gestured|nodded|smiled|frowned|sighed)',
            r'\.\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:thought|wondered|realized|remembered|considered)',
            r'\.\s+(The\s+\w+|Then\s+\w+|Suddenly\s+\w+|Meanwhile\s+\w+)',
        ]
        
        # Find action boundary points
        action_points = []
        for pattern in action_patterns:
            for match in re.finditer(pattern, segment):
                action_points.append(match.start() + 1)  # Split after the period
        
        if not action_points:
            return [segment]  # No action boundaries found
        
        # Sort split points
        action_points = sorted(set(action_points))
        
        # Split at action points
        last_pos = 0
        for split_pos in action_points:
            chunk = segment[last_pos:split_pos].strip()
            if chunk:
                splits.append(chunk)
            last_pos = split_pos
        
        # Add remaining content
        if last_pos < len(segment):
            remaining = segment[last_pos:].strip()
            if remaining:
                splits.append(remaining)
        
        return splits
    
    def _enhanced_sentence_splitting(self, segment: str) -> List[str]:
        """
        Enhanced sentence-based splitting with dialogue awareness.
        
        Fallback method that ensures no segment is too large while preserving meaning.
        """
        sentences = self._split_sentences_preserve_dialogue(segment)
        
        splits = []
        current_split = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would make the split too large
            if (current_split and 
                len(current_split) + len(sentence) > self.max_segment_length):
                splits.append(current_split.strip())
                current_split = sentence
            else:
                if current_split:
                    current_split += " " + sentence
                else:
                    current_split = sentence
        
        # Add remaining content
        if current_split.strip():
            splits.append(current_split.strip())
        
        return splits
    
    def _is_good_split(self, splits: List[str], original_segment: str) -> bool:
        """
        Evaluate whether a split result is good enough to use.
        
        Criteria:
        - All splits are reasonable length
        - Content is preserved
        - Splits are meaningful (not too fragmented)
        """
        if not splits or len(splits) == 1:
            return False
        
        # Check split lengths
        for split in splits:
            if len(split.strip()) < self.min_segment_length:
                return False
            if len(split) > self.max_segment_length * 1.2:  # Allow some flexibility
                return False
        
        # Check content preservation
        combined_length = sum(len(split) for split in splits)
        if abs(combined_length - len(original_segment)) > len(original_segment) * 0.1:
            return False  # Too much content lost or added
        
        # Check fragmentation
        if len(splits) > 6:  # Too many fragments
            return False
        
        # Check that each split has meaningful content
        meaningful_splits = 0
        for split in splits:
            # Must contain either dialogue or substantial narrative
            if (any(marker in split for marker in ['"', '"', '"', "'", '—']) or
                len(split.strip()) > 50):
                meaningful_splits += 1
        
        if meaningful_splits < len(splits) * 0.8:  # At least 80% should be meaningful
            return False
        
        return True