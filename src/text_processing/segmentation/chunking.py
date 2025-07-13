import re
from tqdm import tqdm
from fuzzywuzzy import fuzz
from config import settings

class ChunkManager:
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.overlap_size = settings.OVERLAP_SIZE
        
        # NEW: Rolling context configuration
        self.context_window = 5  # Number of recent speakers to track
        self.context_segments = 3  # Number of recent segments to include

    def create_chunks(self, text, scene_breaks=None):
        """
        Splits text into chunks based on semantic boundaries (sentences/paragraphs)
        with specified overlap, prioritizing scene breaks.
        """
        chunks = []
        current_pos = 0
        text_len = len(text)

        # Sort scene breaks and ensure they are unique
        if scene_breaks:
            scene_breaks = sorted(list(set(scene_breaks)))
        else:
            scene_breaks = []

        while current_pos < text_len:
            # Try to find a scene break within the current chunk range
            ideal_end_pos = min(current_pos + self.chunk_size, text_len)
            actual_end_pos = ideal_end_pos
            
            # Look for a scene break near the end of the ideal chunk
            # Prioritize scene breaks within the last 10% of the chunk, or within overlap_size
            scene_break_found = False
            for sb_pos in scene_breaks:
                if current_pos < sb_pos < ideal_end_pos:
                    # If a scene break is within the chunk, and not too early
                    if sb_pos > current_pos + self.chunk_size * 0.8 or sb_pos > ideal_end_pos - self.overlap_size:
                        actual_end_pos = sb_pos
                        scene_break_found = True
                        break # Take the first suitable scene break
            
            chunk = text[current_pos:actual_end_pos]

            if actual_end_pos < text_len and not scene_break_found: # Not the last chunk and no scene break used
                # Find a good split point (sentence or paragraph end) within the chunk
                # Look for paragraph break first, then sentence end
                split_point = -1
                
                # Search for paragraph break within the last 10% of the chunk, or up to overlap_size
                search_start_in_chunk = max(0, len(chunk) - self.overlap_size)
                paragraph_breaks = [m.start() for m in re.finditer(r'\n\n', chunk[search_start_in_chunk:])]
                if paragraph_breaks:
                    split_point = search_start_in_chunk + paragraph_breaks[-1] + 2 # +2 for \n\n
                if split_point == -1: # No paragraph break found, look for sentence end
                    sentence_ends = [m.start() for m in re.finditer(r'[.!?]\s+', chunk[search_start_in_chunk:])]
                    if sentence_ends:
                        split_point = search_start_in_chunk + sentence_ends[-1] + 1 # +1 for the punctuation
                
                if split_point != -1: # Found a semantic split point
                    chunk = chunk[:split_point].strip()
                    next_start_pos = current_pos + split_point - self.overlap_size
                else:
                    # Fallback: if no semantic split, just cut and overlap
                    next_start_pos = current_pos + self.chunk_size - self.overlap_size
            else:
                next_start_pos = actual_end_pos # If scene break used or last chunk, next starts after this chunk

            chunks.append(chunk.strip())
            current_pos = next_start_pos
            # Ensure current_pos doesn't go backwards or beyond text_len
            current_pos = max(current_pos, actual_end_pos - self.overlap_size) # Ensure minimum overlap
            current_pos = min(current_pos, text_len)

        return chunks

    def merge(self, final_segments, new_structured_segments):
        """
        Merges new structured segments into the final list, handling overlaps
        using fuzzy sequence matching to find the most likely merge point.
        """
        if not final_segments:
            return new_structured_segments

        max_lookahead = min(len(final_segments), len(new_structured_segments), 10) # Look at up to 10 segments for overlap
        if max_lookahead == 0:
            return final_segments + new_structured_segments

        best_match_index = 0
        highest_similarity = 0

        for i in range(1, max_lookahead + 1):
            # Create the text sequences to compare
            final_overlap_text = " ".join([s['text'] for s in final_segments[-i:]])
            new_overlap_text = " ".join([s['text'] for s in new_structured_segments[:i]])

            # Use fuzzy matching to compare the sequences
            similarity = fuzz.ratio(final_overlap_text, new_overlap_text)

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_index = i

        # Only merge if we have a confident match (e.g., > 85% similar)
        if highest_similarity > 85:
            merge_index = best_match_index
            # print(f"DEBUG: Found fuzzy overlap of {merge_index} segments with {highest_similarity}% similarity.") # Optional Debugging
        else:
            merge_index = 0 # No confident overlap, append everything
            # print(f"DEBUG: No confident overlap found (best was {highest_similarity}%). Appending all new segments.") # Optional Debugging

        return final_segments + new_structured_segments[merge_index:]

    # ========================================
    # ROLLING CONTEXT METHODS
    # ========================================
    
    def extract_context_from_processed_segments(self, processed_segments: list) -> dict:
        """
        Extract rolling context information from processed segments for the next chunk.
        
        Args:
            processed_segments: List of processed segments with speaker/text
            
        Returns:
            Dictionary containing context information for the next chunk
        """
        if not processed_segments:
            return {}
        
        context = {
            'recent_speakers': [],
            'recent_segments': [],
            'conversation_flow': [],
            'introduced_characters': set()
        }
        
        # Extract recent speakers (excluding generic ones)
        speakers_found = []
        for segment in reversed(processed_segments[-10:]):  # Last 10 segments
            speaker = segment.get('speaker', '')
            if speaker and speaker not in ['narrator', 'AMBIGUOUS']:
                speakers_found.append(speaker)
                if len(speakers_found) >= self.context_window:
                    break
        
        context['recent_speakers'] = list(reversed(speakers_found))  # Restore chronological order
        
        # Extract recent segments for conversation flow
        recent_segments = []
        for segment in processed_segments[-self.context_segments:]:
            if segment.get('text', '').strip():
                recent_segments.append({
                    'speaker': segment.get('speaker', ''),
                    'text': segment.get('text', '')[:100] + '...' if len(segment.get('text', '')) > 100 else segment.get('text', ''),
                    'is_dialogue': self._is_dialogue_segment(segment)
                })
        
        context['recent_segments'] = recent_segments
        context['conversation_flow'] = recent_segments  # Alias for backward compatibility
        
        # Extract characters introduced in recent segments
        introduced_chars = set()
        for segment in processed_segments[-5:]:  # Last 5 segments
            speaker = segment.get('speaker', '')
            if speaker and speaker not in ['narrator', 'AMBIGUOUS']:
                introduced_chars.add(speaker)
        
        context['introduced_characters'] = introduced_chars
        
        return context
    
    def create_context_hint_for_chunk(self, chunk_index: int, previous_context: dict = None) -> dict:
        """
        Create a context hint for processing a specific chunk.
        
        Args:
            chunk_index: 0-based index of the chunk being processed
            previous_context: Context extracted from previous chunks
            
        Returns:
            Context hint dictionary for the current chunk
        """
        context_hint = {
            'chunk_position': chunk_index,
            'is_continuation': chunk_index > 0
        }
        
        if previous_context:
            # Pass through relevant context information
            context_hint.update({
                'recent_speakers': previous_context.get('recent_speakers', []),
                'conversation_flow': previous_context.get('conversation_flow', []),
                'introduced_characters': list(previous_context.get('introduced_characters', set()))
            })
        
        return context_hint
    
    def merge_contexts(self, existing_context: dict, new_context: dict) -> dict:
        """
        Merge new context information with existing context.
        
        Args:
            existing_context: Existing context from previous chunks
            new_context: New context from current chunk
            
        Returns:
            Merged context dictionary
        """
        if not existing_context:
            return new_context
        
        if not new_context:
            return existing_context
        
        merged = existing_context.copy()
        
        # Merge recent speakers (keep last N unique speakers)
        existing_speakers = existing_context.get('recent_speakers', [])
        new_speakers = new_context.get('recent_speakers', [])
        
        all_speakers = existing_speakers + new_speakers
        unique_speakers = []
        seen = set()
        
        # Keep unique speakers in reverse order, then reverse to get chronological
        for speaker in reversed(all_speakers):
            if speaker not in seen and speaker not in ['narrator', 'AMBIGUOUS']:
                unique_speakers.append(speaker)
                seen.add(speaker)
                if len(unique_speakers) >= self.context_window:
                    break
        
        merged['recent_speakers'] = list(reversed(unique_speakers))
        
        # Merge conversation flow (keep last N segments)
        existing_flow = existing_context.get('conversation_flow', [])
        new_flow = new_context.get('conversation_flow', [])
        
        all_flow = existing_flow + new_flow
        merged['conversation_flow'] = all_flow[-self.context_segments:]
        
        # Merge introduced characters
        existing_chars = set(existing_context.get('introduced_characters', []))
        new_chars = set(new_context.get('introduced_characters', []))
        merged['introduced_characters'] = list(existing_chars | new_chars)
        
        return merged
    
    def _is_dialogue_segment(self, segment: dict) -> bool:
        """
        Check if a segment contains dialogue.
        
        Args:
            segment: Segment dictionary with speaker/text
            
        Returns:
            True if segment appears to be dialogue
        """
        text = segment.get('text', '')
        speaker = segment.get('speaker', '')
        
        # Check for dialogue markers
        dialogue_markers = ['"', '"', '"', "'", '—', '–']
        has_dialogue_markers = any(marker in text for marker in dialogue_markers)
        
        # Check if speaker is a character (not narrator)
        is_character_speaker = speaker not in ['narrator', 'AMBIGUOUS', '']
        
        return has_dialogue_markers or is_character_speaker
    
    # ========================================
    # SLIDING WINDOW METHODS (Ultrathink Architecture - Phase 3)
    # ========================================
    
    def create_sliding_windows(self, text, scene_breaks=None, text_metadata=None):
        """
        Creates sliding windows for processing using the Ultrathink architecture.
        
        This replaces simple chunking with a sliding window approach that provides
        context lines + task lines for each window, maintaining conversation flow.
        
        Args:
            text: Full text to process
            scene_breaks: Optional list of scene break positions
            text_metadata: Metadata including POV analysis for adaptive sizing
            
        Returns:
            List of window dictionaries with context and task lines
        """
        if not settings.SLIDING_WINDOW_ENABLED:
            # Fallback to legacy chunking if sliding windows disabled
            legacy_chunks = self.create_chunks(text, scene_breaks)
            return self._convert_chunks_to_windows(legacy_chunks)
        
        # Get window configuration
        context_size = settings.CONTEXT_WINDOW_SIZE
        task_size = settings.TASK_WINDOW_SIZE
        overlap_ratio = settings.WINDOW_OVERLAP_RATIO
        
        # Adaptive window sizing based on complexity
        if settings.ADAPTIVE_WINDOW_SIZING and text_metadata:
            context_size, task_size = self._adapt_window_sizes(text_metadata, context_size, task_size)
        
        # Split text into lines for window processing
        lines = self._split_text_into_lines(text)
        
        # Create sliding windows
        windows = []
        total_lines = len(lines)
        current_pos = 0
        window_index = 0
        
        while current_pos < total_lines:
            # Calculate window boundaries
            context_start = max(0, current_pos - context_size)
            context_end = current_pos
            task_start = current_pos
            task_end = min(current_pos + task_size, total_lines)
            
            # Extract context lines (background information)
            context_lines = []
            if context_start < context_end:
                context_lines = lines[context_start:context_end]
            
            # Extract task lines (lines to classify)
            task_lines = lines[task_start:task_end]
            
            # Skip if no task lines
            if not task_lines:
                break
            
            # Adjust for scene breaks if provided
            if scene_breaks:
                context_lines, task_lines = self._adjust_window_for_scene_breaks(
                    context_lines, task_lines, scene_breaks, text, context_start, task_end
                )
            
            # Create window
            window = {
                'window_index': window_index,
                'context_lines': context_lines,
                'task_lines': task_lines,
                'context_start_pos': context_start,
                'context_end_pos': context_end,
                'task_start_pos': task_start,
                'task_end_pos': task_end,
                'total_lines': len(context_lines) + len(task_lines),
                'has_context': len(context_lines) > 0
            }
            
            windows.append(window)
            
            # Calculate next position with overlap
            if task_end >= total_lines:
                break  # Last window
            
            overlap_size = int(task_size * overlap_ratio)
            current_pos = task_end - overlap_size
            current_pos = max(current_pos, task_start + 1)  # Ensure progress
            window_index += 1
        
        return windows
    
    def _split_text_into_lines(self, text):
        """
        Split text into logical lines for window processing.
        
        Preserves sentence boundaries and handles various text formats.
        
        Args:
            text: Text to split
            
        Returns:
            List of text lines
        """
        # First split by paragraphs and newlines
        paragraphs = []
        for para in re.split(r'\n\s*\n', text):
            para = para.strip()
            if para:
                paragraphs.append(para)
        
        lines = []
        for paragraph in paragraphs:
            # Split long paragraphs into sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    # Further split very long sentences at logical points
                    if len(sentence) > 500:  # Very long sentence
                        sub_lines = self._split_long_sentence(sentence)
                        lines.extend(sub_lines)
                    else:
                        lines.append(sentence)
        
        return lines
    
    def _split_long_sentence(self, sentence):
        """
        Split a very long sentence at logical break points.
        
        Args:
            sentence: Long sentence to split
            
        Returns:
            List of sentence fragments
        """
        # Try to split at dialogue boundaries first
        if '"' in sentence:
            parts = re.split(r'(".*?")', sentence)
            fragments = []
            for part in parts:
                part = part.strip()
                if part:
                    fragments.append(part)
            return fragments
        
        # Split at comma boundaries if no dialogue
        if ',' in sentence and len(sentence) > 300:
            parts = sentence.split(',')
            fragments = []
            current_fragment = ""
            
            for part in parts:
                if len(current_fragment + part) < 250:
                    current_fragment += part + ","
                else:
                    if current_fragment:
                        fragments.append(current_fragment.rstrip(',').strip())
                    current_fragment = part + ","
            
            if current_fragment:
                fragments.append(current_fragment.rstrip(',').strip())
            
            return fragments if fragments else [sentence]
        
        # Fallback: split at word boundaries
        words = sentence.split()
        fragments = []
        current_fragment = ""
        
        for word in words:
            if len(current_fragment + " " + word) < 250:
                current_fragment += " " + word
            else:
                if current_fragment:
                    fragments.append(current_fragment.strip())
                current_fragment = word
        
        if current_fragment:
            fragments.append(current_fragment.strip())
        
        return fragments if fragments else [sentence]
    
    def _adapt_window_sizes(self, text_metadata, default_context_size, default_task_size):
        """
        Adapt window sizes based on text complexity and POV type.
        
        Args:
            text_metadata: Text metadata including POV analysis
            default_context_size: Default context window size
            default_task_size: Default task window size
            
        Returns:
            Tuple of (adapted_context_size, adapted_task_size)
        """
        pov_analysis = text_metadata.get('pov_analysis', {})
        document_structure = text_metadata.get('document_structure', {})
        
        context_size = default_context_size
        task_size = default_task_size
        
        # Adjust based on POV type
        pov_type = pov_analysis.get('type', 'UNKNOWN')
        if pov_type == 'FIRST_PERSON':
            # First person benefits from more context for narrator continuity
            context_size = int(default_context_size * 1.2)
        elif pov_type == 'MIXED':
            # Mixed POV needs larger context to handle perspective shifts
            context_size = int(default_context_size * 1.5)
            task_size = int(default_task_size * 0.8)  # Smaller task size for accuracy
        
        # Adjust based on dialogue density
        dialogue_density = document_structure.get('dialogue_density', 0.5)
        if dialogue_density > 0.7:  # High dialogue content
            task_size = int(default_task_size * 1.2)  # Larger task windows for dialogue-heavy text
        elif dialogue_density < 0.3:  # Low dialogue content
            context_size = int(default_context_size * 0.8)  # Less context needed for narrative
        
        # Adjust based on complexity score
        complexity_score = document_structure.get('complexity_score', 0.5)
        if complexity_score > 0.8:  # Very complex text
            context_size = int(context_size * 1.3)
            task_size = int(task_size * 0.7)  # Smaller task size for complex content
        
        # Ensure minimum sizes
        context_size = max(context_size, 10)
        task_size = max(task_size, 5)
        
        # Ensure maximum sizes
        context_size = min(context_size, 100)
        task_size = min(task_size, 30)
        
        return context_size, task_size
    
    def _adjust_window_for_scene_breaks(self, context_lines, task_lines, scene_breaks, original_text, context_start, task_end):
        """
        Adjust window boundaries to respect scene breaks.
        
        Args:
            context_lines: Current context lines
            task_lines: Current task lines
            scene_breaks: List of scene break positions in original text
            original_text: Original text for position mapping
            context_start: Start position of context
            task_end: End position of task
            
        Returns:
            Tuple of (adjusted_context_lines, adjusted_task_lines)
        """
        # For now, return unchanged - scene break adjustment is complex
        # This could be enhanced to truncate context at scene boundaries
        # to avoid mixing scenes in the context window
        return context_lines, task_lines
    
    def _convert_chunks_to_windows(self, chunks):
        """
        Convert legacy chunks to window format for backward compatibility.
        
        Args:
            chunks: List of text chunks from legacy chunking
            
        Returns:
            List of window dictionaries
        """
        windows = []
        for i, chunk in enumerate(chunks):
            # Split chunk into lines
            lines = self._split_text_into_lines(chunk)
            
            # Create window with no context (legacy behavior)
            window = {
                'window_index': i,
                'context_lines': [],
                'task_lines': lines,
                'context_start_pos': 0,
                'context_end_pos': 0,
                'task_start_pos': 0,
                'task_end_pos': len(lines),
                'total_lines': len(lines),
                'has_context': False
            }
            windows.append(window)
        
        return windows
