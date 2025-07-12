import re
from tqdm import tqdm
from fuzzywuzzy import fuzz
from config import settings

class ChunkManager:
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.overlap_size = settings.OVERLAP_SIZE

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
