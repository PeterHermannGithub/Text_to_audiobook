import json
import re
from fuzzywuzzy import fuzz, process
from config import settings

class OutputValidator:
    def __init__(self):
        pass

    def validate(self, processed_data, original_text, text_metadata):
        print(f"DEBUG: Validator received processed_data (first 5 segments):\n{json.dumps(processed_data[:5], indent=2)}")
        print(f"DEBUG: Validator received original_text (first 500 chars):\n{original_text[:500]}...")
        """
        Analyzes the structured text (with chunk indices) for common LLM errors and calculates a quality score.
        """
        errors = []
        known_character_names = text_metadata.get('potential_character_names', set())
        
        # Extract only the segment dictionaries for content preservation check
        structured_text_only = [item[0] for item in processed_data]
        structured_content = "".join([item['text'] for item in structured_text_only])
        # Normalize both texts for a more accurate comparison
        normalized_original = re.sub(r'\s+', ' ', original_text).strip()
        normalized_structured = re.sub(r'\s+', ' ', structured_content).strip()
        
        content_similarity_score = fuzz.ratio(normalized_original, normalized_structured)
        
        if content_similarity_score < settings.SIMILARITY_THRESHOLD:
            # This is a major error, so we won't assign it to a specific segment index
            errors.append({"index": -1, "type": "content_preservation", "message": f"Content Preservation Alert: Text similarity is {content_similarity_score}%, which is below the {settings.SIMILARITY_THRESHOLD}% threshold. The LLM may have omitted or hallucinated content."})

        for i, (segment, chunk_idx) in enumerate(processed_data):
            text = segment['text']
            speaker = segment['speaker']

            # Defensive check for NoneType
            if speaker is None:
                errors.append({"index": i, "type": "missing_speaker", "message": f"Missing Speaker: Segment {i} has a null speaker value."})
                continue
            if text is None:
                errors.append({"index": i, "type": "missing_text", "message": f"Missing Text: Segment {i} for speaker '{speaker}' has a null text value."})
                continue

            # Check for empty segments
            if not text.strip():
                errors.append({"index": i, "type": "empty_segment", "message": f"Empty Segment: Segment {i} for speaker '{speaker}' is empty or contains only whitespace."})
                continue # Skip other checks for this segment


        # 3. Calculate Final Quality Score
        final_score = content_similarity_score
        error_penalty = 5 # Deduct 5 points for each error type
        final_score -= len(errors) * error_penalty
        final_score = max(0, final_score) # Ensure score doesn't go below 0

        # Add error flags directly to the segments for the refinement loop
        for error in errors:
            segment_index = error.get("index", -1)
            if 0 <= segment_index < len(processed_data):
                segment_dict = processed_data[segment_index][0] # Get the segment dictionary from the tuple
                if 'errors' not in segment_dict:
                    segment_dict['errors'] = []
                # Avoid adding duplicate error types to the same segment
                if error["type"] not in segment_dict['errors']:
                    segment_dict['errors'].append(error["type"])

        quality_report = {
            "quality_score": final_score,
            "content_similarity": content_similarity_score,
            "error_count": len(errors),
            "errors": [e["message"] for e in errors] # Keep a clean list for printing
        }

        return processed_data, quality_report
