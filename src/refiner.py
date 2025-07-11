import json
import re
from fuzzywuzzy import fuzz, process

# --- Constants ---
MAX_REFINEMENT_ITERATIONS = 2
REFINEMENT_QUALITY_THRESHOLD = 98.0 # Start refinement if score is below this
CONTEXT_WINDOW_SIZE = 250 # Chars to include before/after a segment for context

class OutputRefiner:
    def __init__(self, llm_orchestrator, output_validator):
        self.orchestrator = llm_orchestrator
        self.validator = output_validator

    def refine(self, processed_data, original_text, raw_chunks, text_metadata):
        """
        Orchestrates the iterative refinement process using a batch-based approach.
        """
        for i in range(MAX_REFINEMENT_ITERATIONS):
            print(f"\n--- Refinement Iteration {i + 1}/{MAX_REFINEMENT_ITERATIONS} ---")
            
            # --- PASS 1: Initial Validation and Error Identification ---
            processed_data, quality_report = self.validator.validate(processed_data, original_text, text_metadata)

            # --- PASS 2: Targeted Refinement for AMBIGUOUS Speakers ---
            ambiguous_segments_found = False
            for idx, (segment, chunk_idx) in enumerate(processed_data):
                if segment.get('speaker') == 'AMBIGUOUS':
                    ambiguous_segments_found = True
                    print(f"  - Refining ambiguous segment at index {idx}...")

                    # Get context for the smart prompt
                    # Look back up to 4 segments for context
                    previous_segments_for_context = [s for s, _ in processed_data[max(0, idx - 4):idx]]
                    known_characters = list(text_metadata.get('potential_character_names', set()))

                    # Build the targeted smart prompt
                    refinement_prompt = self._build_refinement_prompt(previous_segments_for_context, segment['text'], known_characters)

                    # Get the corrected speaker from the LLM
                    response_text = self.orchestrator.get_response(refinement_prompt)
                    corrected_speaker = response_text.strip()

                    if corrected_speaker and corrected_speaker != "AMBIGUOUS":
                        segment['speaker'] = corrected_speaker
                    else:
                        print(f"    Warning: Refinement for segment {idx} failed or returned ambiguous. Marking as unfixable.")
                        self._mark_segment_as_unfixable(segment, "ambiguous_speaker")
            
            if not ambiguous_segments_found:
                print("No ambiguous segments found for refinement. Exiting refinement loop.")
                break

            # Re-validate after refinement pass
            processed_data, quality_report = self.validator.validate(processed_data, original_text, text_metadata)
            print(f"New Quality Score after refinement: {quality_report['quality_score']:.2f}% ({quality_report['error_count']} errors)")

            if quality_report['quality_score'] >= REFINEMENT_QUALITY_THRESHOLD:
                print("Quality threshold met. Exiting refinement loop.")
                break
        
        # Final cleanup: Convert any remaining error flags to confidence flags
        for segment_tuple in processed_data:
            segment = segment_tuple[0]
            if 'errors' in segment:
                self._mark_segment_as_unfixable(segment, segment['errors'][0])

        return processed_data, quality_report


    def _build_refinement_prompt(self, previous_segments, ambiguous_text, known_characters):
        """
        Builds a prompt for the LLM to refine ambiguous speaker assignments.
        """
        prompt = "The following text segment has an ambiguous speaker. Based on the preceding segments and the list of known characters, identify the correct speaker. If the speaker is a narrator, respond with 'narrator'. If the speaker is not in the known characters list and is not a narrator, respond with 'UNKNOWN'.\n\n"
        
        if previous_segments:
            prompt += "Previous segments for context:\n"
            for seg in previous_segments:
                prompt += f"- Speaker: {seg.get('speaker', 'narrator')}, Text: {seg.get('text', '')}\n"
            prompt += "\n"

        prompt += f"Ambiguous segment: {ambiguous_text}\n"
        prompt += f"Known characters: {', '.join(known_characters)}\n\n"
        prompt += "Respond only with the speaker's name (e.g., 'John', 'narrator', 'UNKNOWN')."
        return prompt

    def _mark_segment_as_unfixable(self, segment, error_type):
        """
        Marks a segment as unfixable with a specific error type.
        """
        if 'errors' not in segment:
            segment['errors'] = []
        segment['errors'].append(error_type)
        segment['speaker'] = "UNFIXABLE"

