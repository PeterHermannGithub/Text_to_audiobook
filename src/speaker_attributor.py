import re
import spacy
from fuzzywuzzy import fuzz, process
from config import settings

class SpeakerAttributor:
    def __init__(self, nlp_model):
        self.nlp = nlp_model

    def attribute_speakers(self, paragraph_list, text_metadata):
        # Extract metadata parameters
        known_character_names = text_metadata.get('potential_character_names', set())
        dialogue_markers = text_metadata.get('dialogue_markers', set())
        is_script_like = text_metadata.get('is_script_like', False)
        
        structured_output = []
        for paragraph in paragraph_list:
            speaker = "narrator" # Default speaker
            text = paragraph.strip()

            # Rule 2 (Chat/Script Format): If the paragraph matches the –Name: Text format
            script_match = re.match(r'^(?:–|\s|-)?\s*([A-Z][a-zA-Z0-9_\s]*):\s*(.*)', text)
            if script_match and is_script_like:
                potential_speaker = script_match.group(1).strip()
                dialogue_text = script_match.group(2).strip()
                # Check if the extracted name is a known character or looks like one
                if potential_speaker and (potential_speaker in known_character_names or \
                                          (known_character_names and process.extractOne(potential_speaker, list(known_character_names), scorer=fuzz.token_set_ratio)[1] > 85)):
                    speaker = potential_speaker
                    text = dialogue_text # Update text to be just the dialogue
                else:
                    # If it looks like a script format but the name isn't recognized,
                    # still attribute it to "Unknown Speaker" for review.
                    speaker = "Unknown Speaker"
                    text = dialogue_text # Update text to be just the dialogue

            # Rule 1 (Dialogue): If the paragraph contains dialogue markers
            elif any(marker in text for marker in ['"', '"', '"', '—', "'"]):
                speaker = "AMBIGUOUS" # Default for dialogue if no specific name found

                # Try to find speaker from dialogue tags (e.g., "said John")
                dialogue_tag_match = re.search(r'(said|asked|replied|whispered|shouted|muttered|cried|exclaimed|sighed|laughed|nodded|smiled|thought)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)', text, re.IGNORECASE)
                if dialogue_tag_match:
                    potential_speaker = dialogue_tag_match.group(2).strip()
                    if potential_speaker in known_character_names or \
                       (known_character_names and process.extractOne(potential_speaker, list(known_character_names), scorer=fuzz.token_set_ratio)[1] > 85):
                        speaker = potential_speaker
                else:
                    # Try to find a known character name directly in the paragraph
                    best_match_score = 0
                    best_match_name = None
                    for char_name in known_character_names:
                        # Use token_set_ratio for partial matches within the paragraph
                        score = fuzz.token_set_ratio(char_name, text)
                        if score > best_match_score and score > 75: # Threshold for direct name presence
                            best_match_score = score
                            best_match_name = char_name
                    if best_match_name:
                        speaker = best_match_name
            
            structured_output.append({"speaker": speaker, "text": text})
        return structured_output
