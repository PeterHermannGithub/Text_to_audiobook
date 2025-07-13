import re
import json

class PromptFactory:
    """
    Factory class responsible for constructing various prompts for the LLM.
    Separates prompt engineering logic from LLM orchestration.
    """

    def __init__(self):
        pass

    def create_structuring_prompt(self, text_content, context_hint="", text_metadata=None):
        """
        DEPRECATED: Old segmentation prompt. Use create_speaker_classification_prompt instead.
        Maintained for backward compatibility during transition.
        """
        return self.create_speaker_classification_prompt(text_content, text_metadata)
    
    def create_speaker_classification_prompt(self, numbered_lines, text_metadata=None):
        """
        Creates a speaker classification prompt for the LLM.
        
        The LLM receives pre-segmented, numbered lines and must return exactly 
        the same number of speaker names in a JSON array.
        
        Args:
            numbered_lines: List of strings, each representing a pre-segmented line
            text_metadata: Metadata containing character names and format info
            
        Returns:
            String prompt for speaker classification
        """
        if not numbered_lines:
            return ""
            
        # Extract context information from metadata
        character_profiles = text_metadata.get('character_profiles', []) if text_metadata else []
        character_names = list(text_metadata.get('potential_character_names', set())) if text_metadata else []
        dialogue_markers = text_metadata.get('dialogue_markers', set()) if text_metadata else set()
        is_script_like = text_metadata.get('is_script_like', False) if text_metadata else False
        
        # Build enhanced character context using rich profiles
        character_context = ""
        if character_profiles:
            context_lines = []
            for profile in character_profiles[:10]:  # Limit to top 10 characters
                profile_line = f"- {profile['name']}"
                
                # Add pronouns for gender context
                if profile['pronouns']:
                    gender_hint = self._infer_gender_from_pronouns(profile['pronouns'])
                    if gender_hint:
                        profile_line += f" ({gender_hint})"
                
                # Add titles if present
                if profile['titles']:
                    profile_line += f" [titles: {', '.join(profile['titles'][:3])}]"
                
                # Add aliases if present
                if profile['aliases']:
                    profile_line += f" [aliases: {', '.join(profile['aliases'][:2])}]"
                
                context_lines.append(profile_line)
            
            character_context = f"\n\nKNOWN CHARACTERS:\n" + "\n".join(context_lines)
        elif character_names:
            # Fallback to simple names for backward compatibility
            character_context = f"\n\nKNOWN CHARACTERS: {', '.join(sorted(character_names)[:15])}"
        
        format_context = ""
        if is_script_like:
            format_context = "\n\nFORMAT NOTE: This text contains script-style dialogue (Character: speech)."
        
        dialogue_context = ""
        if dialogue_markers:
            dialogue_context = f"\n\nDIALOGUE MARKERS: {', '.join(list(dialogue_markers)[:5])}"
        
        # Create numbered line display
        numbered_display = ""
        for i, line in enumerate(numbered_lines, 1):
            numbered_display += f"{i}. {line}\n"
        
        num_lines = len(numbered_lines)
        
        return f"""TASK: Identify the speaker for each numbered line below.

RULES:
1. Return ONLY a JSON array with exactly {num_lines} speaker names
2. Use these speaker types:
   - "narrator" for descriptive/narrative text
   - Character names for dialogue (e.g. "John", "Sarah")
   - "AMBIGUOUS" if unclear who is speaking{character_context}{format_context}

CRITICAL DISTINCTION:
- If a line DESCRIBES a character's actions/state → "narrator"
- If a line contains words SPOKEN by a character → character name
- Character name in text ≠ character speaking!

EXAMPLES:

[INPUT]
1. The old castle stood before them.
2. "Let's go inside," Sarah said.
3. John nodded in agreement.
[OUTPUT] ["narrator", "Sarah", "narrator"]

[INPUT]
1. JOHN: I can't believe this.
2. "What do you mean?" 
3. The silence stretched between them.
[OUTPUT] ["John", "AMBIGUOUS", "narrator"]

[INPUT]
1. "Hello there," John said.
2. Mary smiled back at him.
3. "How are you today?"
[OUTPUT] ["John", "narrator", "AMBIGUOUS"]

[INPUT - CHARACTER NAME vs SPEAKER EXAMPLES]
1. John walked toward the door.
2. John's blood was boiling with anger.
3. "I'm ready," John said.
4. John shot forward like a meteor.
[OUTPUT] ["narrator", "narrator", "John", "narrator"]

[INPUT - MIXED CONTENT EXAMPLES]
1. [You dare...!]
2. "Next."
3. The magic power was affecting John.
[OUTPUT] ["AMBIGUOUS", "AMBIGUOUS", "narrator"]

NOW CLASSIFY THESE {num_lines} LINES:

{numbered_display.strip()}

JSON ARRAY:"""

    def create_json_correction_prompt(self, malformed_json_text):
        """
        Creates a prompt to instruct the LLM to correct malformed JSON output.
        """
        return f"""The previous response was malformed JSON. You MUST correct it and provide a valid JSON array of strings.

Here is the malformed JSON that needs correction:
---
{malformed_json_text}
---

Your response MUST be ONLY the corrected JSON array of strings. Do NOT include any other text, explanations, or conversational filler. The JSON must be perfectly valid and complete, do NOT truncate it.
"""

    def create_character_description_prompt(self, character_name, dialogue_sample):
        """
        Creates a prompt to instruct the LLM to describe a character for voice casting.
        """
        return f"""Based on the following dialogue from '{character_name}', provide a concise, one-sentence description of their likely voice characteristics. 
        Focus on age, gender, and tone.

        Example: 'An elderly male with a deep, commanding voice.'
        Example: 'A young female with a bright, energetic voice.'

        Dialogue sample:
        ---
        {dialogue_sample}
        ---
        """

    def create_emotion_annotation_prompt(self, sentence):
        """
        Creates a prompt to instruct the LLM to annotate a sentence with emotions.
        """
        return f"""Analyze the following sentence and return a JSON object with three keys: "emotion", "pitch", and "speaking_rate".

        RULES:
        1.  "emotion" should be a single descriptive word (e.g., "happy", "sad", "angry", "calm").
        2.  "pitch" should be a float between -20.0 and 20.0.
        3.  "speaking_rate" should be a float between 0.25 and 4.0.

        EXAMPLE INPUT:
        "I'm so excited to go to the park!"

        EXAMPLE OUTPUT:
        {{
            "emotion": "excited",
            "pitch": 5.0,
            "speaking_rate": 1.2
        }}

        Now, process the following sentence:
        ---
        {sentence}
        ---
        """

    def _sanitize_text_for_llm(self, text):
        """
        Sanitizes text by replacing problematic Unicode characters and normalizing whitespace.
        """
        # Replace problematic Unicode quotes
        text = text.replace('「', '"').replace('」', '"')
        text = text.replace('『', '"').replace('』', '"')
        text = text.replace('《', '"').replace('》', '"')
        text = text.replace('〈', '"').replace('〉', '"')

        # Normalize all line endings to Unix-style
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Collapse multiple newlines into at most two (for paragraph breaks)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        
        # Collapse any remaining horizontal whitespace (spaces, tabs)
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text
    
    def _infer_gender_from_pronouns(self, pronouns):
        """
        Infer gender hint from pronoun list for LLM context.
        
        Args:
            pronouns: List of pronouns associated with character
            
        Returns:
            String gender hint or None
        """
        pronoun_set = set(p.lower() for p in pronouns)
        
        # Check for male pronouns
        male_pronouns = {'he', 'him', 'his', 'himself'}
        if pronoun_set & male_pronouns:
            return 'male'
        
        # Check for female pronouns
        female_pronouns = {'she', 'her', 'hers', 'herself'}
        if pronoun_set & female_pronouns:
            return 'female'
        
        # Check for neutral/plural pronouns
        neutral_pronouns = {'they', 'them', 'their', 'theirs', 'themselves'}
        if pronoun_set & neutral_pronouns:
            return 'neutral'
        
        return None
