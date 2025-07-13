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
        
        return f"""🎯 SPEAKER CLASSIFICATION TASK

⚠️ CRITICAL: You MUST output ONLY a valid JSON array. No explanations, no text, ONLY the JSON array.

📋 FORMAT REQUIREMENT:
Your response must be EXACTLY this format: ["speaker1", "speaker2", "speaker3", ...]
- Must be valid JSON
- Must contain exactly {num_lines} strings
- Must use double quotes, not single quotes
- No trailing commas

👥 SPEAKER TYPES:
- "narrator" = Text describing actions, thoughts, or scenes
- Character names = Text spoken by characters (e.g., "John", "Sarah")  
- "AMBIGUOUS" = Cannot determine who is speaking

🔥 KEY RULE: Character mentioned ≠ Character speaking
- "John walked to the door" → "narrator" (describes John)
- "I'm leaving," John said → "John" (John speaks)

📚 EXAMPLES WITH EXACT REQUIRED OUTPUT:

Example 1:
[INPUT]
1. The storm was approaching fast.
2. "We need to take shelter," Sarah warned.
3. Everyone nodded in agreement.
[REQUIRED OUTPUT] ["narrator", "Sarah", "narrator"]

Example 2:
[INPUT]  
1. ALICE: This is impossible!
2. "What do you mean?"
3. The room fell silent.
[REQUIRED OUTPUT] ["Alice", "AMBIGUOUS", "narrator"]

Example 3:
[INPUT]
1. "Ready?" Tom asked nervously.
2. Lisa checked her equipment once more.
3. "Let's do this."
[REQUIRED OUTPUT] ["Tom", "narrator", "AMBIGUOUS"]

Example 4 - CHARACTER NAME vs SPEAKER:
[INPUT]
1. Marcus stepped into the arena.
2. The crowd was cheering for Marcus.
3. "I won't lose!" Marcus shouted.
4. Marcus's heart was pounding.
[REQUIRED OUTPUT] ["narrator", "narrator", "Marcus", "narrator"]

Example 5 - MIXED CONTENT:
[INPUT]
1. [System alert: Danger detected]
2. "Help us!"
3. The explosion shook the building.
[REQUIRED OUTPUT] ["AMBIGUOUS", "AMBIGUOUS", "narrator"]{character_context}{format_context}

🎯 YOUR TASK - CLASSIFY THESE {num_lines} LINES:

{numbered_display.strip()}

⚠️ RESPOND WITH ONLY THE JSON ARRAY - NO OTHER TEXT:"""

    def create_json_correction_prompt(self, malformed_json_text):
        """
        Creates a prompt to instruct the LLM to correct malformed JSON output.
        """
        return f"""🚨 JSON FIX REQUIRED

Your previous response was not valid JSON. You MUST fix it and respond with ONLY valid JSON.

❌ BROKEN JSON:
{malformed_json_text}

✅ REQUIRED FORMAT: 
["speaker1", "speaker2", "speaker3"]

🔧 COMMON FIXES NEEDED:
- Add missing quotes around strings
- Remove trailing commas
- Use double quotes, not single
- Ensure proper array format

⚠️ RESPOND WITH ONLY THE CORRECTED JSON ARRAY - NO OTHER TEXT:"""

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
