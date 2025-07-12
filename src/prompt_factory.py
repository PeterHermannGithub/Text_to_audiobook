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
        Builds a sophisticated prompt for the LLM to structure raw text into a
        JSON array of strings, each representing a distinct paragraph.
        """
        sanitized_text_content = self._sanitize_text_for_llm(text_content)

        return f"""Split the following text into a JSON array of strings. Each string in the array must be a distinct paragraph.

        RULES:
        1.  The output MUST be a valid, complete JSON array. Do not truncate it.
        2.  Your response MUST contain ONLY the JSON array. Do NOT include any other text or explanations.

        EXAMPLE INPUT:
        Paragraph one.

        Paragraph two.

        EXAMPLE OUTPUT:
        ["Paragraph one.", "Paragraph two."]

        Now, split the following text:
        ---
{sanitized_text_content}
---
        """

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
