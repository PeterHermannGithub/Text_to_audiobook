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
        Builds a highly constrained, robust prompt for the generative AI model
        to structure raw text into a dialogue-focused JSON format.
        """
        # 1. SANITIZE FIRST: Call the sanitization method on the input text.
        #    This logic must happen BEFORE the final prompt is constructed.
        sanitized_text_content = self._sanitize_text_for_llm(text_content)

        # 2. CONSTRUCT AND RETURN THE FINAL PROMPT:
        return f"""Split the following text into a JSON array of strings. Each string in the array must be a distinct paragraph.

Example Input:
Paragraph one.

Paragraph two.

Example Output:
["Paragraph one.", "Paragraph two."]

Now, split the following text. IMPORTANT: Only output the JSON array. Do not include any other text or explanations. Keep the response concise and complete, do NOT truncate the JSON.
---
{sanitized_text_content}
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
