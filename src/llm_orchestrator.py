import json
import time
import requests
import os
import re
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import google.api_core.exceptions
import google.generativeai as genai
from .prompt_factory import PromptFactory
from config import settings

class LLMOrchestrator:
    def __init__(self, config):
        self.engine = config['engine']
        self.local_model = config.get('local_model')
        self.ollama_url = settings.OLLAMA_URL
        self.log_dir = os.path.join(os.getcwd(), settings.LOG_DIR)
        self.prompt_factory = PromptFactory()

        if self.engine == 'gcp':
            if not config.get('project_id') or not config.get('location'):
                raise ValueError("Project ID and location are required for GCP engine.")
            self.model = genai.GenerativeModel(settings.GCP_LLM_MODEL)

    def build_prompt(self, text_content):
        return self.prompt_factory.create_structuring_prompt(text_content)

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((requests.exceptions.RequestException, google.api_core.exceptions.ServiceUnavailable))
    )
    def get_structured_response(self, prompt):
        """
        Gets and parses a structured JSON response from the LLM.
        """
        response_text = self._get_llm_response(prompt)
        return self._parse_structured_json(response_text, prompt)

    def _get_llm_response(self, prompt):
        if self.engine == 'gcp':
            response = self.model.generate_content(prompt)
            return response.text
        else: # local
            payload = {
                "model": self.local_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0}
            }
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            return response.json().get('response', '')

    def _parse_structured_json(self, response_text, original_prompt):
        """
        Parses the LLM's output, expecting a JSON array of objects.
        """
        cleaned_text = self._extract_json_from_text(response_text)
        if not cleaned_text:
            self._log_error("No JSON array found in the response.", original_prompt, response_text)
            return []

        try:
            data = json.loads(cleaned_text)
            # Basic validation of the structure
            if isinstance(data, list) and all(isinstance(item, dict) and 'speaker' in item and 'text' in item for item in data):
                return data
            else:
                self._log_error("JSON data does not match the required structure.", original_prompt, response_text)
                return []
        except json.JSONDecodeError as e:
            self._log_error(f"JSON parsing failed: {e}", original_prompt, response_text)
            return []

    def _extract_json_from_text(self, text):
        """
        Extracts the JSON array string from a given text, handling markdown code blocks
        and extraneous text.
        """
        # Remove markdown code block fences (```json, ```, etc.)
        text = re.sub(r'^```(?:json)?\s*|```\s*$', '', text, flags=re.MULTILINE).strip()
        
        # Find the first '[' and the last ']' to isolate the JSON array
        start_index = text.find('[')
        end_index = text.rfind(']')

        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_string = text[start_index : end_index + 1]
            # Clean up any potential trailing commas before ] or }
            json_string = re.sub(r',\s*\]', ']', json_string)
            json_string = re.sub(r',\s*\}', '}', json_string)
            return json_string
        
        return None

    def _log_error(self, error_message, prompt, response_text):
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_filename = os.path.join(self.log_dir, f"llm_parsing_error_{timestamp}.log")
        with open(log_filename, "w", encoding="utf-8") as f:
            f.write(f"--- ERROR ---\n{error_message}\n\n--- PROMPT ---\n{prompt}\n\n--- RAW RESPONSE ---\n{response_text}")
        print(f"Error processing LLM response. Details logged to {log_filename}")