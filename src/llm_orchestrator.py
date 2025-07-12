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



# --- Constants ---
CONTEXT_WINDOW_SIZE = 250 # Chars to include before/after a segment for context

class LLMOrchestrator:
    def __init__(self, config):
        self.engine = config['engine']
        self.local_model = config.get('local_model')
        self.ollama_url = settings.OLLAMA_URL
        self.log_dir = os.path.join(os.getcwd(), settings.LOG_DIR) # Initialize log_dir here
        self.prompt_factory = PromptFactory() # Instantiate PromptFactory

        if self.engine == 'gcp':
            if not config.get('project_id') or not config.get('location'):
                raise ValueError("Project ID and location are required for GCP engine.")
            # aiplatform.init(project=config['project_id'], location=config['location'])
            self.model = genai.GenerativeModel(settings.GCP_LLM_MODEL)

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((requests.exceptions.RequestException, 
                                       google.api_core.exceptions.ServiceUnavailable, 
                                       google.api_core.exceptions.ResourceExhausted, 
                                       google.api_core.exceptions.InternalServerError))
    )
    def get_response(self, prompt):
        """
        Handles LLM call based on engine choice with retry logic.
        """
        response_text = None
        if self.engine == 'gcp':
            response = self.model.generate_content(prompt)
            response_text = response.text
        else: # local
            payload = {
                "model": self.local_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0
                }
            }
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            response_json = response.json()
            response_text = response_json.get('response', '')
        return response_text

    def build_prompt(self, text_content):
        """
        Builds a highly constrained, robust prompt for the generative AI model.
        """
        return self.prompt_factory.create_structuring_prompt(text_content)

    def validate_and_parse(self, response_text, original_prompt=None, retries=3):
        """
        Parses the LLM's output and converts it into the internal JSON format.
        Includes retry logic and cleaning for malformed JSON.
        Logs failed outputs and their prompts for debugging.
        """
        current_response_text = response_text
        current_prompt = original_prompt

        for attempt in range(retries):
            if not current_response_text:
                self._log_error("Empty LLM response.", current_prompt, current_response_text)
                return []

            try:
                data = json.loads(current_response_text)
                if isinstance(data, list) and all(isinstance(item, str) for item in data):
                    return data
                else:
                    error_message = f"Expected list of strings, got {type(data)} or invalid items."
                    self._log_error(error_message, current_prompt, current_response_text)
                    # If structure is wrong, try cleaning and re-prompting
                    cleaned_response = self._clean_llm_response(current_response_text)
                    if cleaned_response != current_response_text:
                        current_response_text = cleaned_response
                        continue # Try parsing cleaned response
                    else:
                        # If cleaning didn't help, re-prompt
                        current_prompt = self.prompt_factory.create_json_correction_prompt(current_response_text)
                        current_response_text = self.get_response(current_prompt)
                        continue

            except json.JSONDecodeError as e:
                error_message = f"JSON parsing failed: {e}"
                self._log_error(error_message, current_prompt, current_response_text)
                
                # Attempt to clean the response and retry parsing
                cleaned_response = self._clean_llm_response(current_response_text)
                if cleaned_response != current_response_text:
                    current_response_text = cleaned_response
                    continue # Try parsing cleaned response
                else:
                    # If cleaning didn't help, re-prompt the LLM for valid JSON
                    current_prompt = self.prompt_factory.create_json_correction_prompt(current_response_text)
                    current_response_text = self.get_response(current_prompt)
                    continue
        
        # If all retries fail
        self._log_error("Failed to parse LLM response after multiple attempts.", original_prompt, response_text)
        return []

    def _log_error(self, error_message, prompt, response_text):
        self._ensure_log_directory_exists()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_filename = os.path.join(self.log_dir, f"llm_parsing_error_{timestamp}.log")
        with open(log_filename, "w", encoding="utf-8") as f:
            f.write(f"--- ERROR MESSAGE ---\n{error_message}\n\n")
            f.write(f"--- PROMPT ---\n{prompt}\n\n")
            f.write(f"--- RAW RESPONSE ---\n{response_text}\n")
        print(f"Error: {error_message}. Logged to {log_filename}")

    def _ensure_log_directory_exists(self):
        """
        Ensures the 'logs' directory exists in the project root.
        """
        log_dir = os.path.join(os.getcwd(), "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir

    

    

    def _clean_llm_response(self, response_text):
        """
        Attempts to clean the LLM response to isolate the JSON array.
        This handles cases where the LLM adds conversational text around the JSON.
        """
        # Find the first occurrence of '[' and the last occurrence of ']'
        first_bracket = response_text.find('[')
        last_bracket = response_text.rfind(']')

        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            # Extract the substring between and including the brackets
            json_string = response_text[first_bracket : last_bracket + 1]
            return json_string
        else:
            # If brackets are not found, return the original text (parsing will fail)
            return response_text