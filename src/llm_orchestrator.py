import json
import time
import requests
import os
import re
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import google.api_core.exceptions
import google.generativeai as genai
from .prompt_factory import PromptFactory



# --- Constants ---
CONTEXT_WINDOW_SIZE = 250 # Chars to include before/after a segment for context

class LLMOrchestrator:
    def __init__(self, config):
        self.engine = config['engine']
        self.local_model = config.get('local_model')
        self.ollama_url = config.get('ollama_url', "http://localhost:11434/api/generate")
        self.log_dir = os.path.join(os.getcwd(), "logs") # Initialize log_dir here
        self.prompt_factory = PromptFactory() # Instantiate PromptFactory

        if self.engine == 'gcp':
            if not config.get('project_id') or not config.get('location'):
                raise ValueError("Project ID and location are required for GCP engine.")
            # aiplatform.init(project=config['project_id'], location=config['location'])
            self.model = genai.GenerativeModel('gemini-1.0-pro')

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

    def validate_and_parse(self, response_text, prompt=None):
        """
        Parses the LLM's XML-like output and converts it into the internal JSON format.
        Logs failed outputs and their prompts for debugging.
        """
        if not response_text:
            return []

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            self._ensure_log_directory_exists()
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            log_filename = os.path.join(self.log_dir, f"failed_json_parsing_{timestamp}.log")
            with open(log_filename, "w", encoding="utf-8") as f:
                f.write(f"--- PROMPT ---\n{prompt}\n\n")
                f.write(f"--- RAW RESPONSE ---\n{response_text}\n")
                f.write(f"--- ERROR ---\n{str(e)}\n")
            print(f"Error: JSON parsing failed. Logged to {log_filename}")
            return []

        if not isinstance(data, list):
            self._ensure_log_directory_exists()
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            log_filename = os.path.join(self.log_dir, f"failed_json_parsing_{timestamp}.log")
            with open(log_filename, "w", encoding="utf-8") as f:
                f.write(f"--- PROMPT ---\n{prompt}\n\n")
                f.write(f"--- RAW RESPONSE ---\n{response_text}\n")
                f.write(f"--- ERROR ---\nExpected list, got {type(data)}\n")
            print(f"Error: LLM did not return a list. Logged to {log_filename}")
            return []

        for item in data:
            if not isinstance(item, str):
                self._ensure_log_directory_exists()
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                log_filename = os.path.join(self.log_dir, f"failed_json_parsing_{timestamp}.log")
                with open(log_filename, "w", encoding="utf-8") as f:
                    f.write(f"--- PROMPT ---\n{prompt}\n\n")
                    f.write(f"--- RAW RESPONSE ---\n{response_text}\n")
                    f.write(f"--- ERROR ---\nExpected string in list, got {type(item)}\n")
                print(f"Error: Invalid item in list: {item}. Logged to {log_filename}")
                return []
        
        return data

    

    

    def _ensure_log_directory_exists(self):
        """
        Ensures the 'logs' directory exists in the project root.
        """
        log_dir = os.path.join(os.getcwd(), "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir