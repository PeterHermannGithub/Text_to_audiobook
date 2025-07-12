
import json
import time
import requests
import subprocess
from google.cloud import aiplatform
import google.generativeai as genai
from google.cloud import texttospeech
from config import settings
from src.prompt_factory import PromptFactory

class VoiceCaster:
    """Suggests voice profiles for characters based on structured text."""

    def __init__(self, engine=settings.DEFAULT_LLM_ENGINE, project_id=None, location=None, local_model=settings.DEFAULT_LOCAL_MODEL, voice_quality="premium"):
        self.engine = engine
        self.local_model = local_model
        self.project_id = project_id
        self.location = location
        self.voice_quality = voice_quality
        self.prompt_factory = PromptFactory()

        if self.engine == 'gcp':
            if not project_id or not location:
                raise ValueError("Project ID and location are required for GCP engine.")
            aiplatform.init(project=project_id, location=location)
            self.llm_model = genai.GenerativeModel(settings.GCP_LLM_MODEL)
        elif self.engine == 'local':
            self.ollama_url = settings.OLLAMA_URL
        
        try:
            self.tts_client = texttospeech.TextToSpeechClient() if project_id else None
        except Exception as e:
            print(f"Warning: Could not initialize GCP TTS client. GCP voice suggestions will be disabled. Error: {e}")
            self.tts_client = None

    def cast_voices(self, structured_text):
        """
        Identifies unique speakers and suggests voice profiles.
        """
        unique_speakers = self._get_unique_speakers(structured_text)
        voice_profiles = {}

        available_gcp_voices = self._get_available_gcp_voices()
        available_local_voices = self._get_available_local_voices()

        for speaker in unique_speakers:
            if speaker.lower() == "narrator":
                voice_profiles[speaker] = {
                    "suggested_voice": "en-US-Wavenet-D" if self.voice_quality == "premium" else "en-US-Standard-D",
                    "engine": "gcp",
                    "notes": "Default narrator voice."
                }
                continue

            character_description = self._get_character_description(speaker, structured_text)
            
            suggested_voice, engine = self._suggest_voice_from_description(
                character_description, 
                available_gcp_voices, 
                available_local_voices
            )

            voice_profiles[speaker] = {
                "suggested_voice": suggested_voice.name if hasattr(suggested_voice, 'name') else suggested_voice,
                "engine": engine,
                "notes": character_description
            }
        return voice_profiles

    def _get_unique_speakers(self, structured_text):
        """
        Extracts all unique speaker names from the structured text.
        """
        return sorted(list(set(segment['speaker'] for segment in structured_text)))

    def _get_character_description(self, speaker, structured_text):
        """
        Uses LLM to get a brief description of the character based on their lines.
        """
        sample_lines = [segment['text'] for segment in structured_text if segment['speaker'] == speaker][:5]
        sample_text = " ".join(sample_lines)
        
        if not sample_text:
            return f"No lines found for {speaker}."

        prompt = self.prompt_factory.create_character_description_prompt(speaker, sample_text)
        
        if self.engine == 'gcp':
            try:
                response = self.llm_model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                return f"Error getting GCP description for {speaker}: {e}"
        else: # local
            try:
                payload = {
                    "model": self.local_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0}
                }
                response = requests.post(self.ollama_url, json=payload)
                response.raise_for_status()
                return json.loads(response.text).get('response', '').strip()
            except requests.exceptions.RequestException as e:
                return f"Error getting local description for {speaker}: {e}"
            except json.JSONDecodeError:
                return f"Error decoding local LLM response for {speaker}."

    def _get_available_gcp_voices(self):
        """
        Fetches a list of available Google Cloud TTS voices.
        """
        if not self.tts_client:
            return []
        try:
            return self.tts_client.list_voices().voices
        except Exception as e:
            print(f"Warning: Could not fetch GCP voices. GCP voice suggestions will be disabled. Error: {e}")
            return []

    def _get_available_local_voices(self):
        """
        Fetches a list of available local TTS voices (e.g., from coqui-tts).
        Assumes 'tts --list_models' command is available.
        """
        try:
            result = subprocess.run(['tts', '--list_models'], capture_output=True, text=True, check=True)
            return ["tts_models/en/ljspeech/tacotron2-DDC", "tts_models/en/vctk/vits"]
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"Info: Local TTS command not found or failed. Local voice suggestions will be disabled. Error: {e}")
            return []

    def _suggest_voice_from_description(self, description, gcp_voices, local_voices):
        """
        Suggests a suitable voice based on the character description and quality preference.
        """
        description_lower = description.lower()
        male_keywords = ['male', 'man', 'boy', 'he']
        female_keywords = ['female', 'woman', 'girl', 'she']
        is_male = any(kw in description_lower for kw in male_keywords)
        is_female = any(kw in description_lower for kw in female_keywords)

        # Filter GCP voices based on quality
        if self.voice_quality == "premium":
            quality_keywords = ["wavenet", "neural2"]
        else:
            quality_keywords = ["standard"]

        filtered_gcp_voices = [v for v in gcp_voices if any(q in v.name.lower() for q in quality_keywords) and "en-us" in v.name.lower()]

        # 1. Search within the filtered GCP voices
        for voice in filtered_gcp_voices:
            if is_male and voice.ssml_gender == texttospeech.SsmlVoiceGender.MALE:
                return voice, "gcp"
            if is_female and voice.ssml_gender == texttospeech.SsmlVoiceGender.FEMALE:
                return voice, "gcp"

        # 2. Fallback to any voice in the filtered list if gender match fails
        if filtered_gcp_voices:
            return filtered_gcp_voices[0], "gcp"

        # 3. Fallback to local voices if no GCP voices match
        if local_voices:
            return local_voices[0], "local"

        # 4. Final fallback to any available GCP voice if all else fails
        if gcp_voices:
            return gcp_voices[0], "gcp"
            
        return None, None

