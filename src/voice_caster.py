
import json
import time
import requests
from google.cloud import aiplatform
import google.generativeai as genai
from google.cloud import texttospeech
from config import settings

class VoiceCaster:
    """Suggests voice profiles for characters based on structured text."""

    def __init__(self, engine=settings.DEFAULT_LLM_ENGINE, project_id=None, location=None, local_model=settings.DEFAULT_LOCAL_MODEL):
        self.engine = engine
        self.local_model = local_model
        self.project_id = project_id
        self.location = location

        if self.engine == 'gcp':
            if not project_id or not location:
                raise ValueError("Project ID and location are required for GCP engine.")
            aiplatform.init(project=project_id, location=location)
            self.llm_model = genai.GenerativeModel(settings.GCP_LLM_MODEL)
        elif self.engine == 'local':
            self.ollama_url = settings.OLLAMA_URL
        
        self.tts_client = texttospeech.TextToSpeechClient()

    def cast_voices(self, structured_text):
        """
        Identifies unique speakers and suggests voice profiles.

        Args:
            structured_text (list): List of dictionaries with 'speaker' and 'text'.

        Returns:
            dict: A dictionary of voice profiles for each unique speaker.
        """
        unique_speakers = self._get_unique_speakers(structured_text)
        voice_profiles = {}

        available_voices = self._get_available_gcp_voices()

        for speaker in unique_speakers:
            if speaker == "narrator":
                # Default narrator voice
                voice_profiles[speaker] = {
                    "suggested_voice": "en-US-Wavenet-D", # A common, clear male voice
                    "engine": "gcp",
                    "notes": "Default narrator voice."
                }
                continue

            # Get character description using LLM
            character_description = self._get_character_description(speaker, structured_text)
            
            # Suggest voice based on description
            suggested_voice = self._suggest_voice_from_description(character_description, available_voices)

            voice_profiles[speaker] = {
                "suggested_voice": suggested_voice.name if suggested_voice else None,
                "engine": "gcp",
                "notes": character_description
            }
        return voice_profiles

    def _get_unique_speakers(self, structured_text):
        """
        Extracts all unique speaker names from the structured text.
        """
        speakers = set()
        for segment in structured_text:
            speakers.add(segment['speaker'])
        return list(speakers)

    def _get_character_description(self, speaker, structured_text):
        """
        Uses LLM to get a brief description of the character based on their lines.
        """
        # Collect a sample of lines for the speaker
        sample_lines = []
        for segment in structured_text:
            if segment['speaker'] == speaker:
                sample_lines.append(segment['text'])
            if len(sample_lines) >= 5: # Limit sample to 5 lines
                break
        
        sample_text = " ".join(sample_lines)
        if not sample_text:
            return f"No lines found for {speaker}."

        prompt = f"""Based on the following text spoken by or describing '{speaker}', provide a brief description of their likely gender, age, and personality traits. Focus on characteristics relevant for voice casting. Example: 'Male, adult, calm and authoritative.'

        Text sample for '{speaker}': """{sample_text}"""
        """
        
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
                    "options": {
                        "temperature": 0.0
                    }
                }
                response = requests.post(self.ollama_url, json=payload)
                response.raise_for_status()
                return json.loads(response.text).get('response').strip()
            except requests.exceptions.RequestException as e:
                return f"Error getting local description for {speaker}: {e}"
            except json.JSONDecodeError:
                return f"Error decoding local LLM response for {speaker}."

    def _get_available_gcp_voices(self):
        """
        Fetches a list of available Google Cloud TTS voices.
        """
        voices = self.tts_client.list_voices().voices
        return voices

    def _suggest_voice_from_description(self, description, available_voices):
        """
        Suggests a suitable GCP voice based on the character description.
        This is a simplified matching logic and can be improved.
        """
        # Simple keyword matching for demonstration
        description_lower = description.lower()

        # Prioritize Wavenet voices for quality
        for voice in available_voices:
            if "wavenet" in voice.name.lower() and "en-us" in voice.name.lower():
                if "male" in description_lower and voice.ssml_gender == texttospeech.SsmlVoiceGender.MALE:
                    return voice
                if "female" in description_lower and voice.ssml_gender == texttospeech.SsmlVoiceGender.FEMALE:
                    return voice
        
        # Fallback to standard voices if no wavenet match
        for voice in available_voices:
            if "standard" in voice.name.lower() and "en-us" in voice.name.lower():
                if "male" in description_lower and voice.ssml_gender == texttospeech.SsmlVoiceGender.MALE:
                    return voice
                if "female" in description_lower and voice.ssml_gender == texttospeech.SsmlVoiceGender.FEMALE:
                    return voice

        # If no specific match, return the first English voice found
        for voice in available_voices:
            if "en-us" in voice.name.lower():
                return voice
        
        return None

