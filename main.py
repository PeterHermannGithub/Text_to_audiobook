import re
import os
import fitz  # PyMuPDF
import json
from pathlib import Path
from collections import defaultdict
import random
from pydub import AudioSegment

# Use one of these TTS engines (uncomment as needed):
# For Google Cloud TTS (requires account setup)
from google.cloud import texttospeech

# For pyttsx3 (works offline, lower quality but free)
import pyttsx3

# For ElevenLabs (high quality, limited free tier)
# import requests

class PDFTextExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        
    def extract_text(self):
        """Extract text from PDF file"""
        doc = fitz.open(self.pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

class DialogueProcessor:
    def __init__(self, text):
        self.text = text
        self.characters = set()
        self.narrator_id = "narrator"
        self.system_id = "system"
        
    def process_text(self):
        """Process text to identify different speech patterns"""
        segments = []
        remaining_text = self.text
        
        # Process the text in order of pattern specificity
        
        # 1. Process dashed dialogues (–character: text)
        dash_pattern = r'–([^:]+):(.*?)(?=–|$|\n\n)'
        dash_matches = re.findall(dash_pattern, remaining_text, re.DOTALL)
        
        for speaker, content in dash_matches:
            speaker = speaker.strip()
            content = content.strip()
            self.characters.add(speaker)
            
            # Find the position in the text and mark for removal
            match_text = f"–{speaker}: {content}"
            pos = remaining_text.find(match_text)
            if pos >= 0:
                # Add any narrative text before this dialogue
                if pos > 0:
                    narrative = remaining_text[:pos].strip()
                    if narrative:
                        segments.append({
                            "speaker": self.narrator_id,
                            "content": narrative,
                            "type": "narrative"
                        })
                
                # Add the dialogue
                segments.append({
                    "speaker": speaker,
                    "content": content,
                    "type": "dialogue"
                })
                
                # Remove processed text
                remaining_text = remaining_text[pos + len(match_text):].lstrip()
        
        # 2. Process bracketed system messages
        system_pattern = r'\[(.*?)\]'
        system_matches = re.findall(system_pattern, remaining_text)
        
        for content in system_matches:
            content = content.strip()
            
            # Find the position in the text and mark for removal
            match_text = f"[{content}]"
            pos = remaining_text.find(match_text)
            if pos >= 0:
                # Add any narrative text before this system message
                if pos > 0:
                    narrative = remaining_text[:pos].strip()
                    if narrative:
                        segments.append({
                            "speaker": self.narrator_id,
                            "content": narrative,
                            "type": "narrative"
                        })
                
                # Add the system message
                segments.append({
                    "speaker": self.system_id,
                    "content": content,
                    "type": "system"
                })
                
                # Remove processed text
                remaining_text = remaining_text[pos + len(match_text):].lstrip()
        
        # 3. Process quoted inner thoughts or direct speech
        quote_pattern = r'"([^"]*)"'
        remaining_segments = []
        last_end = 0
        
        for match in re.finditer(quote_pattern, remaining_text):
            content = match.group(1).strip()
            start_pos = match.start()
            end_pos = match.end()
            
            # Check if there's text before the quote
            if start_pos > last_end:
                narrative = remaining_text[last_end:start_pos].strip()
                if narrative:
                    remaining_segments.append({
                        "speaker": self.narrator_id,
                        "content": narrative,
                        "type": "narrative"
                    })
            
            # Add the quoted text
            remaining_segments.append({
                "speaker": self.narrator_id,  # Initially assign to narrator
                "content": content,
                "type": "quoted"  # Mark as quoted to potentially assign to a character later
            })
            
            last_end = end_pos
        
        # Add any remaining text as narrative
        if last_end < len(remaining_text):
            narrative = remaining_text[last_end:].strip()
            if narrative:
                remaining_segments.append({
                    "speaker": self.narrator_id,
                    "content": narrative,
                    "type": "narrative"
                })
        
        # Add remaining segments to the main segments list
        segments.extend(remaining_segments)
        
        # Sort all segments by their appearance order in the text (would require position tracking)
        # For now, we keep the order of processing, prioritizing dialogue and system messages
        
        # Add standard characters
        self.characters.add(self.narrator_id)
        self.characters.add(self.system_id)
        
        return segments
    
    def get_characters(self):
        """Return the set of identified characters"""
        return self.characters

class CharacterVoiceMapper:
    def __init__(self):
        self.character_profiles = {}
        
    def import_character_profiles(self, profiles_path):
        """Import character profiles from a JSON file"""
        if os.path.exists(profiles_path):
            with open(profiles_path, 'r') as f:
                self.character_profiles = json.load(f)
                
    def create_profile_template(self, characters, output_path="character_profiles.json"):
        """Create a template JSON file for character profiles"""
        template = {}
        
        for character in characters:
            template[character] = {
                "gender": "unknown",  # male/female/unknown
                "age": "adult",  # child/young/adult/elderly
                "personality": [],  # list of traits: calm, energetic, serious, cheerful, etc.
                "voice_preference": ""  # optional specific voice id
            }
            
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=4)
            
        return output_path
    
    def match_character_to_voice(self, character, available_voices, engine="pyttsx3"):
        """Match a character to an appropriate voice based on profile"""
        profile = self.character_profiles.get(character, {})
        
        # If there's a specific voice preference, use it
        if profile.get("voice_preference"):
            for voice in available_voices:
                if voice["id"] == profile.get("voice_preference"):
                    return voice
        
        # Match based on gender
        gender = profile.get("gender", "unknown")
        matching_voices = []
        
        if engine == "google":
            gender_key = "ssml_gender"
            male_value = "MALE"
            female_value = "FEMALE"
        elif engine == "elevenlabs":
            gender_key = "gender"
            male_value = "male"
            female_value = "female"
        else:  # pyttsx3
            # pyttsx3 voices often have "male" or "female" in their name
            gender_key = "name"
            male_value = "male"
            female_value = "female"
        
        for voice in available_voices:
            if gender == "male" and male_value.lower() in str(voice.get(gender_key, "")).lower():
                matching_voices.append(voice)
            elif gender == "female" and female_value.lower() in str(voice.get(gender_key, "")).lower():
                matching_voices.append(voice)
        
        # If we found matching voices by gender, choose one randomly
        if matching_voices:
            return random.choice(matching_voices)
        
        # Otherwise, choose any voice randomly
        return random.choice(available_voices) if available_voices else None
    
    def create_character_voice_map(self, characters, available_voices, engine="pyttsx3"):
        """Create a mapping of characters to voices"""
        character_voices = {}
        
        # Special assignments for narrator and system
        for special_character in ["narrator", "system"]:
            if special_character in characters:
                # Find suitable voices for these special roles
                if special_character == "narrator":
                    # Try to find a neutral, clear voice for narrator
                    for voice in available_voices:
                        if "neutral" in str(voice).lower() or "clear" in str(voice).lower():
                            character_voices[special_character] = voice
                            break
                elif special_character == "system":
                    # Try to find a robotic or computerized voice for system messages
                    for voice in available_voices:
                        if "robot" in str(voice).lower() or "computer" in str(voice).lower():
                            character_voices[special_character] = voice
                            break
                
                # If no special voice found, assign randomly
                if special_character not in character_voices:
                    character_voices[special_character] = random.choice(available_voices) if available_voices else None
                
                # Remove from set to process
                if special_character in characters:
                    characters.remove(special_character)
        
        # Assign remaining characters
        used_voices = set()
        for character in characters:
            # Try to find a voice that hasn't been used yet
            available = [v for v in available_voices if v not in used_voices]
            
            # If all voices have been used, allow reuse
            if not available:
                available = available_voices
            
            # Match based on character profile
            voice = self.match_character_to_voice(character, available, engine)
            
            if voice:
                character_voices[character] = voice
                used_voices.add(voice)
        
        return character_voices

class TTSEngine:
    """Abstract base class for TTS engines"""
    def get_available_voices(self):
        raise NotImplementedError("Subclasses must implement get_available_voices()")
    
    def generate_speech(self, text, voice, output_file):
        raise NotImplementedError("Subclasses must implement generate_speech()")

class PyttsxEngine(TTSEngine):
    """Local TTS engine using pyttsx3"""
    def __init__(self):
        self.engine = pyttsx3.init()
    
    def get_available_voices(self):
        """Get available voices from pyttsx3"""
        voices = self.engine.getProperty('voices')
        return [{"id": i, "name": voice.name, "languages": voice.languages} for i, voice in enumerate(voices)]
    
    def generate_speech(self, text, voice, output_file):
        """Generate speech using pyttsx3"""
        # Set voice
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[voice["id"]].id)
        
        # Set rate and volume
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        
        # Generate audio file
        self.engine.save_to_file(text, output_file)
        self.engine.runAndWait()
        
        return output_file

class GoogleTTSEngine(TTSEngine):
    """TTS engine using Google Cloud Text-to-Speech"""
    def __init__(self, credentials_path=None):
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        self.client = texttospeech.TextToSpeechClient()
    
    def get_available_voices(self):
        """Get available voices from Google Cloud TTS"""
        response = self.client.list_voices()
        return [{"id": voice.name, "name": voice.name, "ssml_gender": voice.ssml_gender} 
                for voice in response.voices if "en-" in voice.name]
    
    def generate_speech(self, text, voice, output_file):
        """Generate speech using Google Cloud TTS"""
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        voice_params = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name=voice["name"],
            ssml_gender=voice.get("ssml_gender", texttospeech.SsmlVoiceGender.NEUTRAL)
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        response = self.client.synthesize_speech(
            input=synthesis_input, voice=voice_params, audio_config=audio_config
        )
        
        with open(output_file, "wb") as out:
            out.write(response.audio_content)
            
        return output_file

class ElevenLabsEngine(TTSEngine):
    """TTS engine using ElevenLabs API"""
    def __init__(self, api_key):
        self.api_key = api_key
    
    def get_available_voices(self):
        """Get available voices from ElevenLabs API"""
        url = "https://api.elevenlabs.io/v1/voices"
        headers = {"xi-api-key": self.api_key}
        
        response = requests.get(url, headers=headers)
        voices = response.json().get("voices", [])
        
        return [{"id": voice["voice_id"], "name": voice["name"], "gender": self._infer_gender(voice)} 
                for voice in voices]
    
    def _infer_gender(self, voice):
        """Infer gender from voice labels or name"""
        # This is a simple heuristic and might not be accurate
        name = voice.get("name", "").lower()
        labels = voice.get("labels", {})
        
        if "gender" in labels:
            return labels["gender"]
        
        male_indicators = ["male", "man", "boy", "guy", "masculine"]
        female_indicators = ["female", "woman", "girl", "feminine"]
        
        for indicator in male_indicators:
            if indicator in name:
                return "male"
                
        for indicator in female_indicators:
            if indicator in name:
                return "female"
                
        return "unknown"
    
    def generate_speech(self, text, voice, output_file):
        """Generate speech using ElevenLabs API"""
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice['id']}"
        
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        response = requests.post(url, json=data, headers=headers)
        
        with open(output_file, "wb") as out:
            out.write(response.content)
            
        return output_file

class AudiobookCreator:
    def __init__(self, output_dir="output_audio"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(exist_ok=True)
    
    def create_audiobook(self, pdf_path, tts_engine="pyttsx3", character_profiles=None, 
                         output_file="audiobook.mp3", credentials=None):
        """Create an audiobook from a PDF file"""
        # Extract text from PDF
        extractor = PDFTextExtractor(pdf_path)
        text = extractor.extract_text()
        
        # Process dialogue
        processor = DialogueProcessor(text)
        segments = processor.process_text()
        characters = processor.get_characters()
        
        # Initialize TTS engine
        if tts_engine == "google":
            engine = GoogleTTSEngine(credentials_path=credentials)
        elif tts_engine == "elevenlabs":
            engine = ElevenLabsEngine(api_key=credentials)
        else:
            engine = PyttsxEngine()
        
        # Get available voices
        available_voices = engine.get_available_voices()
        
        # Map characters to voices
        mapper = CharacterVoiceMapper()
        if character_profiles:
            mapper.import_character_profiles(character_profiles)
        else:
            # Create a template for character profiles
            profile_path = mapper.create_profile_template(characters)
            print(f"Character profile template created at: {profile_path}")
            print("Edit this file to customize character voices, then run again with --character-profiles option")
        
        character_voices = mapper.create_character_voice_map(characters, available_voices, tts_engine)
        
        # Generate speech for each segment
        audio_files = []
        
        for i, segment in enumerate(segments):
            speaker = segment["speaker"]
            content = segment["content"]
            segment_type = segment["type"]
            
            if not content.strip():
                continue
            
            # Get voice for this speaker
            voice = character_voices.get(speaker)
            
            if not voice:
                print(f"Warning: No voice assigned to character '{speaker}'. Using narrator voice.")
                voice = character_voices.get("narrator")
            
            # Add appropriate speech marks for quoted text
            if segment_type == "quoted":
                # Add pauses and speech marks in the synthesized speech
                content = f'<break time="300ms"/> "{content}" <break time="300ms"/>'
            
            # Generate speech
            output_file_path = f"{self.output_dir}/segment_{i:04d}_{speaker}.mp3"
            engine.generate_speech(content, voice, output_file_path)
            audio_files.append(output_file_path)
            
            print(f"Generated speech for segment {i+1}/{len(segments)}: {speaker}")
        
        # Combine audio files
        self.combine_audio_files(audio_files, output_file)
        
        return output_file
    
    # Replace pydub with simpler code for testing
# Instead of using AudioSegment, just use simple file operations

def combine_audio_files(audio_files, output_file="audiobook.mp3"):
    """Simple concatenation of audio files (requires ffmpeg in PATH)"""
    import subprocess
    
    # Create a file list for ffmpeg
    with open('file_list.txt', 'w') as f:
        for audio_file in audio_files:
            f.write(f"file '{audio_file}'\n")
    
    # Use ffmpeg to concatenate files
    subprocess.call([
        'ffmpeg', '-f', 'concat', '-safe', '0', 
        '-i', 'file_list.txt', '-c', 'copy', output_file
    ])
    
    # Clean up
    import os
    os.remove('file_list.txt')
    return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert a PDF novel to an audiobook with different voices")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", default="audiobook.mp3", help="Output audiobook filename")
    parser.add_argument("--engine", choices=["pyttsx3", "google", "elevenlabs"], default="pyttsx3",
                      help="TTS engine to use")
    parser.add_argument("--credentials", help="Path to credentials file or API key")
    parser.add_argument("--character-profiles", help="Path to character profiles JSON file")
    
    args = parser.parse_args()
    
    creator = AudiobookCreator()
    creator.create_audiobook(
        args.pdf_path,
        tts_engine=args.engine,
        character_profiles=args.character_profiles,
        output_file=args.output,
        credentials=args.credentials
    )