import os
import subprocess
from google.cloud import texttospeech
from pydub import AudioSegment
from tqdm import tqdm

from config import settings

class AudioGenerator:
    """Generates the final audiobook file from structured text and voice profiles."""

    def __init__(self, project_id=None, location=None):
        self.project_id = project_id
        self.location = location
        self.gcp_tts_client = texttospeech.TextToSpeechClient() if project_id else None
        self.temp_dir = os.path.join(settings.OUTPUT_DIR, "temp")
        os.makedirs(self.temp_dir, exist_ok=True)

    def generate_audiobook(self, structured_text, voice_profiles, output_filename):
        """
        Orchestrates the generation of the audiobook.

        Args:
            structured_text (list): The structured text with speaker information.
            voice_profiles (dict): The voice profiles for each speaker.
            output_filename (str): The desired name for the final output MP3 file.
        """
        print("\nGenerating audio segments...")
        audio_segments = []
        for i, segment in enumerate(tqdm(structured_text, desc="Generating Audio")):
            speaker = segment['speaker']
            text = segment['text']
            voice_profile = voice_profiles.get(speaker)

            if not voice_profile:
                print(f"Warning: No voice profile found for speaker '{speaker}'. Skipping segment.")
                continue

            temp_audio_path = os.path.join(self.temp_dir, f"segment_{i}.mp3")

            try:
                pitch = segment.get('pitch', 0.0)
                speaking_rate = segment.get('speaking_rate', 1.0)

                if voice_profile['engine'] == 'gcp':
                    self._synthesize_gcp_tts(text, voice_profile['suggested_voice'], temp_audio_path, pitch, speaking_rate)
                elif voice_profile['engine'] == 'local':
                    self._synthesize_local_tts(text, voice_profile['suggested_voice'], temp_audio_path)
                else:
                    print(f"Warning: Unsupported TTS engine '{voice_profile['engine']}' for speaker '{speaker}'. Skipping.")
                    continue
                
                audio_segments.append(AudioSegment.from_mp3(temp_audio_path))
            except Exception as e:
                print(f"Error generating audio for segment {i} ('{speaker}'): {e}")

        if not audio_segments:
            print("No audio segments were generated. Audiobook creation failed.")
            return

        print("\nCombining audio segments into final audiobook...")
        final_audio = sum(audio_segments)
        output_path = os.path.join(settings.OUTPUT_DIR, output_filename)
        final_audio.export(output_path, format="mp3")

        self._cleanup_temp_files()
        print(f"\nAudiobook saved successfully to {output_path}")

    def _synthesize_gcp_tts(self, text, voice_name, output_path, pitch=0.0, speaking_rate=1.0):
        if not self.gcp_tts_client:
            raise ValueError("GCP TTS client is not initialized. Cannot synthesize audio.")

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name=voice_name
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            pitch=pitch,
            speaking_rate=speaking_rate
        )

        response = self.gcp_tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        with open(output_path, "wb") as out:
            out.write(response.audio_content)

    def _synthesize_local_tts(self, text, model_name, output_path):
        """Synthesizes audio using a local TTS engine (e.g., coqui-tts)."""
        try:
            subprocess.run(
                [
                    'tts',
                    '--text', text,
                    '--model_name', model_name,
                    '--out_path', output_path
                ],
                check=True,
                capture_output=True,
                text=True
            )
        except FileNotFoundError:
            raise RuntimeError("The 'tts' command was not found. Please ensure Coqui-TTS is installed and in your PATH.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Local TTS synthesis failed: {e.stderr}")

    def _cleanup_temp_files(self):
        """Removes the temporary audio segment files."""
        for f in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, f))
        os.rmdir(self.temp_dir)
