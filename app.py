import argparse
import os
import json
from src.text_processing.text_extractor import TextExtractor
from src.text_structurer import TextStructurer
from src.output.voice_caster import VoiceCaster
from src.output.audio_generator import AudioGenerator
from src.emotion_annotator import EmotionAnnotator
from config import settings

def main():
    parser = argparse.ArgumentParser(description="Convert a document to an audiobook.")
    parser.add_argument("input_file", nargs='?', help="Path to the input file (.txt, .md, .pdf, .docx, .epub, .mobi). Required if --structured-input-file is not used.")
    parser.add_argument("--structured-input-file", help="Path to a pre-structured JSON file. If provided, skips text extraction and structuring.")
    parser.add_argument("--output-filename", help="The desired name for the final output MP3 file.")
    parser.add_argument("--skip-voice-casting", action="store_true", help="If set, skips the voice casting phase.")
    parser.add_argument("--add-emotions", action="store_true", help="If set, adds emotional annotations to the text segments.")
    parser.add_argument("--voice-quality", default="premium", choices=["standard", "premium"], help="The quality of the GCP voices to use.")
    parser.add_argument("--engine", default=settings.DEFAULT_LLM_ENGINE, choices=["local", "gcp"], help="AI engine to use for text structuring and character description (LLM). Default is local.")
    parser.add_argument("--model", default=settings.DEFAULT_LOCAL_MODEL, choices=["mistral", "llama3"], help="Local model to use if --engine is 'local'.")
    parser.add_argument("--project_id", help="Google Cloud project ID. Required if --engine is 'gcp' or if --skip-voice-casting is not set.")
    parser.add_argument("--location", default=settings.GCP_LOCATION, help="Google Cloud location. Required if --engine is 'gcp' or if --skip-voice-casting is not set.")
    parser.add_argument("--debug-llm", action="store_true", help="Enable detailed LLM interaction logging (prompts, responses, processing steps). Debug logs are written to logs/llm_debug.log")
    args = parser.parse_args()

    # Handle debug LLM flag - override settings if enabled
    if args.debug_llm:
        print("🔍 LLM Debug Mode Enabled - Detailed logging will be written to logs/llm_debug.log")
        print("   This will log all prompts, responses, and processing steps for debugging purposes.")
        settings.LLM_DEBUG_LOGGING = True
        # Ensure we have detailed logging for the main system too
        settings.LOG_LEVEL = "DEBUG"
        settings.CONSOLE_LOG_LEVEL = "DEBUG"

    if not args.input_file and not args.structured_input_file:
        parser.error("Either input_file or --structured-input-file must be provided.")

    if args.engine == 'gcp' or not args.skip_voice_casting:
        if not args.project_id:
            parser.error("Google Cloud project ID is required when using --engine gcp or when not skipping voice casting.")

    structured_text = None
    output_filename_base = None

    try:
        if args.structured_input_file:
            structured_input_path = os.path.abspath(args.structured_input_file)
            if not os.path.exists(structured_input_path):
                print(f"Error: Structured input file not found at {structured_input_path}")
                return
            print(f"Loading structured text from {structured_input_path}...")
            with open(structured_input_path, 'r', encoding='utf-8') as f:
                structured_text = json.load(f)
            print("Structured text loaded successfully.")
            output_filename_base = os.path.splitext(os.path.basename(structured_input_path))[0].replace("_structured", "")
        else:
            input_path = os.path.abspath(args.input_file)
            if not os.path.exists(input_path):
                print(f"Error: Input file not found at {input_path}")
                return
            print(f"Extracting text from {input_path}...")
            extractor = TextExtractor()
            raw_text = extractor.extract(input_path)
            print("Text extracted successfully.")

            structurer = TextStructurer(
                engine=args.engine,
                project_id=args.project_id,
                location=args.location,
                local_model=args.model
            )
            structured_text = structurer.structure_text(raw_text)
            
            output_filename_base = os.path.splitext(os.path.basename(input_path))[0]
            structured_text_output_path = os.path.join(settings.OUTPUT_DIR, output_filename_base + "_structured.json")
            with open(structured_text_output_path, 'w', encoding='utf-8') as f:
                json.dump(structured_text, f, indent=2)
            
            print(f"\nStructured text saved to {structured_text_output_path}")

        print("\n--- Structured Text Sample ---")
        print(json.dumps(structured_text[:5], indent=2))

        if args.add_emotions:
            emotion_annotator = EmotionAnnotator(structurer.llm_orchestrator) # Reuse LLM orchestrator
            structured_text = emotion_annotator.annotate_emotions(structured_text)

        if not args.skip_voice_casting:
            print("\nCasting voices for characters... (This may take a moment)")
            voice_caster = VoiceCaster(
                engine=args.engine,
                project_id=args.project_id,
                location=args.location,
                local_model=args.model,
                voice_quality=args.voice_quality
            )
            voice_profiles = voice_caster.cast_voices(structured_text)

            voice_profiles_output_path = os.path.join(settings.OUTPUT_DIR, output_filename_base + "_voice_profiles.json")
            with open(voice_profiles_output_path, 'w', encoding='utf-8') as f:
                json.dump(voice_profiles, f, indent=2)
            
            print(f"\nVoice profiles saved to {voice_profiles_output_path}")
            print("\n--- Suggested Voice Profiles Sample ---")
            print(json.dumps(voice_profiles, indent=2))

            if args.output_filename:
                audio_generator = AudioGenerator(project_id=args.project_id, location=args.location)
                audio_generator.generate_audiobook(structured_text, voice_profiles, args.output_filename)
        else:
            print("\nSkipping voice casting as requested.")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()