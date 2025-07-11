import argparse
import os
import json
from src.text_extractor import TextExtractor
from src.text_structurer import TextStructurer
# from src.voice_caster import VoiceCaster

def main():
    parser = argparse.ArgumentParser(description="Convert a document to an audiobook.")
    parser.add_argument("input_file", nargs='?', help="Path to the input file (.txt, .md, .pdf, .docx, .epub, .mobi). Required if --structured-input-file is not used.")
    parser.add_argument("--structured-input-file", help="Path to a pre-structured JSON file. If provided, skips text extraction and structuring.")
    parser.add_argument("--skip-voice-casting", action="store_true", help="If set, skips the voice casting phase.")
    parser.add_argument("--engine", default="local", choices=["local", "gcp"], help="AI engine to use for text structuring and character description (LLM). Default is local.")
    parser.add_argument("--model", default="mistral", choices=["mistral", "llama3"], help="Local model to use if --engine is 'local'.")
    parser.add_argument("--project_id", help="Google Cloud project ID. Required if --engine is 'gcp' or if --skip-voice-casting is not set.")
    parser.add_argument("--location", default="us-central1", help="Google Cloud location. Required if --engine is 'gcp' or if --skip-voice-casting is not set.")
    args = parser.parse_args()

    if not args.input_file and not args.structured_input_file:
        parser.error("Either input_file or --structured-input-file must be provided.")

    # Conditional requirement for project_id and location
    if args.engine == 'gcp' or not args.skip_voice_casting:
        if not args.project_id:
            parser.error("Google Cloud project ID is required when using --engine gcp or when not skipping voice casting.")
        # Location has a default, so only check if engine is gcp and it's explicitly None (unlikely with default)

    structured_text = None
    output_filename_base = None

    try:
        if args.structured_input_file:
            # Start from pre-structured JSON
            structured_input_path = os.path.abspath(args.structured_input_file)
            if not os.path.exists(structured_input_path):
                print(f"Error: Structured input file not found at {structured_input_path}")
                return
            print(f"Loading structured text from {structured_input_path}...")
            with open(structured_input_path, 'r', encoding='utf-8') as f:
                structured_text = json.load(f)
            print("Structured text loaded successfully.")
            # Determine output_filename_base from structured_input_file for consistency
            output_filename_base = os.path.splitext(os.path.basename(structured_input_path))[0].replace("_structured", "")
        else:
            # Phase 1: Text Extraction (Always Local)
            input_path = os.path.abspath(args.input_file)
            if not os.path.exists(input_path):
                print(f"Error: Input file not found at {input_path}")
                return
            print(f"Extracting text from {input_path}...")
            extractor = TextExtractor()
            raw_text = extractor.extract(input_path)
            print("Text extracted successfully.")

            # Phase 2: Text Structuring (Controlled by --engine)
            structurer = TextStructurer(
                engine=args.engine,
                project_id=args.project_id, # Will be None if not required/provided
                location=args.location,     # Will be default or None if not required/provided
                local_model=args.model
            )
            structured_text = structurer.structure_text(raw_text)
            print(f"DEBUG: structured_text after structure_text call: type={type(structured_text)}, content_sample={structured_text[:5] if isinstance(structured_text, list) else structured_text}")
            
            output_filename_base = os.path.splitext(os.path.basename(input_path))[0]
            structured_text_output_path = os.path.join("output", output_filename_base + "_structured.json")
            with open(structured_text_output_path, 'w', encoding='utf-8') as f:
                json.dump(structured_text, f, indent=2)
            
            print(f"\nStructured text saved to {structured_text_output_path}")

        # Print a sample of the structured data (regardless of source)
        print("\n--- Structured Text Sample ---")
        print(json.dumps(structured_text[:5], indent=2))

        if not args.skip_voice_casting:
            # Phase 3: Voice Casting (LLM part controlled by --engine, TTS part always GCP)
            print("\nCasting voices for characters... (This may take a moment)")
            voice_caster = VoiceCaster(
                engine=args.engine, # This controls the LLM for character description
                project_id=args.project_id, # Required for GCP TTS client
                location=args.location,     # Required for GCP TTS client
                local_model=args.model
            )
            voice_profiles = voice_caster.cast_voices(structured_text)

            voice_profiles_output_path = os.path.join("config", output_filename_base + "_voice_profiles.json")
            with open(voice_profiles_output_path, 'w', encoding='utf-8') as f:
                json.dump(voice_profiles, f, indent=2)
            
            print(f"\nVoice profiles saved to {voice_profiles_output_path}")
            print("\n--- Suggested Voice Profiles Sample ---")
            print(json.dumps(voice_profiles, indent=2))
        else:
            print("\nSkipping voice casting as requested.")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()