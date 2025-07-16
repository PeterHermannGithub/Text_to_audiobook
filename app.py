import argparse
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from src.text_processing.text_extractor import TextExtractor
from src.text_structurer import TextStructurer
from src.distributed_pipeline_orchestrator import DistributedPipelineOrchestrator, DistributedProcessingConfig
from src.output.voice_caster import VoiceCaster
from src.output.audio_generator import AudioGenerator
from src.emotion_annotator import EmotionAnnotator
from config import settings

def main():
    parser = argparse.ArgumentParser(description="Convert a document to an audiobook with distributed processing support.")
    
    # Input/Output arguments
    parser.add_argument("input_file", nargs='?', help="Path to the input file (.txt, .md, .pdf, .docx, .epub, .mobi). Required if --structured-input-file is not used.")
    parser.add_argument("--structured-input-file", help="Path to a pre-structured JSON file. If provided, skips text extraction and structuring.")
    parser.add_argument("--output-filename", help="The desired name for the final output MP3 file.")
    
    # Audio processing arguments
    parser.add_argument("--skip-voice-casting", action="store_true", help="If set, skips the voice casting phase.")
    parser.add_argument("--add-emotions", action="store_true", help="If set, adds emotional annotations to the text segments.")
    parser.add_argument("--voice-quality", default="premium", choices=["standard", "premium"], help="The quality of the GCP voices to use.")
    
    # LLM engine arguments
    parser.add_argument("--engine", default=settings.DEFAULT_LLM_ENGINE, choices=["local", "gcp"], help="AI engine to use for text structuring and character description (LLM). Default is local.")
    parser.add_argument("--model", default=settings.DEFAULT_LOCAL_MODEL, choices=["mistral", "llama3"], help="Local model to use if --engine is 'local'.")
    parser.add_argument("--project_id", help="Google Cloud project ID. Required if --engine is 'gcp' or if --skip-voice-casting is not set.")
    parser.add_argument("--location", default=settings.GCP_LOCATION, help="Google Cloud location. Required if --engine is 'gcp' or if --skip-voice-casting is not set.")
    
    # Distributed processing arguments
    parser.add_argument("--distributed", action="store_true", help="Enable distributed processing using Kafka, Spark, and LLM Pool.")
    parser.add_argument("--processing-mode", default="local", choices=["local", "distributed", "hybrid"], 
                       help="Processing mode: local (traditional), distributed (full pipeline), or hybrid (mixed). Default is local.")
    parser.add_argument("--enable-kafka", action="store_true", help="Enable Kafka event-driven processing.")
    parser.add_argument("--enable-spark", action="store_true", help="Enable Spark distributed validation.")
    parser.add_argument("--enable-caching", action="store_true", help="Enable Redis caching for intermediate results.")
    parser.add_argument("--enable-monitoring", action="store_true", help="Enable Prometheus metrics and health monitoring.")
    
    # Performance and scaling arguments
    parser.add_argument("--workers", type=int, default=3, help="Number of worker threads/processes for distributed processing.")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Size of text chunks for processing.")
    parser.add_argument("--quality-threshold", type=float, default=0.85, help="Quality threshold for text processing.")
    
    # Debugging and logging arguments
    parser.add_argument("--debug-llm", action="store_true", help="Enable detailed LLM interaction logging (prompts, responses, processing steps). Debug logs are written to logs/llm_debug.log")
    parser.add_argument("--debug-distributed", action="store_true", help="Enable detailed distributed processing logging.")
    parser.add_argument("--performance-monitoring", action="store_true", help="Enable performance monitoring and metrics collection.")
    
    args = parser.parse_args()

    # Handle debug flags - override settings if enabled
    if args.debug_llm:
        print("🔍 LLM Debug Mode Enabled - Detailed logging will be written to logs/llm_debug.log")
        print("   This will log all prompts, responses, and processing steps for debugging purposes.")
        settings.LLM_DEBUG_LOGGING = True
        # Ensure we have detailed logging for the main system too
        settings.LOG_LEVEL = "DEBUG"
        settings.CONSOLE_LOG_LEVEL = "DEBUG"
    
    if args.debug_distributed:
        print("🔍 Distributed Processing Debug Mode Enabled")
        settings.LOG_LEVEL = "DEBUG"
        settings.CONSOLE_LOG_LEVEL = "DEBUG"
    
    # Handle distributed processing mode
    if args.distributed:
        args.processing_mode = "distributed"
        args.enable_kafka = True
        args.enable_spark = True
        args.enable_caching = True
        args.enable_monitoring = True
        print("🚀 Distributed Processing Mode Enabled")
        print("   Using Kafka, Spark, LLM Pool, and Redis for scalable processing")
    
    # Create distributed processing configuration
    distributed_config = DistributedProcessingConfig(
        processing_mode=args.processing_mode,
        enable_kafka=args.enable_kafka,
        enable_spark=args.enable_spark,
        enable_caching=args.enable_caching,
        enable_monitoring=args.enable_monitoring,
        chunk_size=args.chunk_size,
        quality_threshold=args.quality_threshold,
        llm_pool_size=args.workers,
        performance_monitoring=args.performance_monitoring
    )

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

            # Initialize text processing engine based on mode
            if args.processing_mode == "local":
                print("📝 Using local text structuring...")
                structurer = TextStructurer(
                    engine=args.engine,
                    project_id=args.project_id,
                    location=args.location,
                    local_model=args.model
                )
                structured_text = structurer.structure_text(raw_text)
                
            else:
                print(f"🔧 Initializing distributed processing pipeline ({args.processing_mode} mode)...")
                
                # Create distributed orchestrator
                orchestrator = DistributedPipelineOrchestrator(
                    config=distributed_config,
                    engine=args.engine,
                    local_model=args.model
                )
                
                # Display system status
                health_status = orchestrator.get_health_status()
                print(f"📊 System Health: {health_status['status']}")
                
                if health_status['status'] == 'healthy':
                    print("✅ All distributed components are operational")
                else:
                    print("⚠️  Some components may have issues - check logs for details")
                
                # Process text using distributed pipeline
                print("🚀 Processing text through distributed pipeline...")
                result = orchestrator.process_text(raw_text)
                
                if result.success:
                    structured_text = result.processed_segments
                    print(f"✅ Distributed processing completed in {result.processing_time:.2f}s")
                    
                    # Display performance metrics
                    if result.performance_metrics:
                        print(f"📈 Performance Metrics:")
                        print(f"   - Processing Mode: {result.performance_metrics.get('processing_mode', 'unknown')}")
                        print(f"   - Total Segments: {len(structured_text)}")
                        if 'cache_stats' in result.performance_metrics:
                            cache_stats = result.performance_metrics['cache_stats']
                            print(f"   - Cache Hits: {cache_stats.get('hits', 0)}")
                            print(f"   - Cache Sets: {cache_stats.get('sets', 0)}")
                    
                    # Display quality report
                    if result.quality_report:
                        print(f"📋 Quality Report:")
                        print(f"   - Quality Score: {result.quality_report.get('quality_score', 0.0):.2f}")
                        print(f"   - Total Segments: {result.quality_report.get('total_segments', 0)}")
                        issues = result.quality_report.get('issues', [])
                        if issues:
                            print(f"   - Issues Found: {len(issues)}")
                
                else:
                    print(f"❌ Distributed processing failed: {result.error_details}")
                    print("🔄 Falling back to local processing...")
                    
                    # Fallback to local processing
                    structurer = TextStructurer(
                        engine=args.engine,
                        project_id=args.project_id,
                        location=args.location,
                        local_model=args.model
                    )
                    structured_text = structurer.structure_text(raw_text)
                
                # Get processing statistics
                processing_stats = orchestrator.get_processing_stats()
                print(f"📊 Processing Statistics:")
                print(f"   - Total Jobs: {processing_stats.get('total_jobs', 0)}")
                print(f"   - Success Rate: {processing_stats.get('successful_jobs', 0)}/{processing_stats.get('total_jobs', 0)}")
                print(f"   - Average Processing Time: {processing_stats.get('avg_processing_time', 0.0):.2f}s")
                
                # Shutdown orchestrator
                orchestrator.shutdown()
            
            output_filename_base = os.path.splitext(os.path.basename(input_path))[0]
            structured_text_output_path = os.path.join(settings.OUTPUT_DIR, output_filename_base + "_structured.json")
            with open(structured_text_output_path, 'w', encoding='utf-8') as f:
                json.dump(structured_text, f, indent=2)
            
            print(f"\nStructured text saved to {structured_text_output_path}")

        print("\n--- Structured Text Sample ---")
        print(json.dumps(structured_text[:5], indent=2))

        if args.add_emotions:
            print("🎭 Adding emotional annotations...")
            
            # Get LLM orchestrator based on processing mode
            if args.processing_mode == "local":
                # For local processing, use the structurer's LLM orchestrator
                emotion_annotator = EmotionAnnotator(structurer.llm_orchestrator)
            else:
                # For distributed processing, create a new LLM orchestrator
                from src.attribution.llm.orchestrator import LLMOrchestrator
                llm_orchestrator = LLMOrchestrator({
                    'engine': args.engine,
                    'project_id': args.project_id,
                    'location': args.location,
                    'local_model': args.model
                })
                emotion_annotator = EmotionAnnotator(llm_orchestrator)
            
            structured_text = emotion_annotator.annotate_emotions(structured_text)

        if not args.skip_voice_casting:
            print("\n🎙️  Casting voices for characters... (This may take a moment)")
            voice_caster = VoiceCaster(
                engine=args.engine,
                project_id=args.project_id,
                location=args.location,
                local_model=args.model,
                voice_quality=args.voice_quality
            )
            
            # Voice casting works with the structured text regardless of processing mode
            voice_profiles = voice_caster.cast_voices(structured_text)

            voice_profiles_output_path = os.path.join(settings.OUTPUT_DIR, output_filename_base + "_voice_profiles.json")
            with open(voice_profiles_output_path, 'w', encoding='utf-8') as f:
                json.dump(voice_profiles, f, indent=2)
            
            print(f"\n📁 Voice profiles saved to {voice_profiles_output_path}")
            print("\n--- Suggested Voice Profiles Sample ---")
            print(json.dumps(voice_profiles, indent=2))

            if args.output_filename:
                print(f"\n🎧 Generating audiobook: {args.output_filename}")
                audio_generator = AudioGenerator(project_id=args.project_id, location=args.location)
                audio_generator.generate_audiobook(structured_text, voice_profiles, args.output_filename)
        else:
            print("\n⏭️  Skipping voice casting as requested.")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()