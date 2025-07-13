import time
import logging
import os
from tqdm import tqdm
import spacy

from config import settings
from .llm_orchestrator import LLMOrchestrator
from .chunking import ChunkManager
from .preprocessor import TextPreprocessor
from .simplified_validator import SimplifiedValidator
from .refiner import OutputRefiner
from .contextual_refiner import ContextualRefiner
from .deterministic_segmenter import DeterministicSegmenter
from .rule_based_attributor import RuleBasedAttributor
from .unfixable_recovery import UnfixableRecoverySystem
from .output_formatter import OutputFormatter

class TextStructurer:
    """Structures raw text into a dialogue-focused JSON format using an LLM."""

    def __init__(self, engine=settings.DEFAULT_LLM_ENGINE, project_id=None, location=None, local_model=settings.DEFAULT_LOCAL_MODEL):
        self.engine = engine
        self.local_model = local_model
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initializing TextStructurer with engine: {engine}, model: {local_model}")
        
        try:
            self.llm_orchestrator = LLMOrchestrator({
                'engine': self.engine,
                'project_id': project_id,
                'location': location,
                'local_model': local_model
            })
            self.chunk_manager = ChunkManager()
            self.preprocessor = TextPreprocessor(self._load_spacy_model())
            self.deterministic_segmenter = DeterministicSegmenter()
            self.rule_based_attributor = RuleBasedAttributor()
            self.validator = SimplifiedValidator()  # Use new simplified validator
            self.refiner = OutputRefiner(self.llm_orchestrator, self.validator)
            self.contextual_refiner = ContextualRefiner(self.llm_orchestrator)
            self.unfixable_recovery = UnfixableRecoverySystem()
            self.output_formatter = OutputFormatter()
            self.logger.info("TextStructurer initialization completed successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize TextStructurer: {e}", exc_info=True)
            raise

    def _setup_logging(self):
        """Setup comprehensive logging with separate levels for console and file."""
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.getcwd(), settings.LOG_DIR)
        os.makedirs(log_dir, exist_ok=True)
        
        # Clear any existing handlers to avoid duplicates
        logging.getLogger().handlers.clear()
        
        # Get log levels from settings
        file_log_level = getattr(logging, getattr(settings, 'FILE_LOG_LEVEL', 'DEBUG').upper(), logging.DEBUG)
        console_log_level = getattr(logging, getattr(settings, 'CONSOLE_LOG_LEVEL', 'INFO').upper(), logging.INFO)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(levelname)s - %(name)s - %(message)s'
        )
        
        # Create file handler (detailed logging)
        file_handler = logging.FileHandler(os.path.join(log_dir, 'text_structurer.log'))
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(detailed_formatter)
        
        # Create console handler (less verbose)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        console_handler.setFormatter(simple_formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Allow all levels, handlers will filter
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    def _load_spacy_model(self):
        nlp_model = None
        try:
            nlp_model = spacy.load(settings.SPACY_MODEL)
            print(f"spaCy model '{settings.SPACY_MODEL}' loaded successfully.")
        except OSError:
            print("\n---")
            print(f"Warning: spaCy model '{settings.SPACY_MODEL}' not found.")
            print("Character name detection will be less accurate.")
            print(f"For better results, run: python -m spacy download {settings.SPACY_MODEL}")
            print("---\n")
        return nlp_model

    def structure_text(self, text_content):
        """
        Uses the selected AI engine to structure the text with comprehensive error handling.
        """
        start_time = time.time()
        self.logger.info(f"Starting text structuring with {self.engine} engine ({self.local_model if self.engine == 'local' else 'gemini-1.0-pro'})")
        print(f"\nUsing {self.engine} engine ({self.local_model if self.engine == 'local' else 'gemini-1.0-pro'})...")

        try:
            # Pre-process text for structural hints
            self.logger.info("Starting text preprocessing...")
            text_metadata = self.preprocessor.analyze(text_content)
            self.logger.info(f"Preprocessing completed. Found {len(text_metadata.get('potential_character_names', set()))} potential characters")

            # Create chunks
            self.logger.info("Creating text chunks...")
            chunks = self.chunk_manager.create_chunks(text_content, text_metadata['scene_breaks'])
            self.logger.info(f"Created {len(chunks)} chunks for processing")
            
            all_structured_segments = []
            processed_data_with_chunks = []
            failed_chunks = []
            
            for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
                try:
                    self.logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
                    
                    # Phase 2: Deterministic segmentation first
                    numbered_lines = self.deterministic_segmenter.segment_text(chunk, text_metadata)
                    
                    if not numbered_lines:
                        self.logger.warning(f"Empty segmentation for chunk {i+1}, using fallback")
                        # Fallback: simple sentence splitting converted to numbered lines
                        fallback_segments = self._fallback_text_splitting(chunk)
                        numbered_lines = [{"line_id": i+1, "text": seg} for i, seg in enumerate(fallback_segments)]
                    
                    # Phase 2.5: Rule-based attribution first pass
                    attributed_lines = self.rule_based_attributor.process_lines(numbered_lines, text_metadata)
                    
                    # Separate lines that need AI processing from those already attributed
                    pending_ai_lines = self.rule_based_attributor.get_pending_lines(attributed_lines)
                    rule_attributed_lines = self.rule_based_attributor.get_attributed_lines(attributed_lines)
                    
                    self.logger.debug(f"Chunk {i+1}: {len(rule_attributed_lines)} rule-attributed, {len(pending_ai_lines)} need AI")
                    
                    # Phase 3: LLM processing for remaining lines (speaker classification only)
                    if pending_ai_lines:
                        # Extract just the text content for LLM classification
                        text_lines = [line['text'] for line in pending_ai_lines]
                        
                        # NEW APPROACH: Use speaker classification instead of text segmentation
                        speaker_classifications = self.llm_orchestrator.get_speaker_classifications(text_lines, text_metadata)
                        
                        if not speaker_classifications or len(speaker_classifications) != len(text_lines):
                            self.logger.warning(f"LLM classification failed for chunk {i+1}, using fallback")
                            speaker_classifications = ["AMBIGUOUS"] * len(text_lines)
                        
                        # Combine text lines with AI-classified speakers
                        ai_structured_data = []
                        for text_line, speaker in zip(text_lines, speaker_classifications):
                            ai_structured_data.append({"speaker": speaker, "text": text_line})
                    else:
                        ai_structured_data = []
                    
                    # Combine rule-based and AI attributions
                    rule_structured_data = [{"speaker": line['speaker'], "text": line['text']} for line in rule_attributed_lines]
                    structured_data_for_chunk = rule_structured_data + ai_structured_data
                    
                    # Add chunk index information for validation
                    for segment in structured_data_for_chunk:
                        processed_data_with_chunks.append((segment, i))
                    
                    all_structured_segments = self.chunk_manager.merge(
                        all_structured_segments, structured_data_for_chunk
                    )
                    
                except Exception as chunk_error:
                    self.logger.error(f"Failed to process chunk {i+1}: {chunk_error}", exc_info=True)
                    failed_chunks.append(i)
                    
                    # Attempt fallback processing
                    try:
                        self.logger.info(f"Attempting fallback processing for chunk {i+1}")
                        fallback_segments = self._fallback_chunk_processing(chunk, text_metadata)
                        for segment in fallback_segments:
                            processed_data_with_chunks.append((segment, i))
                        all_structured_segments.extend(fallback_segments)
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback processing also failed for chunk {i+1}: {fallback_error}")
                        continue

            if failed_chunks:
                self.logger.warning(f"Failed to process {len(failed_chunks)} chunks: {failed_chunks}")
                
        except Exception as preprocessing_error:
            self.logger.error(f"Critical error in preprocessing stage: {preprocessing_error}", exc_info=True)
            raise

        print(f"Initial processing completed. Running validation and refinement...")
        
        # Run validation and contextual refinement on the complete dataset
        try:
            validated_data, quality_report = self.validator.validate(processed_data_with_chunks, text_content, text_metadata)
            print(f"Initial quality score: {quality_report['quality_score']:.2f}% ({quality_report['error_count']} errors)")
            
            # Extract segments for contextual refinement (remove chunk indices temporarily)
            segments_only = [segment for segment, chunk_idx in validated_data]
            
            # Run contextual refinement on AMBIGUOUS speakers with segment tracking
            print("Running contextual refinement for AMBIGUOUS speakers...")
            original_segment_count = len(segments_only)
            contextually_refined_segments = self.contextual_refiner.refine_ambiguous_speakers(segments_only, text_metadata)
            refined_segment_count = len(contextually_refined_segments)
            
            # ROBUSTNESS: Track segment count changes
            if refined_segment_count != original_segment_count:
                print(f"Contextual refinement changed segment count: {original_segment_count} -> {refined_segment_count}")
                self.logger.info(f"Segment count changed during contextual refinement: {original_segment_count} -> {refined_segment_count}")
            
            # Count refinements made
            ambiguous_before = sum(1 for seg in segments_only if seg.get('speaker') == 'AMBIGUOUS')
            ambiguous_after = sum(1 for seg in contextually_refined_segments if seg.get('speaker') == 'AMBIGUOUS')
            refined_count = ambiguous_before - ambiguous_after
            
            if refined_count > 0:
                print(f"Contextual refinement resolved {refined_count} AMBIGUOUS speakers")
            
            # Run UNFIXABLE recovery system
            print("Running UNFIXABLE recovery system...")
            unfixable_before = sum(1 for seg in contextually_refined_segments if seg.get('speaker') == 'UNFIXABLE')
            
            if unfixable_before > 0:
                recovered_segments = self.unfixable_recovery.recover_unfixable_segments(contextually_refined_segments, text_metadata)
                unfixable_after = sum(1 for seg in recovered_segments if seg.get('speaker') == 'UNFIXABLE')
                recovered_count = unfixable_before - unfixable_after
                
                if recovered_count > 0:
                    print(f"UNFIXABLE recovery resolved {recovered_count} UNFIXABLE speakers")
                else:
                    print("No UNFIXABLE speakers were recovered")
                    
                contextually_refined_segments = recovered_segments
            else:
                print("No UNFIXABLE segments found for recovery")
            
            # ROBUSTNESS FIX: Re-add chunk indices with proper bounds checking
            contextually_refined_data = []
            for i, seg in enumerate(contextually_refined_segments):
                try:
                    # Use original index if available, otherwise use a safe default
                    if i < len(validated_data):
                        chunk_idx = validated_data[i][1]
                    else:
                        # If segments were merged/removed, use the last available chunk index
                        chunk_idx = validated_data[-1][1] if validated_data else 0
                    contextually_refined_data.append((seg, chunk_idx))
                except (IndexError, TypeError) as e:
                    # Defensive programming: if anything goes wrong, use safe defaults
                    self.logger.warning(f"Index mapping issue at segment {i}: {e}")
                    contextually_refined_data.append((seg, 0))  # Default chunk index
            
            # Run final validation after UNFIXABLE recovery to get updated quality score
            final_validated_data, updated_quality_report = self.validator.validate(contextually_refined_data, text_content, text_metadata)
            print(f"Quality after UNFIXABLE recovery: {updated_quality_report['quality_score']:.2f}% "
                  f"({updated_quality_report['error_analysis']['total_errors']} errors, "
                  f"{updated_quality_report['attribution_metrics']['unfixable_segments']} unfixable)")
            
            # Use updated quality report for decision making
            current_quality_score = updated_quality_report['quality_score']
            
            # Run traditional refinement if quality is still below threshold
            if current_quality_score < settings.REFINEMENT_QUALITY_THRESHOLD:
                print("Running additional traditional refinement...")
                refined_data, final_quality_report = self.refiner.refine(final_validated_data, text_content, chunks, text_metadata)
                print(f"Final quality score: {final_quality_report['quality_score']:.2f}% ({final_quality_report['error_count']} errors)")
                # Extract just the segments without chunk indices for return
                final_segments = [segment for segment, chunk_idx in refined_data]
            else:
                print(f"Quality acceptable after UNFIXABLE recovery ({current_quality_score:.1f}%).")
                # Extract just the segments without chunk indices for return
                final_segments = [segment for segment, chunk_idx in final_validated_data]
                
        except Exception as e:
            import traceback
            self.logger.error(f"Validation/refinement pipeline failed: {e}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            print(f"Warning: Validation/refinement failed: {e}")
            print("Returning unvalidated results...")
            
            # ROBUSTNESS: Ensure we have valid segments to return
            try:
                final_segments = all_structured_segments if all_structured_segments else []
                print(f"Returning {len(final_segments)} unvalidated segments")
            except Exception as fallback_error:
                self.logger.error(f"Even fallback failed: {fallback_error}")
                final_segments = []  # Last resort: empty list

        # ROBUSTNESS: Validate segments before formatting
        print("Validating segment integrity before formatting...")
        validated_final_segments = self._validate_segment_integrity(final_segments)
        
        # Apply final output formatting with robust error handling
        print("Applying output formatting and cleanup...")
        try:
            formatted_segments = self.output_formatter.format_output(validated_final_segments, preserve_metadata=True)
            
            # Log formatting results
            original_count = len(validated_final_segments)
            formatted_count = len(formatted_segments)
            if formatted_count != original_count:
                print(f"Output formatting: {original_count} -> {formatted_count} segments (removed {original_count - formatted_count} empty)")
                
        except Exception as e:
            self.logger.warning(f"Output formatting failed: {e}")
            print(f"Warning: Output formatting failed, using unformatted segments: {e}")
            formatted_segments = validated_final_segments  # Use validated segments if formatting fails

        end_time = time.time()
        print(f"Text structuring completed in {end_time - start_time:.2f} seconds.")

        return formatted_segments

    def _fallback_text_splitting(self, text):
        """Fallback method for splitting text when LLM fails."""
        self.logger.info("Using fallback text splitting method")
        import re
        
        # Split on paragraph breaks, then sentences
        paragraphs = re.split(r'\n\s*\n', text.strip())
        if not paragraphs or len(paragraphs) == 1:
            # If no paragraph breaks, split on sentences
            sentences = re.split(r'[.!?]+\s+', text.strip())
            return [s.strip() for s in sentences if s.strip()]
        
        return [p.strip() for p in paragraphs if p.strip()]

    def _fallback_chunk_processing(self, chunk, text_metadata):
        """Complete fallback processing for a chunk when all else fails."""
        self.logger.info("Using complete fallback processing")
        
        # Simple text splitting
        paragraphs = self._fallback_text_splitting(chunk)
        
        # Basic speaker attribution without LLM
        segments = []
        for paragraph in paragraphs:
            speaker = "narrator"  # Default to narrator
            text = paragraph.strip()
            
            # Very basic dialogue detection
            if '"' in text or '"' in text or '"' in text:
                speaker = "AMBIGUOUS"
            
            segments.append({"speaker": speaker, "text": text})
        
        return segments
    
    def _validate_segment_integrity(self, segments):
        """
        ROBUSTNESS: Validate segment integrity and fix common issues.
        
        Args:
            segments: List of segments to validate
            
        Returns:
            List of validated segments with fixed issues
        """
        if not segments:
            return []
        
        validated_segments = []
        fixed_count = 0
        
        for i, segment in enumerate(segments):
            try:
                # Ensure segment is a dictionary
                if not isinstance(segment, dict):
                    self.logger.warning(f"Segment {i} is not a dictionary: {type(segment)}")
                    continue
                
                # Ensure required fields exist
                if 'speaker' not in segment:
                    segment['speaker'] = 'narrator'  # Default speaker
                    fixed_count += 1
                    
                if 'text' not in segment:
                    segment['text'] = ''  # Empty text
                    fixed_count += 1
                
                # Ensure text is a string
                if not isinstance(segment.get('text'), str):
                    segment['text'] = str(segment.get('text', ''))
                    fixed_count += 1
                
                # Ensure speaker is a string
                if not isinstance(segment.get('speaker'), str):
                    segment['speaker'] = str(segment.get('speaker', 'narrator'))
                    fixed_count += 1
                
                # Skip completely empty segments
                if not segment.get('text', '').strip():
                    continue
                
                validated_segments.append(segment)
                
            except Exception as e:
                self.logger.warning(f"Error validating segment {i}: {e}")
                # Skip problematic segments
                continue
        
        if fixed_count > 0:
            self.logger.info(f"Fixed {fixed_count} segment integrity issues")
            print(f"Fixed {fixed_count} segment integrity issues")
        
        return validated_segments
