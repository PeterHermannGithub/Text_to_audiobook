import json
import time
from tqdm import tqdm
import spacy


from .preprocessor import TextPreprocessor
from .chunking import ChunkManager
from .llm_orchestrator import LLMOrchestrator
from .validator import OutputValidator
from .refiner import OutputRefiner
from .speaker_attributor import SpeakerAttributor

# --- Constants ---
MAX_REFINEMENT_ITERATIONS = 2
REFINEMENT_QUALITY_THRESHOLD = 98.0 # Start refinement if score is below this

class TextStructurer:
    """Structures raw text into a dialogue-focused JSON format using an LLM."""

    def __init__(self, engine='local', project_id=None, location=None, local_model='mistral',
                 chunk_size=2500, overlap_size=500):
        """
        Initializes the TextStructurer.

        Args:
            engine (str): The AI engine to use ('local' or 'gcp').
            project_id (str, optional): The Google Cloud project ID. Required for 'gcp' engine.
            location (str, optional): The Google Cloud location. Required for 'gcp' engine.
            local_model (str): The local model to use ('mistral' or 'llama3').
            chunk_size (int): Target size of text chunks for LLM processing.
                              Tune this based on the LLM's context window (e.g., 4000 tokens).
            overlap_size (int): Size of overlap between consecutive chunks.
        """
        self.engine = engine
        self.local_model = local_model
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

        # Initialize helper classes
        self.preprocessor = TextPreprocessor(self._load_spacy_model())
        self.chunk_manager = ChunkManager(self.chunk_size, self.overlap_size)
        self.llm_orchestrator = LLMOrchestrator({
            'engine': self.engine,
            'project_id': project_id,
            'location': location,
            'local_model': self.local_model
        })
        self.output_validator = OutputValidator()
        self.output_refiner = OutputRefiner(self.llm_orchestrator, self.output_validator)
        self.speaker_attributor = None # Will be initialized after pre-processing

    def _load_spacy_model(self):
        nlp_model = None
        try:
            nlp_model = spacy.load("en_core_web_sm")
            print("spaCy model 'en_core_web_sm' loaded successfully.")
        except OSError:
            print("\n---")
            print("Warning: spaCy model 'en_core_web_sm' not found.")
            print("Character name detection will be less accurate.")
            print("For better results, run: python -m spacy download en_core_web_sm")
            print("---\n")
        return nlp_model

    def structure_text(self, text_content):
        """
        Uses the selected AI engine to structure the text.
        """
        start_time = time.time()
        print(f"\nUsing {self.engine} engine ({self.local_model if self.engine == 'local' else 'gemini-1.0-pro'})...")

        # --- Pre-processing ---
        print("Pre-processing text for structural hints...")
        text_metadata = self.preprocessor.analyze(text_content)
        print(f"Detected dialogue markers: {text_metadata['dialogue_markers']}")
        print(f"Detected scene breaks: {len(text_metadata['scene_breaks'])} locations")
        print(f"Potential character names: {list(text_metadata['potential_character_names'])}")
        print(f"Is script-like: {text_metadata['is_script_like']}")
        # --- End Pre-processing ---

        # Initialize SpeakerAttributor after text_metadata is available
        self.speaker_attributor = SpeakerAttributor(self._load_spacy_model(), text_metadata)

        # Initialize processed_data
        processed_data = []

        if len(text_content) <= self.chunk_size:
            # Process as a single chunk if it fits
            prompt = self.llm_orchestrator.build_prompt(text_content)
            response_text = self.llm_orchestrator.get_response(prompt)
            paragraph_list = self.llm_orchestrator.validate_and_parse(response_text, prompt)
            structured_data_from_llm = self.speaker_attributor.attribute_speakers(paragraph_list)
            # Convert list[dict] to list[tuple(dict, int)] for consistency
            processed_data = [(s, 0) for s in structured_data_from_llm]
        else:
            # --- Main Structuring Loop ---
            all_processed_segments = []

            # ChunkManager is now responsible for creating large, overlapping chunks of raw text.
            chunks = self.chunk_manager.create_chunks(text_content, text_metadata['scene_breaks'])
            
            for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
                prompt = self.llm_orchestrator.build_prompt(chunk)
                
                # The LLMOrchestrator sends this prompt to the LLM.
                response_text = self.llm_orchestrator.get_response(prompt)
                
                # The LLMOrchestrator returns a list of paragraphs (strings)
                paragraph_list = self.llm_orchestrator.validate_and_parse(response_text, prompt)
                
                # SpeakerAttributor assigns speakers to the paragraphs
                structured_data_for_chunk = self.speaker_attributor.attribute_speakers(paragraph_list)
                
                # Merging: The ChunkManager's _merge_segments logic is used to intelligently merge
                # the lists of dictionaries from each chunk.
                all_processed_segments = self.chunk_manager.merge(
                    all_processed_segments, structured_data_for_chunk
                )
            # Convert list[dict] to list[tuple(dict, int)] for consistency
            processed_data = [(s, 0) for s in all_processed_segments] # Use dummy chunk_idx 0 after merging

        end_time = time.time()
        print(f"Text structuring completed in {end_time - start_time:.2f} seconds.")

        if not processed_data:
            return []

        # --- Post-processing and Validation ---
        print("\nPost-processing and validating structured text...")
        processed_data, quality_report = self.output_validator.validate(
            processed_data, text_content, text_metadata
        )
        print(f"Validation complete. Quality Score: {quality_report['quality_score']:.2f}%")
        if quality_report['errors']:
            print("Validation Errors Found:")
            for error in quality_report['errors']:
                print(f"- {error}")
        # --- End Post-processing ---

        # --- Iterative Refinement ---
        if quality_report['quality_score'] < REFINEMENT_QUALITY_THRESHOLD:
            print(f"\nInitial quality score is below {REFINEMENT_QUALITY_THRESHOLD}%. Starting iterative refinement...")
            # Pass the raw chunks and metadata to the refinement process
            processed_data, quality_report = self.output_refiner.refine(processed_data, text_content, chunks, text_metadata)
        # --- End Refinement ---

        return [item[0] for item in processed_data] # Return only the segment dictionaries