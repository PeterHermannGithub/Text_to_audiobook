import time
from tqdm import tqdm
import spacy

from config import settings
from .llm_orchestrator import LLMOrchestrator
from .chunking import ChunkManager
from .preprocessor import TextPreprocessor
from .speaker_attributor import SpeakerAttributor

class TextStructurer:
    """Structures raw text into a dialogue-focused JSON format using an LLM."""

    def __init__(self, engine=settings.DEFAULT_LLM_ENGINE, project_id=None, location=None, local_model=settings.DEFAULT_LOCAL_MODEL):
        self.engine = engine
        self.local_model = local_model
        self.llm_orchestrator = LLMOrchestrator({
            'engine': self.engine,
            'project_id': project_id,
            'location': location,
            'local_model': local_model
        })
        self.chunk_manager = ChunkManager()
        self.preprocessor = TextPreprocessor(self._load_spacy_model())
        self.speaker_attributor = SpeakerAttributor(self._load_spacy_model()) # Initialize SpeakerAttributor

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
        Uses the selected AI engine to structure the text.
        """
        start_time = time.time()
        print(f"\nUsing {self.engine} engine ({self.local_model if self.engine == 'local' else 'gemini-1.0-pro'})...")

        # Pre-process text for structural hints
        text_metadata = self.preprocessor.analyze(text_content)

        # Create chunks
        chunks = self.chunk_manager.create_chunks(text_content, text_metadata['scene_breaks'])
        
        all_structured_segments = []
        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            # LLM now only splits into paragraphs (strings)
            prompt = self.llm_orchestrator.build_prompt(chunk)
            paragraph_list = self.llm_orchestrator.get_structured_response(prompt) # This now returns list of strings
            
            # SpeakerAttributor assigns speakers to the paragraphs
            structured_data_for_chunk = self.speaker_attributor.attribute_speakers(paragraph_list, text_metadata)
            
            all_structured_segments = self.chunk_manager.merge(
                all_structured_segments, structured_data_for_chunk
            )

        end_time = time.time()
        print(f"Text structuring completed in {end_time - start_time:.2f} seconds.")

        return all_structured_segments
