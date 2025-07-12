# Text_to_audiobook - Advanced Audiobook Generation

A sophisticated Python application that transforms various document formats into engaging audiobooks with AI-powered voice synthesis, character voice differentiation, and intelligent dialogue processing.

## 🚀 **Quick Start**

### 1. Prerequisites
- Python 3.7+ with pip
- FFmpeg for audio processing
- Ollama installed and running (for local LLM processing)
- Google Cloud Project with Text-to-Speech API enabled (for voice casting and audio generation)

### 2. Installation
```bash
# Clone or download project files
cd Text_to_audiobook

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Basic Usage
```bash
# Run with default local model (defined in config/settings.py) for text structuring & character description
# Requires Google Cloud Project ID for voice casting
python app.py input/your_book.pdf --project_id "your-gcp-project-id"

# Run with a different local model (e.g., llama3, if pulled) for text structuring & character description
# Requires Google Cloud Project ID for voice casting
python app.py input/your_book.docx --engine local --model llama3 --project_id "your-gcp-project-id"

# Run with Google Cloud LLM (Gemini) for text structuring & character description
# Requires Google Cloud Project ID for voice casting
python app.py input/your_book.epub --engine gcp --project_id "your-gcp-project-id"
```

## ✨ **Key Features**

- **📚 Multi-Format Text Extraction**: Supports `.txt`, `.md`, `.pdf`, `.docx`, `.epub`, `.mobi`.
- **🗣️ AI-Powered Text Structuring**: Converts raw text into structured JSON, separating narration and dialogue, using either local (Mistral or Ollama) or Google Cloud (Gemini) LLMs, with speaker attribution, validation, and refinement.
- **🎭 Character Voice Casting**: Identifies unique speakers and suggests suitable Google Cloud TTS voices.
- **🎵 Audio Production Pipeline**: (Future) Professional audio concatenation with FFmpeg.
- **⚙️ Modular Architecture**: Designed for easy extension and maintenance.

## 📁 **Project Structure**

```
├── venv/                     # Python Virtual Environment
├── config/                    # For user-editable configurations
│   ├── settings.py            # Centralized application settings
│   └── voice_profiles.json    # (Generated) Suggested voice mappings
├── input/                     # Place source documents here
├── output/                    # Generated files are stored here
│   ├── <book_name>.json       # Structured text output from Phase 2
│   └── temp/                  # (Future) For temporary audio segments
├── src/                       # Core application logic
│   ├── text_extractor.py      # PHASE 1: Handles all file reading
│   ├── text_structurer.py     # PHASE 2: Orchestrates text structuring with LLM and deterministic parsing
│   ├── llm_orchestrator.py    # Handles communication with LLM (local or GCP) and response validation
│   ├── prompt_factory.py      # Generates prompts for the LLM
│   ├── speaker_attributor.py  # Assigns speakers to text segments
│   ├── voice_caster.py        # (Placeholder) For future Phase 3 logic
│   ├── audio_generator.py     # (Placeholder) For future Phase 4 logic
│   ├── preprocessor.py        # Pre-processes text for structural hints
│   ├── chunking.py            # Manages text chunking for LLM processing
│   ├── validator.py           # Validates the structured output
│   ├── refiner.py             # Refines structured output based on validation errors
│   └── utils.py               # (Placeholder) For shared helper functions
├── app.py                     # Main application entry point & CLI handler
├── requirements.txt           # Project dependencies
├── README.md                  # This file: Guide for human developers
└── GEMINI.md                  # Comprehensive guide for AI assistants
```

## 🛣️ **Roadmap / Future Plans**

This project is under active development. Key upcoming phases and improvements include:

*   **Phase 4: Audio Generation**: Implement the core logic to convert structured text into audio using Google Cloud TTS and concatenate segments with FFmpeg.
*   **Robust Character Description**: Enhance `VoiceCaster` to infer character traits (gender, age, personality, voice tone) *solely* from text, without relying on LLM's external knowledge.
*   **Improved Text Structuring**: Refine LLM prompts and add pre/post-processing steps to ensure more consistent and accurate dialogue/narration separation for unseen texts.
*   **Error Handling & User Feedback**: Implement more comprehensive error handling and provide clearer user feedback throughout the process.
*   **Configuration Management**: Externalize API keys and other sensitive configurations.
*   **Testing**: Add unit and integration tests for all modules.

---

**Note**: For detailed technical documentation, implementation specifics, and AI assistant guidance, please refer to the `GEMINI.md` file.
**Git Ignore**: Log files, input/output data, and virtual environments are ignored by Git. Refer to `.gitignore` for details.