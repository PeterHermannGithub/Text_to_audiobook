# GEMINI.md - AI Assistant Guide for Text_to_audiobook

**Project Goal:** To convert text from various document formats into a structured, voice-ready script, which can then be used to generate a multi-character audiobook.

This document provides a comprehensive technical overview for AI assistants to understand, maintain, and extend this project.

---

## 🚀 Core Workflow

The application operates in a sequential, multi-phase process:

1.  **Phase 1: Text Extraction**
    *   **Input:** A file path provided via command-line argument.
    *   **Process:** The `TextExtractor` class reads the file and extracts its raw text content.
    *   **Output:** A single string of raw text.

2.  **Phase 2: Text Structuring**
    *   **Input:** The raw text string.
    *   **Process:** The `TextStructurer` class orchestrates the following:
        *   **Chunking:** The `ChunkManager` creates large, overlapping chunks of raw text.
        *   **Prompting:** For each chunk, the `PromptFactory` creates a "markup" prompt for the LLM.
        *   **LLM Interaction:** The `LLMOrchestrator` sends the prompt to the LLM, which returns the text with special XML-like tags (`<seg>`) identifying speakers and paragraphs.
        *   **Parsing:** The `MarkupParser` parses this marked-up text into a clean list of dictionaries (JSON format).
        *   **Merging:** The `ChunkManager` intelligently merges the lists of dictionaries from each chunk.
    *   **Output:** A JSON file (`output/<book_name>.json`) containing a list of objects, where each object has a `speaker` and `text` key.

3.  **Phase 3: Voice Casting (Future)**
    *   **Input:** The structured JSON from Phase 2.
    *   **Process:** Will analyze the unique speakers and suggest appropriate TTS voices.
    *   **Output:** A `voice_profiles.json` configuration file.

4.  **Phase 4: Audio Generation (Future)**
    *   **Input:** The structured JSON and the `voice_profiles.json`.
    *   **Process:** Will use a TTS engine to generate audio for each text segment with the assigned voice.
    *   **Output:** A final audiobook file (e.g., `output.mp3`).

---

## 📁 Project Structure & Key Components

```
/Text_to_audiobook/
├── .venv/                     # Python Virtual Environment
├── config/                    # For user-editable configurations
│   └── (voice_profiles.json)  # (Future) Voice mappings
├── input/                     # Place source documents here
├── output/                    # Generated files are stored here
│   ├── <book_name>.json       # Structured text output from Phase 2
│   └── temp/                  # (Future) For temporary audio segments
├── src/                       # Core application logic
│   ├── __init__.py
│   ├── text_extractor.py      # PHASE 1: Handles all file reading
│   ├── text_structurer.py     # PHASE 2: Orchestrates text structuring with LLM and deterministic parsing
│   ├── llm_orchestrator.py    # Handles communication with LLM (local or GCP) and response validation
│   ├── prompt_factory.py      # Generates prompts for the LLM
│   ├── markup_parser.py       # Parses LLM's XML-like output into structured JSON
│   ├── voice_caster.py        # (Placeholder) For future Phase 3 logic
│   ├── audio_generator.py     # (Placeholder) For future Phase 4 logic
│   ├── preprocessor.py        # Pre-processes text for structural hints
│   ├── chunking.py            # Manages text chunking for LLM processing
│   ├── validator.py           # Validates the structured output
│   ├── refiner.py             # Refines structured output based on validation errors
│   └── utils.py               # (Placeholder) For shared helper functions
├── app.py                     # Main application entry point & CLI handler
├── requirements.txt           # Project dependencies
├── README.md                  # Guide for human developers
└── GEMINI.md                  # This file: guide for AI assistants
```

### Component Deep Dive:

*   **`app.py`**: 
    *   **Purpose:** The main entry point that orchestrates the workflow.
    *   **Logic:** Parses command-line arguments (`input_file`, `--engine`, `--model`), initializes the `TextExtractor` and `TextStructurer`, and calls their methods in sequence.

*   **`src/text_extractor.py`**:
    *   **Class:** `TextExtractor`
    *   **Purpose:** To abstract the file reading process.
    *   **Logic:** Contains a dictionary mapping file extensions (`.txt`, `.pdf`, `.epub`, etc.) to their respective reading methods (`_read_pdf`, `_read_docx`). The `extract()` method is the single public interface.
    *   **Dependencies:** `PyMuPDF`, `python-docx`, `EbookLib`, `mobi`, `BeautifulSoup4`.

*   **`src/text_structurer.py`**:
    *   **Class:** `TextStructurer`
    *   **Purpose:** Orchestrates the entire text structuring process, coordinating between `TextPreprocessor`, `ChunkManager`, `LLMOrchestrator`, `OutputValidator`, and `OutputRefiner`.

*   **`src/llm_orchestrator.py`**:
    *   **Class:** `LLMOrchestrator`
    *   **Purpose:** Manages communication with the LLM (local Ollama or Google Cloud Vertex AI) and handles the initial parsing of the LLM's raw response using `MarkupParser`.
    *   **Engines:**
        *   **`local` (Default):** Connects to an Ollama server at `http://localhost:11434`. Uses the `requests` library. Supports different local models (`mistral`, `llama3`).
        *   **`gcp`:** Connects to the Google Cloud Vertex AI API using the `google-cloud-aiplatform` library. Requires `project_id` and `location` for initialization.

*   **`src/prompt_factory.py`**:
    *   **Class:** `PromptFactory`
    *   **Purpose:** Centralized factory for generating all prompts sent to the LLM, ensuring consistency and separation of prompt engineering logic.

*   **`src/markup_parser.py`**:
    *   **Class:** `MarkupParser`
    *   **Purpose:** Deterministically parses the XML-like output from the LLM into a clean, structured JSON format using BeautifulSoup.

*   **`src/preprocessor.py`**:
    *   **Class:** `TextPreprocessor`
    *   **Purpose:** Analyzes raw text to extract structural hints like dialogue markers, scene breaks, and potential character names.

*   **`src/chunking.py`**:
    *   **Class:** `ChunkManager`
    *   **Purpose:** Manages the splitting of large texts into smaller, overlapping chunks suitable for LLM processing, and merges the structured segments back together.

*   **`src/validator.py`**:
    *   **Class:** `OutputValidator`
    *   **Purpose:** Validates the structured output against the original text to ensure fidelity and identify errors.

*   **`src/refiner.py`**:
    *   **Class:** `OutputRefiner`
    *   **Purpose:** Iteratively refines the structured output by sending problematic segments back to the LLM for correction.

---

## 🛠️ How to Run & Develop

1.  **Setup Virtual Environment:**
    ```bash
    # Navigate to project root
    cd /mnt/c/Dev/Projects_gemini/Text_to_audiobook

    # Create and activate venv if it doesn't exist
    python -m venv venv
    source venv/bin/activate

    # Install dependencies
    pip install -r requirements.txt
    ```

2.  **Setup Local AI Server (Ollama - Manual Step for User):
    ```bash
    # (In a separate terminal) Install Ollama
    curl -fsSL https://ollama.com/install.sh | sh

    # (In a separate terminal) Pull models
    ollama pull mistral
    ollama pull llama3
    ```

3.  **Execute the Application:**
    *   Place a test file in the `/input` directory (e.g., `input/my_book.epub`).
    *   Run the app from the project root directory.

    ```bash
    # Run with default local model (mistral)
    python app.py input/my_book.epub

    # Run with a different local model
    python app.py input/my_book.epub --model llama3

    # Run with Google Cloud engine (requires credentials)
    python app.py input/my_book.epub --engine gcp --project_id "your-gcp-project-id"
    ```

---

## 🚨 Common Issues & Troubleshooting

*   **Ollama Connection Error:** Ensure the Ollama server is running. Run `ollama list` in a separate terminal to verify.
*   **Malformed JSON Output:** The `_validate_and_parse` method in `text_structurer.py` is the first place to debug. The prompt in `_build_prompt` may need to be adjusted for different models.
*   **Unsupported File Type:** Add a new read method and its file extension to the `supported_formats` dictionary in `text_extractor.py`.
*   **GCP Authentication:** Ensure the user has run `gcloud auth application-default login` and that the specified `project_id` is correct.

---

## 🚫 Git Ignore Policy

To maintain a clean repository and avoid committing unnecessary or sensitive files, the following are ignored by Git:

*   **Virtual Environments:** `venv/`, `env/`, `ENV/`
*   **Python Bytecode:** `__pycache__/`, `*.py[cod]`, `*$py.class`
*   **Build Artifacts:** `*.so`, `.Python`, `build/`, `develop-eggs/`, `dist/`, `downloads/`, `eggs/`, `.eggs/`, `lib/`, `lib64/`, `parts/`, `sdist/`, `var/`, `wheels/`, `*.egg-info/`, `*.egg`
*   **Generated Audio Files:** `*.wav`, `*.mp3`, `*.m4a`
*   **OS-specific Files:** `.DS_Store`, `Thumbs.db`
*   **Log Files:** `logs/`, `*.log`
*   **Input/Output Data:** `input/`, `output/`, `*.json`

This policy ensures that only essential source code and configuration files are tracked in the repository. For a complete list of ignored patterns, refer to the `.gitignore` file.