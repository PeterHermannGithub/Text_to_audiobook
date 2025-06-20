# Text_to_audiobook - PDF to Audiobook Converter

## Context & Purpose
Text_to_audiobook is a sophisticated Python application that transforms static PDF documents into engaging audiobooks with AI-powered voice synthesis, character voice differentiation, and intelligent dialogue processing.

## Overview
A comprehensive PDF-to-audiobook conversion system that creates natural-sounding narrations with character voice differentiation and intelligent dialogue processing. The application solves the accessibility challenge of consuming written content through audio while maintaining the richness of the original text through voice modulation and character recognition.

## Architecture Overview

### Core Architecture Pattern
- **Modular Class-Based Design**: Separate classes for each major functionality
- **Multiple TTS Engine Support**: Pluggable architecture for different text-to-speech providers
- **Pipeline Processing**: Sequential text extraction, dialogue processing, voice mapping, and audio generation
- **Character Voice Mapping**: Intelligent assignment of distinct voices to different characters

### Key Components

#### Text Processing Layer
- **PDFTextExtractor** (`/main.py:20`) - Extracts and cleans text from PDF files using PyMuPDF
- **DialogueProcessor** (`/main.py:32`) - Identifies speakers, dialogue sections, and narrative text
- **CharacterVoiceMapper** (`/main.py:165`) - Maps characters to appropriate voices based on profiles

#### TTS Engine Layer
- **TTSEngine** (`/main.py:281`) - Abstract base class for text-to-speech engines
- **PyttsxEngine** (`/main.py:289`) - Local TTS using pyttsx3 (offline, free)
- **GoogleTTSEngine** (`/main.py:315`) - Google Cloud Text-to-Speech integration
- **ElevenLabsEngine** (`/main.py:351`) - ElevenLabs API integration (high quality)

#### Audio Processing Layer
- **AudiobookCreator** (`/main.py:414`) - Main orchestration class for entire conversion process
- **Audio Combination** (`/main.py:492`) - FFmpeg-based audio file concatenation

#### Diagnostics Layer
- **diagnose.py** (`/diagnose.py:1`) - System diagnostics and dependency verification

## Technology Stack

### Core Technologies
- **Language**: Python 3.7+
- **PDF Processing**: PyMuPDF (fitz) for text extraction
- **Text Processing**: Regular expressions, natural language processing
- **Audio Processing**: FFmpeg for audio concatenation
- **Serialization**: JSON for character profiles and configuration

### TTS Engine Dependencies
- **Local TTS**: pyttsx3 (offline, cross-platform)
- **Google Cloud**: google-cloud-texttospeech SDK
- **ElevenLabs**: requests library for API integration
- **Audio Processing**: pydub, AudioSegment (optional)

### System Dependencies
- **FFmpeg**: For audio file concatenation and processing
- **Audio Drivers**: Platform-specific audio system support
- **Virtual Environment**: Isolated dependency management

## Key Features

### 1. Multiple TTS Engine Support
- **pyttsx3**: Offline, free, lower quality but universal compatibility
- **Google Cloud TTS**: High-quality cloud-based synthesis with extensive voice selection
- **ElevenLabs**: Premium quality AI voices with natural emotional expression
- **Pluggable Architecture**: Easy to add new TTS providers

```python
# TTS Engine Selection
if tts_engine == "google":
    engine = GoogleTTSEngine(credentials_path=credentials)
elif tts_engine == "elevenlabs":
    engine = ElevenLabsEngine(api_key=credentials)
else:
    engine = PyttsxEngine()
```

### 2. Intelligent Dialogue Detection
- **Pattern Recognition**: Identifies different dialogue formats (dashes, quotes, brackets)
- **Speaker Identification**: Extracts character names from dialogue tags
- **Context Preservation**: Maintains narrative flow between dialogue sections
- **System Messages**: Special handling for bracketed system/action text

```python
# Dialogue Pattern Processing
dash_pattern = r'–([^:]+):(.*?)(?=–|$|\n\n)'
quote_pattern = r'"([^"]*)"'
system_pattern = r'\[(.*?)\]'
```

### 3. Character Voice Mapping
- **Profile-Based Assignment**: JSON configuration for character voice preferences
- **Gender Matching**: Automatic voice selection based on character gender
- **Voice Diversity**: Ensures different characters get distinct voices
- **Special Roles**: Dedicated voice assignment for narrator and system messages

```python
# Character Profile Structure
{
    "character_name": {
        "gender": "male/female/unknown",
        "age": "child/young/adult/elderly",
        "personality": ["calm", "energetic", "serious"],
        "voice_preference": "specific_voice_id"
    }
}
```

### 4. Advanced Text Processing
- **PDF Text Extraction**: Robust handling of various PDF formats and layouts
- **Text Cleaning**: Removes formatting artifacts and normalizes content
- **Segment Organization**: Logical division of content for voice assignment
- **Content Validation**: Ensures text quality and readability

### 5. Audio Production Pipeline
- **Segment Generation**: Individual audio files for each dialogue/narrative segment
- **Voice Consistency**: Maintains character voice assignments throughout
- **Audio Concatenation**: Seamless joining of segments into final audiobook
- **Quality Control**: Configurable speech parameters (rate, volume, pitch)

## Usage Examples

### 1. Basic Conversion
```bash
# Simple PDF to audiobook conversion
python main.py book.pdf --output my_audiobook.mp3

# Using specific TTS engine
python main.py book.pdf --engine google --credentials path/to/credentials.json
```

### 2. Advanced Character Mapping
```bash
# Generate character profile template
python main.py book.pdf --engine pyttsx3
# Edit generated character_profiles.json
# Run with custom profiles
python main.py book.pdf --character-profiles character_profiles.json
```

### 3. System Diagnostics
```bash
# Check system compatibility
python diagnose.py
# Verify all dependencies and audio system
```

## Conversion Process Workflow

### 1. PDF Analysis
- Extract raw text from PDF using PyMuPDF
- Clean and normalize text formatting
- Preserve document structure and chapter divisions
- Handle various PDF layouts and encodings

### 2. Dialogue Detection
- Scan text for dialogue patterns using regex
- Identify speakers and character names
- Separate narrative text from character speech
- Process system messages and action descriptions

### 3. Voice Mapping
- Generate or load character profile configurations
- Match characters to available TTS voices
- Assign unique voices to ensure character distinction
- Configure special voices for narrator and system roles

### 4. Audio Generation
- Convert each text segment to speech using selected TTS engine
- Apply voice-specific parameters (pitch, rate, volume)
- Generate individual audio files for each segment
- Maintain consistent audio quality and format

### 5. Audio Assembly
- Concatenate individual segments using FFmpeg
- Ensure seamless transitions between segments
- Optimize final audio file for playbook compatibility
- Generate single audiobook file for distribution

## Integration Points

### 1. Cloud TTS Services
- **Google Cloud**: Requires service account credentials and API enablement
- **ElevenLabs**: Requires API key and subscription for extended usage
- **Rate Limiting**: Intelligent handling of API rate limits and quotas

### 2. System Audio
- **Audio Drivers**: Platform-specific audio system integration
- **Output Formats**: Support for MP3, WAV, and other audio formats
- **Quality Settings**: Configurable audio quality and compression

### 3. File System
- **Input Support**: PDF files with text content (not image-based)
- **Output Directory**: Organized file structure for segments and final output
- **Temporary Files**: Efficient cleanup of intermediate audio segments

## Limitations

### 1. PDF Content Constraints
- **Text-Based PDFs Only**: Cannot process image-based or scanned PDFs
- **Complex Layouts**: May struggle with multi-column or complex formatting
- **Language Support**: Limited to languages supported by chosen TTS engine
- **Character Encoding**: May have issues with non-standard character encodings

### 2. TTS Engine Limitations
- **Voice Quality**: Quality varies significantly between engines
- **Natural Expression**: Limited emotional and contextual expression
- **Pronunciation**: May mispronounce names, technical terms, or foreign words
- **Rate Limits**: Cloud services have usage quotas and rate limiting

### 3. Audio Processing Constraints
- **FFmpeg Dependency**: Requires FFmpeg installation for audio concatenation
- **Large File Handling**: Memory usage increases with document size
- **Real-time Processing**: Cannot generate audio faster than TTS engine speed
- **Audio Quality**: Final quality limited by TTS engine capabilities

## Performance Considerations

### 1. Processing Speed
- **TTS Engine Speed**: Google Cloud fastest, ElevenLabs high quality but slower
- **Document Size**: Processing time scales with document length
- **Parallel Processing**: Sequential generation (potential for future parallelization)
- **Network Dependency**: Cloud engines require stable internet connection

### 2. Resource Usage
- **Memory**: Holds entire document text in memory during processing
- **Disk Space**: Temporary files for each segment plus final audiobook
- **CPU Usage**: Local TTS engines (pyttsx3) use more CPU resources
- **Network**: Cloud services consume bandwidth for audio generation

## Related Files
- **Source Code**: `main.py` - Complete application implementation
- **Diagnostics**: `diagnose.py` - System compatibility checker
- **Configuration**: Character profile JSON templates
- **Dependencies**: Virtual environment with required packages

## Metadata
- **Last Updated**: 2025-04-21
- **Complexity**: Advanced
- **Dependencies**: PyMuPDF, pyttsx3, FFmpeg, Google Cloud TTS (optional), ElevenLabs (optional)
- **Related Projects**: Standalone Python application for PDF processing

## External Resources
- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)
- [Google Cloud Text-to-Speech](https://cloud.google.com/text-to-speech)
- [ElevenLabs API Documentation](https://docs.elevenlabs.io/)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)