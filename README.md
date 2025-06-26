# Text_to_audiobook - PDF to Audiobook Converter

A sophisticated Python application that transforms PDF documents into engaging audiobooks with AI-powered voice synthesis, character voice differentiation, and intelligent dialogue processing.

## 🚀 **Quick Start**

### 1. Prerequisites
- Python 3.7+ with pip
- FFmpeg for audio processing
- 4GB+ RAM (8GB recommended for large documents)

### 2. Installation
```bash
# Clone or download project files
cd Text_to_audiobook

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Quick Test
```bash
# Verify system setup
python diagnose.py

# Convert a PDF to audiobook
python main.py your_book.pdf
```

## ✨ **Key Features**

- **🎭 Multiple TTS Engines**: pyttsx3 (local), Google Cloud TTS (high quality), ElevenLabs (premium AI voices)
- **🗣️ Character Voice Mapping**: Intelligent assignment of distinct voices to different characters
- **📖 Intelligent Dialogue Detection**: Recognizes speech patterns and separates dialogue from narrative
- **🎵 Audio Production Pipeline**: Professional audio concatenation with FFmpeg
- **⚙️ Pluggable Architecture**: Easy to extend with new TTS providers
- **🔧 Character Profiles**: JSON-based voice customization for specific characters

## 📁 **Project Structure**

```
├── main.py                    # Complete application with TTS engines
├── diagnose.py               # System diagnostics and dependency verification
├── requirements.txt          # Python package dependencies
├── character_profiles.json   # Generated character voice configurations
├── venv/                     # Virtual environment (created during setup)
└── output/                   # Generated audio files and segments
```

## 🛠️ **Usage Guide**

### Basic Conversion
```bash
# Simple PDF to audiobook conversion
python main.py book.pdf

# Specify output file
python main.py book.pdf --output my_audiobook.mp3
```

### TTS Engine Selection
```bash
# Local TTS (free, offline)
python main.py book.pdf --engine pyttsx3

# Google Cloud TTS (high quality, requires credentials)
python main.py book.pdf --engine google --credentials path/to/credentials.json

# ElevenLabs AI voices (premium quality, requires API key)
python main.py book.pdf --engine elevenlabs --credentials YOUR_API_KEY
```

### Character Voice Customization
```bash
# Generate character profile template
python main.py book.pdf --engine pyttsx3
# Edit generated character_profiles.json
# Run with custom profiles
python main.py book.pdf --character-profiles character_profiles.json
```

## ⚙️ **Configuration**

### Core Dependencies
```bash
# Required packages
pip install PyMuPDF      # PDF text extraction
pip install pyttsx3      # Local TTS engine
pip install pydub        # Audio processing

# Optional cloud TTS
pip install google-cloud-texttospeech  # Google Cloud TTS
pip install requests     # For ElevenLabs API
```

### FFmpeg Installation
```bash
# Windows (using Chocolatey)
choco install ffmpeg

# macOS (using Homebrew)
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Verify installation
ffmpeg -version
```

### TTS Engine Setup

#### Google Cloud Text-to-Speech
1. Create Google Cloud project at [console.cloud.google.com](https://console.cloud.google.com/)
2. Enable Text-to-Speech API
3. Create service account and download JSON credentials
4. Set environment variable: `export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"`

#### ElevenLabs Setup
1. Create account at [elevenlabs.io](https://elevenlabs.io/)
2. Get API key from profile settings
3. Use API key with `--credentials YOUR_API_KEY`

## 🎯 **How It Works**

### 1. PDF Analysis
- Extracts text from PDF using PyMuPDF
- Preserves document structure and formatting
- Handles various PDF layouts and encodings

### 2. Dialogue Detection
- Identifies different dialogue formats (quotes, dashes, brackets)
- Extracts character names from dialogue tags
- Separates narrative text from character speech

### 3. Voice Mapping
- Generates character profile configurations
- Assigns unique voices to ensure character distinction
- Supports custom voice preferences via JSON profiles

### 4. Audio Generation
- Converts text segments using selected TTS engine
- Maintains voice consistency throughout the audiobook
- Generates individual audio files for each segment

### 5. Audio Assembly
- Concatenates segments using FFmpeg
- Creates seamless transitions between parts
- Outputs single audiobook file for distribution

## 🔧 **System Diagnostics**

```bash
# Check all dependencies and system compatibility
python diagnose.py

# Expected output:
# ✅ Python version compatible
# ✅ All required packages installed
# ✅ FFmpeg available
# ✅ Audio system functional
# ✅ TTS engines accessible
```

## 🚨 **Troubleshooting**

### Common Issues
```bash
# FFmpeg not found
export PATH="$PATH:/path/to/ffmpeg/bin"

# Permission errors (Windows)
python -m pip install --user package_name

# No audio output
# Linux: sudo apt install alsa-utils pulseaudio
# macOS: Check System Preferences > Sound
# Windows: Check Device Manager > Sound devices

# PDF extraction fails
# Ensure PDF contains text (not scanned images)
# Try with different PDF files
```

### TTS Engine Issues
- **pyttsx3**: Verify system voices are installed
- **Google Cloud**: Check credentials and API billing
- **ElevenLabs**: Verify API key and account limits

## 🎨 **Character Voice Profiles**

### Profile Structure
```json
{
    "character_name": {
        "gender": "male/female/unknown",
        "age": "child/young/adult/elderly",
        "personality": ["calm", "energetic", "serious"],
        "voice_preference": "specific_voice_id"
    }
}
```

### Voice Assignment
- **Automatic**: Based on character gender and available voices
- **Manual**: Edit character_profiles.json for specific assignments
- **Special Roles**: Dedicated voices for narrator and system messages

## 📈 **Performance Considerations**

- **Processing Speed**: Google Cloud fastest, ElevenLabs highest quality but slower
- **Document Size**: Processing time scales with length (1-5 pages: 1-5 minutes)
- **Memory Usage**: Holds entire document in memory during processing
- **Storage**: Temporary files for segments plus final audiobook

## 🔒 **Limitations**

- **PDF Types**: Text-based PDFs only (cannot process scanned images)
- **Complex Layouts**: May struggle with multi-column formatting
- **Language Support**: Limited to TTS engine capabilities
- **Cloud Dependencies**: Google Cloud and ElevenLabs require internet connection

## 📞 **Support & Documentation**

For complete technical documentation, implementation details, and troubleshooting guides, see the **CLAUDE.md** file in this repository.

---

**Note**: This README.md and CLAUDE.md are the only two documentation files for this project. All development information is consolidated into these two sources.