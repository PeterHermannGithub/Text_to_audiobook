# Text_to_audiobook Setup Guide

## Context & Purpose
Complete installation and configuration guide for the PDF to audiobook converter, including system dependencies, TTS engine setup, and troubleshooting for various platforms.

## Prerequisites

### System Requirements
- **Python**: 3.7 or higher with pip package manager
- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Memory**: Minimum 4GB RAM (8GB recommended for large documents)
- **Storage**: 1GB free space for dependencies and temporary files
- **Audio System**: Functional audio drivers for local TTS testing

### External Dependencies
- **FFmpeg**: Required for audio file concatenation and processing
- **Audio Drivers**: Platform-specific audio system support
- **Internet Connection**: Required for cloud TTS services (Google Cloud, ElevenLabs)

### Optional Cloud Services
- **Google Cloud Account**: For Google Cloud Text-to-Speech (high quality, paid)
- **ElevenLabs Account**: For premium AI voices (limited free tier)

## Installation Steps

### 1. Python Environment Setup
```bash
# Verify Python installation
python --version  # Should be 3.7+
python -m pip --version

# Create project directory
mkdir pdf_to_audiobook
cd pdf_to_audiobook

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Download Project Files
```bash
# Option 1: Download from repository
git clone https://github.com/username/Text_to_audiobook.git
cd Text_to_audiobook

# Option 2: Manual download
# Download main.py and diagnose.py to your project directory
```

### 3. Install Python Dependencies
```bash
# Install core dependencies
pip install PyMuPDF  # PDF text extraction
pip install pyttsx3   # Local TTS engine
pip install pydub     # Audio processing (optional)

# Install cloud TTS dependencies (optional)
pip install google-cloud-texttospeech  # Google Cloud TTS
pip install requests  # For ElevenLabs API

# Create requirements.txt for future reference
pip freeze > requirements.txt
```

### 4. FFmpeg Installation

#### Windows Installation
```bash
# Option 1: Using Chocolatey
choco install ffmpeg

# Option 2: Manual installation
# 1. Download FFmpeg from https://ffmpeg.org/download.html
# 2. Extract to C:\ffmpeg
# 3. Add C:\ffmpeg\bin to system PATH
# 4. Restart command prompt

# Verify installation
ffmpeg -version
```

#### macOS Installation
```bash
# Using Homebrew (recommended)
brew install ffmpeg

# Using MacPorts
sudo port install ffmpeg

# Verify installation
ffmpeg -version
```

#### Linux Installation
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# CentOS/RHEL
sudo yum install ffmpeg
# or
sudo dnf install ffmpeg

# Verify installation
ffmpeg -version
```

## Configuration

### 1. Basic System Verification
```bash
# Run diagnostic script to verify all dependencies
python diagnose.py

# Expected output should show:
# ✅ Python version compatible
# ✅ All required packages installed
# ✅ FFmpeg available
# ✅ Audio system functional
```

### 2. TTS Engine Configuration

#### Local TTS (pyttsx3) - Default
```python
# No additional configuration required
# Test with simple command:
python -c "import pyttsx3; engine = pyttsx3.init(); engine.say('Test'); engine.runAndWait()"
```

#### Google Cloud Text-to-Speech Setup
```bash
# 1. Create Google Cloud Project
# Go to https://console.cloud.google.com/
# Create new project or select existing

# 2. Enable Text-to-Speech API
# Navigate to APIs & Services > Library
# Search for "Cloud Text-to-Speech API"
# Click Enable

# 3. Create Service Account
# Go to IAM & Admin > Service Accounts
# Click "Create Service Account"
# Download JSON credentials file

# 4. Set credentials environment variable
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
# Windows:
set GOOGLE_APPLICATION_CREDENTIALS=path\to\credentials.json
```

#### ElevenLabs Setup
```bash
# 1. Create ElevenLabs account at https://elevenlabs.io/
# 2. Navigate to your profile settings
# 3. Copy your API key
# 4. Store API key securely (do not commit to version control)

# Test ElevenLabs connection (replace YOUR_API_KEY)
python -c "import requests; print(requests.get('https://api.elevenlabs.io/v1/voices', headers={'xi-api-key': 'YOUR_API_KEY'}).status_code)"
```

### 3. Character Profile Configuration
```bash
# Generate character profile template
python main.py sample.pdf --engine pyttsx3

# This creates character_profiles.json:
{
    "narrator": {
        "gender": "unknown",
        "age": "adult",
        "personality": [],
        "voice_preference": ""
    },
    "Character Name": {
        "gender": "male",
        "age": "young",
        "personality": ["energetic", "cheerful"],
        "voice_preference": "specific_voice_id"
    }
}

# Edit profiles to customize character voices
# Save and run conversion with profiles:
python main.py sample.pdf --character-profiles character_profiles.json
```

## Verification

### 1. Basic Functionality Test
```bash
# Create a simple test PDF with text content
# Or download a sample PDF with dialogue

# Test basic conversion
python main.py test_document.pdf --output test_audiobook.mp3

# Expected behavior:
# - Text extraction from PDF
# - Character detection and voice assignment
# - Audio segment generation
# - Final audiobook creation
```

### 2. TTS Engine Testing
```bash
# Test local TTS
python main.py test.pdf --engine pyttsx3 --output local_test.mp3

# Test Google Cloud TTS (requires credentials)
python main.py test.pdf --engine google --credentials path/to/credentials.json --output google_test.mp3

# Test ElevenLabs (requires API key)
python main.py test.pdf --engine elevenlabs --credentials YOUR_API_KEY --output elevenlabs_test.mp3
```

### 3. Audio Quality Verification
```bash
# Check generated audio files
# Verify they play correctly on your system
# Test character voice differentiation
# Check audio quality and clarity

# Verify final audiobook file
# Should be a single MP3 file combining all segments
# Check for smooth transitions between segments
```

### 4. Performance Testing
```bash
# Test with different document sizes
# Small document (1-5 pages): Should complete in 1-5 minutes
# Medium document (10-50 pages): May take 10-30 minutes
# Large document (100+ pages): Could take 1-3 hours

# Monitor system resources during conversion
# Check for memory leaks or excessive CPU usage
```

## Troubleshooting

### Common Installation Issues

#### Python/Pip Issues
```bash
# Permission errors on Windows
python -m pip install --user package_name

# SSL certificate errors
pip install --trusted-host pypi.org --trusted-host pypi.python.org package_name

# Virtual environment activation issues
# Windows: Ensure execution policy allows scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### FFmpeg Installation Issues
```bash
# FFmpeg not found in PATH
# Windows: Add FFmpeg bin directory to system PATH
# macOS/Linux: Use package manager or compile from source

# Permission denied errors
# Ensure user has write permissions to output directory
chmod 755 /path/to/output/directory
```

#### Audio System Issues
```bash
# No audio output device
# Linux: Install ALSA or PulseAudio
sudo apt install alsa-utils pulseaudio

# macOS: Check System Preferences > Sound
# Windows: Check Device Manager > Sound devices
```

### TTS Engine Issues

#### pyttsx3 Problems
```bash
# No voices available
# Windows: Reinstall speech platform
# macOS: Check Accessibility settings
# Linux: Install espeak or festival
sudo apt install espeak espeak-data

# Voice not changing
# Verify voice ID in available voices list
python -c "import pyttsx3; engine = pyttsx3.init(); voices = engine.getProperty('voices'); [print(f'{i}: {v.name}') for i, v in enumerate(voices)]"
```

#### Google Cloud TTS Issues
```bash
# Authentication errors
# Verify credentials file path and format
# Check Google Cloud project billing is enabled
# Verify Text-to-Speech API is enabled

# Quota exceeded errors
# Check API usage in Google Cloud Console
# Consider upgrading to higher quota tier
```

#### ElevenLabs Issues
```bash
# API key errors
# Verify API key is correct and active
# Check account status and usage limits

# Rate limiting
# Add delays between requests for large documents
# Consider upgrading to higher tier for increased limits
```

### Processing Issues

#### PDF Text Extraction Problems
```bash
# Empty text extraction
# Verify PDF contains text (not scanned images)
# Try with different PDF files
# Check PDF permissions and encryption

# Garbled or incorrect text
# PDF may have encoding issues
# Try saving PDF with different settings
# Use OCR software for image-based PDFs
```

#### Character Detection Issues
```bash
# No characters detected
# Check dialogue format in PDF
# Verify dialogue patterns match regex patterns in code
# Manually edit character_profiles.json if needed

# Incorrect character assignment
# Review and edit character profiles
# Adjust dialogue detection patterns
# Use manual character mapping
```

#### Audio Generation Problems
```bash
# Audio files not created
# Check write permissions to output directory
# Verify TTS engine is working correctly
# Check for special characters in text

# Poor audio quality
# Try different TTS engine
# Adjust voice parameters (rate, volume)
# Use higher quality cloud services
```

### Performance Optimization

#### Memory Usage
```bash
# For large documents, process in chunks
# Clear intermediate files regularly
# Use streaming processing for very large PDFs
# Monitor memory usage during conversion
```

#### Processing Speed
```bash
# Use faster TTS engines (Google Cloud > ElevenLabs > pyttsx3)
# Process shorter segments for faster feedback
# Use SSD storage for temporary files
# Close other applications during processing
```

## Development and Customization

### Adding New TTS Engines
```python
# Create new engine class inheriting from TTSEngine
class NewTTSEngine(TTSEngine):
    def get_available_voices(self):
        # Implement voice discovery
        pass
    
    def generate_speech(self, text, voice, output_file):
        # Implement text-to-speech conversion
        pass
```

### Customizing Dialogue Detection
```python
# Modify patterns in DialogueProcessor class
# Add new dialogue formats or adjust existing ones
# Update character extraction logic
# Add support for different languages
```

### Audio Post-Processing
```bash
# Add audio effects using FFmpeg
ffmpeg -i input.mp3 -af "highpass=f=200,lowpass=f=3000" output.mp3

# Normalize audio levels
ffmpeg -i input.mp3 -af loudnorm output.mp3

# Add background music or effects
ffmpeg -i voice.mp3 -i music.mp3 -filter_complex "[0:a][1:a]amix=inputs=2:duration=first" output.mp3
```

## Deployment and Distribution

### Standalone Application
```bash
# Use PyInstaller to create executable
pip install pyinstaller
pyinstaller --onefile main.py

# Include required data files
pyinstaller --onefile --add-data "character_profiles.json;." main.py
```

### Docker Containerization
```dockerfile
# Example Dockerfile
FROM python:3.9-slim

RUN apt-get update && apt-get install -y ffmpeg

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py diagnose.py ./
CMD ["python", "main.py"]
```

## Maintenance and Updates

### Regular Maintenance
1. **Update Dependencies**: Keep Python packages current
2. **Test TTS Engines**: Verify cloud service APIs remain functional
3. **Audio Quality**: Test output quality with different document types
4. **Performance**: Monitor processing times and optimize as needed

### Backup and Recovery
```bash
# Backup important configurations
cp character_profiles.json character_profiles_backup.json

# Version control for customizations
git init
git add main.py diagnose.py character_profiles.json
git commit -m "Initial audiobook converter setup"
```

## Metadata
- **Last Updated**: 2025-04-21
- **Complexity**: Intermediate-Advanced
- **Dependencies**: Python 3.7+, FFmpeg, TTS engines
- **Platforms**: Windows, macOS, Linux