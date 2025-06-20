# Save this as diagnose.py
import sys
import os
import subprocess

def print_section(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

print_section("Python Information")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")

print_section("Installed Packages")
try:
    result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                           capture_output=True, text=True)
    print(result.stdout)
except Exception as e:
    print(f"Error listing packages: {e}")

print_section("FFmpeg Check")
try:
    result = subprocess.run(['ffmpeg', '-version'], 
                           capture_output=True, text=True)
    print("FFmpeg is installed:")
    print(result.stdout[:200] + "...")  # First 200 chars
except Exception as e:
    print(f"FFmpeg not found or error: {e}")

print_section("Audio Module Check")
modules_to_check = [
    'pyaudio', 'pyaudioop', 'wave', 'pyttsx3', 'pydub.audio_segment'
]

for module in modules_to_check:
    try:
        __import__(module)
        print(f"✅ {module} is available")
    except ImportError:
        print(f"❌ {module} is NOT available")

print_section("Simple Audio Test")
try:
    import pyttsx3
    engine = pyttsx3.init()
    print("pyttsx3 initialized successfully")
    
    voices = engine.getProperty('voices')
    print(f"Found {len(voices)} voices")
    
    # Test voice properties
    if voices:
        print(f"First voice ID: {voices[0].id}")
        print(f"First voice name: {voices[0].name}")
    
    # Test simple speech
    engine.say("Testing audio system")
    engine.runAndWait()
    print("Audio test completed")
except Exception as e:
    print(f"Error in audio test: {e}")