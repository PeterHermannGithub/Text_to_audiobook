#!/usr/bin/env python3
"""
Test enhanced prompts using rich character profiles.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.prompt_factory import PromptFactory
from src.preprocessor import TextPreprocessor
import spacy

def test_enhanced_prompt_generation():
    """Test prompt generation with rich character profiles."""
    print("=== Testing Enhanced Prompt Generation ===")
    
    test_text = '''Mr. John Smith walked into the room. He was tired after a long day.
    
"Hello there," John said with a smile. "How are you today?"
Mary looked up from her book. She closed it carefully. "I'm doing well, thank you for asking."

Dr. Sarah Johnson, also known as "The Scientist," entered the laboratory.'''

    # Load spaCy and preprocess text
    try:
        nlp = spacy.load("en_core_web_sm")
        print("spaCy model loaded successfully.")
    except OSError:
        print("spaCy model not found, using basic processing.")
        nlp = None
    
    preprocessor = TextPreprocessor(nlp)
    metadata = preprocessor.analyze(test_text)
    
    print(f"Detected {len(metadata['character_profiles'])} character profiles:")
    for profile in metadata['character_profiles']:
        print(f"  - {profile['name']}: {profile['pronouns']} {profile['titles']} {profile['aliases']}")
    print()
    
    # Test classification prompt with rich profiles
    test_lines = [
        'Mr. John Smith walked into the room.',
        '"Hello there," John said with a smile.',
        'Mary looked up from her book.',
        'Dr. Sarah Johnson entered the laboratory.'
    ]
    
    prompt_factory = PromptFactory()
    prompt = prompt_factory.create_speaker_classification_prompt(test_lines, metadata)
    
    print("Generated classification prompt:")
    print("=" * 60)
    print(prompt[:1500] + "..." if len(prompt) > 1500 else prompt)
    print("=" * 60)
    print()
    
    # Check if rich character context is included
    if "KNOWN CHARACTERS:" in prompt:
        print("✓ Rich character context included in prompt")
        
        # Extract the character section
        start = prompt.find("KNOWN CHARACTERS:")
        end = prompt.find("\n\n", start + len("KNOWN CHARACTERS:"))
        if end == -1:
            end = len(prompt)
        character_section = prompt[start:end]
        
        print("Character context section:")
        print(character_section)
        print()
        
        # Check for enhanced features
        if "(male)" in character_section or "(female)" in character_section:
            print("✓ Gender hints from pronouns included")
        if "[titles:" in character_section:
            print("✓ Character titles included")
        if "[aliases:" in character_section:
            print("✓ Character aliases included")
    else:
        print("✗ Character context not found in prompt")

def test_gender_inference():
    """Test gender inference from pronouns."""
    print("=== Testing Gender Inference ===")
    
    prompt_factory = PromptFactory()
    
    test_cases = [
        (['he', 'his', 'him'], 'male'),
        (['she', 'her', 'hers'], 'female'),
        (['they', 'them', 'their'], 'neutral'),
        (['it', 'its'], None),
        (['he', 'she'], 'male'),  # Male pronouns take precedence
        ([], None)
    ]
    
    for pronouns, expected in test_cases:
        result = prompt_factory._infer_gender_from_pronouns(pronouns)
        print(f"Pronouns {pronouns} -> {result} (expected: {expected})")
        assert result == expected, f"Expected {expected}, got {result}"
    
    print("✓ All gender inference tests passed")
    print()

def test_backward_compatibility():
    """Test that the system still works with old-style character names."""
    print("=== Testing Backward Compatibility ===")
    
    # Test with old-style metadata (no character_profiles)
    old_metadata = {
        'potential_character_names': {'John', 'Mary', 'Sarah'},
        'dialogue_markers': {'"'},
        'is_script_like': False
    }
    
    test_lines = ['John said hello.', 'Mary replied.', 'Sarah nodded.']
    
    prompt_factory = PromptFactory()
    prompt = prompt_factory.create_speaker_classification_prompt(test_lines, old_metadata)
    
    print("Generated prompt with old-style metadata:")
    if "KNOWN CHARACTERS: John, Mary, Sarah" in prompt:
        print("✓ Backward compatibility maintained")
    else:
        print("✗ Backward compatibility broken")
        print("Character section:", prompt[prompt.find("KNOWN CHARACTERS"):prompt.find("KNOWN CHARACTERS")+100])
    print()

if __name__ == "__main__":
    print("Testing Enhanced Prompt Generation...\n")
    
    test_enhanced_prompt_generation()
    test_gender_inference()
    test_backward_compatibility()
    
    print("Enhanced prompt generation tests completed!")