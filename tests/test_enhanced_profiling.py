#!/usr/bin/env python3
"""
Test enhanced character profiling with pronouns, aliases, and titles.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.text_processing.preprocessor import TextPreprocessor, CharacterProfile
import spacy

def test_character_profile_class():
    """Test the CharacterProfile class functionality."""
    print("=== Testing CharacterProfile Class ===")
    
    profile = CharacterProfile("John Doe")
    
    # Test adding pronouns
    profile.add_pronoun("he", 0.8)
    profile.add_pronoun("his", 0.9)
    
    # Test adding aliases
    profile.add_alias("Johnny", 0.7)
    profile.add_alias("The Hero", 0.6)
    
    # Test adding titles
    profile.add_title("Mr.", 0.9)
    profile.add_title("Captain", 0.8)
    
    # Test dictionary conversion
    profile_dict = profile.to_dict()
    
    print(f"Profile for {profile.name}:")
    print(f"  Pronouns: {profile_dict['pronouns']}")
    print(f"  Aliases: {profile_dict['aliases']}")  
    print(f"  Titles: {profile_dict['titles']}")
    print(f"  Confidence: {profile_dict['confidence']:.2f}")
    print()

def test_enhanced_preprocessing():
    """Test enhanced character profiling with complex text."""
    print("=== Testing Enhanced Character Profiling ===")
    
    test_text = '''Mr. John Smith walked into the room. He was tired after a long day.
    
"Hello there," John said with a smile. "How are you today?"
Mary looked up from her book. She closed it carefully. "I'm doing well, thank you for asking."

Kim Dokja stared at his smartphone screen. The notification was clear.
Kim Dokja: "This can't be real."

Dr. Sarah Johnson, also known as "The Scientist," entered the laboratory.
She had been working on this project for months. Her research was groundbreaking.

CAPTAIN: All hands on deck!
SAILOR: Aye aye, sir!'''

    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
        print("spaCy model loaded successfully.")
    except OSError:
        print("spaCy model not found, using fallback.")
        nlp = None
    
    # Test preprocessing
    preprocessor = TextPreprocessor(nlp)
    metadata = preprocessor.analyze(test_text)
    
    print(f"Found {len(metadata['character_profiles'])} character profiles:")
    print()
    
    for i, profile_dict in enumerate(metadata['character_profiles'], 1):
        print(f"Profile {i}: {profile_dict['name']}")
        print(f"  Pronouns: {profile_dict['pronouns']}")
        print(f"  Aliases: {profile_dict['aliases']}")
        print(f"  Titles: {profile_dict['titles']}")
        print(f"  Confidence: {profile_dict['confidence']:.2f}")
        print()
    
    # Test backward compatibility
    print("Backward compatibility - character names:")
    print(f"  {sorted(metadata['potential_character_names'])}")
    print()
    
    # Test other metadata
    print(f"Dialogue markers: {metadata['dialogue_markers']}")
    print(f"Script-like format: {metadata['is_script_like']}")
    print(f"Scene breaks: {len(metadata['scene_breaks'])}")

def test_pronoun_detection():
    """Test pronoun detection accuracy."""
    print("=== Testing Pronoun Detection ===")
    
    test_cases = [
        ("John walked in. He was smiling.", ["he"]),
        ("Mary opened the door. She was surprised.", ["she"]),
        ("The team arrived. They were ready.", ["they"]),
        ("Dr. Smith examined the patient. His diagnosis was accurate.", ["his"]),
    ]
    
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("spaCy model not found, skipping pronoun tests.")
        return
    
    preprocessor = TextPreprocessor(nlp)
    
    for text, expected_pronouns in test_cases:
        metadata = preprocessor.analyze(text)
        profiles = metadata['character_profiles']
        
        print(f"Text: {repr(text)}")
        if profiles:
            profile = profiles[0]
            actual_pronouns = profile['pronouns']
            print(f"  Expected: {expected_pronouns}")
            print(f"  Detected: {actual_pronouns}")
            print(f"  Match: {set(expected_pronouns).issubset(set(actual_pronouns))}")
        else:
            print(f"  No profiles detected")
        print()

def test_alias_detection():
    """Test alias detection patterns."""
    print("=== Testing Alias Detection ===")
    
    test_cases = [
        'Kim Dokja, also known as The Fool, entered the room.',
        'The Hero, also called John Smith, stood up.',
        '"The Scientist," Dr. Johnson said quietly.',
        'Captain Marvel thought, "I am the strongest."'
    ]
    
    try:
        nlp = spacy.load("en_core_web_sm") 
    except OSError:
        print("spaCy model not found, using basic detection.")
        nlp = None
    
    preprocessor = TextPreprocessor(nlp)
    
    for text in test_cases:
        metadata = preprocessor.analyze(text)
        profiles = metadata['character_profiles']
        
        print(f"Text: {repr(text)}")
        for profile in profiles:
            if profile['aliases']:
                print(f"  {profile['name']} aliases: {profile['aliases']}")
        print()

if __name__ == "__main__":
    print("Testing Enhanced Character Profiling...\n")
    
    test_character_profile_class()
    test_enhanced_preprocessing()
    test_pronoun_detection()
    test_alias_detection()
    
    print("Enhanced character profiling tests completed!")