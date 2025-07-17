#!/usr/bin/env python
"""Memory profiling for text structuring components."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from text_structurer import TextStructurer
from config import settings

@profile
def test_text_structuring_memory():
    """Profile memory usage during text structuring."""
    # Create test text content
    test_content = '''
Chapter 1

"Hello there," said Alice to Bob.

Bob looked up from his book. "Oh, hello Alice. How are you today?"

"I'm doing well, thank you," Alice replied with a smile.

The narrator observed their conversation with interest.

"Would you like to join me for lunch?" Bob asked.

"That sounds lovely," Alice responded.
''' * 100  # Multiply to create substantial content

    # Initialize text structurer (mock mode to avoid LLM calls)
    structurer = TextStructurer(engine='local', local_model='mock')
    
    # Process the text
    try:
        result = structurer.structure_text(test_content)
        return len(result) if result else 0
    except Exception as e:
        print(f"Note: Text structuring failed (expected in CI): {e}")
        return 0

@profile
def test_large_content_structuring():
    """Profile memory usage with larger content."""
    # Create substantial test content
    dialogue_content = '''
"This is a longer conversation," Character A said thoughtfully.

Character B nodded in agreement. "Yes, we need to process larger amounts of text."

The narrator added context between the spoken words.

"Memory profiling helps us understand resource usage," Character A continued.

"Absolutely," Character B replied. "This is important for performance optimization."
''' * 500  # Create very large content

    # Initialize text structurer
    structurer = TextStructurer(engine='local', local_model='mock')
    
    try:
        result = structurer.structure_text(dialogue_content)
        return len(result) if result else 0
    except Exception as e:
        print(f"Note: Large content structuring failed (expected in CI): {e}")
        return 0

if __name__ == "__main__":
    print("Running text structuring memory profiling...")
    result1 = test_text_structuring_memory()
    print(f"Text structuring result: {result1} segments")
    
    print("Running large content memory profiling...")
    result2 = test_large_content_structuring()
    print(f"Large content result: {result2} segments")
    
    print("Memory profiling completed!")