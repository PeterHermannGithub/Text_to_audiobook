#!/usr/bin/env python
"""Memory profiling for text extraction components."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from text_processing.text_extractor import TextExtractor
import tempfile
import fitz  # PyMuPDF

@profile
def test_pdf_memory_usage():
    """Profile memory usage during PDF text extraction."""
    extractor = TextExtractor()
    
    # Create a test PDF in memory
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        # Create a simple PDF for testing
        doc = fitz.open()
        page = doc.new_page()
        text = "This is a test PDF content. " * 1000  # Create substantial content
        page.insert_text((72, 72), text)
        doc.save(tmp.name)
        doc.close()
        
        # Extract text and measure memory
        extracted_text = extractor.extract(tmp.name)
        
        # Cleanup
        os.unlink(tmp.name)
        
        return len(extracted_text)

@profile
def test_large_text_memory_usage():
    """Profile memory usage during large text processing."""
    extractor = TextExtractor()
    
    # Create a large text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
        large_text = "This is test content for memory profiling. " * 10000
        tmp.write(large_text)
        tmp.flush()
        
        # Extract text and measure memory
        extracted_text = extractor.extract(tmp.name)
        
        # Cleanup
        os.unlink(tmp.name)
        
        return len(extracted_text)

if __name__ == "__main__":
    print("Running PDF memory profiling...")
    result1 = test_pdf_memory_usage()
    print(f"PDF extraction result: {result1} characters")
    
    print("Running large text memory profiling...")
    result2 = test_large_text_memory_usage()
    print(f"Text extraction result: {result2} characters")
    
    print("Memory profiling completed!")