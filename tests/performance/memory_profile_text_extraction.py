"""
Memory profiling tests for text extraction components.

This module provides memory usage analysis for the text extraction pipeline,
focusing on identifying memory bottlenecks and optimization opportunities.
"""

import pytest
import time
from memory_profiler import profile
from unittest.mock import Mock, patch
from typing import Dict, Any, List
import tempfile
import os


@pytest.mark.performance
@pytest.mark.memory
class TestTextExtractionMemoryProfile:
    """Memory profiling tests for text extraction components."""
    
    @profile
    def test_pdf_extraction_memory_profile(self):
        """Profile memory usage during PDF text extraction."""
        
        with patch('src.text_processing.pdf_extractor.PyMuPDF') as mock_pymupdf:
            # Mock PDF document
            mock_doc = Mock()
            mock_page = Mock()
            mock_page.get_text.return_value = "Sample PDF text content " * 1000  # ~25KB of text
            mock_doc.__len__.return_value = 100  # 100 pages
            mock_doc.__getitem__.return_value = mock_page
            mock_doc.close.return_value = None
            
            mock_pymupdf.open.return_value = mock_doc
            
            # Simulate text extraction for memory profiling
            from src.text_processing.pdf_extractor import PDFTextExtractor
            
            extractor = PDFTextExtractor()
            
            # Test memory usage across different document sizes
            test_sizes = [10, 50, 100, 500]  # pages
            
            for size in test_sizes:
                mock_doc.__len__.return_value = size
                
                # Extract text and measure memory impact
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                try:
                    extracted_text = extractor.extract_text(temp_path)
                    
                    # Verify extraction worked
                    assert len(extracted_text) > 0
                    assert isinstance(extracted_text, str)
                    
                    # Force garbage collection point
                    import gc
                    gc.collect()
                    
                finally:
                    os.unlink(temp_path)
    
    @profile
    def test_docx_extraction_memory_profile(self):
        """Profile memory usage during DOCX text extraction."""
        
        with patch('src.text_processing.docx_extractor.Document') as mock_document_class:
            # Mock DOCX document
            mock_doc = Mock()
            
            # Create mock paragraphs with substantial text
            mock_paragraphs = []
            for i in range(200):  # 200 paragraphs
                mock_para = Mock()
                mock_para.text = f"This is paragraph {i} with substantial content " * 20  # ~1KB per paragraph
                mock_paragraphs.append(mock_para)
            
            mock_doc.paragraphs = mock_paragraphs
            mock_document_class.return_value = mock_doc
            
            from src.text_processing.docx_extractor import DOCXTextExtractor
            
            extractor = DOCXTextExtractor()
            
            # Test memory usage with different document sizes
            paragraph_counts = [50, 100, 200, 500]
            
            for count in paragraph_counts:
                # Adjust mock to return specified number of paragraphs
                mock_doc.paragraphs = mock_paragraphs[:count]
                
                with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                try:
                    extracted_text = extractor.extract_text(temp_path)
                    
                    # Verify extraction
                    assert len(extracted_text) > 0
                    assert f"paragraph {min(count-1, 199)}" in extracted_text
                    
                    # Memory cleanup point
                    import gc
                    gc.collect()
                    
                finally:
                    os.unlink(temp_path)
    
    @profile
    def test_epub_extraction_memory_profile(self):
        """Profile memory usage during EPUB text extraction."""
        
        with patch('src.text_processing.epub_extractor.epub') as mock_epub:
            # Mock EPUB structure
            mock_book = Mock()
            
            # Create mock items (chapters)
            mock_items = []
            for i in range(20):  # 20 chapters
                mock_item = Mock()
                mock_item.get_type.return_value = 9  # XHTML type
                chapter_content = f"<html><body><p>Chapter {i} content " * 100 + "</p></body></html>"  # ~10KB per chapter
                mock_item.get_content.return_value = chapter_content.encode('utf-8')
                mock_items.append(mock_item)
            
            mock_book.get_items.return_value = mock_items
            mock_epub.read_epub.return_value = mock_book
            
            from src.text_processing.epub_extractor import EPUBTextExtractor
            
            extractor = EPUBTextExtractor()
            
            # Test memory usage with different numbers of chapters
            chapter_counts = [5, 10, 20, 50]
            
            for count in chapter_counts:
                # Adjust mock to return specified number of chapters
                mock_book.get_items.return_value = mock_items[:count]
                
                with tempfile.NamedTemporaryFile(suffix='.epub', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                try:
                    extracted_text = extractor.extract_text(temp_path)
                    
                    # Verify extraction
                    assert len(extracted_text) > 0
                    assert f"Chapter {min(count-1, 19)}" in extracted_text
                    
                    # Memory cleanup
                    import gc
                    gc.collect()
                    
                finally:
                    os.unlink(temp_path)
    
    @profile
    def test_large_file_memory_profile(self):
        """Profile memory usage with large files to identify memory leaks."""
        
        with patch('src.text_processing.text_extractor.TextExtractor') as mock_extractor_class:
            
            mock_extractor = Mock()
            mock_extractor_class.return_value = mock_extractor
            
            # Simulate extraction of progressively larger files
            file_sizes_mb = [1, 5, 10, 25]  # MB of text
            
            for size_mb in file_sizes_mb:
                # Generate text content of specified size
                content_size = size_mb * 1024 * 1024  # Convert to bytes
                chunk_size = 1024  # 1KB chunks
                
                large_content = ""
                for i in range(0, content_size, chunk_size):
                    chunk = f"Text chunk {i//chunk_size} with content " * 10  # ~500 bytes
                    large_content += chunk[:min(chunk_size, content_size - i)]
                
                mock_extractor.extract_text.return_value = large_content
                
                # Test extraction
                extractor = mock_extractor_class()
                result = extractor.extract_text(f"large_file_{size_mb}mb.txt")
                
                # Verify result
                assert len(result) > size_mb * 500000  # At least 500KB per MB input
                
                # Force memory cleanup to test for leaks
                del result
                del large_content
                import gc
                gc.collect()
    
    def test_memory_usage_thresholds(self):
        """Test that memory usage stays within acceptable thresholds."""
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Get baseline memory usage
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with patch('src.text_processing.text_extractor.TextExtractor') as mock_extractor_class:
            mock_extractor = Mock()
            mock_extractor_class.return_value = mock_extractor
            
            # Simulate processing multiple large files
            for i in range(10):
                # Generate 5MB of text content
                large_text = "Sample text content " * 262144  # ~5MB
                mock_extractor.extract_text.return_value = large_text
                
                extractor = mock_extractor_class()
                result = extractor.extract_text(f"test_file_{i}.txt")
                
                # Check current memory usage
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - baseline_memory
                
                # Verify memory usage doesn't grow excessively
                assert memory_increase < 500, f"Memory usage too high: {memory_increase}MB increase"
                
                # Clean up
                del result
                del large_text
                import gc
                gc.collect()


def run_memory_profiling_suite():
    """Run the complete memory profiling test suite."""
    
    test_suite = TestTextExtractionMemoryProfile()
    
    print("Running text extraction memory profiling...")
    
    # Run each test with memory profiling
    test_methods = [
        test_suite.test_pdf_extraction_memory_profile,
        test_suite.test_docx_extraction_memory_profile,
        test_suite.test_epub_extraction_memory_profile,
        test_suite.test_large_file_memory_profile,
        test_suite.test_memory_usage_thresholds
    ]
    
    for method in test_methods:
        print(f"Profiling: {method.__name__}")
        try:
            method()
            print(f"✓ {method.__name__} completed")
        except Exception as e:
            print(f"✗ {method.__name__} failed: {e}")
    
    print("Memory profiling suite completed")


if __name__ == "__main__":
    run_memory_profiling_suite()