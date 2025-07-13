import os
import re
import docx
import fitz  # PyMuPDF
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import mobi

class TextExtractor:
    """Extracts text from various file formats."""

    def __init__(self):
        self.supported_formats = {
            ".txt": self._read_txt,
            ".md": self._read_txt,
            ".pdf": self._read_pdf,
            ".docx": self._read_docx,
            ".epub": self._read_epub,
            ".mobi": self._read_mobi,
        }

    def extract(self, file_path):
        """
        Extracts text from the given file based on its extension.

        Args:
            file_path (str): The absolute path to the file.

        Returns:
            str: The extracted text content.
        
        Raises:
            ValueError: If the file format is not supported.
        """
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {ext}")
        
        return self.supported_formats[ext](file_path)

    def _read_txt(self, file_path):
        """Reads text from a .txt or .md file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _read_pdf(self, file_path):
        """Reads text from a .pdf file with content filtering."""
        doc = fitz.open(file_path)
        
        # Extract text from all pages first
        all_pages_text = []
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            all_pages_text.append({
                'page_num': page_num,
                'text': page_text,
                'content_type': self._classify_page_content(page_text)
            })
        
        # Filter and process content
        filtered_text = self._filter_pdf_content(all_pages_text)
        return filtered_text

    def _read_docx(self, file_path):
        """Reads text from a .docx file."""
        doc = docx.Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text

    def _read_epub(self, file_path):
        """Reads text from an .epub file."""
        book = epub.read_epub(file_path)
        text = ""
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text += soup.get_text() + "\n"
        return text

    def _read_mobi(self, file_path):
        """Reads text from a .mobi file."""
        temp_dir, _ = mobi.extract(file_path)
        text = ""
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(('.html', '.htm')):
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        soup = BeautifulSoup(f.read(), 'html.parser')
                        text += soup.get_text() + "\n"
        return text
    
    def _classify_page_content(self, page_text):
        """
        Classify the content type of a PDF page.
        
        Args:
            page_text (str): Text content of the page
            
        Returns:
            str: Content type ('toc', 'chapter_header', 'metadata', 'story', 'mixed')
        """
        if not page_text.strip():
            return 'empty'
        
        # Clean and normalize text for analysis
        text = page_text.strip()
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Count different types of content indicators
        chapter_listing_count = 0
        dialogue_count = 0
        narrative_count = 0
        
        # Patterns for different content types
        chapter_listing_pattern = re.compile(r'^Chapter\s+\d+.*?[IVX]+\s*$', re.IGNORECASE)
        chapter_number_pattern = re.compile(r'^\d+\.\s*$')
        dialogue_markers = ['"', '"', '"', '―', '—', "'"]
        
        for line in lines:
            # Check for chapter listings (TOC pattern)
            if (chapter_listing_pattern.match(line) or 
                re.match(r'^Chapter\s+\d+:', line) or
                (len(line) < 100 and 'Chapter' in line and any(char in line for char in ['–', '—', ':']))):
                chapter_listing_count += 1
            
            # Check for standalone chapter numbers
            elif chapter_number_pattern.match(line):
                chapter_listing_count += 1
                
            # Check for dialogue
            elif any(marker in line for marker in dialogue_markers):
                dialogue_count += 1
                
            # Check for narrative prose (longer sentences)
            elif len(line) > 50 and '.' in line:
                narrative_count += 1
        
        total_lines = len(lines)
        if total_lines == 0:
            return 'empty'
        
        # Calculate ratios
        chapter_ratio = chapter_listing_count / total_lines
        dialogue_ratio = dialogue_count / total_lines
        narrative_ratio = narrative_count / total_lines
        
        # Classify based on content ratios
        if chapter_ratio > 0.6:  # More than 60% chapter listings
            return 'toc'
        elif chapter_ratio > 0.3 and dialogue_ratio < 0.1:  # Mixed TOC
            return 'mixed_toc'
        elif dialogue_ratio > 0.3 or narrative_ratio > 0.3:  # Significant story content
            return 'story'
        elif any(word in text.lower() for word in ['epilogue', 'prologue', 'author', 'preface']):
            return 'chapter_header'
        elif len(text) < 200:  # Very short content
            return 'metadata'
        else:
            return 'mixed'
    
    def _filter_pdf_content(self, pages_data):
        """
        Filter PDF content to extract only story-relevant text.
        
        Args:
            pages_data (list): List of page dictionaries with content type classification
            
        Returns:
            str: Filtered text content
        """
        story_text = []
        story_started = False
        consecutive_toc_pages = 0
        
        for page_data in pages_data:
            content_type = page_data['content_type']
            page_text = page_data['text']
            
            # Skip empty pages
            if content_type == 'empty':
                continue
            
            # Track consecutive TOC pages
            if content_type in ['toc', 'mixed_toc']:
                consecutive_toc_pages += 1
                # If we haven't started story content yet, skip TOC
                if not story_started:
                    continue
                # If we have too many consecutive TOC pages after story started, it might be another TOC section
                elif consecutive_toc_pages > 2:
                    continue
            else:
                consecutive_toc_pages = 0
            
            # Include story content
            if content_type == 'story':
                story_started = True
                story_text.append(self._clean_story_text(page_text))
            
            # Include chapter headers if story has started
            elif content_type == 'chapter_header' and story_started:
                cleaned_text = self._clean_chapter_header(page_text)
                if cleaned_text:
                    story_text.append(cleaned_text)
            
            # Include mixed content if story has started
            elif content_type == 'mixed' and story_started:
                cleaned_text = self._clean_mixed_content(page_text)
                if cleaned_text:
                    story_text.append(cleaned_text)
        
        return '\n\n'.join(story_text) if story_text else '\n'.join(page['text'] for page in pages_data)
    
    def _clean_story_text(self, text):
        """Clean story text by removing artifacts and formatting issues."""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # Remove standalone page numbers
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove header/footer artifacts (repeated short lines)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip very short lines that might be artifacts (but keep dialogue)
            if len(line) < 3 and not any(marker in line for marker in ['"', '"', '"', '―', '—']):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _clean_chapter_header(self, text):
        """Extract meaningful chapter headers, skip pure TOC entries."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Filter out pure chapter listing lines
        meaningful_lines = []
        for line in lines:
            # Skip lines that look like TOC entries
            if re.match(r'^Chapter\s+\d+.*?[IVX]+\s*$', line, re.IGNORECASE):
                continue
            elif re.match(r'^\d+\.\s*$', line):
                continue
            else:
                meaningful_lines.append(line)
        
        return '\n'.join(meaningful_lines) if meaningful_lines else ''
    
    def _clean_mixed_content(self, text):
        """Clean mixed content by filtering out TOC elements."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        cleaned_lines = []
        for line in lines:
            # Skip chapter listing patterns
            if (re.match(r'^Chapter\s+\d+.*?[IVX]+\s*$', line, re.IGNORECASE) or
                re.match(r'^\d+\.\s*$', line) or
                (len(line) < 50 and 'Chapter' in line and any(char in line for char in ['–', '—']))):
                continue
            else:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines) if cleaned_lines else ''