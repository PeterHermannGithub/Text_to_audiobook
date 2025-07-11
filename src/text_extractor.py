import os
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
        """Reads text from a .pdf file."""
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

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