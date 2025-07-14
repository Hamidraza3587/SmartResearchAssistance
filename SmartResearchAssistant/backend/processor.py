import re
import logging
import os
from typing import List, Tuple, Optional, Dict, Any
import pymupdf  # PyMuPDF
from .chunker import TextChunker

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handles document processing including text extraction, cleaning, and chunking.
    Supports PDF and TXT file formats with configurable chunking.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the DocumentProcessor with chunking configuration.
        
        Args:
            chunk_size: Size of each text chunk in characters (default: 1000)
            chunk_overlap: Overlap between chunks in characters (default: 200)
        """
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract and clean text from a PDF file using PyMuPDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted and cleaned text as a string
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            RuntimeError: If there's an error reading the PDF or if no text could be extracted
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        doc = None
        try:
            logger.info(f"Extracting text from PDF: {os.path.basename(file_path)}")
            doc = pymupdf.open(file_path)
            
            # Check if the document is encrypted
            if doc.needs_pass:
                raise RuntimeError("PDF is encrypted and requires a password")
            
            # Check if the document has any pages
            if len(doc) == 0:
                raise RuntimeError("PDF contains no pages")
            
            # Extract text from each page
            text_parts = []
            for page_num, page in enumerate(doc, 1):
                try:
                    page_text = page.get_text("text")
                    if page_text.strip():
                        text_parts.append(page_text)
                    else:
                        logger.warning(f"Page {page_num} appears to be empty or contains no text")
                except Exception as page_error:
                    logger.warning(f"Error extracting text from page {page_num}: {str(page_error)}")
            
            # Combine all page texts
            full_text = "\n\n".join(text_parts)
            
            if not full_text.strip():
                # Try alternative text extraction method if the standard one fails
                logger.info("Standard text extraction failed, trying alternative method...")
                full_text = self._extract_text_alternative(doc)
                
                if not full_text.strip():
                    raise RuntimeError("The document appears to be empty or contains no extractable text. "
                                     "This might be a scanned document or an image-based PDF.")
            
            return self._clean_text(full_text)
            
        except Exception as e:
            error_msg = f"Error extracting text from PDF {os.path.basename(file_path)}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
        finally:
            if doc:
                doc.close()
    
    def _extract_text_alternative(self, doc) -> str:
        """Alternative text extraction method for problematic PDFs."""
        text_parts = []
        for page in doc:
            # Try different text extraction methods
            for method in ["text", "blocks", "words"]:
                try:
                    if method == "text":
                        page_text = page.get_text("text")
                    elif method == "blocks":
                        blocks = page.get_text("blocks")
                        page_text = "\n".join(block[4] for block in blocks if block[4].strip())
                    elif method == "words":
                        words = page.get_text("words")
                        page_text = " ".join(word[4] for word in words if word[4].strip())
                    
                    if page_text.strip():
                        text_parts.append(page_text)
                        break
                except Exception as e:
                    logger.debug(f"Text extraction method '{method}' failed: {str(e)}")
        
        return "\n\n".join(text_parts)
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """
        Extract and clean text from a plain text file with proper encoding handling.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Extracted text as a single string
            
        Raises:
            FileNotFoundError: If the text file doesn't exist
            RuntimeError: If there's an error reading the file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found: {file_path}")
            
        try:
            logger.info(f"Reading text file: {os.path.basename(file_path)}")
            # Try UTF-8 first, fall back to other common encodings if needed
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                        if content.strip():
                            return content.strip()
                except UnicodeDecodeError:
                    continue
            
            # If we get here, all encodings failed
            raise UnicodeDecodeError(f"Could not decode {file_path} with any of the attempted encodings")
                
        except Exception as e:
            error_msg = f"Error reading text file {os.path.basename(file_path)}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def process_document(self, file_path: str) -> List[Tuple[str, int]]:
        """
        Process a document (PDF or TXT) and return clean, chunked text with metadata.
        
        Args:
            file_path: Path to the document file (must be .pdf or .txt)
            
        Returns:
            List of (chunk_text, chunk_number) tuples where chunk_text is the 
            processed text and chunk_number is its 1-based position
            
        Raises:
            ValueError: For unsupported file formats or processing errors
        """
        try:
            # Validate file exists and has a supported extension
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Extract text based on file type
            if file_ext == '.pdf':
                text = self.extract_text_from_pdf(file_path)
            elif file_ext == '.txt':
                text = self.extract_text_from_txt(file_path)
            else:
                raise ValueError(
                    f"Unsupported file format: {file_ext}. "
                    "Only .pdf and .txt files are supported."
                )
            
            if not text.strip():
                raise ValueError("The document appears to be empty or contains no extractable text.")
            
            # Clean and normalize the text
            text = self._clean_text(text)
            
            # Split into chunks
            chunks = self.chunker.chunk_text(text)
            
            if not chunks:
                raise ValueError("No valid text chunks could be created from the document.")
            
            # Add 1-based chunk numbers and return
            return [(chunk, i+1) for i, chunk in enumerate(chunks)]
            
        except Exception as e:
            error_msg = f"Error processing document {os.path.basename(file_path)}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize the extracted text by:
        1. Removing excessive whitespace and line breaks
        2. Normalizing quotes and dashes
        3. Fixing common OCR/PDF extraction artifacts
        4. Ensuring proper spacing around punctuation
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned and normalized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Replace various whitespace characters with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize different types of quotes and apostrophes
        text = re.sub(r'[\u2018\u2019\u201A\u201B\u2032\u2035]', "'", text)  # Single quotes
        text = re.sub(r'[\u201C\u201D\u201E\u201F\u2033\u2036]', '"', text)  # Double quotes
        
        # Normalize different types of dashes
        text = re.sub(r'[\u2010-\u2015]', '-', text)  # Various dash/hyphen characters
        
        # Fix common OCR/PDF extraction issues
        text = re.sub(r'(\w)- (\w)', r'\1\2', text)  # Fix hyphenated words split across lines
        text = re.sub(r'(\w) - (\w)', r'\1-\2', text)  # Fix spaces around hyphens
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([(\[{"])\s+', r'\1', text)  # Remove space after opening brackets/quotes
        text = re.sub(r'\s+([)\]}"])', r'\1', text)  # Remove space before closing brackets/quotes
        
        # Fix multiple consecutive punctuation marks
        text = re.sub(r'([.,;:!?])(?=[.,;:!?])', '', text)
        
        # Normalize paragraph breaks (two or more newlines)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove any remaining control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        return text.strip()
