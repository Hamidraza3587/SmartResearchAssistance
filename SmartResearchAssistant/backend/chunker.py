import re
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class TextChunker:
    """
    Handles intelligent text chunking with support for paragraph and sentence boundaries.
    Implements configurable chunk size and overlap for optimal document processing.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the TextChunker with configuration.
        
        Args:
            chunk_size: Target size of each chunk in characters (default: 1000)
            chunk_overlap: Number of characters to overlap between chunks (default: 200)
        """
        self.chunk_size = max(100, chunk_size)  # Enforce minimum chunk size
        self.chunk_overlap = max(0, min(chunk_overlap, chunk_size // 2))  # Ensure reasonable overlap
        
        # Compile regex patterns for better performance
        self.sentence_endings = re.compile(r'(?<=[.!?])\s+')  # Split on sentence boundaries
        self.paragraph_separator = re.compile(r'\n\s*\n+')  # Split on paragraph boundaries
        self.whitespace = re.compile(r'\s+')  # Match any whitespace
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks of approximately chunk_size characters,
        respecting sentence and paragraph boundaries where possible.
        
        Args:
            text: The input text to be chunked
            
        Returns:
            List of text chunks
        """
        if not text or not isinstance(text, str):
            return []
        
        # Normalize whitespace first
        text = self.whitespace.sub(' ', text).strip()
        
        # If text is smaller than chunk size, return as is
        if len(text) <= self.chunk_size:
            return [text]
        
        # First try to split by paragraphs
        chunks = self._split_by_paragraphs(text)
        
        # If paragraphs are too large, split by sentences
        if any(len(chunk) > self.chunk_size * 1.5 for chunk in chunks):
            chunks = self._split_by_sentences(text)
        
        # If still too large, split by fixed size with overlap
        if any(len(chunk) > self.chunk_size * 1.5 for chunk in chunks):
            chunks = self._split_fixed_size(text)
        
        return chunks
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text into chunks based on paragraph boundaries."""
        paragraphs = [p.strip() for p in self.paragraph_separator.split(text) if p.strip()]
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If adding this paragraph would exceed chunk size (with some buffer)
            if current_chunk and current_size + len(para) > self.chunk_size * 1.1:
                chunks.append('\n\n'.join(current_chunk))
                # Start a new chunk with overlap if possible
                overlap_start = max(0, len(current_chunk) - 2)  # Use last 2 paragraphs as overlap
                current_chunk = current_chunk[overlap_start:]
                current_size = sum(len(p) for p in current_chunk) + (len(current_chunk) - 1) * 2  # +2 for newlines
            
            current_chunk.append(para)
            current_size += len(para) + (2 if current_chunk else 0)  # +2 for newlines
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text into chunks based on sentence boundaries."""
        # First split into sentences
        sentences = [s.strip() for s in self.sentence_endings.split(text) if s.strip()]
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed chunk size
            if current_chunk and current_size + len(sentence) > self.chunk_size * 1.1:
                chunks.append(' '.join(current_chunk))
                # Start a new chunk with overlap if possible
                overlap_sentences = min(2, len(current_chunk) - 1)  # Use last 1-2 sentences as overlap
                if overlap_sentences > 0:
                    current_chunk = current_chunk[-overlap_sentences:]
                    current_size = sum(len(s) for s in current_chunk) + len(current_chunk) - 1  # +1 for spaces
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += len(sentence) + (1 if current_chunk else 0)  # +1 for space
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _split_fixed_size(self, text: str) -> List[str]:
        """Split text into fixed-size chunks with overlap."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position for this chunk
            end = min(start + self.chunk_size, text_length)
            
            # If we're not at the end, try to find a good breaking point
            if end < text_length:
                # Look for a sentence or paragraph end within the last 20% of the chunk
                lookback = int(self.chunk_size * 0.2)
                lookahead = int(self.chunk_size * 0.1)
                
                # First try to find a paragraph break
                para_break = text.rfind('\n\n', end - lookback, end + lookahead)
                if para_break != -1 and para_break > start:
                    end = para_break + 2  # Include the newlines
                else:
                    # Then try to find a sentence end
                    sentence_break = max(
                        text.rfind('. ', end - lookback, end + lookahead),
                        text.rfind('! ', end - lookback, end + lookahead),
                        text.rfind('? ', end - lookback, end + lookahead)
                    )
                    if sentence_break != -1 and sentence_break > start:
                        end = sentence_break + 1  # Include the space
            
            # Add the chunk
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move the start position, accounting for overlap
            if end == text_length:
                break
                
            next_start = end - min(self.chunk_overlap, end - start)
            if next_start <= start:  # Ensure we make progress
                next_start = end
                
            start = next_start
        
        return chunks
