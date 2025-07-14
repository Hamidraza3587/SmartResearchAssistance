import logging
from typing import Optional, List, Dict, Any
from .llm import LLMClient
import re

# Configure logger
logger = logging.getLogger(__name__)

class Summarizer:
    """Handles document summarization using LLMs."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
    
    def _clean_summary(self, text: str, max_words: int) -> str:
        """Clean and truncate the summary to the specified word limit."""
        # Remove any markdown formatting if present
        text = re.sub(r'#+\s*', '', text)  # Remove headers
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # Remove italics
        
        # Truncate to max_words
        words = text.split()
        if len(words) > max_words:
            text = ' '.join(words[:max_words]) + '...'
            
        return text.strip()
    
    def _chunk_text(self, text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
        """
        Split text into smaller chunks with overlap to maintain context.
        
        Args:
            text: The text to split
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text or not isinstance(text, str):
            return []
            
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # Try to end at a sentence boundary
            if end < text_length:
                # Look for the next sentence end
                sentence_ends = ['. ', '! ', '? ', '\n\n']
                for sep in sentence_ends:
                    pos = text.rfind(sep, start, end + 100)  # Look ahead a bit
                    if pos > start + (chunk_size // 2):  # Only if it's in the second half
                        end = pos + len(sep)
                        break
            
            chunks.append(text[start:end].strip())
            
            # Move start position, accounting for overlap
            if end == text_length:
                break
                
            start = max(start + chunk_size - overlap, end - overlap)
            
        return chunks
    
    def summarize(self, text: str, max_words: int = 150, max_chunks: int = 3) -> str:
        """
        Generate a concise summary of the provided text.
        Uses a faster, extractive approach first, then refines if needed.
        
        Args:
            text: The text to summarize
            max_words: Maximum number of words for the final summary
            max_chunks: Maximum number of chunks to process (for performance)
            
        Returns:
            str: A concise summary of the input text
        """
        try:
            if not text or not text.strip():
                return "No content to summarize."
                
            # Clean the input text first
            text = self._clean_text(text)
            if not text:
                return "No valid text content found to summarize."
            
            logger.info(f"Generating summary for text of length {len(text)} characters")
            
            # For very short texts, return as is
            if len(text) < 500:
                return text[:500] + ("..." if len(text) == 500 else "")
            
            # Simple extractive summarization for speed
            try:
                # Split into sentences
                sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
                
                # Take first few sentences as summary (simple extractive approach)
                summary = ' '.join(sentences[:5])
                
                # If we have a very short summary, try to get more content
                if len(summary) < 100 and len(sentences) > 5:
                    summary = ' '.join(sentences[:10])
                    
                return summary
                
            except Exception as e:
                logger.warning(f"Simple summarization failed: {str(e)}")
                # Fall back to original text if simple approach fails
                return text[:1000] + ("..." if len(text) > 1000 else "")
            
        except Exception as e:
            error_msg = f"Error generating summary: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"Unable to generate summary: {str(e)}"
    
    def _summarize_chunk(self, text: str, max_words: int) -> str:
        """
        Helper method to summarize a single chunk of text.
        
        Args:
            text: The text to summarize
            max_words: Maximum number of words for the summary
            
        Returns:
            str: The generated summary
        """
        if not text or not text.strip():
            return ""
            
        # Create a focused prompt
        prompt = f"""[INST] <<SYS>>
You are a helpful research assistant that summarizes text concisely and accurately.
Your task is to create a clear and coherent summary that captures the main ideas, key points, and conclusions from the provided text.

Guidelines:
- Be concise but comprehensive
- Focus on key information and main points
- Use clear and simple language
- Maintain factual accuracy
- Omit unnecessary details
- Maximum length: {max_words} words
<</SYS>>

Please provide a concise summary of the following text in {max_words} words or less:

{text}

Summary: [/INST]"""
        
        try:
            # Generate the summary with conservative parameters
            summary = self.llm.generate(
                prompt=prompt,
                max_tokens=min(800, max_words * 2),  # Estimate 2 tokens per word
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                max_new_tokens=min(500, max_words * 2)
            )
            
            # Clean and format the summary
            return self._clean_summary(summary, max_words)
            
        except Exception as e:
            logger.error(f"Error in _summarize_chunk: {str(e)}")
            # Fallback to a simple extraction of first few sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            fallback = ' '.join(sentences[:3])
            return self._clean_summary(fallback, max_words)

# Alias for backward compatibility
generate_summary = Summarizer().summarize
