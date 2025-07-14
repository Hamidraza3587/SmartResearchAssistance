from typing import List, Dict, Any, Optional, Tuple
from .llm import LLMClient
from .embeddings import EmbeddingModel
import logging
import re

logger = logging.getLogger(__name__)

class QAEngine:
    """Handles question answering over documents using retrieval-augmented generation."""
    
    def __init__(self, embedding_model: Optional[EmbeddingModel] = None, 
                 llm_client: Optional[LLMClient] = None):
        """
        Initialize the QA Engine.
        
        Args:
            embedding_model: Pre-initialized embedding model (optional)
            llm_client: Pre-initialized LLM client (optional)
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.llm = llm_client or LLMClient()
    
    def index_document(self, chunks: List[Tuple[str, int]]):
        """
        Index document chunks for retrieval.
        
        Args:
            chunks: List of (text, chunk_number) tuples
        """
        self.embedding_model.create_embeddings(chunks)
    
    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Format sources for inclusion in the answer."""
        if not sources:
            return "No specific sources were referenced."
            
        source_list = []
        for i, source in enumerate(sources, 1):
            chunk_num = source.get('chunk', '?')
            score = source.get('relevance_score', 0)
            source_list.append(f"- Chunk {chunk_num} (Relevance: {score:.2f})")
        
        return "\n".join(source_list)
    
    def _format_context(self, chunks: List[Tuple[str, int, float]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Format context from relevant chunks."""
        context_parts = []
        sources = []
        
        for chunk_text, chunk_num, score in chunks:
            context_parts.append(f"[Chunk {chunk_num}, Relevance: {score:.2f}]\n{chunk_text}")
            sources.append({"chunk": chunk_num, "relevance_score": score})
        
        return "\n\n".join(context_parts), sources
    
    def _extract_final_answer(self, text: str) -> str:
        """Extract just the answer part from the model's response."""
        # Remove any system-like messages or instructions
        text = re.sub(r'\[INST\].*?\[/INST\]', '', text, flags=re.DOTALL)
        text = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', text, flags=re.DOTALL)
        
        # Remove any markdown formatting
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`.*?`', '', text)
        
        # Clean up whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def answer_question(self, question: str, max_chars: int = 4000) -> Dict[str, Any]:
        """
        Answer a question based on the indexed document using RAG (Retrieval-Augmented Generation).
        
        Args:
            question: The question to answer
            max_chars: Maximum number of characters to include in the context
            
        Returns:
            Dict containing the answer, sources, and context
        """
        try:
            # Search for relevant chunks using semantic search
            relevant_chunks = self.embedding_model.search(question, k=3)
            
            if not relevant_chunks:
                return {
                    "answer": "I couldn't find any relevant information to answer this question.",
                    "sources": [],
                    "context": ""
                }
            
            # Prepare context from relevant chunks
            context, sources = self._format_context(relevant_chunks)
            
            # Format the prompt for the LLM
            prompt = f"""[INST] <<SYS>>
You are a helpful research assistant that answers questions based on the provided context.
- If the context contains the answer, provide a clear and concise response.
- If the context doesn't contain enough information, say that you don't know.
- Always cite the relevant chunk numbers in your answer.
- If the question cannot be answered from the context, say so.
<</SYS>>

Context:
{context}

Question: {question}

Please provide a clear and concise answer based on the context above. 
If the context doesn't contain the answer, say "I couldn't find enough information to answer this question."
At the end, list the relevant chunk numbers in parentheses, like (Chunk 1, Chunk 3).

Answer: [/INST]"""
            
            # Generate answer using the LLM
            answer = self.llm.generate(
                prompt=prompt,
                max_tokens=1024,
                temperature=0.2,  # Lower temperature for more focused answers
                top_p=0.9,
                do_sample=True
            )
            
            # Clean up the answer
            answer = self._extract_final_answer(answer)
            
            return {
                "answer": answer,
                "sources": sources,
                "context": context
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}", exc_info=True)
            return {
                "answer": f"Sorry, I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "context": ""
            }
