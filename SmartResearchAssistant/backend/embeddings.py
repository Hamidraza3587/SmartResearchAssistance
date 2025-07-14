import numpy as np
import faiss
from typing import List, Tuple, Optional, Union
import logging
import torch
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import login
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Handles text embeddings and similarity search using FAISS with Hugging Face models."""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        """
        Initialize the embedding model and FAISS index.
        
        Args:
            model_name: Name of the Hugging Face model to use for embeddings
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.index = None
        self.chunks = []
        self.chunk_numbers = []
        self.embedding_size = None
        
        # Get Hugging Face token
        self.token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if self.token:
            login(token=self.token)
        
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading embedding model {self.model_name} on {self.device}...")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.token
            )
            
            self.model = AutoModel.from_pretrained(
                self.model_name,
                token=self.token,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            # Set embedding size based on model config
            self.embedding_size = self.model.config.hidden_size
            
            logger.info(f"Embedding model {self.model_name} loaded successfully.")
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def create_embeddings(self, chunks: List[Tuple[str, int]]) -> np.ndarray:
        """
        Create embeddings for a list of text chunks.
        
        Args:
            chunks: List of (text, chunk_number) tuples
            
        Returns:
            np.ndarray: Array of embeddings
        """
        texts = [chunk[0] for chunk in chunks]
        self.chunk_numbers = [chunk[1] for chunk in chunks]
        self.chunks = texts
        
        logger.info(f"Creating embeddings for {len(texts)} chunks...")
        
        # Tokenize input texts
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Get sentence embeddings using mean pooling
        embeddings = self._mean_pooling(
            model_output,
            encoded_input['attention_mask']
        )
        
        # Convert to numpy and normalize
        embeddings = embeddings.cpu().numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_size)  # Using inner product for cosine similarity
        self.index.add(embeddings.astype('float32'))
        
        return embeddings
    
    def search(self, query: str, k: int = 3) -> List[Tuple[str, int, float]]:
        """
        Search for the most similar chunks to the query.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of (chunk_text, chunk_number, similarity_score) tuples
        """
        if self.index is None or not self.chunks:
            raise ValueError("No documents have been indexed yet.")
        
        # Encode the query
        encoded_query = self.tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            model_output = self.model(**encoded_query)
        
        # Get query embedding
        query_embedding = self._mean_pooling(
            model_output,
            encoded_query['attention_mask']
        )
        
        # Normalize query embedding
        query_embedding = query_embedding.cpu().numpy()
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search the index (using inner product for cosine similarity)
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Get the results
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0:  # Skip invalid indices
                continue
            chunk_text = self.chunks[idx]
            chunk_number = self.chunk_numbers[idx]
            # Convert to similarity score (0 to 1)
            similarity = (score + 1) / 2  # Convert from [-1, 1] to [0, 1]
            results.append((chunk_text, chunk_number, float(similarity)))
        
        return results
