"""Embedding functionality for text processing."""
from typing import List, Union, Literal, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from .tokenizer import Tokenizer
from dotenv import load_dotenv

class Embedder:
    """Embedder class that supports both OpenAI and sentence-transformers models."""
    
    def __init__(self, model_type: Literal["openai", "sentence-transformers"], 
                 model_name: str,
                 tokenizer: Tokenizer,
                 max_tokens: int,
                 dimensions: Optional[int] = None): # <-- Added optional dimensions
        """
        Initialize the embedder.
        
        Args:
            model_type: Type of model to use ("openai" or "sentence-transformers")
            model_name: Name of the model
            tokenizer: Tokenizer instance for validation
            max_tokens: Maximum tokens allowed by the model
            dimensions: The number of dimensions the resulting output embeddings should have.
                        Only supported in text-embedding-3 and later models.
        """
        self.model_type = model_type
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.dimensions = dimensions # <-- Store dimensions
        
        if model_type == "openai":
            load_dotenv()
            self._client = OpenAI()  # Initialize OpenAI client
        else:  # sentence-transformers
            self._model = SentenceTransformer(model_name)
    
    def _validate_chunks(self, chunks: List[str]) -> List[str]:
        """
        Validate chunks against token limit and filter out invalid ones.
        
        Args:
            chunks: List of text chunks to validate
            
        Returns:
            List[str]: List of valid chunks
        """
        valid_chunks = []
        for chunk in chunks:
            if self.tokenizer.validate_chunk_size(chunk, self.max_tokens):
                valid_chunks.append(chunk)
            else:
                print(f"Warning: Chunk exceeds token limit ({self.max_tokens}) and will be skipped")
        return valid_chunks
    
    def _embed_openai(self, chunks: List[str]) -> List[List[float]]:
        """
        Embed chunks using OpenAI API in a single batch request.
        
        Args:
            chunks: List of text chunks to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        # Create the request body, adding dimensions if specified
        request_body = {
            "input": chunks,  # Pass the entire list of chunks for batch processing 
            "model": self.model_name
        }
        if self.dimensions:
            request_body["dimensions"] = self.dimensions # Optional parameter for newer models 

        # Make a single API call for all chunks
        response = self._client.embeddings.create(**request_body)
        
        # Extract the embedding vector from each object in the response
        # The response data is a list of embedding objects 
        embeddings = [item.embedding for item in response.data]
        return embeddings
    
    def _embed_sentence_transformers(self, chunks: List[str]) -> List[List[float]]:
        """
        Embed chunks using sentence-transformers.
        
        Args:
            chunks: List of text chunks to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        embeddings = self._model.encode(chunks, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        """
        Embed a list of text chunks.
        
        Args:
            chunks: List of text chunks to embed
            
        Returns:
            List[List[float]]: List of embedding vectors as float32
        """
        # Validate chunks
        valid_chunks = self._validate_chunks(chunks)
        
        if not valid_chunks:
            return []
        
        # Get embeddings
        if self.model_type == "openai":
            embeddings = self._embed_openai(valid_chunks)
        else:  # sentence-transformers
            embeddings = self._embed_sentence_transformers(valid_chunks)
        
        # Convert to float32
        return [np.array(embedding, dtype=np.float32).tolist() for embedding in embeddings]
    
    def format_embeddings_for_output(self, embeddings: List[List[float]]) -> str:
        """
        Format embeddings for output file.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            str: Formatted string representation of embeddings
        """
        formatted = []
        for i, embedding in enumerate(embeddings):
            formatted.append(f"=== Embedding {i+1} ===\n")
            formatted.append(f"Vector Length: {len(embedding)}\n")
            formatted.append("Values:\n")
            # Format as a single line of comma-separated values
            formatted.append(",".join(map(str, embedding)))
            formatted.append("\n\n")
        return "\n".join(formatted)