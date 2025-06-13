"""Tokenizer functionality for text processing."""
from typing import Optional, Literal
import tiktoken
from sentence_transformers import SentenceTransformer

class Tokenizer:
    """Tokenizer class that supports both OpenAI and sentence-transformers models."""
    
    def __init__(self, model_type: Literal["openai", "sentence-transformers"], 
                 model_name: Optional[str] = None):
        """
        Initialize the tokenizer.
        
        Args:
            model_type: Type of model to use ("openai" or "sentence-transformers")
            model_name: Name of the model (required for sentence-transformers)
        """
        self.model_type = model_type
        self.model_name = model_name
        self._tokenizer = None
        
        if model_type == "openai":
            self._tokenizer = tiktoken.encoding_for_model(model_name or "text-embedding-3-small")
        elif model_type == "sentence-transformers":
            if not model_name:
                raise ValueError("model_name is required for sentence-transformers")
            model = SentenceTransformer(model_name)
            self._tokenizer = model.tokenizer
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            int: Number of tokens
        """
        if self.model_type == "openai":
            return len(self._tokenizer.encode(text))
        else:  # sentence-transformers
            return len(self._tokenizer.encode(text, add_special_tokens=False))
    
    def validate_chunk_size(self, text: str, max_tokens: int) -> bool:
        """
        Validate if a text chunk is within the maximum token limit.
        
        Args:
            text: The text to validate
            max_tokens: Maximum allowed tokens
            
        Returns:
            bool: True if text is within limit, False otherwise
        """
        return self.count_tokens(text) <= max_tokens 