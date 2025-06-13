"""Configuration settings for the document processing pipeline."""
import os

# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   
OUTPUT_DIR = os.path.join(BASE_DIR, "out")  


# File paths
MARKDOWN_PATH = "sample.md"
DEFAULT_OUTPUT_FILENAME = "main_out.txt"

# Text splitting configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
KEEP_SEPARATOR = True

# Tokenizer configuration
TOKENIZER_TYPE = "openai"  # or "sentence-transformers"
TOKENIZER_MODEL = "text-embedding-3-small"  # for OpenAI text-embedding-3-large
# TOKENIZER_MODEL = "all-MiniLM-L6-v2"  # for sentence-transformers
MAX_TOKENS = 1536  # Maximum tokens allowed by the model 

# Oracle database configuration
ORACLE_BATCH_SIZE = 100  # Default batch size for Oracle inserts 