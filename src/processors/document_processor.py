"""Document processing functionality."""
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document

def load_document(file_path: str) -> List[Document]:
    """
    Load and clean markdown document.
    
    Args:
        file_path (str): Path to the markdown file
        
    Returns:
        List[Document]: List of processed documents
    """
    loader = UnstructuredMarkdownLoader(file_path, mode="elements")
    return loader.load()

def create_text_splitter(chunk_size: int, chunk_overlap: int, 
                        separators: List[str], keep_separator: bool) -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter with specified configuration.
    
    Args:
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Overlap between chunks
        separators (List[str]): List of separators to use
        keep_separator (bool): Whether to keep separators in output
        
    Returns:
        RecursiveCharacterTextSplitter: Configured text splitter
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        keep_separator=keep_separator
    ) 