"""Logging functionality for the document processing pipeline."""
import logging
from typing import List
from langchain.schema import Document

def setup_logger() -> logging.Logger:
    """
    Set up and configure logger.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger('document_processor')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def log_processing_steps(logger: logging.Logger, file_path: str, chunks: List[Document]) -> None:
    """
    Log processing steps and results.
    
    Args:
        logger (logging.Logger): Logger instance
        file_path (str): Path to processed file
        chunks (List[Document]): List of processed chunks
    """
    logger.info(f"Processing file: {file_path}")
    logger.info(f"Successfully split document into {len(chunks)} chunks") 