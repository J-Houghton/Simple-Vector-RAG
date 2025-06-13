"""Utility functions for URL extraction."""
import re

def extract_urls(text: str) -> list[str]:
    """
    Extract URLs from text using regex pattern matching.
    
    Args:
        text (str): The text to extract URLs from
        
    Returns:
        list[str]: List of found URLs
    """
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text) 