import logging
import os
import trafilatura
from bs4 import BeautifulSoup
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Import Rust integration if available
try:
    from scraper.rust_integration import extract_with_rust
    RUST_EXTRACTOR_AVAILABLE = True
except ImportError:
    RUST_EXTRACTOR_AVAILABLE = False
    # Provide a fallback implementation to avoid unbound references
    def extract_with_rust(url: str, html_content: Optional[str] = None, 
                         use_api: bool = True) -> Tuple[Optional[str], Optional[str], str]:
        """Fallback implementation when Rust extractor is unavailable"""
        logger.debug("Rust extractor not available, using fallback")
        return None, None, html_content or ""

# Environment variable to control whether to use Rust extractor
USE_RUST_EXTRACTOR = os.environ.get('USE_RUST_EXTRACTOR', 'true').lower() in ('true', '1', 'yes')

def extract_content(url, html_content):
    """
    Extract the main content from a webpage
    
    Args:
        url (str): The URL of the webpage
        html_content (bytes): The raw HTML content
        
    Returns:
        tuple: (title, extracted_text, raw_html)
    """
    try:
        # Convert bytes to string for storage
        raw_html = html_content.decode('utf-8', errors='replace')
        
        # Try to use Rust extractor if available and enabled
        if RUST_EXTRACTOR_AVAILABLE and USE_RUST_EXTRACTOR:
            try:
                logger.debug(f"Attempting to extract content from {url} using Rust extractor")
                title, extracted_text, _ = extract_with_rust(url, raw_html)
                
                if title and extracted_text:
                    logger.info(f"Successfully extracted content from {url} using Rust extractor")
                    return title, extracted_text, raw_html
                else:
                    logger.warning(f"Rust extractor failed for {url}, falling back to Python extraction")
            except Exception as e:
                logger.warning(f"Error using Rust extractor for {url}: {str(e)}, falling back to Python extraction")
        
        # Extract title using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        title = soup.title.string if soup.title else ""
        
        # Try to extract content with Trafilatura
        logger.debug(f"Extracting content from {url} using Trafilatura")
        extracted_text = trafilatura.extract(html_content)
        
        # If Trafilatura fails, fall back to simple text extraction
        if not extracted_text:
            logger.warning(f"Trafilatura failed to extract content from {url}, falling back to basic extraction")
            extracted_text = soup.get_text(separator=' ', strip=True)
        
        return title, extracted_text, raw_html
    
    except Exception as e:
        logger.error(f"Error extracting content from {url}: {str(e)}")
        # Return basic information in case of error
        return "", "", html_content.decode('utf-8', errors='replace')

def clean_text(text):
    """
    Clean extracted text for better quality
    
    Args:
        text (str): The text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    cleaned = ' '.join(text.split())
    
    # Remove problematic Unicode characters
    cleaned = ''.join(c for c in cleaned if ord(c) >= 32 or c in '\n\r\t')
    
    # Replace multiple newlines with just two
    import re
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    return cleaned
