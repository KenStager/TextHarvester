"""
Integration with the Rust content extractor for improved performance.
This module provides a bridge to the Rust-based content extraction engine.
"""
import json
import logging
import os
import subprocess
import requests
from typing import Dict, Optional, Tuple, Union, Any, cast

# Configure logging
logger = logging.getLogger(__name__)

# Constants
RUST_EXTRACTOR_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                  "rust_extractor", "target", "release", "rust_extractor")
RUST_API_URL = os.environ.get("RUST_EXTRACTOR_API_URL", "http://localhost:8888")

class RustExtractorClient:
    """Client for the Rust content extractor"""
    
    def __init__(self, use_api: bool = True, api_url: Optional[str] = None):
        """
        Initialize the Rust extractor client
        
        Args:
            use_api: Whether to use the API server (True) or direct process calls (False)
            api_url: URL of the Rust extractor API server (default: from env or http://localhost:8888)
        """
        self.use_api = use_api
        self.api_url = api_url or RUST_API_URL
        self._check_setup()
    
    def _check_setup(self) -> bool:
        """
        Check if the Rust extractor is properly set up
        
        Returns:
            bool: True if setup is valid, False otherwise
        """
        if not self.use_api:
            # Check if binary exists
            if not os.path.exists(RUST_EXTRACTOR_PATH):
                logger.warning(f"Rust extractor binary not found at {RUST_EXTRACTOR_PATH}")
                logger.warning("Falling back to Python-based extraction")
                return False
        else:
            # Check if API is reachable
            try:
                response = requests.get(f"{self.api_url}/api/health", timeout=2)
                if response.status_code != 200:
                    logger.warning(f"Rust extractor API returned status {response.status_code}")
                    logger.warning("Falling back to Python-based extraction")
                    return False
            except requests.RequestException as e:
                logger.warning(f"Failed to connect to Rust extractor API: {e}")
                logger.warning("Falling back to Python-based extraction")
                return False
        
        return True
    
    def extract_content(self, url: str, html_content: Optional[str] = None, 
                       clean_text: bool = True) -> Dict[str, Any]:
        """
        Extract content from a URL or HTML using the Rust extractor
        
        Args:
            url: The URL to extract content from
            html_content: Optional HTML content (if provided, URL is only used for reference)
            clean_text: Whether to clean the extracted text
            
        Returns:
            dict: Extraction results with text, title, metadata, etc.
        """
        if self.use_api:
            return self._extract_via_api(url, html_content, clean_text)
        else:
            return self._extract_via_process(url, html_content, clean_text)
    
    def _extract_via_api(self, url: str, html_content: Optional[str], 
                        clean_text: bool) -> Dict[str, Any]:
        """Extract content using the Rust API server"""
        try:
            payload = {
                "url": url,
                "options": {"clean_text": clean_text}
            }
            
            if html_content:
                payload["html_content"] = html_content
            
            response = requests.post(
                f"{self.api_url}/api/extract",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Rust extractor API error: {response.status_code} - {response.text}")
                raise ValueError(f"API extraction failed with status {response.status_code}")
                
            return cast(Dict[str, Any], response.json())
            
        except requests.RequestException as e:
            logger.error(f"Error calling Rust extractor API: {e}")
            raise
    
    def _extract_via_process(self, url: str, html_content: Optional[str], 
                           clean_text: bool) -> Dict[str, Any]:
        """Extract content by calling the Rust binary directly"""
        try:
            cmd = [RUST_EXTRACTOR_PATH, "extract", url, "--format", "json"]
            
            if not clean_text:
                cmd.append("--no-clean")
                
            # If we have HTML content, we need to pass it through stdin
            # This is more complex and would require temporary files in a real implementation
            if html_content:
                logger.warning("Direct process extraction doesn't support HTML content input")
                logger.warning("URL will be fetched directly")
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return cast(Dict[str, Any], json.loads(result.stdout))
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Rust extractor process error: {e.stderr}")
            raise ValueError(f"Process extraction failed with code {e.returncode}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Rust extractor output: {e}")
            raise

def extract_with_rust(url: str, html_content: Optional[str] = None, 
                    use_api: bool = True) -> Tuple[Optional[str], Optional[str], str]:
    """
    Extract content using the Rust extractor
    
    Args:
        url: The URL to extract content from
        html_content: Optional HTML content (if provided, URL is only used for reference)
        use_api: Whether to use the API server (True) or direct process calls (False)
        
    Returns:
        tuple: (title, extracted_text, raw_html)
    """
    client = RustExtractorClient(use_api=use_api)
    html = html_content or ""
    
    try:
        result = client.extract_content(url, html)
        
        if not result.get("success", False):
            error = result.get("error", "Unknown error")
            logger.error(f"Rust extraction failed: {error}")
            return None, None, html
        
        # Extract title and text, ensuring they're the correct types
        title = result.get("title")
        text = result.get("text", "")
        
        # Return with proper types
        return title, text, html
        
    except Exception as e:
        logger.exception(f"Error using Rust extractor: {e}")
        return None, None, html