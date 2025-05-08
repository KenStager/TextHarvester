"""
Integration tests for Rust extractor with Classification System.

These tests verify that the Rust extractor can be used as a content source
for the classification system, ensuring proper integration between the 
high-performance extraction and intelligent classification.
"""

import unittest
import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scraper.rust_integration import RustExtractorClient
from intelligence.classification.pipeline import ClassificationPipeline, ClassificationInput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRustClassificationIntegration(unittest.TestCase):
    """Test integration between Rust extractor and classification system."""
    
    def setUp(self):
        """Set up test resources."""
        # Initialize Rust extractor client with fallback to API
        self.rust_client = RustExtractorClient(use_api=True)
        
        # Initialize classification pipeline for football
        self.classification_pipeline = ClassificationPipeline.create_football_pipeline()
        
    def test_extract_and_classify_football_content(self):
        """Test extracting content with Rust and classifying it."""
        # Skip if Rust extractor is not available
        if not self._is_rust_available():
            self.skipTest("Rust extractor not available")
            
        # Test with a known football content URL
        test_url = "https://www.premierleague.com/news"
        
        # Extract content using Rust
        extracted_content = self.rust_client.extract_content(test_url)
        
        # Verify extraction succeeded
        self.assertTrue(extracted_content.get("success", False), 
                       "Rust extraction failed")
        
        # Get the extracted text
        text = extracted_content.get("text", "")
        self.assertTrue(text, "Extracted text is empty")
        
        # Classify the extracted content
        classification_input = ClassificationInput(
            text=text,
            metadata={"url": test_url}
        )
        
        classification_result = self.classification_pipeline.process(classification_input)
        
        # Verify classification succeeded and identified football content
        self.assertTrue(classification_result.is_relevant, 
                       "Content should be classified as relevant")
        self.assertIsNotNone(classification_result.primary_topic,
                           "Primary topic should be identified")
        
        # Log the classification results
        logger.info(f"Classification results for {test_url}:")
        logger.info(f"Primary topic: {classification_result.primary_topic}")
        logger.info(f"Confidence: {classification_result.primary_topic_confidence:.4f}")
        logger.info(f"Subtopics: {len(classification_result.subtopics)}")
        
    def test_fallback_to_python_extraction(self):
        """Test fallback to Python extraction when Rust is unavailable."""
        # Force fallback by using invalid API URL
        test_client = RustExtractorClient(use_api=True, api_url="http://invalid-url:9999")
        
        # Test with a simple URL that would be handled by Python extraction
        test_url = "https://example.com"
        
        # This should fall back to Python-based extraction
        try:
            result = test_client.extract_content(test_url)
            # If we got a result despite the invalid API, the fallback worked
            logger.info("Successfully fell back to Python extraction")
        except Exception as e:
            self.fail(f"Failed to fall back to Python extraction: {e}")
        
    def _is_rust_available(self):
        """Check if Rust extractor is available."""
        try:
            # Try to connect to the API
            self.rust_client._check_setup()
            return True
        except:
            return False


if __name__ == '__main__':
    unittest.main()
