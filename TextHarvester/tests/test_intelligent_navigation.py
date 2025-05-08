"""
Tests for the intelligent navigation system in the TextHarvester web crawler.
"""
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scraper.crawler import WebCrawler
from scraper.path_intelligence import LinkIntelligence, evaluate_page_quality
from models import ScrapingStatus, ScrapingConfiguration

class TestIntelligentNavigation(unittest.TestCase):
    """Test cases for the intelligent navigation features."""
    
    def setUp(self):
        """Set up test environment with mocked dependencies."""
        # Mock the job and configuration
        self.crawler = WebCrawler(job_id=1)
        self.crawler.job = MagicMock()
        self.crawler.configuration = MagicMock()
        self.crawler.configuration.max_depth = 2
        self.crawler.link_intelligence = LinkIntelligence()
        
        # Set up some test data
        self.crawler.url_parents = {
            'http://example.com/high_quality': 'http://example.com/',
            'http://example.com/low_quality': 'http://example.com/',
            'http://example.com/no_parent': None
        }
        
        self.crawler.page_quality_scores = {
            'http://example.com/': 0.6,
            'http://example.com/high_quality': 0.9,
            'http://example.com/low_quality': 0.3
        }
        
        self.crawler.domain_quality_scores = {
            'example.com': (0.8, 10),  # (score, count)
            'lowquality.com': (0.3, 5)
        }
        
        # Mock HTML content
        self.crawler.content_html_cache = {
            'http://example.com/': '<html><body><a href="/high_quality">High Quality Link</a></body></html>'
        }
    
    @patch('scraper.crawler.get_domain_from_url')
    def test_should_extend_depth_standard_depth(self, mock_get_domain):
        """Test that URLs within standard depth always return True."""
        mock_get_domain.return_value = 'example.com'
        
        # URLs at depth 1 (within max_depth=2) should always return True
        result = self.crawler.should_extend_depth('http://example.com/page', 1)
        self.assertTrue(result, "URLs within standard depth should always proceed")
    
    @patch('scraper.crawler.get_domain_from_url')
    def test_should_extend_depth_exceeds_absolute_max(self, mock_get_domain):
        """Test that URLs exceeding absolute max depth return False."""
        mock_get_domain.return_value = 'example.com'
        
        # URLs beyond absolute max (max_depth + 2) should always return False
        result = self.crawler.should_extend_depth('http://example.com/page', 5)
        self.assertFalse(result, "URLs beyond absolute max depth should not proceed")
    
    @patch('scraper.crawler.get_domain_from_url')
    def test_should_extend_depth_high_quality_parent(self, mock_get_domain):
        """Test that URLs with high-quality parents get extended depth."""
        mock_get_domain.return_value = 'example.com'
        
        # URLs with high-quality parents should extend beyond standard depth
        result = self.crawler.should_extend_depth('http://example.com/high_quality', 3)
        self.assertTrue(result, "URLs with high-quality parents should get extended depth")
    
    @patch('scraper.crawler.get_domain_from_url')
    def test_should_extend_depth_low_quality_parent(self, mock_get_domain):
        """Test that URLs with low-quality parents don't get extended depth."""
        mock_get_domain.return_value = 'example.com'
        
        # URLs with low-quality parents should not extend beyond standard depth
        result = self.crawler.should_extend_depth('http://example.com/low_quality', 3)
        self.assertFalse(result, "URLs with low-quality parents should not get extended depth")
    
    @patch('scraper.crawler.get_domain_from_url')
    def test_should_extend_depth_high_quality_domain(self, mock_get_domain):
        """Test that URLs from high-quality domains get extended depth."""
        mock_get_domain.return_value = 'example.com'
        
        # URLs from high-quality domains should extend beyond standard depth
        result = self.crawler.should_extend_depth('http://example.com/new_page', 3)
        self.assertTrue(result, "URLs from high-quality domains should get extended depth")
    
    @patch('scraper.crawler.get_domain_from_url')
    def test_should_extend_depth_low_quality_domain(self, mock_get_domain):
        """Test that URLs from low-quality domains don't get extended depth."""
        mock_get_domain.return_value = 'lowquality.com'
        
        # URLs from low-quality domains should not extend beyond standard depth
        result = self.crawler.should_extend_depth('http://lowquality.com/page', 3)
        self.assertFalse(result, "URLs from low-quality domains should not get extended depth")
    
    @patch('scraper.path_intelligence.LinkIntelligence.score_link')
    @patch('scraper.crawler.get_domain_from_url')
    def test_should_extend_depth_promising_link(self, mock_get_domain, mock_score_link):
        """Test that promising links get extended depth."""
        mock_get_domain.return_value = 'newdomain.com'
        mock_score_link.return_value = 0.9
        
        # Set up parent info
        self.crawler.url_parents['http://newdomain.com/promising'] = 'http://example.com/'
        
        # High-scoring links should extend beyond standard depth
        result = self.crawler.should_extend_depth('http://newdomain.com/promising', 3)
        self.assertTrue(result, "High-scoring links should get extended depth")
    
    @patch('scraper.path_intelligence.LinkIntelligence.score_link')
    @patch('scraper.crawler.get_domain_from_url')
    def test_should_extend_depth_unpromising_link(self, mock_get_domain, mock_score_link):
        """Test that unpromising links don't get extended depth."""
        mock_get_domain.return_value = 'anotherdomain.com'
        mock_score_link.return_value = 0.5
        
        # Set up parent info
        self.crawler.url_parents['http://anotherdomain.com/unpromising'] = 'http://example.com/'
        
        # Low-scoring links should not extend beyond standard depth
        result = self.crawler.should_extend_depth('http://anotherdomain.com/unpromising', 3)
        self.assertFalse(result, "Low-scoring links should not get extended depth")


if __name__ == '__main__':
    unittest.main()
