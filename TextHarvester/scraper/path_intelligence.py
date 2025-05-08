"""
Path intelligence utilities for smarter web crawling.
Provides link scoring, prioritization, and navigation recommendations
to improve the quality of data collection.
"""

import re
import logging
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# Common patterns that indicate high-value content
CONTENT_INDICATORS = [
    r'article', r'post', r'content', r'blog', r'news', r'publication',
    r'research', r'paper', r'study', r'report', r'analysis', r'review'
]

# Patterns that often indicate low-value pages
LOW_VALUE_INDICATORS = [
    r'login', r'signup', r'cart', r'account', r'privacy', r'terms',
    r'contact', r'about', r'faq', r'help', r'support', r'tag', r'category',
    r'page=\d+', r'sort=', r'filter=', r'search=', r'comment'
]

# URL path segments that typically indicate high-quality hubs
HUB_INDICATORS = [
    r'publications', r'articles', r'blog', r'research', r'resources', 
    r'docs', r'documentation', r'papers', r'journals'
]

class LinkIntelligence:
    """
    Provides intelligent link scoring and prioritization based on multiple factors:
    - Link text relevance
    - URL structure analysis
    - Anchor context assessment
    - Page position heuristics
    """
    
    def __init__(self, base_domain=None, content_keywords=None):
        """
        Initialize the link intelligence engine.
        
        Args:
            base_domain (str): The primary domain being crawled
            content_keywords (list): Optional list of domain-specific content keywords
        """
        self.base_domain = base_domain
        self.content_keywords = content_keywords or []
        
    def score_link(self, link_url, anchor_element=None, source_url=None):
        """
        Score a link based on multiple factors to determine its crawl priority.
        Higher scores indicate links more likely to lead to valuable content.
        
        Args:
            link_url (str): The URL to score
            anchor_element (bs4.Tag): The BeautifulSoup anchor tag containing this link
            source_url (str): The URL of the page containing this link
            
        Returns:
            float: A score from 0.0 (lowest priority) to 1.0 (highest priority)
        """
        scores = []
        
        # Base score starts at 0.5
        base_score = 0.5
        scores.append(base_score)
        
        # 1. URL structure analysis
        url_score = self._analyze_url_structure(link_url, source_url)
        scores.append(url_score)
        
        # 2. Link text relevance (if anchor element is provided)
        if anchor_element is not None:
            text_score = self._analyze_link_text(anchor_element)
            scores.append(text_score)
            
            # 3. Anchor context - look at surrounding text
            context_score = self._analyze_anchor_context(anchor_element)
            scores.append(context_score)
            
            # 4. Link prominence - where in the page it appears
            prominence_score = self._analyze_link_prominence(anchor_element)
            scores.append(prominence_score)
            
        # Calculate weighted average (with more weight on URL and link text)
        # Use different weights if no anchor element is provided
        if anchor_element is not None:
            final_score = (base_score * 0.1 + 
                          url_score * 0.3 + 
                          text_score * 0.3 + 
                          context_score * 0.2 + 
                          prominence_score * 0.1)
        else:
            final_score = (base_score * 0.2 + url_score * 0.8)
            
        # Ensure score is in proper range
        return max(0.0, min(1.0, final_score))
    
    def _analyze_url_structure(self, url, source_url=None):
        """
        Analyze URL structure for content quality indicators.
        
        Args:
            url (str): The URL to analyze
            source_url (str): The URL containing this link
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        score = 0.5  # Neutral starting point
        parsed = urlparse(url)
        path = parsed.path.lower()
        query = parse_qs(parsed.query)
        
        # Penalize query parameters that usually indicate pagination/filtering
        if query:
            query_penalty = min(0.1 * len(query), 0.3)
            score -= query_penalty
            
            # But promote certain query parameters that might indicate valuable content
            if 'id' in query or 'article' in query or 'post' in query:
                score += 0.15
        
        # Check for positive content indicators in path
        for indicator in CONTENT_INDICATORS:
            if re.search(indicator, path):
                score += 0.15
                break
                
        # Check for hub indicators that might lead to content collections
        for hub in HUB_INDICATORS:
            if re.search(hub, path):
                score += 0.2
                break
        
        # Penalize paths that typically have low content value
        for indicator in LOW_VALUE_INDICATORS:
            if re.search(indicator, path):
                score -= 0.2
                break
        
        # File extensions that typically indicate valuable content
        if path.endswith(('.html', '.htm', '.php', '.asp', '.aspx')):
            score += 0.05
        elif path.endswith(('.pdf', '.doc', '.docx')):
            score += 0.2  # Document formats usually contain significant content
            
        # Clean paths with few segments often indicate important pages
        segments = [s for s in path.split('/') if s]
        if 1 <= len(segments) <= 2:
            score += 0.1
        
        # Domain comparison - same domain links often better for cohesive crawling
        if source_url:
            source_domain = urlparse(source_url).netloc
            link_domain = parsed.netloc
            if source_domain == link_domain:
                score += 0.05
        
        # Normalize final score
        return max(0.0, min(1.0, score))
    
    def _analyze_link_text(self, anchor):
        """
        Analyze anchor text for content quality indicators.
        
        Args:
            anchor (bs4.Tag): The BeautifulSoup anchor tag
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        score = 0.5
        
        # Get the link text
        if not anchor.string and not anchor.text:
            return 0.4  # Empty link text is slightly negative
        
        link_text = anchor.text.lower()
        
        # Skip empty or very short link text
        if len(link_text) < 3:
            return 0.4
            
        # Longer link text is often more descriptive and valuable
        text_length = len(link_text)
        if text_length > 20:
            score += 0.15
        elif text_length > 10:
            score += 0.1
            
        # Check for content indicator words in link text
        for indicator in CONTENT_INDICATORS:
            if re.search(indicator, link_text):
                score += 0.2
                break
                
        # Penalize common low-value link text
        for indicator in LOW_VALUE_INDICATORS:
            if re.search(indicator, link_text):
                score -= 0.2
                break
                
        # Reward links with domain keywords
        for keyword in self.content_keywords:
            if keyword.lower() in link_text:
                score += 0.15
                break
                
        # "Read more", "Full article", etc. often lead to content
        read_more_patterns = ['read more', 'full article', 'continue reading', 
                              'view article', 'learn more', 'read article']
        for pattern in read_more_patterns:
            if pattern in link_text:
                score += 0.2
                break
                
        return max(0.0, min(1.0, score))
    
    def _analyze_anchor_context(self, anchor):
        """
        Analyze text around the anchor for content relevance.
        
        Args:
            anchor (bs4.Tag): The BeautifulSoup anchor tag
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        score = 0.5
        
        # Try to get the parent paragraph or div
        parent = anchor.parent
        if not parent:
            return score
            
        # Get siblings and surrounding text
        context_text = ""
        
        # Look at previous sibling
        prev_sib = anchor.previous_sibling
        if prev_sib and isinstance(prev_sib, (str, Tag)):
            if isinstance(prev_sib, str):
                context_text += prev_sib
            else:
                context_text += prev_sib.get_text()
                
        # Look at next sibling
        next_sib = anchor.next_sibling
        if next_sib and isinstance(next_sib, (str, Tag)):
            if isinstance(next_sib, str):
                context_text += next_sib
            else:
                context_text += next_sib.get_text()
                
        # If we have parent text, use that too
        if parent.name in ['p', 'div', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            context_text += parent.get_text()
            
        context_text = context_text.lower()
        
        # Look for content indicators in context
        for indicator in CONTENT_INDICATORS:
            if re.search(indicator, context_text):
                score += 0.1
                break
                
        # Check for timestamps/dates which often indicate content pages
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # 01/30/2023
            r'\d{4}-\d{1,2}-\d{1,2}',    # 2023-01-30
            r'jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec \d{1,2}',  # Jan 30
            r'january|february|march|april|may|june|july|august|september|october|november|december'
        ]
        for pattern in date_patterns:
            if re.search(pattern, context_text):
                score += 0.15
                break
                
        # Look for author mentions
        author_patterns = ['author', 'written by', 'posted by', 'by:']
        for pattern in author_patterns:
            if pattern in context_text:
                score += 0.15
                break
                
        return max(0.0, min(1.0, score))
    
    def _analyze_link_prominence(self, anchor):
        """
        Analyze the link's position in the page structure.
        
        Args:
            anchor (bs4.Tag): The BeautifulSoup anchor tag
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        score = 0.5
        
        # Check if the link is in a headline
        parents = [p.name for p in anchor.parents]
        if any(h in parents for h in ['h1', 'h2', 'h3']):
            score += 0.3
        elif any(h in parents for h in ['h4', 'h5', 'h6']):
            score += 0.2
            
        # Check if link is in main content areas
        content_areas = ['main', 'article', 'section', 'content']
        if any(area in parents for area in content_areas):
            score += 0.2
            
        # Links in navigation are less likely to be content
        nav_areas = ['nav', 'navbar', 'menu', 'sidebar', 'footer']
        if any(area in parents for area in nav_areas) or 'nav' in parents:
            score -= 0.2
            
        # Links with images might be more prominent
        if anchor.find('img'):
            score += 0.1
            
        # Check for list items - sometimes indicates content listings
        if 'li' in parents:
            parent_ul = anchor.find_parent('ul') or anchor.find_parent('ol')
            if parent_ul:
                # Large lists might be content collections
                list_items = parent_ul.find_all('li')
                if len(list_items) > 3 and len(list_items) < 20:
                    score += 0.1
                    
        return max(0.0, min(1.0, score))

def evaluate_page_quality(html_content, url):
    """
    Evaluate the quality and content richness of a page.
    Used for adaptive depth decisions.
    
    Args:
        html_content (str/bytes): The HTML content of the page
        url (str): The URL of the page
        
    Returns:
        tuple: (quality_score, content_metrics)
            - quality_score: Float from 0.0 to 1.0
            - content_metrics: Dict with page characteristics
    """
    if isinstance(html_content, bytes):
        html_content = html_content.decode('utf-8', errors='replace')
        
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements that don't contribute to content
        for s in soup(['script', 'style', 'nav', 'footer']):
            s.decompose()
            
        # Extract main text
        text = soup.get_text(separator=' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Content length metrics
        word_count = len(text.split())
        char_count = len(text)
        
        # Get paragraph count - good indicator of content richness
        paragraphs = soup.find_all('p')
        paragraph_count = len(paragraphs)
        
        # Text-to-HTML ratio (higher is usually better)
        html_length = len(html_content)
        text_ratio = char_count / html_length if html_length > 0 else 0
        
        # Count outbound links (pages with many outbound links often have value)
        links = soup.find_all('a', href=True)
        link_count = len(links)
        
        # Content indicators
        has_article_element = bool(soup.find('article'))
        has_headings = bool(soup.find(['h1', 'h2', 'h3']))
        has_images = len(soup.find_all('img')) > 0
        
        # Calculate overall quality score
        score = 0.0
        
        # Word count scoring
        if word_count > 1000:
            score += 0.3
        elif word_count > 500:
            score += 0.2
        elif word_count > 200:
            score += 0.1
            
        # Paragraph count scoring
        if paragraph_count > 10:
            score += 0.2
        elif paragraph_count > 5:
            score += 0.1
            
        # Text ratio scoring
        if text_ratio > 0.5:
            score += 0.2
        elif text_ratio > 0.25:
            score += 0.1
            
        # Content structure indicators
        if has_article_element:
            score += 0.1
        if has_headings:
            score += 0.1
        if has_images:
            score += 0.1
            
        # Normalize score
        final_score = min(1.0, score)
        
        # Return both score and metrics
        content_metrics = {
            'word_count': word_count,
            'paragraph_count': paragraph_count,
            'text_ratio': text_ratio,
            'link_count': link_count,
            'has_article': has_article_element,
            'has_headings': has_headings,
            'has_images': has_images
        }
        
        return final_score, content_metrics
        
    except Exception as e:
        logger.warning(f"Error evaluating page quality: {e}")
        return 0.3, {'error': str(e)}  # Default to low-medium quality on error
