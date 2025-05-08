"""
Fast Relevance Filtering System.

This module provides efficient first-pass filtering to quickly identify
content that may be relevant to a given domain, optimized for high throughput.
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional, Union, Any
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from intelligence.utils.text_processing import normalize_text, extract_keywords
from intelligence.classification.topic_taxonomy import TopicTaxonomy, TopicNode

logger = logging.getLogger(__name__)


class FastFilter:
    """
    Fast content filtering system to quickly determine potential relevance.
    
    Provides multiple filtering strategies including keyword matching and TF-IDF
    for rapid early-stage filtering before more computationally expensive
    classification methods are applied.
    """
    
    def __init__(self, taxonomy: Optional[TopicTaxonomy] = None, 
                 threshold: float = 0.3, strategy: str = "tfidf"):
        """
        Initialize the fast filter.
        
        Args:
            taxonomy: Optional taxonomy to extract keywords from
            threshold: Score threshold for relevance (0.0 to 1.0)
            strategy: Filtering strategy to use ('keyword', 'tfidf', or 'hybrid')
        """
        self.threshold = threshold
        self.strategy = strategy
        self.taxonomy = taxonomy
        
        # Internal state
        self._keyword_set = set()
        self._domain_keywords = {}
        self._tfidf_vectorizer = None
        self._tfidf_feature_names = []
        self._domain_vectors = {}
        
        # If taxonomy is provided, extract keywords from it
        if taxonomy:
            self._extract_taxonomy_keywords(taxonomy)
    
    def _extract_taxonomy_keywords(self, taxonomy: TopicTaxonomy) -> None:
        """
        Extract keywords from a taxonomy for filtering.
        
        Args:
            taxonomy: The taxonomy to extract keywords from
        """
        # Get all nodes
        all_nodes = taxonomy.get_all_nodes()
        
        # Extract keywords by domain (using root node names as domains)
        root_nodes = taxonomy.root_nodes
        
        for root_node in root_nodes:
            domain_name = root_node.name.lower()
            self._domain_keywords[domain_name] = set()
            
            # Get all nodes under this root
            nodes_in_domain = [root_node]
            self._get_all_child_nodes(root_node, nodes_in_domain)
            
            # Extract keywords from all nodes in this domain
            for node in nodes_in_domain:
                if node.keywords:
                    # Add original keywords
                    self._domain_keywords[domain_name].update([k.lower() for k in node.keywords])
                    
                    # Add node name and parts of the name
                    name_parts = re.split(r'\s+', node.name.lower())
                    self._domain_keywords[domain_name].update(name_parts)
            
            # Add to global keyword set
            self._keyword_set.update(self._domain_keywords[domain_name])
        
        logger.info(f"Extracted {len(self._keyword_set)} keywords from taxonomy")
        
        # If using TF-IDF, prepare the vectorizer
        if self.strategy in ("tfidf", "hybrid"):
            self._prepare_tfidf_vectorizer()
    
    def _get_all_child_nodes(self, node: TopicNode, result_list: List[TopicNode]) -> None:
        """
        Recursively get all child nodes of a node.
        
        Args:
            node: The parent node
            result_list: List to append child nodes to
        """
        for child in node.children:
            result_list.append(child)
            self._get_all_child_nodes(child, result_list)
    
    def _prepare_tfidf_vectorizer(self) -> None:
        """
        Prepare the TF-IDF vectorizer with domain keywords.
        """
        # Prepare documents for TF-IDF
        documents = []
        domain_docs = {}
        
        for domain, keywords in self._domain_keywords.items():
            # Create a document from the keywords for this domain
            domain_doc = " ".join(sorted(list(keywords)))
            documents.append(domain_doc)
            domain_docs[domain] = domain_doc
        
        # Fit the vectorizer
        self._tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        
        # Fit the vectorizer and get feature names
        self._tfidf_vectorizer.fit(documents)
        self._tfidf_feature_names = self._tfidf_vectorizer.get_feature_names_out()
        
        # Generate domain vectors
        for domain, doc in domain_docs.items():
            domain_vector = self._tfidf_vectorizer.transform([doc])
            self._domain_vectors[domain] = domain_vector
    
    def add_domain_keywords(self, domain: str, keywords: List[str]) -> None:
        """
        Add keywords for a specific domain.
        
        Args:
            domain: Name of the domain
            keywords: List of keywords to add
        """
        if domain not in self._domain_keywords:
            self._domain_keywords[domain] = set()
            
        # Add keywords for this domain
        keywords_lower = [k.lower() for k in keywords]
        self._domain_keywords[domain].update(keywords_lower)
        
        # Update global keyword set
        self._keyword_set.update(keywords_lower)
        
        # If using TF-IDF, re-prepare the vectorizer
        if self.strategy in ("tfidf", "hybrid"):
            self._prepare_tfidf_vectorizer()
            
        logger.info(f"Added {len(keywords)} keywords for domain '{domain}'")
    
    def is_potentially_relevant(self, text: str, domain: Optional[str] = None) -> Tuple[bool, float]:
        """
        Quickly check if text is potentially relevant based on keyword presence.
        
        Args:
            text: Text to check for relevance
            domain: Optional specific domain to check against
            
        Returns:
            Tuple of (is_relevant, confidence)
        """
        # Normalize the text
        normalized_text = normalize_text(text)
        
        # Combine strategies based on configuration
        if self.strategy == "keyword":
            return self._keyword_filter(normalized_text, domain)
        elif self.strategy == "tfidf":
            return self._tfidf_filter(normalized_text, domain)
        elif self.strategy == "hybrid":
            # Combine both strategies - first check keywords, then TF-IDF if needed
            is_relevant, confidence = self._keyword_filter(normalized_text, domain)
            
            # If keyword filter is unsure (close to threshold), use TF-IDF for confirmation
            if 0.2 <= confidence <= 0.4:
                tfidf_relevant, tfidf_confidence = self._tfidf_filter(normalized_text, domain)
                # Average the confidences with higher weight for TF-IDF
                combined_confidence = (confidence + (2 * tfidf_confidence)) / 3
                return combined_confidence >= self.threshold, combined_confidence
            
            return is_relevant, confidence
        else:
            raise ValueError(f"Unknown filtering strategy: {self.strategy}")
    
    def _keyword_filter(self, text: str, domain: Optional[str] = None) -> Tuple[bool, float]:
        """
        Filter based on keyword presence.
        
        Args:
            text: Text to filter
            domain: Optional specific domain
            
        Returns:
            Tuple of (is_relevant, confidence)
        """
        # Get the keywords to use
        if domain and domain.lower() in self._domain_keywords:
            keywords = self._domain_keywords[domain.lower()]
        else:
            keywords = self._keyword_set
        
        if not keywords:
            logger.warning("No keywords available for filtering")
            return False, 0.0
        
        # Extract potential keywords from the text
        text_keywords = extract_keywords(text)
        
        # Count keyword matches
        matches = set(text_keywords).intersection(keywords)
        match_count = len(matches)
        
        # Calculate confidence based on matches and text length
        # More matches = higher confidence
        word_count = len(text.split())
        min_words = 50  # Minimum words for full confidence calculation
        
        if word_count < min_words:
            # Short text gets lower max confidence
            max_confidence = 0.7 * (word_count / min_words)
        else:
            max_confidence = 0.7  # Max confidence for keyword matching
            
        # Calculate confidence - more matches give higher confidence
        # Using logarithmic scale to avoid diminishing returns
        if match_count > 0:
            # Log base 10 of (1 + match_count) scaled to max_confidence
            # This gives a curve that rises quickly for early matches
            confidence = max_confidence * min(1.0, np.log10(1 + match_count) / np.log10(10))
        else:
            confidence = 0.0
            
        return confidence >= self.threshold, confidence
    
    def _tfidf_filter(self, text: str, domain: Optional[str] = None) -> Tuple[bool, float]:
        """
        Filter based on TF-IDF similarity.
        
        Args:
            text: Text to filter
            domain: Optional specific domain
            
        Returns:
            Tuple of (is_relevant, confidence)
        """
        if not self._tfidf_vectorizer:
            logger.warning("TF-IDF vectorizer not prepared, falling back to keyword filter")
            return self._keyword_filter(text, domain)
        
        # Transform the input text
        text_vector = self._tfidf_vectorizer.transform([text])
        
        # If domain specified, calculate similarity to that domain
        if domain and domain.lower() in self._domain_vectors:
            domain_vector = self._domain_vectors[domain.lower()]
            similarity = self._cosine_similarity(text_vector, domain_vector)
            confidence = float(similarity)
        else:
            # Calculate similarity to all domains and take the max
            similarities = []
            for domain_name, domain_vector in self._domain_vectors.items():
                similarity = self._cosine_similarity(text_vector, domain_vector)
                similarities.append(similarity)
            
            if similarities:
                confidence = float(max(similarities))
            else:
                confidence = 0.0
        
        return confidence >= self.threshold, confidence
    
    @staticmethod
    def _cosine_similarity(vec1, vec2) -> float:
        """
        Calculate cosine similarity between two sparse vectors.
        
        Args:
            vec1: First sparse vector
            vec2: Second sparse vector
            
        Returns:
            Cosine similarity value
        """
        # Convert to arrays for dot product
        vec1_array = vec1.toarray().flatten()
        vec2_array = vec2.toarray().flatten()
        
        # Calculate dot product
        dot_product = np.dot(vec1_array, vec2_array)
        
        # Calculate norms
        norm1 = np.linalg.norm(vec1_array)
        norm2 = np.linalg.norm(vec2_array)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        # Return cosine similarity
        return dot_product / (norm1 * norm2)
    
    def get_matching_keywords(self, text: str, domain: Optional[str] = None, 
                              max_keywords: int = 10) -> List[str]:
        """
        Get the keywords that match the text for a given domain.
        
        Args:
            text: Text to find matching keywords in
            domain: Optional specific domain
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of matching keywords
        """
        # Get the keywords to use
        if domain and domain.lower() in self._domain_keywords:
            keywords = self._domain_keywords[domain.lower()]
        else:
            keywords = self._keyword_set
        
        if not keywords:
            return []
        
        # Extract potential keywords from the text
        text_keywords = extract_keywords(text)
        
        # Find matching keywords
        matches = sorted(list(set(text_keywords).intersection(keywords)))
        
        # Return up to max_keywords
        return matches[:max_keywords]
    
    @staticmethod
    def create_from_football_taxonomy() -> 'FastFilter':
        """
        Create a FastFilter instance specifically for football filtering.
        
        Returns:
            Configured FastFilter for football content
        """
        from intelligence.classification.taxonomies.football import get_premier_league_taxonomy
        
        # Get the football taxonomy
        football_taxonomy = get_premier_league_taxonomy()
        
        # Create filter with hybrid strategy
        filter_instance = FastFilter(
            taxonomy=football_taxonomy,
            threshold=0.25,  # Lower threshold for higher recall
            strategy="hybrid"
        )
        
        # Add additional football-specific keywords that might not be in the taxonomy
        additional_keywords = [
            "goal", "score", "stadium", "referee", "season", "transfer", 
            "premier", "league", "football", "soccer", "match", "fixture",
            "standing", "table", "points", "lineup", "formation", "assist",
            "clean sheet", "injury", "substitution", "card", "offside", "var",
            "coach", "manager", "halftime", "fulltime", "win", "lose", "draw"
        ]
        filter_instance.add_domain_keywords("football", additional_keywords)
        
        # Add keywords to disambiguate from American football
        filter_instance.add_domain_keywords("football_disambiguation", [
            "pitch", "goalkeeper", "striker", "centre-back", "offside",
            "premier league", "fa cup", "champions league", "relegation",
            "manchester", "liverpool", "chelsea", "arsenal", "tottenham",
            "booking", "yellow card", "red card", "corner", "free kick"
        ])
        
        return filter_instance
