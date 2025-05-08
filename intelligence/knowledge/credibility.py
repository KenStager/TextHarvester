"""
Source Credibility Scoring

This module provides the CredibilityScorer class for evaluating and managing
the credibility of content sources to weight information appropriately.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime, timedelta
from urllib.parse import urlparse
import re

from app import db
from db.models.knowledge_base import (
    SourceCredibility, KnowledgeSource, KnowledgeNode, KnowledgeEdge
)

logger = logging.getLogger(__name__)


class CredibilityScorer:
    """
    Evaluates and manages the credibility of content sources
    to weight information based on source reliability.
    """
    
    def __init__(self, domain: str = None):
        """
        Initialize the credibility scorer.
        
        Args:
            domain: Optional domain to filter operations by (e.g., 'football')
        """
        self.domain = domain
        logger.info(f"Initialized credibility scorer for domain: {domain}")
    
    def evaluate_source(self, 
                       source_url: str, 
                       force_update: bool = False) -> Dict[str, Any]:
        """
        Evaluate or retrieve the credibility of a source.
        
        Args:
            source_url: URL of the source to evaluate
            force_update: Whether to force re-evaluation if a score exists
            
        Returns:
            Dictionary with credibility scores
        """
        # Normalize the URL
        normalized_url = self._normalize_url(source_url)
        
        # Check if source has been evaluated recently
        existing = SourceCredibility.query.filter_by(source_url=normalized_url).first()
        
        if existing and not force_update:
            # Check if evaluation is recent enough
            if existing.updated_at > datetime.utcnow() - timedelta(days=30):
                return {
                    "source_url": existing.source_url,
                    "domain": existing.domain,
                    "overall_score": existing.overall_score,
                    "domain_expertise": existing.domain_expertise,
                    "accuracy_score": existing.accuracy_score,
                    "bias_score": existing.bias_score,
                    "transparency_score": existing.transparency_score,
                    "consistency_score": existing.consistency_score,
                    "evaluation_count": existing.evaluation_count,
                    "last_evaluated": existing.last_evaluated.isoformat() if existing.last_evaluated else None
                }
        
        # Extract the domain from the URL
        parsed_url = urlparse(normalized_url)
        domain = parsed_url.netloc
        
        # Perform credibility evaluation
        evaluation = self._evaluate_source_credibility(normalized_url, domain)
        
        # Update or create credibility record
        if existing:
            # Update existing record
            existing.overall_score = evaluation["overall_score"]
            existing.domain_expertise = evaluation["domain_expertise"]
            existing.accuracy_score = evaluation["accuracy_score"]
            existing.bias_score = evaluation["bias_score"]
            existing.transparency_score = evaluation["transparency_score"]
            existing.consistency_score = evaluation["consistency_score"]
            existing.updated_at = datetime.utcnow()
            existing.last_evaluated = datetime.utcnow()
            existing.evaluation_count += 1
            
            db.session.commit()
            
            logger.info(f"Updated credibility for {normalized_url}: {evaluation['overall_score']}")
            
        else:
            # Create new record
            credibility = SourceCredibility(
                source_url=normalized_url,
                domain=domain,
                overall_score=evaluation["overall_score"],
                domain_expertise=evaluation["domain_expertise"],
                accuracy_score=evaluation["accuracy_score"],
                bias_score=evaluation["bias_score"],
                transparency_score=evaluation["transparency_score"],
                consistency_score=evaluation["consistency_score"],
                last_evaluated=datetime.utcnow(),
                evaluation_count=1
            )
            
            db.session.add(credibility)
            db.session.commit()
            
            logger.info(f"Created credibility for {normalized_url}: {evaluation['overall_score']}")
        
        return evaluation
    
    def _normalize_url(self, url: str) -> str:
        """
        Normalize a URL for consistent storage and comparison.
        
        Args:
            url: The URL to normalize
            
        Returns:
            Normalized URL
        """
        # Parse the URL
        parsed = urlparse(url)
        
        # Ensure scheme is present
        if not parsed.scheme:
            parsed = urlparse(f"https://{url}")
        
        # Remove common tracking parameters
        query_params = []
        if parsed.query:
            skip_params = {'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
                          'fbclid', 'gclid', 'msclkid', 'ref', 'source', 'ref_src'}
            
            for param in parsed.query.split('&'):
                if '=' in param:
                    name, value = param.split('=', 1)
                    if name.lower() not in skip_params:
                        query_params.append(f"{name}={value}")
                else:
                    query_params.append(param)
        
        # Reconstruct the URL
        netloc = parsed.netloc.lower()
        path = parsed.path
        
        # Remove trailing slash from path if present
        if path.endswith('/') and len(path) > 1:
            path = path[:-1]
        
        # Remove www. prefix if present
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        
        # Reconstruct query string if any params remain
        query = '&'.join(query_params) if query_params else ''
        
        # Build the normalized URL
        normalized = f"{parsed.scheme}://{netloc}{path}"
        if query:
            normalized += f"?{query}"
        
        return normalized
    
    def _evaluate_source_credibility(self, url: str, domain: str) -> Dict[str, Any]:
        """
        Evaluate the credibility of a source based on multiple factors.
        
        Args:
            url: The normalized URL
            domain: The domain of the URL
            
        Returns:
            Dictionary with credibility scores
        """
        # This would involve multiple evaluation techniques
        # For the implementation, we'll use a simplified approach based on:
        # 1. Domain reputation (could be pre-configured or learned)
        # 2. Previous accuracy (based on verification of information from this source)
        # 3. Consistency (how consistent information from this source has been)
        # 4. Domain expertise (how knowledgeable the source is in specific domains)
        
        # In a production system, these would be based on actual evaluation
        # For now, we'll use some default values with slight randomization
        # In the football domain, we might have pre-configured values for known sources
        
        domain_expertise = {}
        accuracy_score = 0.7  # Default moderate accuracy
        bias_score = 0.5      # Default neutral bias
        transparency_score = 0.6  # Default moderate transparency
        consistency_score = 0.7  # Default good consistency
        
        # Domain-specific settings
        if self.domain == "football":
            # Domain expertise for football sources
            if any(football_site in domain for football_site in 
                  ["fifa.com", "uefa.com", "premierleague.com", "thefa.com"]):
                # Official sources get high scores
                domain_expertise = {"football": 0.95}
                accuracy_score = 0.9
                transparency_score = 0.85
            
            elif any(sports_site in domain for sports_site in 
                    ["espn.com", "skysports.com", "goal.com", "bbc.co.uk/sport"]):
                # Major sports sites get good scores
                domain_expertise = {"football": 0.85, "sports": 0.8}
                accuracy_score = 0.8
                transparency_score = 0.75
            
            elif any(news_site in domain for news_site in 
                    ["guardian.co.uk", "telegraph.co.uk", "nytimes.com"]):
                # General news with sports sections
                domain_expertise = {"football": 0.7, "general_news": 0.8}
                accuracy_score = 0.8
                transparency_score = 0.8
            
            elif any(blog_site in domain for blog_site in 
                    ["wordpress.com", "blogspot.com", "medium.com"]):
                # Blogs get lower initial credibility
                domain_expertise = {"football": 0.5}
                accuracy_score = 0.6
                transparency_score = 0.5
            
            else:
                # Unknown sources get average scores until evaluated
                domain_expertise = {"football": 0.6}
        
        # Calculate overall score
        # Weight factors: accuracy (0.4), domain expertise (0.3), consistency (0.2), transparency (0.1)
        domain_expertise_score = domain_expertise.get(self.domain, 0.5) if self.domain else 0.5
        
        overall_score = (
            accuracy_score * 0.4 +
            domain_expertise_score * 0.3 +
            consistency_score * 0.2 +
            transparency_score * 0.1
        )
        
        return {
            "source_url": url,
            "domain": domain,
            "overall_score": overall_score,
            "domain_expertise": domain_expertise,
            "accuracy_score": accuracy_score,
            "bias_score": bias_score,
            "transparency_score": transparency_score,
            "consistency_score": consistency_score
        }
    
    def evaluate_content_sources(self, content_id: int) -> Dict[str, Any]:
        """
        Evaluate all sources referenced in a content item.
        
        Args:
            content_id: ID of the content to evaluate sources for
            
        Returns:
            Dictionary with evaluation results
        """
        # Get all sources for this content
        sources = KnowledgeSource.query.filter_by(content_id=content_id).all()
        
        results = {
            "content_id": content_id,
            "source_count": len(sources),
            "evaluations": []
        }
        
        # Get the content source URL from the scraper database
        # This requires accessing the scraper content model
        from db.models.scraper import ScraperContent
        content = ScraperContent.query.get(content_id)
        
        if content and content.url:
            # Evaluate the main content source
            evaluation = self.evaluate_source(content.url)
            results["primary_source_evaluation"] = evaluation
        
        return results
    
    def recalculate_confidence_with_credibility(self, node_id: int) -> Dict[str, Any]:
        """
        Recalculate a node's confidence based on source credibility.
        
        Args:
            node_id: ID of the node to recalculate confidence for
            
        Returns:
            Dictionary with recalculation results
        """
        node = KnowledgeNode.query.get(node_id)
        if not node:
            logger.warning(f"Node not found for confidence recalculation: {node_id}")
            return {
                "status": "error",
                "message": "Node not found"
            }
        
        # Get all sources for this node
        sources = KnowledgeSource.query.filter_by(node_id=node_id).all()
        
        if not sources:
            return {
                "status": "info",
                "message": "No sources found for this node",
                "node_id": node_id,
                "confidence": node.confidence
            }
        
        # Calculate weighted confidence
        total_weighted_confidence = 0
        total_weight = 0
        
        for source in sources:
            # Get the content for this source
            from db.models.scraper import ScraperContent
            content = ScraperContent.query.get(source.content_id)
            
            if not content or not content.url:
                # Use source's confidence as is
                total_weighted_confidence += source.confidence
                total_weight += 1
                continue
            
            # Get source credibility
            credibility = SourceCredibility.query.filter_by(source_url=self._normalize_url(content.url)).first()
            
            if credibility:
                # Weight the source confidence by credibility
                source_weight = credibility.overall_score
                total_weighted_confidence += source.confidence * source_weight
                total_weight += source_weight
            else:
                # No credibility information, use source's confidence as is
                total_weighted_confidence += source.confidence
                total_weight += 1
        
        # Calculate final confidence
        if total_weight > 0:
            new_confidence = total_weighted_confidence / total_weight
        else:
            new_confidence = node.confidence
        
        # Update node confidence
        old_confidence = node.confidence
        node.confidence = new_confidence
        db.session.commit()
        
        logger.info(f"Recalculated confidence for node {node_id}: {old_confidence} -> {new_confidence}")
        
        return {
            "status": "success",
            "node_id": node_id,
            "old_confidence": old_confidence,
            "new_confidence": new_confidence,
            "source_count": len(sources)
        }
    
    def recalculate_edge_confidence(self, edge_id: int) -> Dict[str, Any]:
        """
        Recalculate an edge's confidence based on source credibility.
        
        Args:
            edge_id: ID of the edge to recalculate confidence for
            
        Returns:
            Dictionary with recalculation results
        """
        edge = KnowledgeEdge.query.get(edge_id)
        if not edge:
            logger.warning(f"Edge not found for confidence recalculation: {edge_id}")
            return {
                "status": "error",
                "message": "Edge not found"
            }
        
        # Get all sources for this edge
        sources = KnowledgeSource.query.filter_by(edge_id=edge_id).all()
        
        if not sources:
            return {
                "status": "info",
                "message": "No sources found for this edge",
                "edge_id": edge_id,
                "confidence": edge.confidence
            }
        
        # Calculate weighted confidence
        total_weighted_confidence = 0
        total_weight = 0
        
        for source in sources:
            # Get the content for this source
            from db.models.scraper import ScraperContent
            content = ScraperContent.query.get(source.content_id)
            
            if not content or not content.url:
                # Use source's confidence as is
                total_weighted_confidence += source.confidence
                total_weight += 1
                continue
            
            # Get source credibility
            credibility = SourceCredibility.query.filter_by(source_url=self._normalize_url(content.url)).first()
            
            if credibility:
                # Weight the source confidence by credibility
                source_weight = credibility.overall_score
                total_weighted_confidence += source.confidence * source_weight
                total_weight += source_weight
            else:
                # No credibility information, use source's confidence as is
                total_weighted_confidence += source.confidence
                total_weight += 1
        
        # Calculate final confidence
        if total_weight > 0:
            new_confidence = total_weighted_confidence / total_weight
        else:
            new_confidence = edge.confidence
        
        # Update edge confidence
        old_confidence = edge.confidence
        edge.confidence = new_confidence
        db.session.commit()
        
        logger.info(f"Recalculated confidence for edge {edge_id}: {old_confidence} -> {new_confidence}")
        
        return {
            "status": "success",
            "edge_id": edge_id,
            "old_confidence": old_confidence,
            "new_confidence": new_confidence,
            "source_count": len(sources)
        }
    
    def get_source_ranking(self, topic: str = None, limit: int = 10) -> List[Dict]:
        """
        Get a ranking of sources by credibility.
        
        Args:
            topic: Optional topic to filter by
            limit: Maximum number of results
            
        Returns:
            List of sources with credibility scores
        """
        query = SourceCredibility.query
        
        if topic and self.domain:
            # If we're looking for a specific topic within the domain
            # Filter sources with expertise in both the domain and topic
            query = query.filter(
                SourceCredibility.domain_expertise.contains({self.domain: True})
            )
        
        # Order by overall score descending
        query = query.order_by(SourceCredibility.overall_score.desc())
        
        sources = query.limit(limit).all()
        
        result = []
        for source in sources:
            result.append({
                "source_url": source.source_url,
                "domain": source.domain,
                "overall_score": source.overall_score,
                "domain_expertise": source.domain_expertise
            })
        
        return result
