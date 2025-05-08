"""
Contradiction Detection and Resolution

This module provides the ContradictionDetector class for identifying and managing
contradictions in the knowledge graph, enabling conflict resolution and uncertainty handling.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime

from app import db
from db.models.knowledge_base import (
    KnowledgeNode, KnowledgeEdge, KnowledgeSource, KnowledgeContradiction,
    ClaimNode, EntityNode, SourceCredibility
)

logger = logging.getLogger(__name__)


class ContradictionDetector:
    """
    Identifies and manages contradictions in the knowledge graph,
    enabling conflict resolution and uncertainty representation.
    """
    
    def __init__(self, domain: str = None):
        """
        Initialize the contradiction detector.
        
        Args:
            domain: Optional domain to filter operations by (e.g., 'football')
        """
        self.domain = domain
        logger.info(f"Initialized contradiction detector for domain: {domain}")
    
    def detect_claim_contradictions(self, 
                                   new_claim_id: int = None,
                                   detect_all: bool = False) -> List[Dict]:
        """
        Detect contradictions between claims in the knowledge graph.
        
        Args:
            new_claim_id: Optional ID of a new claim to check against existing claims
            detect_all: Whether to detect all contradictions (resource-intensive)
            
        Returns:
            List of dictionaries with contradiction information
        """
        contradictions = []
        
        # If checking a specific claim
        if new_claim_id and not detect_all:
            new_claim = ClaimNode.query.get(new_claim_id)
            if not new_claim:
                logger.warning(f"Claim not found for contradiction detection: {new_claim_id}")
                return []
            
            # Get existing claims that might contradict the new claim
            potential_contradictions = self._find_potential_claim_contradictions(new_claim)
            
            # Check each potential contradiction
            for existing_claim in potential_contradictions:
                # Skip self-comparison
                if existing_claim.id == new_claim_id:
                    continue
                
                # Determine if the claims contradict each other
                contradiction_result = self._check_claim_contradiction(new_claim, existing_claim)
                
                if contradiction_result["contradicts"]:
                    # Record the contradiction
                    contradiction = self._record_contradiction(
                        primary_node_id=new_claim.id,
                        contradicting_node_id=existing_claim.id,
                        contradiction_type=contradiction_result["type"],
                        description=contradiction_result["description"]
                    )
                    
                    if contradiction:
                        contradictions.append({
                            "contradiction_id": contradiction.id,
                            "primary_node_id": new_claim.id,
                            "contradicting_node_id": existing_claim.id,
                            "contradiction_type": contradiction_result["type"],
                            "description": contradiction_result["description"]
                        })
        
        # If detecting all contradictions
        elif detect_all:
            # This is potentially very resource-intensive
            # Get all claims in the domain
            claim_query = ClaimNode.query
            if self.domain:
                claim_query = claim_query.filter(ClaimNode.domain == self.domain)
            
            claims = claim_query.all()
            
            # Compare each claim with every other claim
            for i, claim1 in enumerate(claims[:-1]):
                for claim2 in claims[i+1:]:
                    # Check if the claims contradict each other
                    contradiction_result = self._check_claim_contradiction(claim1, claim2)
                    
                    if contradiction_result["contradicts"]:
                        # Check if this contradiction already exists
                        existing = KnowledgeContradiction.query.filter(
                            ((KnowledgeContradiction.primary_node_id == claim1.id) &
                             (KnowledgeContradiction.contradicting_node_id == claim2.id)) |
                            ((KnowledgeContradiction.primary_node_id == claim2.id) &
                             (KnowledgeContradiction.contradicting_node_id == claim1.id))
                        ).first()
                        
                        if not existing:
                            # Record the contradiction
                            # Use the more confident claim as primary
                            if (claim1.confidence or 0) >= (claim2.confidence or 0):
                                primary = claim1
                                contradicting = claim2
                            else:
                                primary = claim2
                                contradicting = claim1
                            
                            contradiction = self._record_contradiction(
                                primary_node_id=primary.id,
                                contradicting_node_id=contradicting.id,
                                contradiction_type=contradiction_result["type"],
                                description=contradiction_result["description"]
                            )
                            
                            if contradiction:
                                contradictions.append({
                                    "contradiction_id": contradiction.id,
                                    "primary_node_id": primary.id,
                                    "contradicting_node_id": contradicting.id,
                                    "contradiction_type": contradiction_result["type"],
                                    "description": contradiction_result["description"]
                                })
        
        return contradictions
    
    def _find_potential_claim_contradictions(self, claim: ClaimNode) -> List[ClaimNode]:
        """
        Find potential contradictions for a claim based on content similarity
        and entity references.
        
        Args:
            claim: The claim to find potential contradictions for
            
        Returns:
            List of potentially contradicting claims
        """
        # This could use more sophisticated NLP techniques for better detection
        # Here we're using a simple approach: find claims with similar entities
        
        # Get entities mentioned in the claim
        claim_entities = set()
        for edge in claim.outgoing_edges:
            if edge.relationship_type == "mentions" and isinstance(edge.target, EntityNode):
                claim_entities.add(edge.target.id)
        
        if not claim_entities:
            # No entities to match on, use text similarity instead
            # This is a basic implementation - could be enhanced with embedding similarity
            return ClaimNode.query.filter(
                ClaimNode.id != claim.id,
                ClaimNode.domain == claim.domain,
                ClaimNode.content.ilike(f"%{claim.content[:30]}%")
            ).all()
        
        # Find claims that mention the same entities
        potential_contradictions = []
        entity_mentions = []
        
        for entity_id in claim_entities:
            # Find edges mentioning this entity
            mention_edges = KnowledgeEdge.query.filter_by(
                relationship_type="mentions",
                target_id=entity_id
            ).all()
            
            # Get the source claim nodes
            for edge in mention_edges:
                source_claim = ClaimNode.query.get(edge.source_id)
                if source_claim and source_claim.id != claim.id:
                    entity_mentions.append(source_claim)
        
        # Remove duplicates
        seen_ids = set()
        for claim in entity_mentions:
            if claim.id not in seen_ids:
                potential_contradictions.append(claim)
                seen_ids.add(claim.id)
        
        return potential_contradictions
    
    def _check_claim_contradiction(self, 
                                  claim1: ClaimNode, 
                                  claim2: ClaimNode) -> Dict[str, Any]:
        """
        Check if two claims contradict each other.
        
        Args:
            claim1: First claim
            claim2: Second claim
            
        Returns:
            Dictionary with contradiction results
        """
        # This is a simplified implementation
        # More sophisticated contradiction detection would use NLP and logical analysis
        
        # For now, we'll use some simple heuristics:
        # 1. Check for sentiment opposition on the same topic
        # 2. Check for direct negation patterns
        # 3. Check for temporal conflicts
        
        result = {
            "contradicts": False,
            "type": None,
            "description": None
        }
        
        # Check sentiment contradiction (if both have sentiment values)
        if claim1.sentiment is not None and claim2.sentiment is not None:
            # Check for significant sentiment difference on same entities
            sentiment_diff = abs(claim1.sentiment - claim2.sentiment)
            if sentiment_diff > 1.0:  # Significant difference
                # Check if they're about the same entities
                claim1_entities = {edge.target_id for edge in claim1.outgoing_edges if edge.relationship_type == "mentions"}
                claim2_entities = {edge.target_id for edge in claim2.outgoing_edges if edge.relationship_type == "mentions"}
                
                common_entities = claim1_entities.intersection(claim2_entities)
                if common_entities:
                    result["contradicts"] = True
                    result["type"] = "sentiment_conflict"
                    result["description"] = f"Opposing sentiments ({claim1.sentiment:.1f} vs {claim2.sentiment:.1f}) about the same entities"
                    return result
        
        # Check for direct negation patterns
        # This is a very basic approach - could be improved with NLP
        negation_markers = ["not", "n't", "never", "no", "disagree", "incorrect", "false", "untrue"]
        
        # Check if one claim directly negates the other
        for marker in negation_markers:
            if marker in claim1.content.lower() and claim2.content.lower() in claim1.content.lower():
                result["contradicts"] = True
                result["type"] = "direct_negation"
                result["description"] = f"Claim 1 directly negates Claim 2 using '{marker}'"
                return result
            
            if marker in claim2.content.lower() and claim1.content.lower() in claim2.content.lower():
                result["contradicts"] = True
                result["type"] = "direct_negation"
                result["description"] = f"Claim 2 directly negates Claim 1 using '{marker}'"
                return result
        
        # Check content for factual conflicts through substring matching
        # This is a very simplistic approach - in a real implementation,
        # we would use semantic analysis, NLI, or similar techniques
        if claim1.claim_type == "factual" and claim2.claim_type == "factual":
            # Very simple overlap detection
            if len(claim1.content) > 20 and len(claim2.content) > 20:
                claim1_words = set(claim1.content.lower().split())
                claim2_words = set(claim2.content.lower().split())
                
                common_words = claim1_words.intersection(claim2_words)
                
                # If they share a significant portion of words but are not too similar
                if len(common_words) > min(len(claim1_words), len(claim2_words)) * 0.5:
                    # Check for potential opposing statements
                    # This is very simplified - real implementation would need deeper analysis
                    for opposing_pair in [("win", "lose"), ("victory", "defeat"), ("true", "false"),
                                         ("confirm", "deny"), ("agree", "disagree")]:
                        if (opposing_pair[0] in claim1.content.lower() and opposing_pair[1] in claim2.content.lower()) or \
                           (opposing_pair[1] in claim1.content.lower() and opposing_pair[0] in claim2.content.lower()):
                            result["contradicts"] = True
                            result["type"] = "factual_conflict"
                            result["description"] = f"Opposing factual claims with contradicting terms ({opposing_pair[0]}/{opposing_pair[1]})"
                            return result
        
        return result
    
    def detect_entity_contradictions(self, entity_id: int = None) -> List[Dict]:
        """
        Detect contradictions in entity attributes or relationships.
        
        Args:
            entity_id: Optional ID of an entity to check
            
        Returns:
            List of dictionaries with contradiction information
        """
        contradictions = []
        
        # Get entity node
        if entity_id:
            entity_node = EntityNode.query.get(entity_id)
            if not entity_node:
                logger.warning(f"Entity not found for contradiction detection: {entity_id}")
                return []
            
            # Check for attribute contradictions
            attribute_contradictions = self._check_entity_attribute_contradictions(entity_node)
            contradictions.extend(attribute_contradictions)
            
            # Check for relationship contradictions
            relationship_contradictions = self._check_entity_relationship_contradictions(entity_node)
            contradictions.extend(relationship_contradictions)
        
        return contradictions
    
    def _check_entity_attribute_contradictions(self, entity_node: EntityNode) -> List[Dict]:
        """
        Check for contradictions in entity attributes based on different sources.
        
        Args:
            entity_node: The entity node to check
            
        Returns:
            List of dictionaries with contradiction information
        """
        # This would require analysis of attribute provenance
        # For now, we'll return an empty list as this is highly domain-specific
        return []
    
    def _check_entity_relationship_contradictions(self, entity_node: EntityNode) -> List[Dict]:
        """
        Check for contradictions in entity relationships.
        
        Args:
            entity_node: The entity node to check
            
        Returns:
            List of dictionaries with contradiction information
        """
        contradictions = []
        
        # Check for mutually exclusive relationships
        # This is domain-specific, but here's an example for football
        if self.domain == "football":
            # Check for contradicting team memberships
            team_edges = []
            for edge in entity_node.outgoing_edges:
                if edge.relationship_type == "plays_for":
                    team_edges.append(edge)
            
            # If a player has multiple active team relationships
            if len(team_edges) > 1:
                # Check if they overlap in time
                for i, edge1 in enumerate(team_edges[:-1]):
                    for edge2 in team_edges[i+1:]:
                        overlap = self._check_temporal_overlap(edge1, edge2)
                        
                        if overlap["overlaps"]:
                            # Football players typically don't play for multiple teams simultaneously
                            # Record contradiction
                            contradiction = self._record_contradiction(
                                primary_node_id=entity_node.id,
                                contradicting_node_id=entity_node.id,  # Self-contradiction in relationships
                                contradiction_type="relationship_conflict",
                                description=f"Entity has conflicting 'plays_for' relationships to multiple teams at the same time"
                            )
                            
                            if contradiction:
                                contradictions.append({
                                    "contradiction_id": contradiction.id,
                                    "primary_node_id": entity_node.id,
                                    "contradicting_node_id": entity_node.id,
                                    "contradiction_type": "relationship_conflict",
                                    "description": f"Entity has conflicting 'plays_for' relationships to multiple teams at the same time",
                                    "relationships": [edge1.id, edge2.id]
                                })
        
        return contradictions
    
    def _check_temporal_overlap(self, edge1: KnowledgeEdge, edge2: KnowledgeEdge) -> Dict[str, Any]:
        """
        Check if two temporal edges overlap in time.
        
        Args:
            edge1: First edge with temporal attributes
            edge2: Second edge with temporal attributes
            
        Returns:
            Dictionary with overlap results
        """
        result = {
            "overlaps": False,
            "start": None,
            "end": None
        }
        
        # If either edge has no temporal information, assume they overlap
        if not edge1.valid_from and not edge1.valid_to and not edge2.valid_from and not edge2.valid_to:
            result["overlaps"] = True
            return result
        
        # Check for overlap
        edge1_start = edge1.valid_from or datetime.min
        edge1_end = edge1.valid_to or datetime.max
        
        edge2_start = edge2.valid_from or datetime.min
        edge2_end = edge2.valid_to or datetime.max
        
        # Check if one range contains the other
        if edge1_start <= edge2_start <= edge1_end:
            result["overlaps"] = True
            result["start"] = edge2_start
            result["end"] = min(edge1_end, edge2_end)
        elif edge2_start <= edge1_start <= edge2_end:
            result["overlaps"] = True
            result["start"] = edge1_start
            result["end"] = min(edge1_end, edge2_end)
        
        return result
    
    def _record_contradiction(self, 
                             primary_node_id: int,
                             contradicting_node_id: int,
                             contradiction_type: str,
                             description: str = None) -> Optional[KnowledgeContradiction]:
        """
        Record a contradiction in the database.
        
        Args:
            primary_node_id: ID of the primary node
            contradicting_node_id: ID of the contradicting node
            contradiction_type: Type of contradiction
            description: Optional description of the contradiction
            
        Returns:
            The created contradiction record or None if it already exists
        """
        # Check if this contradiction already exists
        existing = KnowledgeContradiction.query.filter_by(
            primary_node_id=primary_node_id,
            contradicting_node_id=contradicting_node_id,
            contradiction_type=contradiction_type
        ).first()
        
        if existing:
            logger.info(f"Contradiction already exists: {existing.id}")
            return existing
        
        # Create new contradiction
        contradiction = KnowledgeContradiction(
            primary_node_id=primary_node_id,
            contradicting_node_id=contradicting_node_id,
            contradiction_type=contradiction_type,
            description=description,
            resolution_status="unresolved"
        )
        
        db.session.add(contradiction)
        db.session.commit()
        
        logger.info(f"Recorded contradiction: {contradiction.id} - {contradiction_type}")
        return contradiction
    
    def resolve_contradiction(self, 
                             contradiction_id: int,
                             resolution_status: str,
                             resolution_notes: str = None) -> Optional[KnowledgeContradiction]:
        """
        Resolve a contradiction with the specified status.
        
        Args:
            contradiction_id: ID of the contradiction to resolve
            resolution_status: Resolution status (resolved_primary, resolved_contradicting, resolved_both_valid)
            resolution_notes: Optional notes about the resolution
            
        Returns:
            The updated contradiction record or None if not found
        """
        contradiction = KnowledgeContradiction.query.get(contradiction_id)
        if not contradiction:
            logger.warning(f"Contradiction not found for resolution: {contradiction_id}")
            return None
        
        valid_statuses = ["resolved_primary", "resolved_contradicting", "resolved_both_valid"]
        if resolution_status not in valid_statuses:
            logger.warning(f"Invalid resolution status: {resolution_status}")
            return None
        
        # Update contradiction
        contradiction.resolution_status = resolution_status
        contradiction.resolution_notes = resolution_notes
        contradiction.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        logger.info(f"Resolved contradiction {contradiction_id} as {resolution_status}")
        return contradiction
    
    def get_unresolved_contradictions(self, 
                                     contradiction_type: str = None,
                                     limit: int = 100) -> List[KnowledgeContradiction]:
        """
        Get unresolved contradictions for review.
        
        Args:
            contradiction_type: Optional type of contradictions to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of unresolved contradictions
        """
        query = KnowledgeContradiction.query.filter_by(resolution_status="unresolved")
        
        if contradiction_type:
            query = query.filter_by(contradiction_type=contradiction_type)
        
        # If domain is specified, filter primary nodes by domain
        if self.domain:
            primary_node = db.aliased(KnowledgeNode)
            query = query.join(primary_node, KnowledgeContradiction.primary_node_id == primary_node.id)
            query = query.filter(primary_node.domain == self.domain)
        
        return query.limit(limit).all()
