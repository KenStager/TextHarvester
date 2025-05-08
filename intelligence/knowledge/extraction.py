"""
Knowledge Extraction

This module provides the KnowledgeExtractor class for extracting knowledge
from processed content, including entities, relationships, claims, and events.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime
import re

from app import db
from db.models.knowledge_base import (
    KnowledgeNode, EntityNode, ConceptNode, EventNode, ClaimNode, KnowledgeEdge
)
from db.models.entity_models import Entity, EntityMention
from db.models.content_intelligence import ContentClassification

logger = logging.getLogger(__name__)


class KnowledgeExtractor:
    """
    Extracts structured knowledge from processed content and
    converts it into knowledge graph nodes and edges.
    """
    
    def __init__(self, domain: str = None, graph_manager = None):
        """
        Initialize the knowledge extractor.
        
        Args:
            domain: Optional domain to filter operations by (e.g., 'football')
            graph_manager: Optional KnowledgeGraph instance to use for node/edge creation
        """
        self.domain = domain
        self.graph_manager = graph_manager
        
        # Import here to avoid circular imports
        if not graph_manager:
            from .graph import KnowledgeGraph
            self.graph_manager = KnowledgeGraph(domain=domain)
            
        logger.info(f"Initialized knowledge extractor for domain: {domain}")
    
    def extract_from_content(self, content_id: int) -> Dict[str, Any]:
        """
        Extract knowledge from a processed content item.
        
        Args:
            content_id: ID of the content to process
            
        Returns:
            Dictionary with extraction results
        """
        from db.models.content_intelligence import EnhancedContent
        
        # Get the enhanced content
        enhanced_content = EnhancedContent.query.filter_by(content_id=content_id).first()
        if not enhanced_content:
            logger.warning(f"No enhanced content found for content_id: {content_id}")
            return {"status": "error", "message": "No enhanced content found"}
        
        # Get entity mentions from the content
        entity_mentions = EntityMention.query.filter_by(content_id=content_id).all()
        
        # Get content classification
        classification = ContentClassification.query.filter_by(
            content_id=content_id, 
            is_primary=True
        ).first()
        
        results = {
            "content_id": content_id,
            "extracted_entities": [],
            "extracted_relationships": [],
            "extracted_claims": [],
            "extracted_events": []
        }
        
        # Extract entities from entity mentions
        entity_nodes = self._extract_entities(entity_mentions, content_id)
        results["extracted_entities"] = entity_nodes
        
        # Extract relationships between entities
        relationships = self._extract_relationships(entity_mentions, entity_nodes, content_id)
        results["extracted_relationships"] = relationships
        
        # Extract claims if enhanced content has them
        if enhanced_content.enhanced_metadata and "claims" in enhanced_content.enhanced_metadata:
            claims = self._extract_claims(enhanced_content.enhanced_metadata["claims"], content_id)
            results["extracted_claims"] = claims
        
        # Extract events if enhanced content has them
        if enhanced_content.enhanced_metadata and "events" in enhanced_content.enhanced_metadata:
            events = self._extract_events(enhanced_content.enhanced_metadata["events"], content_id)
            results["extracted_events"] = events
        
        logger.info(f"Extracted knowledge from content {content_id}: "
                   f"{len(results['extracted_entities'])} entities, "
                   f"{len(results['extracted_relationships'])} relationships, "
                   f"{len(results['extracted_claims'])} claims, "
                   f"{len(results['extracted_events'])} events")
        
        return results
    
    def _extract_entities(self, 
                         entity_mentions: List[EntityMention],
                         content_id: int) -> List[Dict]:
        """
        Extract entity nodes from entity mentions.
        
        Args:
            entity_mentions: List of entity mentions from the content
            content_id: ID of the content being processed
            
        Returns:
            List of dictionaries with entity node information
        """
        entity_nodes = []
        mention_to_node = {}  # Maps entity mention ID to knowledge node ID
        
        # Group mentions by entity
        entity_groups = {}
        for mention in entity_mentions:
            if mention.entity_id:
                if mention.entity_id not in entity_groups:
                    entity_groups[mention.entity_id] = []
                entity_groups[mention.entity_id].append(mention)
        
        # Create or update entity nodes for each entity
        for entity_id, mentions in entity_groups.items():
            # Get the entity
            entity = Entity.query.get(entity_id)
            if not entity:
                logger.warning(f"Entity not found: {entity_id}")
                continue
            
            # Extract confidence from mentions
            confidence = sum(mention.confidence or 1.0 for mention in mentions) / len(mentions)
            
            # Create entity attributes from entity metadata
            attributes = {}
            if entity.metadata:
                attributes.update(entity.metadata)
            
            # Add mention texts as alternative names
            mention_texts = set(mention.mention_text for mention in mentions)
            if len(mention_texts) > 1:
                attributes["alternative_names"] = list(mention_texts)
            
            # Determine entity tags based on entity type
            tags = []
            if entity.entity_type_id:
                entity_type = entity.entity_type
                if entity_type:
                    tags.append(entity_type.name)
                    # Add parent types as tags
                    parent_type = entity_type.parent
                    while parent_type:
                        tags.append(parent_type.name)
                        parent_type = parent_type.parent
            
            # Create or update entity node
            entity_name = entity.canonical_name or entity.name
            entity_node = self.graph_manager.create_entity_node(
                name=entity_name,
                entity_id=entity_id,
                attributes=attributes,
                tags=tags,
                confidence=confidence
            )
            
            # Store mapping from mentions to this node
            for mention in mentions:
                mention_to_node[mention.id] = entity_node.id
            
            # Add source reference
            self.graph_manager.add_source(
                node_id=entity_node.id,
                content_id=content_id,
                confidence=confidence,
                extraction_method="entity_mention",
                excerpt=mentions[0].mention_text,
                context=mentions[0].context_before + " " + mentions[0].context_after
            )
            
            entity_nodes.append({
                "node_id": entity_node.id,
                "entity_id": entity_id,
                "name": entity_name,
                "type": entity.entity_type.name if entity.entity_type else None,
                "confidence": confidence,
                "mentions": len(mentions)
            })
        
        return entity_nodes
    
    def _extract_relationships(self, 
                              entity_mentions: List[EntityMention],
                              entity_nodes: List[Dict],
                              content_id: int) -> List[Dict]:
        """
        Extract relationships between entities based on co-occurrence and patterns.
        
        Args:
            entity_mentions: List of entity mentions from the content
            entity_nodes: List of entity nodes extracted
            content_id: ID of the content being processed
            
        Returns:
            List of dictionaries with relationship information
        """
        relationships = []
        
        # Map entity IDs to node IDs
        entity_to_node = {node["entity_id"]: node["node_id"] for node in entity_nodes}
        
        # Extract co-occurrence relationships
        # Two entities mentioned within N tokens of each other might be related
        if self.domain == "football":
            relationships.extend(self._extract_football_relationships(
                entity_mentions, entity_to_node, content_id))
        else:
            # Generic co-occurrence relationships
            relationships.extend(self._extract_cooccurrence_relationships(
                entity_mentions, entity_to_node, content_id))
            
        return relationships
    
    def _extract_cooccurrence_relationships(self,
                                          entity_mentions: List[EntityMention],
                                          entity_to_node: Dict[int, int],
                                          content_id: int) -> List[Dict]:
        """
        Extract relationships based on entity co-occurrence.
        
        Args:
            entity_mentions: List of entity mentions
            entity_to_node: Mapping from entity IDs to node IDs
            content_id: Content being processed
            
        Returns:
            List of dictionaries with relationship information
        """
        relationships = []
        MAX_DISTANCE = 100  # Maximum character distance for co-occurrence
        
        # Sort mentions by position in text
        sorted_mentions = sorted(entity_mentions, key=lambda m: m.start_char)
        
        for i, mention1 in enumerate(sorted_mentions[:-1]):
            if not mention1.entity_id or mention1.entity_id not in entity_to_node:
                continue
                
            source_node_id = entity_to_node[mention1.entity_id]
            
            # Look at subsequent mentions within MAX_DISTANCE
            for mention2 in sorted_mentions[i+1:]:
                if mention2.start_char - mention1.end_char > MAX_DISTANCE:
                    break
                    
                if not mention2.entity_id or mention2.entity_id not in entity_to_node:
                    continue
                    
                if mention2.entity_id == mention1.entity_id:
                    continue  # Skip same entity
                    
                target_node_id = entity_to_node[mention2.entity_id]
                
                # Determine relationship type based on entity types
                relationship_type = "co_occurs_with"
                confidence = 0.7  # Base confidence for co-occurrence
                
                # Extract text between mentions to look for relationship indicators
                text_between = mention1.context_after[:mention2.start_char - mention1.end_char]
                
                # Create relationship
                edge = self.graph_manager.create_edge(
                    source_id=source_node_id,
                    target_id=target_node_id,
                    relationship_type=relationship_type,
                    confidence=confidence
                )
                
                if edge:
                    # Add source
                    self.graph_manager.add_source(
                        edge_id=edge.id,
                        content_id=content_id,
                        confidence=confidence,
                        extraction_method="co_occurrence",
                        excerpt=f"{mention1.mention_text} ... {mention2.mention_text}",
                        context=f"{mention1.context_before} {mention1.mention_text} {text_between} {mention2.mention_text} {mention2.context_after}"
                    )
                    
                    relationships.append({
                        "edge_id": edge.id,
                        "source_node_id": source_node_id,
                        "target_node_id": target_node_id,
                        "relationship_type": relationship_type,
                        "confidence": confidence
                    })
        
        return relationships
    
    def _extract_football_relationships(self,
                                       entity_mentions: List[EntityMention],
                                       entity_to_node: Dict[int, int],
                                       content_id: int) -> List[Dict]:
        """
        Extract football-specific relationships between entities.
        
        Args:
            entity_mentions: List of entity mentions
            entity_to_node: Mapping from entity IDs to node IDs
            content_id: Content being processed
            
        Returns:
            List of dictionaries with relationship information
        """
        relationships = []
        
        # Patterns for football relationships
        patterns = [
            {
                "pattern": r"(played|playing|plays)\s+for",
                "relationship": "plays_for",
                "source_type": "PERSON.PLAYER",
                "target_type": "TEAM"
            },
            {
                "pattern": r"(manages|managing|managed)\s+",
                "relationship": "manages",
                "source_type": "PERSON.MANAGER",
                "target_type": "TEAM"
            },
            {
                "pattern": r"(scored|scoring|scores)\s+",
                "relationship": "scored_against",
                "source_type": "PERSON.PLAYER",
                "target_type": "TEAM"
            },
            {
                "pattern": r"(signed|signing|signs)\s+",
                "relationship": "signed",
                "source_type": "TEAM",
                "target_type": "PERSON.PLAYER"
            },
            {
                "pattern": r"(won|winning|wins)\s+against",
                "relationship": "won_against",
                "source_type": "TEAM",
                "target_type": "TEAM"
            }
        ]
        
        # Group mentions by entity type
        mentions_by_type = {}
        for mention in entity_mentions:
            if not mention.entity_id or mention.entity_id not in entity_to_node:
                continue
                
            entity = Entity.query.get(mention.entity_id)
            if not entity or not entity.entity_type:
                continue
                
            entity_type = entity.entity_type.name
            if entity_type not in mentions_by_type:
                mentions_by_type[entity_type] = []
                
            mentions_by_type[entity_type].append(mention)
        
        # For each pattern, look for matching entity pairs
        for pattern in patterns:
            if pattern["source_type"] not in mentions_by_type or pattern["target_type"] not in mentions_by_type:
                continue
                
            for source_mention in mentions_by_type[pattern["source_type"]]:
                for target_mention in mentions_by_type[pattern["target_type"]]:
                    # Skip if same entity
                    if source_mention.entity_id == target_mention.entity_id:
                        continue
                        
                    # Check if they're close to each other
                    if abs(source_mention.start_char - target_mention.start_char) > 200:
                        continue
                        
                    # Get text around the mentions
                    text = source_mention.context_before + " " + source_mention.mention_text + " " + \
                           source_mention.context_after + " " + target_mention.context_before + " " + \
                           target_mention.mention_text + " " + target_mention.context_after
                    
                    # Check if the pattern matches
                    if re.search(pattern["pattern"], text, re.IGNORECASE):
                        source_node_id = entity_to_node[source_mention.entity_id]
                        target_node_id = entity_to_node[target_mention.entity_id]
                        
                        edge = self.graph_manager.create_edge(
                            source_id=source_node_id,
                            target_id=target_node_id,
                            relationship_type=pattern["relationship"],
                            confidence=0.85  # Higher confidence for pattern matches
                        )
                        
                        if edge:
                            # Add source
                            self.graph_manager.add_source(
                                edge_id=edge.id,
                                content_id=content_id,
                                confidence=0.85,
                                extraction_method="pattern_match",
                                excerpt=f"{source_mention.mention_text} {pattern['relationship']} {target_mention.mention_text}",
                                context=text
                            )
                            
                            relationships.append({
                                "edge_id": edge.id,
                                "source_node_id": source_node_id,
                                "target_node_id": target_node_id,
                                "relationship_type": pattern["relationship"],
                                "confidence": 0.85
                            })
        
        return relationships
    
    def _extract_claims(self, claims_data: List[Dict], content_id: int) -> List[Dict]:
        """
        Extract claim nodes from claims data in enhanced content.
        
        Args:
            claims_data: List of claims from enhanced content
            content_id: ID of the content being processed
            
        Returns:
            List of dictionaries with claim node information
        """
        extracted_claims = []
        
        for claim_data in claims_data:
            claim_text = claim_data.get("text")
            if not claim_text:
                continue
                
            claim_type = claim_data.get("type", "factual")
            sentiment = claim_data.get("sentiment")
            confidence = claim_data.get("confidence", 0.8)
            
            # Create claim node
            claim_node = self.graph_manager.create_claim_node(
                content=claim_text,
                name=claim_data.get("title"),
                claim_type=claim_type,
                sentiment=sentiment,
                confidence=confidence
            )
            
            # Add source
            self.graph_manager.add_source(
                node_id=claim_node.id,
                content_id=content_id,
                confidence=confidence,
                extraction_method="claim_extraction",
                excerpt=claim_text
            )
            
            # Link claim to entities it mentions
            if "entities" in claim_data:
                for entity_id in claim_data["entities"]:
                    if entity_id in [node["entity_id"] for node in extracted_claims]:
                        node_id = next(node["node_id"] for node in extracted_claims if node["entity_id"] == entity_id)
                        
                        edge = self.graph_manager.create_edge(
                            source_id=claim_node.id,
                            target_id=node_id,
                            relationship_type="mentions",
                            confidence=confidence
                        )
                        
                        if edge:
                            self.graph_manager.add_source(
                                edge_id=edge.id,
                                content_id=content_id,
                                confidence=confidence,
                                extraction_method="claim_entity_link",
                                excerpt=claim_text
                            )
            
            extracted_claims.append({
                "node_id": claim_node.id,
                "text": claim_text,
                "type": claim_type,
                "confidence": confidence
            })
        
        return extracted_claims
    
    def _extract_events(self, events_data: List[Dict], content_id: int) -> List[Dict]:
        """
        Extract event nodes from events data in enhanced content.
        
        Args:
            events_data: List of events from enhanced content
            content_id: ID of the content being processed
            
        Returns:
            List of dictionaries with event node information
        """
        extracted_events = []
        
        for event_data in events_data:
            event_name = event_data.get("name")
            if not event_name:
                continue
                
            start_date = event_data.get("start_date")
            if isinstance(start_date, str):
                try:
                    start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                except ValueError:
                    start_date = None
            
            end_date = event_data.get("end_date")
            if isinstance(end_date, str):
                try:
                    end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                except ValueError:
                    end_date = None
                    
            location = event_data.get("location")
            description = event_data.get("description")
            confidence = event_data.get("confidence", 0.8)
            
            # Create tags from event type
            tags = []
            if "type" in event_data:
                tags.append(event_data["type"])
            
            # Create event node
            event_node = self.graph_manager.create_event_node(
                name=event_name,
                start_date=start_date,
                end_date=end_date,
                location=location,
                content=description,
                tags=tags,
                confidence=confidence
            )
            
            # Add source
            self.graph_manager.add_source(
                node_id=event_node.id,
                content_id=content_id,
                confidence=confidence,
                extraction_method="event_extraction",
                excerpt=event_name + (f" - {description}" if description else "")
            )
            
            # Link event to entities involved
            if "entities" in event_data:
                for entity_id in event_data["entities"]:
                    if entity_id in [node["entity_id"] for node in extracted_events]:
                        node_id = next(node["node_id"] for node in extracted_events if node["entity_id"] == entity_id)
                        
                        edge = self.graph_manager.create_edge(
                            source_id=event_node.id,
                            target_id=node_id,
                            relationship_type="involves",
                            confidence=confidence
                        )
                        
                        if edge:
                            self.graph_manager.add_source(
                                edge_id=edge.id,
                                content_id=content_id,
                                confidence=confidence,
                                extraction_method="event_entity_link",
                                excerpt=event_name
                            )
            
            extracted_events.append({
                "node_id": event_node.id,
                "name": event_name,
                "start_date": start_date,
                "location": location,
                "confidence": confidence
            })
        
        return extracted_events
