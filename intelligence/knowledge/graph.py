"""
Knowledge Graph Management

This module provides the KnowledgeGraph class for working with the knowledge graph,
including operations for creating, querying, and manipulating nodes and relationships.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
from sqlalchemy import and_, or_, func
from sqlalchemy.orm import aliased

from app import db
from db.models.knowledge_base import (
    KnowledgeNode, KnowledgeEdge, KnowledgeSource, 
    KnowledgeContradiction, EntityNode, ConceptNode, 
    EventNode, ClaimNode, KnowledgeTag
)
from db.models.entity_models import Entity
from intelligence.utils.text_processing import normalize_text

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    Manages operations on the knowledge graph, including creation,
    querying, and manipulation of nodes and relationships.
    """
    
    def __init__(self, domain: str = None):
        """
        Initialize the knowledge graph manager.
        
        Args:
            domain: Optional domain to filter operations by (e.g., 'football')
        """
        self.domain = domain
        logger.info(f"Initialized knowledge graph manager for domain: {domain}")
    
    # Node Operations
    
    def get_node(self, node_id: int) -> Optional[KnowledgeNode]:
        """
        Retrieve a knowledge node by ID.
        
        Args:
            node_id: The ID of the node to retrieve
            
        Returns:
            The knowledge node or None if not found
        """
        return KnowledgeNode.query.get(node_id)
    
    def find_nodes(self, 
                  name: str = None, 
                  node_type: str = None, 
                  limit: int = 100) -> List[KnowledgeNode]:
        """
        Find knowledge nodes matching the given criteria.
        
        Args:
            name: Optional name to search for (uses partial matching)
            node_type: Optional node type to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of matching knowledge nodes
        """
        query = KnowledgeNode.query
        
        if self.domain:
            query = query.filter(KnowledgeNode.domain == self.domain)
            
        if name:
            query = query.filter(KnowledgeNode.name.ilike(f"%{name}%"))
            
        if node_type:
            query = query.filter(KnowledgeNode.node_type == node_type)
            
        return query.limit(limit).all()
    
    def create_entity_node(self, 
                         name: str, 
                         entity_id: int = None,
                         attributes: Dict = None,
                         tags: List[str] = None,
                         confidence: float = 1.0) -> EntityNode:
        """
        Create a new entity node in the knowledge graph.
        
        Args:
            name: The name of the entity
            entity_id: Optional reference to an Entity in the entity_models
            attributes: Optional dictionary of entity attributes
            tags: Optional list of tags to associate with the node
            confidence: Confidence score for this entity (0.0-1.0)
            
        Returns:
            The created entity node
        """
        # Normalize the entity name
        canonical_name = normalize_text(name).lower()
        
        # Check if entity already exists with this canonical name
        existing = EntityNode.query.filter(
            EntityNode.canonical_name == canonical_name,
            EntityNode.domain == self.domain
        ).first()
        
        if existing:
            logger.info(f"Entity node already exists: {existing.id} - {existing.name}")
            return existing
        
        # Create new entity node
        entity_node = EntityNode(
            name=name,
            canonical_name=canonical_name,
            domain=self.domain or 'general',
            entity_id=entity_id,
            attributes=attributes or {},
            confidence=confidence
        )
        
        # Add tags
        if tags:
            for tag_name in tags:
                tag = KnowledgeTag.query.filter_by(name=tag_name).first()
                if not tag:
                    tag = KnowledgeTag(name=tag_name)
                    db.session.add(tag)
                entity_node.tags.append(tag)
        
        db.session.add(entity_node)
        db.session.commit()
        
        logger.info(f"Created entity node: {entity_node.id} - {entity_node.name}")
        return entity_node
    
    def create_concept_node(self, 
                          name: str,
                          content: str = None,
                          attributes: Dict = None,
                          tags: List[str] = None,
                          confidence: float = 1.0) -> ConceptNode:
        """
        Create a new concept node in the knowledge graph.
        
        Args:
            name: The name of the concept
            content: Optional textual description of the concept
            attributes: Optional dictionary of concept attributes
            tags: Optional list of tags to associate with the node
            confidence: Confidence score for this concept (0.0-1.0)
            
        Returns:
            The created concept node
        """
        # Normalize the concept name
        canonical_name = normalize_text(name).lower()
        
        # Check if concept already exists with this canonical name
        existing = ConceptNode.query.filter(
            ConceptNode.canonical_name == canonical_name,
            ConceptNode.domain == self.domain
        ).first()
        
        if existing:
            logger.info(f"Concept node already exists: {existing.id} - {existing.name}")
            return existing
        
        # Create new concept node
        concept_node = ConceptNode(
            name=name,
            canonical_name=canonical_name,
            domain=self.domain or 'general',
            content=content,
            attributes=attributes or {},
            confidence=confidence
        )
        
        # Add tags
        if tags:
            for tag_name in tags:
                tag = KnowledgeTag.query.filter_by(name=tag_name).first()
                if not tag:
                    tag = KnowledgeTag(name=tag_name)
                    db.session.add(tag)
                concept_node.tags.append(tag)
        
        db.session.add(concept_node)
        db.session.commit()
        
        logger.info(f"Created concept node: {concept_node.id} - {concept_node.name}")
        return concept_node
    
    def create_event_node(self, 
                        name: str,
                        start_date: datetime = None,
                        end_date: datetime = None,
                        location: str = None,
                        content: str = None,
                        attributes: Dict = None,
                        tags: List[str] = None,
                        confidence: float = 1.0) -> EventNode:
        """
        Create a new event node in the knowledge graph.
        
        Args:
            name: The name of the event
            start_date: Optional start date of the event
            end_date: Optional end date of the event
            location: Optional location of the event
            content: Optional textual description of the event
            attributes: Optional dictionary of event attributes
            tags: Optional list of tags to associate with the node
            confidence: Confidence score for this event (0.0-1.0)
            
        Returns:
            The created event node
        """
        # Check for similar events to avoid duplication
        existing = None
        if start_date:
            # If we have a start date, check for events with same name on same date
            existing = EventNode.query.filter(
                EventNode.name.ilike(f"%{name}%"),
                EventNode.start_date == start_date,
                EventNode.domain == self.domain
            ).first()
        
        if existing:
            logger.info(f"Similar event node already exists: {existing.id} - {existing.name}")
            return existing
        
        # Create new event node
        event_node = EventNode(
            name=name,
            domain=self.domain or 'general',
            start_date=start_date,
            end_date=end_date,
            location=location,
            content=content,
            attributes=attributes or {},
            confidence=confidence
        )
        
        # Add tags
        if tags:
            for tag_name in tags:
                tag = KnowledgeTag.query.filter_by(name=tag_name).first()
                if not tag:
                    tag = KnowledgeTag(name=tag_name)
                    db.session.add(tag)
                event_node.tags.append(tag)
        
        db.session.add(event_node)
        db.session.commit()
        
        logger.info(f"Created event node: {event_node.id} - {event_node.name}")
        return event_node
    
    def create_claim_node(self, 
                        content: str,
                        name: str = None,
                        claim_type: str = "factual",
                        sentiment: float = None,
                        attributes: Dict = None,
                        tags: List[str] = None,
                        confidence: float = 1.0) -> ClaimNode:
        """
        Create a new claim node in the knowledge graph.
        
        Args:
            content: The text of the claim
            name: Optional short name/title for the claim (generated from content if not provided)
            claim_type: Type of claim (factual, opinion, prediction)
            sentiment: Optional sentiment score (-1.0 to 1.0)
            attributes: Optional dictionary of claim attributes
            tags: Optional list of tags to associate with the node
            confidence: Confidence score for this claim (0.0-1.0)
            
        Returns:
            The created claim node
        """
        # Generate name from content if not provided
        if not name:
            name = content[:100] + ('...' if len(content) > 100 else '')
        
        # Create new claim node
        claim_node = ClaimNode(
            name=name,
            content=content,
            domain=self.domain or 'general',
            claim_type=claim_type,
            sentiment=sentiment,
            attributes=attributes or {},
            confidence=confidence
        )
        
        # Add tags
        if tags:
            for tag_name in tags:
                tag = KnowledgeTag.query.filter_by(name=tag_name).first()
                if not tag:
                    tag = KnowledgeTag(name=tag_name)
                    db.session.add(tag)
                claim_node.tags.append(tag)
        
        db.session.add(claim_node)
        db.session.commit()
        
        logger.info(f"Created claim node: {claim_node.id} - {claim_node.name[:30]}...")
        return claim_node
    
    def update_node(self, 
                   node_id: int, 
                   attributes: Dict = None,
                   content: str = None,
                   tags: List[str] = None,
                   confidence: float = None) -> Optional[KnowledgeNode]:
        """
        Update an existing knowledge node.
        
        Args:
            node_id: The ID of the node to update
            attributes: Optional dictionary of attributes to update
            content: Optional content to update
            tags: Optional list of tags to set (replaces existing tags)
            confidence: Optional confidence score to update
            
        Returns:
            The updated node or None if not found
        """
        node = self.get_node(node_id)
        if not node:
            logger.warning(f"Node not found for update: {node_id}")
            return None
        
        # Update attributes dictionary (merge with existing)
        if attributes:
            current_attrs = node.attributes or {}
            current_attrs.update(attributes)
            node.attributes = current_attrs
        
        # Update content if provided
        if content is not None:
            node.content = content
            
        # Update confidence if provided
        if confidence is not None:
            node.confidence = confidence
        
        # Update tags if provided
        if tags:
            # Clear existing tags
            node.tags = []
            
            # Add new tags
            for tag_name in tags:
                tag = KnowledgeTag.query.filter_by(name=tag_name).first()
                if not tag:
                    tag = KnowledgeTag(name=tag_name)
                    db.session.add(tag)
                node.tags.append(tag)
        
        # Update timestamp
        node.updated_at = datetime.utcnow()
        
        db.session.commit()
        logger.info(f"Updated node: {node.id} - {node.name}")
        
        return node
    
    def delete_node(self, node_id: int) -> bool:
        """
        Delete a knowledge node and all its relationships.
        
        Args:
            node_id: The ID of the node to delete
            
        Returns:
            True if the node was deleted, False if not found
        """
        node = self.get_node(node_id)
        if not node:
            logger.warning(f"Node not found for deletion: {node_id}")
            return False
        
        # Note: relationships will be automatically deleted due to
        # cascade="all, delete-orphan" in the relationship definitions
        
        db.session.delete(node)
        db.session.commit()
        
        logger.info(f"Deleted node: {node_id}")
        return True
    
    # Edge Operations
    
    def get_edge(self, edge_id: int) -> Optional[KnowledgeEdge]:
        """
        Retrieve a knowledge edge by ID.
        
        Args:
            edge_id: The ID of the edge to retrieve
            
        Returns:
            The knowledge edge or None if not found
        """
        return KnowledgeEdge.query.get(edge_id)
    
    def create_edge(self, 
                   source_id: int,
                   target_id: int,
                   relationship_type: str,
                   weight: float = 1.0,
                   confidence: float = 1.0,
                   attributes: Dict = None,
                   valid_from: datetime = None,
                   valid_to: datetime = None) -> Optional[KnowledgeEdge]:
        """
        Create a new edge between two knowledge nodes.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            relationship_type: Type of relationship
            weight: Weight of the relationship (default 1.0)
            confidence: Confidence in this relationship (0.0-1.0)
            attributes: Optional dictionary of edge attributes
            valid_from: Optional date when relationship became valid
            valid_to: Optional date when relationship ended
            
        Returns:
            The created edge or None if either node does not exist
        """
        # Verify that both nodes exist
        source_node = self.get_node(source_id)
        target_node = self.get_node(target_id)
        
        if not source_node or not target_node:
            logger.warning(f"Cannot create edge: source or target node not found. "
                          f"Source: {source_id}, Target: {target_id}")
            return None
        
        # Check if similar edge already exists
        existing = KnowledgeEdge.query.filter_by(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type
        ).first()
        
        if existing:
            logger.info(f"Similar edge already exists: {existing.id}")
            return existing
        
        # Create the edge
        edge = KnowledgeEdge(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            weight=weight,
            confidence=confidence,
            attributes=attributes or {},
            valid_from=valid_from,
            valid_to=valid_to
        )
        
        db.session.add(edge)
        db.session.commit()
        
        logger.info(f"Created edge: {edge.id} - {relationship_type} from {source_id} to {target_id}")
        return edge
    
    def update_edge(self, 
                   edge_id: int,
                   weight: float = None,
                   confidence: float = None,
                   attributes: Dict = None,
                   valid_to: datetime = None) -> Optional[KnowledgeEdge]:
        """
        Update an existing edge.
        
        Args:
            edge_id: ID of the edge to update
            weight: Optional new weight
            confidence: Optional new confidence score
            attributes: Optional dictionary of attributes to update
            valid_to: Optional date when relationship ended
            
        Returns:
            The updated edge or None if not found
        """
        edge = self.get_edge(edge_id)
        if not edge:
            logger.warning(f"Edge not found for update: {edge_id}")
            return None
        
        # Update attributes if provided
        if weight is not None:
            edge.weight = weight
            
        if confidence is not None:
            edge.confidence = confidence
            
        if valid_to is not None:
            edge.valid_to = valid_to
        
        # Update attributes dictionary (merge with existing)
        if attributes:
            current_attrs = edge.attributes or {}
            current_attrs.update(attributes)
            edge.attributes = current_attrs
        
        # Update timestamp
        edge.updated_at = datetime.utcnow()
        
        db.session.commit()
        logger.info(f"Updated edge: {edge.id}")
        
        return edge
    
    def delete_edge(self, edge_id: int) -> bool:
        """
        Delete a knowledge edge.
        
        Args:
            edge_id: The ID of the edge to delete
            
        Returns:
            True if the edge was deleted, False if not found
        """
        edge = self.get_edge(edge_id)
        if not edge:
            logger.warning(f"Edge not found for deletion: {edge_id}")
            return False
        
        db.session.delete(edge)
        db.session.commit()
        
        logger.info(f"Deleted edge: {edge_id}")
        return True
    
    # Graph Traversal and Queries
    
    def get_node_neighbors(self, 
                          node_id: int,
                          direction: str = 'both',
                          relationship_types: List[str] = None,
                          limit: int = 100) -> List[Dict]:
        """
        Get the neighbors of a node.
        
        Args:
            node_id: ID of the node
            direction: 'outgoing', 'incoming', or 'both'
            relationship_types: Optional list of relationship types to filter by
            limit: Maximum number of neighbors to return
            
        Returns:
            List of dictionaries with node and edge information
        """
        node = self.get_node(node_id)
        if not node:
            logger.warning(f"Node not found for getting neighbors: {node_id}")
            return []
        
        results = []
        
        # Get outgoing edges
        if direction in ['outgoing', 'both']:
            query = node.outgoing_edges
            
            if relationship_types:
                query = [e for e in query if e.relationship_type in relationship_types]
                
            for edge in query[:limit]:
                results.append({
                    'direction': 'outgoing',
                    'node': edge.target,
                    'edge': edge
                })
        
        # Get incoming edges
        if direction in ['incoming', 'both']:
            remaining_limit = limit - len(results)
            if remaining_limit > 0:
                query = node.incoming_edges
                
                if relationship_types:
                    query = [e for e in query if e.relationship_type in relationship_types]
                    
                for edge in query[:remaining_limit]:
                    results.append({
                        'direction': 'incoming',
                        'node': edge.source,
                        'edge': edge
                    })
        
        return results
    
    def find_path(self, 
                 start_node_id: int,
                 end_node_id: int,
                 max_depth: int = 3) -> List[Dict]:
        """
        Find a path between two nodes in the knowledge graph.
        
        Args:
            start_node_id: ID of the starting node
            end_node_id: ID of the ending node
            max_depth: Maximum path length to consider
            
        Returns:
            List of dictionaries representing the path, or empty list if no path found
        """
        # Breadth-first search implementation
        visited = set()
        queue = [[(start_node_id, None, None)]]  # Each item is a path [(node_id, edge_id, direction)]
        
        while queue:
            path = queue.pop(0)
            node_id, _, _ = path[-1]
            
            if node_id == end_node_id:
                # Found a path, convert to result format
                result_path = []
                for i, (node_id, edge_id, direction) in enumerate(path):
                    node = self.get_node(node_id)
                    if i > 0:  # Skip edge for the first node
                        edge = self.get_edge(edge_id)
                        result_path.append({
                            'node': node,
                            'edge': edge,
                            'direction': direction
                        })
                    else:
                        result_path.append({
                            'node': node,
                            'edge': None,
                            'direction': None
                        })
                return result_path
            
            if node_id in visited:
                continue
                
            visited.add(node_id)
            
            if len(path) >= max_depth:
                continue
            
            # Get neighbors
            neighbors = self.get_node_neighbors(node_id)
            for neighbor in neighbors:
                if neighbor['node'].id not in visited:
                    new_path = path.copy()
                    new_path.append((
                        neighbor['node'].id, 
                        neighbor['edge'].id,
                        neighbor['direction']
                    ))
                    queue.append(new_path)
        
        # No path found
        return []
    
    def get_subgraph(self, 
                    central_node_id: int,
                    max_nodes: int = 50,
                    max_depth: int = 2) -> Dict:
        """
        Get a subgraph centered around a specific node.
        
        Args:
            central_node_id: ID of the central node
            max_nodes: Maximum number of nodes to include
            max_depth: Maximum depth to traverse
            
        Returns:
            Dictionary with nodes and edges of the subgraph
        """
        central_node = self.get_node(central_node_id)
        if not central_node:
            logger.warning(f"Central node not found for subgraph: {central_node_id}")
            return {'nodes': [], 'edges': []}
        
        # Breadth-first traversal
        nodes = {central_node_id: central_node}
        edges = {}
        queue = [(central_node_id, 0)]  # (node_id, depth)
        visited = {central_node_id}
        
        while queue and len(nodes) < max_nodes:
            node_id, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            # Get neighbors
            neighbors = self.get_node_neighbors(node_id)
            for neighbor in neighbors:
                neighbor_id = neighbor['node'].id
                edge_id = neighbor['edge'].id
                
                # Add edge if not already added
                if edge_id not in edges:
                    edges[edge_id] = neighbor['edge']
                
                # Add node and enqueue if not visited
                if neighbor_id not in visited and len(nodes) < max_nodes:
                    nodes[neighbor_id] = neighbor['node']
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, depth + 1))
        
        return {
            'nodes': list(nodes.values()),
            'edges': list(edges.values())
        }
    
    # Knowledge Source Management
    
    def add_source(self, 
                  node_id: int = None,
                  edge_id: int = None,
                  content_id: int = None,
                  confidence: float = 1.0,
                  extraction_method: str = None,
                  excerpt: str = None,
                  context: str = None) -> KnowledgeSource:
        """
        Add a source for a knowledge node or edge.
        
        Args:
            node_id: Optional ID of the knowledge node
            edge_id: Optional ID of the knowledge edge
            content_id: ID of the content source
            confidence: Confidence in this source (0.0-1.0)
            extraction_method: Method used to extract this knowledge
            excerpt: Text excerpt where the knowledge was found
            context: Additional context around the excerpt
            
        Returns:
            The created knowledge source
        """
        if not node_id and not edge_id:
            raise ValueError("Either node_id or edge_id must be provided")
            
        source = KnowledgeSource(
            node_id=node_id,
            edge_id=edge_id,
            content_id=content_id,
            confidence=confidence,
            extraction_method=extraction_method,
            excerpt=excerpt,
            context=context
        )
        
        db.session.add(source)
        db.session.commit()
        
        entity_type = "node" if node_id else "edge"
        entity_id = node_id if node_id else edge_id
        logger.info(f"Added source for {entity_type}:{entity_id} from content:{content_id}")
        
        return source
    
    def get_node_sources(self, node_id: int) -> List[KnowledgeSource]:
        """
        Get all sources for a knowledge node.
        
        Args:
            node_id: ID of the knowledge node
            
        Returns:
            List of knowledge sources
        """
        return KnowledgeSource.query.filter_by(node_id=node_id).all()
    
    def get_edge_sources(self, edge_id: int) -> List[KnowledgeSource]:
        """
        Get all sources for a knowledge edge.
        
        Args:
            edge_id: ID of the knowledge edge
            
        Returns:
            List of knowledge sources
        """
        return KnowledgeSource.query.filter_by(edge_id=edge_id).all()
