"""
Knowledge Querying

This module provides the KnowledgeQuerier class for querying and exploring
the knowledge graph, including pathfinding, pattern matching, and aggregation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime
from sqlalchemy import and_, or_, not_, func, text
from sqlalchemy.orm import aliased

from app import db
from db.models.knowledge_base import (
    KnowledgeNode, KnowledgeEdge, KnowledgeTag, KnowledgeQuery,
    EntityNode, ConceptNode, EventNode, ClaimNode
)

logger = logging.getLogger(__name__)


class KnowledgeQuerier:
    """
    Provides utilities for querying and exploring the knowledge graph,
    including path finding, pattern matching, and aggregation.
    """
    
    def __init__(self, domain: str = None):
        """
        Initialize the knowledge querier.
        
        Args:
            domain: Optional domain to filter operations by (e.g., 'football')
        """
        self.domain = domain
        logger.info(f"Initialized knowledge querier for domain: {domain}")
    
    def search_nodes(self, 
                    query: str = None,
                    node_types: List[str] = None,
                    tags: List[str] = None,
                    min_confidence: float = 0.0,
                    limit: int = 20,
                    offset: int = 0) -> Dict[str, Any]:
        """
        Search for nodes matching the given criteria.
        
        Args:
            query: Text query to search in node names and content
            node_types: Optional list of node types to filter by
            tags: Optional list of tags to filter by
            min_confidence: Minimum confidence threshold
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            Dictionary with search results
        """
        # Start building the query
        base_query = KnowledgeNode.query
        
        # Apply domain filter if specified
        if self.domain:
            base_query = base_query.filter(KnowledgeNode.domain == self.domain)
        
        # Apply text search if specified
        if query:
            base_query = base_query.filter(
                or_(
                    KnowledgeNode.name.ilike(f"%{query}%"),
                    KnowledgeNode.content.ilike(f"%{query}%"),
                    KnowledgeNode.attributes.cast(db.String).ilike(f"%{query}%")
                )
            )
        
        # Apply node type filter if specified
        if node_types:
            base_query = base_query.filter(KnowledgeNode.node_type.in_(node_types))
        
        # Apply confidence threshold
        if min_confidence > 0:
            base_query = base_query.filter(KnowledgeNode.confidence >= min_confidence)
        
        # Apply tag filter if specified
        if tags:
            for tag in tags:
                tag_obj = KnowledgeTag.query.filter_by(name=tag).first()
                if tag_obj:
                    base_query = base_query.filter(KnowledgeNode.tags.any(KnowledgeTag.id == tag_obj.id))
        
        # Count total results
        total_count = base_query.count()
        
        # Apply pagination
        results = base_query.order_by(KnowledgeNode.confidence.desc()).offset(offset).limit(limit).all()
        
        # Log the query for analytics
        self._log_query(
            query_text=query,
            query_type="node_search",
            parameters={
                "node_types": node_types,
                "tags": tags,
                "min_confidence": min_confidence,
                "domain": self.domain
            },
            result_count=len(results)
        )
        
        # Format results
        formatted_results = []
        for node in results:
            formatted_results.append(self._format_node_result(node))
        
        return {
            "total": total_count,
            "offset": offset,
            "limit": limit,
            "results": formatted_results
        }
    
    def _format_node_result(self, node: KnowledgeNode) -> Dict[str, Any]:
        """
        Format a node into a standardized result dictionary.
        
        Args:
            node: The knowledge node to format
            
        Returns:
            Dictionary with node information
        """
        result = {
            "id": node.id,
            "type": node.node_type,
            "name": node.name,
            "domain": node.domain,
            "confidence": node.confidence,
            "tags": [tag.name for tag in node.tags],
            "created_at": node.created_at.isoformat() if node.created_at else None,
            "updated_at": node.updated_at.isoformat() if node.updated_at else None
        }
        
        # Add type-specific information
        if node.node_type == "entity" and isinstance(node, EntityNode):
            result["entity_id"] = node.entity_id
            
        elif node.node_type == "concept" and isinstance(node, ConceptNode):
            result["content"] = node.content
            
        elif node.node_type == "event" and isinstance(node, EventNode):
            result["start_date"] = node.start_date.isoformat() if node.start_date else None
            result["end_date"] = node.end_date.isoformat() if node.end_date else None
            result["location"] = node.location
            
        elif node.node_type == "claim" and isinstance(node, ClaimNode):
            result["claim_type"] = node.claim_type
            result["content"] = node.content
            result["sentiment"] = node.sentiment
            result["is_refuted"] = node.is_refuted
        
        # Add attributes if present
        if node.attributes:
            result["attributes"] = node.attributes
        
        # Add relationship counts
        result["outgoing_edges_count"] = len(node.outgoing_edges)
        result["incoming_edges_count"] = len(node.incoming_edges)
        
        return result
    
    def get_node_relationships(self, 
                              node_id: int,
                              direction: str = "both",
                              relationship_types: List[str] = None,
                              limit: int = 20) -> Dict[str, Any]:
        """
        Get the relationships of a specific node.
        
        Args:
            node_id: ID of the node
            direction: "outgoing", "incoming", or "both"
            relationship_types: Optional list of relationship types to filter by
            limit: Maximum number of relationships to return
            
        Returns:
            Dictionary with relationship information
        """
        node = KnowledgeNode.query.get(node_id)
        if not node:
            logger.warning(f"Node not found for relationships: {node_id}")
            return {
                "status": "error",
                "message": "Node not found"
            }
        
        relationships = {
            "node_id": node_id,
            "node_name": node.name,
            "node_type": node.node_type,
            "outgoing": [],
            "incoming": []
        }
        
        # Get outgoing relationships
        if direction in ["outgoing", "both"]:
            outgoing_query = node.outgoing_edges
            
            # Filter by relationship type if specified
            if relationship_types:
                outgoing_query = [e for e in outgoing_query if e.relationship_type in relationship_types]
            
            # Add relationships to result
            for edge in outgoing_query[:limit]:
                target = KnowledgeNode.query.get(edge.target_id)
                relationships["outgoing"].append({
                    "edge_id": edge.id,
                    "relationship_type": edge.relationship_type,
                    "target_id": edge.target_id,
                    "target_name": target.name if target else "Unknown",
                    "target_type": target.node_type if target else "unknown",
                    "confidence": edge.confidence,
                    "attributes": edge.attributes
                })
        
        # Get incoming relationships
        if direction in ["incoming", "both"]:
            incoming_query = node.incoming_edges
            
            # Filter by relationship type if specified
            if relationship_types:
                incoming_query = [e for e in incoming_query if e.relationship_type in relationship_types]
            
            # Add relationships to result
            remaining_limit = limit - len(relationships["outgoing"]) if direction == "both" else limit
            for edge in incoming_query[:remaining_limit]:
                source = KnowledgeNode.query.get(edge.source_id)
                relationships["incoming"].append({
                    "edge_id": edge.id,
                    "relationship_type": edge.relationship_type,
                    "source_id": edge.source_id,
                    "source_name": source.name if source else "Unknown",
                    "source_type": source.node_type if source else "unknown",
                    "confidence": edge.confidence,
                    "attributes": edge.attributes
                })
        
        # Log the query for analytics
        self._log_query(
            query_text=f"relationships for node {node_id}",
            query_type="node_relationships",
            parameters={
                "node_id": node_id,
                "direction": direction,
                "relationship_types": relationship_types
            },
            result_count=len(relationships["outgoing"]) + len(relationships["incoming"])
        )
        
        return relationships
    
    def find_paths(self, 
                  start_node_id: int,
                  end_node_id: int,
                  max_length: int = 3,
                  min_confidence: float = 0.0) -> Dict[str, Any]:
        """
        Find paths between two nodes in the knowledge graph.
        
        Args:
            start_node_id: ID of the starting node
            end_node_id: ID of the ending node
            max_length: Maximum path length
            min_confidence: Minimum confidence threshold for edges
            
        Returns:
            Dictionary with path information
        """
        # Verify nodes exist
        start_node = KnowledgeNode.query.get(start_node_id)
        end_node = KnowledgeNode.query.get(end_node_id)
        
        if not start_node:
            logger.warning(f"Start node not found: {start_node_id}")
            return {
                "status": "error",
                "message": "Start node not found"
            }
            
        if not end_node:
            logger.warning(f"End node not found: {end_node_id}")
            return {
                "status": "error",
                "message": "End node not found"
            }
        
        # Find paths using breadth-first search
        paths = self._find_paths_bfs(
            start_node_id=start_node_id,
            end_node_id=end_node_id,
            max_length=max_length,
            min_confidence=min_confidence
        )
        
        # Format the path results
        formatted_paths = []
        
        for path in paths:
            formatted_path = []
            
            for i, step in enumerate(path):
                node_id, edge_id, direction = step
                
                node = KnowledgeNode.query.get(node_id)
                
                if i > 0:  # Not the first node
                    edge = KnowledgeEdge.query.get(edge_id)
                    
                    formatted_path.append({
                        "node": {
                            "id": node.id,
                            "name": node.name,
                            "type": node.node_type
                        },
                        "edge": {
                            "id": edge.id,
                            "type": edge.relationship_type,
                            "direction": direction,
                            "confidence": edge.confidence
                        }
                    })
                else:  # First node has no incoming edge
                    formatted_path.append({
                        "node": {
                            "id": node.id,
                            "name": node.name,
                            "type": node.node_type
                        },
                        "edge": None
                    })
            
            formatted_paths.append(formatted_path)
        
        # Log the query for analytics
        self._log_query(
            query_text=f"paths from {start_node_id} to {end_node_id}",
            query_type="path_finding",
            parameters={
                "start_node_id": start_node_id,
                "end_node_id": end_node_id,
                "max_length": max_length,
                "min_confidence": min_confidence
            },
            result_count=len(formatted_paths)
        )
        
        return {
            "status": "success",
            "start_node": {
                "id": start_node.id,
                "name": start_node.name,
                "type": start_node.node_type
            },
            "end_node": {
                "id": end_node.id,
                "name": end_node.name,
                "type": end_node.node_type
            },
            "paths_found": len(formatted_paths),
            "paths": formatted_paths
        }
    
    def _find_paths_bfs(self, 
                       start_node_id: int,
                       end_node_id: int,
                       max_length: int,
                       min_confidence: float) -> List[List[Tuple]]:
        """
        Find paths between nodes using breadth-first search.
        
        Args:
            start_node_id: ID of the starting node
            end_node_id: ID of the ending node
            max_length: Maximum path length
            min_confidence: Minimum confidence threshold for edges
            
        Returns:
            List of paths, where each path is a list of (node_id, edge_id, direction) tuples
        """
        # Initialize search
        visited = set()
        queue = [[(start_node_id, None, None)]]  # Each path is a list of (node_id, edge_id, direction) tuples
        complete_paths = []
        
        while queue:
            path = queue.pop(0)
            current_node_id = path[-1][0]
            
            # Check if we've reached the destination
            if current_node_id == end_node_id:
                complete_paths.append(path)
                continue
            
            # Skip if we've already visited this node on this path
            if current_node_id in {node_id for node_id, _, _ in path[:-1]}:
                continue
            
            # Skip if path is already at maximum length
            if len(path) > max_length:
                continue
            
            # Mark current node as visited
            visited.add(current_node_id)
            
            # Get the current node
            current_node = KnowledgeNode.query.get(current_node_id)
            
            # Expand outgoing edges
            for edge in current_node.outgoing_edges:
                # Skip if confidence is below threshold
                if edge.confidence < min_confidence:
                    continue
                
                # Create a new path with this edge
                new_path = path.copy()
                new_path.append((edge.target_id, edge.id, "outgoing"))
                queue.append(new_path)
            
            # Expand incoming edges
            for edge in current_node.incoming_edges:
                # Skip if confidence is below threshold
                if edge.confidence < min_confidence:
                    continue
                
                # Create a new path with this edge
                new_path = path.copy()
                new_path.append((edge.source_id, edge.id, "incoming"))
                queue.append(new_path)
        
        return complete_paths
    
    def get_subgraph(self, 
                    node_id: int,
                    max_depth: int = 2,
                    max_nodes: int = 50,
                    relationship_types: List[str] = None,
                    min_confidence: float = 0.0) -> Dict[str, Any]:
        """
        Get a subgraph centered around a specific node.
        
        Args:
            node_id: ID of the central node
            max_depth: Maximum path length from central node
            max_nodes: Maximum number of nodes to include
            relationship_types: Optional list of relationship types to filter by
            min_confidence: Minimum confidence threshold for edges
            
        Returns:
            Dictionary with subgraph information
        """
        # Check if node exists
        central_node = KnowledgeNode.query.get(node_id)
        if not central_node:
            logger.warning(f"Node not found for subgraph: {node_id}")
            return {
                "status": "error",
                "message": "Node not found"
            }
        
        # Explore the neighborhood using breadth-first search
        nodes = {node_id: central_node}  # Map of node_id to node
        edges = {}  # Map of edge_id to edge
        
        # Queue contains (node_id, depth) tuples
        queue = [(node_id, 0)]
        visited = {node_id}
        
        while queue and len(nodes) < max_nodes:
            current_id, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            current_node = nodes[current_id]
            
            # Process outgoing edges
            for edge in current_node.outgoing_edges:
                # Skip if confidence is below threshold
                if edge.confidence < min_confidence:
                    continue
                
                # Skip if relationship type doesn't match filter
                if relationship_types and edge.relationship_type not in relationship_types:
                    continue
                
                # Add the edge to our subgraph
                edges[edge.id] = edge
                
                # Add the target node if not already visited
                if edge.target_id not in visited and len(nodes) < max_nodes:
                    target_node = KnowledgeNode.query.get(edge.target_id)
                    if target_node:
                        nodes[edge.target_id] = target_node
                        visited.add(edge.target_id)
                        queue.append((edge.target_id, depth + 1))
            
            # Process incoming edges
            for edge in current_node.incoming_edges:
                # Skip if confidence is below threshold
                if edge.confidence < min_confidence:
                    continue
                
                # Skip if relationship type doesn't match filter
                if relationship_types and edge.relationship_type not in relationship_types:
                    continue
                
                # Add the edge to our subgraph
                edges[edge.id] = edge
                
                # Add the source node if not already visited
                if edge.source_id not in visited and len(nodes) < max_nodes:
                    source_node = KnowledgeNode.query.get(edge.source_id)
                    if source_node:
                        nodes[edge.source_id] = source_node
                        visited.add(edge.source_id)
                        queue.append((edge.source_id, depth + 1))
        
        # Format the result
        formatted_nodes = [self._format_node_result(node) for node in nodes.values()]
        
        formatted_edges = []
        for edge in edges.values():
            formatted_edges.append({
                "id": edge.id,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "type": edge.relationship_type,
                "confidence": edge.confidence,
                "weight": edge.weight,
                "attributes": edge.attributes
            })
        
        # Log the query for analytics
        self._log_query(
            query_text=f"subgraph around node {node_id}",
            query_type="subgraph",
            parameters={
                "node_id": node_id,
                "max_depth": max_depth,
                "max_nodes": max_nodes,
                "relationship_types": relationship_types,
                "min_confidence": min_confidence
            },
            result_count=len(formatted_nodes)
        )
        
        return {
            "status": "success",
            "central_node": {
                "id": central_node.id,
                "name": central_node.name,
                "type": central_node.node_type
            },
            "node_count": len(formatted_nodes),
            "edge_count": len(formatted_edges),
            "nodes": formatted_nodes,
            "edges": formatted_edges
        }
    
    def find_shortest_path(self, 
                          start_node_id: int,
                          end_node_id: int,
                          max_length: int = 5,
                          min_confidence: float = 0.0) -> Dict[str, Any]:
        """
        Find the shortest path between two nodes.
        
        Args:
            start_node_id: ID of the starting node
            end_node_id: ID of the ending node
            max_length: Maximum path length to consider
            min_confidence: Minimum confidence threshold for edges
            
        Returns:
            Dictionary with path information
        """
        # Use the find_paths method with limit=1 to get shortest path
        path_results = self.find_paths(
            start_node_id=start_node_id,
            end_node_id=end_node_id,
            max_length=max_length,
            min_confidence=min_confidence
        )
        
        if path_results.get("status") == "error":
            return path_results
        
        if not path_results.get("paths"):
            return {
                "status": "success",
                "path_found": False,
                "message": "No path found"
            }
        
        # Find the shortest path
        shortest_path = min(path_results["paths"], key=len)
        
        return {
            "status": "success",
            "path_found": True,
            "path_length": len(shortest_path),
            "path": shortest_path
        }
    
    def search_by_pattern(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for subgraphs matching a specified pattern.
        
        Args:
            pattern: Dictionary describing the pattern to match
            
        Returns:
            Dictionary with matching subgraphs
        """
        # This is a simplified implementation
        # A full pattern matching system would need a more sophisticated approach
        
        results = {
            "status": "success",
            "matches": []
        }
        
        try:
            # Extract pattern components
            node_patterns = pattern.get("nodes", [])
            edge_patterns = pattern.get("edges", [])
            
            if not node_patterns:
                return {
                    "status": "error",
                    "message": "Pattern must include at least one node"
                }
            
            # Start with the first node pattern
            first_node_pattern = node_patterns[0]
            
            # Query for nodes matching the first pattern
            start_nodes_query = KnowledgeNode.query
            
            # Apply node type filter
            if "type" in first_node_pattern:
                start_nodes_query = start_nodes_query.filter(
                    KnowledgeNode.node_type == first_node_pattern["type"]
                )
            
            # Apply name filter
            if "name" in first_node_pattern:
                start_nodes_query = start_nodes_query.filter(
                    KnowledgeNode.name.ilike(f"%{first_node_pattern['name']}%")
                )
            
            # Apply domain filter
            if self.domain:
                start_nodes_query = start_nodes_query.filter(
                    KnowledgeNode.domain == self.domain
                )
            
            # Apply confidence filter
            if "min_confidence" in first_node_pattern:
                start_nodes_query = start_nodes_query.filter(
                    KnowledgeNode.confidence >= first_node_pattern["min_confidence"]
                )
            
            # Get the matching start nodes
            start_nodes = start_nodes_query.all()
            
            # For each start node, try to match the pattern
            for start_node in start_nodes:
                # For simplicity, we'll just check if the node has the right connections
                # A full implementation would recursively match the entire pattern
                
                matches = True
                
                # Check each edge pattern
                for edge_pattern in edge_patterns:
                    source_idx = edge_pattern.get("source_idx", 0)
                    target_idx = edge_pattern.get("target_idx", 1)
                    relationship_type = edge_pattern.get("type")
                    
                    # Only handle patterns starting from the first node for simplicity
                    if source_idx != 0:
                        continue
                    
                    # Check if the edge exists
                    matching_edges = []
                    
                    if relationship_type:
                        matching_edges = [e for e in start_node.outgoing_edges 
                                        if e.relationship_type == relationship_type]
                    else:
                        matching_edges = start_node.outgoing_edges
                    
                    if not matching_edges:
                        matches = False
                        break
                
                if matches:
                    # This node matches the pattern
                    results["matches"].append({
                        "match_id": start_node.id,
                        "match_type": start_node.node_type,
                        "match_name": start_node.name
                    })
            
            results["count"] = len(results["matches"])
            
            # Log the query for analytics
            self._log_query(
                query_text=f"pattern search with {len(node_patterns)} nodes and {len(edge_patterns)} edges",
                query_type="pattern_search",
                parameters=pattern,
                result_count=len(results["matches"])
            )
            
            return results
            
        except Exception as e:
            logger.exception(f"Error in pattern search: {e}")
            return {
                "status": "error",
                "message": f"Error in pattern search: {str(e)}"
            }
    
    def query_claims_by_entity(self, 
                              entity_id: int,
                              claim_type: str = None,
                              min_confidence: float = 0.0,
                              limit: int = 20) -> Dict[str, Any]:
        """
        Query claims related to a specific entity.
        
        Args:
            entity_id: ID of the entity
            claim_type: Optional type of claims to filter by
            min_confidence: Minimum confidence threshold
            limit: Maximum number of results
            
        Returns:
            Dictionary with matching claims
        """
        # Get the entity
        entity = EntityNode.query.get(entity_id)
        if not entity:
            logger.warning(f"Entity not found for claim query: {entity_id}")
            return {
                "status": "error",
                "message": "Entity not found"
            }
        
        # Find claims that mention this entity
        # These would be connected by "mentions" relationships
        claims = []
        
        # First get all "mentions" edges where this entity is the target
        mention_edges = KnowledgeEdge.query.filter_by(
            target_id=entity_id,
            relationship_type="mentions"
        ).all()
        
        # Get the source nodes (claims)
        for edge in mention_edges:
            claim = ClaimNode.query.get(edge.source_id)
            
            if claim:
                # Apply filters
                if claim_type and claim.claim_type != claim_type:
                    continue
                    
                if min_confidence > 0 and claim.confidence < min_confidence:
                    continue
                
                # Add to results
                claims.append(claim)
                
                # Stop if we've reached the limit
                if len(claims) >= limit:
                    break
        
        # Format results
        formatted_claims = []
        for claim in claims:
            formatted_claims.append({
                "id": claim.id,
                "content": claim.content,
                "claim_type": claim.claim_type,
                "sentiment": claim.sentiment,
                "confidence": claim.confidence,
                "is_refuted": claim.is_refuted
            })
        
        # Log the query for analytics
        self._log_query(
            query_text=f"claims mentioning entity {entity_id}",
            query_type="entity_claims",
            parameters={
                "entity_id": entity_id,
                "claim_type": claim_type,
                "min_confidence": min_confidence
            },
            result_count=len(formatted_claims)
        )
        
        return {
            "status": "success",
            "entity": {
                "id": entity.id,
                "name": entity.name,
                "type": entity.node_type
            },
            "claim_count": len(formatted_claims),
            "claims": formatted_claims
        }
    
    def query_events_by_timeframe(self, 
                                 start_date: datetime = None,
                                 end_date: datetime = None,
                                 tags: List[str] = None,
                                 limit: int = 20) -> Dict[str, Any]:
        """
        Query events within a specific timeframe.
        
        Args:
            start_date: Optional start date for the timeframe
            end_date: Optional end date for the timeframe
            tags: Optional list of tags to filter by
            limit: Maximum number of results
            
        Returns:
            Dictionary with matching events
        """
        # Build query for event nodes
        query = EventNode.query
        
        # Apply domain filter if specified
        if self.domain:
            query = query.filter(EventNode.domain == self.domain)
        
        # Apply date filters
        if start_date:
            query = query.filter(or_(
                EventNode.start_date >= start_date,
                EventNode.end_date >= start_date
            ))
            
        if end_date:
            query = query.filter(or_(
                EventNode.start_date <= end_date,
                EventNode.start_date == None
            ))
        
        # Apply tag filters
        if tags:
            for tag in tags:
                tag_obj = KnowledgeTag.query.filter_by(name=tag).first()
                if tag_obj:
                    query = query.filter(EventNode.tags.any(KnowledgeTag.id == tag_obj.id))
        
        # Order by start date
        query = query.order_by(EventNode.start_date)
        
        # Get results
        events = query.limit(limit).all()
        
        # Format results
        formatted_events = []
        for event in events:
            formatted_events.append({
                "id": event.id,
                "name": event.name,
                "start_date": event.start_date.isoformat() if event.start_date else None,
                "end_date": event.end_date.isoformat() if event.end_date else None,
                "location": event.location,
                "content": event.content,
                "confidence": event.confidence,
                "tags": [tag.name for tag in event.tags]
            })
        
        # Log the query for analytics
        self._log_query(
            query_text=f"events between {start_date} and {end_date}",
            query_type="timeframe_events",
            parameters={
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "tags": tags
            },
            result_count=len(formatted_events)
        )
        
        return {
            "status": "success",
            "event_count": len(formatted_events),
            "events": formatted_events
        }
    
    def _log_query(self, 
                  query_text: str,
                  query_type: str,
                  parameters: Dict = None,
                  result_count: int = None,
                  execution_time_ms: int = None):
        """
        Log a query for analytics.
        
        Args:
            query_text: Text representation of the query
            query_type: Type of query (e.g., "node_search", "path_finding")
            parameters: Optional dictionary of query parameters
            result_count: Optional count of results
            execution_time_ms: Optional execution time in milliseconds
        """
        try:
            query_log = KnowledgeQuery(
                query_text=query_text,
                query_type=query_type,
                parameters=parameters,
                result_count=result_count,
                execution_time_ms=execution_time_ms
            )
            
            db.session.add(query_log)
            db.session.commit()
            
        except Exception as e:
            logger.warning(f"Failed to log query: {e}")
            db.session.rollback()
