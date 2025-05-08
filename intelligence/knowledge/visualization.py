"""
Knowledge Graph Visualization

This module provides utilities for visualizing the knowledge graph,
including network graphs, hierarchies, and temporal views.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set, Any
import json

from db.models.knowledge_base import (
    KnowledgeNode, KnowledgeEdge, KnowledgeTag,
    EntityNode, ConceptNode, EventNode, ClaimNode
)

logger = logging.getLogger(__name__)


class GraphVisualizationGenerator:
    """
    Generates visualization data for knowledge graph components,
    supporting various visualization formats and libraries.
    """
    
    def __init__(self, domain: str = None):
        """
        Initialize the visualization generator.
        
        Args:
            domain: Optional domain to filter operations by (e.g., 'football')
        """
        self.domain = domain
        logger.info(f"Initialized graph visualization generator for domain: {domain}")
    
    def generate_graph_data(self, 
                           node_ids: List[int] = None,
                           relationship_types: List[str] = None,
                           max_nodes: int = 100) -> Dict[str, Any]:
        """
        Generate graph data for visualization.
        
        Args:
            node_ids: Optional list of node IDs to include
            relationship_types: Optional list of relationship types to include
            max_nodes: Maximum number of nodes to include
            
        Returns:
            Dictionary with graph data in a format suitable for visualization
        """
        # Query for nodes and edges
        nodes = []
        edges = []
        
        # If specific nodes are requested
        if node_ids:
            # Get the specified nodes
            for node_id in node_ids[:max_nodes]:
                node = KnowledgeNode.query.get(node_id)
                if node:
                    nodes.append(self._format_node_for_visualization(node))
                    
                    # Get related edges
                    for edge in node.outgoing_edges:
                        # Filter by relationship type if specified
                        if relationship_types and edge.relationship_type not in relationship_types:
                            continue
                            
                        # Only include edges between nodes in our list
                        if edge.target_id in node_ids:
                            edges.append(self._format_edge_for_visualization(edge))
        else:
            # Get nodes from the domain
            query = KnowledgeNode.query
            
            if self.domain:
                query = query.filter(KnowledgeNode.domain == self.domain)
            
            # Limit to max_nodes
            nodes_from_db = query.limit(max_nodes).all()
            
            # Format nodes
            for node in nodes_from_db:
                nodes.append(self._format_node_for_visualization(node))
            
            # Get node IDs
            node_ids = [node["id"] for node in nodes]
            
            # Get edges between these nodes
            for node_id in node_ids:
                node = KnowledgeNode.query.get(node_id)
                
                for edge in node.outgoing_edges:
                    # Only include edges between nodes in our list
                    if edge.target_id in node_ids:
                        # Filter by relationship type if specified
                        if relationship_types and edge.relationship_type not in relationship_types:
                            continue
                            
                        edges.append(self._format_edge_for_visualization(edge))
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    def _format_node_for_visualization(self, node: KnowledgeNode) -> Dict[str, Any]:
        """
        Format a node for visualization.
        
        Args:
            node: The knowledge node to format
            
        Returns:
            Dictionary with node visualization data
        """
        # Base node data
        node_data = {
            "id": node.id,
            "label": node.name,
            "type": node.node_type,
            "domain": node.domain,
            "tags": [tag.name for tag in node.tags]
        }
        
        # Add type-specific styling
        if node.node_type == "entity":
            node_data["color"] = "#4CAF50"  # Green
            node_data["shape"] = "circle"
            
        elif node.node_type == "concept":
            node_data["color"] = "#2196F3"  # Blue
            node_data["shape"] = "rectangle"
            
        elif node.node_type == "event":
            node_data["color"] = "#FFC107"  # Amber
            node_data["shape"] = "diamond"
            
            # Add date information for events
            if isinstance(node, EventNode):
                node_data["start_date"] = node.start_date.isoformat() if node.start_date else None
                node_data["end_date"] = node.end_date.isoformat() if node.end_date else None
            
        elif node.node_type == "claim":
            node_data["color"] = "#F44336"  # Red
            node_data["shape"] = "triangle"
            
            # Add claim type
            if isinstance(node, ClaimNode):
                node_data["claim_type"] = node.claim_type
        
        # Set size based on confidence
        confidence = node.confidence or 0.5
        node_data["size"] = 5 + confidence * 10  # Size 5-15 based on confidence
        
        # Add tooltip information
        node_data["title"] = f"{node.name} ({node.node_type})"
        
        return node_data
    
    def _format_edge_for_visualization(self, edge: KnowledgeEdge) -> Dict[str, Any]:
        """
        Format an edge for visualization.
        
        Args:
            edge: The knowledge edge to format
            
        Returns:
            Dictionary with edge visualization data
        """
        edge_data = {
            "id": edge.id,
            "from": edge.source_id,
            "to": edge.target_id,
            "label": edge.relationship_type,
            "type": edge.relationship_type
        }
        
        # Set width based on weight
        weight = edge.weight or 1.0
        edge_data["width"] = 1 + weight * 2  # Width 1-3 based on weight
        
        # Set opacity based on confidence
        confidence = edge.confidence or 0.5
        edge_data["opacity"] = confidence
        
        # Add temporal information if available
        if edge.valid_from or edge.valid_to:
            edge_data["valid_from"] = edge.valid_from.isoformat() if edge.valid_from else None
            edge_data["valid_to"] = edge.valid_to.isoformat() if edge.valid_to else None
            
            # Add timeline info
            edge_data["timeline"] = True
        
        # Add tooltip information
        edge_data["title"] = f"{edge.relationship_type} ({edge.confidence:.2f})"
        
        return edge_data
    
    def generate_temporal_view(self, 
                              start_date,
                              end_date,
                              node_type: str = "event",
                              tags: List[str] = None) -> Dict[str, Any]:
        """
        Generate timeline visualization data.
        
        Args:
            start_date: Start date for the timeline
            end_date: End date for the timeline
            node_type: Type of nodes to include (default: event)
            tags: Optional list of tags to filter by
            
        Returns:
            Dictionary with timeline data
        """
        # Query for nodes within the time range
        query = KnowledgeNode.query.filter(KnowledgeNode.node_type == node_type)
        
        if self.domain:
            query = query.filter(KnowledgeNode.domain == self.domain)
        
        # Apply tag filters
        if tags:
            for tag in tags:
                tag_obj = KnowledgeTag.query.filter_by(name=tag).first()
                if tag_obj:
                    query = query.filter(KnowledgeNode.tags.any(KnowledgeTag.id == tag_obj.id))
        
        # Get nodes
        if node_type == "event":
            # For events, filter by date range
            nodes = EventNode.query.filter(
                EventNode.start_date >= start_date,
                EventNode.start_date <= end_date
            ).all()
        else:
            # For other node types, just get all
            nodes = query.all()
        
        # Format for timeline visualization
        timeline_items = []
        
        for node in nodes:
            item = {
                "id": node.id,
                "content": node.name,
                "type": node.node_type
            }
            
            # Add date information
            if node_type == "event" and isinstance(node, EventNode):
                item["start"] = node.start_date.isoformat() if node.start_date else None
                item["end"] = node.end_date.isoformat() if node.end_date else None
                
                if not item["start"]:
                    continue  # Skip events without a start date
                
                # Add location if available
                if node.location:
                    item["location"] = node.location
            else:
                # For non-events, use created_at as the date
                item["start"] = node.created_at.isoformat() if node.created_at else None
                
                if not item["start"]:
                    continue  # Skip nodes without a date
            
            # Add style information
            item["className"] = f"timeline-item-{node.node_type}"
            
            # Add to timeline
            timeline_items.append(item)
        
        return {
            "timeline": True,
            "items": timeline_items
        }
    
    def generate_hierarchy_data(self, 
                               root_node_id: int,
                               max_depth: int = 3) -> Dict[str, Any]:
        """
        Generate hierarchical visualization data.
        
        Args:
            root_node_id: ID of the root node
            max_depth: Maximum depth to traverse
            
        Returns:
            Dictionary with hierarchy data
        """
        # Get root node
        root_node = KnowledgeNode.query.get(root_node_id)
        if not root_node:
            logger.warning(f"Root node not found: {root_node_id}")
            return {
                "status": "error",
                "message": "Root node not found"
            }
        
        # Generate the hierarchy data recursively
        hierarchy_data = self._build_hierarchy(root_node, max_depth, 0)
        
        return {
            "hierarchy": True,
            "root": hierarchy_data
        }
    
    def _build_hierarchy(self, 
                        node: KnowledgeNode,
                        max_depth: int,
                        current_depth: int) -> Dict[str, Any]:
        """
        Recursively build a hierarchy structure.
        
        Args:
            node: Current node
            max_depth: Maximum depth to traverse
            current_depth: Current depth in the hierarchy
            
        Returns:
            Dictionary with hierarchy data for this node and its children
        """
        # Format node data
        node_data = {
            "id": node.id,
            "name": node.name,
            "type": node.node_type,
            "confidence": node.confidence
        }
        
        # If we've reached max depth, don't include children
        if current_depth >= max_depth:
            return node_data
        
        # Get children via part_of or similar relationships
        hierarchy_relationships = ["part_of", "is_a", "instance_of", "subclass_of", "belongs_to"]
        
        children = []
        
        # Look for incoming edges with hierarchy relationships
        for edge in node.incoming_edges:
            if edge.relationship_type in hierarchy_relationships:
                child_node = KnowledgeNode.query.get(edge.source_id)
                if child_node:
                    # Recursively build child hierarchy
                    child_data = self._build_hierarchy(
                        child_node, 
                        max_depth, 
                        current_depth + 1
                    )
                    children.append(child_data)
        
        # Add children if any were found
        if children:
            node_data["children"] = children
        
        return node_data
    
    def export_to_vis_js(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export graph data to Vis.js format for network visualization.
        
        Args:
            graph_data: Graph data from generate_graph_data
            
        Returns:
            Dictionary with Vis.js formatted data
        """
        # Vis.js uses a specific format for nodes and edges
        nodes = []
        edges = []
        
        # Format nodes
        for node in graph_data.get("nodes", []):
            vis_node = {
                "id": node["id"],
                "label": node["label"],
                "title": node.get("title", node["label"]),
                "color": node.get("color", "#ccc"),
                "shape": node.get("shape", "dot"),
                "size": node.get("size", 10),
                "group": node.get("type", "node")
            }
            nodes.append(vis_node)
        
        # Format edges
        for edge in graph_data.get("edges", []):
            vis_edge = {
                "id": edge["id"],
                "from": edge["from"],
                "to": edge["to"],
                "label": edge["label"],
                "width": edge.get("width", 1),
                "title": edge.get("title", edge["label"]),
                "arrows": "to",
                "smooth": {
                    "type": "curvedCW",
                    "roundness": 0.2
                }
            }
            
            # Add color based on type
            if "type" in edge:
                if edge["type"] in ["plays_for", "member_of", "part_of"]:
                    vis_edge["color"] = {"color": "#4CAF50", "opacity": edge.get("opacity", 1.0)}
                elif edge["type"] in ["manages", "leads", "directs"]:
                    vis_edge["color"] = {"color": "#2196F3", "opacity": edge.get("opacity", 1.0)}
                elif edge["type"] in ["mentions", "refers_to", "about"]:
                    vis_edge["color"] = {"color": "#FFC107", "opacity": edge.get("opacity", 1.0)}
                else:
                    vis_edge["color"] = {"color": "#9E9E9E", "opacity": edge.get("opacity", 1.0)}
            
            edges.append(vis_edge)
        
        return {
            "format": "vis_js",
            "nodes": nodes,
            "edges": edges,
            "options": {
                "nodes": {
                    "font": {
                        "face": "Roboto",
                        "size": 14
                    }
                },
                "edges": {
                    "font": {
                        "face": "Roboto",
                        "size": 12
                    },
                    "length": 200
                },
                "physics": {
                    "enabled": True,
                    "solver": "forceAtlas2Based",
                    "forceAtlas2Based": {
                        "gravitationalConstant": -50,
                        "springLength": 100,
                        "springConstant": 0.08
                    },
                    "stabilization": {
                        "iterations": 150
                    }
                }
            }
        }
    
    def export_to_d3_force(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export graph data to D3.js force layout format.
        
        Args:
            graph_data: Graph data from generate_graph_data
            
        Returns:
            Dictionary with D3.js formatted data
        """
        # D3 force layout format
        nodes = []
        links = []
        
        # Format nodes
        for node in graph_data.get("nodes", []):
            d3_node = {
                "id": node["id"],
                "name": node["label"],
                "group": node.get("type", "node"),
                "r": node.get("size", 10),  # radius
                "fill": node.get("color", "#ccc")
            }
            nodes.append(d3_node)
        
        # Format edges
        for edge in graph_data.get("edges", []):
            d3_link = {
                "source": edge["from"],
                "target": edge["to"],
                "value": edge.get("width", 1),
                "label": edge["label"],
                "type": edge.get("type", "link")
            }
            links.append(d3_link)
        
        return {
            "format": "d3_force",
            "nodes": nodes,
            "links": links
        }
