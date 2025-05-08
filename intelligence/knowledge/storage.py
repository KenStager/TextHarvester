"""
Knowledge Storage

This module provides the KnowledgeStorage class for efficient storage
and retrieval of knowledge graph data, including batch operations.
"""

import logging
import json
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import func, and_, or_

from app import db
from db.models.knowledge_base import (
    KnowledgeNode, KnowledgeEdge, KnowledgeSource, 
    KnowledgeContradiction, KnowledgeTag
)

logger = logging.getLogger(__name__)


class KnowledgeStorage:
    """
    Manages efficient storage and retrieval of knowledge graph data,
    including batch operations and persistence.
    """
    
    def __init__(self, domain: str = None):
        """
        Initialize the knowledge storage manager.
        
        Args:
            domain: Optional domain to filter operations by (e.g., 'football')
        """
        self.domain = domain
        logger.info(f"Initialized knowledge storage manager for domain: {domain}")
    
    def batch_store_nodes(self, nodes_data: List[Dict]) -> Dict[str, Any]:
        """
        Store multiple knowledge nodes in a batch operation.
        
        Args:
            nodes_data: List of dictionaries with node data
            
        Returns:
            Dictionary with operation results
        """
        results = {
            "total": len(nodes_data),
            "successful": 0,
            "failed": 0,
            "node_ids": [],
            "errors": []
        }
        
        for i, node_data in enumerate(nodes_data):
            try:
                # Extract common fields
                node_type = node_data.get("node_type", "entity")
                name = node_data.get("name")
                
                if not name:
                    results["failed"] += 1
                    results["errors"].append({
                        "index": i,
                        "error": "Missing required field 'name'"
                    })
                    continue
                
                # Create new node based on type
                node = None
                
                if node_type == "entity":
                    from .graph import KnowledgeGraph
                    graph = KnowledgeGraph(domain=self.domain)
                    node = graph.create_entity_node(
                        name=name,
                        entity_id=node_data.get("entity_id"),
                        attributes=node_data.get("attributes"),
                        tags=node_data.get("tags"),
                        confidence=node_data.get("confidence", 1.0)
                    )
                
                elif node_type == "concept":
                    from .graph import KnowledgeGraph
                    graph = KnowledgeGraph(domain=self.domain)
                    node = graph.create_concept_node(
                        name=name,
                        content=node_data.get("content"),
                        attributes=node_data.get("attributes"),
                        tags=node_data.get("tags"),
                        confidence=node_data.get("confidence", 1.0)
                    )
                
                elif node_type == "event":
                    from .graph import KnowledgeGraph
                    graph = KnowledgeGraph(domain=self.domain)
                    node = graph.create_event_node(
                        name=name,
                        start_date=node_data.get("start_date"),
                        end_date=node_data.get("end_date"),
                        location=node_data.get("location"),
                        content=node_data.get("content"),
                        attributes=node_data.get("attributes"),
                        tags=node_data.get("tags"),
                        confidence=node_data.get("confidence", 1.0)
                    )
                
                elif node_type == "claim":
                    from .graph import KnowledgeGraph
                    graph = KnowledgeGraph(domain=self.domain)
                    node = graph.create_claim_node(
                        content=node_data.get("content") or name,
                        name=name,
                        claim_type=node_data.get("claim_type", "factual"),
                        sentiment=node_data.get("sentiment"),
                        attributes=node_data.get("attributes"),
                        tags=node_data.get("tags"),
                        confidence=node_data.get("confidence", 1.0)
                    )
                
                else:
                    results["failed"] += 1
                    results["errors"].append({
                        "index": i,
                        "error": f"Invalid node_type: {node_type}"
                    })
                    continue
                
                if node:
                    results["successful"] += 1
                    results["node_ids"].append(node.id)
                    
                    # Add source if provided
                    if "source" in node_data and "content_id" in node_data["source"]:
                        from .graph import KnowledgeGraph
                        graph = KnowledgeGraph(domain=self.domain)
                        graph.add_source(
                            node_id=node.id,
                            content_id=node_data["source"]["content_id"],
                            confidence=node_data["source"].get("confidence", 1.0),
                            extraction_method=node_data["source"].get("extraction_method"),
                            excerpt=node_data["source"].get("excerpt"),
                            context=node_data["source"].get("context")
                        )
                else:
                    results["failed"] += 1
                    results["errors"].append({
                        "index": i,
                        "error": "Failed to create node"
                    })
            
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({
                    "index": i,
                    "error": str(e)
                })
                logger.exception(f"Error storing node at index {i}: {e}")
        
        return results
    
    def batch_store_edges(self, edges_data: List[Dict]) -> Dict[str, Any]:
        """
        Store multiple knowledge edges in a batch operation.
        
        Args:
            edges_data: List of dictionaries with edge data
            
        Returns:
            Dictionary with operation results
        """
        results = {
            "total": len(edges_data),
            "successful": 0,
            "failed": 0,
            "edge_ids": [],
            "errors": []
        }
        
        for i, edge_data in enumerate(edges_data):
            try:
                # Extract required fields
                source_id = edge_data.get("source_id")
                target_id = edge_data.get("target_id")
                relationship_type = edge_data.get("relationship_type")
                
                if not source_id or not target_id or not relationship_type:
                    results["failed"] += 1
                    results["errors"].append({
                        "index": i,
                        "error": "Missing required fields: source_id, target_id, or relationship_type"
                    })
                    continue
                
                # Create the edge
                from .graph import KnowledgeGraph
                graph = KnowledgeGraph(domain=self.domain)
                
                edge = graph.create_edge(
                    source_id=source_id,
                    target_id=target_id,
                    relationship_type=relationship_type,
                    weight=edge_data.get("weight", 1.0),
                    confidence=edge_data.get("confidence", 1.0),
                    attributes=edge_data.get("attributes"),
                    valid_from=edge_data.get("valid_from"),
                    valid_to=edge_data.get("valid_to")
                )
                
                if edge:
                    results["successful"] += 1
                    results["edge_ids"].append(edge.id)
                    
                    # Add source if provided
                    if "source" in edge_data and "content_id" in edge_data["source"]:
                        graph.add_source(
                            edge_id=edge.id,
                            content_id=edge_data["source"]["content_id"],
                            confidence=edge_data["source"].get("confidence", 1.0),
                            extraction_method=edge_data["source"].get("extraction_method"),
                            excerpt=edge_data["source"].get("excerpt"),
                            context=edge_data["source"].get("context")
                        )
                else:
                    results["failed"] += 1
                    results["errors"].append({
                        "index": i,
                        "error": "Failed to create edge"
                    })
            
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({
                    "index": i,
                    "error": str(e)
                })
                logger.exception(f"Error storing edge at index {i}: {e}")
        
        return results
    
    def import_knowledge_graph(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import a complete knowledge graph from a dictionary representation.
        
        Args:
            graph_data: Dictionary with nodes and edges data
            
        Returns:
            Dictionary with import results
        """
        results = {
            "nodes_imported": 0,
            "edges_imported": 0,
            "tags_imported": 0,
            "failed": 0,
            "errors": []
        }
        
        # Start a transaction
        try:
            # First import tags
            tags_data = graph_data.get("tags", [])
            tag_map = {}  # Map original IDs to new IDs
            
            for tag_data in tags_data:
                try:
                    name = tag_data.get("name")
                    if not name:
                        continue
                        
                    # Check if tag already exists
                    tag = KnowledgeTag.query.filter_by(name=name).first()
                    if not tag:
                        tag = KnowledgeTag(
                            name=name,
                            description=tag_data.get("description")
                        )
                        db.session.add(tag)
                        db.session.flush()  # Get ID without committing
                    
                    # Map original ID to new ID
                    if "id" in tag_data:
                        tag_map[tag_data["id"]] = tag.id
                        
                    results["tags_imported"] += 1
                
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(str(e))
            
            # Import nodes
            nodes_data = graph_data.get("nodes", [])
            node_map = {}  # Map original IDs to new IDs
            
            for node_data in nodes_data:
                try:
                    # Extract node data
                    original_id = node_data.get("id")
                    node_type = node_data.get("node_type", "entity")
                    name = node_data.get("name")
                    
                    if not name:
                        results["failed"] += 1
                        results["errors"].append(f"Missing name for node: {original_id}")
                        continue
                    
                    # Adjust tags using the tag map
                    tags = []
                    for tag_id in node_data.get("tag_ids", []):
                        if tag_id in tag_map:
                            tag = KnowledgeTag.query.get(tag_map[tag_id])
                            if tag:
                                tags.append(tag.name)
                    
                    # Create node based on type
                    node = None
                    if node_type == "entity":
                        node = KnowledgeNode(
                            node_type="entity",
                            name=name,
                            canonical_name=node_data.get("canonical_name"),
                            domain=self.domain or node_data.get("domain", "general"),
                            content=node_data.get("content"),
                            attributes=node_data.get("attributes"),
                            confidence=node_data.get("confidence", 1.0),
                            verified=node_data.get("verified", False)
                        )
                    
                    elif node_type == "concept":
                        node = KnowledgeNode(
                            node_type="concept",
                            name=name,
                            canonical_name=node_data.get("canonical_name"),
                            domain=self.domain or node_data.get("domain", "general"),
                            content=node_data.get("content"),
                            attributes=node_data.get("attributes"),
                            confidence=node_data.get("confidence", 1.0),
                            verified=node_data.get("verified", False)
                        )
                    
                    # Add other node types as needed...
                    
                    if node:
                        db.session.add(node)
                        db.session.flush()  # Get ID without committing
                        
                        # Add tags
                        for tag_name in tags:
                            tag = KnowledgeTag.query.filter_by(name=tag_name).first()
                            if tag:
                                node.tags.append(tag)
                        
                        # Map original ID to new ID
                        if original_id:
                            node_map[original_id] = node.id
                            
                        results["nodes_imported"] += 1
                
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(str(e))
                    logger.exception(f"Error importing node: {e}")
            
            # Import edges
            edges_data = graph_data.get("edges", [])
            
            for edge_data in edges_data:
                try:
                    # Extract edge data
                    original_source_id = edge_data.get("source_id")
                    original_target_id = edge_data.get("target_id")
                    relationship_type = edge_data.get("relationship_type")
                    
                    if not original_source_id or not original_target_id or not relationship_type:
                        results["failed"] += 1
                        results["errors"].append("Missing required edge data")
                        continue
                    
                    # Map to new IDs
                    source_id = node_map.get(original_source_id)
                    target_id = node_map.get(original_target_id)
                    
                    if not source_id or not target_id:
                        results["failed"] += 1
                        results["errors"].append(f"Could not map source or target IDs: {original_source_id} -> {original_target_id}")
                        continue
                    
                    # Create edge
                    edge = KnowledgeEdge(
                        source_id=source_id,
                        target_id=target_id,
                        relationship_type=relationship_type,
                        weight=edge_data.get("weight", 1.0),
                        confidence=edge_data.get("confidence", 1.0),
                        attributes=edge_data.get("attributes"),
                        valid_from=edge_data.get("valid_from"),
                        valid_to=edge_data.get("valid_to")
                    )
                    
                    db.session.add(edge)
                    results["edges_imported"] += 1
                
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(str(e))
                    logger.exception(f"Error importing edge: {e}")
            
            # Commit the transaction
            db.session.commit()
            
        except Exception as e:
            db.session.rollback()
            results["errors"].append(f"Transaction failed: {str(e)}")
            logger.exception(f"Import transaction failed: {e}")
        
        return results
    
    def export_knowledge_graph(self, 
                              include_nodes: bool = True,
                              include_edges: bool = True,
                              node_types: List[str] = None,
                              relationship_types: List[str] = None,
                              min_confidence: float = 0.0,
                              exclude_sources: bool = False) -> Dict[str, Any]:
        """
        Export the knowledge graph to a dictionary representation.
        
        Args:
            include_nodes: Whether to include nodes
            include_edges: Whether to include edges
            node_types: Optional list of node types to include
            relationship_types: Optional list of relationship types to include
            min_confidence: Minimum confidence threshold
            exclude_sources: Whether to exclude source information
            
        Returns:
            Dictionary with the knowledge graph data
        """
        export_data = {
            "metadata": {
                "exported_at": datetime.utcnow().isoformat(),
                "domain": self.domain,
                "node_count": 0,
                "edge_count": 0,
                "tag_count": 0
            },
            "tags": [],
            "nodes": [],
            "edges": []
        }
        
        try:
            # Export tags
            tags = KnowledgeTag.query.all()
            
            for tag in tags:
                export_data["tags"].append({
                    "id": tag.id,
                    "name": tag.name,
                    "description": tag.description
                })
            
            export_data["metadata"]["tag_count"] = len(export_data["tags"])
            
            # Export nodes if requested
            if include_nodes:
                # Build query
                query = KnowledgeNode.query
                
                if self.domain:
                    query = query.filter(KnowledgeNode.domain == self.domain)
                
                if node_types:
                    query = query.filter(KnowledgeNode.node_type.in_(node_types))
                
                if min_confidence > 0:
                    query = query.filter(KnowledgeNode.confidence >= min_confidence)
                
                nodes = query.all()
                
                for node in nodes:
                    node_data = {
                        "id": node.id,
                        "node_type": node.node_type,
                        "name": node.name,
                        "canonical_name": node.canonical_name,
                        "domain": node.domain,
                        "content": node.content,
                        "attributes": node.attributes,
                        "confidence": node.confidence,
                        "verified": node.verified,
                        "created_at": node.created_at.isoformat() if node.created_at else None,
                        "updated_at": node.updated_at.isoformat() if node.updated_at else None,
                        "tag_ids": [tag.id for tag in node.tags]
                    }
                    
                    # Add sources if requested
                    if not exclude_sources:
                        node_data["sources"] = []
                        for source in node.sources:
                            node_data["sources"].append({
                                "content_id": source.content_id,
                                "extraction_method": source.extraction_method,
                                "confidence": source.confidence,
                                "excerpt": source.excerpt,
                                "created_at": source.created_at.isoformat() if source.created_at else None
                            })
                    
                    export_data["nodes"].append(node_data)
                
                export_data["metadata"]["node_count"] = len(export_data["nodes"])
            
            # Export edges if requested
            if include_edges:
                # Build query
                query = KnowledgeEdge.query
                
                if relationship_types:
                    query = query.filter(KnowledgeEdge.relationship_type.in_(relationship_types))
                
                if min_confidence > 0:
                    query = query.filter(KnowledgeEdge.confidence >= min_confidence)
                
                # If domain is specified, filter edges where both nodes are in the domain
                if self.domain:
                    source_node = db.aliased(KnowledgeNode)
                    target_node = db.aliased(KnowledgeNode)
                    
                    query = (query
                            .join(source_node, KnowledgeEdge.source_id == source_node.id)
                            .join(target_node, KnowledgeEdge.target_id == target_node.id)
                            .filter(source_node.domain == self.domain)
                            .filter(target_node.domain == self.domain))
                
                edges = query.all()
                
                for edge in edges:
                    edge_data = {
                        "id": edge.id,
                        "source_id": edge.source_id,
                        "target_id": edge.target_id,
                        "relationship_type": edge.relationship_type,
                        "weight": edge.weight,
                        "confidence": edge.confidence,
                        "attributes": edge.attributes,
                        "valid_from": edge.valid_from.isoformat() if edge.valid_from else None,
                        "valid_to": edge.valid_to.isoformat() if edge.valid_to else None,
                        "created_at": edge.created_at.isoformat() if edge.created_at else None,
                        "updated_at": edge.updated_at.isoformat() if edge.updated_at else None
                    }
                    
                    # Add sources if requested
                    if not exclude_sources:
                        edge_data["sources"] = []
                        for source in edge.sources:
                            edge_data["sources"].append({
                                "content_id": source.content_id,
                                "extraction_method": source.extraction_method,
                                "confidence": source.confidence,
                                "excerpt": source.excerpt,
                                "created_at": source.created_at.isoformat() if source.created_at else None
                            })
                    
                    export_data["edges"].append(edge_data)
                
                export_data["metadata"]["edge_count"] = len(export_data["edges"])
            
            return export_data
            
        except Exception as e:
            logger.exception(f"Error exporting knowledge graph: {e}")
            return {
                "metadata": {
                    "error": str(e),
                    "exported_at": datetime.utcnow().isoformat()
                },
                "tags": [],
                "nodes": [],
                "edges": []
            }
    
    def export_to_json_file(self, 
                           filepath: str,
                           **kwargs) -> Dict[str, Any]:
        """
        Export the knowledge graph to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
            **kwargs: Additional arguments for export_knowledge_graph
            
        Returns:
            Dictionary with export results
        """
        try:
            # Get the export data
            export_data = self.export_knowledge_graph(**kwargs)
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            return {
                "status": "success",
                "filepath": filepath,
                "node_count": export_data["metadata"]["node_count"],
                "edge_count": export_data["metadata"]["edge_count"],
                "tag_count": export_data["metadata"]["tag_count"]
            }
            
        except Exception as e:
            logger.exception(f"Error exporting to JSON file: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def import_from_json_file(self, filepath: str) -> Dict[str, Any]:
        """
        Import the knowledge graph from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Dictionary with import results
        """
        try:
            # Read from file
            with open(filepath, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            # Import the data
            import_results = self.import_knowledge_graph(graph_data)
            
            return {
                "status": "success",
                "filepath": filepath,
                "nodes_imported": import_results["nodes_imported"],
                "edges_imported": import_results["edges_imported"],
                "tags_imported": import_results["tags_imported"],
                "failed": import_results["failed"],
                "errors": import_results["errors"]
            }
            
        except Exception as e:
            logger.exception(f"Error importing from JSON file: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def clear_domain_knowledge(self, confirm_domain: str = None) -> Dict[str, Any]:
        """
        Clear all knowledge for the current domain.
        
        Args:
            confirm_domain: Domain name confirmation (must match self.domain)
            
        Returns:
            Dictionary with results
        """
        if not self.domain:
            return {
                "status": "error",
                "error": "No domain specified. Set domain on initialization."
            }
        
        if not confirm_domain or confirm_domain != self.domain:
            return {
                "status": "error",
                "error": f"Domain confirmation '{confirm_domain}' does not match current domain '{self.domain}'"
            }
        
        try:
            # Find nodes in this domain
            nodes = KnowledgeNode.query.filter_by(domain=self.domain).all()
            node_ids = [node.id for node in nodes]
            
            # Count edges associated with these nodes
            edges_count = KnowledgeEdge.query.filter(
                or_(
                    KnowledgeEdge.source_id.in_(node_ids),
                    KnowledgeEdge.target_id.in_(node_ids)
                )
            ).count()
            
            # Delete nodes (will cascade to edges and sources due to relationship config)
            for node in nodes:
                db.session.delete(node)
            
            db.session.commit()
            
            return {
                "status": "success",
                "domain": self.domain,
                "nodes_deleted": len(nodes),
                "edges_deleted": edges_count
            }
            
        except Exception as e:
            db.session.rollback()
            logger.exception(f"Error clearing domain knowledge: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge storage.
        
        Returns:
            Dictionary with storage statistics
        """
        try:
            stats = {
                "nodes": {
                    "total": KnowledgeNode.query.count(),
                    "by_type": {},
                    "by_domain": {}
                },
                "edges": {
                    "total": KnowledgeEdge.query.count(),
                    "by_type": {}
                },
                "tags": {
                    "total": KnowledgeTag.query.count()
                },
                "sources": {
                    "total": KnowledgeSource.query.count()
                },
                "contradictions": {
                    "total": KnowledgeContradiction.query.count(),
                    "unresolved": KnowledgeContradiction.query.filter_by(resolution_status="unresolved").count()
                }
            }
            
            # Node stats by type
            node_type_counts = db.session.query(
                KnowledgeNode.node_type, 
                func.count(KnowledgeNode.id)
            ).group_by(KnowledgeNode.node_type).all()
            
            for node_type, count in node_type_counts:
                stats["nodes"]["by_type"][node_type] = count
            
            # Node stats by domain
            if self.domain:
                stats["nodes"]["current_domain"] = KnowledgeNode.query.filter_by(domain=self.domain).count()
            else:
                domain_counts = db.session.query(
                    KnowledgeNode.domain, 
                    func.count(KnowledgeNode.id)
                ).group_by(KnowledgeNode.domain).all()
                
                for domain, count in domain_counts:
                    stats["nodes"]["by_domain"][domain] = count
            
            # Edge stats by type
            edge_type_counts = db.session.query(
                KnowledgeEdge.relationship_type, 
                func.count(KnowledgeEdge.id)
            ).group_by(KnowledgeEdge.relationship_type).all()
            
            for edge_type, count in edge_type_counts:
                stats["edges"]["by_type"][edge_type] = count
            
            return stats
            
        except Exception as e:
            logger.exception(f"Error getting storage stats: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
