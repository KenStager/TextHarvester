"""
Knowledge Graph API Routes

This module provides API routes for interacting with the knowledge graph,
including querying, management, and processing.
"""

import logging
from flask import Blueprint, request, jsonify
from datetime import datetime

from intelligence.knowledge.graph import KnowledgeGraph
from intelligence.knowledge.extraction import KnowledgeExtractor
from intelligence.knowledge.storage import KnowledgeStorage
from intelligence.knowledge.conflict import ContradictionDetector
from intelligence.knowledge.credibility import CredibilityScorer
from intelligence.knowledge.queries import KnowledgeQuerier
from intelligence.knowledge.pipeline import KnowledgePipeline
from intelligence.knowledge.visualization import GraphVisualizationGenerator

logger = logging.getLogger(__name__)

# Create blueprint
bp = Blueprint('knowledge', __name__, url_prefix='/knowledge')


# Node Routes
@bp.route('/nodes', methods=['GET'])
def get_nodes():
    """Get nodes matching search criteria."""
    # Extract query parameters
    query = request.args.get('q')
    node_types = request.args.getlist('type')
    tags = request.args.getlist('tag')
    domain = request.args.get('domain')
    limit = request.args.get('limit', 20, type=int)
    offset = request.args.get('offset', 0, type=int)
    min_confidence = request.args.get('min_confidence', 0.0, type=float)
    
    # Initialize querier with domain
    querier = KnowledgeQuerier(domain=domain)
    
    # Search for nodes
    results = querier.search_nodes(
        query=query,
        node_types=node_types or None,
        tags=tags or None,
        min_confidence=min_confidence,
        limit=limit,
        offset=offset
    )
    
    return jsonify(results)


@bp.route('/nodes/<int:node_id>', methods=['GET'])
def get_node(node_id):
    """Get a specific node by ID."""
    domain = request.args.get('domain')
    graph = KnowledgeGraph(domain=domain)
    
    node = graph.get_node(node_id)
    if not node:
        return jsonify({"error": "Node not found"}), 404
    
    # Get node details including relationships
    relationships = KnowledgeQuerier(domain=domain).get_node_relationships(
        node_id=node_id,
        limit=100
    )
    
    # Format node details
    result = {
        "id": node.id,
        "name": node.name,
        "type": node.node_type,
        "domain": node.domain,
        "canonical_name": node.canonical_name,
        "content": node.content,
        "attributes": node.attributes,
        "confidence": node.confidence,
        "verified": node.verified,
        "created_at": node.created_at.isoformat() if node.created_at else None,
        "updated_at": node.updated_at.isoformat() if node.updated_at else None,
        "tags": [{"id": tag.id, "name": tag.name} for tag in node.tags],
        "relationships": relationships
    }
    
    return jsonify(result)


@bp.route('/nodes', methods=['POST'])
def create_node():
    """Create a new knowledge node."""
    data = request.json
    domain = data.get('domain')
    
    if not data or not data.get('name') or not data.get('node_type'):
        return jsonify({"error": "Missing required fields: name, node_type"}), 400
    
    graph = KnowledgeGraph(domain=domain)
    
    try:
        # Create node based on type
        node_type = data.get('node_type')
        
        if node_type == 'entity':
            node = graph.create_entity_node(
                name=data.get('name'),
                entity_id=data.get('entity_id'),
                attributes=data.get('attributes'),
                tags=data.get('tags'),
                confidence=data.get('confidence', 1.0)
            )
        
        elif node_type == 'concept':
            node = graph.create_concept_node(
                name=data.get('name'),
                content=data.get('content'),
                attributes=data.get('attributes'),
                tags=data.get('tags'),
                confidence=data.get('confidence', 1.0)
            )
        
        elif node_type == 'event':
            # Parse dates if provided
            start_date = None
            if data.get('start_date'):
                try:
                    start_date = datetime.fromisoformat(data.get('start_date').replace('Z', '+00:00'))
                except ValueError:
                    pass
            
            end_date = None
            if data.get('end_date'):
                try:
                    end_date = datetime.fromisoformat(data.get('end_date').replace('Z', '+00:00'))
                except ValueError:
                    pass
            
            node = graph.create_event_node(
                name=data.get('name'),
                start_date=start_date,
                end_date=end_date,
                location=data.get('location'),
                content=data.get('content'),
                attributes=data.get('attributes'),
                tags=data.get('tags'),
                confidence=data.get('confidence', 1.0)
            )
        
        elif node_type == 'claim':
            node = graph.create_claim_node(
                content=data.get('content') or data.get('name'),
                name=data.get('name'),
                claim_type=data.get('claim_type', 'factual'),
                sentiment=data.get('sentiment'),
                attributes=data.get('attributes'),
                tags=data.get('tags'),
                confidence=data.get('confidence', 1.0)
            )
        
        else:
            return jsonify({"error": f"Invalid node type: {node_type}"}), 400
        
        # Add source information if provided
        if 'source' in data and 'content_id' in data['source']:
            graph.add_source(
                node_id=node.id,
                content_id=data['source']['content_id'],
                confidence=data['source'].get('confidence', 1.0),
                extraction_method=data['source'].get('extraction_method'),
                excerpt=data['source'].get('excerpt'),
                context=data['source'].get('context')
            )
        
        return jsonify({
            "status": "success",
            "node_id": node.id,
            "message": f"Created {node_type} node: {node.name}"
        }), 201
    
    except Exception as e:
        logger.exception(f"Error creating node: {e}")
        return jsonify({"error": str(e)}), 500


@bp.route('/nodes/<int:node_id>', methods=['PUT', 'PATCH'])
def update_node(node_id):
    """Update an existing knowledge node."""
    data = request.json
    domain = data.get('domain')
    
    if not data:
        return jsonify({"error": "No update data provided"}), 400
    
    graph = KnowledgeGraph(domain=domain)
    
    try:
        # Extract update fields
        attributes = data.get('attributes')
        content = data.get('content') if 'content' in data else None
        tags = data.get('tags')
        confidence = data.get('confidence') if 'confidence' in data else None
        
        # Update the node
        node = graph.update_node(
            node_id=node_id,
            attributes=attributes,
            content=content,
            tags=tags,
            confidence=confidence
        )
        
        if not node:
            return jsonify({"error": "Node not found"}), 404
        
        return jsonify({
            "status": "success",
            "node_id": node.id,
            "message": f"Updated node: {node.name}"
        })
    
    except Exception as e:
        logger.exception(f"Error updating node: {e}")
        return jsonify({"error": str(e)}), 500


@bp.route('/nodes/<int:node_id>', methods=['DELETE'])
def delete_node(node_id):
    """Delete a knowledge node."""
    domain = request.args.get('domain')
    graph = KnowledgeGraph(domain=domain)
    
    try:
        # Delete the node
        success = graph.delete_node(node_id)
        
        if not success:
            return jsonify({"error": "Node not found"}), 404
        
        return jsonify({
            "status": "success",
            "message": f"Deleted node: {node_id}"
        })
    
    except Exception as e:
        logger.exception(f"Error deleting node: {e}")
        return jsonify({"error": str(e)}), 500


# Edge Routes
@bp.route('/edges', methods=['POST'])
def create_edge():
    """Create a new knowledge edge between nodes."""
    data = request.json
    domain = data.get('domain')
    
    if not data or not data.get('source_id') or not data.get('target_id') or not data.get('relationship_type'):
        return jsonify({"error": "Missing required fields: source_id, target_id, relationship_type"}), 400
    
    graph = KnowledgeGraph(domain=domain)
    
    try:
        # Create the edge
        edge = graph.create_edge(
            source_id=data.get('source_id'),
            target_id=data.get('target_id'),
            relationship_type=data.get('relationship_type'),
            weight=data.get('weight', 1.0),
            confidence=data.get('confidence', 1.0),
            attributes=data.get('attributes'),
            valid_from=data.get('valid_from'),
            valid_to=data.get('valid_to')
        )
        
        if not edge:
            return jsonify({"error": "Could not create edge; check if nodes exist"}), 400
        
        # Add source information if provided
        if 'source' in data and 'content_id' in data['source']:
            graph.add_source(
                edge_id=edge.id,
                content_id=data['source']['content_id'],
                confidence=data['source'].get('confidence', 1.0),
                extraction_method=data['source'].get('extraction_method'),
                excerpt=data['source'].get('excerpt'),
                context=data['source'].get('context')
            )
        
        return jsonify({
            "status": "success",
            "edge_id": edge.id,
            "message": f"Created edge: {edge.relationship_type} from {edge.source_id} to {edge.target_id}"
        }), 201
    
    except Exception as e:
        logger.exception(f"Error creating edge: {e}")
        return jsonify({"error": str(e)}), 500


@bp.route('/edges/<int:edge_id>', methods=['GET'])
def get_edge(edge_id):
    """Get a specific edge by ID."""
    domain = request.args.get('domain')
    graph = KnowledgeGraph(domain=domain)
    
    edge = graph.get_edge(edge_id)
    if not edge:
        return jsonify({"error": "Edge not found"}), 404
    
    # Get source and target nodes
    source_node = graph.get_node(edge.source_id)
    target_node = graph.get_node(edge.target_id)
    
    # Format edge details
    result = {
        "id": edge.id,
        "relationship_type": edge.relationship_type,
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "source": {
            "id": source_node.id,
            "name": source_node.name,
            "type": source_node.node_type
        } if source_node else None,
        "target": {
            "id": target_node.id,
            "name": target_node.name,
            "type": target_node.node_type
        } if target_node else None,
        "weight": edge.weight,
        "confidence": edge.confidence,
        "attributes": edge.attributes,
        "valid_from": edge.valid_from.isoformat() if edge.valid_from else None,
        "valid_to": edge.valid_to.isoformat() if edge.valid_to else None,
        "created_at": edge.created_at.isoformat() if edge.created_at else None,
        "updated_at": edge.updated_at.isoformat() if edge.updated_at else None
    }
    
    return jsonify(result)


@bp.route('/edges/<int:edge_id>', methods=['PUT', 'PATCH'])
def update_edge(edge_id):
    """Update an existing knowledge edge."""
    data = request.json
    domain = data.get('domain')
    
    if not data:
        return jsonify({"error": "No update data provided"}), 400
    
    graph = KnowledgeGraph(domain=domain)
    
    try:
        # Extract update fields
        weight = data.get('weight') if 'weight' in data else None
        confidence = data.get('confidence') if 'confidence' in data else None
        attributes = data.get('attributes')
        valid_to = data.get('valid_to')
        
        if valid_to:
            try:
                valid_to = datetime.fromisoformat(valid_to.replace('Z', '+00:00'))
            except ValueError:
                valid_to = None
        
        # Update the edge
        edge = graph.update_edge(
            edge_id=edge_id,
            weight=weight,
            confidence=confidence,
            attributes=attributes,
            valid_to=valid_to
        )
        
        if not edge:
            return jsonify({"error": "Edge not found"}), 404
        
        return jsonify({
            "status": "success",
            "edge_id": edge.id,
            "message": f"Updated edge: {edge.relationship_type} from {edge.source_id} to {edge.target_id}"
        })
    
    except Exception as e:
        logger.exception(f"Error updating edge: {e}")
        return jsonify({"error": str(e)}), 500


@bp.route('/edges/<int:edge_id>', methods=['DELETE'])
def delete_edge(edge_id):
    """Delete a knowledge edge."""
    domain = request.args.get('domain')
    graph = KnowledgeGraph(domain=domain)
    
    try:
        # Delete the edge
        success = graph.delete_edge(edge_id)
        
        if not success:
            return jsonify({"error": "Edge not found"}), 404
        
        return jsonify({
            "status": "success",
            "message": f"Deleted edge: {edge_id}"
        })
    
    except Exception as e:
        logger.exception(f"Error deleting edge: {e}")
        return jsonify({"error": str(e)}), 500


# Query Routes
@bp.route('/query/paths', methods=['GET'])
def find_paths():
    """Find paths between nodes in the knowledge graph."""
    # Extract query parameters
    start_node_id = request.args.get('start', type=int)
    end_node_id = request.args.get('end', type=int)
    max_length = request.args.get('max_length', 3, type=int)
    min_confidence = request.args.get('min_confidence', 0.0, type=float)
    domain = request.args.get('domain')
    
    if not start_node_id or not end_node_id:
        return jsonify({"error": "Missing required parameters: start, end"}), 400
    
    # Initialize querier with domain
    querier = KnowledgeQuerier(domain=domain)
    
    # Find paths
    results = querier.find_paths(
        start_node_id=start_node_id,
        end_node_id=end_node_id,
        max_length=max_length,
        min_confidence=min_confidence
    )
    
    return jsonify(results)


@bp.route('/query/subgraph', methods=['GET'])
def get_subgraph():
    """Get a subgraph centered around a specific node."""
    # Extract query parameters
    node_id = request.args.get('node_id', type=int)
    max_depth = request.args.get('max_depth', 2, type=int)
    max_nodes = request.args.get('max_nodes', 50, type=int)
    relationship_types = request.args.getlist('relationship_type')
    min_confidence = request.args.get('min_confidence', 0.0, type=float)
    domain = request.args.get('domain')
    
    if not node_id:
        return jsonify({"error": "Missing required parameter: node_id"}), 400
    
    # Initialize querier with domain
    querier = KnowledgeQuerier(domain=domain)
    
    # Get subgraph
    results = querier.get_subgraph(
        node_id=node_id,
        max_depth=max_depth,
        max_nodes=max_nodes,
        relationship_types=relationship_types or None,
        min_confidence=min_confidence
    )
    
    return jsonify(results)


@bp.route('/query/claims', methods=['GET'])
def query_claims():
    """Query claims related to an entity."""
    # Extract query parameters
    entity_id = request.args.get('entity_id', type=int)
    claim_type = request.args.get('claim_type')
    min_confidence = request.args.get('min_confidence', 0.0, type=float)
    limit = request.args.get('limit', 20, type=int)
    domain = request.args.get('domain')
    
    if not entity_id:
        return jsonify({"error": "Missing required parameter: entity_id"}), 400
    
    # Initialize querier with domain
    querier = KnowledgeQuerier(domain=domain)
    
    # Query claims
    results = querier.query_claims_by_entity(
        entity_id=entity_id,
        claim_type=claim_type,
        min_confidence=min_confidence,
        limit=limit
    )
    
    return jsonify(results)


@bp.route('/query/events', methods=['GET'])
def query_events():
    """Query events within a specific timeframe."""
    # Extract query parameters
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    tags = request.args.getlist('tag')
    limit = request.args.get('limit', 20, type=int)
    domain = request.args.get('domain')
    
    # Parse dates
    start_date_parsed = None
    if start_date:
        try:
            start_date_parsed = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        except ValueError:
            return jsonify({"error": "Invalid start_date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)"}), 400
    
    end_date_parsed = None
    if end_date:
        try:
            end_date_parsed = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        except ValueError:
            return jsonify({"error": "Invalid end_date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)"}), 400
    
    # Initialize querier with domain
    querier = KnowledgeQuerier(domain=domain)
    
    # Query events
    results = querier.query_events_by_timeframe(
        start_date=start_date_parsed,
        end_date=end_date_parsed,
        tags=tags or None,
        limit=limit
    )
    
    return jsonify(results)


# Processing Routes
@bp.route('/process', methods=['POST'])
def process_content():
    """Process a content item through the knowledge pipeline."""
    data = request.json
    
    if not data or not data.get('content_id'):
        return jsonify({"error": "Missing required field: content_id"}), 400
    
    domain = data.get('domain')
    content_id = data.get('content_id')
    
    # Initialize pipeline with domain
    pipeline = KnowledgePipeline(domain=domain)
    
    try:
        # Process the content
        results = pipeline.process_content(content_id)
        return jsonify(results)
    
    except Exception as e:
        logger.exception(f"Error processing content {content_id}: {e}")
        return jsonify({
            "status": "error",
            "content_id": content_id,
            "message": str(e)
        }), 500


@bp.route('/process/batch', methods=['POST'])
def batch_process():
    """Process multiple content items in batch."""
    data = request.json
    
    if not data or not data.get('content_ids'):
        return jsonify({"error": "Missing required field: content_ids"}), 400
    
    domain = data.get('domain')
    content_ids = data.get('content_ids')
    
    # Initialize pipeline with domain
    pipeline = KnowledgePipeline(domain=domain)
    
    try:
        # Process content in batch
        results = pipeline.batch_process(content_ids)
        return jsonify(results)
    
    except Exception as e:
        logger.exception(f"Error in batch processing: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@bp.route('/process/unprocessed', methods=['POST'])
def process_unprocessed():
    """Find and process unprocessed content."""
    data = request.json
    
    domain = data.get('domain')
    limit = data.get('limit', 100, type=int)
    
    # Initialize pipeline with domain
    pipeline = KnowledgePipeline(domain=domain)
    
    try:
        # Process unprocessed content
        results = pipeline.process_unprocessed_content(limit=limit)
        return jsonify(results)
    
    except Exception as e:
        logger.exception(f"Error processing unprocessed content: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# Contradiction Routes
@bp.route('/contradictions', methods=['GET'])
def get_contradictions():
    """Get unresolved contradictions for review."""
    # Extract query parameters
    contradiction_type = request.args.get('type')
    limit = request.args.get('limit', 100, type=int)
    domain = request.args.get('domain')
    
    # Initialize conflict detector with domain
    detector = ContradictionDetector(domain=domain)
    
    # Get unresolved contradictions
    contradictions = detector.get_unresolved_contradictions(
        contradiction_type=contradiction_type,
        limit=limit
    )
    
    # Format results
    results = []
    for contradiction in contradictions:
        primary_node = KnowledgeGraph().get_node(contradiction.primary_node_id)
        contradicting_node = KnowledgeGraph().get_node(contradiction.contradicting_node_id)
        
        results.append({
            "id": contradiction.id,
            "contradiction_type": contradiction.contradiction_type,
            "description": contradiction.description,
            "primary_node": {
                "id": primary_node.id,
                "name": primary_node.name,
                "type": primary_node.node_type
            } if primary_node else None,
            "contradicting_node": {
                "id": contradicting_node.id,
                "name": contradicting_node.name,
                "type": contradicting_node.node_type
            } if contradicting_node else None,
            "created_at": contradiction.created_at.isoformat() if contradiction.created_at else None
        })
    
    return jsonify({
        "contradictions": results,
        "count": len(results)
    })


@bp.route('/contradictions/<int:contradiction_id>/resolve', methods=['POST'])
def resolve_contradiction(contradiction_id):
    """Resolve a contradiction with the specified status."""
    data = request.json
    
    if not data or not data.get('resolution_status'):
        return jsonify({"error": "Missing required field: resolution_status"}), 400
    
    domain = data.get('domain')
    resolution_status = data.get('resolution_status')
    resolution_notes = data.get('resolution_notes')
    
    # Initialize conflict detector with domain
    detector = ContradictionDetector(domain=domain)
    
    try:
        # Resolve the contradiction
        contradiction = detector.resolve_contradiction(
            contradiction_id=contradiction_id,
            resolution_status=resolution_status,
            resolution_notes=resolution_notes
        )
        
        if not contradiction:
            return jsonify({"error": "Contradiction not found"}), 404
        
        return jsonify({
            "status": "success",
            "contradiction_id": contradiction.id,
            "resolution_status": contradiction.resolution_status,
            "message": f"Resolved contradiction as {resolution_status}"
        })
    
    except Exception as e:
        logger.exception(f"Error resolving contradiction {contradiction_id}: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# Visualization Routes
@bp.route('/visualize/graph', methods=['GET'])
def visualize_graph():
    """Generate graph visualization data."""
    # Extract query parameters
    node_ids = request.args.getlist('node_id', type=int)
    relationship_types = request.args.getlist('relationship_type')
    max_nodes = request.args.get('max_nodes', 100, type=int)
    format_type = request.args.get('format', 'vis_js')
    domain = request.args.get('domain')
    
    # Initialize visualization generator with domain
    generator = GraphVisualizationGenerator(domain=domain)
    
    # Generate graph data
    graph_data = generator.generate_graph_data(
        node_ids=node_ids or None,
        relationship_types=relationship_types or None,
        max_nodes=max_nodes
    )
    
    # Format data for specific visualization library
    if format_type == 'vis_js':
        return jsonify(generator.export_to_vis_js(graph_data))
    elif format_type == 'd3_force':
        return jsonify(generator.export_to_d3_force(graph_data))
    else:
        return jsonify(graph_data)


@bp.route('/visualize/timeline', methods=['GET'])
def visualize_timeline():
    """Generate timeline visualization data."""
    # Extract query parameters
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    node_type = request.args.get('node_type', 'event')
    tags = request.args.getlist('tag')
    domain = request.args.get('domain')
    
    # Parse dates
    start_date_parsed = None
    if start_date:
        try:
            start_date_parsed = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        except ValueError:
            return jsonify({"error": "Invalid start_date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)"}), 400
    
    end_date_parsed = None
    if end_date:
        try:
            end_date_parsed = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        except ValueError:
            return jsonify({"error": "Invalid end_date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)"}), 400
    
    # Initialize visualization generator with domain
    generator = GraphVisualizationGenerator(domain=domain)
    
    # Generate timeline data
    timeline_data = generator.generate_temporal_view(
        start_date=start_date_parsed,
        end_date=end_date_parsed,
        node_type=node_type,
        tags=tags or None
    )
    
    return jsonify(timeline_data)


@bp.route('/visualize/hierarchy', methods=['GET'])
def visualize_hierarchy():
    """Generate hierarchical visualization data."""
    # Extract query parameters
    root_node_id = request.args.get('root_node_id', type=int)
    max_depth = request.args.get('max_depth', 3, type=int)
    domain = request.args.get('domain')
    
    if not root_node_id:
        return jsonify({"error": "Missing required parameter: root_node_id"}), 400
    
    # Initialize visualization generator with domain
    generator = GraphVisualizationGenerator(domain=domain)
    
    # Generate hierarchy data
    hierarchy_data = generator.generate_hierarchy_data(
        root_node_id=root_node_id,
        max_depth=max_depth
    )
    
    return jsonify(hierarchy_data)


# Storage Management Routes
@bp.route('/storage/stats', methods=['GET'])
def get_storage_stats():
    """Get statistics about the knowledge storage."""
    domain = request.args.get('domain')
    
    # Initialize storage manager with domain
    storage = KnowledgeStorage(domain=domain)
    
    # Get storage statistics
    stats = storage.get_storage_stats()
    
    return jsonify(stats)


@bp.route('/storage/export', methods=['GET'])
def export_knowledge():
    """Export the knowledge graph to a dictionary."""
    # Extract query parameters
    include_nodes = request.args.get('include_nodes', True, type=bool)
    include_edges = request.args.get('include_edges', True, type=bool)
    node_types = request.args.getlist('node_type')
    relationship_types = request.args.getlist('relationship_type')
    min_confidence = request.args.get('min_confidence', 0.0, type=float)
    exclude_sources = request.args.get('exclude_sources', False, type=bool)
    domain = request.args.get('domain')
    
    # Initialize storage manager with domain
    storage = KnowledgeStorage(domain=domain)
    
    # Export knowledge graph
    export_data = storage.export_knowledge_graph(
        include_nodes=include_nodes,
        include_edges=include_edges,
        node_types=node_types or None,
        relationship_types=relationship_types or None,
        min_confidence=min_confidence,
        exclude_sources=exclude_sources
    )
    
    return jsonify(export_data)


@bp.route('/storage/import', methods=['POST'])
def import_knowledge():
    """Import knowledge graph data."""
    data = request.json
    domain = data.get('domain')
    
    if not data:
        return jsonify({"error": "No graph data provided"}), 400
    
    # Initialize storage manager with domain
    storage = KnowledgeStorage(domain=domain)
    
    try:
        # Import knowledge graph
        results = storage.import_knowledge_graph(data)
        
        return jsonify({
            "status": "success",
            "nodes_imported": results["nodes_imported"],
            "edges_imported": results["edges_imported"],
            "tags_imported": results["tags_imported"],
            "failed": results["failed"],
            "errors": results["errors"]
        })
    
    except Exception as e:
        logger.exception(f"Error importing knowledge graph: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@bp.route('/storage/clear', methods=['POST'])
def clear_domain_knowledge():
    """Clear all knowledge for a specific domain."""
    data = request.json
    
    if not data or not data.get('domain') or not data.get('confirm_domain'):
        return jsonify({"error": "Missing required fields: domain, confirm_domain"}), 400
    
    domain = data.get('domain')
    confirm_domain = data.get('confirm_domain')
    
    # Initialize storage manager with domain
    storage = KnowledgeStorage(domain=domain)
    
    try:
        # Clear domain knowledge
        results = storage.clear_domain_knowledge(confirm_domain=confirm_domain)
        
        if results.get("status") == "error":
            return jsonify(results), 400
        
        return jsonify(results)
    
    except Exception as e:
        logger.exception(f"Error clearing domain knowledge: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
