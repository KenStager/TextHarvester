"""
Knowledge Base Models

This module defines the database models for the Knowledge Management System,
including knowledge nodes, relationships, source tracking, and credibility metrics.
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text, Table
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB

from app import db


# Association table for knowledge node tags
knowledge_node_tags = Table(
    'knowledge_node_tags',
    db.Model.metadata,
    Column('node_id', Integer, ForeignKey('knowledge_nodes.id'), primary_key=True),
    Column('tag_id', Integer, ForeignKey('knowledge_tags.id'), primary_key=True)
)


class KnowledgeTag(db.Model):
    """Tags for categorizing knowledge nodes"""
    __tablename__ = 'knowledge_tags'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    nodes = relationship("KnowledgeNode", secondary=knowledge_node_tags, back_populates="tags")

    def __repr__(self):
        return f"<KnowledgeTag {self.name}>"


class KnowledgeNode(db.Model):
    """
    Knowledge nodes represent entities, concepts, events, or other knowledge units
    in the knowledge graph.
    """
    __tablename__ = 'knowledge_nodes'

    id = Column(Integer, primary_key=True)
    node_type = Column(String(50), nullable=False)  # entity, concept, event, claim, etc.
    name = Column(String(255), nullable=False)
    canonical_name = Column(String(255), nullable=True)
    domain = Column(String(50), nullable=False)  # e.g., football, science, general
    content = Column(Text, nullable=True)
    attributes = Column(JSONB, nullable=True)  # Flexible storage for domain-specific attributes
    confidence = Column(Float, default=1.0)
    verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # For hierarchical relationships (e.g., taxonomy)
    parent_id = Column(Integer, ForeignKey('knowledge_nodes.id'), nullable=True)
    children = relationship("KnowledgeNode", 
                           backref=db.backref('parent', remote_side=[id]),
                           cascade="all, delete-orphan")

    # Entity relationships
    entity_id = Column(Integer, ForeignKey('entities.id'), nullable=True)
    entity = relationship("Entity", backref="knowledge_nodes")
    
    # Relationships
    tags = relationship("KnowledgeTag", secondary=knowledge_node_tags, back_populates="nodes")
    outgoing_edges = relationship("KnowledgeEdge", foreign_keys="KnowledgeEdge.source_id", 
                               back_populates="source", cascade="all, delete-orphan")
    incoming_edges = relationship("KnowledgeEdge", foreign_keys="KnowledgeEdge.target_id", 
                               back_populates="target", cascade="all, delete-orphan")
    sources = relationship("KnowledgeSource", back_populates="node", cascade="all, delete-orphan")
    contradictions = relationship("KnowledgeContradiction", 
                                back_populates="primary_node",
                                foreign_keys="KnowledgeContradiction.primary_node_id",
                                cascade="all, delete-orphan")

    __mapper_args__ = {
        'polymorphic_on': node_type,
        'polymorphic_identity': 'node'
    }

    def __repr__(self):
        return f"<KnowledgeNode {self.name} ({self.node_type})>"


class EntityNode(KnowledgeNode):
    """Knowledge node representing an entity"""
    __mapper_args__ = {
        'polymorphic_identity': 'entity',
    }


class ConceptNode(KnowledgeNode):
    """Knowledge node representing a concept"""
    __mapper_args__ = {
        'polymorphic_identity': 'concept',
    }


class EventNode(KnowledgeNode):
    """Knowledge node representing an event"""
    __tablename__ = 'knowledge_event_nodes'

    id = Column(Integer, ForeignKey('knowledge_nodes.id'), primary_key=True)
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    location = Column(String(255), nullable=True)

    __mapper_args__ = {
        'polymorphic_identity': 'event',
    }


class ClaimNode(KnowledgeNode):
    """Knowledge node representing a claim or statement"""
    __tablename__ = 'knowledge_claim_nodes'

    id = Column(Integer, ForeignKey('knowledge_nodes.id'), primary_key=True)
    claim_type = Column(String(50), nullable=True)  # e.g., factual, opinion, prediction
    sentiment = Column(Float, nullable=True)  # -1.0 to 1.0
    is_refuted = Column(Boolean, default=False)
    refutation_explanation = Column(Text, nullable=True)

    __mapper_args__ = {
        'polymorphic_identity': 'claim',
    }


class KnowledgeEdge(db.Model):
    """
    Knowledge edges represent relationships between knowledge nodes
    in the knowledge graph.
    """
    __tablename__ = 'knowledge_edges'

    id = Column(Integer, primary_key=True)
    relationship_type = Column(String(100), nullable=False)  # e.g., has_part, plays_for, located_in
    source_id = Column(Integer, ForeignKey('knowledge_nodes.id'), nullable=False)
    target_id = Column(Integer, ForeignKey('knowledge_nodes.id'), nullable=False)
    weight = Column(Float, default=1.0)
    confidence = Column(Float, default=1.0)
    attributes = Column(JSONB, nullable=True)  # Flexible storage for relationship-specific attributes
    valid_from = Column(DateTime, nullable=True)  # For temporal relationships
    valid_to = Column(DateTime, nullable=True)  # For temporal relationships
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    source = relationship("KnowledgeNode", foreign_keys=[source_id], back_populates="outgoing_edges")
    target = relationship("KnowledgeNode", foreign_keys=[target_id], back_populates="incoming_edges")
    sources = relationship("KnowledgeSource", back_populates="edge", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<KnowledgeEdge {self.relationship_type}: {self.source_id} -> {self.target_id}>"


class KnowledgeSource(db.Model):
    """
    Tracks the sources of knowledge nodes and edges,
    including the content they were extracted from and the confidence.
    """
    __tablename__ = 'knowledge_sources'

    id = Column(Integer, primary_key=True)
    node_id = Column(Integer, ForeignKey('knowledge_nodes.id'), nullable=True)
    edge_id = Column(Integer, ForeignKey('knowledge_edges.id'), nullable=True)
    content_id = Column(Integer, ForeignKey('scraper_content.id'), nullable=False)
    extraction_method = Column(String(100), nullable=True)  # e.g., rule-based, NER, inference
    confidence = Column(Float, default=1.0)
    excerpt = Column(Text, nullable=True)  # The text excerpt where this knowledge was found
    context = Column(Text, nullable=True)  # Additional context around the excerpt
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    node = relationship("KnowledgeNode", back_populates="sources")
    edge = relationship("KnowledgeEdge", back_populates="sources")
    content = relationship("ScraperContent")

    def __repr__(self):
        entity_type = "node" if self.node_id else "edge"
        entity_id = self.node_id if self.node_id else self.edge_id
        return f"<KnowledgeSource {entity_type}:{entity_id} from content:{self.content_id}>"


class SourceCredibility(db.Model):
    """
    Tracks credibility scores for content sources
    to weight information based on source reliability.
    """
    __tablename__ = 'source_credibility'

    id = Column(Integer, primary_key=True)
    source_url = Column(String(255), nullable=False, unique=True)
    domain = Column(String(100), nullable=False)
    overall_score = Column(Float, default=0.5)  # 0.0 to 1.0
    domain_expertise = Column(JSONB, nullable=True)  # Domain-specific expertise scores
    accuracy_score = Column(Float, nullable=True)  # Based on fact-checking
    bias_score = Column(Float, nullable=True)  # Political or topical bias assessment
    transparency_score = Column(Float, nullable=True)  # How transparent about sources/methods
    consistency_score = Column(Float, nullable=True)  # How consistent over time
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_evaluated = Column(DateTime, default=datetime.utcnow)
    evaluation_count = Column(Integer, default=1)

    def __repr__(self):
        return f"<SourceCredibility {self.source_url}: {self.overall_score}>"


class KnowledgeContradiction(db.Model):
    """
    Tracks contradictory information within the knowledge base,
    allowing for conflict resolution and uncertainty representation.
    """
    __tablename__ = 'knowledge_contradictions'

    id = Column(Integer, primary_key=True)
    primary_node_id = Column(Integer, ForeignKey('knowledge_nodes.id'), nullable=False)
    contradicting_node_id = Column(Integer, ForeignKey('knowledge_nodes.id'), nullable=False)
    contradiction_type = Column(String(100), nullable=False)  # e.g., factual, temporal, logical
    description = Column(Text, nullable=True)
    resolution_status = Column(String(50), default='unresolved')  # unresolved, resolved_primary, resolved_contradicting, resolved_both_valid
    resolution_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    primary_node = relationship("KnowledgeNode", foreign_keys=[primary_node_id], back_populates="contradictions")
    contradicting_node = relationship("KnowledgeNode", foreign_keys=[contradicting_node_id])

    def __repr__(self):
        return f"<KnowledgeContradiction {self.id}: {self.primary_node_id} vs {self.contradicting_node_id}>"


class KnowledgeQuery(db.Model):
    """
    Tracks knowledge queries and their results
    for analytics and query optimization.
    """
    __tablename__ = 'knowledge_queries'

    id = Column(Integer, primary_key=True)
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50), nullable=False)  # e.g., path, neighborhood, attribute
    parameters = Column(JSONB, nullable=True)
    result_count = Column(Integer, nullable=True)
    execution_time_ms = Column(Integer, nullable=True)
    user_id = Column(Integer, nullable=True)  # If user authentication is implemented
    user_feedback = Column(Integer, nullable=True)  # User rating of results, if collected
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<KnowledgeQuery {self.id}: {self.query_text[:30]}...>"
