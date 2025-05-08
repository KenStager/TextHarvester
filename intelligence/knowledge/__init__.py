"""
Knowledge Management System

This package contains components for creating, managing, and querying
a knowledge graph built from extracted content entities and relationships.

The knowledge management system organizes information extracted from content
into a structured, queryable knowledge base that connects entities, concepts,
events, and claims with their relationships and sources.

Key components:
- Knowledge graph management
- Knowledge extraction from processed content
- Knowledge storage and retrieval
- Contradiction detection and resolution
- Source credibility assessment
- Knowledge querying and exploration
"""

from .graph import KnowledgeGraph
from .extraction import KnowledgeExtractor
from .storage import KnowledgeStorage
from .pipeline import KnowledgePipeline
from .credibility import CredibilityScorer
from .queries import KnowledgeQuerier
from .conflict import ContradictionDetector
