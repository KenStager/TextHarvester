"""
Tests for the Knowledge Graph module.

This module contains tests for the KnowledgeGraph class and its functionality
for managing knowledge nodes and edges.
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

from intelligence.knowledge.graph import KnowledgeGraph
from db.models.knowledge_base import (
    KnowledgeNode, KnowledgeEdge, EntityNode, 
    ConceptNode, EventNode, ClaimNode
)


class TestKnowledgeGraph(unittest.TestCase):
    """Tests for the KnowledgeGraph class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.graph = KnowledgeGraph(domain="test")
        
        # Mock session for database operations
        self.session_patch = patch('app.db.session')
        self.mock_session = self.session_patch.start()
        
        # Mock query objects
        self.query_patch = patch('db.models.knowledge_base.KnowledgeNode.query')
        self.mock_query = self.query_patch.start()
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.session_patch.stop()
        self.query_patch.stop()
    
    def test_init(self):
        """Test initialization of KnowledgeGraph."""
        self.assertEqual(self.graph.domain, "test")
    
    def test_get_node(self):
        """Test getting a node by ID."""
        # Arrange
        mock_node = MagicMock(spec=KnowledgeNode)
        mock_node.id = 1
        mock_node.name = "Test Node"
        self.mock_query.get.return_value = mock_node
        
        # Act
        result = self.graph.get_node(1)
        
        # Assert
        self.mock_query.get.assert_called_once_with(1)
        self.assertEqual(result, mock_node)
    
    def test_find_nodes(self):
        """Test finding nodes by criteria."""
        # Arrange
        mock_node = MagicMock(spec=KnowledgeNode)
        mock_node.id = 1
        mock_node.name = "Test Node"
        
        mock_filter = MagicMock()
        mock_filter.filter.return_value = mock_filter
        mock_filter.limit.return_value = [mock_node]
        self.mock_query.filter.return_value = mock_filter
        
        # Act
        result = self.graph.find_nodes(name="Test", node_type="entity", limit=10)
        
        # Assert
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], mock_node)
    
    def test_create_entity_node(self):
        """Test creating an entity node."""
        # Arrange
        mock_node = MagicMock(spec=EntityNode)
        mock_node.id = 1
        mock_node.name = "Test Entity"
        mock_node.domain = "test"
        
        mock_filter = MagicMock()
        mock_filter.filter.return_value = mock_filter
        mock_filter.first.return_value = None  # No existing node
        self.mock_query.filter.return_value = mock_filter
        
        # Mock EntityNode creation
        with patch('db.models.knowledge_base.EntityNode', return_value=mock_node):
            # Act
            result = self.graph.create_entity_node(
                name="Test Entity",
                entity_id=123,
                attributes={"attribute1": "value1"},
                tags=["tag1", "tag2"],
                confidence=0.9
            )
            
            # Assert
            self.mock_session.add.assert_called_once_with(mock_node)
            self.mock_session.commit.assert_called_once()
            self.assertEqual(result, mock_node)
    
    def test_create_edge(self):
        """Test creating an edge between nodes."""
        # Arrange
        mock_source = MagicMock(spec=KnowledgeNode)
        mock_source.id = 1
        mock_source.name = "Source Node"
        
        mock_target = MagicMock(spec=KnowledgeNode)
        mock_target.id = 2
        mock_target.name = "Target Node"
        
        mock_edge = MagicMock(spec=KnowledgeEdge)
        mock_edge.id = 1
        mock_edge.source_id = 1
        mock_edge.target_id = 2
        mock_edge.relationship_type = "test_relation"
        
        # Mock node retrieval
        self.mock_query.get.side_effect = [mock_source, mock_target]
        
        # Mock edge query
        mock_edge_query = MagicMock()
        mock_edge_query.filter_by.return_value = mock_edge_query
        mock_edge_query.first.return_value = None  # No existing edge
        
        with patch('db.models.knowledge_base.KnowledgeEdge.query', mock_edge_query), \
             patch('db.models.knowledge_base.KnowledgeEdge', return_value=mock_edge):
            # Act
            result = self.graph.create_edge(
                source_id=1,
                target_id=2,
                relationship_type="test_relation",
                weight=1.5,
                confidence=0.8
            )
            
            # Assert
            self.mock_session.add.assert_called_once_with(mock_edge)
            self.mock_session.commit.assert_called_once()
            self.assertEqual(result, mock_edge)
    
    def test_delete_node(self):
        """Test deleting a node."""
        # Arrange
        mock_node = MagicMock(spec=KnowledgeNode)
        mock_node.id = 1
        mock_node.name = "Test Node"
        self.mock_query.get.return_value = mock_node
        
        # Act
        result = self.graph.delete_node(1)
        
        # Assert
        self.mock_session.delete.assert_called_once_with(mock_node)
        self.mock_session.commit.assert_called_once()
        self.assertTrue(result)
    
    def test_update_node(self):
        """Test updating a node."""
        # Arrange
        mock_node = MagicMock(spec=KnowledgeNode)
        mock_node.id = 1
        mock_node.name = "Test Node"
        mock_node.attributes = {"existing": "value"}
        mock_node.content = "Original content"
        mock_node.confidence = 0.7
        mock_node.tags = []
        
        self.mock_query.get.return_value = mock_node
        
        # Act
        result = self.graph.update_node(
            node_id=1,
            attributes={"new": "attribute"},
            content="Updated content",
            confidence=0.8
        )
        
        # Assert
        self.assertEqual(mock_node.attributes, {"existing": "value", "new": "attribute"})
        self.assertEqual(mock_node.content, "Updated content")
        self.assertEqual(mock_node.confidence, 0.8)
        self.mock_session.commit.assert_called_once()
        self.assertEqual(result, mock_node)


if __name__ == '__main__':
    unittest.main()
