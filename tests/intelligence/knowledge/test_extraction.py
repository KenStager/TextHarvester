"""
Tests for the Knowledge Extraction module.

This module contains tests for the KnowledgeExtractor class and its functionality
for extracting knowledge from processed content.
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

from intelligence.knowledge.extraction import KnowledgeExtractor
from db.models.knowledge_base import (
    KnowledgeNode, KnowledgeEdge, EntityNode, 
    ConceptNode, EventNode, ClaimNode
)
from db.models.entity_models import Entity, EntityMention
from db.models.content_intelligence import ContentClassification, EnhancedContent


class TestKnowledgeExtractor(unittest.TestCase):
    """Tests for the KnowledgeExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock graph manager
        self.mock_graph = MagicMock()
        self.extractor = KnowledgeExtractor(domain="test", graph_manager=self.mock_graph)
        
        # Mock session for database operations
        self.session_patch = patch('app.db.session')
        self.mock_session = self.session_patch.start()
        
        # Mock query objects
        self.enhanced_content_query_patch = patch('db.models.content_intelligence.EnhancedContent.query')
        self.mock_enhanced_content_query = self.enhanced_content_query_patch.start()
        
        self.entity_mention_query_patch = patch('db.models.entity_models.EntityMention.query')
        self.mock_entity_mention_query = self.entity_mention_query_patch.start()
        
        self.content_classification_query_patch = patch('db.models.content_intelligence.ContentClassification.query')
        self.mock_content_classification_query = self.content_classification_query_patch.start()
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.session_patch.stop()
        self.enhanced_content_query_patch.stop()
        self.entity_mention_query_patch.stop()
        self.content_classification_query_patch.stop()
    
    def test_init(self):
        """Test initialization of KnowledgeExtractor."""
        self.assertEqual(self.extractor.domain, "test")
        self.assertEqual(self.extractor.graph_manager, self.mock_graph)
    
    def test_extract_from_content(self):
        """Test extracting knowledge from content."""
        # Arrange
        content_id = 123
        
        # Mock enhanced content
        mock_enhanced_content = MagicMock(spec=EnhancedContent)
        mock_enhanced_content.content_id = content_id
        mock_enhanced_content.enhanced_metadata = {
            "claims": [
                {"text": "Test claim", "type": "factual", "confidence": 0.8}
            ],
            "events": [
                {"name": "Test event", "start_date": "2023-01-01T00:00:00", "confidence": 0.9}
            ]
        }
        
        mock_filter = MagicMock()
        mock_filter.first.return_value = mock_enhanced_content
        self.mock_enhanced_content_query.filter_by.return_value = mock_filter
        
        # Mock entity mentions
        mock_entity_mention = MagicMock(spec=EntityMention)
        mock_entity_mention.entity_id = 456
        mock_entity_mention.mention_text = "Test Entity"
        mock_entity_mention.start_char = 10
        mock_entity_mention.end_char = 20
        mock_entity_mention.context_before = "Before "
        mock_entity_mention.context_after = " after"
        mock_entity_mention.confidence = 0.9
        
        mock_entity_filter = MagicMock()
        mock_entity_filter.all.return_value = [mock_entity_mention]
        self.mock_entity_mention_query.filter_by.return_value = mock_entity_filter
        
        # Mock content classification
        mock_classification = MagicMock(spec=ContentClassification)
        mock_classification.content_id = content_id
        mock_classification.is_primary = True
        
        mock_class_filter = MagicMock()
        mock_class_filter.first.return_value = mock_classification
        self.mock_content_classification_query.filter_by.return_value = mock_class_filter
        
        # Mock entity
        mock_entity = MagicMock(spec=Entity)
        mock_entity.id = 456
        mock_entity.name = "Test Entity"
        mock_entity.canonical_name = "test entity"
        
        # Mock entity type
        mock_entity_type = MagicMock()
        mock_entity_type.name = "PERSON"
        mock_entity_type.parent = None
        
        mock_entity.entity_type = mock_entity_type
        
        # Mock Entity query
        with patch('db.models.entity_models.Entity.query') as mock_entity_query:
            mock_entity_query.get.return_value = mock_entity
            
            # Mock graph manager responses
            mock_entity_node = MagicMock(spec=EntityNode)
            mock_entity_node.id = 789
            self.mock_graph.create_entity_node.return_value = mock_entity_node
            
            mock_claim_node = MagicMock(spec=ClaimNode)
            mock_claim_node.id = 101
            self.mock_graph.create_claim_node.return_value = mock_claim_node
            
            mock_event_node = MagicMock(spec=EventNode)
            mock_event_node.id = 102
            self.mock_graph.create_event_node.return_value = mock_event_node
            
            # Act
            result = self.extractor.extract_from_content(content_id)
            
            # Assert
            self.assertEqual(result["content_id"], content_id)
            self.assertEqual(len(result["extracted_entities"]), 1)
            self.assertEqual(result["extracted_entities"][0]["node_id"], 789)
            self.assertEqual(len(result["extracted_claims"]), 1)
            self.assertEqual(len(result["extracted_events"]), 1)
            
            # Verify graph manager calls
            self.mock_graph.create_entity_node.assert_called_once()
            self.mock_graph.add_source.assert_called()
            self.mock_graph.create_claim_node.assert_called_once()
            self.mock_graph.create_event_node.assert_called_once()
    
    def test_extract_entities(self):
        """Test extracting entity nodes from entity mentions."""
        # Arrange
        content_id = 123
        
        # Mock entity mention
        mock_entity_mention = MagicMock(spec=EntityMention)
        mock_entity_mention.entity_id = 456
        mock_entity_mention.mention_text = "Test Entity"
        mock_entity_mention.confidence = 0.9
        mock_entity_mention.context_before = "Before "
        mock_entity_mention.context_after = " after"
        
        entity_mentions = [mock_entity_mention]
        
        # Mock entity
        mock_entity = MagicMock(spec=Entity)
        mock_entity.id = 456
        mock_entity.name = "Test Entity"
        mock_entity.canonical_name = "test entity"
        mock_entity.metadata = {"key": "value"}
        
        # Mock entity type
        mock_entity_type = MagicMock()
        mock_entity_type.name = "PERSON"
        mock_entity_type.parent = None
        
        mock_entity.entity_type = mock_entity_type
        
        # Mock Entity query
        with patch('db.models.entity_models.Entity.query') as mock_entity_query:
            mock_entity_query.get.return_value = mock_entity
            
            # Mock graph manager response
            mock_entity_node = MagicMock(spec=EntityNode)
            mock_entity_node.id = 789
            self.mock_graph.create_entity_node.return_value = mock_entity_node
            
            # Act
            result = self.extractor._extract_entities(entity_mentions, content_id)
            
            # Assert
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["node_id"], 789)
            self.assertEqual(result[0]["entity_id"], 456)
            self.assertEqual(result[0]["name"], "test entity")
            self.assertEqual(result[0]["type"], "PERSON")
            
            # Verify graph manager calls
            self.mock_graph.create_entity_node.assert_called_once_with(
                name="test entity",
                entity_id=456,
                attributes={"key": "value", "alternative_names": ["Test Entity"]},
                tags=["PERSON"],
                confidence=0.9
            )
            
            self.mock_graph.add_source.assert_called_once()
    
    def test_extract_football_relationships(self):
        """Test extracting football-specific relationships."""
        # Arrange
        content_id = 123
        
        # Mock entity mentions
        player_mention = MagicMock(spec=EntityMention)
        player_mention.entity_id = 456
        player_mention.mention_text = "Ronaldo"
        player_mention.start_char = 10
        player_mention.end_char = 17
        player_mention.context_before = "Football player "
        player_mention.context_after = " plays for the team"
        
        team_mention = MagicMock(spec=EntityMention)
        team_mention.entity_id = 789
        team_mention.mention_text = "Manchester United"
        team_mention.start_char = 30
        team_mention.end_char = 47
        team_mention.context_before = "Ronaldo plays for "
        team_mention.context_after = " in the Premier League"
        
        entity_mentions = [player_mention, team_mention]
        
        # Entity to node mapping
        entity_to_node = {
            456: 901,  # Player entity to node mapping
            789: 902   # Team entity to node mapping
        }
        
        # Mock Entity query
        with patch('db.models.entity_models.Entity.query') as mock_entity_query:
            # Mock player entity
            mock_player = MagicMock(spec=Entity)
            mock_player.id = 456
            
            # Mock player entity type
            mock_player_type = MagicMock()
            mock_player_type.name = "PERSON.PLAYER"
            mock_player_type.parent = None
            mock_player.entity_type = mock_player_type
            
            # Mock team entity
            mock_team = MagicMock(spec=Entity)
            mock_team.id = 789
            
            # Mock team entity type
            mock_team_type = MagicMock()
            mock_team_type.name = "TEAM"
            mock_team_type.parent = None
            mock_team.entity_type = mock_team_type
            
            # Set up mock to return appropriate entity based on ID
            def get_entity(entity_id):
                if entity_id == 456:
                    return mock_player
                elif entity_id == 789:
                    return mock_team
                return None
            
            mock_entity_query.get.side_effect = get_entity
            
            # Mock graph manager response for edge creation
            mock_edge = MagicMock(spec=KnowledgeEdge)
            mock_edge.id = 123
            self.mock_graph.create_edge.return_value = mock_edge
            
            # Act
            # Set domain to "football" to enable football-specific extraction
            self.extractor.domain = "football"
            result = self.extractor._extract_football_relationships(entity_mentions, entity_to_node, content_id)
            
            # Assert
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["edge_id"], 123)
            self.assertEqual(result[0]["relationship_type"], "plays_for")
            
            # Verify graph manager calls
            self.mock_graph.create_edge.assert_called_once_with(
                source_id=901,  # Player node ID
                target_id=902,  # Team node ID
                relationship_type="plays_for",
                confidence=0.85
            )
            
            self.mock_graph.add_source.assert_called_once()


if __name__ == '__main__':
    unittest.main()
