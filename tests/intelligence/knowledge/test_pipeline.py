"""
Tests for the Knowledge Pipeline module.

This module contains tests for the KnowledgePipeline class and its functionality
for processing content through the complete knowledge workflow.
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

from intelligence.knowledge.pipeline import KnowledgePipeline
from intelligence.base_pipeline import BasePipeline
from db.models.knowledge_base import (
    KnowledgeNode, KnowledgeEdge, EntityNode, 
    ConceptNode, EventNode, ClaimNode
)


class TestKnowledgePipeline(unittest.TestCase):
    """Tests for the KnowledgePipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock pipeline components
        self.mock_extractor = MagicMock()
        self.mock_storage = MagicMock()
        self.mock_conflict_detector = MagicMock()
        self.mock_credibility_scorer = MagicMock()
        self.mock_graph = MagicMock()
        
        # Create pipeline with mocked components
        with patch('intelligence.knowledge.extraction.KnowledgeExtractor', return_value=self.mock_extractor), \
             patch('intelligence.knowledge.storage.KnowledgeStorage', return_value=self.mock_storage), \
             patch('intelligence.knowledge.conflict.ContradictionDetector', return_value=self.mock_conflict_detector), \
             patch('intelligence.knowledge.credibility.CredibilityScorer', return_value=self.mock_credibility_scorer), \
             patch('intelligence.knowledge.graph.KnowledgeGraph', return_value=self.mock_graph):
            self.pipeline = KnowledgePipeline(domain="test")
    
    def test_init(self):
        """Test initialization of KnowledgePipeline."""
        self.assertEqual(self.pipeline.domain, "test")
        self.assertEqual(self.pipeline.extractor, self.mock_extractor)
        self.assertEqual(self.pipeline.storage, self.mock_storage)
        self.assertEqual(self.pipeline.conflict_detector, self.mock_conflict_detector)
        self.assertEqual(self.pipeline.credibility_scorer, self.mock_credibility_scorer)
        self.assertEqual(self.pipeline.graph, self.mock_graph)
    
    @patch('db.models.scraper.ScraperContent')
    @patch('db.models.content_intelligence.EnhancedContent')
    def test_process_content(self, mock_enhanced_content_class, mock_scraper_content_class):
        """Test processing content through the pipeline."""
        # Arrange
        content_id = 123
        
        # Mock content
        mock_content = MagicMock()
        mock_content.id = content_id
        mock_content.title = "Test Content"
        mock_content.url = "https://example.com/test"
        
        # Mock ScraperContent query
        mock_content_query = MagicMock()
        mock_content_query.get.return_value = mock_content
        mock_scraper_content_class.query = mock_content_query
        
        # Mock EnhancedContent
        mock_enhanced = MagicMock()
        mock_enhanced.content_id = content_id
        
        # Mock EnhancedContent query
        mock_enhanced_query = MagicMock()
        mock_enhanced_query.filter_by.return_value.first.return_value = mock_enhanced
        mock_enhanced_content_class.query = mock_enhanced_query
        
        # Mock extraction results
        extraction_results = {
            "extracted_entities": [
                {"node_id": 456, "name": "Entity1", "type": "PERSON"}
            ],
            "extracted_relationships": [
                {"edge_id": 789, "source_node_id": 456, "target_node_id": 457, "relationship_type": "knows"}
            ],
            "extracted_claims": [
                {"node_id": 101, "text": "Test claim", "type": "factual"}
            ],
            "extracted_events": []
        }
        self.mock_extractor.extract_from_content.return_value = extraction_results
        
        # Mock credibility results
        credibility_results = {
            "source_url": "https://example.com/test",
            "overall_score": 0.8,
            "domain_expertise": {"test": 0.75}
        }
        self.mock_credibility_scorer.evaluate_source.return_value = credibility_results
        
        # Mock contradiction detection
        self.mock_conflict_detector.detect_claim_contradictions.return_value = []
        self.mock_conflict_detector.detect_entity_contradictions.return_value = []
        
        # Mock confidence recalculation
        update_result = {
            "status": "success",
            "node_id": 456,
            "old_confidence": 0.7,
            "new_confidence": 0.75
        }
        self.mock_credibility_scorer.recalculate_confidence_with_credibility.return_value = update_result
        self.mock_credibility_scorer.recalculate_edge_confidence.return_value = update_result
        
        # Mock storage stats
        storage_stats = {
            "nodes": {"total": 100},
            "edges": {"total": 150},
            "contradictions": {"unresolved": 5}
        }
        self.mock_storage.get_storage_stats.return_value = storage_stats
        
        # Act
        result = self.pipeline.process_content(content_id)
        
        # Assert
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["content_id"], content_id)
        self.assertEqual(result["title"], "Test Content")
        self.assertEqual(result["url"], "https://example.com/test")
        
        # Verify component calls
        self.mock_extractor.extract_from_content.assert_called_once_with(content_id)
        self.mock_credibility_scorer.evaluate_source.assert_called_once_with("https://example.com/test")
        self.mock_conflict_detector.detect_claim_contradictions.assert_called_once()
        self.mock_conflict_detector.detect_entity_contradictions.assert_called_once_with(entity_id=456)
        self.mock_credibility_scorer.recalculate_confidence_with_credibility.assert_called_once_with(node_id=456)
        self.mock_credibility_scorer.recalculate_edge_confidence.assert_called_once_with(edge_id=789)
        self.mock_storage.get_storage_stats.assert_called_once()
    
    @patch('db.models.scraper.ScraperContent')
    @patch('db.models.content_intelligence.EnhancedContent')
    def test_process_content_not_found(self, mock_enhanced_content_class, mock_scraper_content_class):
        """Test processing content that doesn't exist."""
        # Arrange
        content_id = 123
        
        # Mock ScraperContent query to return None
        mock_content_query = MagicMock()
        mock_content_query.get.return_value = None
        mock_scraper_content_class.query = mock_content_query
        
        # Act
        result = self.pipeline.process_content(content_id)
        
        # Assert
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "Content not found")
        
        # Verify no component calls were made
        self.mock_extractor.extract_from_content.assert_not_called()
        self.mock_credibility_scorer.evaluate_source.assert_not_called()
    
    @patch('db.models.scraper.ScraperContent')
    @patch('db.models.content_intelligence.EnhancedContent')
    def test_process_content_not_enhanced(self, mock_enhanced_content_class, mock_scraper_content_class):
        """Test processing content that hasn't been enhanced."""
        # Arrange
        content_id = 123
        
        # Mock content
        mock_content = MagicMock()
        mock_content.id = content_id
        mock_content.title = "Test Content"
        
        # Mock ScraperContent query
        mock_content_query = MagicMock()
        mock_content_query.get.return_value = mock_content
        mock_scraper_content_class.query = mock_content_query
        
        # Mock EnhancedContent query to return None
        mock_enhanced_query = MagicMock()
        mock_enhanced_query.filter_by.return_value.first.return_value = None
        mock_enhanced_content_class.query = mock_enhanced_query
        
        # Act
        result = self.pipeline.process_content(content_id)
        
        # Assert
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "Content has not been enhanced for intelligence extraction")
        
        # Verify no extraction was performed
        self.mock_extractor.extract_from_content.assert_not_called()
    
    def test_batch_process(self):
        """Test batch processing of content."""
        # Arrange
        content_ids = [123, 456, 789]
        
        # Mock process_content to return success for first two IDs and error for the third
        def mock_process(content_id):
            if content_id == 789:
                return {
                    "status": "error",
                    "message": "Content not found",
                    "content_id": content_id
                }
            else:
                return {
                    "status": "success",
                    "content_id": content_id,
                    "steps": {}
                }
        
        # Patch the process_content method
        with patch.object(self.pipeline, 'process_content', side_effect=mock_process):
            # Act
            result = self.pipeline.batch_process(content_ids)
            
            # Assert
            self.assertEqual(result["total"], 3)
            self.assertEqual(result["successful"], 2)
            self.assertEqual(result["failed"], 1)
            self.assertEqual(len(result["errors"]), 1)
            self.assertEqual(result["errors"][0]["content_id"], 789)


if __name__ == '__main__':
    unittest.main()
