"""
Unit tests for the classification system.

This module contains unit tests for the Topic Classification System components.
"""

import unittest
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from intelligence.classification.topic_taxonomy import TopicTaxonomy, TopicNode
from intelligence.classification.fast_filter import FastFilter
from intelligence.classification.classifiers import TopicClassifier, ClassificationResult
from intelligence.classification.pipeline import ClassificationPipeline, ClassificationInput
from intelligence.classification.taxonomies.football import get_premier_league_taxonomy


class TestTopicTaxonomy(unittest.TestCase):
    """Tests for the TopicTaxonomy class."""
    
    def setUp(self):
        """Set up test taxonomy."""
        self.taxonomy = TopicTaxonomy("test_taxonomy", "Test taxonomy")
        
        # Create a simple taxonomy
        root = TopicNode("Root", ["root", "main"], "Root node")
        self.taxonomy.add_root_node(root)
        
        child1 = TopicNode("Child1", ["child", "first"], "First child")
        child2 = TopicNode("Child2", ["child", "second"], "Second child")
        
        root.add_child(child1)
        root.add_child(child2)
        
        grandchild = TopicNode("Grandchild", ["grand", "third"], "Grandchild")
        child1.add_child(grandchild)
    
    def test_get_node_by_name(self):
        """Test retrieving node by name."""
        node = self.taxonomy.get_node_by_name("Child1")
        self.assertIsNotNone(node)
        self.assertEqual(node.name, "Child1")
        
        # Non-existent node
        node = self.taxonomy.get_node_by_name("NonExistent")
        self.assertIsNone(node)
    
    def test_get_all_nodes(self):
        """Test retrieving all nodes."""
        nodes = self.taxonomy.get_all_nodes()
        self.assertEqual(len(nodes), 4)  # Root + 2 children + 1 grandchild
    
    def test_get_leaf_nodes(self):
        """Test retrieving leaf nodes."""
        nodes = self.taxonomy.get_leaf_nodes()
        self.assertEqual(len(nodes), 2)  # Child2 and Grandchild are leaves
        names = [node.name for node in nodes]
        self.assertIn("Child2", names)
        self.assertIn("Grandchild", names)
    
    def test_find_path_to_node(self):
        """Test finding path to a node."""
        grandchild = self.taxonomy.get_node_by_name("Grandchild")
        path = self.taxonomy.find_path_to_node(grandchild.id)
        
        self.assertEqual(len(path), 3)
        self.assertEqual(path[0].name, "Root")
        self.assertEqual(path[1].name, "Child1")
        self.assertEqual(path[2].name, "Grandchild")


class TestFastFilter(unittest.TestCase):
    """Tests for the FastFilter class."""
    
    def setUp(self):
        """Set up test filter."""
        # Create a simple taxonomy for testing
        taxonomy = TopicTaxonomy("test_taxonomy", "Test taxonomy")
        root = TopicNode("Football", ["football", "soccer", "goal"], "Football content")
        taxonomy.add_root_node(root)
        
        self.filter = FastFilter(taxonomy, threshold=0.3, strategy="keyword")
    
    def test_is_potentially_relevant(self):
        """Test relevance filtering."""
        # Relevant text
        relevant_text = "Manchester United won the football match with a last-minute goal."
        is_relevant, confidence = self.filter.is_potentially_relevant(relevant_text)
        self.assertTrue(is_relevant)
        self.assertGreaterEqual(confidence, 0.3)
        
        # Irrelevant text
        irrelevant_text = "The stock market saw significant gains yesterday with tech stocks leading the rally."
        is_relevant, confidence = self.filter.is_potentially_relevant(irrelevant_text)
        self.assertFalse(is_relevant)
        self.assertLess(confidence, 0.3)
    
    def test_get_matching_keywords(self):
        """Test keyword matching."""
        text = "The football match ended with a spectacular goal in the final minutes."
        keywords = self.filter.get_matching_keywords(text)
        
        self.assertIn("football", keywords)
        self.assertIn("goal", keywords)


class TestFootballTaxonomy(unittest.TestCase):
    """Tests for the football taxonomy."""
    
    def setUp(self):
        """Set up test environment."""
        self.football_taxonomy = get_premier_league_taxonomy()
    
    def test_taxonomy_structure(self):
        """Test the structure of the football taxonomy."""
        # Check root node
        self.assertEqual(len(self.football_taxonomy.root_nodes), 1)
        root = self.football_taxonomy.root_nodes[0]
        self.assertEqual(root.name, "Football")
        
        # Check Premier League node
        pl_node = None
        for child in root.children:
            if child.name == "Premier League":
                pl_node = child
                break
        
        self.assertIsNotNone(pl_node, "Premier League node not found")
        
        # Check categories under Premier League
        category_names = [node.name for node in pl_node.children]
        expected_categories = ["Teams", "Players", "Managers", "Transfers", "Matches"]
        
        for category in expected_categories:
            self.assertIn(category, category_names, f"Category '{category}' not found")
    
    def test_teams_coverage(self):
        """Test that all Premier League teams are included."""
        # Find the Teams node
        teams_node = None
        pl_node = self.football_taxonomy.get_node_by_name("Premier League")
        
        for child in pl_node.children:
            if child.name == "Teams":
                teams_node = child
                break
        
        self.assertIsNotNone(teams_node, "Teams node not found")
        
        # Check team count (should be 20 for Premier League)
        self.assertEqual(len(teams_node.children), 20, "Should have exactly 20 Premier League teams")
        
        # Check some specific teams
        team_names = [node.name for node in teams_node.children]
        expected_teams = ["Manchester United", "Liverpool", "Arsenal", "Chelsea", "Manchester City"]
        
        for team in expected_teams:
            self.assertIn(team, team_names, f"Team '{team}' not found")


class TestClassificationPipeline(unittest.TestCase):
    """Tests for the classification pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a simplified pipeline for testing
        # Using only FastFilter component for speed
        self.pipeline = ClassificationPipeline(domain_name="football")
        
        # Override hierarchical classifier with a mock
        self.pipeline.hierarchical_classifier = None
    
    def test_fast_filter_only(self):
        """Test classification with only the fast filter."""
        # Football content
        football_text = "Liverpool beat Manchester United 2-0 at Anfield on Saturday."
        result = self.pipeline.process(football_text)
        
        # Since we've disabled the hierarchical classifier, the result should
        # be based only on the fast filter
        self.assertFalse(result.is_relevant)  # Without hierarchical classifier, should be False
        
    def test_input_conversions(self):
        """Test different input formats."""
        # String input
        result1 = self.pipeline.process("Some text about football.")
        self.assertIsNotNone(result1)
        
        # Dict input
        result2 = self.pipeline.process({
            "text": "Some text about football.",
            "content_id": 123
        })
        self.assertIsNotNone(result2)
        self.assertEqual(result2.content_id, 123)
        
        # ClassificationInput
        input_obj = ClassificationInput(
            text="Some text about football.",
            content_id=456,
            metadata={"source": "test"}
        )
        result3 = self.pipeline.process(input_obj)
        self.assertIsNotNone(result3)
        self.assertEqual(result3.content_id, 456)


if __name__ == "__main__":
    unittest.main()
