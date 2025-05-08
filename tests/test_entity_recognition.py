"""
Unit tests for the Entity Recognition System.

This module contains unit tests for the Entity Recognition System components.
"""

import unittest
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from intelligence.entities.entity_types import EntityType, EntityAttribute, EntityTypeRegistry
from intelligence.entities.ner_model import Entity, CustomNERModel
from intelligence.entities.linking import KnowledgeBase, EntityLinker, EntityCandidate
from intelligence.entities.pipeline import EntityExtractionPipeline, EntityExtractionInput
from intelligence.entities.taxonomies.football_entities import create_football_entity_registry


class TestEntityTypes(unittest.TestCase):
    """Tests for the entity type system."""
    
    def setUp(self):
        """Set up test entity types."""
        # Create a simple entity type hierarchy
        self.person = EntityType("PERSON", "A human individual")
        
        # Add attributes
        self.person.add_attribute(EntityAttribute("name", "Full name of the person"))
        self.person.add_attribute(EntityAttribute("nationality", "Nationality of the person"))
        
        # Add child types
        self.player = EntityType("PLAYER", "A sports player", parent=self.person)
        self.player.add_attribute(EntityAttribute("team", "Current team"))
        self.player.add_attribute(EntityAttribute("position", "Playing position"))
        
        self.manager = EntityType("MANAGER", "A team manager", parent=self.person)
        self.manager.add_attribute(EntityAttribute("team", "Current team"))
        
        # Create a registry
        self.registry = EntityTypeRegistry(domain="test")
        self.registry.add_root_type(self.person)
    
    def test_entity_type_hierarchy(self):
        """Test entity type hierarchy relationships."""
        # Check parent-child relationships
        self.assertEqual(self.player.parent, self.person)
        self.assertEqual(self.manager.parent, self.person)
        self.assertIn(self.player, self.person.children)
        self.assertIn(self.manager, self.person.children)
    
    def test_get_full_name(self):
        """Test getting full hierarchical names."""
        self.assertEqual(self.person.get_full_name(), "PERSON")
        self.assertEqual(self.player.get_full_name(), "PERSON.PLAYER")
        self.assertEqual(self.manager.get_full_name(), "PERSON.MANAGER")
    
    def test_is_subtype_of(self):
        """Test subtype checking."""
        self.assertTrue(self.player.is_subtype_of("PERSON"))
        self.assertTrue(self.manager.is_subtype_of("PERSON"))
        self.assertFalse(self.person.is_subtype_of("PLAYER"))
        self.assertTrue(self.player.is_subtype_of(self.person))
    
    def test_get_all_attributes(self):
        """Test attribute inheritance."""
        # Person attributes
        person_attrs = self.person.get_all_attributes()
        self.assertEqual(len(person_attrs), 2)
        attr_names = {a.name for a in person_attrs}
        self.assertEqual(attr_names, {"name", "nationality"})
        
        # Player attributes (including inherited)
        player_attrs = self.player.get_all_attributes()
        self.assertEqual(len(player_attrs), 4)
        attr_names = {a.name for a in player_attrs}
        self.assertEqual(attr_names, {"name", "nationality", "team", "position"})


class TestEntityTypeRegistry(unittest.TestCase):
    """Tests for the entity type registry."""
    
    def setUp(self):
        """Set up test entity type registry."""
        self.registry = create_football_entity_registry()
    
    def test_get_type_by_name(self):
        """Test retrieving entity types by name."""
        team_type = self.registry.get_type_by_name("TEAM")
        self.assertIsNotNone(team_type)
        self.assertEqual(team_type.name, "TEAM")
        
        # Non-existent type
        nonexistent_type = self.registry.get_type_by_name("NONEXISTENT")
        self.assertIsNone(nonexistent_type)
    
    def test_get_subtypes(self):
        """Test getting subtypes of an entity type."""
        person_subtypes = self.registry.get_subtypes("PERSON")
        
        # Check that we have some subtypes
        self.assertGreater(len(person_subtypes), 0)
        
        # Check that all subtypes are correctly related
        for subtype in person_subtypes:
            self.assertTrue(subtype.is_subtype_of("PERSON"))
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        # Convert to dict
        registry_dict = self.registry.to_dict()
        
        # Check basic properties
        self.assertEqual(registry_dict["domain"], "football")
        self.assertIsInstance(registry_dict["types"], list)
        self.assertGreater(len(registry_dict["types"]), 0)
        
        # Convert back to registry
        new_registry = EntityTypeRegistry.from_dict(registry_dict)
        
        # Check that we have the same number of root types
        self.assertEqual(len(new_registry.root_types), len(self.registry.root_types))
        
        # Check that we have the same number of total types
        self.assertEqual(len(new_registry.get_all_types()), len(self.registry.get_all_types()))


class TestEntity(unittest.TestCase):
    """Tests for the Entity class."""
    
    def test_entity_creation(self):
        """Test creating entity objects."""
        entity = Entity(
            text="Liverpool", 
            label="TEAM", 
            start_char=10, 
            end_char=19, 
            confidence=0.95
        )
        
        self.assertEqual(entity.text, "Liverpool")
        self.assertEqual(entity.label, "TEAM")
        self.assertEqual(entity.start_char, 10)
        self.assertEqual(entity.end_char, 19)
        self.assertEqual(entity.confidence, 0.95)
        self.assertIsNone(entity.kb_id)
    
    def test_to_dict_and_from_dict(self):
        """Test entity serialization."""
        original = Entity(
            text="Mohamed Salah", 
            label="PLAYER", 
            start_char=20, 
            end_char=33, 
            confidence=0.9,
            kb_id="player_123"
        )
        
        # Convert to dict
        entity_dict = original.to_dict()
        
        # Check dict properties
        self.assertEqual(entity_dict["text"], "Mohamed Salah")
        self.assertEqual(entity_dict["label"], "PLAYER")
        self.assertEqual(entity_dict["start_char"], 20)
        self.assertEqual(entity_dict["end_char"], 33)
        self.assertEqual(entity_dict["confidence"], 0.9)
        self.assertEqual(entity_dict["kb_id"], "player_123")
        
        # Convert back to entity
        recreated = Entity.from_dict(entity_dict)
        
        # Check equality
        self.assertEqual(recreated.text, original.text)
        self.assertEqual(recreated.label, original.label)
        self.assertEqual(recreated.start_char, original.start_char)
        self.assertEqual(recreated.end_char, original.end_char)
        self.assertEqual(recreated.confidence, original.confidence)
        self.assertEqual(recreated.kb_id, original.kb_id)


class TestKnowledgeBase(unittest.TestCase):
    """Tests for the knowledge base."""
    
    def setUp(self):
        """Set up test knowledge base."""
        self.kb = KnowledgeBase(name="test_kb", domain="football")
        
        # Add test entities
        self.kb.add_entity(
            kb_id="team_liverpool",
            name="Liverpool",
            entity_type="TEAM",
            attributes={"country": "England", "stadium": "Anfield"},
            aliases=["The Reds", "LFC"]
        )
        
        self.kb.add_entity(
            kb_id="player_salah",
            name="Mohamed Salah",
            entity_type="PLAYER",
            attributes={"team": "Liverpool", "position": "Forward", "nationality": "Egypt"},
            aliases=["Mo Salah", "Egyptian King"]
        )
        
        self.kb.add_entity(
            kb_id="player_trent",
            name="Trent Alexander-Arnold",
            entity_type="PLAYER",
            attributes={"team": "Liverpool", "position": "Defender", "nationality": "England"},
            aliases=["TAA"]
        )
    
    def test_get_entity(self):
        """Test retrieving entities by ID."""
        entity = self.kb.get_entity("team_liverpool")
        self.assertIsNotNone(entity)
        self.assertEqual(entity["name"], "Liverpool")
        self.assertEqual(entity["type"], "TEAM")
        
        # Non-existent entity
        nonexistent = self.kb.get_entity("nonexistent")
        self.assertIsNone(nonexistent)
    
    def test_get_entities_by_type(self):
        """Test retrieving entities by type."""
        players = self.kb.get_entities_by_type("PLAYER")
        self.assertEqual(len(players), 2)
        player_names = {p["name"] for p in players}
        self.assertEqual(player_names, {"Mohamed Salah", "Trent Alexander-Arnold"})
    
    def test_find_candidates(self):
        """Test finding candidate entities."""
        # Test exact match
        candidates = self.kb.find_candidates("Liverpool")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].name, "Liverpool")
        self.assertEqual(candidates[0].score, 1.0)  # Exact match
        
        # Test alias match
        candidates = self.kb.find_candidates("The Reds")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].name, "Liverpool")
        self.assertGreaterEqual(candidates[0].score, 0.8)  # Alias match
        
        # Test fuzzy match
        candidates = self.kb.find_candidates("Liverpol")  # Typo
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].name, "Liverpool")
        self.assertLess(candidates[0].score, 1.0)  # Not exact match
        
        # Test with type filter
        candidates = self.kb.find_candidates("Liverpool", entity_type="TEAM")
        self.assertEqual(len(candidates), 1)
        
        # Test with wrong type filter
        candidates = self.kb.find_candidates("Liverpool", entity_type="PLAYER")
        self.assertEqual(len(candidates), 0)  # No matches due to type filter
    
    def test_disambiguate_entity(self):
        """Test entity disambiguation."""
        # Create test entity and context
        entity = Entity(text="Salah", label="PLAYER", start_char=0, end_char=5)
        context = "Salah scored a goal for Liverpool against Manchester United."
        
        # Get candidates
        candidates = [
            EntityCandidate(
                kb_id="player_salah",
                name="Mohamed Salah",
                score=0.8,
                type="PLAYER",
                attributes={"team": "Liverpool"}
            ),
            EntityCandidate(
                kb_id="player_other_salah",
                name="Ahmed Salah",
                score=0.7,
                type="PLAYER",
                attributes={"team": "Other Team"}
            )
        ]
        
        # Disambiguate
        linked_entity = self.kb.disambiguate_entity(entity, context, candidates)
        
        # Check result
        self.assertIsNotNone(linked_entity)
        self.assertEqual(linked_entity.kb_id, "player_salah")  # Should choose the right Salah based on context


class TestEntityLinker(unittest.TestCase):
    """Tests for the entity linker."""
    
    def setUp(self):
        """Set up test entity linker."""
        # Create a simple football knowledge base
        self.kb = KnowledgeBase(name="test_kb", domain="football")
        
        # Add test entities
        self.kb.add_entity(
            kb_id="team_liverpool",
            name="Liverpool",
            entity_type="TEAM",
            attributes={"country": "England", "stadium": "Anfield"},
            aliases=["The Reds", "LFC"]
        )
        
        self.kb.add_entity(
            kb_id="team_manchester_united",
            name="Manchester United",
            entity_type="TEAM",
            attributes={"country": "England", "stadium": "Old Trafford"},
            aliases=["Man Utd", "United", "The Red Devils"]
        )
        
        # Create entity linker with this knowledge base
        self.linker = EntityLinker(knowledge_base=self.kb, domain="football")
    
    def test_link_entity(self):
        """Test linking a single entity."""
        # Create test entity and context
        entity = Entity(text="Liverpool", label="TEAM", start_char=0, end_char=9)
        context = "Liverpool won the match 2-1."
        
        # Link entity
        linked_entity = self.linker.link_entity(entity, context)
        
        # Check result
        self.assertIsNotNone(linked_entity)
        self.assertEqual(linked_entity.kb_id, "team_liverpool")
        self.assertEqual(linked_entity.kb_name, "Liverpool")
        self.assertEqual(linked_entity.type, "TEAM")
    
    def test_link_entities(self):
        """Test linking multiple entities."""
        # Create test entities
        entities = [
            Entity(text="Liverpool", label="TEAM", start_char=0, end_char=9),
            Entity(text="Man Utd", label="TEAM", start_char=20, end_char=27)
        ]
        
        # Create context
        text = "Liverpool played against Man Utd in a thrilling match."
        
        # Link entities
        linked_entities = self.linker.link_entities(entities, text)
        
        # Check result
        self.assertEqual(len(linked_entities), 2)
        
        # Check Liverpool
        self.assertEqual(linked_entities[0].kb_id, "team_liverpool")
        self.assertEqual(linked_entities[0].kb_name, "Liverpool")
        
        # Check Man Utd
        self.assertEqual(linked_entities[1].kb_id, "team_manchester_united")
        self.assertEqual(linked_entities[1].kb_name, "Manchester United")


class TestEntityExtractionPipeline(unittest.TestCase):
    """Tests for the entity extraction pipeline."""
    
    def setUp(self):
        """Set up test pipeline."""
        # Create a simplified pipeline for testing
        self.pipeline = EntityExtractionPipeline(domain="football")
    
    def test_process_simple_text(self):
        """Test processing a simple text."""
        text = "Liverpool beat Manchester United 2-1 at Anfield."
        
        # Process text
        result = self.pipeline.process(text)
        
        # Check that we got some entities
        self.assertGreater(len(result.entities), 0)
        
        # Check processing time
        self.assertGreater(result.processing_time, 0)
        
        # Check that we have entity counts
        self.assertGreater(len(result.entity_counts), 0)
    
    def test_input_conversions(self):
        """Test different input formats."""
        # String input
        result1 = self.pipeline.process("Liverpool")
        self.assertIsNotNone(result1)
        
        # Dict input
        result2 = self.pipeline.process({
            "text": "Liverpool",
            "content_id": 123
        })
        self.assertIsNotNone(result2)
        self.assertEqual(result2.content_id, 123)
        
        # Full input object
        input_obj = EntityExtractionInput(
            text="Liverpool",
            content_id=456,
            metadata={"source": "test"}
        )
        result3 = self.pipeline.process(input_obj)
        self.assertIsNotNone(result3)
        self.assertEqual(result3.content_id, 456)
    
    def test_entity_distribution_analysis(self):
        """Test entity distribution analysis."""
        text = """
        Liverpool secured a dramatic 2-1 victory over Manchester United at Anfield
        on Sunday, with Mohamed Salah scoring the winning goal in the 82nd minute.
        Manager Jurgen Klopp praised his team's resilience after falling behind to
        a Bruno Fernandes strike early in the second half.
        """
        
        # Analyze entity distribution
        analysis = self.pipeline.analyze_entity_distribution(text)
        
        # Check result structure
        self.assertIn("total_entities", analysis)
        self.assertIn("counts_by_type", analysis)
        self.assertIn("entity_density", analysis)
        self.assertIn("position_distribution", analysis)
        self.assertIn("co_occurrence", analysis)
        
        # Check that we have some entities
        self.assertGreater(analysis["total_entities"], 0)
        
        # Check that density is reasonable
        self.assertGreaterEqual(analysis["entity_density"], 0)
        self.assertLessEqual(analysis["entity_density"], 1)


if __name__ == "__main__":
    unittest.main()
