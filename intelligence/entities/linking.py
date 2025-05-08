"""
Entity Linking and Disambiguation System.

This module provides functionality for linking extracted entities to a knowledge base,
resolving ambiguities, and maintaining entity relationships.
"""

import logging
import json
import os
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass, field
import re
import math
from collections import defaultdict

from intelligence.config import MODELS_DIR
from intelligence.entities.ner_model import Entity
from db.models.entity_models import Entity as DbEntity, EntityMention as DbEntityMention

logger = logging.getLogger(__name__)


@dataclass
class EntityCandidate:
    """A candidate entity for linking."""
    kb_id: str
    name: str
    score: float
    type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "kb_id": self.kb_id,
            "name": self.name,
            "score": self.score,
            "type": self.type,
            "attributes": self.attributes
        }


@dataclass
class LinkedEntity:
    """An entity linked to a knowledge base entry."""
    mention: Entity
    kb_id: str
    score: float
    kb_name: str
    type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "mention": self.mention.to_dict(),
            "kb_id": self.kb_id,
            "score": self.score,
            "kb_name": self.kb_name,
            "type": self.type,
            "attributes": self.attributes
        }


class KnowledgeBase:
    """
    Knowledge base for entity linking.
    
    This class provides a lookup system for entities, support for disambiguation,
    and management of entity relationships.
    """
    
    def __init__(self, name: str, domain: str = "generic"):
        """
        Initialize the knowledge base.
        
        Args:
            name: Name of the knowledge base
            domain: Domain of the knowledge base
        """
        self.name = name
        self.domain = domain
        self.entities = {}  # Map of kb_id to entity data
        self.name_to_ids = defaultdict(list)  # Map of normalized name to list of kb_ids
        self.aliases = defaultdict(list)  # Map of alias to list of kb_ids
        self.types = defaultdict(list)  # Map of type to list of kb_ids
        
    def add_entity(self, kb_id: str, name: str, entity_type: str, 
                  attributes: Dict[str, Any] = None, aliases: List[str] = None) -> None:
        """
        Add an entity to the knowledge base.
        
        Args:
            kb_id: Unique identifier for the entity
            name: Primary name of the entity
            entity_type: Type of the entity
            attributes: Optional attributes for the entity
            aliases: Optional aliases for the entity
        """
        # Add to main entity dictionary
        self.entities[kb_id] = {
            "kb_id": kb_id,
            "name": name,
            "type": entity_type,
            "attributes": attributes or {},
            "aliases": aliases or []
        }
        
        # Add to name index
        normalized_name = self._normalize_text(name)
        self.name_to_ids[normalized_name].append(kb_id)
        
        # Add to type index
        self.types[entity_type].append(kb_id)
        
        # Add aliases
        if aliases:
            for alias in aliases:
                normalized_alias = self._normalize_text(alias)
                self.aliases[normalized_alias].append(kb_id)
                
    def add_entities_from_list(self, entities: List[Dict[str, Any]]) -> None:
        """
        Add multiple entities from a list.
        
        Args:
            entities: List of entity dictionaries
        """
        for entity in entities:
            self.add_entity(
                kb_id=entity.get("kb_id", f"{self.domain}_{len(self.entities)}"),
                name=entity["name"],
                entity_type=entity["type"],
                attributes=entity.get("attributes", {}),
                aliases=entity.get("aliases", [])
            )
            
    def get_entity(self, kb_id: str) -> Optional[Dict[str, Any]]:
        """
        Get entity by ID.
        
        Args:
            kb_id: Entity ID to look up
            
        Returns:
            Entity data or None if not found
        """
        return self.entities.get(kb_id)
    
    def get_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """
        Get all entities of a specific type.
        
        Args:
            entity_type: Type to filter by
            
        Returns:
            List of matching entities
        """
        return [self.entities[kb_id] for kb_id in self.types.get(entity_type, [])]
    
    def find_candidates(self, text: str, entity_type: Optional[str] = None,
                       threshold: float = 0.5, max_candidates: int = 5) -> List[EntityCandidate]:
        """
        Find candidate entities that match the text.
        
        Args:
            text: Text to match
            entity_type: Optional type to filter by
            threshold: Minimum similarity score
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of candidate entities
        """
        normalized_text = self._normalize_text(text)
        candidates = []
        
        # Check exact name matches
        for kb_id in self.name_to_ids.get(normalized_text, []):
            entity = self.entities[kb_id]
            # Filter by type if specified
            if entity_type and not self._is_compatible_type(entity["type"], entity_type):
                continue
                
            candidates.append(EntityCandidate(
                kb_id=kb_id,
                name=entity["name"],
                score=1.0,  # Exact match
                type=entity["type"],
                attributes=entity["attributes"]
            ))
            
        # Check alias matches
        for kb_id in self.aliases.get(normalized_text, []):
            entity = self.entities[kb_id]
            # Filter by type if specified
            if entity_type and not self._is_compatible_type(entity["type"], entity_type):
                continue
                
            # Don't add duplicates
            if any(c.kb_id == kb_id for c in candidates):
                continue
                
            candidates.append(EntityCandidate(
                kb_id=kb_id,
                name=entity["name"],
                score=0.9,  # Alias match (slightly lower than exact)
                type=entity["type"],
                attributes=entity["attributes"]
            ))
            
        # If we have exact or alias matches, no need for fuzzy matching
        if candidates:
            return sorted(candidates, key=lambda c: c.score, reverse=True)[:max_candidates]
            
        # Try fuzzy matching if no exact matches found
        fuzzy_candidates = self._fuzzy_match(
            normalized_text, 
            entity_type=entity_type,
            threshold=threshold
        )
        
        # Combine and sort by score
        all_candidates = candidates + fuzzy_candidates
        all_candidates.sort(key=lambda c: c.score, reverse=True)
        
        return all_candidates[:max_candidates]
    
    def _fuzzy_match(self, text: str, entity_type: Optional[str] = None,
                    threshold: float = 0.5) -> List[EntityCandidate]:
        """
        Find entities with fuzzy matching.
        
        Args:
            text: Text to match
            entity_type: Optional type to filter by
            threshold: Minimum similarity score
            
        Returns:
            List of candidate entities
        """
        candidates = []
        
        # Filter entities by type if specified
        if entity_type:
            entity_ids = self.types.get(entity_type, [])
            # Also include subtypes
            for type_name, type_ids in self.types.items():
                if self._is_compatible_type(type_name, entity_type) and type_name != entity_type:
                    entity_ids.extend(type_ids)
        else:
            entity_ids = list(self.entities.keys())
        
        # Calculate similarity for each entity
        for kb_id in entity_ids:
            entity = self.entities[kb_id]
            
            # Calculate similarity with name
            name_similarity = self._calculate_similarity(text, entity["name"])
            
            # Calculate similarity with aliases
            alias_similarities = [
                self._calculate_similarity(text, alias) for alias in entity["aliases"]
            ]
            
            # Use highest similarity score
            max_similarity = max([name_similarity] + alias_similarities)
            
            # Add if above threshold
            if max_similarity >= threshold:
                candidates.append(EntityCandidate(
                    kb_id=kb_id,
                    name=entity["name"],
                    score=max_similarity,
                    type=entity["type"],
                    attributes=entity["attributes"]
                ))
        
        # Sort by score
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates
    
    def disambiguate_entity(self, entity: Entity, context: str, 
                          candidates: Optional[List[EntityCandidate]] = None) -> Optional[LinkedEntity]:
        """
        Disambiguate an entity mention using context.
        
        Args:
            entity: Entity mention to disambiguate
            context: Context text around the entity
            candidates: Optional pre-calculated candidates
            
        Returns:
            Linked entity or None if no match found
        """
        # Get candidates if not provided
        if not candidates:
            candidates = self.find_candidates(entity.text, entity_type=entity.label)
            
        if not candidates:
            return None
            
        # If only one candidate, return it directly
        if len(candidates) == 1:
            return LinkedEntity(
                mention=entity,
                kb_id=candidates[0].kb_id,
                score=candidates[0].score,
                kb_name=candidates[0].name,
                type=candidates[0].type,
                attributes=candidates[0].attributes
            )
            
        # Multiple candidates, use context for disambiguation
        # Calculate context relevance for each candidate
        candidates_with_context = []
        
        for candidate in candidates:
            # Get entity data
            entity_data = self.entities[candidate.kb_id]
            
            # Calculate context relevance
            context_score = self._calculate_context_relevance(
                context, entity_data["name"], entity_data["attributes"], entity_data["aliases"]
            )
            
            # Combined score: name similarity * context relevance
            combined_score = candidate.score * (0.5 + 0.5 * context_score)
            
            candidates_with_context.append((candidate, combined_score))
            
        # Sort by combined score
        candidates_with_context.sort(key=lambda x: x[1], reverse=True)
        
        # Get best candidate
        best_candidate, best_score = candidates_with_context[0]
        
        # Only link if score is high enough
        if best_score >= 0.5:
            return LinkedEntity(
                mention=entity,
                kb_id=best_candidate.kb_id,
                score=best_score,
                kb_name=best_candidate.name,
                type=best_candidate.type,
                attributes=best_candidate.attributes
            )
            
        return None
    
    def link_entities(self, entities: List[Entity], text: str) -> List[LinkedEntity]:
        """
        Link multiple entities to knowledge base entries.
        
        Args:
            entities: List of entity mentions
            text: Full text containing the entities
            
        Returns:
            List of linked entities
        """
        linked_entities = []
        
        for entity in entities:
            # Get context around the entity
            context = self._extract_context(text, entity, window=100)
            
            # Get candidates
            candidates = self.find_candidates(entity.text, entity_type=entity.label)
            
            # Disambiguate
            linked_entity = self.disambiguate_entity(entity, context, candidates)
            
            if linked_entity:
                linked_entities.append(linked_entity)
                
        return linked_entities
    
    def save_to_file(self, filepath: Optional[str] = None) -> str:
        """
        Save knowledge base to a file.
        
        Args:
            filepath: Optional path to save to
            
        Returns:
            Path where the file was saved
        """
        if not filepath:
            # Create default filepath
            kb_dir = os.path.join(MODELS_DIR, "kb")
            os.makedirs(kb_dir, exist_ok=True)
            filepath = os.path.join(kb_dir, f"{self.domain}_{self.name}.json")
            
        # Prepare data
        kb_data = {
            "name": self.name,
            "domain": self.domain,
            "entities": list(self.entities.values())
        }
        
        # Save to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(kb_data, f, indent=2)
            
        logger.info(f"Saved knowledge base to {filepath}")
        return filepath
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'KnowledgeBase':
        """
        Load knowledge base from a file.
        
        Args:
            filepath: Path to load from
            
        Returns:
            Loaded knowledge base
        """
        with open(filepath, "r", encoding="utf-8") as f:
            kb_data = json.load(f)
            
        # Create knowledge base
        kb = cls(name=kb_data["name"], domain=kb_data.get("domain", "generic"))
        
        # Add entities
        kb.add_entities_from_list(kb_data["entities"])
        
        logger.info(f"Loaded knowledge base from {filepath} with {len(kb.entities)} entities")
        return kb
    
    def save_to_database(self) -> List[DbEntity]:
        """
        Save knowledge base to the database.
        
        Returns:
            List of database entity objects
        """
        db_entities = []
        
        for kb_id, entity_data in self.entities.items():
            # Create or update database entity
            db_entity = DbEntity.get_by_kb_id(kb_id)
            
            if not db_entity:
                db_entity = DbEntity(
                    name=entity_data["name"],
                    entity_type_id=self._get_db_type_id(entity_data["type"]),
                    kb_id=kb_id,
                    canonical_name=entity_data["name"],
                    metadata=entity_data["attributes"]
                )
                db_entity.save()
            else:
                # Update existing
                db_entity.name = entity_data["name"]
                db_entity.entity_type_id = self._get_db_type_id(entity_data["type"])
                db_entity.canonical_name = entity_data["name"]
                db_entity.metadata = entity_data["attributes"]
                db_entity.save()
                
            db_entities.append(db_entity)
            
        logger.info(f"Saved {len(db_entities)} entities to database")
        return db_entities
    
    def _get_db_type_id(self, type_name: str) -> int:
        """
        Get database entity type ID for a type name.
        
        Args:
            type_name: Type name to look up
            
        Returns:
            Database entity type ID
        """
        # In a real implementation, this would look up the ID in the database
        # For this stub, we'll return a placeholder
        return 1
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize text for matching.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def _is_compatible_type(actual_type: str, expected_type: str) -> bool:
        """
        Check if an actual type is compatible with expected type.
        
        Args:
            actual_type: Actual entity type
            expected_type: Expected entity type
            
        Returns:
            True if compatible, False otherwise
        """
        # Exact match
        if actual_type == expected_type:
            return True
            
        # Check if actual type is a subtype of expected type
        # In a hierarchy like "PERSON.PLAYER", "PERSON.PLAYER" is a subtype of "PERSON"
        return actual_type.startswith(expected_type + ".")
    
    @staticmethod
    def _calculate_similarity(text1: str, text2: str) -> float:
        """
        Calculate similarity between two strings.
        
        Args:
            text1: First string
            text2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize texts
        norm1 = KnowledgeBase._normalize_text(text1)
        norm2 = KnowledgeBase._normalize_text(text2)
        
        # Check for exact match
        if norm1 == norm2:
            return 1.0
            
        # Check for containment
        if norm1 in norm2 or norm2 in norm1:
            # Length ratio determines score
            len_ratio = min(len(norm1), len(norm2)) / max(len(norm1), len(norm2))
            return 0.8 * len_ratio + 0.2
            
        # Calculate Levenshtein distance
        distance = KnowledgeBase._levenshtein_distance(norm1, norm2)
        max_len = max(len(norm1), len(norm2))
        
        if max_len == 0:
            return 1.0
            
        # Convert distance to similarity
        return 1.0 - (distance / max_len)
    
    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """
        Calculate Levenshtein (edit) distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Edit distance
        """
        if len(s1) < len(s2):
            return KnowledgeBase._levenshtein_distance(s2, s1)
            
        # len(s1) >= len(s2)
        if len(s2) == 0:
            return len(s1)
            
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]
    
    @staticmethod
    def _calculate_context_relevance(context: str, name: str, attributes: Dict[str, Any],
                                   aliases: List[str]) -> float:
        """
        Calculate relevance of an entity to a context.
        
        Args:
            context: Context text
            name: Entity name
            attributes: Entity attributes
            aliases: Entity aliases
            
        Returns:
            Relevance score between 0 and 1
        """
        # Normalize context
        context_lower = context.lower()
        
        # Count occurrences of name
        name_count = context_lower.count(name.lower())
        
        # Count occurrences of aliases
        alias_count = sum(context_lower.count(alias.lower()) for alias in aliases)
        
        # Count occurrences of attribute values
        attribute_count = 0
        for key, value in attributes.items():
            if isinstance(value, str) and len(value) > 3:
                attribute_count += context_lower.count(value.lower())
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and len(item) > 3:
                        attribute_count += context_lower.count(item.lower())
                        
        # Calculate total relevance
        total_count = name_count + alias_count + attribute_count
        
        # Normalize to a score between 0 and 1
        # More occurrences increase score but with diminishing returns
        return min(1.0, math.log(1 + total_count) / 2)
    
    @staticmethod
    def _extract_context(text: str, entity: Entity, window: int = 100) -> str:
        """
        Extract context around an entity.
        
        Args:
            text: Full text
            entity: Entity to extract context for
            window: Character window on each side
            
        Returns:
            Context text
        """
        start = max(0, entity.start_char - window)
        end = min(len(text), entity.end_char + window)
        
        return text[start:end]


class FootballKnowledgeBase(KnowledgeBase):
    """
    Football-specific knowledge base with specialized disambiguation rules.
    
    This class extends the generic KnowledgeBase with football-specific features
    like team awareness for player disambiguation and handling of name variations.
    """
    
    def __init__(self):
        """Initialize the football knowledge base."""
        super().__init__(name="football", domain="football")
        
        # Add team relationships
        self.player_teams = {}  # Map of player KB ID to team KB ID
        self.team_players = defaultdict(list)  # Map of team KB ID to list of player KB IDs
        
        # Add additional indexes
        self.positions = defaultdict(list)  # Map of position to list of player KB IDs
        
    def add_entity(self, kb_id: str, name: str, entity_type: str,
                  attributes: Dict[str, Any] = None, aliases: List[str] = None) -> None:
        """
        Add an entity to the knowledge base with football-specific processing.
        
        Args:
            kb_id: Unique identifier for the entity
            name: Primary name of the entity
            entity_type: Type of the entity
            attributes: Optional attributes for the entity
            aliases: Optional aliases for the entity
        """
        # Call parent method
        super().add_entity(kb_id, name, entity_type, attributes, aliases)
        
        # Add football-specific indexing
        attributes = attributes or {}
        
        if entity_type.startswith("PERSON.PLAYER"):
            # Add position index
            position = attributes.get("position")
            if position:
                self.positions[position].append(kb_id)
                
            # Add team relationship
            team = attributes.get("team")
            if team:
                # Find team KB ID
                team_ids = [tid for tid, t in self.entities.items() 
                           if t["type"].startswith("TEAM") and t["name"] == team]
                
                if team_ids:
                    team_id = team_ids[0]
                    self.player_teams[kb_id] = team_id
                    self.team_players[team_id].append(kb_id)
    
    def disambiguate_entity(self, entity: Entity, context: str,
                          candidates: Optional[List[EntityCandidate]] = None) -> Optional[LinkedEntity]:
        """
        Disambiguate a football entity with specialized rules.
        
        Args:
            entity: Entity mention to disambiguate
            context: Context text around the entity
            candidates: Optional pre-calculated candidates
            
        Returns:
            Linked entity or None if no match found
        """
        # Get candidates if not provided
        if not candidates:
            candidates = self.find_candidates(entity.text, entity_type=entity.label)
            
        if not candidates:
            return None
            
        # If only one candidate, return it directly
        if len(candidates) == 1:
            return LinkedEntity(
                mention=entity,
                kb_id=candidates[0].kb_id,
                score=candidates[0].score,
                kb_name=candidates[0].name,
                type=candidates[0].type,
                attributes=candidates[0].attributes
            )
            
        # For player entities, try to use team context
        if entity.label in ["PERSON", "PLAYER"] and len(candidates) > 1:
            # Extract team mentions from context
            team_mentions = self._extract_team_mentions(context)
            
            if team_mentions:
                # Try to match players with teams
                for candidate in candidates:
                    player_team_id = self.player_teams.get(candidate.kb_id)
                    if player_team_id and player_team_id in team_mentions:
                        # Boost score for players from mentioned teams
                        return LinkedEntity(
                            mention=entity,
                            kb_id=candidate.kb_id,
                            score=min(1.0, candidate.score * 1.3),  # Boost score
                            kb_name=candidate.name,
                            type=candidate.type,
                            attributes=candidate.attributes
                        )
        
        # Fall back to generic disambiguation
        return super().disambiguate_entity(entity, context, candidates)
    
    def _extract_team_mentions(self, text: str) -> Set[str]:
        """
        Extract team mentions from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            Set of team KB IDs mentioned in text
        """
        team_mentions = set()
        
        # Check for team names and aliases
        for kb_id, entity_data in self.entities.items():
            if entity_data["type"].startswith("TEAM"):
                name_lower = entity_data["name"].lower()
                if name_lower in text.lower():
                    team_mentions.add(kb_id)
                    continue
                    
                # Check aliases
                for alias in entity_data["aliases"]:
                    if alias.lower() in text.lower():
                        team_mentions.add(kb_id)
                        break
                        
        return team_mentions
    
    @classmethod
    def create_from_football_data(cls) -> 'FootballKnowledgeBase':
        """
        Create a football knowledge base from predefined data.
        
        Returns:
            Initialized football knowledge base
        """
        from intelligence.entities.taxonomies.football_entities import (
            get_premier_league_team_entities,
            get_football_competitions
        )
        
        # Create knowledge base
        kb = cls()
        
        # Add teams
        for team_data in get_premier_league_team_entities():
            kb_id = f"team_{team_data['name'].replace(' ', '_').lower()}"
            
            # Add aliases
            aliases = team_data.get("nickname", []) + [team_data.get("short_name", "")]
            
            # Add entity
            kb.add_entity(
                kb_id=kb_id,
                name=team_data["name"],
                entity_type="TEAM.CLUB",
                attributes={
                    "country": team_data.get("country"),
                    "city": team_data.get("city"),
                    "stadium": team_data.get("stadium"),
                    "founded": team_data.get("founded"),
                    "league": team_data.get("league"),
                    "colors": team_data.get("colors", [])
                },
                aliases=aliases
            )
            
        # Add competitions
        for comp_data in get_football_competitions():
            kb_id = f"competition_{comp_data['name'].replace(' ', '_').lower()}"
            
            # Add entity
            kb.add_entity(
                kb_id=kb_id,
                name=comp_data["name"],
                entity_type=comp_data["type"],
                attributes={
                    "country": comp_data.get("country"),
                    "organizer": comp_data.get("organizer"),
                    "teams": comp_data.get("teams"),
                    "season": comp_data.get("season"),
                    "founded": comp_data.get("founded")
                }
            )
            
        logger.info(f"Created football knowledge base with {len(kb.entities)} entities")
        return kb


class EntityLinker:
    """
    Entity linking system that connects mentions to knowledge base entries.
    
    This class provides a workflow for linking entity mentions to knowledge base
    entries, handling disambiguation, and managing the linking process.
    """
    
    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None, domain: str = "football"):
        """
        Initialize the entity linker.
        
        Args:
            knowledge_base: Optional knowledge base to use
            domain: Domain for the linker
        """
        self.domain = domain
        
        # Use provided knowledge base or create a domain-specific one
        if knowledge_base:
            self.kb = knowledge_base
        elif domain == "football":
            self.kb = FootballKnowledgeBase.create_from_football_data()
        else:
            self.kb = KnowledgeBase(name=f"{domain}_kb", domain=domain)
            
    def link_entities(self, entities: List[Entity], text: str) -> List[LinkedEntity]:
        """
        Link entity mentions to knowledge base entries.
        
        Args:
            entities: List of entity mentions
            text: Full text containing the entities
            
        Returns:
            List of linked entities
        """
        return self.kb.link_entities(entities, text)
    
    def link_entity(self, entity: Entity, context: str) -> Optional[LinkedEntity]:
        """
        Link a single entity with context.
        
        Args:
            entity: Entity mention to link
            context: Context around the entity
            
        Returns:
            Linked entity or None if no match found
        """
        candidates = self.kb.find_candidates(entity.text, entity_type=entity.label)
        return self.kb.disambiguate_entity(entity, context, candidates)
    
    def suggest_new_entity(self, entity: Entity, entity_type: str) -> Dict[str, Any]:
        """
        Suggest a new entity for the knowledge base.
        
        Args:
            entity: Entity mention to create entity from
            entity_type: Type for the new entity
            
        Returns:
            Suggested entity data
        """
        # Create a unique KB ID
        kb_id = f"{self.domain}_{entity_type.lower()}_{len(self.kb.entities)}"
        
        # Create entity data
        entity_data = {
            "kb_id": kb_id,
            "name": entity.text,
            "type": entity_type,
            "attributes": {},
            "aliases": []
        }
        
        return entity_data
    
    def add_suggested_entity(self, entity_data: Dict[str, Any]) -> str:
        """
        Add a suggested entity to the knowledge base.
        
        Args:
            entity_data: Entity data to add
            
        Returns:
            KB ID of the added entity
        """
        self.kb.add_entity(
            kb_id=entity_data["kb_id"],
            name=entity_data["name"],
            entity_type=entity_data["type"],
            attributes=entity_data["attributes"],
            aliases=entity_data["aliases"]
        )
        
        logger.info(f"Added new entity: {entity_data['name']} ({entity_data['type']})")
        return entity_data["kb_id"]
    
    def save_to_file(self, filepath: Optional[str] = None) -> str:
        """
        Save the knowledge base to a file.
        
        Args:
            filepath: Optional path to save to
            
        Returns:
            Path where the file was saved
        """
        return self.kb.save_to_file(filepath)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'EntityLinker':
        """
        Load an entity linker from a file.
        
        Args:
            filepath: Path to load from
            
        Returns:
            Loaded entity linker
        """
        kb = KnowledgeBase.load_from_file(filepath)
        return cls(knowledge_base=kb, domain=kb.domain)
    
    def save_to_database(self) -> None:
        """Save the knowledge base to the database."""
        self.kb.save_to_database()
    
    def save_linked_entities(self, content_id: int, linked_entities: List[LinkedEntity]) -> List[DbEntityMention]:
        """
        Save linked entities to the database.
        
        Args:
            content_id: ID of the content
            linked_entities: List of linked entities to save
            
        Returns:
            List of database entity mention objects
        """
        db_mentions = []
        
        for linked_entity in linked_entities:
            # Get entity ID for KB ID
            entity_id = self._get_db_entity_id(linked_entity.kb_id)
            
            if not entity_id:
                continue
                
            # Create entity mention
            db_mention = DbEntityMention(
                content_id=content_id,
                entity_id=entity_id,
                start_char=linked_entity.mention.start_char,
                end_char=linked_entity.mention.end_char,
                mention_text=linked_entity.mention.text,
                confidence=linked_entity.score
            )
            db_mention.save()
            
            db_mentions.append(db_mention)
            
        logger.info(f"Saved {len(db_mentions)} entity mentions for content ID {content_id}")
        return db_mentions
    
    def _get_db_entity_id(self, kb_id: str) -> Optional[int]:
        """
        Get database entity ID for a KB ID.
        
        Args:
            kb_id: KB ID to look up
            
        Returns:
            Database entity ID or None if not found
        """
        # In a real implementation, this would look up the ID in the database
        # For this stub, we'll return a placeholder
        return 1
