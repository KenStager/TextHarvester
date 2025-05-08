"""
Custom NER Model Implementation.

This module provides custom Named Entity Recognition model integration
with support for domain-specific entity types and pattern matching.
"""

import logging
import os
import json
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Iterator
from dataclasses import dataclass
import time
import re
from pathlib import Path

import spacy
from spacy.pipeline import EntityRuler
from spacy.tokens import Doc, Span
from spacy.language import Language
from spacy.training import Example
try:
    from spacy.util import minibatch, compounding
except ImportError:
    # For older spaCy versions
    from spacy.util import minibatch
    compounding = None

from intelligence.utils.model_utils import get_model_path
from intelligence.utils.text_processing import normalize_text
from intelligence.config import MODELS_DIR
from intelligence.entities.entity_types import EntityTypeRegistry

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Extracted entity information."""
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float = 1.0
    kb_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "label": self.label,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": self.confidence,
            "kb_id": self.kb_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """Create from dictionary representation."""
        return cls(
            text=data["text"],
            label=data["label"],
            start_char=data["start_char"],
            end_char=data["end_char"],
            confidence=data.get("confidence", 1.0),
            kb_id=data.get("kb_id")
        )
    
    @classmethod
    def from_spacy_span(cls, span: Span, confidence: float = 1.0) -> 'Entity':
        """Create from spaCy span."""
        return cls(
            text=span.text,
            label=span.label_,
            start_char=span.start_char,
            end_char=span.end_char,
            confidence=confidence,
            kb_id=None
        )


class CustomNERModel:
    """
    Custom NER model based on spaCy with domain-specific enhancements.
    
    This class provides a wrapper around spaCy's NER capabilities with
    additional features for domain-specific entity recognition, including
    pattern matching, context rules, and confidence scoring.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm", domain: str = "football",
                 use_gpu: bool = False, load_patterns: bool = True):
        """
        Initialize the NER model.
        
        Args:
            model_name: Base spaCy model to use
            domain: Domain for specialized entity recognition
            use_gpu: Whether to use GPU acceleration
            load_patterns: Whether to load domain-specific patterns
        """
        self.model_name = model_name
        self.domain = domain
        self.use_gpu = use_gpu
        
        # Load base spaCy model
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except IOError:
            logger.warning(f"Could not load {model_name}, downloading...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
        
        # Initialize entity ruler if needed
        if load_patterns:
            self._add_entity_ruler()
            self._load_domain_patterns()
    
    def _add_entity_ruler(self) -> None:
        """Add entity ruler to the pipeline if not already present."""
        # Check if entity ruler already exists
        if "entity_ruler" not in self.nlp.pipe_names:
            # Add entity ruler before the NER component
            config = {"phrase_matcher_attr": "LOWER"}
            ruler = self.nlp.add_pipe("entity_ruler", before="ner", config=config)
            self.entity_ruler = ruler
        else:
            # Get existing entity ruler
            self.entity_ruler = self.nlp.get_pipe("entity_ruler")
    
    def _load_domain_patterns(self) -> None:
        """Load domain-specific NER patterns."""
        patterns_file = os.path.join(MODELS_DIR, "patterns", f"{self.domain}_patterns.jsonl")
        
        if os.path.exists(patterns_file):
            try:
                self.entity_ruler.from_disk(patterns_file)
                logger.info(f"Loaded NER patterns from {patterns_file}")
            except Exception as e:
                logger.error(f"Error loading patterns from {patterns_file}: {e}")
        else:
            logger.info(f"No patterns file found at {patterns_file}, using default patterns")
            self._create_default_patterns()
    
    def _create_default_patterns(self) -> None:
        """Create default patterns based on domain."""
        if self.domain == "football":
            self._create_football_patterns()
        else:
            logger.warning(f"No default patterns available for domain: {self.domain}")
    
    def _create_football_patterns(self) -> None:
        """Create football-specific NER patterns."""
        from intelligence.entities.taxonomies.football_entities import (
            get_premier_league_team_entities,
            get_football_competitions
        )
        
        patterns = []
        
        # Add team patterns
        for team in get_premier_league_team_entities():
            # Main team name
            patterns.append({"label": "TEAM", "pattern": team["name"]})
            
            # Short name
            if "short_name" in team:
                patterns.append({"label": "TEAM", "pattern": team["short_name"]})
            
            # Nicknames
            for nickname in team.get("nickname", []):
                if len(nickname) > 3:  # Avoid short ambiguous nicknames
                    patterns.append({"label": "TEAM", "pattern": nickname})
            
            # Stadium as a VENUE
            if "stadium" in team:
                patterns.append({"label": "VENUE", "pattern": team["stadium"]})
        
        # Add competition patterns
        for comp in get_football_competitions():
            patterns.append({"label": "COMPETITION", "pattern": comp["name"]})
        
        # Add common football positions
        for position in ["goalkeeper", "defender", "midfielder", "forward", "striker", "winger",
                        "center-back", "centre-back", "full-back", "left-back", "right-back"]:
            patterns.append({"label": "POSITION", "pattern": position})
            
        # Add football-specific event terms
        for event in ["goal", "penalty", "free kick", "corner", "yellow card", "red card",
                     "offside", "substitution", "injury", "transfer", "tackle", "save"]:
            patterns.append({"label": "EVENT", "pattern": event})
        
        # Add patterns to entity ruler
        self.entity_ruler.add_patterns(patterns)
        logger.info(f"Added {len(patterns)} football-specific patterns")
    
    def save_model(self, output_dir: Optional[str] = None) -> str:
        """
        Save the NER model to disk.
        
        Args:
            output_dir: Directory to save the model (default: domain-specific path)
            
        Returns:
            Path where the model was saved
        """
        if not output_dir:
            output_dir = os.path.join(MODELS_DIR, "ner", self.domain)
            
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        self.nlp.to_disk(output_dir)
        logger.info(f"Saved NER model to {output_dir}")
        
        # Save patterns separately for easier updates
        if hasattr(self, "entity_ruler"):
            patterns_dir = os.path.join(MODELS_DIR, "patterns")
            os.makedirs(patterns_dir, exist_ok=True)
            patterns_file = os.path.join(patterns_dir, f"{self.domain}_patterns.jsonl")
            self.entity_ruler.to_disk(patterns_file)
            logger.info(f"Saved patterns to {patterns_file}")
        
        return output_dir
    
    @classmethod
    def load_model(cls, model_dir: str, domain: str = "football") -> 'CustomNERModel':
        """
        Load a previously saved NER model.
        
        Args:
            model_dir: Directory where the model is saved
            domain: Domain of the model
            
        Returns:
            Loaded CustomNERModel instance
        """
        instance = cls(model_name=None, domain=domain, load_patterns=False)
        
        # Load the model from disk
        instance.nlp = spacy.load(model_dir)
        logger.info(f"Loaded NER model from {model_dir}")
        
        # Get entity ruler if present
        if "entity_ruler" in instance.nlp.pipe_names:
            instance.entity_ruler = instance.nlp.get_pipe("entity_ruler")
        
        return instance
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of extracted entities
        """
        # Process the text
        doc = self.nlp(text)
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            confidence = getattr(ent, "_.confidence", 1.0)  # Default confidence
            entity = Entity.from_spacy_span(ent, confidence)
            entities.append(entity)
        
        return entities
    
    def extract_entities_with_context(self, text: str, context_window: int = 50) -> List[Dict[str, Any]]:
        """
        Extract entities with surrounding context.
        
        Args:
            text: Text to extract entities from
            context_window: Number of characters for context window
            
        Returns:
            List of entities with context
        """
        # Get entities
        entities = self.extract_entities(text)
        
        # Add context to each entity
        result = []
        for entity in entities:
            # Calculate context boundaries
            start_context = max(0, entity.start_char - context_window)
            end_context = min(len(text), entity.end_char + context_window)
            
            # Extract context
            context_before = text[start_context:entity.start_char]
            context_after = text[entity.end_char:end_context]
            
            # Create result with context
            entity_with_context = entity.to_dict()
            entity_with_context.update({
                "context_before": context_before,
                "context_after": context_after
            })
            
            result.append(entity_with_context)
        
        return result
    
    def add_patterns(self, patterns: List[Dict[str, Any]]) -> None:
        """
        Add entity patterns to the model.
        
        Args:
            patterns: List of patterns to add (each with 'label' and 'pattern')
        """
        if not hasattr(self, "entity_ruler"):
            self._add_entity_ruler()
            
        self.entity_ruler.add_patterns(patterns)
        logger.info(f"Added {len(patterns)} new patterns")
    
    def add_entities_as_patterns(self, entities: List[Dict[str, Any]]) -> None:
        """
        Add entities directly as patterns.
        
        Args:
            entities: List of entity dictionaries with 'text' and 'label'
        """
        patterns = [{"label": entity["label"], "pattern": entity["text"]} 
                   for entity in entities if len(entity["text"]) > 1]
        
        self.add_patterns(patterns)
    
    def train(self, training_data: List[Dict[str, Any]], n_iter: int = 10) -> Dict[str, Any]:
        """
        Train the NER model on custom data.
        
        Args:
            training_data: List of training examples
            n_iter: Number of training iterations
            
        Returns:
            Dictionary with training metrics
        """
        # Prepare training examples
        examples = []
        for item in training_data:
            text = item["text"]
            entities = [(ent["start_char"], ent["end_char"], ent["label"]) 
                       for ent in item["entities"]]
            
            # Create example
            doc = self.nlp.make_doc(text)
            example = Example.from_dict(doc, {"entities": entities})
            examples.append(example)
        
        # Get the NER component
        ner = self.nlp.get_pipe("ner")
        
        # Train the model
        start_time = time.time()
        losses = {}
        
        # Create compounding batch sizes if not available directly
        if compounding is None:
            # Simple batch size increaser
            batches = lambda i: min(4 + (i * 2), 32)
        else:
            # Use spaCy's compounding
            batches = compounding(4.0, 32.0, 1.001)
        
        # Train in batches
        for i in range(n_iter):
            examples_shuffled = examples.copy()
            random.shuffle(examples_shuffled)
            
            # Get batch size for this iteration
            batch_size = next(batches) if callable(batches) else batches(i)
            
            for batch in minibatch(examples_shuffled, size=batch_size):
                self.nlp.update(batch, drop=0.5, losses=losses)
                
            logger.info(f"Training iteration {i+1}/{n_iter}, loss: {losses.get('ner', 0):.4f}")
        
        training_time = time.time() - start_time
        
        # Return metrics
        metrics = {
            "iterations": n_iter,
            "examples": len(examples),
            "training_time": training_time,
            "final_loss": losses.get("ner", 0)
        }
        
        logger.info(f"Completed NER training with {len(examples)} examples in {training_time:.2f}s")
        return metrics


class FootballNERModel(CustomNERModel):
    """
    Football-specific NER model with specialized enhancements.
    
    This class extends the CustomNERModel with football-specific features
    like team formation recognition, player position detection, and match
    event extraction.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm", use_gpu: bool = False):
        """
        Initialize the football NER model.
        
        Args:
            model_name: Base spaCy model to use
            use_gpu: Whether to use GPU acceleration
        """
        super().__init__(model_name=model_name, domain="football", use_gpu=use_gpu)
        
        # Add football-specific components
        self._add_formation_detector()
        self._add_player_position_detector()
        self._add_match_event_detector()
    
    def _add_formation_detector(self) -> None:
        """Add custom component to detect football formations."""
        # Create formation detection component
        @Language.component("formation_detector")
        def formation_detector(doc):
            # Pattern for common football formations like 4-4-2, 3-5-2, etc.
            formation_pattern = re.compile(r'\b([3-5])[-–]([2-5])[-–]([1-3])\b')
            
            matches = formation_pattern.finditer(doc.text)
            new_ents = list(doc.ents)
            
            for match in matches:
                start, end = match.span()
                # Find the token span that corresponds to the character span
                span = doc.char_span(start, end, label="FORMATION")
                if span is not None:
                    new_ents.append(span)
            
            # Filter out any overlapping spans
            filtered_ents = self._filter_overlapping_spans(new_ents)
            doc.ents = filtered_ents
            return doc
        
        # Add component to pipeline
        self.nlp.add_pipe("formation_detector", after="ner")
    
    def _add_player_position_detector(self) -> None:
        """Add custom component to detect player positions and roles."""
        # Create position detection component
        @Language.component("position_detector")
        def position_detector(doc):
            # Common football positions and roles
            positions = {
                "goalkeeper": "POSITION.GOALKEEPER",
                "goalie": "POSITION.GOALKEEPER",
                "keeper": "POSITION.GOALKEEPER",
                "gk": "POSITION.GOALKEEPER",
                "defender": "POSITION.DEFENDER",
                "centre-back": "POSITION.DEFENDER",
                "center-back": "POSITION.DEFENDER",
                "right-back": "POSITION.DEFENDER",
                "left-back": "POSITION.DEFENDER",
                "full-back": "POSITION.DEFENDER",
                "cb": "POSITION.DEFENDER",
                "rb": "POSITION.DEFENDER",
                "lb": "POSITION.DEFENDER",
                "midfielder": "POSITION.MIDFIELDER",
                "central midfielder": "POSITION.MIDFIELDER",
                "defensive midfielder": "POSITION.MIDFIELDER",
                "attacking midfielder": "POSITION.MIDFIELDER",
                "cdm": "POSITION.MIDFIELDER",
                "cam": "POSITION.MIDFIELDER",
                "cm": "POSITION.MIDFIELDER",
                "winger": "POSITION.MIDFIELDER",
                "playmaker": "POSITION.MIDFIELDER",
                "forward": "POSITION.FORWARD",
                "striker": "POSITION.FORWARD",
                "center forward": "POSITION.FORWARD",
                "centre forward": "POSITION.FORWARD",
                "cf": "POSITION.FORWARD",
                "st": "POSITION.FORWARD"
            }
            
            new_ents = list(doc.ents)
            text_lower = doc.text.lower()
            
            for position, label in positions.items():
                # Find all occurrences of this position
                for match in re.finditer(r'\b' + re.escape(position) + r'\b', text_lower):
                    start, end = match.span()
                    # Find the token span that corresponds to the character span
                    span = doc.char_span(start, end, label=label.split('.')[0])
                    if span is not None:
                        new_ents.append(span)
            
            # Filter out any overlapping spans
            filtered_ents = self._filter_overlapping_spans(new_ents)
            doc.ents = filtered_ents
            return doc
        
        # Add component to pipeline
        self.nlp.add_pipe("position_detector", after="formation_detector")
    
    def _add_match_event_detector(self) -> None:
        """Add custom component to detect match events."""
        # Create match event detection component
        @Language.component("match_event_detector")
        def match_event_detector(doc):
            # Common match events
            events = {
                "goal": "EVENT.GOAL",
                "scored": "EVENT.GOAL",
                "scoring": "EVENT.GOAL", 
                "penalty": "EVENT.PENALTY",
                "free kick": "EVENT.FREEKICK",
                "freekick": "EVENT.FREEKICK",
                "corner": "EVENT.CORNER",
                "yellow card": "EVENT.CARD",
                "red card": "EVENT.CARD",
                "booking": "EVENT.CARD",
                "sent off": "EVENT.CARD",
                "substitution": "EVENT.SUBSTITUTION",
                "substituted": "EVENT.SUBSTITUTION",
                "injury": "EVENT.INJURY",
                "injured": "EVENT.INJURY",
                "save": "EVENT.SAVE",
                "saves": "EVENT.SAVE",
                "tackle": "EVENT.TACKLE",
                "tackled": "EVENT.TACKLE",
                "offside": "EVENT.OFFSIDE"
            }
            
            new_ents = list(doc.ents)
            text_lower = doc.text.lower()
            
            for event, label in events.items():
                # Find all occurrences of this event
                for match in re.finditer(r'\b' + re.escape(event) + r'\b', text_lower):
                    start, end = match.span()
                    # Find the token span that corresponds to the character span
                    span = doc.char_span(start, end, label=label.split('.')[0])
                    if span is not None:
                        new_ents.append(span)
            
            # Filter out any overlapping spans
            filtered_ents = self._filter_overlapping_spans(new_ents)
            doc.ents = filtered_ents
            return doc
        
        # Add component to pipeline
        self.nlp.add_pipe("match_event_detector", after="position_detector")
    
    @staticmethod
    def _filter_overlapping_spans(spans: List[Span]) -> List[Span]:
        """
        Filter out overlapping spans, keeping those with highest priority.
        
        Args:
            spans: List of spans to filter
            
        Returns:
            Filtered list of spans
        """
        # Sort spans by length (shorter spans have higher priority)
        sorted_spans = sorted(spans, key=lambda span: (span.end - span.start, span.start))
        
        # Keep track of which character positions have been consumed
        seen_tokens = set()
        filtered_spans = []
        
        for span in sorted_spans:
            # Check if any token in the span has been seen
            if any(token.i in seen_tokens for token in span):
                continue
                
            # Add span to filtered spans
            filtered_spans.append(span)
            
            # Mark all tokens in the span as seen
            seen_tokens.update(token.i for token in span)
        
        return filtered_spans
    
    def extract_match_events(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract football match events with associated information.
        
        Args:
            text: Match report or description
            
        Returns:
            List of extracted match events with associated entities
        """
        # Process text
        doc = self.nlp(text)
        
        # Create entity index by label
        entities_by_label = {}
        for ent in doc.ents:
            label = ent.label_
            if label not in entities_by_label:
                entities_by_label[label] = []
            entities_by_label[label].append(Entity.from_spacy_span(ent))
        
        # Extract match events
        events = []
        
        # Look for goals
        goals = entities_by_label.get("EVENT", [])
        goals = [e for e in goals if e.text.lower() in ["goal", "scored", "scoring"]]
        
        # Associate players with goals
        for goal in goals:
            # Find the closest PERSON entity before and after the goal
            closest_person = self._find_closest_entity(
                goal, entities_by_label.get("PERSON", []), window=100
            )
            
            closest_team = self._find_closest_entity(
                goal, entities_by_label.get("TEAM", []), window=100
            )
            
            # Create event with associated entities
            event = {
                "type": "GOAL",
                "text": goal.text,
                "span": (goal.start_char, goal.end_char),
                "entities": {
                    "scorer": closest_person.to_dict() if closest_person else None,
                    "team": closest_team.to_dict() if closest_team else None
                }
            }
            
            events.append(event)
        
        # Extract other types of events - example for cards
        cards = entities_by_label.get("EVENT", [])
        cards = [e for e in cards if e.text.lower() in ["yellow card", "red card", "booking", "sent off"]]
        
        for card in cards:
            closest_person = self._find_closest_entity(
                card, entities_by_label.get("PERSON", []), window=100
            )
            
            # Create event
            event = {
                "type": "CARD",
                "text": card.text,
                "span": (card.start_char, card.end_char),
                "entities": {
                    "player": closest_person.to_dict() if closest_person else None
                }
            }
            
            events.append(event)
        
        return events
    
    @staticmethod
    def _find_closest_entity(target: Entity, entities: List[Entity], window: int = 50) -> Optional[Entity]:
        """
        Find the closest entity to a target entity.
        
        Args:
            target: Target entity
            entities: List of entities to search
            window: Character window to consider
            
        Returns:
            Closest entity if found, None otherwise
        """
        if not entities:
            return None
            
        closest = None
        min_distance = float('inf')
        
        for entity in entities:
            # Calculate distance between target and entity
            if entity.end_char <= target.start_char:
                # Entity is before target
                distance = target.start_char - entity.end_char
            elif entity.start_char >= target.end_char:
                # Entity is after target
                distance = entity.start_char - target.end_char
            else:
                # Entities overlap
                distance = 0
                
            # Update closest if this entity is closer and within window
            if distance < min_distance and distance <= window:
                min_distance = distance
                closest = entity
                
        return closest
    
    def extract_teams_with_players(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract teams and associated players from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            Dictionary mapping team entities to lists of player entities
        """
        # Get all entities
        doc = self.nlp(text)
        
        # Extract teams and players
        teams = []
        players = []
        
        for ent in doc.ents:
            if ent.label_ == "TEAM":
                teams.append(Entity.from_spacy_span(ent))
            elif ent.label_ == "PERSON":
                players.append(Entity.from_spacy_span(ent))
        
        # Associate players with teams
        result = {}
        
        for team in teams:
            # Find players associated with this team
            team_players = []
            
            # Look for patterns like "X of Team" or "Team's X"
            for player in players:
                # Check for "Player of Team" pattern
                player_end_idx = player.end_char
                of_index = text.find(" of ", player_end_idx, player_end_idx + 10)
                if of_index != -1:
                    # Check if team appears after "of"
                    if text[of_index + 4:of_index + 4 + len(team.text)].lower() == team.text.lower():
                        team_players.append(player.to_dict())
                        continue
                
                # Check for "Team's Player" pattern
                team_possessive = team.text + "'s"
                if team_possessive in text[:player.start_char][-30:]:
                    team_players.append(player.to_dict())
                    continue
                
                # Check for "Team player" pattern (adjacent)
                if text[team.end_char:team.end_char + 30].find(player.text) != -1:
                    team_players.append(player.to_dict())
                    continue
            
            # Add to result
            result[team.text] = team_players
        
        return result
    
    def analyze_match_report(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a football match report.
        
        Args:
            text: Match report text
            
        Returns:
            Dictionary with extracted teams, players, events, and statistics
        """
        # Process text
        doc = self.nlp(text)
        
        # Extract all entities
        entities = {
            "teams": [],
            "players": [],
            "venues": [],
            "competitions": [],
            "events": [],
            "positions": [],
            "dates": [],
            "times": [],
            "results": []
        }
        
        # Basic entity extraction
        for ent in doc.ents:
            if ent.label_ == "TEAM":
                entities["teams"].append(Entity.from_spacy_span(ent).to_dict())
            elif ent.label_ == "PERSON":
                entities["players"].append(Entity.from_spacy_span(ent).to_dict())
            elif ent.label_ == "VENUE":
                entities["venues"].append(Entity.from_spacy_span(ent).to_dict())
            elif ent.label_ == "COMPETITION":
                entities["competitions"].append(Entity.from_spacy_span(ent).to_dict())
            elif ent.label_ == "EVENT":
                entities["events"].append(Entity.from_spacy_span(ent).to_dict())
            elif ent.label_ == "POSITION":
                entities["positions"].append(Entity.from_spacy_span(ent).to_dict())
            elif ent.label_ == "DATE":
                entities["dates"].append(Entity.from_spacy_span(ent).to_dict())
            elif ent.label_ == "TIME":
                entities["times"].append(Entity.from_spacy_span(ent).to_dict())
        
        # Extract score pattern (e.g., "2-1", "3:0")
        score_pattern = re.compile(r'\b(\d{1,2})[-:–](\d{1,2})\b')
        for match in score_pattern.finditer(text):
            score_text = match.group()
            entities["results"].append({
                "text": score_text,
                "start_char": match.start(),
                "end_char": match.end(),
                "home_score": int(match.group(1)),
                "away_score": int(match.group(2))
            })
        
        # Extract relationships
        relationships = {
            "team_players": self.extract_teams_with_players(text),
            "match_events": self.extract_match_events(text)
        }
        
        # Try to determine the main teams and score
        main_result = None
        if entities["results"] and len(entities["teams"]) >= 2:
            # Find the most prominent result
            main_result = {
                "score": entities["results"][0],
                "home_team": entities["teams"][0]["text"],
                "away_team": entities["teams"][1]["text"] if len(entities["teams"]) > 1 else None
            }
        
        return {
            "entities": entities,
            "relationships": relationships,
            "main_result": main_result,
            "title": self._extract_title(text)
        }
    
    @staticmethod
    def _extract_title(text: str) -> Optional[str]:
        """
        Try to extract a title from the text.
        
        Args:
            text: Text to extract title from
            
        Returns:
            Extracted title or None
        """
        # Try to find the first sentence or line
        lines = text.split('\n')
        first_non_empty = next((line.strip() for line in lines if line.strip()), None)
        
        if first_non_empty and len(first_non_empty) <= 150:
            return first_non_empty
        
        # If first line is too long, try to find a shorter first sentence
        sentences = text.split('.')
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) <= 150:
                return first_sentence
            elif len(first_sentence) > 150:
                # Try to truncate to a reasonable title length
                return first_sentence[:100] + "..."
        
        return None
