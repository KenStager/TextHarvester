"""
Entity Extraction Pipeline.

This module implements the complete end-to-end entity extraction pipeline,
coordinating entity recognition, linking, and storage.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass, field

from intelligence.base_pipeline import BasePipeline
from intelligence.utils.text_processing import normalize_text
from intelligence.entities.ner_model import Entity, CustomNERModel, FootballNERModel
from intelligence.entities.linking import EntityLinker, LinkedEntity
from db.models.entity_models import EntityMention as DbEntityMention

logger = logging.getLogger(__name__)


@dataclass
class EntityExtractionInput:
    """Data class for entity extraction pipeline input."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_id: Optional[int] = None


@dataclass
class EntityExtractionOutput:
    """Data class for entity extraction pipeline output."""
    content_id: Optional[int]
    entities: List[Union[Entity, LinkedEntity]]
    linked_entities: List[LinkedEntity]
    unlinked_entities: List[Entity]
    entity_counts: Dict[str, int]
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "content_id": self.content_id,
            "entities": [e.to_dict() for e in self.entities],
            "linked_entities": [e.to_dict() for e in self.linked_entities],
            "unlinked_entities": [e.to_dict() for e in self.unlinked_entities],
            "entity_counts": self.entity_counts,
            "processing_time": self.processing_time
        }


class EntityExtractionPipeline(BasePipeline):
    """
    Entity extraction pipeline for recognizing and linking entities.
    
    This pipeline handles the end-to-end process of extracting entities from text,
    linking them to knowledge base entries, and storing the results.
    """
    
    def __init__(self, ner_model: Optional[CustomNERModel] = None,
                 entity_linker: Optional[EntityLinker] = None,
                 domain: str = "football", confidence_threshold: float = 0.5):
        """
        Initialize the entity extraction pipeline.
        
        Args:
            ner_model: Optional NER model to use
            entity_linker: Optional entity linker to use
            domain: Domain for entity extraction
            confidence_threshold: Confidence threshold for entities
        """
        super().__init__("entity_extraction")
        
        self.domain = domain
        self.confidence_threshold = confidence_threshold
        
        # Initialize components if not provided
        self.ner_model = ner_model or self._initialize_ner_model()
        self.entity_linker = entity_linker or self._initialize_entity_linker()
    
    def _initialize_ner_model(self) -> CustomNERModel:
        """
        Initialize the NER model based on domain.
        
        Returns:
            Initialized NER model
        """
        if self.domain == "football":
            return FootballNERModel()
        else:
            return CustomNERModel(domain=self.domain)
    
    def _initialize_entity_linker(self) -> EntityLinker:
        """
        Initialize the entity linker based on domain.
        
        Returns:
            Initialized entity linker
        """
        return EntityLinker(domain=self.domain)
    
    def process(self, input_data: Union[EntityExtractionInput, Dict, str]) -> EntityExtractionOutput:
        """
        Process input through the entity extraction pipeline.
        
        Args:
            input_data: Input data to process (EntityExtractionInput, Dict, or str)
            
        Returns:
            Entity extraction results
        """
        # Convert input to EntityExtractionInput if needed
        if isinstance(input_data, str):
            input_data = EntityExtractionInput(text=input_data)
        elif isinstance(input_data, dict):
            input_data = EntityExtractionInput(**input_data)
        
        # Validate input
        if not input_data.text:
            logger.warning("Empty text provided to entity extraction pipeline")
            return EntityExtractionOutput(
                content_id=input_data.content_id,
                entities=[],
                linked_entities=[],
                unlinked_entities=[],
                entity_counts={},
                processing_time=0.0
            )
        
        # Start timing
        start_time = time.time()
        
        try:
            # Step 1: Extract entities
            entities = self._extract_entities(input_data.text)
            
            # Step 2: Link entities
            linked_entities = self._link_entities(entities, input_data.text)
            
            # Separate linked and unlinked entities
            unlinked_entities = [e for e in entities if not any(le.mention.text == e.text and 
                                                              le.mention.start_char == e.start_char 
                                                              for le in linked_entities)]
            
            # Step 3: Count entities by type
            entity_counts = self._count_entities(entities)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create output
            output = EntityExtractionOutput(
                content_id=input_data.content_id,
                entities=entities,
                linked_entities=linked_entities,
                unlinked_entities=unlinked_entities,
                entity_counts=entity_counts,
                processing_time=processing_time
            )
            
            logger.info(f"Extracted {len(entities)} entities ({len(linked_entities)} linked) in {processing_time:.4f}s")
            return output
            
        except Exception as e:
            logger.error(f"Error in entity extraction pipeline: {str(e)}", exc_info=True)
            processing_time = time.time() - start_time
            
            # Return empty output on error
            return EntityExtractionOutput(
                content_id=input_data.content_id,
                entities=[],
                linked_entities=[],
                unlinked_entities=[],
                entity_counts={},
                processing_time=processing_time
            )
    
    def _extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of extracted entities
        """
        entities = self.ner_model.extract_entities(text)
        
        # Filter by confidence threshold
        entities = [e for e in entities if e.confidence >= self.confidence_threshold]
        
        return entities
    
    def _link_entities(self, entities: List[Entity], text: str) -> List[LinkedEntity]:
        """
        Link entities to knowledge base entries.
        
        Args:
            entities: Extracted entities
            text: Original text
            
        Returns:
            List of linked entities
        """
        return self.entity_linker.link_entities(entities, text)
    
    def _count_entities(self, entities: List[Entity]) -> Dict[str, int]:
        """
        Count entities by type.
        
        Args:
            entities: List of entities
            
        Returns:
            Dictionary mapping entity types to counts
        """
        counts = {}
        
        for entity in entities:
            entity_type = entity.label
            counts[entity_type] = counts.get(entity_type, 0) + 1
            
        return counts
    
    def save_entities_to_database(self, output: EntityExtractionOutput) -> bool:
        """
        Save extracted entities to the database.
        
        Args:
            output: Entity extraction output
            
        Returns:
            True if successful, False otherwise
        """
        if not output.content_id:
            logger.warning("Cannot save entities with no content ID")
            return False
            
        try:
            # Save linked entities
            db_mentions = self.entity_linker.save_linked_entities(
                output.content_id, output.linked_entities
            )
            
            # Save unlinked entities
            for entity in output.unlinked_entities:
                # Create entity mention without entity reference
                db_mention = DbEntityMention(
                    content_id=output.content_id,
                    entity_id=None,
                    start_char=entity.start_char,
                    end_char=entity.end_char,
                    mention_text=entity.text,
                    confidence=entity.confidence
                )
                db_mention.save()
                db_mentions.append(db_mention)
                
            logger.info(f"Saved {len(db_mentions)} entity mentions to database for content ID {output.content_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving entities to database: {str(e)}", exc_info=True)
            return False
    
    def extract_entities_with_context(self, text: str, context_window: int = 50) -> List[Dict[str, Any]]:
        """
        Extract entities with surrounding context.
        
        Args:
            text: Text to extract entities from
            context_window: Number of characters for context window
            
        Returns:
            List of entities with context
        """
        return self.ner_model.extract_entities_with_context(text, context_window)
    
    def analyze_entity_distribution(self, text: str) -> Dict[str, Any]:
        """
        Analyze entity distribution in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with entity distribution statistics
        """
        # Extract entities
        entities = self._extract_entities(text)
        
        # Count by type
        counts_by_type = self._count_entities(entities)
        
        # Calculate density
        total_words = len(text.split())
        entity_density = len(entities) / total_words if total_words > 0 else 0
        
        # Calculate distribution by position
        position_bins = 10
        bin_size = len(text) / position_bins
        position_distribution = [0] * position_bins
        
        for entity in entities:
            position = int(entity.start_char / bin_size)
            if position >= position_bins:
                position = position_bins - 1
            position_distribution[position] += 1
            
        # Analyze co-occurrence
        co_occurrence = {}
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                # Skip if same entity type
                if entity1.label == entity2.label:
                    continue
                    
                pair = (entity1.label, entity2.label)
                if pair not in co_occurrence:
                    co_occurrence[pair] = 0
                co_occurrence[pair] += 1
                
        # Format co-occurrence as list
        co_occurrence_list = [
            {"types": list(pair), "count": count}
            for pair, count in co_occurrence.items()
        ]
        
        return {
            "total_entities": len(entities),
            "counts_by_type": counts_by_type,
            "entity_density": entity_density,
            "position_distribution": position_distribution,
            "co_occurrence": co_occurrence_list
        }
    
    @classmethod
    def create_football_pipeline(cls) -> 'EntityExtractionPipeline':
        """
        Create a pre-configured pipeline for football.
        
        Returns:
            Configured pipeline for football
        """
        # Initialize football-specific components
        ner_model = FootballNERModel()
        entity_linker = EntityLinker(domain="football")
        
        return cls(
            ner_model=ner_model,
            entity_linker=entity_linker,
            domain="football",
            confidence_threshold=0.6
        )
