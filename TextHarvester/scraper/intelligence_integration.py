"""
Intelligence Integration Module

This module provides integration between the TextHarvester scraper and the intelligence components,
allowing for automatic content classification and entity extraction.
"""

import logging
import sys
import os
import time
from typing import Optional, Dict, List, Any, Tuple

# Configure logging
logger = logging.getLogger(__name__)

# Add intelligence module to the Python path if it's not already there
parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Try to import intelligence components
try:
    from intelligence.classification.pipeline import (
        ClassificationPipeline, ClassificationInput, ClassificationOutput
    )
    from intelligence.entities.pipeline import (
        EntityExtractionPipeline, EntityExtractionInput, EntityExtractionOutput
    )
    from intelligence.base_pipeline import WorkItem
    INTELLIGENCE_AVAILABLE = True
    logger.info("Intelligence modules successfully imported")
except ImportError as e:
    INTELLIGENCE_AVAILABLE = False
    logger.warning(f"Intelligence modules not available, falling back to basic processing: {e}")


class IntelligenceProcessor:
    """
    Handles the integration between the scraper and intelligence components,
    processing content through classification and entity extraction pipelines.
    """
    
    def __init__(self, domain: str = "football", enable_classification: bool = True, 
                 enable_entity_extraction: bool = True):
        """
        Initialize the intelligence processor.
        
        Args:
            domain: Domain for intelligence processing (e.g., "football")
            enable_classification: Whether to enable content classification
            enable_entity_extraction: Whether to enable entity extraction
        """
        self.domain = domain
        self.enable_classification = enable_classification and INTELLIGENCE_AVAILABLE
        self.enable_entity_extraction = enable_entity_extraction and INTELLIGENCE_AVAILABLE
        
        # Initialize pipelines on first use to avoid startup delays
        self._classification_pipeline = None
        self._entity_extraction_pipeline = None
        
        if not INTELLIGENCE_AVAILABLE:
            logger.warning("Intelligence components not available. Make sure the intelligence module is installed.")
    
    @property
    def classification_pipeline(self) -> Optional[object]:
        """Lazy-load the classification pipeline."""
        if self.enable_classification and not self._classification_pipeline:
            try:
                if self.domain == "football":
                    self._classification_pipeline = ClassificationPipeline.create_football_pipeline()
                else:
                    self._classification_pipeline = ClassificationPipeline(domain_name=self.domain)
                logger.info(f"Initialized classification pipeline for domain: {self.domain}")
            except Exception as e:
                logger.error(f"Failed to initialize classification pipeline: {str(e)}")
                self.enable_classification = False
        return self._classification_pipeline
    
    @property
    def entity_extraction_pipeline(self) -> Optional[object]:
        """Lazy-load the entity extraction pipeline."""
        if self.enable_entity_extraction and not self._entity_extraction_pipeline:
            try:
                if self.domain == "football":
                    self._entity_extraction_pipeline = EntityExtractionPipeline.create_football_pipeline()
                else:
                    self._entity_extraction_pipeline = EntityExtractionPipeline(domain=self.domain)
                logger.info(f"Initialized entity extraction pipeline for domain: {self.domain}")
            except Exception as e:
                logger.error(f"Failed to initialize entity extraction pipeline: {str(e)}")
                self.enable_entity_extraction = False
        return self._entity_extraction_pipeline
    
    def process_content(self, content) -> Dict[str, Any]:
        """
        Process content through intelligence pipelines.
        
        Args:
            content: ScrapedContent object to process
            
        Returns:
            Dictionary with processing results
        """
        results = {
            "classification": None,
            "entities": None,
            "processing_time": 0
        }
        
        # Check if intelligence components are available
        if not INTELLIGENCE_AVAILABLE:
            logger.warning("Intelligence processing skipped - modules not available")
            return results
            
        start_time = time.time()
        
        # Process through classification pipeline if enabled
        if self.enable_classification and self.classification_pipeline:
            try:
                classification_result = self.classify_content(content)
                if classification_result:
                    results["classification"] = classification_result
                    # Save classification to database
                    self.save_classification(content.id, classification_result)
            except Exception as e:
                logger.error(f"Error classifying content {content.id}: {str(e)}")
        
        # Process through entity extraction pipeline if enabled
        if self.enable_entity_extraction and self.entity_extraction_pipeline:
            try:
                entity_result = self.extract_entities(content)
                if entity_result:
                    results["entities"] = entity_result
                    # Save entities to database
                    self.save_entities(content.id, entity_result)
            except Exception as e:
                logger.error(f"Error extracting entities from content {content.id}: {str(e)}")
        
        # Calculate total processing time
        results["processing_time"] = time.time() - start_time
        
        return results
    
    def classify_content(self, content) -> Optional[Any]:
        """
        Classify content using the classification pipeline.
        
        Args:
            content: ScrapedContent object to classify
            
        Returns:
            Classification output or None if classification failed
        """
        if not self.classification_pipeline:
            return None
            
        # Prepare input data
        input_data = ClassificationInput(
            text=content.extracted_text,
            content_id=content.id,
            metadata={
                "url": content.url,
                "title": content.title,
                "crawl_depth": content.crawl_depth
            }
        )
        
        try:
            # Process through classification pipeline
            result = self.classification_pipeline.process(input_data)
            logger.info(f"Classified content {content.id} as '{result.primary_topic}' with confidence {result.primary_topic_confidence:.4f}")
            return result
        except Exception as e:
            logger.error(f"Classification error for content {content.id}: {str(e)}")
            return None
    
    def extract_entities(self, content) -> Optional[Any]:
        """
        Extract entities from content using the entity extraction pipeline.
        
        Args:
            content: ScrapedContent object to extract entities from
            
        Returns:
            Entity extraction output or None if extraction failed
        """
        if not self.entity_extraction_pipeline:
            return None
            
        # Prepare input data
        input_data = EntityExtractionInput(
            text=content.extracted_text,
            content_id=content.id,
            metadata={
                "url": content.url,
                "title": content.title,
                "crawl_depth": content.crawl_depth
            }
        )
        
        try:
            # Process through entity extraction pipeline
            result = self.entity_extraction_pipeline.process(input_data)
            logger.info(f"Extracted {len(result.entities)} entities from content {content.id}")
            return result
        except Exception as e:
            logger.error(f"Entity extraction error for content {content.id}: {str(e)}")
            return None
    
    def save_classification(self, content_id: int, result) -> bool:
        """
        Save classification results to database.
        
        Args:
            content_id: ID of content being classified
            result: Classification output
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try import with correct module path
            from TextHarvester.app import db
            from TextHarvester.models_update import ContentClassification
        except ImportError:
            try:
                # Fallback for direct imports
                from app import db
                from models_update import ContentClassification
            except ImportError:
                logger.error("Could not import database modules - database operations disabled")
                return False
        
        try:
            # Create classification record
            classification = ContentClassification(
                content_id=content_id,
                is_relevant=result.is_relevant,
                confidence=result.confidence,
                primary_topic=result.primary_topic,
                primary_topic_id=result.primary_topic_id,
                primary_topic_confidence=result.primary_topic_confidence,
                subtopics=result.subtopics,
                processing_time=result.processing_time
            )
            
            # Add to database
            db.session.add(classification)
            db.session.commit()
            logger.debug(f"Saved classification for content {content_id}")
            return True
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error saving classification for content {content_id}: {str(e)}")
            return False
    
    def save_entities(self, content_id: int, result) -> bool:
        """
        Save extracted entities to database.
        
        Args:
            content_id: ID of content entities were extracted from
            result: Entity extraction output
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try import with correct module path
            from TextHarvester.app import db
            from TextHarvester.models_update import ContentEntity
        except ImportError:
            try:
                # Fallback for direct imports
                from app import db
                from models_update import ContentEntity
            except ImportError:
                logger.error("Could not import database modules - database operations disabled")
                return False
        
        try:
            entity_records = []
            
            # Save linked entities first
            for entity in result.linked_entities:
                entity_record = ContentEntity(
                    content_id=content_id,
                    entity_type=entity.entity.type if hasattr(entity, 'entity') else entity.mention.label,
                    entity_text=entity.mention.text,
                    start_char=entity.mention.start_char,
                    end_char=entity.mention.end_char,
                    confidence=entity.confidence,
                    entity_id=entity.entity.id if hasattr(entity, 'entity') else None,
                    metadata={
                        "linked": True,
                        "entity_name": entity.entity.name if hasattr(entity, 'entity') else None,
                        "link_score": entity.link_score if hasattr(entity, 'link_score') else None
                    }
                )
                entity_records.append(entity_record)
            
            # Save unlinked entities
            for entity in result.unlinked_entities:
                entity_record = ContentEntity(
                    content_id=content_id,
                    entity_type=entity.label,
                    entity_text=entity.text,
                    start_char=entity.start_char,
                    end_char=entity.end_char,
                    confidence=entity.confidence,
                    entity_id=None,
                    metadata={"linked": False}
                )
                entity_records.append(entity_record)
            
            # Add all records to database
            db.session.add_all(entity_records)
            db.session.commit()
            logger.debug(f"Saved {len(entity_records)} entities for content {content_id}")
            return True
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error saving entities for content {content_id}: {str(e)}")
            return False
