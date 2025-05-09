"""
Classification Pipeline Implementation.

This module implements the complete end-to-end classification pipeline,
coordinating the various classification components and managing the
overall classification process.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass

from intelligence.base_pipeline import BasePipeline
from intelligence.utils.text_processing import normalize_text, extract_keywords
from intelligence.classification.fast_filter import FastFilter
from intelligence.classification.classifiers import (
    TopicClassifier, HierarchicalClassifier, 
    ClassificationResult, FootballClassifier
)
from intelligence.classification.topic_taxonomy import TopicTaxonomy

logger = logging.getLogger(__name__)


@dataclass
class ClassificationInput:
    """Data class for classification pipeline input."""
    text: str
    metadata: Dict[str, Any] = None
    content_id: Optional[int] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ClassificationOutput:
    """Data class for classification pipeline output."""
    content_id: Optional[int]
    is_relevant: bool
    confidence: float
    primary_topic: Optional[str] = None
    primary_topic_id: Optional[str] = None
    primary_topic_confidence: float = 0.0
    subtopics: List[Dict[str, Any]] = None
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.subtopics is None:
            self.subtopics = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "content_id": self.content_id,
            "is_relevant": self.is_relevant,
            "confidence": self.confidence,
            "primary_topic": self.primary_topic,
            "primary_topic_id": self.primary_topic_id,
            "primary_topic_confidence": self.primary_topic_confidence,
            "subtopics": self.subtopics,
            "processing_time": self.processing_time
        }


class ClassificationPipeline(BasePipeline):
    """
    End-to-end classification pipeline.
    
    This pipeline coordinates the process of classifying content, using
    fast filtering for initial screening and hierarchical classification
    for detailed topic categorization.
    """
    
    def __init__(self, taxonomy: Optional[TopicTaxonomy] = None,
                 fast_filter: Optional[FastFilter] = None,
                 hierarchical_classifier: Optional[HierarchicalClassifier] = None,
                 confidence_threshold: float = 0.5,
                 domain_name: str = "football"):
        """
        Initialize the classification pipeline.
        
        Args:
            taxonomy: Optional taxonomy to use for classification
            fast_filter: Optional fast filter for initial screening
            hierarchical_classifier: Optional hierarchical classifier for detailed classification
            confidence_threshold: Confidence threshold for relevance
            domain_name: Name of the domain (used for football-specific configuration)
        """
        super().__init__("classification")
        
        self.taxonomy = taxonomy
        self.fast_filter = fast_filter
        self.hierarchical_classifier = hierarchical_classifier
        self.confidence_threshold = confidence_threshold
        self.domain_name = domain_name
        
        # Initialize components if not provided
        if self.domain_name == "football" and not all([self.taxonomy, self.fast_filter, self.hierarchical_classifier]):
            self._initialize_football_components()
            
    def _initialize_football_components(self) -> None:
        """Initialize pipeline components specifically for football."""
        from intelligence.classification.taxonomies.football import get_premier_league_taxonomy
        
        logger.info("Initializing football-specific classification components")
        
        # Initialize taxonomy if not provided
        if not self.taxonomy:
            self.taxonomy = get_premier_league_taxonomy()
            
        # Initialize fast filter if not provided
        if not self.fast_filter:
            self.fast_filter = FastFilter.create_from_football_taxonomy()
            
        # Initialize hierarchical classifier if not provided
        if not self.hierarchical_classifier:
            self.hierarchical_classifier = FootballClassifier.create(
                confidence_threshold=self.confidence_threshold
            )
    
    def process(self, input_data: Union[ClassificationInput, Dict, str]) -> ClassificationOutput:
        """
        Process input through the classification pipeline.
        
        Args:
            input_data: Input data to process (ClassificationInput, Dict, or str)
            
        Returns:
            Classification results
        """
        # Convert input to ClassificationInput if needed
        if isinstance(input_data, str):
            input_data = ClassificationInput(text=input_data)
        elif isinstance(input_data, dict):
            input_data = ClassificationInput(**input_data)
            
        # Validate input
        if not input_data.text:
            logger.warning("Empty text provided to classification pipeline")
            return ClassificationOutput(
                content_id=input_data.content_id,
                is_relevant=False,
                confidence=1.0,  # High confidence that empty text is not relevant
                processing_time=0.0
            )
            
        # Start timing
        start_time = time.time()
        
        try:
            # Step 1: Fast filtering
            is_relevant, fast_confidence = self._apply_fast_filter(input_data.text)
            
            # If not potentially relevant, return quickly
            if not is_relevant:
                processing_time = time.time() - start_time
                logger.info(f"Fast filter determined content is not relevant (confidence: {fast_confidence:.4f})")
                
                return ClassificationOutput(
                    content_id=input_data.content_id,
                    is_relevant=False,
                    confidence=fast_confidence,
                    processing_time=processing_time
                )
                
            # Step 2: Hierarchical classification
            hierarchical_result = self._apply_hierarchical_classification(input_data.text)
            
            # Step 3: Process results
            output = self._process_classification_results(
                input_data, hierarchical_result, fast_confidence
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            output.processing_time = processing_time
            
            logger.info(f"Classified content as '{output.primary_topic}' with confidence {output.primary_topic_confidence:.4f}")
            return output
            
        except Exception as e:
            logger.error(f"Error in classification pipeline: {str(e)}", exc_info=True)
            processing_time = time.time() - start_time
            
            # Return a default output on error
            return ClassificationOutput(
                content_id=input_data.content_id,
                is_relevant=False,
                confidence=0.0,
                processing_time=processing_time
            )
    
    def _apply_fast_filter(self, text: str) -> Tuple[bool, float]:
        """
        Apply fast filtering to quickly determine if content is potentially relevant.
        
        Args:
            text: Text to filter
            
        Returns:
            Tuple of (is_relevant, confidence)
        """
        if not self.fast_filter:
            logger.warning("No fast filter available, assuming content is relevant")
            # Basic keyword matching as fallback
            basic_keywords = {
                "football": ["football", "soccer", "goal", "match", "team", "player", "league", "cup"],
                "technology": ["technology", "computer", "software", "hardware", "app", "digital", "tech"],
                "business": ["business", "company", "market", "finance", "stock", "economy", "investment"],
                "general": ["news", "report", "article", "story", "today", "update", "information"]
            }
            
            # Check if relevant to the domain
            domain_keywords = basic_keywords.get(self.domain_name.lower(), basic_keywords["general"])
            
            # Simple keyword counting
            text_lower = text.lower()
            matches = sum(1 for keyword in domain_keywords if keyword in text_lower)
            confidence = min(0.9, 0.3 + (matches * 0.1))  # Scale from 0.3 to 0.9 based on matches
            
            return True, confidence
            
        is_relevant, confidence = self.fast_filter.is_potentially_relevant(
            text, domain=self.domain_name
        )
        
        # Get matching keywords for logging
        matching_keywords = self.fast_filter.get_matching_keywords(
            text, domain=self.domain_name, max_keywords=5
        )
        
        if matching_keywords:
            keyword_str = ", ".join(matching_keywords)
            logger.debug(f"Fast filter found keywords: {keyword_str}")
            
        return is_relevant, confidence
    
    def _apply_hierarchical_classification(self, text: str) -> ClassificationResult:
        """
        Apply hierarchical classification to determine specific topics.
        
        Args:
            text: Text to classify
            
        Returns:
            Hierarchical classification result
        """
        if not self.hierarchical_classifier:
            logger.warning("No hierarchical classifier available, using default classification")
            
            # Use the default ClassificationResult factory method
            return ClassificationResult.create_default_for_domain(domain=self.domain_name, confidence=0.7)
            
        return self.hierarchical_classifier.predict(text, max_depth=3)
    
    def _process_classification_results(self, input_data: ClassificationInput,
                                       hierarchical_result: ClassificationResult,
                                       fast_confidence: float) -> ClassificationOutput:
        """
        Process classification results into a standardized output.
        
        Args:
            input_data: Original input data
            hierarchical_result: Results from hierarchical classification
            fast_confidence: Confidence from fast filtering
            
        Returns:
            Processed classification output
        """
        # Determine primary topic
        primary_topic = hierarchical_result.node_name
        primary_topic_id = hierarchical_result.node_id
        primary_confidence = hierarchical_result.confidence
        
        # Extract subtopics from children
        subtopics = []
        for child in hierarchical_result.children:
            subtopic = {
                "topic": child.node_name,
                "topic_id": child.node_id,
                "confidence": child.confidence
            }
            
            # Add child's children as subsubtopics if present
            if child.children:
                subtopic["subtopics"] = [
                    {
                        "topic": subchild.node_name,
                        "topic_id": subchild.node_id,
                        "confidence": subchild.confidence
                    }
                    for subchild in child.children
                ]
                
            subtopics.append(subtopic)
            
        # Sort subtopics by confidence
        subtopics.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Determine overall confidence
        # Combining fast filtering and hierarchical classification
        # with more weight to hierarchical results
        overall_confidence = (primary_confidence * 0.7) + (fast_confidence * 0.3)
        
        # Create output
        return ClassificationOutput(
            content_id=input_data.content_id,
            is_relevant=primary_confidence >= self.confidence_threshold,
            confidence=overall_confidence,
            primary_topic=primary_topic,
            primary_topic_id=primary_topic_id,
            primary_topic_confidence=primary_confidence,
            subtopics=subtopics
        )
    
    def save_classification_to_database(self, output: ClassificationOutput) -> bool:
        """
        Save classification results to the database.
        
        Args:
            output: Classification output to save
            
        Returns:
            True if successful, False otherwise
        """
        # This would normally interact with the database models
        # For now, we'll just log the action
        if output.content_id:
            logger.info(f"Would save classification for content ID {output.content_id} to database")
            return True
        else:
            logger.warning("Cannot save classification with no content ID")
            return False
            
    @staticmethod
    def create_football_pipeline() -> 'ClassificationPipeline':
        """
        Create a pre-configured classification pipeline for football content.
        
        Returns:
            Configured ClassificationPipeline for football
        """
        return ClassificationPipeline(domain_name="football")
