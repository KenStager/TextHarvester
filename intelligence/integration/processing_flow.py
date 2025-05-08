"""
Processing Flow Coordination.

This module provides the overall processing flow coordination for the
Content Intelligence Platform, orchestrating the different processing pipelines.
"""

import logging
import sys
import os
from typing import Dict, List, Optional, Union, Any, Set
from pathlib import Path

# Add the project root to path to ensure imports work properly
sys.path.append(str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


class ProcessingFlow:
    """
    Coordinator for end-to-end content processing.
    
    This class orchestrates the flow of content through the different
    processing pipelines of the Content Intelligence Platform.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the processing flow coordinator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.pipelines = {}
        
        # Initialize required pipelines
        self._initialize_pipelines()
    
    def _initialize_pipelines(self):
        """Initialize the required processing pipelines."""
        try:
            # Import the pipelines
            from intelligence.classification.pipeline import ClassificationPipeline
            from intelligence.entities.pipeline import EntityExtractionPipeline
            
            # Initialize classification pipeline
            if 'classification' not in self.pipelines:
                logger.info("Initializing classification pipeline")
                self.pipelines['classification'] = ClassificationPipeline.create_football_pipeline()
            
            # Initialize entity extraction pipeline
            if 'entities' not in self.pipelines:
                logger.info("Initializing entity extraction pipeline")
                self.pipelines['entities'] = EntityExtractionPipeline.create_football_pipeline()
            
            # Additional pipelines would be initialized here as they're implemented:
            # - Temporal analysis
            # - Content enrichment
            # - Knowledge management
            
            logger.info(f"Initialized {len(self.pipelines)} processing pipelines")
            
        except ImportError as e:
            logger.error(f"Failed to initialize pipelines: {e}")
    
    def process_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process content through the intelligence pipelines.
        
        This method implements the complete processing workflow for a single
        content item, running it through all the necessary pipelines.
        
        Args:
            content: Dictionary with content data
            
        Returns:
            Dictionary with processing results
        """
        content_text = content.get('extracted_text', '')
        content_id = content.get('id')
        
        if not content_text:
            return {
                'content_id': content_id,
                'error': 'No text content provided'
            }
        
        # Start with an empty results dictionary
        results = {
            'content_id': content_id,
            'processing_metadata': {
                'word_count': len(content_text.split()),
                'char_count': len(content_text)
            }
        }
        
        try:
            # Step 1: Topic Classification
            if 'classification' in self.pipelines:
                logger.info(f"Classifying content {content_id}")
                
                classification_input = {
                    'text': content_text,
                    'content_id': content_id,
                    'metadata': content
                }
                
                classification_result = self.pipelines['classification'].process(classification_input)
                results['classification'] = classification_result.to_dict()
                
                # Determine if we should continue processing based on relevance
                is_relevant = classification_result.is_relevant
                
                # If not relevant, we can skip further processing
                if not is_relevant and not self.config.get('process_all', False):
                    logger.info(f"Content {content_id} not relevant, skipping further processing")
                    return results
            else:
                # If no classification pipeline, assume content is relevant
                is_relevant = True
            
            # Step 2: Entity Extraction
            if 'entities' in self.pipelines:
                logger.info(f"Extracting entities from content {content_id}")
                
                entity_input = {
                    'text': content_text,
                    'content_id': content_id,
                    'metadata': content
                }
                
                entity_result = self.pipelines['entities'].process(entity_input)
                results['entities'] = entity_result.to_dict()
            
            # Step 3: Temporal Analysis (if implemented)
            if 'temporal' in self.pipelines:
                logger.info(f"Analyzing temporal information in content {content_id}")
                
                temporal_input = {
                    'text': content_text,
                    'content_id': content_id,
                    'metadata': content,
                    'entities': results.get('entities', {})
                }
                
                temporal_result = self.pipelines['temporal'].process(temporal_input)
                results['temporal'] = temporal_result.to_dict()
            
            # Step 4: Content Enrichment (if implemented)
            if 'enrichment' in self.pipelines:
                logger.info(f"Enriching content {content_id}")
                
                enrichment_input = {
                    'text': content_text,
                    'content_id': content_id,
                    'metadata': content,
                    'classification': results.get('classification', {}),
                    'entities': results.get('entities', {}),
                    'temporal': results.get('temporal', {})
                }
                
                enrichment_result = self.pipelines['enrichment'].process(enrichment_input)
                results['enrichment'] = enrichment_result.to_dict()
            
            # Save results to database
            self._save_results(content_id, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing content {content_id}: {str(e)}")
            results['error'] = str(e)
            return results
    
    def _save_results(self, content_id: int, results: Dict[str, Any]) -> None:
        """
        Save processing results to the database.
        
        Args:
            content_id: ID of the content
            results: Processing results to save
        """
        if not content_id:
            return
            
        try:
            # Import database models
            from db.models.content_intelligence import (
                ContentClassification, 
                EntityMention
            )
            
            logger.info(f"Saving processing results for content {content_id}")
            
            # In a real implementation, this would save the results to the database
            # For now, we'll just log that we would save the results
            
            logger.info(f"Would save results for content {content_id} with "
                         f"{len(results.get('classification', {}))} classification entries, "
                         f"{len(results.get('entities', {}).get('entities', []))} entities")
                         
        except ImportError as e:
            logger.warning(f"Could not save results to database: {e}")
    
    def process_by_domain(self, domain: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process all scraped content for a specific domain.
        
        This method is used to process content for a specific domain,
        such as football, that may have been scraped previously.
        
        Args:
            domain: Domain to process (e.g., 'football')
            options: Processing options
            
        Returns:
            Dictionary with processing results
        """
        from intelligence.integration.batch_processor import BatchProcessor
        
        # Create a batch processor
        processor = BatchProcessor()
        
        # Configure domain-specific options
        domain_options = {
            'domain': domain,
            'min_word_count': 100
        }
        
        # Merge with provided options
        if options:
            domain_options.update(options)
        
        # Process the backlog
        result = processor.process_backlog(domain_options)
        
        return result
