"""
Integration Hooks for TextHarvester Scraper.

This module provides integration hooks between the TextHarvester web scraper
and the Content Intelligence Platform components, allowing for automatic
processing of scraped content.
"""

import logging
import sys
import os
from typing import Dict, List, Optional, Union, Any, Set, Callable
import threading
import time
from pathlib import Path

# Add the project root to path to ensure imports work properly
sys.path.append(str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

# Configuration for intelligence processing
DEFAULT_CONFIG = {
    'enabled': True,
    'process_in_realtime': False,
    'batch_size': 50,
    'min_word_count': 100,
    'max_queue_size': 1000,
    'processing_threads': 2,
    'domains': {
        'football': {
            'enabled': True,
            'priority': 'high'
        }
    }
}

class IntegrationManager:
    """
    Manager for integrating scraped content with the Content Intelligence Platform.
    
    This class provides hooks that can be called from the scraper to process
    newly scraped content through the intelligence pipelines.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    @classmethod
    def get_instance(cls):
        """Singleton pattern to ensure only one integration manager exists."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def __init__(self):
        """Initialize the integration manager."""
        self.config = DEFAULT_CONFIG.copy()
        self.processing_queue = []
        self.queue_lock = threading.RLock()
        self.is_processing = False
        self.stop_requested = False
        self.processing_thread = None
        self.callbacks = {
            'on_content_processed': [],
            'on_batch_completed': [],
            'on_error': []
        }
        
        # Try to import intelligence components
        try:
            from intelligence.classification.pipeline import ClassificationPipeline
            from intelligence.entities.pipeline import EntityExtractionPipeline
            
            # Initialize pipelines - defer actual creation until needed
            self.classification_pipeline = None
            self.entity_pipeline = None
            
            logger.info("Intelligence components found and ready for integration")
            self.intelligence_available = True
        except ImportError as e:
            logger.warning(f"Intelligence components not available: {e}")
            self.intelligence_available = False
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """
        Register a callback function for a specific event type.
        
        Args:
            event_type: Type of event ('on_content_processed', 'on_batch_completed', 'on_error')
            callback: Function to call when the event occurs
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the integration manager.
        
        Args:
            config: Configuration dictionary to merge with defaults
        """
        self.config.update(config)
        logger.info(f"Integration manager configured: {self.config}")
    
    def _trigger_callbacks(self, event_type: str, data: Any) -> None:
        """
        Trigger all callbacks for a specific event type.
        
        Args:
            event_type: Type of event
            data: Data to pass to the callbacks
        """
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in {event_type} callback: {str(e)}")
    
    def post_content_hook(self, content: Dict[str, Any]) -> None:
        """
        Hook called after content is scraped.
        
        This hook is intended to be called from the scraper's content extraction
        process to integrate with the intelligence platform.
        
        Args:
            content: Dict containing scraped content information
        """
        if not self.config['enabled'] or not self.intelligence_available:
            return
        
        # Check if content meets minimum requirements
        if not self._should_process_content(content):
            return
        
        # If real-time processing is enabled, process immediately
        if self.config['process_in_realtime']:
            self._process_content(content)
        else:
            # Otherwise, add to queue for batch processing
            with self.queue_lock:
                # Check if queue is full
                if len(self.processing_queue) >= self.config['max_queue_size']:
                    logger.warning("Processing queue is full, dropping oldest content")
                    self.processing_queue.pop(0)  # Remove oldest item
                
                self.processing_queue.append(content)
                
                # Start processing thread if not already running
                if not self.is_processing and not self.processing_thread:
                    self._start_processing_thread()
    
    def post_job_hook(self, job_data: Dict[str, Any]) -> None:
        """
        Hook called after a scraping job completes.
        
        Args:
            job_data: Dict containing job information
        """
        if not self.config['enabled'] or not self.intelligence_available:
            return
        
        logger.info(f"Scraping job {job_data.get('id')} completed, scheduling batch processing")
        
        # Process any remaining items in the queue
        self._process_batch(force=True)
    
    def _should_process_content(self, content: Dict[str, Any]) -> bool:
        """
        Determine if content should be processed based on configuration rules.
        
        Args:
            content: Dict containing scraped content information
            
        Returns:
            True if content should be processed, False otherwise
        """
        # Skip content with too little text
        word_count = content.get('word_count', 0)
        if word_count < self.config['min_word_count']:
            return False
        
        # Additional filters could be added here
        # - Content type filtering
        # - Domain filtering
        # - Language detection
        
        return True
    
    def _process_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single content item through the intelligence pipelines.
        
        Args:
            content: Dict containing scraped content information
            
        Returns:
            Dict with processing results
        """
        if not self.intelligence_available:
            return {'error': 'Intelligence components not available'}
        
        try:
            # Ensure pipelines are initialized
            self._initialize_pipelines_if_needed()
            
            content_text = content.get('extracted_text', '')
            content_id = content.get('id')
            
            # Run classification pipeline
            classification_result = self.classification_pipeline.process({
                'text': content_text,
                'content_id': content_id,
                'metadata': content
            })
            
            results = {
                'content_id': content_id,
                'classification': classification_result.to_dict()
            }
            
            # If content is relevant to our domains, extract entities
            if classification_result.is_relevant:
                entity_result = self.entity_pipeline.process({
                    'text': content_text,
                    'content_id': content_id,
                    'metadata': content
                })
                
                results['entities'] = entity_result.to_dict()
            
            # Save results to database if content_id is provided
            if content_id:
                self._save_results_to_database(content_id, results)
            
            # Trigger callbacks
            self._trigger_callbacks('on_content_processed', results)
            
            return results
            
        except Exception as e:
            error_info = {
                'content_id': content.get('id'),
                'error': str(e),
                'content': content  # Include original content for reference
            }
            logger.error(f"Error processing content {content.get('id')}: {str(e)}")
            self._trigger_callbacks('on_error', error_info)
            return {'error': str(e)}
    
    def _save_results_to_database(self, content_id: int, results: Dict[str, Any]) -> None:
        """
        Save processing results to the database.
        
        This method links the scraped content to the intelligence processing 
        results in the database.
        
        Args:
            content_id: ID of the scraped content
            results: Dict with processing results
        """
        try:
            # Import database models
            from db.models.content_intelligence import (
                ContentClassification, 
                ContentEntity, 
                EntityMention
            )
            
            # In a real implementation, this would save the results to the database
            # For now, just log that we would save results
            logger.info(f"Would save processing results for content {content_id} to database")
            
        except ImportError as e:
            logger.warning(f"Could not save results to database: {e}")
    
    def _process_batch(self, force: bool = False) -> None:
        """
        Process a batch of content from the queue.
        
        Args:
            force: If True, process all remaining items regardless of batch size
        """
        batch = []
        
        # Get a batch of items from the queue with proper locking
        with self.queue_lock:
            if force:
                # Take all remaining items
                batch = self.processing_queue[:]
                self.processing_queue = []
            else:
                # Take up to batch_size items
                batch_size = min(len(self.processing_queue), self.config['batch_size'])
                if batch_size > 0:
                    batch = self.processing_queue[:batch_size]
                    self.processing_queue = self.processing_queue[batch_size:]
        
        if not batch:
            return
            
        logger.info(f"Processing batch of {len(batch)} content items")
        
        # Process each item in the batch
        results = []
        for content in batch:
            if self.stop_requested:
                break
                
            result = self._process_content(content)
            results.append(result)
        
        # Trigger batch completed callbacks
        self._trigger_callbacks('on_batch_completed', results)
    
    def _start_processing_thread(self) -> None:
        """Start the background processing thread."""
        if self.is_processing:
            return
            
        self.stop_requested = False
        self.is_processing = True
        
        def processing_loop():
            logger.info("Starting batch processing thread")
            
            while not self.stop_requested:
                # Check if we have enough items to process a batch
                with self.queue_lock:
                    queue_size = len(self.processing_queue)
                
                if queue_size >= self.config['batch_size']:
                    self._process_batch()
                elif queue_size > 0:
                    # If we have some items but not a full batch, wait a bit
                    # to see if more items arrive before processing
                    time.sleep(5)
                    
                    # Process anyway if items are still waiting
                    with self.queue_lock:
                        if len(self.processing_queue) > 0:
                            self._process_batch()
                else:
                    # No items to process, wait a bit
                    time.sleep(1)
            
            # Process any remaining items before stopping
            self._process_batch(force=True)
            
            logger.info("Batch processing thread stopped")
            self.is_processing = False
            self.processing_thread = None
        
        # Create and start the thread
        self.processing_thread = threading.Thread(target=processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_processing(self) -> None:
        """Stop the background processing thread."""
        if self.is_processing:
            logger.info("Requesting batch processing thread to stop")
            self.stop_requested = True
    
    def _initialize_pipelines_if_needed(self) -> None:
        """Initialize the intelligence pipelines if they haven't been created yet."""
        if not self.intelligence_available:
            return
            
        # Import here to avoid circular imports
        from intelligence.classification.pipeline import ClassificationPipeline
        from intelligence.entities.pipeline import EntityExtractionPipeline
        
        # Initialize classification pipeline if needed
        if self.classification_pipeline is None:
            logger.info("Initializing classification pipeline")
            self.classification_pipeline = ClassificationPipeline.create_football_pipeline()
        
        # Initialize entity pipeline if needed
        if self.entity_pipeline is None:
            logger.info("Initializing entity extraction pipeline")
            self.entity_pipeline = EntityExtractionPipeline.create_football_pipeline()
    
    def process_content_by_id(self, content_id: int) -> Dict[str, Any]:
        """
        Process a specific content item by ID.
        
        This method can be called to process a content item on demand,
        outside of the normal post-processing flow.
        
        Args:
            content_id: ID of the scraped content to process
            
        Returns:
            Dict with processing results
        """
        if not self.intelligence_available:
            return {'error': 'Intelligence components not available'}
        
        try:
            # Import database model
            from models import ScrapedContent, ContentMetadata
            
            # Query the content from the database
            # We'd normally use a Flask app context here, but for simplicity
            # we'll assume the caller has handled that
            content = ScrapedContent.query.get(content_id)
            if not content:
                return {'error': f'Content with ID {content_id} not found'}
            
            # Get metadata if available
            metadata = None
            if hasattr(content, 'content_metadata') and content.content_metadata:
                metadata = content.content_metadata
            
            # Create a content dict for processing
            content_dict = {
                'id': content.id,
                'url': content.url,
                'title': content.title,
                'extracted_text': content.extracted_text,
                'crawl_depth': content.crawl_depth,
                'job_id': content.job_id,
                'created_at': content.created_at.isoformat() if content.created_at else None,
                'word_count': metadata.word_count if metadata else None,
                'language': metadata.language if metadata else None
            }
            
            # Process the content
            return self._process_content(content_dict)
            
        except Exception as e:
            logger.error(f"Error processing content {content_id}: {str(e)}")
            return {'error': str(e)}


def process_content_batch(content_ids: List[int], options: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Process a batch of content items by ID.
    
    This function is a convenience wrapper for processing multiple content
    items with a single call.
    
    Args:
        content_ids: List of content IDs to process
        options: Optional processing options
        
    Returns:
        Dict with processing results
    """
    manager = IntegrationManager.get_instance()
    
    results = {
        'total': len(content_ids),
        'successful': 0,
        'failed': 0,
        'items': []
    }
    
    for content_id in content_ids:
        try:
            result = manager.process_content_by_id(content_id)
            results['items'].append(result)
            
            if 'error' not in result:
                results['successful'] += 1
            else:
                results['failed'] += 1
                
        except Exception as e:
            results['failed'] += 1
            results['items'].append({
                'content_id': content_id,
                'error': str(e)
            })
    
    return results


def setup_scraper_integration(app=None):
    """
    Set up the integration with the scraper.
    
    This function should be called from the scraper's initialization code
    to set up the integration hooks.
    
    Args:
        app: Optional Flask app instance for database context
    """
    # Initialize the integration manager
    manager = IntegrationManager.get_instance()
    
    try:
        # Import the scraper's content extractor module
        from scraper.content_extractor import extract_content
        from models import ScrapedContent
        
        # Store the original extract_content function
        original_extract_content = extract_content
        
        # Create a wrapped function that calls our hook
        def wrapped_extract_content(url, content, *args, **kwargs):
            # Call the original function
            title, extracted_text, raw_html = original_extract_content(url, content, *args, **kwargs)
            
            # Call our post-processing hook if we have text content
            if extracted_text:
                # We don't have the full content object here, just the extracted data
                # The actual hook integration would be better in the fetch_url method
                # of the WebCrawler class, but this is a non-invasive approach
                content_dict = {
                    'url': url,
                    'title': title,
                    'extracted_text': extracted_text,
                    'word_count': len(extracted_text.split()) if extracted_text else 0
                }
                
                # Call the hook
                try:
                    manager.post_content_hook(content_dict)
                except Exception as e:
                    logger.error(f"Error in post-content hook: {str(e)}")
            
            # Return the original result
            return title, extracted_text, raw_html
        
        # Replace the original function with our wrapped version
        # NOTE: This approach is somewhat hacky and would be better implemented
        # with a proper plugin system in the scraper. For a production system,
        # consider modifying the scraper code directly to call our hooks.
        import scraper.content_extractor
        scraper.content_extractor.extract_content = wrapped_extract_content
        
        logger.info("Scraper integration set up successfully")
        return True
        
    except ImportError as e:
        logger.error(f"Could not set up scraper integration: {e}")
        return False


def register_api_routes(app):
    """
    Register API routes for the intelligence integration.
    
    Args:
        app: Flask app instance
    """
    from flask import Blueprint, request, jsonify
    
    # Create a blueprint for the intelligence API
    intelligence_bp = Blueprint('intelligence', __name__, url_prefix='/intelligence')
    
    @intelligence_bp.route('/process/<int:content_id>', methods=['POST'])
    def process_content(content_id):
        """Process a specific content item."""
        manager = IntegrationManager.get_instance()
        result = manager.process_content_by_id(content_id)
        return jsonify(result)
    
    @intelligence_bp.route('/process_batch', methods=['POST'])
    def process_batch():
        """Process a batch of content items."""
        data = request.get_json()
        content_ids = data.get('content_ids', [])
        options = data.get('options', {})
        
        if not content_ids:
            return jsonify({'error': 'No content IDs provided'}), 400
        
        results = process_content_batch(content_ids, options)
        return jsonify(results)
    
    @intelligence_bp.route('/configure', methods=['POST'])
    def configure():
        """Configure the integration manager."""
        data = request.get_json()
        manager = IntegrationManager.get_instance()
        manager.configure(data)
        return jsonify({'status': 'success'})
    
    # Register the blueprint with the app
    app.register_blueprint(intelligence_bp)
    
    logger.info("Registered intelligence API routes")
