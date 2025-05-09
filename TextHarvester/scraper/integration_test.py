"""
Test script for intelligence integration.

This script tests the integration between the TextHarvester scraper and the intelligence components.
It can be used to verify that the integration is working correctly without running a full crawl job.
"""

import os
import sys
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path so we can import app
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    # Import the app and models
    from app import app, db
    from models import ScrapedContent, ContentMetadata
    from models_update import ContentClassification, ContentEntity
    from scraper.intelligence_integration import IntelligenceProcessor, INTELLIGENCE_AVAILABLE
    
    def test_intelligence_integration():
        """
        Test the integration between the TextHarvester scraper and the intelligence components.
        This function creates a test content object and processes it through the intelligence pipelines.
        """
        logger.info("Starting intelligence integration test")
        
        if not INTELLIGENCE_AVAILABLE:
            logger.error("Intelligence modules not available! Test cannot proceed.")
            return
        
        with app.app_context():
            # Create a test content object
            test_content = ScrapedContent(
                job_id=1,  # Use a dummy job ID
                url="https://test.example.com/football-article",
                title="Premier League: Manchester City vs Liverpool Preview",
                raw_html="<html><body><h1>Premier League: Manchester City vs Liverpool Preview</h1><p>A crucial match between Manchester City and Liverpool will take place on Saturday. Pep Guardiola's team will face Jurgen Klopp's side in what promises to be an exciting game. Kevin De Bruyne is expected to return from injury.</p></body></html>",
                extracted_text="Premier League: Manchester City vs Liverpool Preview\n\nA crucial match between Manchester City and Liverpool will take place on Saturday. Pep Guardiola's team will face Jurgen Klopp's side in what promises to be an exciting game. Kevin De Bruyne is expected to return from injury.",
                crawl_depth=0,
                processing_time=100  # 100ms
            )
            
            # Add metadata
            test_metadata = ContentMetadata(
                word_count=len(test_content.extracted_text.split()),
                char_count=len(test_content.extracted_text),
                language="en",
                content_type="text/html",
                extra_data={}
            )
            
            # Link metadata to content
            test_content.content_metadata = test_metadata
            
            # Add to database temporarily for testing
            db.session.add(test_content)
            db.session.flush()  # This assigns an ID without committing
            
            try:
                # Create intelligence processor with both features enabled
                processor = IntelligenceProcessor(
                    domain="football",
                    enable_classification=True,
                    enable_entity_extraction=True
                )
                
                # Process the content
                logger.info(f"Processing test content ID: {test_content.id}")
                results = processor.process_content(test_content)
                
                # Log results
                logger.info(f"Intelligence processing completed in {results['processing_time']:.2f}s")
                
                if results.get('classification'):
                    classification = results['classification']
                    logger.info(f"Classification results:")
                    logger.info(f"  Primary topic: {classification.primary_topic}")
                    logger.info(f"  Confidence: {classification.primary_topic_confidence:.4f}")
                    logger.info(f"  Is relevant: {classification.is_relevant}")
                    
                    # Check if classification was saved to database
                    db_classification = ContentClassification.query.filter_by(content_id=test_content.id).first()
                    if db_classification:
                        logger.info(f"Classification saved to database successfully!")
                    else:
                        logger.warning("Classification not found in database!")
                else:
                    logger.warning("No classification results returned!")
                
                if results.get('entities'):
                    entities = results['entities']
                    logger.info(f"Entity extraction results:")
                    logger.info(f"  Total entities: {len(entities.entities)}")
                    logger.info(f"  Linked entities: {len(entities.linked_entities)}")
                    logger.info(f"  Unlinked entities: {len(entities.unlinked_entities)}")
                    
                    # List top 5 entities by type
                    if entities.entities:
                        logger.info("  Top entities:")
                        for i, entity in enumerate(entities.entities[:5]):
                            if hasattr(entity, 'label'):
                                # Unlinked entity
                                logger.info(f"    {i+1}. {entity.label}: {entity.text} (conf: {entity.confidence:.2f})")
                            else:
                                # Linked entity
                                logger.info(f"    {i+1}. {entity.mention.label}: {entity.mention.text} (conf: {entity.confidence:.2f})")
                    
                    # Check if entities were saved to database
                    db_entities = ContentEntity.query.filter_by(content_id=test_content.id).all()
                    if db_entities:
                        logger.info(f"Entities saved to database successfully! Count: {len(db_entities)}")
                    else:
                        logger.warning("Entities not found in database!")
                else:
                    logger.warning("No entity extraction results returned!")
                
                # Test successful
                logger.info("Intelligence integration test completed successfully!")
                return True
            
            except Exception as e:
                logger.error(f"Error during intelligence integration test: {str(e)}")
                return False
            
            finally:
                # Clean up test data - rollback without committing
                db.session.rollback()
                logger.info("Test cleanup completed")
    
    if __name__ == "__main__":
        success = test_intelligence_integration()
        if success:
            print("Intelligence integration test PASSED")
            sys.exit(0)
        else:
            print("Intelligence integration test FAILED")
            sys.exit(1)

except ImportError as e:
    logger.error(f"Failed to import required modules: {str(e)}")
    logger.error("Make sure you've installed all dependencies and are running from the correct directory.")
    sys.exit(1)
