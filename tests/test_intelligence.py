"""
Test script for the intelligence features of TextHarvester.

This script tests the classification and entity extraction pipelines
to verify they are working correctly.
"""

import os
import sys
import logging
import json
from datetime import datetime

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_intelligence")

# Sample texts for testing
SAMPLE_TEXTS = {
    "football": """
        Manchester United secured a dramatic 2-1 victory over Manchester City in the derby match at Old Trafford. 
        Marcus Rashford scored the winning goal in the 82nd minute, capitalizing on a defensive error by City's backline. 
        The result puts United just three points behind City in the Premier League table. 
        Erik ten Hag praised his team's resilience while Pep Guardiola admitted his side wasn't clinical enough in front of goal.
    """,
    "technology": """
        Apple unveiled its latest iPhone model today, featuring a revolutionary new camera system and the fastest mobile processor on the market. 
        The iPhone 15 Pro includes a 10x optical zoom lens and can record 8K video. 
        CEO Tim Cook called it "the most advanced smartphone we've ever created." 
        The device will be available next month starting at $999, with pre-orders beginning this Friday.
    """,
    "business": """
        Tesla reported record quarterly earnings, exceeding Wall Street expectations. 
        The electric vehicle maker posted a profit of $3.2 billion on $24.3 billion in revenue. 
        CEO Elon Musk announced plans to expand production capacity at the company's Berlin and Texas factories. 
        Tesla shares rose 7% in after-hours trading following the announcement.
    """
}

def test_classification_pipeline():
    """Test the classification pipeline."""
    logger.info("Testing classification pipeline...")
    
    try:
        # Import classification pipeline
        from intelligence.classification.pipeline import ClassificationPipeline, ClassificationInput
        
        # Create pipeline
        pipeline = ClassificationPipeline(domain_name="general")
        
        results = {}
        
        # Test with each sample text
        for domain, text in SAMPLE_TEXTS.items():
            logger.info(f"Testing classification with {domain} text...")
            
            input_data = ClassificationInput(text=text)
            result = pipeline.process(input_data)
            
            # Log result
            logger.info(f"Classification result: {result.primary_topic} ({result.primary_topic_confidence:.4f})")
            
            results[domain] = {
                "primary_topic": result.primary_topic,
                "primary_topic_id": result.primary_topic_id,
                "confidence": result.primary_topic_confidence,
                "is_relevant": result.is_relevant,
                "processing_time": result.processing_time
            }
        
        # Print summary
        logger.info("Classification test results:")
        for domain, result in results.items():
            logger.info(f"- {domain}: {result['primary_topic']} ({result['confidence']:.4f})")
            
        return True, results
    
    except Exception as e:
        logger.error(f"Classification test failed: {str(e)}", exc_info=True)
        return False, None

def test_entity_extraction_pipeline():
    """Test the entity extraction pipeline."""
    logger.info("Testing entity extraction pipeline...")
    
    try:
        # Import entity extraction pipeline
        from intelligence.entities.pipeline import EntityExtractionPipeline, EntityExtractionInput
        
        # Create pipeline
        pipeline = EntityExtractionPipeline(domain="general")
        
        results = {}
        
        # Test with each sample text
        for domain, text in SAMPLE_TEXTS.items():
            logger.info(f"Testing entity extraction with {domain} text...")
            
            input_data = EntityExtractionInput(text=text)
            result = pipeline.process(input_data)
            
            # Log result
            logger.info(f"Extracted {len(result.entities)} entities")
            
            # Count entities by type
            entity_types = {}
            for entity in result.entities:
                entity_type = entity.label
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            # Log entity types
            for entity_type, count in entity_types.items():
                logger.info(f"- {entity_type}: {count} entities")
            
            results[domain] = {
                "total_entities": len(result.entities),
                "entity_types": entity_types,
                "processing_time": result.processing_time
            }
        
        # Print summary
        logger.info("Entity extraction test results:")
        for domain, result in results.items():
            logger.info(f"- {domain}: {result['total_entities']} entities found")
            
        return True, results
    
    except Exception as e:
        logger.error(f"Entity extraction test failed: {str(e)}", exc_info=True)
        return False, None

def test_intelligence_integration():
    """Test the intelligence integration module."""
    logger.info("Testing intelligence integration...")
    
    try:
        # Create a mock content object
        class MockContent:
            def __init__(self, id, url, title, extracted_text):
                self.id = id
                self.url = url
                self.title = title
                self.extracted_text = extracted_text
                self.crawl_depth = 1
        
        # Import intelligence integration
        from TextHarvester.scraper.intelligence_integration import IntelligenceProcessor
        
        # Create processor
        processor = IntelligenceProcessor(domain="general")
        
        results = {}
        
        # Test with each sample text
        for i, (domain, text) in enumerate(SAMPLE_TEXTS.items()):
            logger.info(f"Testing intelligence integration with {domain} text...")
            
            # Create mock content
            content = MockContent(
                id=i+1,
                url=f"https://example.com/{domain}",
                title=f"Sample {domain.capitalize()} Article",
                extracted_text=text
            )
            
            # Process content
            result = processor.process_content(content)
            
            # Log result
            if result["classification"]:
                logger.info(f"Classification: {result['classification'].primary_topic}")
            else:
                logger.info("No classification result")
                
            if result["entities"]:
                logger.info(f"Entities: {len(result['entities'].entities)} found")
            else:
                logger.info("No entity extraction result")
            
            results[domain] = {
                "has_classification": result["classification"] is not None,
                "has_entities": result["entities"] is not None,
                "processing_time": result["processing_time"]
            }
        
        # Print summary
        logger.info("Intelligence integration test results:")
        for domain, result in results.items():
            logger.info(f"- {domain}: Classification={result['has_classification']}, Entities={result['has_entities']}")
            
        return True, results
    
    except Exception as e:
        logger.error(f"Intelligence integration test failed: {str(e)}", exc_info=True)
        return False, None

def run_all_tests():
    """Run all intelligence tests."""
    logger.info("Running all intelligence tests...")
    
    results = {
        "classification": None,
        "entity_extraction": None,
        "integration": None
    }
    
    # Test classification
    logger.info("\n" + "="*80 + "\nTESTING CLASSIFICATION\n" + "="*80)
    success, result = test_classification_pipeline()
    results["classification"] = {
        "success": success,
        "results": result
    }
    
    # Test entity extraction
    logger.info("\n" + "="*80 + "\nTESTING ENTITY EXTRACTION\n" + "="*80)
    success, result = test_entity_extraction_pipeline()
    results["entity_extraction"] = {
        "success": success,
        "results": result
    }
    
    # Test integration
    logger.info("\n" + "="*80 + "\nTESTING INTELLIGENCE INTEGRATION\n" + "="*80)
    success, result = test_intelligence_integration()
    results["integration"] = {
        "success": success,
        "results": result
    }
    
    # Print overall results
    logger.info("\n" + "="*80 + "\nTEST SUMMARY\n" + "="*80)
    for test_name, test_result in results.items():
        status = "PASSED" if test_result["success"] else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"intelligence_test_results_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Test results saved to {result_file}")
    
    return results

if __name__ == "__main__":
    run_all_tests()
