"""
Demonstration of integrating the classification system with the scraper.

This script shows how to hook the classification pipeline into the
existing scraper's post-processing workflow.
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from intelligence.classification.pipeline import ClassificationPipeline, ClassificationInput

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# This would typically be imported from your scraper module
# For this example, we'll create a simple placeholder
class ScraperContent:
    """Simple placeholder for scraper content."""
    def __init__(self, content_id, text, source_url, title=None, metadata=None):
        self.id = content_id
        self.text = text
        self.source_url = source_url
        self.title = title or ""
        self.metadata = metadata or {}

def classify_scraped_content(content_items, save_to_db=False):
    """
    Classify scraped content using the classification pipeline.
    
    Args:
        content_items: List of ScraperContent objects
        save_to_db: Whether to save classification results to database
        
    Returns:
        List of (content, classification_result) tuples
    """
    # Initialize the classification pipeline
    pipeline = ClassificationPipeline.create_football_pipeline()
    
    results = []
    
    for content in content_items:
        # Create classification input
        input_data = ClassificationInput(
            text=content.text,
            content_id=content.id,
            metadata={
                "source_url": content.source_url,
                "title": content.title,
                **content.metadata
            }
        )
        
        # Process through the pipeline
        classification_result = pipeline.process(input_data)
        
        # Save to database if requested
        if save_to_db and classification_result.is_relevant:
            pipeline.save_classification_to_database(classification_result)
        
        # Add to results
        results.append((content, classification_result))
    
    return results

def main():
    """Demonstrate integration with scraper."""
    # Sample scraped content
    content_items = [
        ScraperContent(
            content_id=1,
            text="""
            Manchester City have won the Premier League title for the fourth consecutive season,
            setting a new record in English football. Pep Guardiola's side secured the title with
            a 4-0 victory over West Ham on the final day of the season.
            """,
            source_url="https://example.com/sports/city-champions",
            title="Man City Make History with Fourth Consecutive Premier League Title",
            metadata={"scrape_date": "2023-05-28"}
        ),
        ScraperContent(
            content_id=2,
            text="""
            The transfer saga of the summer has concluded as Kylian Mbappé has officially
            signed for Real Madrid on a five-year contract. The French superstar leaves
            Paris Saint-Germain after seven seasons at the club.
            """,
            source_url="https://example.com/sports/mbappe-madrid",
            title="Mbappé Joins Real Madrid in Blockbuster Transfer",
            metadata={"scrape_date": "2023-06-15"}
        ),
        ScraperContent(
            content_id=3,
            text="""
            Scientists have developed a new battery technology that could revolutionize
            electric vehicles. The new solid-state battery offers twice the energy density
            of current lithium-ion batteries and can be charged in just 10 minutes.
            """,
            source_url="https://example.com/tech/battery-breakthrough",
            title="Revolutionary Battery Technology to Transform Electric Vehicles",
            metadata={"scrape_date": "2023-06-10"}
        )
    ]
    
    print("Classifying scraped content...")
    results = classify_scraped_content(content_items)
    
    for content, classification in results:
        print(f"\n--- Content ID: {content.id} ---")
        print(f"Title: {content.title}")
        print(f"Source: {content.source_url}")
        print(f"Is relevant: {classification.is_relevant} (confidence: {classification.confidence:.4f})")
        
        if classification.is_relevant:
            print(f"Primary topic: {classification.primary_topic}")
            
            if classification.subtopics:
                print("Top subtopics:")
                for subtopic in classification.subtopics[:2]:
                    print(f"  - {subtopic['topic']} (confidence: {subtopic['confidence']:.4f})")
        
        print(f"Processing time: {classification.processing_time:.4f} seconds")

if __name__ == "__main__":
    main()
