"""
Basic Pipeline Example
====================

This script demonstrates the usage of the base pipeline functionality.
"""

import sys
import os
import time
import logging
from typing import Dict, List, Any
import random

# Add parent directory to path to import from TextHarvester package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from our intelligence package
from intelligence import BasePipeline, ParallelPipeline, WorkItem
from intelligence.utils import clean_text, normalize_football_teams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextCleaningPipeline(BasePipeline):
    """Example pipeline that cleans and normalizes text."""
    
    def __init__(self, config=None):
        """Initialize the pipeline."""
        super().__init__('text_cleaning', config)
        
    def process_item(self, item: WorkItem) -> WorkItem:
        """Process a single work item by cleaning its text."""
        logger.info(f"Processing item {item.item_id}")
        
        # Get text from item
        text = item.data
        
        # Clean the text
        cleaned_text = clean_text(text)
        
        # Apply domain-specific normalization if domain is specified
        domain = item.metadata.get('domain')
        if domain == 'football':
            cleaned_text = normalize_football_teams(cleaned_text)
            
        # Set the result
        item.result = cleaned_text
        
        # Simulate processing time
        time.sleep(0.1)
        
        return item


class FootballEntityExtractionPipeline(BasePipeline):
    """Example pipeline that extracts football-related entities."""
    
    def __init__(self, config=None):
        """Initialize the pipeline."""
        super().__init__('football_entity_extraction', config)
        
        # Define patterns for entity extraction (simplified)
        self.team_patterns = {
            'Manchester United': 'TEAM',
            'Liverpool': 'TEAM',
            'Arsenal': 'TEAM',
            'Chelsea': 'TEAM',
            'Manchester City': 'TEAM'
        }
        
        self.player_patterns = {
            'Salah': 'PLAYER',
            'Kane': 'PLAYER',
            'Fernandes': 'PLAYER',
            'De Bruyne': 'PLAYER',
            'Rashford': 'PLAYER'
        }
        
    def process_item(self, item: WorkItem) -> WorkItem:
        """Process a single work item by extracting entities."""
        logger.info(f"Processing item {item.item_id}")
        
        # Get text from item
        text = item.data
        
        # Simple entity extraction (using string matching for demonstration)
        entities = []
        
        # Extract teams
        for team_name, entity_type in self.team_patterns.items():
            if team_name in text:
                entities.append({
                    'text': team_name,
                    'type': entity_type,
                    'start': text.index(team_name),
                    'end': text.index(team_name) + len(team_name)
                })
        
        # Extract players
        for player_name, entity_type in self.player_patterns.items():
            if player_name in text:
                entities.append({
                    'text': player_name,
                    'type': entity_type,
                    'start': text.index(player_name),
                    'end': text.index(player_name) + len(player_name)
                })
        
        # Set the result
        item.result = {
            'text': text,
            'entities': entities
        }
        
        # Simulate processing time
        time.sleep(0.2)
        
        return item
        
    def postprocess(self, results: List[WorkItem]) -> Dict[str, Any]:
        """Postprocess work items into a summary."""
        all_entities = []
        
        for item in results:
            if item.error is None and item.result:
                all_entities.extend(item.result['entities'])
                
        # Group by entity type
        entity_types = {}
        for entity in all_entities:
            entity_type = entity['type']
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
        return {
            'total_entities': len(all_entities),
            'entity_types': entity_types,
            'entities': all_entities
        }


def simple_processor(text):
    """Example processor function for use with ParallelPipeline."""
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Count words
    word_count = len(cleaned_text.split())
    
    # Return processed result
    return {
        'cleaned_text': cleaned_text,
        'word_count': word_count,
        'original_length': len(text),
        'cleaned_length': len(cleaned_text)
    }


def main():
    """Main function to demonstrate pipeline usage."""
    # Sample data
    football_texts = [
        "Manchester United secured a dramatic late victory against Liverpool at Old Trafford.",
        "Kane scored twice as Tottenham defeated Arsenal in the north London derby.",
        "De Bruyne's injury is a major blow for Manchester City ahead of their match with Chelsea.",
        "Salah continues his impressive form for Liverpool with another goal against Everton.",
        "The title race between Manchester City and Arsenal is heating up with five games to go.",
        "Manchester United have agreed terms with Rashford on a new contract extension.",
        "Chelsea announced the signing of a new striker from Barcelona for £65 million.",
        "The Premier League match between Liverpool and Manchester United ended in a thrilling 2-2 draw.",
        "Arsenal's young team has shown remarkable progress under their new manager.",
        "Fernandes will captain Manchester United in their Champions League fixture."
    ]
    
    # Example 1: Basic text cleaning pipeline
    logger.info("=== Example 1: Basic Text Cleaning Pipeline ===")
    
    # Create pipeline
    cleaning_pipeline = TextCleaningPipeline()
    
    # Process data
    cleaned_texts = cleaning_pipeline.process(football_texts)
    
    # Show results
    logger.info(f"Processed {len(cleaned_texts)} texts")
    for i, text in enumerate(cleaned_texts[:3]):
        logger.info(f"Sample {i+1}: {text[:50]}...")
    
    # Example 2: Football entity extraction pipeline
    logger.info("\n=== Example 2: Football Entity Extraction Pipeline ===")
    
    # Create pipeline with configuration
    entity_pipeline = FootballEntityExtractionPipeline({
        'batch_size': 3  # Process in smaller batches
    })
    
    # Process data
    entity_results = entity_pipeline.process(football_texts)
    
    # Show results
    logger.info(f"Extracted {entity_results['total_entities']} entities")
    logger.info(f"Entity types: {entity_results['entity_types']}")
    
    # Show some example entities
    for i, entity in enumerate(entity_results['entities'][:5]):
        logger.info(f"Entity {i+1}: {entity['text']} ({entity['type']})")
    
    # Example 3: Parallel processing pipeline
    logger.info("\n=== Example 3: Parallel Processing Pipeline ===")
    
    # Create parallel pipeline
    parallel_pipeline = ParallelPipeline('text_stats', simple_processor, {
        'max_workers': 4,
        'use_processes': False  # Use threads instead of processes
    })
    
    # Process data in parallel
    parallel_results = parallel_pipeline.process_parallel(football_texts)
    
    # Show results
    logger.info(f"Processed {len(parallel_results)} texts in parallel")
    
    # Calculate average statistics
    avg_word_count = sum(result['word_count'] for result in parallel_results) / len(parallel_results)
    avg_reduction = sum(1 - (result['cleaned_length'] / result['original_length']) 
                         for result in parallel_results) / len(parallel_results)
    
    logger.info(f"Average word count: {avg_word_count:.1f}")
    logger.info(f"Average text reduction: {avg_reduction:.1%}")
    
    # Example 4: Asynchronous processing
    logger.info("\n=== Example 4: Asynchronous Processing ===")
    
    # Create a new pipeline
    async_pipeline = TextCleaningPipeline({
        'max_workers': 3
    })
    
    # Generate more sample data
    more_texts = [
        f"Sample text {i} with random words: {' '.join(random.sample(['football', 'match', 'goal', 'player', 'team', 'league', 'cup', 'stadium', 'fans', 'coach'], 5))}"
        for i in range(20)
    ]
    
    # Start asynchronous processing
    async_pipeline.process_async(more_texts)
    
    # Monitor progress
    logger.info(f"Started asynchronous processing of {len(more_texts)} texts")
    
    for _ in range(5):
        state = async_pipeline.state.to_dict()
        logger.info(f"Progress: {state['processed_items']}/{len(more_texts)} items processed")
        time.sleep(0.5)
    
    # Wait for completion
    logger.info("Waiting for completion...")
    async_pipeline.wait_for_completion()
    
    # Get results
    results = async_pipeline.get_results()
    logger.info(f"Async processing completed: {len(results.successful_items)} successful, {len(results.failed_items)} failed")
    
    # Example 5: Streaming processing
    logger.info("\n=== Example 5: Stream Processing ===")
    
    # Create generator function for streaming
    def text_stream_generator():
        for i in range(10):
            yield f"Streaming text {i}: The match between Team A and Team B ended in a {random.randint(0, 5)}-{random.randint(0, 5)} draw."
            time.sleep(0.2)
    
    # Create pipeline
    stream_pipeline = TextCleaningPipeline()
    
    # Process stream
    logger.info("Processing text stream...")
    for i, result in enumerate(stream_pipeline.process_stream(text_stream_generator(), batch_size=3)):
        logger.info(f"Stream result {i+1}: {result[:30]}...")
    
    logger.info("All examples completed!")


if __name__ == '__main__':
    main()
