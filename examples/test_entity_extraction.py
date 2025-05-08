"""
Test script for the entity extraction system.

This script demonstrates the use of the entity extraction pipeline with
sample football content to identify and link entities.
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from intelligence.entities.pipeline import EntityExtractionPipeline
from intelligence.entities.ner_model import FootballNERModel
from intelligence.entities.linking import EntityLinker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Sample football content
SAMPLE_TEXTS = [
    """
    Liverpool secured a dramatic 2-1 victory over Manchester United at Anfield
    on Sunday, with Mohamed Salah scoring the winning goal in the 82nd minute.
    Manager Jurgen Klopp praised his team's resilience after falling behind to
    a Bruno Fernandes strike early in the second half.
    """,
    
    """
    Chelsea have confirmed the signing of Victor Osimhen from Napoli for a 
    club-record fee of £85 million. The Nigerian striker has signed a five-year 
    contract at Stamford Bridge and will wear the number 9 shirt. Osimhen scored 
    26 goals in Serie A last season, helping Napoli secure Champions League qualification.
    """,
    
    """
    The Premier League has announced a new broadcasting deal worth £6.7 billion
    over the next four seasons. Sky Sports will show 128 matches per season,
    while TNT Sports will broadcast 52 games. Amazon Prime has secured rights
    to stream 20 matches, including all Boxing Day fixtures.
    """
]

def main():
    """Run a test of the entity extraction pipeline."""
    print("Initializing football entity extraction pipeline...")
    pipeline = EntityExtractionPipeline.create_football_pipeline()
    
    print("\nTesting with sample texts:")
    for i, text in enumerate(SAMPLE_TEXTS):
        print(f"\n--- Sample Text {i+1} ---")
        print(f"{text.strip()}\n")
        
        # Process through the pipeline
        result = pipeline.process(text)
        
        # Display results
        print(f"Extracted {len(result.entities)} entities in {result.processing_time:.4f} seconds")
        
        if result.entity_counts:
            print("\nEntity counts by type:")
            for entity_type, count in result.entity_counts.items():
                print(f"  - {entity_type}: {count}")
        
        if result.linked_entities:
            print("\nLinked entities:")
            for linked_entity in result.linked_entities[:5]:  # Show up to 5 linked entities
                print(f"  - {linked_entity.mention.text} ({linked_entity.type})")
                print(f"    KB ID: {linked_entity.kb_id}")
                print(f"    Confidence: {linked_entity.score:.4f}")
        
        if result.unlinked_entities:
            print("\nUnlinked entities:")
            for entity in result.unlinked_entities[:5]:  # Show up to 5 unlinked entities
                print(f"  - {entity.text} ({entity.label})")
                print(f"    Span: {entity.start_char}-{entity.end_char}")
                print(f"    Confidence: {entity.confidence:.4f}")
        
        # Demonstrate additional capabilities
        if i == 0:  # For the first text only
            print("\nAdditional Analysis:")
            analysis = pipeline.analyze_entity_distribution(text)
            print(f"  Entity density: {analysis['entity_density']:.4f} entities per word")
            if analysis['co_occurrence']:
                print("  Common entity co-occurrences:")
                for co in analysis['co_occurrence'][:3]:  # Show top 3
                    print(f"    - {' + '.join(co['types'])}: {co['count']} occurrences")

if __name__ == "__main__":
    main()
