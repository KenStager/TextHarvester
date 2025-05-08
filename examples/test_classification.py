"""
Test script for the classification system.

This script demonstrates the use of the classification pipeline with
sample football content.
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from intelligence.classification.pipeline import ClassificationPipeline
from intelligence.classification.fast_filter import FastFilter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Sample football content
SAMPLE_TEXTS = [
    """
    Manchester United secured a dramatic 2-1 victory over Liverpool at Old Trafford
    on Sunday, with Marcus Rashford scoring the winning goal in the 82nd minute.
    The result moves United to within three points of the top four in the Premier League table.
    """,
    
    """
    The summer transfer window is heating up as Chelsea are reportedly preparing a
    £70 million bid for Napoli striker Victor Osimhen. The Blues are looking to
    strengthen their attacking options ahead of the new Premier League season.
    """,
    
    """
    Scientists have discovered a new species of deep-sea fish in the Mariana Trench.
    The previously unknown species can survive at depths of up to 8,000 meters and
    has unique adaptations to the high-pressure environment.
    """
]

def main():
    """Run a test of the classification pipeline."""
    print("Initializing football classification pipeline...")
    pipeline = ClassificationPipeline.create_football_pipeline()
    
    print("\nTesting with sample texts:")
    for i, text in enumerate(SAMPLE_TEXTS):
        print(f"\n--- Sample Text {i+1} ---")
        print(f"{text.strip()}\n")
        
        # Process through the pipeline
        result = pipeline.process(text)
        
        # Display results
        print(f"Is relevant: {result.is_relevant} (confidence: {result.confidence:.4f})")
        if result.is_relevant:
            print(f"Primary topic: {result.primary_topic} (confidence: {result.primary_topic_confidence:.4f})")
            
            if result.subtopics:
                print("Subtopics:")
                for subtopic in result.subtopics[:3]:  # Show top 3 subtopics
                    print(f"  - {subtopic['topic']} (confidence: {subtopic['confidence']:.4f})")
        
        print(f"Processing time: {result.processing_time:.4f} seconds")

if __name__ == "__main__":
    main()
