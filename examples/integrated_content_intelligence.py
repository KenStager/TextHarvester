"""
Demonstration of integrated Content Intelligence Platform.

This script shows how the Topic Classification System and Entity Recognition System
work together to provide comprehensive content intelligence.
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from intelligence.classification.pipeline import ClassificationPipeline
from intelligence.entities.pipeline import EntityExtractionPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Sample football article
FOOTBALL_ARTICLE = """
MATCH REPORT: Liverpool 3-1 Arsenal - Reds go 8 points clear at the top

Liverpool extended their lead at the top of the Premier League to 8 points with an impressive 3-1 victory over Arsenal at Anfield on Saturday.

Goals from Mohamed Salah, Darwin Nunez and Trent Alexander-Arnold secured the three points for Jurgen Klopp's side after Martin Odegaard had given Arsenal an early lead.

The Gunners started brightly and took the lead in the 12th minute when Odegaard finished clinically after a swift counter-attack orchestrated by Bukayo Saka.

Liverpool responded well to going behind and drew level in the 35th minute when Salah converted from the penalty spot after being fouled by William Saliba in the box.

The Reds dominated the second half and took the lead through Nunez, who headed home Alexander-Arnold's pinpoint cross in the 58th minute. The right-back then added a third himself in the 72nd minute with a spectacular free-kick from 25 yards that gave David Raya no chance.

Arsenal manager Mikel Arteta introduced Gabriel Jesus and Leandro Trossard in an attempt to get back into the game, but Liverpool's defense, marshaled superbly by Virgil van Dijk, held firm.

The result maintains Liverpool's perfect home record this season and puts them in a commanding position at the top of the table ahead of the busy Christmas period.

"It was a tough game against a quality opposition," said Klopp after the match. "We showed great character to come back and in the second half we were really dominant. The atmosphere was incredible and helped us through the difficult moments."

Arteta was disappointed with the result but remained positive about his team's performance. "We started very well but Liverpool showed why they're top of the league. We have to learn from these experiences and keep improving."

Liverpool next face Manchester United at Old Trafford, while Arsenal host Chelsea in a London derby at the Emirates Stadium.
"""

# Sample non-football article for comparison
NON_FOOTBALL_ARTICLE = """
Scientists Discover New Species of Deep-Sea Fish in Mariana Trench

A team of marine biologists has discovered a previously unknown species of deep-sea fish living at a depth of over 8,000 meters in the Mariana Trench, the deepest known point in Earth's oceans.

The fish, temporarily named "Pseudoliparis marianensis" until formal classification is complete, appears to have unique adaptations to the extreme pressure of its deep-sea habitat. Researchers from the Woods Hole Oceanographic Institution captured high-resolution images and several specimens using a specialized deep-sea submersible.

Dr. Sarah Chen, the lead researcher on the expedition, described the discovery as "a significant breakthrough in our understanding of deep-sea ecosystems." The newly discovered fish has a translucent body, poorly developed eyes, and specialized cells that help it withstand pressures exceeding 800 times that at sea level.

"What's particularly fascinating about this species is its metabolic adaptations," explained Dr. Chen. "It appears to have a unique mechanism for energy conservation that allows it to survive in an environment with very limited food resources."

The research team spent three weeks exploring the Mariana Trench, located in the western Pacific Ocean, using advanced underwater technology that allowed them to collect samples and data from extreme depths that have been largely inaccessible to scientists until recently.

The discovery adds to growing evidence that deep-sea trenches, despite their extreme conditions, host diverse ecosystems that have evolved remarkable adaptations. The research findings will be published next month in the journal "Nature Oceanography."

Dr. Thomas Rivera, a marine biologist not involved in the research, called the discovery "a reminder of how much we still have to learn about our oceans, particularly their deepest regions."

The research was funded by the National Oceanic and Atmospheric Administration (NOAA) and will continue with further expeditions planned for next year to study the behavior and ecology of this newly discovered species.
"""

def process_article(article, title="Article"):
    """Process an article through both classification and entity extraction pipelines."""
    print(f"\n{'=' * 80}")
    print(f"PROCESSING: {title}")
    print(f"{'=' * 80}\n")
    
    # Print a snippet of the article
    print(f"Article snippet:\n{article[:300]}...\n")
    
    # Step 1: Topic Classification
    print("STEP 1: TOPIC CLASSIFICATION")
    print("-" * 50)
    
    classification_pipeline = ClassificationPipeline.create_football_pipeline()
    classification_result = classification_pipeline.process(article)
    
    print(f"Is relevant to football: {classification_result.is_relevant} (confidence: {classification_result.confidence:.4f})")
    
    if classification_result.is_relevant:
        print(f"Primary topic: {classification_result.primary_topic} (confidence: {classification_result.primary_topic_confidence:.4f})")
        
        if classification_result.subtopics:
            print("Subtopics:")
            for subtopic in classification_result.subtopics[:3]:  # Top 3 subtopics
                print(f"  - {subtopic['topic']} (confidence: {subtopic['confidence']:.4f})")
    
    print(f"\nClassification time: {classification_result.processing_time:.4f} seconds")
    
    # Step 2: Entity Extraction (only if relevant to football)
    if classification_result.is_relevant:
        print("\nSTEP 2: ENTITY EXTRACTION")
        print("-" * 50)
        
        extraction_pipeline = EntityExtractionPipeline.create_football_pipeline()
        extraction_result = extraction_pipeline.process(article)
        
        print(f"Extracted {len(extraction_result.entities)} entities in {extraction_result.processing_time:.4f} seconds")
        
        if extraction_result.entity_counts:
            print("\nEntity counts by type:")
            for entity_type, count in extraction_result.entity_counts.items():
                print(f"  - {entity_type}: {count}")
        
        if extraction_result.linked_entities:
            print("\nTop linked entities:")
            for linked_entity in extraction_result.linked_entities[:5]:  # Show up to 5 linked entities
                print(f"  - {linked_entity.mention.text} ({linked_entity.type})")
                print(f"    KB Name: {linked_entity.kb_name}")
                print(f"    Confidence: {linked_entity.score:.4f}")
        
        # Step 3: Additional Analysis
        print("\nSTEP 3: INTEGRATED ANALYSIS")
        print("-" * 50)
        
        # Analyze entities by subtopic
        print("Entity distribution across subtopics:")
        
        # This is a simplified demonstration of how you might combine topic and entity information
        # In a real implementation, you would analyze entity occurrences in sections of text 
        # that correspond to different subtopics
        
        for subtopic in classification_result.subtopics[:2]:  # Top 2 subtopics
            subtopic_name = subtopic['topic']
            print(f"\nSubtopic: {subtopic_name}")
            
            # Simple filtering based on entity type relevance to subtopic
            if "Match" in subtopic_name:
                relevant_types = {"TEAM", "PLAYER", "VENUE", "EVENT"}
            elif "Team" in subtopic_name:
                relevant_types = {"TEAM", "PERSON", "PLAYER"}
            elif "Player" in subtopic_name:
                relevant_types = {"PLAYER", "PERSON", "TEAM"}
            else:
                relevant_types = set(extraction_result.entity_counts.keys())
            
            # Count entities of relevant types
            relevant_entities = [e for e in extraction_result.entities if e.label in relevant_types]
            print(f"  Relevant entities: {len(relevant_entities)}")
            
            # List top entities
            if relevant_entities:
                print("  Top entities in this subtopic:")
                entity_samples = relevant_entities[:3]  # Just show 3 examples
                for entity in entity_samples:
                    print(f"    - {entity.text} ({entity.label})")

def main():
    """Demonstrate integrated content intelligence."""
    print("CONTENT INTELLIGENCE PLATFORM DEMONSTRATION\n")
    print("This demonstration shows how Topic Classification and Entity Recognition work together.")
    
    # Process football article
    process_article(FOOTBALL_ARTICLE, "Liverpool vs Arsenal Match Report")
    
    # Process non-football article
    process_article(NON_FOOTBALL_ARTICLE, "Marine Biology Article")

if __name__ == "__main__":
    main()
