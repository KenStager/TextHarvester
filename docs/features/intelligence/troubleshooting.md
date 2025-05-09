# TextHarvester Intelligence Troubleshooting Guide

This guide helps you diagnose and fix issues with the TextHarvester intelligence features.

## Common Issues and Solutions

### Classification Issues

#### Problem: Classification returns "Unknown" with 0.0 confidence

**Symptoms:**
- Classification results show "Unknown" as the primary topic
- Confidence score is 0.0
- Logs show "No hierarchical classifier available" warnings

**Solutions:**
1. Check if you've created the default taxonomy files:
   ```bash
   # Create directory if it doesn't exist
   mkdir -p intelligence/data
   
   # Create a basic taxonomy file
   echo '{"topics":[{"id":"general","name":"General","subtopics":[]}]}' > intelligence/data/default_taxonomy.json
   ```

2. Verify that the `create_default_for_domain` method is implemented in `intelligence/classification/classifiers.py`

3. Try running with mock models:
   ```bash
   python run_mockup.py
   ```

#### Problem: Classification doesn't recognize domain-specific content

**Symptoms:**
- Content is classified as general topics despite being domain-specific
- Confidence scores are lower than expected

**Solutions:**
1. Check domain name spelling and case sensitivity
2. Verify domain-specific taxonomy exists
3. Add more domain keywords to the fast filter:
   ```python
   # In fast_filter.py
   filter_instance.add_domain_keywords("your_domain", ["keyword1", "keyword2",...])
   ```

### Entity Extraction Issues

#### Problem: No entities are extracted

**Symptoms:**
- Entity extraction returns empty lists
- No entity counts in the result

**Solutions:**
1. Check if SpaCy is properly installed:
   ```bash
   pip install spacy
   python -m spacy download en_core_web_sm
   ```

2. Verify pattern files exist:
   ```bash
   # Create patterns directory
   mkdir -p models/patterns
   
   # Create a basic pattern file
   echo '{"label":"ORG","pattern":"Google"}\n{"label":"ORG","pattern":"Microsoft"}' > models/patterns/general_patterns.jsonl
   ```

3. Check the text for recognizable entities (length, formatting)

#### Problem: Custom entities are not recognized

**Symptoms:**
- Basic entities (PERSON, ORG) are found but domain-specific ones are not
- Missing expected entity types in the results

**Solutions:**
1. Check pattern file format:
   ```json
   {"label":"CUSTOM_TYPE","pattern":"Exact Pattern Text"}
   ```

2. Verify the custom entity type is registered:
   ```python
   # In entity_types.py
   registry.add_entity_type(EntityType("CUSTOM_TYPE", "Description"))
   ```

3. Ensure the appropriate domain is set on the pipeline:
   ```python
   pipeline = EntityExtractionPipeline(domain="your_domain")
   ```

### Integration Issues

#### Problem: Database errors when saving intelligence results

**Symptoms:**
- "Error saving classification" or "Error saving entities" in logs
- "No module named 'app'" or similar import errors

**Solutions:**
1. Fix import paths in `intelligence_integration.py`:
   ```python
   try:
       from TextHarvester.app import db
       from TextHarvester.models_update import ContentClassification
   except ImportError:
       try:
           from app import db
           from models_update import ContentClassification
       except ImportError:
           logger.error("Could not import database modules")
   ```

2. Check database configuration:
   - Verify `.env` file has correct DATABASE_URL
   - Make sure tables exist in the database

3. Temporarily disable database operations for testing:
   ```python
   # In intelligence_integration.py, modify save methods to return True without DB operations
   ```

#### Problem: Memory usage issues with large crawls

**Symptoms:**
- Out of memory errors during large crawls
- System slows down significantly when intelligence is enabled

**Solutions:**
1. Reduce batch sizes in config.py
2. Enable lazy loading for all components
3. Implement content length limits:
   ```python
   # In pipeline.py, truncate text before processing
   if len(text) > MAX_LENGTH:
       text = text[:MAX_LENGTH]
   ```

## Diagnostics

### Testing Components Individually

#### Test Classification Only

```python
from intelligence.classification.pipeline import ClassificationPipeline, ClassificationInput

def test_classification(text):
    pipeline = ClassificationPipeline(domain_name="general")
    input_data = ClassificationInput(text=text)
    result = pipeline.process(input_data)
    print(f"Topic: {result.primary_topic}, Confidence: {result.primary_topic_confidence}")
    print(f"Subtopics: {result.subtopics}")
    return result

result = test_classification("Your test text here")
```

#### Test Entity Extraction Only

```python
from intelligence.entities.pipeline import EntityExtractionPipeline, EntityExtractionInput

def test_entity_extraction(text):
    pipeline = EntityExtractionPipeline(domain="general")
    input_data = EntityExtractionInput(text=text)
    result = pipeline.process(input_data)
    print(f"Found {len(result.entities)} entities:")
    for entity in result.entities:
        print(f"- {entity.label}: {entity.text} ({entity.confidence:.2f})")
    return result

result = test_entity_extraction("Google and Microsoft are tech companies. Tim Cook is the CEO of Apple.")
```

### Checking Component Status

#### Verify SpaCy Model

```python
import spacy

def check_spacy():
    try:
        nlp = spacy.load("en_core_web_sm")
        print(f"SpaCy model loaded successfully: {nlp.meta}")
        return True
    except Exception as e:
        print(f"SpaCy model error: {e}")
        return False

check_spacy()
```

#### Check Pattern Files

```python
import os
import json

def check_patterns(domain="general"):
    pattern_file = f"models/patterns/{domain}_patterns.jsonl"
    if not os.path.exists(pattern_file):
        print(f"Pattern file not found: {pattern_file}")
        return False
    
    try:
        with open(pattern_file, 'r') as f:
            patterns = [json.loads(line) for line in f if line.strip()]
        print(f"Found {len(patterns)} patterns for domain '{domain}'")
        return True
    except Exception as e:
        print(f"Error reading pattern file: {e}")
        return False

check_patterns()
```

## Enabling Enhanced Logging

Add detailed logging for better debugging:

```python
import logging

# In your script or notebook
logging.basicConfig(level=logging.DEBUG)

# To enable specific components
logging.getLogger('intelligence.classification').setLevel(logging.DEBUG)
logging.getLogger('intelligence.entities').setLevel(logging.DEBUG)
```

## Complete Test Script

Save this as `test_intelligence_manual.py`:

```python
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_intelligence_manual")

# Add parent directory to path
sys.path.append('.')

sample_text = """
Google and Microsoft are competing in cloud computing. 
Apple announced its new iPhone in September.
The product will cost $999 and be available next month.
Tim Cook presented the keynote at their headquarters in Cupertino.
"""

def test_all():
    logger.info("Testing Intelligence Components")
    
    # Test classification
    logger.info("=== Testing Classification ===")
    from intelligence.classification.pipeline import ClassificationPipeline, ClassificationInput
    pipeline = ClassificationPipeline(domain_name="general")
    input_data = ClassificationInput(text=sample_text)
    result = pipeline.process(input_data)
    logger.info(f"Classification: {result.primary_topic} ({result.primary_topic_confidence:.4f})")
    
    # Test entity extraction
    logger.info("=== Testing Entity Extraction ===")
    from intelligence.entities.pipeline import EntityExtractionPipeline, EntityExtractionInput
    pipeline = EntityExtractionPipeline(domain="general")
    input_data = EntityExtractionInput(text=sample_text)
    result = pipeline.process(input_data)
    logger.info(f"Found {len(result.entities)} entities")
    for entity in result.entities:
        logger.info(f"- {entity.label}: {entity.text}")
    
    logger.info("=== Testing Integration ===")
    from TextHarvester.scraper.intelligence_integration import IntelligenceProcessor
    
    # Mock content object
    class MockContent:
        def __init__(self, id, text):
            self.id = id
            self.extracted_text = text
            self.url = "https://example.com"
            self.title = "Test Content"
            self.crawl_depth = 1
    
    content = MockContent(1, sample_text)
    processor = IntelligenceProcessor(domain="general")
    result = processor.process_content(content)
    
    logger.info("Intelligence test completed")

if __name__ == "__main__":
    test_all()
```

Run with:
```bash
python test_intelligence_manual.py
```
