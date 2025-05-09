# TextHarvester Intelligence Features

## Overview

TextHarvester intelligence features provide content analysis capabilities for web scraped text, including:

1. **Content Classification**: Automatically categorize content into topics and subtopics
2. **Entity Extraction**: Identify and extract named entities like people, organizations, locations, etc.
3. **Intelligence Integration**: Connect these capabilities with web scraping for automated analysis

## Setup Instructions

### 1. Install Dependencies

The intelligence features require additional dependencies beyond the core TextHarvester requirements:

```bash
# Install basic dependencies
pip install -r requirements.txt

# Install intelligence-specific dependencies
pip install -r requirements-intelligence.txt
```

Or use the setup script for a complete setup:

```bash
python setup_intelligence.py
```

This will:
- Install required Python packages
- Download spaCy models
- Set up NLTK data
- Create directory structure
- Generate mock models for testing

### 2. Pattern Files Setup

The entity extraction system requires pattern files for entity recognition. If you don't see these files after installation, create them manually:

```bash
# Create patterns directory if it doesn't exist
mkdir -p models/patterns

# Create a general patterns file
echo '{"label":"ORGANIZATION","pattern":"Google"}\n{"label":"ORGANIZATION","pattern":"Microsoft"}\n{"label":"PERSON","pattern":"Tim Cook"}' > models/patterns/general_patterns.jsonl
```

## Using Intelligence Features

### In Scraping Configurations

Enable intelligence features in your scraping configuration:

```python
config = ScrapingConfiguration(
    name="My Scraping Job",
    source_list_id=1,
    max_depth=2,
    # Intelligence settings
    enable_classification=True,
    enable_entity_extraction=True,
    intelligence_domain="football"  # or "general"
)
```

### Directly from Python

```python
# Content Classification
from intelligence.classification.pipeline import ClassificationPipeline, ClassificationInput

pipeline = ClassificationPipeline(domain_name="football")
result = pipeline.process(ClassificationInput(text="Your text here"))
print(f"Topic: {result.primary_topic}, Confidence: {result.primary_topic_confidence}")

# Entity Extraction
from intelligence.entities.pipeline import EntityExtractionPipeline, EntityExtractionInput

pipeline = EntityExtractionPipeline(domain="football")
result = pipeline.process(EntityExtractionInput(text="Your text here"))
print(f"Found {len(result.entities)} entities")
```

## Testing

You can test the intelligence features using:

```bash
python tests/test_intelligence.py
```

This runs tests on both pipelines and the integration module.

## Architecture

The intelligence module consists of these key components:

1. **Classification Pipeline**: Categorizes content into topics and subtopics
   - Uses both traditional ML and transformer-based models
   - Hierarchical classification for detailed topic analysis
   - Fallback to reasonable defaults when models aren't available

2. **Entity Extraction Pipeline**: Identifies named entities in content
   - Uses spaCy for base entity recognition
   - Enhanced with domain-specific patterns
   - Includes entity linking for knowledge graph integration

3. **Intelligence Integration**: Connects pipelines with the scraper
   - Lazy loading of components to minimize resource usage
   - Error isolation to prevent failures from affecting scraping
   - Configurable through the scraping interface
   - Robust database error handling

4. **Utilities**:
   - Text processing utilities for normalization and cleaning
   - Model management utilities for loading and caching models
   - Configuration management for flexibility

## Troubleshooting

### Missing Dependencies

If you see import errors, install the required dependencies:

```bash
pip install torch transformers sentence-transformers spacy nltk
python -m spacy download en_core_web_sm
```

### No Classification Results

If classification returns "Unknown" with low confidence:
1. Check if pattern files exist in `models/patterns/`
2. Try generating mock models with `python run_mockup.py`
3. Make sure domain configuration files exist in `intelligence/data/`

### Entity Extraction Errors

Common entity extraction issues:
1. Missing pattern files - create them in `models/patterns/`
2. SpaCy model not found - run `python -m spacy download en_core_web_sm`
3. Pattern format issues - use the proper JSONL format

### Database Operation Failures

If database operations fail:
1. Make sure database settings are properly configured in `.env`
2. Check if models exist in `TextHarvester/models_update.py`
3. Verify connections between intelligence components and database

## Extending Intelligence Features

To add new domains or capabilities:

1. Create domain-specific taxonomies in `intelligence/data/`
2. Add domain-specific entity patterns in `models/patterns/[domain]_patterns.jsonl`
3. Update model paths in `intelligence/utils/model_utils.py`
4. Add domain-specific classification default outputs in `create_default_for_domain`

Refer to the `INTELLIGENCE-ROADMAP.md` document for future development plans.

