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

### 2. Initialize Mock Models

For testing without downloading large language models, you can create mockup models:

```bash
python run_mockup.py
```

This creates lightweight test models that allow the intelligence pipelines to run.

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

2. **Entity Extraction Pipeline**: Identifies named entities in content
   - Uses spaCy for base entity recognition
   - Enhanced with domain-specific patterns
   - Includes entity linking for knowledge graph integration

3. **Intelligence Integration**: Connects pipelines with the scraper
   - Lazy loading of components to minimize resource usage
   - Error isolation to prevent failures from affecting scraping
   - Configurable through the scraping interface

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

Make sure the taxonomy files exist in `intelligence/data/`:
- `default_taxonomy.json`
- `football_taxonomy.json` (for football domain)

### Entity Extraction Errors

Ensure mock models were created in the `intelligence/cache/` directory or download proper language models.

## Extending Intelligence Features

To add new domains or capabilities:

1. Create domain-specific taxonomies in `intelligence/data/`
2. Add domain-specific entity types in `intelligence/entities/entity_types.py`
3. Update model paths in `intelligence/utils/model_utils.py`

Refer to the `INTELLIGENCE-ROADMAP.md` document for future development plans.

