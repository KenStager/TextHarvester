# TextHarvester Intelligence Overview

## Introduction

The intelligence module in TextHarvester provides automated analysis capabilities for web-scraped content. It consists of two core components: content classification and entity extraction, both integrated with the web scraping pipeline for seamless operation.

## Core Components

### 1. Classification Pipeline

The classification pipeline categorizes content into topics and subtopics, helping organize large amounts of scraped data intelligently.

#### Key Features:
- **Domain-specific classification**: Supports different domains (general, football, etc.)
- **Fast filtering**: Quick pre-screening for relevant content
- **Hierarchical classification**: Multi-level topic classification
- **Graceful fallbacks**: Provides meaningful defaults when ML models aren't available

#### Usage:
```python
from intelligence.classification.pipeline import ClassificationPipeline, ClassificationInput

pipeline = ClassificationPipeline(domain_name="general")
result = pipeline.process(ClassificationInput(text="Your content here"))

print(f"Primary topic: {result.primary_topic}")
print(f"Confidence: {result.primary_topic_confidence}")
for subtopic in result.subtopics:
    print(f"Subtopic: {subtopic['topic']}")
```

### 2. Entity Extraction Pipeline

The entity extraction pipeline identifies and extracts named entities (people, organizations, locations, etc.) from content.

#### Key Features:
- **Base NER using spaCy**: Leverages spaCy's powerful entity recognition
- **Domain-specific patterns**: Enhanced with custom patterns for specialized domains
- **Entity linking**: Links extracted entities to knowledge base entries (when available)
- **Context extraction**: Provides surrounding context for extracted entities

#### Usage:
```python
from intelligence.entities.pipeline import EntityExtractionPipeline, EntityExtractionInput

pipeline = EntityExtractionPipeline(domain="general")
result = pipeline.process(EntityExtractionInput(text="Your content here"))

print(f"Found {len(result.entities)} entities")
for entity in result.entities:
    print(f"{entity.label}: {entity.text} ({entity.confidence:.2f})")
```

### 3. Intelligence Integration

The integration module connects the classification and entity extraction pipelines with the web scraping system.

#### Key Features:
- **Lazy loading**: Components are loaded only when needed to conserve resources
- **Error isolation**: Prevents intelligence failures from affecting core scraping
- **Database integration**: Stores intelligence results with scraped content
- **Robust error handling**: Gracefully handles missing dependencies or database issues

#### Usage:
```python
from TextHarvester.scraper.intelligence_integration import IntelligenceProcessor

processor = IntelligenceProcessor(
    domain="general",
    enable_classification=True,
    enable_entity_extraction=True
)

results = processor.process_content(content)
```

## Directory Structure

```
intelligence/
├── __init__.py
├── base_pipeline.py        # Base pipeline definition
├── config.py               # Configuration settings
├── classification/         # Classification components
│   ├── __init__.py
│   ├── pipeline.py         # Main classification pipeline
│   ├── classifiers.py      # Classifier implementations
│   ├── fast_filter.py      # Quick relevance filtering
│   ├── topic_taxonomy.py   # Topic hierarchy definitions
│   └── taxonomies/         # Domain-specific taxonomies
├── entities/               # Entity extraction components
│   ├── __init__.py
│   ├── pipeline.py         # Main entity extraction pipeline
│   ├── ner_model.py        # Named entity recognition models
│   ├── linking.py          # Entity linking components
│   ├── entity_types.py     # Entity type definitions
│   └── taxonomies/         # Domain-specific entity types
└── utils/                  # Shared utilities
    ├── __init__.py
    ├── text_processing.py  # Text normalization and cleaning
    └── model_utils.py      # Model loading and management
```

## Pattern Files

Pattern files are used by the entity extraction system to recognize domain-specific entities. They are located in:

```
models/patterns/
├── general_patterns.jsonl  # Patterns for general domain
└── football_patterns.jsonl # Patterns for football domain
```

Each pattern file follows the JSONL format, with each line containing a JSON object with `label` and `pattern` fields:

```json
{"label":"ORGANIZATION","pattern":"Google"}
{"label":"PERSON","pattern":"Tim Cook"}
```

## Default Classification

When machine learning models aren't available, the system falls back to default classifications by domain. These defaults are defined in the `create_default_for_domain` method in `intelligence/classification/classifiers.py`:

```python
@classmethod
def create_default_for_domain(cls, domain: str = "general", confidence: float = 0.7) -> 'ClassificationResult':
    """Create a default classification result for a domain."""
    domain_defaults = {
        "football": {
            "id": "football_news",
            "name": "Football News",
            # ...
        },
        "general": {
            "id": "general_news",
            "name": "General News",
            # ...
        }
        # Additional domains...
    }
    # ...
```

## Extending the Intelligence System

### Adding a New Domain

1. **Create domain-specific patterns**:
   - Add a new pattern file in `models/patterns/{domain}_patterns.jsonl`

2. **Add default classification**:
   - Update `create_default_for_domain` in `classifiers.py` with domain-specific defaults

3. **Add domain taxonomy** (optional):
   - Create a new taxonomy file in `intelligence/classification/taxonomies/{domain}.py`

4. **Add domain-specific entity types** (optional):
   - Create a new entity type file in `intelligence/entities/taxonomies/{domain}_entities.py`

### Enhancing Existing Features

- **Improve classification**: Add domain-specific keywords to the fast filter
- **Enhance entity extraction**: Add more patterns to pattern files
- **Add new entity types**: Extend the entity type registry in `intelligence/entities/entity_types.py`

## Troubleshooting

### Common Issues

1. **"No module named 'app'"**: Import path issue with database operations
   - Solution: Update import paths in intelligence_integration.py

2. **"No fast filter available"**: Missing fast filter implementation
   - Solution: Ensure domain keywords are defined in the fast filter

3. **"No patterns file found"**: Missing entity patterns
   - Solution: Create the required pattern file in models/patterns/

4. **"No hierarchical classifier available"**: Missing classifier implementation
   - Solution: The system will use default classification results

### Testing

Test the intelligence system with:
```bash
python tests/test_intelligence.py
```

This runs tests on all components and shows detailed output about classification and entity extraction results.
