# TextHarvester Intelligence Implementation Notes

## Overview

This document provides implementation notes and technical details for the TextHarvester intelligence features. It's intended for developers working on maintaining or extending these features.

## Key Components Implemented

### 1. Text Processing Utilities

Added to `intelligence/utils/text_processing.py`:

- **normalize_text**: Standardized text preprocessing function used by classification and entity extraction pipelines
  - Handles text cleaning, normalization, stopword removal, and lemmatization
  - Gracefully handles missing dependencies (spaCy)
  - Provides consistent preprocessing across all intelligence components

- **extract_keywords**: Function for extracting relevant keywords from text
  - Uses spaCy when available for linguistic extraction
  - Falls back to word frequency-based extraction when spaCy isn't available
  - Combines named entities, noun chunks, and important words

### 2. Model Utilities

Added to `intelligence/utils/model_utils.py`:

- **get_model_path**: Standard path construction for model files
  - Constructs paths based on domain and model type
  - Creates directories as needed
  - Uses configurable base directories

- **get_embeddings**: Text embedding generation
  - Handles Sentence-Transformers embeddings with fallbacks
  - Gracefully degrades when libraries are missing
  - Supports domain-specific models

### 3. Configuration Settings

Added to `intelligence/config.py`:

- **TAXONOMY_EXPORT_PATH**: Path for taxonomy data
- **MODEL_CACHE_SIZE**: Controls model memory usage
- **MODEL_CACHE_TTL**: Time-to-live for cached models
- **Intelligence-specific directories**: For models, data, and cache

### 4. Database Model Fixes

Fixed SQLAlchemy reserved attribute name conflicts in `db/models/entity_models.py`:

- Renamed `metadata` to `entity_metadata` in Entity model
- Renamed `metadata` to `relation_metadata` in EntityRelationship model
- Created migration script in `db_migrations/update_entity_models.py`

### 5. Directory Structure & Data Files

Created:
- `intelligence/data/`: For taxonomy files and other data
  - `default_taxonomy.json`: General topic taxonomy
  - `football_taxonomy.json`: Football-specific taxonomy
- `intelligence/cache/`: For model storage
  - Subdirectories for different model types and domains

### 6. Mock Model System

Added support for testing without full ML models:

- `intelligence/create_mockup_models.py`: Mock model generation
- `run_mockup.py`: Easy setup script
- Fallback mechanisms in intelligence pipelines

## Implementation Considerations

### Performance Optimization

- Lazy loading of intelligence components
- Model caching with size limits
- Graceful degradation for missing dependencies
- Minimal memory usage for mock models

### Error Handling

- Comprehensive error handling throughout pipelines
- Isolation of intelligence errors from core functionality
- Informative logging at appropriate levels
- Fallback mechanisms for missing components

### Compatibility

- Compatible with existing scraper architecture
- Integration with the UI components
- Support for both development and production environments
- Support for different domains (general, football)

## Known Limitations

1. **Mock Models vs. Real Models**: The mock models provide structure but not actual intelligence. Real models should be integrated for production use.

2. **Limited Domain Support**: Currently supports general and football domains. Other domains need additional taxonomy and entity type definitions.

3. **Performance with Large Datasets**: The current implementation works well for individual documents but may need optimization for large batch processing.

4. **Model Dependencies**: Real transformer models require significant disk space and memory.

## Future Development

1. **Real ML Models**: Replace mock models with pre-trained models
2. **Additional Domains**: Expand beyond football to other domains
3. **Performance Optimization**: Batch processing and parallel execution
4. **Enhanced Entity Linking**: Connect entities with knowledge graphs
5. **UI Improvements**: Better visualization of intelligence results

## Troubleshooting Guide

### Common Issues

1. **Missing Dependencies**:
   - Error: `ImportError: No module named 'transformers'`
   - Solution: Install required packages with `pip install -r requirements-intelligence.txt`

2. **SQLAlchemy Errors**:
   - Error: `Attribute name 'metadata' is reserved when using the Declarative API`
   - Solution: Run the migration script `python db_migrations/update_entity_models.py`

3. **Model Not Found**:
   - Error: `Error loading model: File not found`
   - Solution: Run mockup model creation script `python run_mockup.py`

4. **spaCy Model Missing**:
   - Error: `Can't find model 'en_core_web_sm'`
   - Solution: Install model with `python -m spacy download en_core_web_sm`

## Testing

The tests in `tests/test_intelligence.py` verify:
1. Classification pipeline functionality
2. Entity extraction pipeline functionality
3. Integration with the scraper

Run tests with:
```bash
python tests/test_intelligence.py
```

## Database Migration

When deploying to existing systems with data, run the migration script to update database schema:
```bash
python db_migrations/update_entity_models.py
```

This script safely renames the `metadata` columns to avoid conflicts with SQLAlchemy.
