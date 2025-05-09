# TextHarvester Intelligence Integration

This document provides an overview of the intelligence integration features added to TextHarvester. These features enhance the web scraper with advanced content analysis capabilities, including classification and entity extraction.

## Overview

The intelligence integration connects TextHarvester's web scraping capabilities with advanced content analysis. When enabled, the system will automatically:

1. Classify content into topics and determine relevance
2. Extract named entities (people, organizations, locations, etc.)
3. Store the results in a structured format for analysis

## Features

- **Content Classification**: Automatically categorize content into topics with confidence scores
- **Entity Extraction**: Identify and extract named entities from text with contextual information
- **Domain-Specific Processing**: Specialized processing for different domains (e.g., football)
- **Integration with Crawling**: Intelligence processing happens automatically during crawling
- **Performance Optimizations**: Lazy loading of intelligence components to minimize resource usage

## Setup Instructions

### 1. Database Setup

First, run the database migration script to add the necessary tables and columns:

```bash
cd TextHarvester
python db_migrations/add_intelligence_tables.py
```

This will:
- Add intelligence configuration options to the `scraping_configuration` table
- Create new tables for storing classification and entity data

### 2. Path Configuration

Ensure the intelligence module is in the Python path. The intelligence integration module will attempt to find it automatically, but you may need to adjust your Python path if you encounter import errors.

### 3. Testing Integration

Run the integration test to verify that everything is set up correctly:

```bash
python scraper/integration_test.py
```

This will create a test content item and process it through the intelligence pipelines to ensure that the integration is working properly.

## Using Intelligence Features

### Enabling Intelligence in the Web UI

1. Navigate to the configuration page for a scraping job
2. Click on "Intelligence Configuration"
3. Enable classification and/or entity extraction
4. Select the appropriate domain for your content
5. Save the configuration

### Manually Processing Content

You can also manually process existing content through the intelligence pipelines:

1. View a content item in the admin interface
2. Click on "Analyze with Intelligence"
3. The system will process the content and display the results

### Viewing Intelligence Results

#### Classification Results

Classification results show:
- Primary topic with confidence score
- Whether the content is relevant to the domain
- Subtopics and their confidence scores

#### Entity Results

Entity results show:
- Entity text and type
- Position in the original text
- Confidence score
- Entity linking information (if available)

## Database Schema

The intelligence integration adds the following tables:

### ContentClassification

Stores classification results for each content item:

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| content_id | Integer | FK to ScrapedContent |
| is_relevant | Boolean | Relevance flag |
| confidence | Float | Overall confidence |
| primary_topic | String | Main topic |
| primary_topic_id | String | Topic ID |
| primary_topic_confidence | Float | Topic confidence |
| subtopics | JSON | Subtopics data |
| processing_time | Float | Processing time in seconds |
| created_at | DateTime | Creation timestamp |

### ContentEntity

Stores extracted entities:

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| content_id | Integer | FK to ScrapedContent |
| entity_type | String | Entity type (PERSON, ORG, etc.) |
| entity_text | String | Entity text |
| start_char | Integer | Start position |
| end_char | Integer | End position |
| confidence | Float | Confidence score |
| entity_id | String | Optional entity ID for linking |
| metadata | JSON | Additional entity data |
| created_at | DateTime | Creation timestamp |

## Configuration Options

The following options are available for intelligence configuration:

| Option | Description | Default |
|--------|-------------|---------|
| enable_classification | Enable content classification | False |
| enable_entity_extraction | Enable entity extraction | False |
| intelligence_domain | Domain for intelligence processing | 'football' |
| store_raw_intelligence | Store detailed intelligence results | False |

## Technical Details

### Integration Architecture

The intelligence integration is designed to be:

1. **Optional**: Intelligence features can be enabled or disabled per scraping job
2. **Fault-tolerant**: Failures in intelligence processing won't disrupt scraping
3. **Efficient**: Components are loaded on-demand to minimize resources
4. **Extensible**: New intelligence features can be added easily

### File Structure

- `models_update.py`: New database models for intelligence data
- `scraper/intelligence_integration.py`: Main integration module
- `db_migrations/add_intelligence_tables.py`: Database migration script
- `api/intelligence.py`: API routes for intelligence features
- `templates/admin/intelligence_config.html`: UI template for configuration
- `scraper/integration_test.py`: Test script for intelligence integration

### Integration Flow

1. WebCrawler loads job configuration with intelligence settings
2. If intelligence features are enabled, IntelligenceProcessor is initialized
3. After content extraction, content is passed to the intelligence processor
4. Classification and entity extraction are performed if enabled
5. Results are stored in the database with the content

## Troubleshooting

### Common Issues

- **Import errors**: Make sure the intelligence module is in your Python path
- **Database errors**: Run the migration script to create required tables
- **Performance issues**: Disable intelligence features for large crawl jobs
- **Memory usage**: Consider limiting the number of parallel workers when using intelligence

### Logs

Intelligence-related logs are prefixed with "intelligence_integration" and can be found in the application logs.

## Future Improvements

Potential areas for enhancement:

- Add more intelligence domains beyond football
- Implement result caching to improve performance
- Create analytics dashboards for intelligence data
- Add batch processing for existing content
- Integrate with external knowledge bases for better entity linking
