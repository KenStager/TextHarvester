# TextHarvester Database Design

This document describes the database schema used by TextHarvester, including core scraping tables and intelligence-related models.

## Overview

TextHarvester uses a relational database (PostgreSQL recommended, SQLite supported for development) to store:

1. Source configurations and lists
2. Scraping job configurations and status
3. Collected content and metadata
4. Intelligence analysis results

## Core Tables

### SourceList

Stores collections of web sources to be scraped.

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| name | String | List name |
| description | Text | List description |
| created_at | DateTime | Creation timestamp |

### Source

Stores individual web sources.

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| source_list_id | Integer | FK to SourceList |
| base_url | String | Source base URL |
| name | String | Source name |
| description | Text | Optional description |
| last_checked | DateTime | Last validation timestamp |
| status | String | Source status |

### ScrapingConfiguration

Stores configuration parameters for scraping jobs.

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| name | String | Configuration name |
| source_list_id | Integer | FK to SourceList |
| max_depth | Integer | Maximum crawl depth |
| respect_robots_txt | Boolean | Follow robots.txt rules |
| follow_external_links | Boolean | Follow links to other domains |
| rate_limit_delay | Integer | Delay between requests (ms) |
| user_agent | String | User agent to use |
| max_content_length | Integer | Max content size to store |
| created_at | DateTime | Creation timestamp |
| enable_intelligent_navigation | Boolean | Enable intelligent navigation |
| quality_threshold | Float | Quality score threshold for extending depth |
| max_extended_depth | Integer | Max levels beyond standard depth |
| enable_classification | Boolean | Enable content classification |
| enable_entity_extraction | Boolean | Enable entity extraction |
| intelligence_domain | String | Domain for intelligence processing |
| intelligence_config | JSON | Additional intelligence configuration |

### ScrapingJob

Tracks individual scraping jobs.

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| config_id | Integer | FK to ScrapingConfiguration |
| status | String | Job status |
| start_time | DateTime | Job start time |
| end_time | DateTime | Job end time |
| total_urls | Integer | Total URLs processed |
| successful_urls | Integer | Successfully processed URLs |
| failed_urls | Integer | Failed URLs |
| crawl_log | Text | Summary log |

### ScrapedContent

Stores extracted content.

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| job_id | Integer | FK to ScrapingJob |
| url | String | Content URL |
| title | String | Page title |
| content_text | Text | Extracted text content |
| content_html | Text | Raw HTML (if stored) |
| crawl_depth | Integer | Depth in the crawl |
| crawl_timestamp | DateTime | Time of extraction |
| content_hash | String | Content hash for deduplication |
| word_count | Integer | Number of words |
| language | String | Detected language |
| parent_url | String | Parent URL |
| metadata | JSON | Additional metadata |

## Intelligence Tables

### ContentClassification

Stores classification results for content.

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

Stores extracted entities.

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
| entity_metadata | JSON | Additional entity data |
| created_at | DateTime | Creation timestamp |

### SourceCredibility

Stores credibility scores for content sources.

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| source_domain | String | Source domain |
| credibility_score | Float | 0-1 credibility score |
| evaluation_method | String | How score was determined |
| last_updated | DateTime | Last update timestamp |
| metadata | JSON | Additional metadata |

## Database Migrations

When making changes to the database schema:

1. Create a migration script in `db_migrations/` directory
2. Test migrations both forward and backward
3. Update this documentation to reflect changes

Example migration script structure:

```python
from app import db
from models import ScrapingConfiguration

def migrate():
    # Add new column
    db.engine.execute("ALTER TABLE scraping_configuration ADD COLUMN enable_intelligence BOOLEAN DEFAULT FALSE")
    
    # Update existing records
    db.engine.execute("UPDATE scraping_configuration SET enable_intelligence = FALSE")
    
    print("Migration completed successfully")

if __name__ == "__main__":
    migrate()
```

## Performance Considerations

- Use appropriate indexes for frequently queried columns
- Consider partitioning for large tables (especially ScrapedContent)
- Use batch operations for bulk inserts
- Keep HTML content storage optional for large-scale crawling
- Implement regular database maintenance procedures

## Database Configuration

Database connection is configured through the `DATABASE_URL` environment variable or in the `.env` file:

```
# PostgreSQL (recommended for production)
DATABASE_URL=postgresql://username:password@localhost:5432/textharvester

# SQLite (simple option for development)
# Leave DATABASE_URL unset to use SQLite
```

## ORM Usage

TextHarvester uses SQLAlchemy ORM for database operations. Example usage:

```python
from app import db
from models import ScrapedContent, ScrapingJob

# Query with relationships
contents = ScrapedContent.query.filter(
    ScrapedContent.job_id == ScrapingJob.id,
    ScrapingJob.status == 'completed'
).limit(10).all()

# Batch insert
db.session.bulk_save_objects([
    ScrapedContent(job_id=job_id, url=url, title=title, content_text=text)
    for url, title, text in content_items
])
db.session.commit()
```
