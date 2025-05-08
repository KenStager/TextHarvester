# NLP Corpus Builder with Content Intelligence

A comprehensive web scraping and content intelligence platform for collecting, analyzing, and structuring domain-specific text data. This versatile system helps you gather high-quality text data for training NER (Named Entity Recognition) and text classification models across any field - with a special focus on Premier League football as the demonstration domain.

## Overview

The NLP Corpus Builder is a domain-agnostic tool that enables researchers, data scientists, and AI engineers to efficiently gather and analyze high-quality text data from specific sources. The platform streamlines the entire process from source management to content extraction, classification, and knowledge extraction.

The system is designed to be memory-efficient, handling large-scale crawling jobs with thousands of pages while providing robust error handling and resource management. All content is automatically cleaned, normalized, and prepared for NER/SpanCat annotation and topic classification.

## Architecture

The system follows a modular architecture with these key components:

```
┌─────────────────┐     ┌──────────────────────────────────────┐     ┌───────────────────┐
│                 │     │         TOPIC-SPECIFIC               │     │                   │
│  NLP CORPUS     │     │      CONTENT INTELLIGENCE            │     │     CONTENT       │
│   BUILDER       │──►  │              LAYER                   │──►  │    GENERATION     │
│  (EXISTING)     │     │                                      │     │    FRAMEWORK      │
└─────────────────┘     └──────────────────────────────────────┘     └───────────────────┘
                                        │
                                        ▼
                        ┌──────────────────────────────────────┐
                        │           KNOWLEDGE                  │
                        │          MANAGEMENT                  │
                        │            SYSTEM                    │
                        └──────────────────────────────────────┘
```

## Key Features

### Core Web Scraper
- **Advanced source management system**
  - Organize sources into categorized lists
  - Source validation and testing
  - Predefined source lists for common domains
  - Import/export of source configurations

- **High-performance parallel web crawler**
  - Multi-threaded domain processing
  - Configurable crawl depth and link following
  - Rate limiting and robots.txt compliance
  - User agent rotation and request throttling
  - Robust error handling and retry mechanisms

- **Intelligent content extraction**
  - Precise main content detection with trafilatura
  - Rust-based high-performance extraction (optional)
  - Fallback extraction mechanisms
  - Advanced text cleaning and normalization
  - Metadata extraction and storage

- **Comprehensive job management**
  - Real-time job monitoring and analytics
  - Pausable/stoppable background jobs
  - Detailed logging and diagnostics
  - Job comparison and trend analysis

- **Memory-efficient data export**
  - Streaming export for large datasets
  - Chunking algorithms for NER/SpanCat training
  - Customizable chunk sizes and overlaps
  - Export with or without original HTML
  - JSONL format compatible with annotation tools

### Content Intelligence Layer

- **Topic Classification System** (Implemented)
  - Hierarchical topic taxonomy management
  - Fast filtering for rapid content relevance screening
  - Multiple classifier implementations (SVM, LR, RF)
  - Football-specific taxonomy with detailed categories
  - Prodigy integration for annotation and training

- **Named Entity Recognition** (Implemented)
  - Custom entity type definitions with inheritance
  - Specialized football entity taxonomy
  - Domain-specific entity recognition models
  - Entity linking to knowledge base
  - Relationship extraction between entities

- **Knowledge Management System** (In Development)
  - Knowledge graph representation
  - Entity and relationship storage
  - Contradiction detection
  - Source credibility assessment
  - Querying and visualization tools

## Current Implementation Status

- ✅ **Core Web Scraper**: Fully implemented with high-performance extraction
- ✅ **Database Foundation**: Schema and ORM models implemented
- ✅ **Core Framework**: Configuration, utilities, and base pipeline
- ✅ **Topic Classification**: Complete football-specific implementation
- ✅ **Entity Recognition**: Complete football-specific implementation
- 🔄 **Rust Integration**: Optional high-speed content extraction engine
- 🚧 **Knowledge Management**: In development
- 🔜 **Content Generation**: Planned for future development

## Project Structure

```
TextHarvester/
├── README.md
├── requirements.txt
├── setup.py
├── run_text_harvester.py
├── TextHarvester/
│   ├── app.py
│   ├── main.py
│   ├── models.py
│   ├── data/         # Database and local storage
│   ├── static/       # Web interface assets
│   ├── templates/    # HTML templates
│   ├── scraper/      # Web scraper components
│   │   ├── content_extractor.py
│   │   ├── crawler.py
│   │   ├── export.py
│   │   ├── rust_integration.py
│   │   ├── source_lists.py
│   │   └── utils.py
│   ├── rust_extractor/  # Rust-based extractor (optional)
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   └── README.md
│   └── api/          # API endpoints
│       ├── routes.py
│       └── sources.py
├── intelligence/     # Content intelligence components
│   ├── base_pipeline.py
│   ├── config.py
│   ├── __init__.py
│   ├── classification/  # Topic classification
│   │   ├── classifiers.py
│   │   ├── fast_filter.py
│   │   ├── pipeline.py
│   │   ├── topic_taxonomy.py
│   │   └── taxonomies/
│   │       └── football.py
│   ├── entities/     # Entity recognition
│   │   ├── entity_types.py
│   │   ├── linking.py
│   │   ├── ner_model.py
│   │   ├── pipeline.py
│   │   └── taxonomies/
│   │       └── football_entities.py
│   ├── knowledge/    # Knowledge management (in development)
│   │   └── ...
│   └── utils/        # Utility functions
│       ├── model_utils.py
│       ├── prodigy_integration.py
│       └── text_processing.py
├── db/              # Database models and migrations
│   └── models/
│       ├── topic_taxonomy.py
│       ├── entity_models.py
│       └── content_intelligence.py
├── prodigy/         # Prodigy annotation recipes
│   └── recipes/
│       ├── domain_ner.py
│       └── topic_classification.py
└── tests/           # Test cases
    └── integration/
        └── test_rust_classification.py
```

## Installation

### Prerequisites

- Python 3.9+
- PostgreSQL database (optional, SQLite works for development)
- Rust (optional, for high-performance extraction)

### Quick Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd TextHarvester
```

2. Run the setup script:
```bash
python setup.py
```

3. Run the system:
```bash
python run_text_harvester.py
```

4. Access the web interface:
Open your browser and go to http://localhost:5000

### Manual Setup

1. Set up Python virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the database:
- Copy `.env.example` to `.env` in the TextHarvester directory
- Edit `.env` to configure database connection (SQLite is used by default)

4. Run the application:
```bash
python run_text_harvester.py
```

## Usage

### Web Interface

The web interface provides comprehensive management of all scraping and intelligence functions:

1. **Source Management**:
   - Create and organize sources into lists
   - Test source extraction quality
   - Import/export source configurations

2. **Scraping Jobs**:
   - Configure job parameters (depth, rate limits, etc.)
   - Monitor job progress in real-time
   - View performance metrics and statistics
   - Export results in various formats

3. **Content Intelligence**:
   - View topic classification for extracted content
   - Explore entity recognition results
   - Browse knowledge graph (when implemented)

### API Usage

The system provides a RESTful API for integration with other applications:

```python
import requests

# Start a new scraping job
job_config = {
    "source_list_id": 1,
    "max_depth": 2,
    "follow_links": True,
    "rate_limit": 5  # requests per second
}
response = requests.post("http://localhost:5000/api/jobs", json=job_config)
job_id = response.json()["job_id"]

# Get job status
job_status = requests.get(f"http://localhost:5000/api/jobs/{job_id}").json()

# Export job results
export_config = {
    "job_id": job_id,
    "format": "jsonl",
    "include_html": False,
    "chunking": {
        "enabled": True,
        "size": 500,
        "overlap": 50
    }
}
export_url = requests.post("http://localhost:5000/api/export", json=export_config).json()["url"]
```

## Football Domain Implementation

As a demonstration domain, the system includes specialized components for Premier League football:

### Topic Classification Taxonomy
- Complete Premier League team hierarchy (all 20 teams)
- Player categories (goalkeepers, defenders, midfielders, forwards)
- Matches, transfers, competitions
- Statistics, venues, finances, media coverage

### Entity Recognition
- Team detection with nickname handling
- Player identification
- Match event extraction
- Competition and venue recognition

## Customization

The system is designed to be easily adaptable to new domains:

1. **Create a new taxonomy**:
```python
from intelligence.classification.topic_taxonomy import TopicTaxonomy, TopicNode

# Create root taxonomy
my_taxonomy = TopicTaxonomy(
    name="my_domain",
    description="My custom domain taxonomy"
)

# Create main category
main_category = TopicNode(
    name="Main Category", 
    keywords=["keyword1", "keyword2"]
)
my_taxonomy.add_root_node(main_category)

# Add subcategories
subcategory = TopicNode(
    name="Subcategory",
    keywords=["specific1", "specific2"]
)
main_category.add_child(subcategory)

# Save to database
my_taxonomy.save_to_database()
```

2. **Create domain-specific entity types**:
```python
from intelligence.entities.entity_types import EntityType

# Create custom entity types
DOMAIN_ENTITY_TYPES = {
    "CUSTOM_ENTITY": {
        "subtypes": ["SUBTYPE1", "SUBTYPE2"],
        "attributes": ["attribute1", "attribute2"]
    }
}
```

3. **Implement custom pipelines**:
```python
from intelligence.classification.pipeline import ClassificationPipeline
from intelligence.entities.pipeline import EntityExtractionPipeline

# Create custom pipelines
my_classification_pipeline = ClassificationPipeline(
    taxonomy=my_taxonomy,
    confidence_threshold=0.6,
    domain_name="my_domain"
)

my_entity_pipeline = EntityExtractionPipeline(
    entity_types=DOMAIN_ENTITY_TYPES,
    domain_name="my_domain"
)
```

## Requirements

- Python 3.9+
- PostgreSQL database (optional, SQLite works for development)
- Required Python packages (installed by setup.py):
  - flask & flask-sqlalchemy: Web framework and ORM
  - beautifulsoup4: HTML parsing
  - trafilatura: Main content extraction
  - scikit-learn: Machine learning models
  - requests: HTTP client
  - python-dotenv: Environment variable management

## Optional Components

- **Rust Extractor**: High-performance content extraction (requires Rust installation)
- **Prodigy Integration**: For training custom models (requires Prodigy license)
- **PostgreSQL**: For production deployment (SQLite works for development)

## Common Use Cases

- **Academic Research**: Collect domain-specific corpora for research in linguistics, NLP, or domain-specific studies
- **AI Training Data**: Build custom datasets for training specialized language models or NER systems
- **Market Intelligence**: Monitor specific sources for competitive intelligence and trend analysis
- **Knowledge Base Construction**: Gather domain knowledge for expert systems or semantic databases
- **Educational Content**: Collect learning materials on specific topics for educational applications
- **Media Monitoring**: Track publications across multiple sources for specific topics or entities

## Troubleshooting

### Database Issues
- Check that the data directory exists and has proper permissions
- For SQLite: Delete any corrupted database file
- For PostgreSQL: Verify connection parameters in .env file

### Rust Extractor Issues
- Ensure Rust and Cargo are installed
- Check that the Rust extractor binary is properly built
- Set USE_PYTHON_EXTRACTION=1 in .env to force Python extraction

### Web Interface Issues
- Clear browser cache and cookies
- Check for JavaScript console errors
- Verify that all static assets are loading

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Trafilatura for content extraction
- Flask for web framework
- scikit-learn for machine learning models
- Prodigy for annotation integration
- Rust for high-performance content extraction (optional component)
