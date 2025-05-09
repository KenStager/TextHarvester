# TextHarvester: Web Scraping & Content Intelligence Platform

TextHarvester is a web scraping platform designed to collect, process, and analyze text data from across the web. Originally developed as a dedicated web scraper, the system is now evolving to incorporate intelligence features for content analysis.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## Project Status

- **Web Scraping**: Production-ready with high-performance, parallel crawling
- **Content Processing**: Stable with both Python and Rust-based extraction
- **Intelligence Features**: Under active development - basic classification and entity extraction are implemented

## Key Features

### Web Scraping Engine
- **Parallel Domain Processing**: Multi-threaded crawling optimized for large-scale data collection
- **Intelligent Navigation**: Quality-based crawling that automatically extends depth for high-value content
- **Robust Management**: Rate limiting, retry mechanisms, and job control
- **Source Management**: Organize sources into categorized lists with validation and testing

### Content Processing
- **High-Performance Extraction**: Rust-based extractor with Python fallback
- **Quality Assessment**: Automatic evaluation of content quality and relevance
- **Advanced Cleaning**: Multi-stage text normalization and cleaning

### Intelligence Engine (In Development)
- **Content Classification**: Categorize content by topic with confidence scores
- **Entity Extraction**: Identify and extract named entities with contextual information
- **Domain-Specific Analysis**: Specialized processing for different knowledge domains
- **Pipeline Architecture**: Modular, extensible analysis workflows

### Data Management
- **Efficient Storage**: Optimized database models for content and analysis results
- **Streaming Export**: Memory-efficient export for large datasets
- **Flexible Formats**: Support for various export formats including JSONL for annotation

### Web Interface
- **Interactive Dashboard**: Real-time monitoring of crawling jobs
- **Content Exploration**: Browse, search, and analyze collected content
- **Configuration Management**: Create and manage crawling configurations
- **Intelligence Dashboard**: Comprehensive visualization of classification and entity extraction results
- **Content Intelligence**: View topic classifications and extracted entities for individual content items

## System Architecture

TextHarvester follows a modular architecture with clear separation of concerns:

```
┌───────────────┐     ┌─────────────────┐     ┌──────────────────┐
│               │     │                 │     │                  │
│ Web Scraper   ├────►│ Content         ├────►│ Intelligence     │
│               │     │ Processor       │     │ Engine           │
└───────┬───────┘     └─────────┬───────┘     └──────────┬───────┘
        │                       │                        │
        │                       │                        │
        ▼                       ▼                        ▼
┌───────────────────────────────────────────────────────────────┐
│                       Database Layer                           │
└───────────────────────────────────────────────────────────────┘
                               ▲
                               │
                               ▼
┌───────────────────────────────────────────────────────────────┐
│                        Web Interface                           │
└───────────────────────────────────────────────────────────────┘
```

For a more detailed architecture overview, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Use Cases

- **Research Data Collection**: Gather corpus data for NLP/ML research
- **Media Monitoring**: Track coverage across multiple sources  
- **Competitive Intelligence**: Analyze industry trends and competitor content
- **Content Aggregation**: Build specialized knowledge bases
- **Training Data Creation**: Generate labeled datasets for machine learning

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL
- Optionally: Rust for the high-performance extractor

### Installation

For quick setup:

```bash
# Clone repository
git clone <repository-url>
cd TextHarvester

# Run setup
python setup.py

# Start application
python run_text_harvester.py
```

Then access the web interface at: http://localhost:5000

For detailed installation instructions, see [INSTALLATION.md](INSTALLATION.md).

### First Scraping Job

1. **Create a Source List**
   - Navigate to Sources → New Source List
   - Add sources manually or import from predefined lists

2. **Create a Scraping Configuration**
   - Set crawl depth, rate limiting, and other parameters
   - Enable intelligence features if desired

3. **Start the Job**
   - Click "Start Job" on the configuration page
   - Monitor progress in real-time on the status page

4. **View Intelligence Results**
   - Navigate to the Intelligence dashboard to see classification and entity extraction results
   - View detailed classifications and entities for individual content items

5. **Export Results**
   - Once complete, export data in JSONL format
   - Use for analysis, annotation, or other downstream tasks

## Project Structure

- `api/`: API routes and controllers
- `db_migrations/`: Database migration scripts
- `intelligence/`: Content analysis components
- `rust_extractor/`: High-performance content extraction
- `scraper/`: Core web scraping functionality
- `static/`: Static assets for web UI
- `templates/`: HTML templates
- `tests/`: Test suite

## Documentation

### Core Documentation
- [INSTALLATION.md](INSTALLATION.md): Comprehensive setup instructions
- [ARCHITECTURE.md](ARCHITECTURE.md): Detailed system architecture and design
- [DEVELOPMENT.md](DEVELOPMENT.md): Development workflow and contribution guidelines
- [CHANGELOG.md](CHANGELOG.md): Version history and feature updates
- [INTELLIGENCE-ROADMAP.md](INTELLIGENCE-ROADMAP.md): Current state and future plans for intelligence features

### Technical Documentation
- [docs/components/rust_extractor.md](docs/components/rust_extractor.md): Rust content extraction engine
- [docs/features/intelligent_navigation.md](docs/features/intelligent_navigation.md): Quality-based crawling system
- [docs/database.md](docs/database.md): Database schema and design
- [docs/dashboard/intelligence_visualization.md](docs/dashboard/intelligence_visualization.md): Intelligence dashboard and visualization

## Components

### Rust Extractor

The Rust-based content extractor provides significant performance improvements:

- **5-10x faster** processing than Python-based alternatives
- **50-70% less memory** consumption
- **High accuracy** content extraction
- **Robust handling** of different HTML formats

### Intelligent Navigation

The crawler includes an intelligent navigation system that:
- Adaptively adjusts crawl depth based on content quality
- Tracks parent-child relationships between URLs
- Evaluates page quality using metrics like content length and structure
- Optimizes resource usage by focusing on high-value content paths

### Intelligence Features (In Development)

Current intelligence capabilities include:
- Basic content classification into predefined topics
- Named entity recognition for people, organizations, locations, etc.
- Initial framework for domain-specific analysis
- Pipeline architecture for future expansion

## Contributing

Contributions are welcome! Please see [DEVELOPMENT.md](DEVELOPMENT.md) for guidelines.

Key areas for contribution:
- Additional content extractors
- Intelligence feature development
- Performance improvements
- Documentation and examples
- UI enhancements

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Trafilatura for Python-based content extraction
- SQLAlchemy for database ORM
- Flask for the web framework
- Various open-source NLP libraries for intelligence features