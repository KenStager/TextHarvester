# TextHarvester Architecture Guide

This document describes the high-level architecture of the TextHarvester system, designed to serve as a reference for developers and LLMs working on the project. It outlines the core components, their relationships, and the design principles underlying the system.

## System Overview

TextHarvester is a comprehensive web scraping and content intelligence platform designed for collecting, processing, and analyzing textual data from the web. The system consists of several key components:

1. **Web Scraper**: Core engine for crawling websites and extracting content
2. **Content Processor**: Cleans and normalizes extracted text
3. **Intelligence Engine**: Analyzes content using classification and entity extraction
4. **Database Layer**: Stores configurations, content, and intelligence results
5. **Web Interface**: Provides user management and visualization of results

## Architecture Diagram

```
┌───────────────────────────────────────────────────────────────────┐
│                      TextHarvester System                          │
│                                                                   │
│  ┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐ │
│  │             │     │                 │     │                  │ │
│  │ Web Scraper ├────►│ Content         ├────►│ Intelligence     │ │
│  │             │     │ Processor       │     │ Engine           │ │
│  └──────┬──────┘     └─────────┬───────┘     └──────────┬───────┘ │
│         │                      │                        │         │
│         │                      │                        │         │
│         ▼                      ▼                        ▼         │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                       Database Layer                         │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                               ▲                                    │
│                               │                                    │
│                               ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                        Web Interface                         │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Web Scraper

The web scraper is responsible for crawling websites, fetching HTML content, and managing the crawling process. It's designed to be highly parallel, efficient, and respectful of website policies.

**Key Files:**
- `scraper/crawler.py`: Main crawler class with parallel domain processing
- `scraper/path_intelligence.py`: Intelligent navigation and link scoring
- `scraper/content_extractor.py`: HTML content extraction

**Design Principles:**
- Domain-based parallelism for efficient crawling
- Intelligent depth management based on content quality
- Robust error handling and retry mechanisms
- Rate limiting and robots.txt compliance

### 2. Content Processor

The content processor cleans and normalizes the extracted text, preparing it for storage and analysis. It handles various text formats and encoding issues.

**Key Files:**
- `scraper/content_extractor.py`: Extraction and initial cleaning
- `rust_extractor/src/extractors/text_cleaner.rs`: Advanced text cleaning (Rust implementation)

**Design Principles:**
- Clean text while preserving important formatting
- Handle encoding issues gracefully
- Extract the main content, excluding boilerplate and navigation
- Provide fallback mechanisms when primary extraction fails

### 3. Intelligence Engine

The intelligence engine analyzes the extracted content, classifying it into topics and extracting named entities. It's designed to be modular, with support for different domains and analysis types.

**Key Files:**
- `intelligence/classification/pipeline.py`: Content classification
- `intelligence/entities/pipeline.py`: Entity extraction
- `intelligence/base_pipeline.py`: Base pipeline infrastructure
- `scraper/intelligence_integration.py`: Integration with scraper

**Design Principles:**
- Domain-specific processing (e.g., football)
- Pipeline-based architecture for composable processing
- Lazy loading to minimize resource usage
- Fault tolerance to prevent analysis failures from affecting crawling

### 4. Database Layer

The database layer stores configurations, content, and analysis results. It's designed to be efficient for both storage and retrieval, with support for large volumes of data.

**Key Files:**
- `models.py`: Core database models
- `models_update.py`: Intelligence-related models
- `db_migrations/`: Database migration scripts

**Design Principles:**
- Clear separation of concerns in model design
- Efficient indexing for frequently queried fields
- Support for structured (SQL) and semi-structured (JSON) data
- Transaction safety for concurrent operations

### 5. Web Interface

The web interface provides user-friendly access to the system, including configuration, job management, and result visualization.

**Key Files:**
- `api/routes.py`: Main API routes
- `api/sources.py`: Source management
- `api/intelligence.py`: Intelligence-related routes
- `templates/`: HTML templates

**Design Principles:**
- Clean separation of backend and frontend
- RESTful API design
- Responsive UI for various device sizes
- Real-time updates for job status

## Integration Points

The system has several key integration points:

1. **Scraper ↔ Content Processor**: After HTML is fetched, content is extracted and cleaned
2. **Content Processor ↔ Intelligence Engine**: Cleaned content is analyzed for classification and entities
3. **Intelligence Engine ↔ Database**: Analysis results are stored alongside content
4. **Database ↔ Web Interface**: UI displays content, analysis, and allows configuration

## Data Flow

1. **Configuration**: User configures scraping job via web interface
2. **Crawling**: Scraper crawls websites based on configuration
3. **Extraction**: Content processor extracts and cleans text
4. **Analysis**: Intelligence engine analyzes content (if enabled)
5. **Storage**: Results stored in database
6. **Presentation**: Web interface displays results

## Extension Points

The system is designed to be extensible in several ways:

1. **New Intelligence Features**: Add new analysis types via the pipeline architecture
2. **Custom Extractors**: Implement specialized extractors for different content types
3. **Additional Domains**: Add domain-specific processing for new areas
4. **Export Formats**: Implement new exporters for different downstream systems
5. **Source Types**: Add support for new source types beyond web pages

## Cross-Cutting Concerns

Several concerns are handled across the entire system:

1. **Logging**: Consistent logging with appropriate levels
2. **Error Handling**: Graceful degradation on failures
3. **Performance Optimization**: Efficient resource usage
4. **Security**: Input validation and output sanitization
5. **Concurrency**: Thread-safe operations for parallel processing

## Development Guidelines

1. **Modularity**: Keep components focused on single responsibilities
2. **Testability**: Write code that can be tested in isolation
3. **Configuration over Code**: Make behaviors configurable when possible
4. **Progressive Enhancement**: Add advanced features while maintaining core functionality
5. **Documentation**: Document code, APIs, and architecture decisions

## Future Directions

1. **Machine Learning Enhancements**: More advanced content analysis
2. **Distributed Crawling**: Scale to multiple machines
3. **Real-time Processing**: Stream processing of content
4. **Knowledge Graph Integration**: Connect entities to a knowledge graph
5. **Multi-modal Content**: Extract and analyze images and other media

This architecture guide serves as a high-level overview of the TextHarvester system. For more detailed information on specific components, refer to the respective README files and code documentation.
