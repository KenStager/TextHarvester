# NLP Corpus Builder: Advanced Web Scraper for NER/SpanCat Training Data

A comprehensive, production-ready web scraping and research data collection platform designed for intelligent content extraction and AI-powered research workflows. This versatile system helps you collect domain-specific text data for training NER (Named Entity Recognition) and SpanCat models across any field or discipline - from AI research to competitive intelligence.

## Overview

The NLP Corpus Builder is a domain-agnostic tool that enables researchers, data scientists, and AI engineers to efficiently gather high-quality text data from specific sources. The platform streamlines the entire process from source management to content extraction and export in formats optimized for annotation tools like Prodigy.

The system is designed to be memory-efficient, handling large-scale crawling jobs with thousands of pages while providing robust error handling and resource management. All content is automatically cleaned, normalized, and prepared for NER/SpanCat annotation.

## Key Features

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
  - **Intelligent navigation system for adaptive crawling**

- **Intelligent content extraction**
  - Precise main content detection with trafilatura
  - High-performance Rust-based extraction engine (optional)
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

## Workflow & Use Cases

### Typical Workflow

1. **Source Management**:
   - Create and organize sources into lists based on research domains
   - Test sources for content extraction quality
   - Import from predefined source lists or add custom sources

2. **Configure Scraping Parameters**:
   - Set crawl depth, rate limiting, and other parameters
   - Choose whether to follow external links
   - Configure robots.txt compliance options
   - Benefit from automatic intelligent navigation for quality-based crawling

3. **Run Scraping Jobs**:
   - Start jobs with specific source lists
   - Monitor progress in real-time
   - Pause or stop jobs as needed
   - View detailed statistics and performance metrics

4. **Export & Use Data**:
   - Export data in JSONL format compatible with annotation tools
   - Configure chunking parameters for optimal NER training
   - Use streaming export for large datasets
   - Process data for NER/SpanCat annotation

### Common Use Cases

- **Academic Research**: Collect domain-specific corpora for research in linguistics, NLP, or domain-specific studies
- **AI Training Data**: Build custom datasets for training specialized language models or NER systems
- **Market Intelligence**: Monitor specific sources for competitive intelligence and trend analysis
- **Knowledge Base Construction**: Gather domain knowledge for expert systems or semantic databases
- **Educational Content**: Collect learning materials on specific topics for educational applications
- **Media Monitoring**: Track publications across multiple sources for specific topics or entities

## Requirements

- Python 3.11+
- PostgreSQL database
- Required Python packages:
  - beautifulsoup4: HTML parsing
  - email-validator: Input validation
  - flask & flask-sqlalchemy: Web framework and ORM
  - gunicorn: WSGI HTTP Server
  - psycopg2-binary: PostgreSQL adapter
  - python-dotenv: Environment variable management
  - requests: HTTP client
  - trafilatura: Main content extraction

## Running on Replit

This application is designed to run directly on Replit without any additional configuration:

1. Click the Run button to start the application
2. The application will automatically use the PostgreSQL database configured in Replit
3. Access the web interface via the Webview tab or the URL provided by Replit
4. Begin by adding sources or importing from predefined lists

## Local Setup (outside of Replit)

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Set up a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install beautifulsoup4 email-validator flask flask-sqlalchemy gunicorn psycopg2-binary python-dotenv requests trafilatura
```

### 4. Database Options

#### Option A: PostgreSQL (recommended for production)

Make sure PostgreSQL is installed and running on your system. Create a new database:

```bash
createdb web_scraper
```

Create a `.env` file in the project root with:

```
DATABASE_URL=postgresql://username:password@localhost:5432/web_scraper
SESSION_SECRET=your_secret_key_here
```

Replace `username` and `password` with your PostgreSQL credentials.

#### Option B: SQLite (simple option for development)

For local development, you can use SQLite without any additional configuration. The application will create a SQLite database in the `data` directory if no PostgreSQL connection is provided.

### 5. Run the application locally

```bash
python main.py
```

The application will be accessible at http://localhost:5000

## Running with Gunicorn (Production)

For production-like environments, you can use Gunicorn:

```bash
gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app
```

## Project Structure

The application follows a modular architecture with clear separation of concerns:

- **Core Application**
  - `app.py`: Database configuration and Flask app initialization
  - `main.py`: Application entry point
  - `models.py`: SQLAlchemy database models for all entities

- **API and Web Interface** (`api/` directory)
  - `routes.py`: Main route handlers for job and configuration management
  - `sources.py`: Source management API endpoints
  - `__init__.py`: Blueprint registration and API initialization

- **Scraper Engine** (`scraper/` directory)
  - `crawler.py`: Multi-threaded web crawling system with parallel domain processing
  - `content_extractor.py`: Text extraction from HTML using trafilatura
  - `rust_integration.py`: Integration with high-performance Rust extractor
  - `export.py`: Data export utilities with streaming and chunking capabilities
  - `utils.py`: Helper functions for URL processing, validation, etc.
  - `source_lists.py`: Predefined source lists for various domains

- **Rust Extractor** (`rust_extractor/` directory)
  - High-performance content extraction engine written in Rust
  - 5-10x faster than Python-based alternatives
  - Lower memory footprint for large-scale extractions
  - API server for direct integration with Python scraper

- **Web UI**
  - `static/`: JavaScript, CSS, and image assets
  - `templates/`: Jinja2 HTML templates for the web interface
  - `exports/`: Directory for exported data files

## Technical Implementation Details

### Crawler Architecture

The crawler utilizes a thread-based parallel architecture where:
- Each domain is processed in its own thread to maximize throughput
- A thread-safe registry tracks active jobs and manages graceful shutdown
- Direct SQL updates are used for critical operations to ensure transaction safety
- URL queues are managed per-domain to respect rate limiting requirements
- Intelligent navigation makes quality-based decisions for deeper, more focused crawling

#### Intelligent Navigation System

The crawler now features an advanced intelligent navigation system that:
- **Adaptively adjusts crawl depth** based on content quality metrics rather than using a uniform maximum depth
- **Tracks parent-child relationships** between URLs to provide context for navigation decisions
- **Evaluates page quality** using metrics like word count, paragraph density, and text-to-HTML ratio
- **Makes dynamic depth decisions** based on parent page quality, domain averages, and link scoring
- **Optimizes resource usage** by extending depth only for high-value content paths

This system automatically identifies and prioritizes high-quality content sources, exploring them more deeply while limiting exploration of low-value paths. See the detailed documentation in `docs/intelligent_navigation.md`.

### Content Extraction

Content extraction uses a multi-layer approach:
1. High-performance Rust extractor when available for maximum performance
2. Trafilatura as the primary Python-based extractor with full-document context
3. Fallback to BeautifulSoup-based extraction if needed
4. Text normalization and cleaning for optimal NER training
5. Metadata extraction for additional context

#### Rust vs Python Extraction

The system supports two extraction engines:

- **Python-based Extraction** (Default)
  - Uses trafilatura for high-quality content extraction
  - Good balance between accuracy and performance
  - No additional dependencies beyond Python packages

- **Rust-based Extraction** (Optional, Recommended for Production)
  - 5-10x faster processing than Python-based extraction
  - 50-70% less memory consumption
  - Perfect for high-volume crawling jobs
  - Compatible with the same Python API interface
  - Can be run as standalone API server or direct CLI tool

### Export System

The export system is designed for memory efficiency with large datasets:
1. Streaming response for browser-based downloads without timeout issues
2. Batched database queries to avoid memory pressure
3. Text chunking algorithms that respect sentence boundaries
4. JSONL format generation with comprehensive metadata
5. UTF-8 sanitization to ensure compatibility with annotation tools

## Data Export Format

The application exports data in JSONL format with each line containing a complete JSON object with:

```json
{
  "text": "Chunked text content ready for annotation",
  "meta": {
    "source": "web_scrape",
    "url": "Original source URL",
    "title": "Page title",
    "date": "ISO timestamp",
    "job_id": 123,
    "content_id": 456,
    "config_name": "Configuration name",
    "crawl_depth": 1,
    "chunk_index": 0,
    "chunk_total": 5,
    "language": "en",
    "content_type": "text/html",
    "word_count": 250
  }
}
```

This format is directly compatible with Prodigy and other annotation tools for NER and SpanCat model training.

## Limitations & Considerations

- The system respects robots.txt by default but can be configured to bypass restrictions
- Rate limiting is domain-based to prevent overloading source servers
- For very large crawls (>10,000 pages), consider using multiple smaller jobs
- HTML content can significantly increase database size; consider disabling raw HTML storage for large projects
- External URL following should be used cautiously to prevent unbounded crawls
- The intelligent navigation system extends crawl depth for high-quality content but is capped at a maximum of `max_depth + 2`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas for future development include:

- Additional content extractors for specialized content types
- Enhanced intelligent navigation with machine learning models
- Further refinement of quality metrics and scoring algorithms
- Integration with vector databases for similarity search
- Annotation workflow integration
- Content deduplication algorithms

## License

This project is licensed under the MIT License - see the LICENSE file for details.