# TextHarvester Changelog

All notable changes to the TextHarvester project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Completed implementation of intelligence features
- Added normalize_text and extract_keywords functions to text processing
- Added get_model_path and get_embeddings functions to model utilities
- Created configuration settings for model caching
- Resolved SQLAlchemy reserved attribute name conflicts
- Added mock model system for testing without ML models
- Implemented missing database and taxonomy structures
- Full documentation for intelligence features in README_INTELLIGENCE.md

### Fixed
- Resolved endpoint conflict in routes.py with duplicate view_job_content declarations
- Fixed ContentEntity model to use entity_metadata instead of metadata (SQLAlchemy reserved name)
- Added intelligence blueprint registration in app.py
- Updated database schema to include required intelligence columns
- Fixed classification pipeline to return "General News" with 0.7 confidence by default
- Added create_default_for_domain method to ClassificationResult for better defaults
- Fixed entity extraction by adding general_patterns.jsonl for general domain
- Improved database error handling in intelligence_integration module
- Fixed import path issues in various intelligence modules

## [1.0.0] - 2024-05-08

### Added
- Core web scraping functionality
- Multi-threaded domain processing
- Content extraction and cleaning
- Intelligent navigation based on content quality
- Source management system
- Job monitoring and control
- Export functionality for scraped content
- Admin interface for configuration and monitoring
- Documentation for core components

### Intelligence Features
- Base pipeline architecture for content analysis
- Classification pipeline for categorizing content
- Entity extraction pipeline for named entity recognition
- Football domain-specific processing

### Rust Extractor
- High-performance content extraction in Rust
- API server for integration with Python
- Text cleaning and normalization
- Metadata extraction

## [0.9.0] - 2024-03-15

### Added
- Initial version of TextHarvester
- Basic web scraping functionality
- Simple content extraction
- SQLite database backend
- Command-line interface
- Basic job management

### Changed
- Improved error handling in crawler
- Enhanced content extraction with fallback mechanisms
- Better rate limiting implementation

### Fixed
- Issues with URL parsing and normalization
- Memory leaks during large crawling jobs
- Database connection handling problems

## [0.8.0] - 2024-02-01

### Added
- First prototype of the system
- Simple web crawler implementation
- Basic Flask web interface
- Minimal database models
- Testing infrastructure

## Project Milestones

This section outlines the major milestones in the TextHarvester project development.

### Milestone 1: Core Scraping Framework (Completed)
- Basic web crawling functionality
- Content extraction
- Database storage
- Simple interface

### Milestone 2: Enhanced Scraping Capabilities (Completed)
- Intelligent navigation
- Multi-threaded processing
- Advanced content extraction
- Job management

### Milestone 3: Intelligence Integration (Current)
- Classification pipeline
- Entity extraction
- Intelligence configuration
- Result visualization

### Milestone 4: Advanced Intelligence (Planned)
- Multiple domain support
- Knowledge graph integration
- Relationship extraction
- Advanced entity linking

### Milestone 5: Enterprise Features (Planned)
- User management
- Role-based access control
- API enhancements
- Advanced analytics

## Feature Matrix

This table shows the evolution of key features across major versions:

| Feature                   | 0.8.0 | 0.9.0 | 1.0.0 | Unreleased |
|---------------------------|-------|-------|-------|------------|
| Basic web crawling        | ✅     | ✅     | ✅     | ✅          |
| Content extraction        | ✅     | ✅     | ✅     | ✅          |
| Multi-threaded crawling   | ❌     | ✅     | ✅     | ✅          |
| Intelligent navigation    | ❌     | ❌     | ✅     | ✅          |
| Source management         | ❌     | ❌     | ✅     | ✅          |
| Rust extractor            | ❌     | ❌     | ✅     | ✅          |
| Content classification    | ❌     | ❌     | ❌     | ✅          |
| Entity extraction         | ❌     | ❌     | ❌     | ✅          |
| Domain-specific analysis  | ❌     | ❌     | ❌     | ✅          |
| Knowledge graph           | ❌     | ❌     | ❌     | ❌          |
| Multi-domain support      | ❌     | ❌     | ❌     | ❌          |

## Development Notes

### Version 1.0.0
- First production-ready release
- Complete rewrite of the crawler engine
- Added Rust-based content extractor
- Significantly improved performance and reliability
- Enhanced web interface

### Version 0.9.0
- Major architectural improvements
- Added multi-threading support
- Improved content extraction
- Enhanced error handling
- Began transition to PostgreSQL database

### Version 0.8.0
- Initial prototype
- Single-threaded crawler
- Basic content extraction
- SQLite database
- Minimal functionality
