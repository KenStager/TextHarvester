# TextHarvester Changelog

All notable changes to the TextHarvester project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Intelligence integration for content analysis
- New database models for classification and entity storage
- Configuration options for intelligence features
- API endpoints for managing intelligence features
- UI for intelligence configuration
- Comprehensive Intelligence Dashboard with visualization
- Content classification visualization with confidence scores
- Entity extraction visualization with filtering and highlighting
- Integration of intelligence indicators in content views
- Enhanced job statistics with intelligence metrics
- Documentation for intelligence visualization

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
