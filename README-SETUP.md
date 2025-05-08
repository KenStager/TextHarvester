# TextHarvester Setup Guide

This guide will help you set up and run the TextHarvester system with its Topic Classification components.

## Prerequisites

- Python 3.9+ installed
- (Optional) Rust and Cargo installed for high-performance content extraction
- Required Python packages (installed automatically by setup.py)

## Quick Start

1. **Run the setup script**:
   ```
   python setup.py
   ```
   This will install all required Python dependencies and check for Rust installation.

2. **Run the system**:
   ```
   python run_text_harvester.py
   ```
   This will start both the content extractor and the main application.

3. **Access the web interface**:
   Open your browser and navigate to: http://localhost:5000

## Manual Setup

If you prefer to set up components manually:

1. **Install Python dependencies**:
   ```
   pip install -r requirements.txt
   ```

2. **Configure the database**:
   - Copy `.env.example` to `.env` in the TextHarvester directory
   - Edit `.env` to configure database connection (SQLite is used by default)

3. **Run the components separately**:
   - Start Rust extractor: `python start_rust_extractor.py`
   - Start main application: `python TextHarvester/main.py`

## Configuration Options

- **Database**: Set `DATABASE_URL` in `.env` for PostgreSQL connection
- **Extraction Engine**: Set `USE_PYTHON_EXTRACTION=1` in `.env` to force Python extraction
- **Session Security**: Change `SESSION_SECRET` in `.env` for production use

## Troubleshooting

### Database Issues

If you encounter database errors:
1. Ensure the data directory exists in the TextHarvester folder
2. Check file permissions for SQLite database
3. Clear any corrupt database files: `rm TextHarvester/data/web_scraper.db`

### Rust Extractor Issues

If Rust extraction fails:
1. Make sure Rust and Cargo are installed: https://rustup.rs
2. Try building the extractor manually: `cd rust_extractor && cargo build --release`
3. Force Python extraction: Set `USE_PYTHON_EXTRACTION=1` in `.env`

### Python Dependency Issues

If Python packages are missing:
1. Run `pip install -r requirements.txt`
2. Check for version conflicts with `pip check`

## Topic Classification System

The Topic Classification System is fully integrated and will classify content as it is extracted. The system includes:

1. **Topic Taxonomy**: Hierarchical structure of topics in the football domain
2. **Fast Filtering**: Quick keyword-based relevance detection
3. **Classifiers**: Machine learning models for detailed classification
4. **Integration with Prodigy**: Tools for training and improving classifiers

To train new classification models, use the Prodigy recipes:
```
python -m prodigy textcat.football-topics football_dataset
```

## Development Notes

- The system will use Python-based extraction if Rust is not available
- Debug mode is enabled by default for development
- The SQLite database is created in the TextHarvester/data directory
