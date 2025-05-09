# TextHarvester Installation Guide

This guide provides comprehensive instructions for setting up and running the TextHarvester system with its core components.

## System Requirements

- **Python**: 3.11 or higher
- **Database**: PostgreSQL 14+ (recommended) or SQLite (for development only)
- **Rust**: 1.65+ (optional, for high-performance content extraction)
- **Operating System**: Windows, macOS, or Linux

## Installation Options

There are two main ways to install TextHarvester:

1. **Automated Setup**: Using the setup script (recommended for most users)
2. **Manual Setup**: For advanced users who want more control

## Option 1: Automated Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd TextHarvester
   ```

2. **Run the Setup Script**
   ```bash
   python setup.py
   ```
   
   This script will:
   - Install required Python dependencies
   - Check for optional Rust installation
   - Set up database with default configuration
   - Create necessary directories and files

3. **Run the System**
   ```bash
   python run_text_harvester.py
   ```

   This will start both the content extractor and the main application.

4. **Access the Web Interface**
   
   Open your browser and navigate to: http://localhost:5000

## Option 2: Manual Setup

If you prefer more control over the installation process:

### 1. Python Environment

1. **Create a Virtual Environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### 2. Database Setup

#### PostgreSQL (Recommended for Production)

1. **Install PostgreSQL** if not already installed

2. **Create a Database**
   ```bash
   createdb textharvester
   ```

3. **Configure Connection**
   
   Create a `.env` file in the TextHarvester directory:
   ```
   DATABASE_URL=postgresql://username:password@localhost:5432/textharvester
   SESSION_SECRET=your_secret_key_here
   ```
   Replace `username` and `password` with your PostgreSQL credentials.

#### SQLite (Simple Option for Development)

For local development, you can use SQLite without additional configuration:
   ```
   # In .env file
   # Leave DATABASE_URL unset to use SQLite
   SESSION_SECRET=your_secret_key_here
   ```

The application will create a SQLite database in the `data` directory.

### 3. Initialize Database

```bash
# Create tables
python -c "from app import app, db; app.app_context().push(); db.create_all()"

# Run intelligence migrations if needed
python db_migrations/add_intelligence_tables.py
```

### 4. Rust Extractor (Optional but Recommended)

The Rust extractor provides 5-10x better performance than the Python-based alternative.

1. **Install Rust** using [rustup](https://rustup.rs/) if not already installed

2. **Build the Extractor**
   ```bash
   cd rust_extractor
   cargo build --release
   ```

### 5. Starting Components Manually

1. **Start Rust Extractor** (if using it)
   ```bash
   python start_rust_extractor.py
   # Or on Windows
   start_rust_extractor.bat
   ```

2. **Start Main Application**
   ```bash
   python TextHarvester/main.py
   # Or on Windows
   start_textharvester.bat
   ```

## Configuration Options

TextHarvester can be configured through environment variables or the `.env` file:

### Core Settings

- `DATABASE_URL`: Database connection string (PostgreSQL)
- `SESSION_SECRET`: Secret key for session security
- `TEXTHARVESTER_ENV`: Environment (`development`, `production`, `testing`)
- `PORT`: Web server port (default: 5000)

### Extraction Settings

- `USE_PYTHON_EXTRACTION=1`: Force Python extraction even if Rust is available
- `RUST_EXTRACTOR_URL`: URL for Rust extractor if running as a separate service
- `STORE_RAW_HTML=0`: Disable storage of raw HTML to save database space

### Intelligence Settings

- `ENABLE_INTELLIGENCE=0`: Disable intelligence features by default
- `DEFAULT_INTELLIGENCE_DOMAIN=football`: Default domain for intelligence processing

## Troubleshooting

### Database Issues

If you encounter database errors:
1. Ensure the database exists and is accessible
2. Check file permissions for SQLite database
3. Verify connection string in `.env` file
4. For PostgreSQL, confirm user has proper privileges

### Rust Extractor Issues

If Rust extraction fails:
1. Ensure Rust and Cargo are properly installed
2. Try building the extractor manually: `cd rust_extractor && cargo build --release`
3. Check logs in `logs/rust_extractor.log`
4. Set `USE_PYTHON_EXTRACTION=1` in `.env` to use Python extraction as a fallback

### Python Dependency Issues

If Python packages are missing:
1. Ensure you're using Python 3.11+
2. Try `pip install -r requirements.txt --force-reinstall`
3. Check for version conflicts with `pip check`

### Connection Issues

If you can't access the web interface:
1. Confirm the application is running (check console output)
2. Verify the port isn't in use by another application
3. Try accessing `http://127.0.0.1:5000` instead of `localhost`
4. Check firewall settings if accessing from another machine

## Next Steps

After installation, refer to the main [README.md](README.md) for usage instructions, including:

1. Creating source lists
2. Configuring scraping jobs
3. Running and monitoring scraping operations
4. Exporting and using collected data

For development, see [DEVELOPMENT.md](DEVELOPMENT.md) for coding standards and workflow.
