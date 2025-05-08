# TextHarvester Git Configuration

This document explains the Git configuration for the TextHarvester project, particularly the `.gitignore` setup.

## Git Structure

The project uses a two-level `.gitignore` approach:

1. **Root-level `.gitignore`** - Located at the project root, handling top-level exclusions
2. **Application-level `.gitignore`** - Located in the TextHarvester directory, handling application-specific exclusions

## Excluded File Categories

The following categories of files are excluded from Git tracking:

### Development Files

- **Python bytecode and cache**: `__pycache__/`, `*.py[cod]`, etc.
- **Editor files**: `.vscode/`, `.idea/`, etc.
- **Virtual environments**: `venv/`, `env/`, `prodigy_env/`, etc.

### Data and Outputs

- **Database files**: `*.db`, `*.sqlite`, `*.sqlite3`
- **Export files**: Files in `exports/` (except `.gitkeep`)
- **Logs**: Files in `logs/` (except `.gitkeep`)
- **Data files**: `*.csv`, `*.jsonl`, etc.
- **Intelligence outputs**: `processed_content_*/`, `*_analysis.png`, etc.

### Rust-specific Files

- **Compilation artifacts**: `rust_extractor/target/`, `rust_extractor/Cargo.lock`
- **Backup files**: `**/*.rs.bk`

### Local Configuration

- **Environment files**: `.env`, `.env.local`, etc.
- **Local settings**: `config.local.json`, `settings.local.json`

## Directory Structure

The following directories are maintained with `.gitkeep` files to ensure they exist in the repository while excluding their contents:

- `data/` - For database and persistent data storage
- `exports/` - For output files from scraping jobs
- `logs/` - For application logs

## Usage Notes

1. When adding new file types that should be excluded, update the appropriate `.gitignore` file
2. To force-add a file that matches a pattern in `.gitignore`, use `git add -f <file>`
3. Batch files (`*.bat`) are generally excluded, but `start_rust_extractor.bat` and `start_textharvester.bat` are specifically included

## Best Practices

1. **Never commit:**
   - API keys, passwords, or other credentials
   - Large data files or exports
   - Local configuration specific to your environment
   - Temporary or generated files

2. **Always commit:**
   - Source code
   - Documentation
   - Tests
   - Templates and configuration examples (with sensitive data removed)

## Maintaining Directory Structure

To add a new empty directory to the repository:

1. Create the directory: `mkdir new_directory`
2. Add a `.gitkeep` file: `touch new_directory/.gitkeep`
3. Add an exclusion pattern to `.gitignore`: `new_directory/*`
4. Add an exception for `.gitkeep`: `!new_directory/.gitkeep`
5. Commit the changes: `git add new_directory/.gitkeep .gitignore && git commit -m "Add new_directory structure"`
