# TextHarvester Development Guide

This guide provides detailed information for developers and LLMs working on the TextHarvester project, outlining coding standards, development workflow, testing procedures, and other essential aspects of the development process.

## Setting Up the Development Environment

### Prerequisites

- Python 3.11+
- Rust 1.65+ (for the Rust extractor component)
- PostgreSQL 14+
- Git

### First-Time Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd TextHarvester
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up the Database**
   - Create a PostgreSQL database named `textharvester_dev`
   - Set the environment variable: `DATABASE_URL=postgresql://username:password@localhost:5432/textharvester_dev`
   - Run database creation: `python -c "from app import app, db; app.app_context().push(); db.create_all()"`

5. **Build the Rust Extractor** (optional but recommended)
   ```bash
   cd rust_extractor
   cargo build --release
   ```

6. **Configure Environment Variables**
   Create a `.env` file in the project root with:
   ```
   DATABASE_URL=postgresql://username:password@localhost:5432/textharvester_dev
   SESSION_SECRET=your_dev_secret_key
   TEXTHARVESTER_ENV=development
   ```

### Starting the Application

1. **Run the Flask Application**
   ```bash
   python main.py
   ```

2. **Start the Rust Extractor API** (optional)
   ```bash
   cd rust_extractor
   ./target/release/rust_extractor server
   ```

## Project Structure

```
TextHarvester/
├── api/                  # API routes and controllers
├── db_migrations/        # Database migration scripts
├── intelligence/         # Intelligence components (classification, entities)
├── models/               # Database models
├── rust_extractor/       # Rust-based content extraction (optional)
├── scraper/              # Core web scraping functionality
├── static/               # Static assets for web UI
├── templates/            # HTML templates
├── tests/                # Test suite
├── app.py                # Application initialization
├── main.py               # Entry point
├── models.py             # Core database models
└── requirements.txt      # Python dependencies
```

## Coding Standards

### Python Style Guidelines

- Follow PEP 8 for code style
- Use type hints for function parameters and return values
- Write docstrings for all modules, classes, and functions
- Use meaningful variable and function names
- Limit line length to 100 characters
- Use snake_case for functions and variables, PascalCase for classes

### Rust Style Guidelines

- Follow Rust API Guidelines
- Use `rustfmt` to format code
- Add documentation comments for public functions and types
- Prefer descriptive variable names
- Handle errors explicitly, avoid panicking in library code

### General Principles

- **Single Responsibility**: Each function and class should have one responsibility
- **Error Handling**: Handle errors explicitly, avoid silent failures
- **Testability**: Write code that can be tested in isolation
- **Documentation**: Document code, APIs, and non-obvious behaviors
- **Immutability**: Prefer immutable data structures when possible

## Development Workflow

### Feature Development Process

1. **Issue Creation**
   - Create a detailed issue in the issue tracker
   - Include acceptance criteria and technical details

2. **Branch Creation**
   - Create a feature branch from `main`
   - Use naming convention: `feature/description-issue-number`

3. **Development**
   - Write tests for the new feature
   - Implement the feature
   - Follow coding standards
   - Add or update documentation

4. **Testing**
   - Run tests locally
   - Ensure all tests pass
   - Verify feature meets acceptance criteria

5. **Pull Request**
   - Create a pull request to merge into `main`
   - Fill out the PR template with details of changes
   - Address review comments and feedback

6. **Review and Merge**
   - Code review by another developer or LLM assistant
   - Final testing
   - Merge into `main`

### Working with the Intelligence Module

The intelligence module is a separate component that integrates with the scraper. When working on intelligence features:

1. **Configuration**
   - Intelligence features should be configurable via the UI
   - Default to disabled to conserve resources

2. **Error Handling**
   - Intelligence processing should never cause the main scraping task to fail
   - Use thorough error handling and graceful degradation

3. **Resource Management**
   - Load intelligence components lazily to minimize resource usage
   - Consider the impact on memory and CPU usage during parallel processing

4. **Testing**
   - Test intelligence features both in isolation and integrated with the scraper
   - Use the integration test script to verify correct functionality

## Database Management

### Model Changes

When making changes to the database models:

1. **Update Models**
   - Update the appropriate model files (`models.py`, `models_update.py`)
   - Include proper relationships and constraints

2. **Create Migrations**
   - Create a migration script in `db_migrations/`
   - Test the migration both forward and backward

3. **Document Changes**
   - Update database documentation
   - Include any special handling for existing data

### Querying Guidelines

- Use SQLAlchemy ORM for most queries
- For performance-critical operations, consider using direct SQL via `text()`
- Use appropriate indexes for frequently queried fields
- Optimize queries that operate on large datasets

## Testing

### Test Types

1. **Unit Tests**
   - Test individual functions and classes in isolation
   - Mock dependencies
   - Focus on testing logic and edge cases

2. **Integration Tests**
   - Test interactions between components
   - Use test database for database operations
   - Verify correct end-to-end behavior

3. **Manual Tests**
   - Test user interfaces
   - Verify performance with large datasets
   - Check resource usage during operations

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_file.py

# Run tests with coverage
pytest --cov=.
```

## Working with the Rust Extractor

The Rust extractor is an optional but recommended component that provides faster and more efficient content extraction. When working with it:

1. **Building**
   ```bash
   cd rust_extractor
   cargo build --release
   ```

2. **Testing**
   ```bash
   cd rust_extractor
   cargo test
   ```

3. **Integration with Python**
   - The Python scraper can use the Rust extractor via either:
     - Direct process calls
     - HTTP API calls
   - See `scraper/rust_integration.py` for details

## Debugging

### Python Debugging

- Use debug logging: `logger.debug("Detailed information")`
- For interactive debugging, use the built-in `pdb` module:
  ```python
  import pdb; pdb.set_trace()
  ```
- For more advanced debugging, consider using VS Code's debugger or PyCharm

### Rust Debugging

- Use debug builds for development: `cargo build`
- Add debug logs with the `log` crate
- Use `println!` for quick debugging (remove before committing)
- Consider using `rust-gdb` or `rust-lldb` for complex issues

## Performance Considerations

1. **Memory Management**
   - Be careful with large datasets
   - Process in batches where possible
   - Clean up resources when finished

2. **Parallel Processing**
   - Use thread pools for CPU-bound tasks
   - Consider process pools for memory-isolated tasks
   - Be aware of thread safety in shared resources

3. **Database Operations**
   - Use batch operations for multiple inserts/updates
   - Be careful with large transactions
   - Consider pagination for large result sets

4. **Resource Monitoring**
   - Monitor memory usage during crawling
   - Watch database size growth
   - Be aware of temporary file usage

## Documentation

All new code should include appropriate documentation:

1. **Code Comments**
   - Explain complex algorithms
   - Document non-obvious behaviors
   - Explain why, not just what

2. **Docstrings**
   - Include for all functions, classes, and modules
   - Document parameters, return values, and exceptions
   - Include examples for complex functions

3. **README Files**
   - Update when adding new components
   - Keep architecture documentation current
   - Document API changes

4. **Commit Messages**
   - Write clear, descriptive commit messages
   - Reference issue numbers
   - Explain the rationale for changes

## Common Issues and Solutions

### Import Errors

If you encounter import errors, especially with the intelligence module:

1. Check the Python path: `import sys; print(sys.path)`
2. Ensure the correct directories are in the path
3. If needed, add to the path: `sys.path.append('/path/to/module')`

### Database Connection Issues

If you have trouble connecting to the database:

1. Verify connection string in `.env` file
2. Ensure PostgreSQL is running
3. Check permissions for the database user
4. Try connecting with `psql` to isolate the issue

### Performance Problems

If you encounter performance issues:

1. Check database query performance with `EXPLAIN ANALYZE`
2. Profile Python code with `cProfile`
3. Consider using the Rust extractor for content processing
4. Reduce parallelism if memory usage is high

### Intelligence Module Integration

If intelligence features are not working:

1. Verify the intelligence module is in the Python path
2. Check the configuration to ensure features are enabled
3. Look for specific error messages in the logs
4. Run the integration test script for diagnostics

## Contributing to Documentation

The project documentation is as important as the code itself. When contributing:

1. Update documentation when changing code
2. Keep the architecture diagram current
3. Document known limitations and workarounds
4. Add examples for complex features
5. Use clear language, accessible to both humans and LLMs

This development guide serves as a reference for all developers working on the TextHarvester project. By following these guidelines, we ensure a consistent, high-quality codebase that can continue to evolve and improve over time.
