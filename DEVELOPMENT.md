# TextHarvester Development Guide

This comprehensive guide provides guidelines for developing and contributing to the TextHarvester project, covering setup, workflow, coding standards, and best practices.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Environment Setup](#development-environment-setup)
- [Development Workflow](#development-workflow)
- [LLM-Assisted Development](#llm-assisted-development)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Working with Components](#working-with-components)
- [Database Management](#database-management)
- [Performance Considerations](#performance-considerations)
- [Documentation Guidelines](#documentation-guidelines)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Contributing to the Project](#contributing-to-the-project)

## Code of Conduct

This project adheres to a code of conduct that ensures an open and welcoming environment for all contributors:

- Be respectful and inclusive in all interactions
- Focus on constructive feedback
- Prioritize the community's needs
- Maintain a harassment-free environment
- Value all types of contributions (code, documentation, testing, feedback)

## Development Environment Setup

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Git
- Rust 1.65+ (optional, for the high-performance extractor)

### First-Time Setup

1. **Fork and Clone the Repository**
   
   ```bash
   git clone https://github.com/YOUR-USERNAME/TextHarvester.git
   cd TextHarvester
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Set Up the Database**
   ```bash
   # Create a PostgreSQL database
   createdb textharvester_dev
   
   # Create a .env file with database configuration
   echo "DATABASE_URL=postgresql://username:password@localhost:5432/textharvester_dev" > .env
   echo "SESSION_SECRET=dev_secret_key" >> .env
   echo "TEXTHARVESTER_ENV=development" >> .env
   
   # Initialize database schema
   python -c "from app import app, db; app.app_context().push(); db.create_all()"
   ```

5. **Build Rust Extractor (Optional but Recommended)**
   ```bash
   cd rust_extractor
   cargo build --release
   ```

### Starting the Application

1. **Run the Flask Application**
   ```bash
   python main.py
   ```

2. **Start the Rust Extractor API** (if using it separately)
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

## Development Workflow

### Branching Strategy

- `main`: Main development branch, should always be in a working state
- `feature/feature-name`: For new features
- `bugfix/bug-name`: For bug fixes
- `docs/description`: For documentation updates
- `refactor/description`: For code refactoring
- `test/description`: For adding or updating tests

### Feature Development Process

1. **Issue Creation**
   - Create a detailed issue describing the feature or bug
   - Include acceptance criteria and technical details

2. **Branch Creation**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Development**
   - Write tests for your changes
   - Implement the feature following coding standards
   - Add or update documentation
   - Run tests locally to verify functionality

4. **Commit Changes**
   - Make focused, descriptive commits
   ```bash
   git add .
   git commit -m "Add feature X that does Y"
   ```

5. **Push Changes**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**
   - Open a PR with a clear description of your changes
   - Reference any related issues
   - Fill out the PR template completely

7. **Code Review**
   - Address feedback from reviewers
   - Make necessary adjustments
   - Ensure all tests pass

8. **Merge**
   - After approval, your changes will be merged into the main branch

## LLM-Assisted Development

TextHarvester welcomes contributions assisted by Language Models (LLMs):

### For Humans Working with LLMs

1. **Be specific in your prompts**
   - Include relevant context and constraints
   - Refer to specific files and functions
   - Explain the problem or feature clearly

2. **Review LLM output carefully**
   - Validate logic and ensure it meets requirements
   - Check for potential issues or edge cases
   - Ensure adherence to project standards

3. **Attribute appropriately**
   - Mention LLM assistance in commit messages when appropriate
   - Take responsibility for the final code quality

### For LLMs Assisting Development

1. **Follow project conventions**
   - Adhere to the coding style and patterns in existing code
   - Use consistent naming and documentation formats
   - Follow the architecture principles in documentation

2. **Prioritize these aspects in generated code**
   - Correctness: Code should function as intended
   - Readability: Clear, well-documented code
   - Maintainability: Follow good design practices
   - Security: Follow secure coding practices
   - Performance: Efficient implementation

3. **Include appropriate context**
   - Add docstrings explaining purpose and usage
   - Comment complex logic or algorithms
   - Reference related code or documentation

## Coding Standards

### Python Style Guidelines

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with these adjustments:
  - Line length: 100 characters maximum
  - Use 4 spaces for indentation
  - Use double quotes for strings unless single quotes avoid escaping

- **Type Hints**: Use type hints for all function parameters and return values
- **Naming**: Use descriptive variable and function names
  - Use `snake_case` for functions and variables
  - Use `PascalCase` for classes
- **Docstrings**: All public modules, classes, and functions must have docstrings

Example:
```python
def process_url(url: str, depth: int = 0) -> Tuple[bool, List[str]]:
    """
    Process a URL to extract content and links.
    
    Args:
        url: The URL to process
        depth: Current crawl depth, defaults to 0
        
    Returns:
        A tuple of (success, extracted_links)
        
    Raises:
        ValueError: If the URL is invalid
    """
```

### Rust Style Guidelines

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` for code formatting
- Run `cargo clippy` for linting
- Add documentation comments for public APIs
- Handle errors explicitly, avoid panicking in library code

### General Principles

- **Single Responsibility**: Each function and class should have one responsibility
- **Error Handling**: Handle errors explicitly, avoid silent failures
- **Testability**: Write code that can be tested in isolation
- **Documentation**: Document code, APIs, and non-obvious behaviors
- **Immutability**: Prefer immutable data structures when possible

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

3. **Functional Tests**
   - Test complete functionality from user perspective
   - Verify system works as a whole

4. **Performance Tests**
   - Test resource usage and time efficiency
   - Benchmark critical operations

### Test Coverage

- Aim for at least 80% code coverage
- Critical paths should have 100% coverage
- Write both positive and negative test cases

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_file.py

# Run tests with coverage
pytest --cov=.
```

### Writing Tests

- Test one concept per test function
- Use descriptive test names
- Follow the Arrange-Act-Assert pattern
- Use fixtures and mocks appropriately
- Include both positive and negative test cases

## Working with Components

### Intelligence Module

When working on intelligence features:

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

### Rust Extractor

The Rust extractor provides faster and more efficient content extraction:

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

## Documentation Guidelines

Documentation is as important as code in TextHarvester:

1. **Code Documentation**
   - **Docstrings**: All public modules, classes, and functions
   - **Comments**: Explain complex algorithms and non-obvious behaviors
   - **Explain why**: Focus on explaining why, not just what

2. **Project Documentation**
   - Update README files when adding new components
   - Keep architecture documentation current
   - Document API changes
   - Update examples when changing interfaces

3. **Commit Messages**
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

## Contributing to the Project

### Areas for Contribution

We particularly welcome contributions in these areas:

- Additional intelligence features
- Performance improvements
- New domain support
- Testing and validation
- Documentation enhancements
- UI improvements

### Recognition

All contributors will be recognized in the project's contributors list. We value every contribution, whether it's code, documentation, tests, or feedback.

### Getting Help

If you need help with your contribution:

1. Check the documentation first
2. Search existing issues and discussions
3. Open a new discussion if needed

---

By following these development guidelines, we ensure a consistent, high-quality codebase that can continue to evolve and improve over time. Thank you for contributing to TextHarvester!
