# Contributing to TextHarvester

Thank you for your interest in contributing to TextHarvester! This document provides guidelines and instructions for contributing to the project, both for human developers and language models (LLMs) assisting with development.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [LLM-Assisted Development](#llm-assisted-development)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Documentation Guidelines](#documentation-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Community](#community)

## Code of Conduct

This project adheres to a code of conduct that ensures an open and welcoming environment for all contributors. Key principles include:

- Being respectful and inclusive
- Focusing on constructive feedback
- Prioritizing the community's needs
- Maintaining a harassment-free environment

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Git
- Optionally: Rust 1.65+ (for the Rust extractor component)

### Setting Up Development Environment

1. **Fork the Repository**
   
   First, fork the repository to your own GitHub account.

2. **Clone Your Fork**
   ```bash
   git clone https://github.com/YOUR-USERNAME/TextHarvester.git
   cd TextHarvester
   ```

3. **Set Up Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Set Up Database**
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

5. **Build Rust Extractor (Optional)**
   ```bash
   cd rust_extractor
   cargo build
   ```

6. **Run the Application**
   ```bash
   python main.py
   ```

## Development Workflow

### Branches

- `main`: Main development branch, should always be in a working state
- `feature/feature-name`: For new features
- `bugfix/bug-name`: For bug fixes
- `docs/description`: For documentation updates
- `refactor/description`: For code refactoring
- `test/description`: For adding or updating tests

### Workflow Steps

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   
   Develop your feature, fix, or improvement.

3. **Write Tests**
   
   Add tests for your changes. Ensure all tests pass.

4. **Update Documentation**
   
   Update relevant documentation, including:
   - Code comments and docstrings
   - README or other markdown files
   - API documentation

5. **Commit Changes**
   
   Make focused, descriptive commits:
   ```bash
   git add .
   git commit -m "Add feature X that does Y"
   ```

6. **Push Changes**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create Pull Request**
   
   Open a pull request to the main repository with a clear description of your changes.

## LLM-Assisted Development

TextHarvester welcomes contributions assisted by Language Models (LLMs). When working with LLMs:

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
   - Follow the architecture principles outlined in documentation

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

4. **Reference architectural documentation**
   - ARCHITECTURE.md: For overall system design
   - DEVELOPMENT-GUIDE.md: For specific guidelines
   - README files: For component-specific information

## Pull Request Process

1. **PR Description**
   
   Include a clear description of the changes, the problem they solve, and any relevant issue numbers.

2. **PR Template**
   
   Follow the pull request template, which includes:
   - Purpose of the changes
   - Implementation details
   - Testing performed
   - Documentation updates
   - Breaking changes or dependencies

3. **Code Review**
   
   All PRs require at least one code review from a maintainer or approved contributor.

4. **CI/CD**
   
   Ensure all automated tests and checks pass.

5. **Approval and Merging**
   
   After approval, a maintainer will merge your PR into the main branch.

## Coding Standards

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with these adjustments:
  - Line length: 100 characters maximum
  - Use 4 spaces for indentation
  - Use double quotes for strings unless single quotes avoid escaping

- Use type hints for function parameters and return values
- Use descriptive variable names
- Follow docstring format as shown in existing code

### Rust Code Style

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` for code formatting
- Run `cargo clippy` for linting
- Add documentation comments for public APIs

### General Guidelines

- Keep functions small and focused
- Follow the Single Responsibility Principle
- Write testable code
- Handle errors explicitly
- Log important information at appropriate levels

## Documentation Guidelines

Documentation is as important as code in TextHarvester. Follow these guidelines:

### Code Documentation

- **Docstrings**: All public modules, classes, and functions must have docstrings
- **Format**: Use Google-style docstrings
- **Content**: Document parameters, return values, exceptions, and examples

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

### Project Documentation

- **README files**: Keep them current with changes
- **Architecture documentation**: Update when changing system design
- **Diagrams**: Use plain text diagrams when possible (ASCII, Mermaid)
- **Examples**: Update examples when changing APIs
- **Changelog**: Document significant changes

## Testing Guidelines

### Test Coverage

- Aim for at least 80% code coverage
- Critical paths should have 100% coverage
- Write both unit and integration tests

### Test Types

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **Functional Tests**: Test end-to-end functionality
- **Performance Tests**: Test resource usage and time efficiency

### Writing Tests

- Test one concept per test function
- Use descriptive test names
- Arrange-Act-Assert pattern
- Use fixtures and mocks appropriately
- Include both positive and negative test cases

## Community

### Communication Channels

- GitHub Issues: For bug reports and feature requests
- Discussions: For questions and general discussion
- Pull Requests: For code contributions

### Getting Help

If you need help with your contribution:

1. Check the documentation first
2. Search existing issues and discussions
3. Open a new discussion if needed

### Recognition

All contributors will be recognized in the project's contributors list. We value every contribution, whether it's code, documentation, tests, or feedback.

## Future Contributions

We particularly welcome contributions in these areas:

- Additional intelligence features
- Performance improvements
- New domain support
- Testing and validation
- Documentation enhancements
- UI improvements

Thank you for contributing to TextHarvester! Your efforts help improve the project for everyone.
