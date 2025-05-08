
import sys
import importlib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("extractor_check")

def check_module(module_name):
    try:
        importlib.import_module(module_name)
        logger.info(f"✓ {module_name} is available")
        return True
    except ImportError as e:
        logger.error(f"✗ {module_name} is missing: {e}")
        return False

def main():
    logger.info("Checking required modules for Python-based extraction")
    
    required_modules = [
        "trafilatura",  # Main content extraction library
        "bs4",         # BeautifulSoup for HTML parsing
        "requests",    # HTTP client
        "psycopg2",    # PostgreSQL adapter
        "flask",       # Web framework
        "sqlalchemy"   # ORM
    ]
    
    all_available = True
    for module in required_modules:
        if not check_module(module):
            all_available = False
    
    if all_available:
        logger.info("All required modules for Python-based extraction are available!")
        logger.info("You can run the application with Python extraction enabled.")
    else:
        logger.error("Some required modules are missing.")
        logger.error("Run: pip install -r requirements.txt")
    
    # Check if we can import from the local codebase
    try:
        from scraper.content_extractor import extract_content
        logger.info("✓ Python content extractor is importable")
    except ImportError as e:
        logger.error(f"✗ Python content extractor import failed: {e}")
        all_available = False
    
    return all_available

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
