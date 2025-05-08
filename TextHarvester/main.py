import os
import sys
import logging
import threading
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")

# Check if required packages are available
try:
    from app import app
except ImportError as e:
    logger.error(f"ImportError: {e}")
    logger.error("Flask or other required dependencies not found")
    logger.error("Please run: pip install -r requirements.txt")
    sys.exit(1)

def start_rust_extractor():
    """Start the Rust extractor."""
    try:
        # Add project root to path
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Import the starter script
        from start_rust_extractor import start_rust_server, is_server_running
        
        if is_server_running():
            logger.info("Rust extractor server is already running.")
            return True
        
        logger.info("Starting Rust extractor server...")
        server_process = start_rust_server()
        
        if server_process:
            logger.info(f"Rust extractor server started with PID {server_process.pid}")
            return True
        else:
            logger.warning("Failed to start Rust extractor. Will use Python extraction.")
            return False
    except Exception as e:
        logger.error(f"Error setting up Rust extractor: {e}")
        return False


def init_intelligence_components():
    """Initialize the intelligence components."""
    try:
        # Add project root to path
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Pre-initialize classification pipeline for faster first requests
        try:
            # Import in a way that doesn't use signal in a thread
            from intelligence.classification.pipeline import ClassificationPipeline
            logger.info("Initializing classification pipeline...")
            ClassificationPipeline.create_football_pipeline()
            logger.info("Classification pipeline initialized successfully.")
        except Exception as e:
            logger.warning(f"Could not initialize classification pipeline: {e}")
            logger.warning("The application will still work, but classification features may be limited.")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing intelligence components: {e}")
        return False


# Add a route to initialize components on demand
@app.route('/init-intelligence', methods=['GET'])
def initialize_components_route():
    result = init_intelligence_components()
    if result:
        return {"status": "success", "message": "Intelligence components initialized"}
    else:
        return {"status": "error", "message": "Failed to initialize intelligence components"}, 500


if __name__ == "__main__":
    # Make sure data directory exists
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    logger.info(f"Ensuring data directory exists: {data_dir}")

    # Set environment variable to use Python extraction if not already set
    if "USE_PYTHON_EXTRACTION" not in os.environ:
        os.environ["USE_PYTHON_EXTRACTION"] = "1"
    logger.info(f"Using Python extraction: {os.environ.get('USE_PYTHON_EXTRACTION') == '1'}")
    
    # Only start Rust extractor if we're not using Python extraction
    if os.environ.get("USE_PYTHON_EXTRACTION") != "1":
        logger.info("Attempting to start Rust extractor...")
        start_rust_extractor()
    else:
        logger.info("Using Python extraction only, skipping Rust extractor startup")
    
    # Start the Flask application
    logger.info("Starting TextHarvester application...")
    app.run(host="0.0.0.0", port=5000, debug=True)
