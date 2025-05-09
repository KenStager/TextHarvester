"""
Setup script for TextHarvester intelligence components.

This script installs the required dependencies for the intelligence features
and sets up the necessary data files and directories.
"""

import os
import sys
import shutil
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("setup_intelligence")

def install_dependencies():
    """Install intelligence dependencies from requirements file."""
    logger.info("Installing intelligence dependencies...")
    
    try:
        # Install from requirements file
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", 
            "requirements-intelligence.txt"
        ])
        logger.info("Successfully installed intelligence dependencies")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {str(e)}")
        return False

def setup_spacy_models():
    """Download and set up spaCy models."""
    logger.info("Setting up spaCy models...")
    
    try:
        # Download small English model
        logger.info("Downloading spaCy en_core_web_sm model...")
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ])
        
        # Download medium English model
        logger.info("Downloading spaCy en_core_web_md model...")
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", "en_core_web_md"
        ])
        
        logger.info("Successfully set up spaCy models")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to set up spaCy models: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error setting up spaCy models: {str(e)}")
        return False

def setup_nltk_data():
    """Download and set up NLTK data."""
    logger.info("Setting up NLTK data...")
    
    try:
        import nltk
        
        # Download required NLTK data
        logger.info("Downloading NLTK data...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        logger.info("Successfully set up NLTK data")
        return True
    except Exception as e:
        logger.error(f"Error setting up NLTK data: {str(e)}")
        return False

def setup_data_directories():
    """Set up data directories for intelligence components."""
    logger.info("Setting up data directories...")
    
    try:
        # Create directories if they don't exist
        dirs = [
            "intelligence/data",
            "intelligence/cache",
            "intelligence/cache/models",
            "intelligence/cache/tokenizers"
        ]
        
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        # Check if default taxonomy exists
        taxonomy_path = "intelligence/data/default_taxonomy.json"
        if not os.path.exists(taxonomy_path):
            logger.warning(f"Default taxonomy not found at {taxonomy_path}")
            
            # Create from template if available
            template_path = "intelligence/data/default_taxonomy.json.template"
            if os.path.exists(template_path):
                shutil.copy(template_path, taxonomy_path)
                logger.info(f"Created default taxonomy from template")
        
        logger.info("Successfully set up data directories")
        return True
    except Exception as e:
        logger.error(f"Error setting up data directories: {str(e)}")
        return False

def create_mock_models():
    """Create mock models for testing without downloads."""
    logger.info("Creating mock models for testing...")
    
    try:
        # Run the create_mockup_models script
        from intelligence.create_mockup_models import main as create_models
        success = create_models()
        
        if success:
            logger.info("Successfully created mock models for testing")
        else:
            logger.warning("Failed to create some mock models")
            
        return success
    except Exception as e:
        logger.error(f"Error creating mock models: {str(e)}")
        return False

def verify_installation():
    """Verify the intelligence installation."""
    logger.info("Verifying intelligence installation...")
    
    # Check for required modules
    modules = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("sentence_transformers", "Sentence-Transformers"),
        ("spacy", "spaCy"),
        ("nltk", "NLTK")
    ]
    
    missing_modules = []
    
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            logger.info(f"✓ {display_name} installed")
        except ImportError:
            logger.error(f"✗ {display_name} not installed")
            missing_modules.append(display_name)
    
    # Check for spaCy models
    try:
        import spacy
        models = ["en_core_web_sm", "en_core_web_md"]
        
        for model_name in models:
            try:
                spacy.load(model_name)
                logger.info(f"✓ spaCy model {model_name} installed")
            except OSError:
                logger.error(f"✗ spaCy model {model_name} not installed")
                missing_modules.append(f"spaCy model {model_name}")
    except ImportError:
        pass  # Already reported above
    
    # Check for data directories
    directories = [
        "intelligence/data",
        "intelligence/cache"
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            logger.info(f"✓ Directory {directory} exists")
        else:
            logger.error(f"✗ Directory {directory} does not exist")
            missing_modules.append(f"Directory {directory}")
    
    # Check for taxonomy file
    taxonomy_path = "intelligence/data/default_taxonomy.json"
    if os.path.exists(taxonomy_path):
        logger.info(f"✓ Default taxonomy exists")
    else:
        logger.error(f"✗ Default taxonomy not found")
        missing_modules.append("Default taxonomy")
    
    if missing_modules:
        logger.warning("Installation verification completed with issues")
        logger.warning(f"Missing components: {', '.join(missing_modules)}")
        return False
    else:
        logger.info("Installation verification completed successfully")
        return True

def main():
    """Main setup function."""
    logger.info("Starting intelligence setup...")
    
    # Install dependencies
    if not install_dependencies():
        logger.warning("Failed to install dependencies. Continuing with setup...")
    
    # Set up spaCy models
    if not setup_spacy_models():
        logger.warning("Failed to set up spaCy models. Continuing with setup...")
    
    # Set up NLTK data
    if not setup_nltk_data():
        logger.warning("Failed to set up NLTK data. Continuing with setup...")
    
    # Set up data directories
    if not setup_data_directories():
        logger.error("Failed to set up data directories. Aborting setup.")
        return False
        
    # Create mock models for testing
    if not create_mock_models():
        logger.warning("Failed to create mock models. Intelligence features may not work correctly.")
        # Continue anyway
    
    # Verify installation
    verification_result = verify_installation()
    
    if verification_result:
        logger.info("Intelligence setup completed successfully")
    else:
        logger.warning("Intelligence setup completed with issues")
    
    return verification_result

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
