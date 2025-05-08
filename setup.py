#!/usr/bin/env python
"""
TextHarvester Setup Script

This script helps with the initial setup of the TextHarvester system.
It installs required Python dependencies, checks for Rust installation,
and provides guidance on how to run the system.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("setup")

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"


def check_python_version():
    """Check if the current Python version is compatible."""
    MIN_PYTHON_VERSION = (3, 9, 0)
    current_version = sys.version_info[:3]
    
    if current_version < MIN_PYTHON_VERSION:
        logger.error(f"Python {'.'.join(map(str, MIN_PYTHON_VERSION))} or higher is required.")
        logger.error(f"Current version: {sys.version.split()[0]}")
        return False
    
    logger.info(f"Python version {sys.version.split()[0]} detected (compatible).")
    return True


def install_python_dependencies():
    """Install required Python packages."""
    if not REQUIREMENTS_FILE.exists():
        logger.error(f"Requirements file not found: {REQUIREMENTS_FILE}")
        return False
    
    logger.info("Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])
        logger.info("Python dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False


def check_rust_installation():
    """Check if Rust is installed."""
    try:
        result = subprocess.run(
            ["cargo", "--version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        if result.returncode == 0:
            logger.info(f"Rust is installed: {result.stdout.strip()}")
            return True
        
        logger.warning("Rust command found but returned an error.")
        return False
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("Rust not found. Python-based extraction will be used.")
        logger.info("For better performance, consider installing Rust: https://rustup.rs")
        return False


def print_usage_instructions():
    """Print instructions on how to run the system."""
    logger.info("\n" + "="*80)
    logger.info("TextHarvester Setup Complete!")
    logger.info("="*80)
    logger.info("\nTo run the system:")
    logger.info("  python run_text_harvester.py")
    logger.info("\nThis will start both the Rust extractor (if available) and the main application.")
    logger.info("Open your browser to http://localhost:5000 to access the web interface.")
    
    if not check_rust_installation():
        logger.info("\nNote: For improved performance, consider installing Rust:")
        logger.info("  Windows: https://www.rust-lang.org/tools/install")
        logger.info("  Linux/macOS: Run 'curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh'")
    
    logger.info("\nFor development:")
    logger.info("  1. Start the Rust extractor: python start_rust_extractor.py")
    logger.info("  2. In another terminal, run: python TextHarvester/main.py")
    
    logger.info("\n" + "="*80)


def main():
    """Main setup function."""
    logger.info("Starting TextHarvester setup...")
    
    # Check Python version
    if not check_python_version():
        logger.error("Setup aborted due to incompatible Python version.")
        return 1
    
    # Install Python dependencies
    if not install_python_dependencies():
        logger.error("Setup failed to install required Python dependencies.")
        return 1
    
    # Check for Rust installation
    check_rust_installation()
    
    # Print usage instructions
    print_usage_instructions()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
