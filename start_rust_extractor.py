#!/usr/bin/env python
"""
Script to start the Rust extractor server.

This script ensures the Rust extractor is compiled and running
before starting the main application, providing high-performance
content extraction capabilities to the TextHarvester.
"""

import os
import sys
import time
import subprocess
import logging
import signal
import atexit
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rust_extractor_starter")

# Constants
RUST_DIR = Path(__file__).parent / "rust_extractor"
RUST_BINARY_DIR = RUST_DIR / "target" / "release"
RUST_BINARY_NAME = "rust_extractor"
if sys.platform == "win32":
    RUST_BINARY_NAME += ".exe"
RUST_BINARY_PATH = RUST_BINARY_DIR / RUST_BINARY_NAME
RUST_HOST = "127.0.0.1"  # Use localhost for security
RUST_PORT = 8888


def is_binary_available():
    """Check if the Rust extractor binary exists."""
    return RUST_BINARY_PATH.exists()


def build_rust_extractor():
    """Build the Rust extractor if not already built."""
    logger.info("Building Rust extractor...")
    
    # Skip if USE_PYTHON_EXTRACTION is set
    if os.environ.get("USE_PYTHON_EXTRACTION") == "1":
        logger.info("Skipping Rust build as USE_PYTHON_EXTRACTION is enabled.")
        return False
    
    # Check if Cargo is available
    try:
        subprocess.run(["cargo", "--version"], check=True, stdout=subprocess.PIPE)
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("Rust's Cargo not found. Please install Rust to build the extractor.")
        os.environ["USE_PYTHON_EXTRACTION"] = "1"  # Set flag to use Python extraction
        return False
    
    # Build the extractor
    try:
        build_cmd = ["cargo", "build", "--release"]
        result = subprocess.run(
            build_cmd,
            cwd=RUST_DIR,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info("Rust extractor built successfully.")
        return True
    except subprocess.SubprocessError as e:
        logger.error(f"Failed to build Rust extractor: {e}")
        logger.error(f"Build output: {e.stderr if hasattr(e, 'stderr') else 'No output'}")
        os.environ["USE_PYTHON_EXTRACTION"] = "1"  # Set flag to use Python extraction
        return False


def start_rust_server():
    """Start the Rust extractor server."""
    # Skip if USE_PYTHON_EXTRACTION is set
    if os.environ.get("USE_PYTHON_EXTRACTION") == "1":
        logger.info("Using Python extraction as requested. Rust extractor will not be started.")
        return None
        
    if not is_binary_available():
        if not build_rust_extractor():
            logger.warning("Could not build Rust extractor. Application will use Python extraction.")
            os.environ["USE_PYTHON_EXTRACTION"] = "1"  # Set flag to use Python extraction
            return None
    
    logger.info(f"Starting Rust extractor server on {RUST_HOST}:{RUST_PORT}...")
    
    # Start the server
    try:
        server_cmd = [
            str(RUST_BINARY_PATH),
            "server",
            "--host", RUST_HOST,
            "--port", str(RUST_PORT)
        ]
        
        # Start the server process
        server_process = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Register cleanup to terminate the server when the script exits
        def cleanup():
            if server_process.poll() is None:
                logger.info("Stopping Rust extractor server...")
                try:
                    if sys.platform == "win32":
                        server_process.terminate()
                    else:
                        server_process.send_signal(signal.SIGTERM)
                    server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("Rust server did not terminate gracefully, forcing...")
                    server_process.kill()
        
        atexit.register(cleanup)
        
        # Wait for server to start
        time.sleep(2)
        
        # Check if server started successfully
        if server_process.poll() is not None:
            # Server exited prematurely
            stdout, stderr = server_process.communicate()
            logger.error(f"Rust server failed to start: {stderr}")
            return None
        
        logger.info("Rust extractor server started successfully.")
        return server_process
    
    except Exception as e:
        logger.error(f"Error starting Rust extractor server: {e}")
        return None


def is_server_running():
    """Check if the Rust extractor server is running."""
    import socket
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = False
    
    try:
        result = sock.connect_ex((RUST_HOST, RUST_PORT)) == 0
    finally:
        sock.close()
        
    return result


if __name__ == "__main__":
    # Check if server is already running
    if is_server_running():
        logger.info("Rust extractor server is already running.")
    else:
        server_process = start_rust_server()
        
        if server_process:
            logger.info(f"Rust extractor server running with PID {server_process.pid}")
            
            # Keep the script running to maintain the server
            try:
                while server_process.poll() is None:
                    time.sleep(1)
                
                # If we get here, the server exited
                stdout, stderr = server_process.communicate()
                exit_code = server_process.returncode
                logger.error(f"Rust server exited with code {exit_code}: {stderr}")
            
            except KeyboardInterrupt:
                logger.info("Shutting down Rust extractor server...")
    
    # If script is imported, we just provide the functions
    # and don't run the server automatically
