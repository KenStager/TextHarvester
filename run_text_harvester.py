#!/usr/bin/env python
"""
Convenience script to run the complete TextHarvester system.

This script checks for required dependencies, starts the Rust extractor when available,
and runs the main application with proper error handling.
"""

import os
import sys
import subprocess
import logging
import time
import signal
import atexit
import importlib.util
import pkg_resources
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("system_runner")

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent
RUST_STARTER = PROJECT_ROOT / "start_rust_extractor.py"
MAIN_APP = PROJECT_ROOT / "TextHarvester" / "main.py"


def check_python_dependencies():
    """Check if required Python packages are installed."""
    logger.info("Checking Python dependencies...")
    requirements_path = PROJECT_ROOT / "requirements.txt"
    missing = []

    if requirements_path.exists():
        with open(requirements_path, 'r') as f:
            for line in f:
                # Skip empty lines or comments
                if not line.strip() or line.strip().startswith('#'):
                    continue
                    
                # Extract package name from requirement line
                package = line.split('==')[0].split('>=')[0].split('>')[0].strip()
                
                try:
                    pkg_resources.get_distribution(package)
                except pkg_resources.DistributionNotFound:
                    missing.append(package)
    else:
        logger.warning("requirements.txt not found, skipping dependency check")
    
    return missing


def install_dependencies(missing_packages):
    """Install missing Python dependencies."""
    if not missing_packages:
        return True
        
    logger.info(f"Installing missing dependencies: {', '.join(missing_packages)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False


def check_rust_available():
    """Check if Rust/Cargo is available on the system."""
    try:
        result = subprocess.run(["cargo", "--version"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True)
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def start_process(script_path, process_name):
    """Start a Python script as a subprocess."""
    logger.info(f"Starting {process_name}...")
    
    try:
        # Start the process
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )
        
        logger.info(f"{process_name} started with PID {process.pid}")
        return process
    except Exception as e:
        logger.error(f"Failed to start {process_name}: {e}")
        return None


def monitor_process(process, process_name):
    """Monitor a subprocess and log its output."""
    def read_output(stream, log_func):
        """Read from a stream and log each line."""
        for line in iter(stream.readline, ''):
            if not line:
                break
            log_func(f"[{process_name}] {line.rstrip()}")
    
    # Start threads to read stdout and stderr
    import threading
    stdout_thread = threading.Thread(
        target=read_output, 
        args=(process.stdout, logger.info)
    )
    stderr_thread = threading.Thread(
        target=read_output, 
        args=(process.stderr, logger.error)
    )
    
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()


def main():
    """Run the complete TextHarvester system."""
    processes = []
    
    # Register cleanup handler
    def cleanup():
        logger.info("Shutting down all processes...")
        for p, name in processes:
            if p and p.poll() is None:
                logger.info(f"Terminating {name}...")
                try:
                    if sys.platform == "win32":
                        p.terminate()
                    else:
                        p.send_signal(signal.SIGTERM)
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"{name} did not terminate gracefully, forcing...")
                    p.kill()
    
    atexit.register(cleanup)
    
    try:
        # Check and install Python dependencies
        missing_packages = check_python_dependencies()
        if missing_packages:
            logger.info(f"Missing packages: {', '.join(missing_packages)}")
            user_input = input("Install missing Python packages? (y/n): ").strip().lower()
            if user_input == 'y':
                if not install_dependencies(missing_packages):
                    logger.error("Failed to install dependencies. Please install them manually.")
                    return
            else:
                logger.warning("Running without required dependencies may cause errors.")
        
        # Check if Rust is available
        rust_available = check_rust_available()
        if rust_available:
            logger.info("Rust/Cargo is available. Will use Rust extractor if possible.")
            # Start Rust extractor server
            rust_process = start_process(RUST_STARTER, "Rust Extractor Server")
            if rust_process:
                processes.append((rust_process, "Rust Extractor Server"))
                monitor_process(rust_process, "Rust Extractor")
                
                # Give the Rust server time to start
                time.sleep(2)
        else:
            logger.warning("Rust/Cargo not found. Python-based extraction will be used.")
            logger.warning("For better performance, consider installing Rust: https://rustup.rs")
            # Set an environment variable to disable Rust extractor attempts
            os.environ["USE_PYTHON_EXTRACTION"] = "1"
        
        # Start main application
        app_process = start_process(MAIN_APP, "TextHarvester Application")
        if app_process:
            processes.append((app_process, "TextHarvester Application"))
            monitor_process(app_process, "TextHarvester")
        else:
            logger.error("Failed to start the TextHarvester application.")
            return
        
        # Keep running until interrupted
        logger.info("TextHarvester system started. Press Ctrl+C to stop.")
        
        # Monitor only the main application, don't restart Rust on failure
        app_restart_attempts = 0
        max_restart_attempts = 3
        
        while True:
            # Check main application
            for i, (process, name) in enumerate(processes):
                if name == "TextHarvester Application" and process.poll() is not None:
                    exit_code = process.returncode
                    logger.error(f"{name} exited with code {exit_code}")
                    
                    # Only attempt limited restarts
                    if app_restart_attempts < max_restart_attempts:
                        logger.info(f"Attempting to restart {name} (attempt {app_restart_attempts+1}/{max_restart_attempts})")
                        new_process = start_process(MAIN_APP, name)
                        
                        if new_process:
                            processes[i] = (new_process, name)
                            monitor_process(new_process, name)
                            app_restart_attempts += 1
                        else:
                            logger.error(f"Failed to restart {name}")
                            return
                    else:
                        logger.error(f"Max restart attempts ({max_restart_attempts}) reached for {name}. Exiting.")
                        return
                elif name == "Rust Extractor Server" and process.poll() is not None:
                    # If Rust extractor fails, don't restart it - fallback to Python
                    logger.warning("Rust extractor server has exited. Falling back to Python extraction.")
                    os.environ["USE_PYTHON_EXTRACTION"] = "1"
                    # Remove it from processes to monitor
                    processes.pop(i)
                    break
            
            time.sleep(2)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down...")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
