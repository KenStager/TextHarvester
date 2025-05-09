"""
Run mockup model creation to set up intelligence testing.

This script executes the mockup model creation and 
ensures all required directories exist for the intelligence features.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create intelligence data directory if it doesn't exist
intelligence_data_dir = os.path.join('intelligence', 'data')
os.makedirs(intelligence_data_dir, exist_ok=True)

# Create intelligence cache directory if it doesn't exist  
intelligence_cache_dir = os.path.join('intelligence', 'cache')
os.makedirs(intelligence_cache_dir, exist_ok=True)

# Create model directories
model_dirs = [
    os.path.join('intelligence', 'cache', 'general', 'classification'),
    os.path.join('intelligence', 'cache', 'general', 'ner'),
    os.path.join('intelligence', 'cache', 'football', 'classification'),
    os.path.join('intelligence', 'cache', 'football', 'ner')
]

for dir_path in model_dirs:
    os.makedirs(dir_path, exist_ok=True)
    print(f"Created directory: {dir_path}")

# Run the mockup model creation script
try:
    from intelligence.create_mockup_models import main as create_models
    success = create_models()
    
    if success:
        print("Successfully created mock models for testing")
    else:
        print("Failed to create some mock models")
except Exception as e:
    print(f"Error creating mock models: {str(e)}")

print("\nSetup complete. You can now run tests/test_intelligence.py to test the intelligence features.")
