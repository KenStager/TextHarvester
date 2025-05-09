"""
Create mockup models for testing the intelligence features.

This script creates simple placeholder models to allow the intelligence
pipelines to run without requiring the full models to be downloaded.
"""

import os
import sys
import logging
import pickle
import json
from pathlib import Path

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("create_mockup_models")

from intelligence.utils.model_utils import get_model_path

# List of models to create
MODELS_TO_CREATE = [
    {"domain": "general", "type": "classification", "name": "distilbert_general"},
    {"domain": "football", "type": "classification", "name": "distilbert_football"},
    {"domain": "general", "type": "ner", "name": "bert_ner"},
    {"domain": "football", "type": "ner", "name": "football_entities"},
    {"domain": "general", "type": "embedding", "name": "all-MiniLM-L6-v2"},
    {"domain": "football", "type": "embedding", "name": "football-embeddings"}
]

# Create simple mockup tokenizer
class MockTokenizer:
    """Simple mockup tokenizer for testing."""
    
    def __init__(self, name="test_tokenizer"):
        self.name = name
        
    def tokenize(self, text):
        """Tokenize text into simple word tokens."""
        return text.split()
    
    def __call__(self, texts, **kwargs):
        """Process texts for a model."""
        if isinstance(texts, str):
            texts = [texts]
            
        return {
            "input_ids": [[1, 2, 3, 4, 5]] * len(texts),
            "attention_mask": [[1, 1, 1, 1, 1]] * len(texts)
        }
    
    def save_pretrained(self, path):
        """Save tokenizer to path."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
            json.dump({"name": self.name, "type": "mock"}, f)
        logger.info(f"Saved mock tokenizer to {path}")

# Create simple mockup model
class MockModel:
    """Simple mockup model for testing."""
    
    def __init__(self, name="test_model", model_type="classification"):
        self.name = name
        self.type = model_type
        
    def __call__(self, **kwargs):
        """Process inputs to generate outputs."""
        return MockOutput()
    
    def save_pretrained(self, path):
        """Save model to path."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({"name": self.name, "type": self.type}, f)
        logger.info(f"Saved mock model to {path}")

# Simple mockup output for models
class MockOutput:
    """Simple mockup model output for testing."""
    
    def __init__(self):
        self.logits = [[0.1, 0.2, 0.7]]
        self.last_hidden_state = [[0.1, 0.2, 0.3]]

def create_mockup_model(domain, model_type, model_name):
    """Create a mockup model and tokenizer."""
    # Get model path
    model_path = get_model_path(model_name, domain, model_type)
    model_dir = os.path.dirname(model_path)
    
    # Create model
    model = MockModel(name=model_name, model_type=model_type)
    tokenizer = MockTokenizer(name=f"{model_name}_tokenizer")
    
    # Save model and tokenizer
    model_save_path = os.path.join(model_dir, model_name)
    tokenizer_save_path = os.path.join(model_dir, f"{model_name}_tokenizer")
    
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(tokenizer_save_path, exist_ok=True)
    
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)
    
    # Create additional files for specific model types
    if model_type == "classification":
        # Create label map
        labels = ["football", "technology", "business", "sports", "news"]
        with open(os.path.join(model_dir, f"{model_name}_labels.json"), "w") as f:
            json.dump(labels, f)
    
    elif model_type == "ner":
        # Create entity types
        entity_types = ["PERSON", "ORG", "LOC", "MISC", "TEAM", "PLAYER", "COACH"]
        with open(os.path.join(model_dir, f"{model_name}_entities.json"), "w") as f:
            json.dump(entity_types, f)
    
    logger.info(f"Created mockup {domain}/{model_type}/{model_name}")
    return model_path

def main():
    """Create all mockup models."""
    logger.info("Creating mockup models for testing...")
    
    created_models = []
    
    for model_info in MODELS_TO_CREATE:
        try:
            model_path = create_mockup_model(
                model_info["domain"], 
                model_info["type"], 
                model_info["name"]
            )
            created_models.append(model_path)
        except Exception as e:
            logger.error(f"Error creating mockup model: {str(e)}")
    
    logger.info(f"Created {len(created_models)} mockup models:")
    for model_path in created_models:
        logger.info(f"- {model_path}")
    
    return len(created_models) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
