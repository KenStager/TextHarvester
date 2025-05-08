"""
Prodigy Recipe for Domain-Specific NER Annotation.

This module provides custom Prodigy recipes for annotating entities in domain-specific
text, supporting both manual annotation and pre-annotation with existing models.
"""

import logging
import random
from typing import Dict, List, Tuple, Optional, Union, Any, Iterator, Callable
import time
from pathlib import Path

# These imports would come from actual Prodigy when installed
# For now, they're placeholders
try:
    from prodigy.components.loaders import JSONL
    from prodigy.components.db import connect
    from prodigy.util import set_hashes, log, split_string
    from prodigy import recipe
except ImportError:
    # Create placeholder classes for development without Prodigy
    def recipe(*args, **kwargs):
        """Placeholder for Prodigy recipe decorator."""
        def decorator(func):
            return func
        return decorator
        
    class JSONL:
        """Placeholder for Prodigy JSONL loader."""
        @staticmethod
        def from_path(path):
            return []
            
    def connect(*args, **kwargs):
        """Placeholder for Prodigy database connection."""
        return None
        
    def set_hashes(examples, *args, **kwargs):
        """Placeholder for Prodigy hash setter."""
        return examples
        
    def log(*args, **kwargs):
        """Placeholder for Prodigy logger."""
        logging.info(' '.join(str(arg) for arg in args))
        
    def split_string(string, *args, **kwargs):
        """Placeholder for Prodigy string splitter."""
        return string.split(',')

from intelligence.utils.prodigy_integration import (
    stream_from_scraper, prepare_ner_task,
    save_annotations_for_training
)
from intelligence.entities.ner_model import CustomNERModel, FootballNERModel
from intelligence.entities.entity_types import EntityTypeRegistry


@recipe(
    "ner.football",
    # Dataset name
    dataset=("Dataset to save annotations to", "positional", None, str),
    # Source of examples
    source=("Source file with texts (optional)", "option", "s", str),
    # Entity labels to annotate
    labels=("Comma-separated entity labels", "option", "l", split_string),
    # Database connection to read from
    db_url=("Database connection string", "option", "db", str),
    # Content limit
    limit=("Limit of content items to annotate", "option", "lim", int),
    # Use existing model for pre-annotation
    use_model=("Whether to use an existing model for pre-annotation", "flag", "m", bool),
    # Random order
    random_order=("Whether to randomize example order", "flag", "r", bool),
    # Exclude previously annotated examples
    exclude=("Dataset IDs to exclude examples from", "option", "e", split_string),
)
def football_ner(
    dataset: str,
    source: Optional[str] = None,
    labels: Optional[List[str]] = None,
    db_url: Optional[str] = None,
    limit: int = 100,
    use_model: bool = False,
    random_order: bool = False,
    exclude: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Custom recipe for annotating football entities.
    
    This recipe streams content from either a JSONL file or directly from the
    scraper database and allows annotating football-specific entity types.
    
    Args:
        dataset: Name of dataset to save annotations to
        source: Optional JSONL file with texts to annotate
        labels: Optional list of entity labels (if not provided, uses football entities)
        db_url: Optional database URL to stream content from
        limit: Maximum number of examples to annotate
        use_model: Whether to use an existing model for pre-annotation
        random_order: Whether to randomize the example order
        exclude: Datasets to exclude examples from
        
    Returns:
        Dictionary with session components for Prodigy
    """
    log(f"Loading football NER annotation with dataset '{dataset}'")
    
    # Get labels from football entity types if not provided
    if not labels:
        try:
            from intelligence.entities.taxonomies.football_entities import create_football_entity_registry
            football_registry = create_football_entity_registry()
            
            # Use top-level entity types as labels
            labels = [entity_type.name for entity_type in football_registry.root_types]
            
            # Add common subtypes
            for root_type in football_registry.root_types:
                for child in root_type.children:
                    if child.name in ["PLAYER", "TEAM", "COMPETITION", "VENUE"]:
                        labels.append(child.name)
            
        except ImportError:
            # Fallback to basic football labels
            labels = ["TEAM", "PERSON", "PLAYER", "COMPETITION", "VENUE", "EVENT"]
    
    log(f"Using labels: {', '.join(labels)}")
    
    # Set up stream of examples
    if source:
        # Load from JSONL file
        stream = JSONL(source)
        log(f"Loading examples from {source}")
    else:
        # Stream from scraper database
        stream = stream_from_scraper(db_url, limit=limit)
        log(f"Streaming examples from scraper database")
    
    # Pre-annotate with model if requested
    if use_model:
        log("Pre-annotating examples with football NER model")
        try:
            # Initialize model
            model = FootballNERModel()
            
            # Add pre-annotation to stream
            stream = _add_model_predictions(stream, model, labels)
        except Exception as e:
            log(f"Error initializing NER model: {e}")
            log("Continuing without pre-annotation")
    
    # Prepare NER examples
    stream = prepare_ner_task(stream, labels)
    
    # Add hashes and randomize if needed
    stream = set_hashes(stream, input_keys=("text",))
    
    if random_order:
        stream = list(stream)
        random.shuffle(stream)
    
    # Set up the Prodigy session
    return {
        "view_id": "ner",  # Use Prodigy's NER interface
        "dataset": dataset,
        "stream": stream,
        "update": None,  # No update function for basic NER
        "exclude": exclude or [],
        "config": {
            "labels": labels
        }
    }


@recipe(
    "ner.active-learning",
    # Dataset name
    dataset=("Dataset to save annotations to", "positional", None, str),
    # Input dataset with model predictions
    source_dataset=("Dataset with model predictions", "positional", None, str),
    # Entity labels to annotate
    labels=("Comma-separated entity labels", "option", "l", split_string),
    # Uncertainty threshold (higher means more certain examples)
    threshold=("Uncertainty threshold (0-1)", "option", "t", float),
    # Maximum examples to select
    max_examples=("Maximum examples to select", "option", "n", int),
    # Exclude previously annotated examples
    exclude=("Dataset IDs to exclude examples from", "option", "e", split_string),
)
def ner_active_learning(
    dataset: str,
    source_dataset: str,
    labels: Optional[List[str]] = None,
    threshold: float = 0.5,
    max_examples: int = 100,
    exclude: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Active learning recipe for NER annotation.
    
    This recipe selects examples with uncertain predictions for human review,
    focusing annotation effort on examples where the model is less confident.
    
    Args:
        dataset: Name of dataset to save annotations to
        source_dataset: Dataset with model predictions
        labels: Optional list of entity labels
        threshold: Uncertainty threshold (higher means more certain examples)
        max_examples: Maximum examples to select
        exclude: Datasets to exclude examples from
        
    Returns:
        Dictionary with session components for Prodigy
    """
    log(f"Starting active learning for NER with dataset '{dataset}'")
    
    # Connect to the database
    db = connect()
    if not db.exists(source_dataset):
        raise ValueError(f"Source dataset '{source_dataset}' does not exist")
    
    # Get examples from source dataset
    examples = db.get_dataset(source_dataset)
    log(f"Loaded {len(examples)} examples from source dataset")
    
    # Filter examples by uncertainty
    uncertain_examples = []
    
    for eg in examples:
        # Skip examples without spans (model predictions)
        if "spans" not in eg:
            continue
            
        # Calculate average confidence
        confidences = [span.get("score", 1.0) for span in eg.get("spans", [])]
        
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            uncertainty = 1.0 - avg_confidence
            
            # If uncertainty is above threshold, add to uncertain examples
            if uncertainty >= threshold:
                eg["_uncertainty"] = uncertainty
                uncertain_examples.append(eg)
    
    log(f"Found {len(uncertain_examples)} uncertain examples")
    
    # Sort by uncertainty and limit
    uncertain_examples.sort(key=lambda eg: eg.get("_uncertainty", 0), reverse=True)
    selected_examples = uncertain_examples[:max_examples]
    
    log(f"Selected {len(selected_examples)} examples for annotation")
    
    # Add hashes
    stream = set_hashes(selected_examples, input_keys=("text",))
    
    # Set up the Prodigy session
    return {
        "view_id": "ner",
        "dataset": dataset,
        "stream": stream,
        "update": None,
        "exclude": exclude or [],
        "config": {
            "labels": labels
        }
    }


@recipe(
    "ner.train",
    # Model name
    model=("Base model name or path", "positional", None, str),
    # Output path
    output=("Path to save trained model", "positional", None, str),
    # Dataset name with annotations
    dataset=("Dataset with annotations", "positional", None, str),
    # Entity labels to train
    labels=("Comma-separated entity labels", "option", "l", split_string),
    # Training epochs
    n_iter=("Number of training iterations", "option", "n", int),
    # Dropout rate
    dropout=("Dropout rate", "option", "d", float),
    # Batch size
    batch_size=("Batch size", "option", "b", int),
)
def train_ner_model(
    model: str,
    output: str,
    dataset: str,
    labels: Optional[List[str]] = None,
    n_iter: int = 10,
    dropout: float = 0.2,
    batch_size: int = 4
) -> Dict[str, Any]:
    """
    Recipe to train an NER model from annotations.
    
    This recipe extracts NER annotations from a Prodigy dataset and trains a
    custom NER model.
    
    Args:
        model: Base model name or path
        output: Path to save the trained model
        dataset: Dataset with annotations
        labels: Entity labels to train
        n_iter: Number of training iterations
        dropout: Dropout rate for training
        batch_size: Batch size for training
        
    Returns:
        Dictionary with training results
    """
    import spacy
    
    log(f"Training NER model from dataset '{dataset}'")
    
    # Connect to the database
    db = connect()
    if not db.exists(dataset):
        raise ValueError(f"Dataset '{dataset}' does not exist")
    
    # Get annotations
    examples = db.get_dataset(dataset)
    log(f"Loaded {len(examples)} examples from dataset")
    
    # Prepare training data
    training_data = []
    
    for eg in examples:
        # Skip examples without annotations
        if eg.get("answer") != "accept":
            continue
            
        text = eg.get("text", "")
        spans = eg.get("spans", [])
        
        if not spans:
            continue
            
        # Convert spans to (start, end, label) format
        entities = [(span["start"], span["end"], span["label"]) for span in spans]
        
        training_data.append({"text": text, "entities": entities})
    
    if not training_data:
        log("No valid training examples found")
        return {"status": "error", "message": "No valid training examples found"}
    
    log(f"Prepared {len(training_data)} training examples")
    
    # Load or create base model
    try:
        nlp = spacy.load(model)
        log(f"Loaded base model: {model}")
    except IOError:
        log(f"Creating new model with 'en' pipeline")
        nlp = spacy.blank("en")
    
    # Add or get NER component
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    # Add entity labels
    for label in labels or set(label for _, _, label in [e for d in training_data for e in d["entities"]]):
        ner.add_label(label)
    
    # Train the model
    from intelligence.entities.ner_model import CustomNERModel
    
    # Create custom model
    custom_model = CustomNERModel(model_name=None)
    custom_model.nlp = nlp
    
    # Train with custom data
    metrics = custom_model.train(training_data, n_iter=n_iter)
    
    # Save model
    custom_model.save_model(output_dir=output)
    
    log(f"Trained model saved to {output}")
    return {
        "status": "success",
        "model": output,
        "metrics": metrics
    }


def _add_model_predictions(stream: Iterator[Dict[str, Any]], model: Union[CustomNERModel, FootballNERModel],
                         labels: List[str]) -> Iterator[Dict[str, Any]]:
    """
    Add model predictions to examples stream.
    
    Args:
        stream: Stream of examples
        model: NER model to use for predictions
        labels: Entity labels to include
        
    Returns:
        Stream with added predictions
    """
    for example in stream:
        text = example.get("text", "")
        
        if not text:
            yield example
            continue
            
        # Get entities
        entities = model.extract_entities(text)
        
        # Filter by labels
        if labels:
            entities = [e for e in entities if e.label in labels]
            
        # Convert to Prodigy spans
        spans = []
        for entity in entities:
            spans.append({
                "start": entity.start_char,
                "end": entity.end_char,
                "label": entity.label,
                "score": entity.confidence
            })
            
        # Add spans to example
        example["spans"] = spans
        
        yield example
