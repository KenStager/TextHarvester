"""
Prodigy Recipe for Topic Classification Annotation.

This module provides a custom Prodigy recipe for topic classification annotation,
enabling efficient collection of training data for topic classifiers.
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
    from prodigy.recipes.textcat import manual as textcat_manual
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
    stream_from_scraper, prepare_classification_task,
    save_annotations_for_training
)
from intelligence.classification.topic_taxonomy import TopicTaxonomy
from intelligence.classification.taxonomies.football import get_premier_league_taxonomy


@recipe(
    "textcat.football-topics",
    # Dataset name
    dataset=("Dataset to save annotations to", "positional", None, str),
    # Source of examples
    source=("Source file with texts (optional)", "option", "s", str),
    # Labels to annotate
    labels=("Comma-separated topic labels", "option", "l", split_string),
    # Topic taxonomy to use
    taxonomy=("Topic taxonomy name", "option", "t", str),
    # Database connection to read from
    db_url=("Database connection string", "option", "db", str),
    # Content limit
    limit=("Limit of content items to annotate", "option", "lim", int),
    # Random order
    random_order=("Whether to randomize example order", "flag", "r", bool),
    # Exclude previously annotated examples
    exclude=("Dataset IDs to exclude examples from", "option", "e", split_string),
)
def football_topics(
    dataset: str,
    source: Optional[str] = None,
    labels: Optional[List[str]] = None,
    taxonomy: str = "football",
    db_url: Optional[str] = None,
    limit: int = 100,
    random_order: bool = False,
    exclude: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Custom recipe for annotating football topic classification.
    
    This recipe streams content from either a JSONL file or directly from the
    scraper database and allows annotating topic labels based on the football
    taxonomy.
    
    Args:
        dataset: Name of dataset to save annotations to
        source: Optional JSONL file with texts to annotate
        labels: Optional list of topic labels (if not using taxonomy)
        taxonomy: Topic taxonomy to use for labels
        db_url: Optional database URL to stream content from
        limit: Maximum number of examples to annotate
        random_order: Whether to randomize the example order
        exclude: Datasets to exclude examples from
        
    Returns:
        Dictionary with session components for Prodigy
    """
    log(f"Loading topic classification annotation for football with dataset '{dataset}'")
    
    # Get labels from taxonomy if not provided
    if not labels:
        if taxonomy == "football":
            # Get football taxonomy
            football_tax = get_premier_league_taxonomy()
            
            # Get top-level topics (Premier League subtopics)
            if football_tax.root_nodes:
                premier_league = None
                for child in football_tax.root_nodes[0].children:
                    if child.name == "Premier League":
                        premier_league = child
                        break
                        
                if premier_league:
                    # Use Premier League subtopics as labels
                    labels = [child.name for child in premier_league.children]
                else:
                    # Fallback to root node children
                    labels = [child.name for child in football_tax.root_nodes[0].children]
            else:
                # Fallback to basic football labels
                labels = ["Premier League", "Teams", "Players", "Matches", "Transfers"]
        else:
            # Fallback to basic labels
            labels = ["Football", "Premier League", "Other Sports"]
    
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
    
    # Prepare classification examples
    stream = prepare_classification_task(stream, labels)
    
    # Add hashes and randomize if needed
    stream = set_hashes(stream, input_keys=("text",))
    
    if random_order:
        stream = list(stream)
        random.shuffle(stream)
    
    # Set up the Prodigy session
    return {
        "view_id": "classification",  # Use Prodigy's classification interface
        "dataset": dataset,
        "stream": stream,
        "update": None,  # No update function for basic classification
        "exclude": exclude or [],
        "config": {
            "labels": labels,
            "classification_type": "multiple" if len(labels) > 1 else "binary"
        }
    }


@recipe(
    "textcat.football-binary",
    # Dataset name
    dataset=("Dataset to save annotations to", "positional", None, str),
    # Source of examples
    source=("Source file with texts (optional)", "option", "s", str),
    # Topic to annotate
    topic=("Topic to annotate", "option", "t", str),
    # Database connection to read from
    db_url=("Database connection string", "option", "db", str),
    # Content limit
    limit=("Limit of content items to annotate", "option", "lim", int),
    # Random order
    random_order=("Whether to randomize example order", "flag", "r", bool),
    # Exclude previously annotated examples
    exclude=("Dataset IDs to exclude examples from", "option", "e", split_string),
)
def football_binary(
    dataset: str,
    topic: str = "Football",
    source: Optional[str] = None,
    db_url: Optional[str] = None,
    limit: int = 100,
    random_order: bool = False,
    exclude: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Recipe for binary football topic classification annotation.
    
    This recipe allows for binary classification of whether content belongs
    to a specific football topic or not.
    
    Args:
        dataset: Name of dataset to save annotations to
        topic: Topic to annotate (is/is not this topic)
        source: Optional JSONL file with texts to annotate
        db_url: Optional database URL to stream content from
        limit: Maximum number of examples to annotate
        random_order: Whether to randomize the example order
        exclude: Datasets to exclude examples from
        
    Returns:
        Dictionary with session components for Prodigy
    """
    log(f"Loading binary classification annotation for '{topic}' with dataset '{dataset}'")
    
    # Set up stream of examples
    if source:
        # Load from JSONL file
        stream = JSONL(source)
        log(f"Loading examples from {source}")
    else:
        # Stream from scraper database
        stream = stream_from_scraper(db_url, limit=limit)
        log(f"Streaming examples from scraper database")
    
    # Prepare binary classification examples
    labels = [topic]
    stream = prepare_classification_task(stream, labels)
    
    # Add hashes and randomize if needed
    stream = set_hashes(stream, input_keys=("text",))
    
    if random_order:
        stream = list(stream)
        random.shuffle(stream)
    
    # Set up the Prodigy session
    return {
        "view_id": "classification",  # Use Prodigy's classification interface
        "dataset": dataset,
        "stream": stream,
        "update": None,  # No update function for basic classification
        "exclude": exclude or [],
        "config": {
            "labels": labels,
            "classification_type": "binary"
        }
    }


@recipe(
    "textcat.football-train",
    # Dataset name
    dataset=("Dataset to save annotations to", "positional", None, str),
    # Source dataset with annotations
    source_dataset=("Dataset with annotations", "positional", None, str),
    # Topic to train
    topic=("Topic to train classifier for", "option", "t", str),
    # Output model path
    output=("Path to save trained model", "option", "o", str),
    # Model type
    model_type=("Type of model to train", "option", "m", str),
)
def football_train(
    dataset: str,
    source_dataset: str,
    topic: Optional[str] = None,
    output: Optional[str] = None,
    model_type: str = "svm"
) -> Dict[str, Any]:
    """
    Recipe for training a football topic classifier from annotations.
    
    This recipe extracts annotations from a Prodigy dataset and trains a
    classifier for a specific topic or for multiple topics.
    
    Args:
        dataset: Name of dataset to save training results to
        source_dataset: Dataset with annotations to use for training
        topic: Optional specific topic to train (if None, train all)
        output: Path to save trained model
        model_type: Type of model to train ("svm", "lr", or "rf")
        
    Returns:
        Dictionary with training results
    """
    from intelligence.classification.classifiers import TopicClassifier
    
    log(f"Training classifier from dataset '{source_dataset}'")
    
    # Connect to the database
    db = connect()
    if not db.exists(source_dataset):
        raise ValueError(f"Dataset '{source_dataset}' does not exist")
    
    # Get annotations
    examples = db.get_dataset(source_dataset)
    log(f"Loaded {len(examples)} examples from dataset")
    
    # Prepare training data
    training_data = []
    
    for eg in examples:
        # Skip non-accepted examples
        if eg.get("answer") != "accept":
            continue
            
        text = eg.get("text", "")
        
        # If specific topic is targeted
        if topic:
            is_positive = topic in eg.get("accept", [])
            training_data.append((text, is_positive))
        else:
            # For all topics in the example
            for label in eg.get("accept", []):
                training_data.append((text, True))
    
    if not training_data:
        log("No valid training examples found")
        return {"status": "error", "message": "No valid training examples found"}
    
    log(f"Prepared {len(training_data)} training examples")
    
    # Train the classifier
    classifier = TopicClassifier(
        node_id=topic or "football",
        node_name=topic or "Football",
        model_type=model_type
    )
    
    metrics = classifier.train(training_data)
    
    # Save training results to dataset
    if db.exists(dataset):
        db.drop_dataset(dataset)
    
    db.add_dataset(dataset)
    db.add_examples(
        [
            {
                "meta": {
                    "topic": topic or "football",
                    "model_type": model_type,
                    "metrics": metrics,
                    "training_examples": len(training_data),
                    "source_dataset": source_dataset,
                    "timestamp": time.time()
                }
            }
        ],
        datasets=[dataset]
    )
    
    log(f"Trained classifier with accuracy {metrics['accuracy']:.4f}")
    return {
        "status": "success",
        "dataset": dataset,
        "metrics": metrics
    }
