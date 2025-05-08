"""
Machine Learning Model Utilities
===============================

This module provides common machine learning model utilities for the
Content Intelligence Platform, including model loading and caching,
inference optimization, vectorization, and embedding utilities.
"""

import os
import pickle
import logging
import hashlib
import time
import functools
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from pathlib import Path
import json
import threading
import numpy as np
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

# Optional imports - try to import commonly used ML libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. Some features will be unavailable.")

try:
    from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not installed. Some features will be unavailable.")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not installed. Some features will be unavailable.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("Sentence-Transformers not installed. Some features will be unavailable.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed. Some features will be unavailable.")

# Import configuration settings
from intelligence.config import Config

# Global cache for loaded models
_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.RLock()  # Reentrant lock for thread safety

# Maximum number of models to keep in cache
MAX_CACHE_SIZE = 10

# Model cache expiry time in seconds (1 hour)
CACHE_EXPIRY_TIME = 3600


class ModelCacheItem:
    """Container for a cached model with metadata."""
    
    def __init__(self, model, model_type: str):
        """
        Initialize a model cache item.
        
        Args:
            model: The loaded model.
            model_type (str): Type of model.
        """
        self.model = model
        self.model_type = model_type
        self.last_accessed = time.time()
        self.load_time = time.time()
        self.access_count = 0
    
    def access(self):
        """Update access metadata when model is used."""
        self.last_accessed = time.time()
        self.access_count += 1


def cache_key(model_path: str, model_type: str) -> str:
    """
    Generate a unique key for the model cache.
    
    Args:
        model_path (str): Path to the model.
        model_type (str): Type of model.
        
    Returns:
        str: Unique cache key.
    """
    return f"{model_type}:{model_path}"


def load_model(model_path: str, model_type: str, **kwargs) -> Any:
    """
    Load a model of the specified type from the given path.
    
    Args:
        model_path (str): Path to the model.
        model_type (str): Type of model.
        **kwargs: Additional arguments for model loading.
        
    Returns:
        Any: The loaded model.
        
    Raises:
        ValueError: If the model type is not supported.
        FileNotFoundError: If the model file is not found.
    """
    # Check if model is in cache
    key = cache_key(model_path, model_type)
    
    with _MODEL_CACHE_LOCK:
        cache_item = _MODEL_CACHE.get(key)
        
        if cache_item:
            # Check if cache has expired
            if time.time() - cache_item.last_accessed < CACHE_EXPIRY_TIME:
                logger.debug(f"Using cached model: {key}")
                cache_item.access()
                return cache_item.model
            else:
                logger.debug(f"Cache expired for model: {key}")
                _MODEL_CACHE.pop(key)
    
    # Load the model based on its type
    try:
        if model_type == 'pickle':
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                
        elif model_type == 'spacy' and SPACY_AVAILABLE:
            model = spacy.load(model_path)
            
        elif model_type == 'huggingface' and TRANSFORMERS_AVAILABLE:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path)
            model = (model, tokenizer)  # Return as tuple
            
        elif model_type == 'huggingface_classification' and TRANSFORMERS_AVAILABLE:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model = (model, tokenizer)  # Return as tuple
            
        elif model_type == 'sentence_transformer' and SENTENCE_TRANSFORMERS_AVAILABLE:
            model = SentenceTransformer(model_path)
            
        elif model_type == 'tfidf' and SKLEARN_AVAILABLE:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                
        elif model_type == 'pytorch' and TORCH_AVAILABLE:
            model = torch.load(model_path, map_location=torch.device('cpu'))
            
        elif model_type == 'json':
            with open(model_path, 'r') as f:
                model = json.load(f)
                
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
    
    # Add to cache
    with _MODEL_CACHE_LOCK:
        # Check if cache is full and remove least recently used model
        if len(_MODEL_CACHE) >= MAX_CACHE_SIZE:
            lru_key = min(_MODEL_CACHE.items(), 
                          key=lambda x: x[1].last_accessed)[0]
            _MODEL_CACHE.pop(lru_key)
            logger.debug(f"Removed LRU model from cache: {lru_key}")
        
        # Add new model to cache
        _MODEL_CACHE[key] = ModelCacheItem(model, model_type)
        logger.debug(f"Added model to cache: {key}")
    
    return model


def save_model(model: Any, model_path: str, model_type: str, **kwargs) -> None:
    """
    Save a model to the specified path.
    
    Args:
        model (Any): The model to save.
        model_path (str): Path to save the model.
        model_type (str): Type of model.
        **kwargs: Additional arguments for model saving.
        
    Raises:
        ValueError: If the model type is not supported.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    try:
        if model_type == 'pickle':
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
                
        elif model_type == 'spacy' and SPACY_AVAILABLE:
            model.to_disk(model_path)
            
        elif model_type in ('huggingface', 'huggingface_classification') and TRANSFORMERS_AVAILABLE:
            if isinstance(model, tuple) and len(model) == 2:
                model_obj, tokenizer = model
                model_obj.save_pretrained(model_path)
                tokenizer.save_pretrained(model_path)
            else:
                model.save_pretrained(model_path)
                
        elif model_type == 'pytorch' and TORCH_AVAILABLE:
            torch.save(model, model_path)
            
        elif model_type == 'json':
            with open(model_path, 'w') as f:
                json.dump(model, f, indent=2)
                
        else:
            raise ValueError(f"Unsupported model type for saving: {model_type}")
            
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise
    
    logger.info(f"Model saved to {model_path}")


def clear_model_cache() -> None:
    """Clear the model cache."""
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()
    logger.debug("Model cache cleared")


def get_model_cache_info() -> Dict[str, Any]:
    """
    Get information about the model cache.
    
    Returns:
        Dict[str, Any]: Cache information.
    """
    with _MODEL_CACHE_LOCK:
        info = {
            "size": len(_MODEL_CACHE),
            "max_size": MAX_CACHE_SIZE,
            "models": []
        }
        
        for key, item in _MODEL_CACHE.items():
            info["models"].append({
                "key": key,
                "type": item.model_type,
                "last_accessed": datetime.fromtimestamp(item.last_accessed).isoformat(),
                "load_time": datetime.fromtimestamp(item.load_time).isoformat(),
                "access_count": item.access_count,
                "age_seconds": time.time() - item.load_time
            })
        
        return info


def with_model(model_path: str, model_type: str):
    """
    Decorator to load a model for a function.
    
    Args:
        model_path (str): Path to the model.
        model_type (str): Type of model.
        
    Returns:
        Callable: Decorated function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            model = load_model(model_path, model_type)
            return func(model, *args, **kwargs)
        return wrapper
    return decorator


# Embedding and Vectorization Utilities

def get_text_embeddings(texts: List[str], model_name: str = 'all-MiniLM-L6-v2', 
                        batch_size: int = 32) -> np.ndarray:
    """
    Get embeddings for a list of texts using sentence transformers.
    
    Args:
        texts (List[str]): Texts to embed.
        model_name (str, optional): Name of the sentence transformer model.
        batch_size (int, optional): Batch size for processing.
        
    Returns:
        np.ndarray: Text embeddings.
        
    Raises:
        ImportError: If sentence-transformers is not installed.
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers is required for text embeddings")
    
    model = load_model(model_name, 'sentence_transformer')
    return model.encode(texts, batch_size=batch_size)


def get_tfidf_vectors(texts: List[str], vectorizer: Optional[Any] = None, 
                      load_path: Optional[str] = None, save_path: Optional[str] = None,
                      **kwargs) -> Tuple[np.ndarray, Any]:
    """
    Get TF-IDF vectors for a list of texts.
    
    Args:
        texts (List[str]): Texts to vectorize.
        vectorizer (Any, optional): Existing TF-IDF vectorizer.
        load_path (str, optional): Path to load vectorizer from.
        save_path (str, optional): Path to save vectorizer to.
        **kwargs: Additional arguments for TfidfVectorizer.
        
    Returns:
        Tuple[np.ndarray, Any]: TF-IDF vectors and vectorizer.
        
    Raises:
        ImportError: If scikit-learn is not installed.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for TF-IDF vectorization")
    
    # Load existing vectorizer if path provided
    if load_path and not vectorizer:
        vectorizer = load_model(load_path, 'pickle')
    
    # Create new vectorizer if none provided
    if not vectorizer:
        vectorizer = TfidfVectorizer(**kwargs)
        vectors = vectorizer.fit_transform(texts)
    else:
        vectors = vectorizer.transform(texts)
    
    # Save vectorizer if path provided
    if save_path:
        save_model(vectorizer, save_path, 'pickle')
    
    return vectors, vectorizer


def compute_similarity_matrix(embeddings: np.ndarray, metric: str = 'cosine') -> np.ndarray:
    """
    Compute similarity matrix between embeddings.
    
    Args:
        embeddings (np.ndarray): Matrix of embeddings.
        metric (str, optional): Similarity metric ('cosine', 'dot', 'euclidean').
        
    Returns:
        np.ndarray: Similarity matrix.
    """
    # Normalize embeddings for cosine similarity
    if metric == 'cosine':
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.maximum(norms, 1e-10)
        return np.dot(normalized, normalized.T)
    
    elif metric == 'dot':
        return np.dot(embeddings, embeddings.T)
    
    elif metric == 'euclidean':
        # Compute pairwise distances and convert to similarity
        # (smaller distance = higher similarity)
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(embeddings)
        max_dist = np.max(distances)
        return 1 - (distances / max_dist)
    
    else:
        raise ValueError(f"Unsupported similarity metric: {metric}")


# Model Performance Monitoring

class ModelPerformanceTracker:
    """Track and log model performance metrics."""
    
    def __init__(self, model_name: str, metric_logger: Optional[Callable] = None):
        """
        Initialize a performance tracker.
        
        Args:
            model_name (str): Name of the model being tracked.
            metric_logger (Callable, optional): Function to log metrics.
        """
        self.model_name = model_name
        self.calls = 0
        self.total_time = 0
        self.max_time = 0
        self.min_time = float('inf')
        self.errors = 0
        self.last_call_time = None
        self.metric_logger = metric_logger
    
    def __enter__(self):
        """Context manager entry."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        end_time = time.time()
        elapsed = end_time - self.start_time
        
        self.calls += 1
        self.total_time += elapsed
        self.max_time = max(self.max_time, elapsed)
        self.min_time = min(self.min_time, elapsed)
        self.last_call_time = elapsed
        
        if exc_type is not None:
            self.errors += 1
        
        # Log metrics if logger provided
        if self.metric_logger:
            self.metric_logger(
                model_name=self.model_name,
                elapsed_time=elapsed,
                success=(exc_type is None)
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dict[str, Any]: Statistics dictionary.
        """
        avg_time = self.total_time / max(1, self.calls)
        
        return {
            "model_name": self.model_name,
            "calls": self.calls,
            "total_time": self.total_time,
            "avg_time": avg_time,
            "min_time": self.min_time if self.min_time != float('inf') else 0,
            "max_time": self.max_time,
            "error_rate": self.errors / max(1, self.calls),
            "last_call_time": self.last_call_time
        }


def tracked_inference(model_name: str, metric_logger: Optional[Callable] = None):
    """
    Decorator to track model inference performance.
    
    Args:
        model_name (str): Name of the model.
        metric_logger (Callable, optional): Function to log metrics.
        
    Returns:
        Callable: Decorated function.
    """
    def decorator(func):
        tracker = ModelPerformanceTracker(model_name, metric_logger)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with tracker:
                return func(*args, **kwargs)
        
        wrapper.get_stats = tracker.get_stats
        return wrapper
    
    return decorator


# Batched inference utilities

def batch_iterator(items: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Create batches from a list of items.
    
    Args:
        items (List[Any]): Items to batch.
        batch_size (int): Size of each batch.
        
    Returns:
        List[List[Any]]: List of batches.
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def batched_inference(func: Callable, items: List[Any], batch_size: int, 
                      show_progress: bool = False, **kwargs) -> List[Any]:
    """
    Apply a function to items in batches.
    
    Args:
        func (Callable): Function to apply.
        items (List[Any]): Items to process.
        batch_size (int): Size of each batch.
        show_progress (bool, optional): Whether to show progress.
        **kwargs: Additional arguments for the function.
        
    Returns:
        List[Any]: Results.
    """
    results = []
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    for i, batch in enumerate(batch_iterator(items, batch_size)):
        if show_progress:
            logger.info(f"Processing batch {i+1}/{total_batches} ({len(batch)} items)")
            
        batch_results = func(batch, **kwargs)
        results.extend(batch_results)
    
    return results


# Optimized inference for different model types

def optimized_huggingface_inference(model, tokenizer, texts: List[str], 
                                  batch_size: int = 8, device: str = None) -> List[Any]:
    """
    Optimized inference for Hugging Face models.
    
    Args:
        model: Hugging Face model.
        tokenizer: Hugging Face tokenizer.
        texts (List[str]): Texts to process.
        batch_size (int, optional): Batch size.
        device (str, optional): Device to use (cpu, cuda).
        
    Returns:
        List[Any]: Model outputs.
    """
    if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
        raise ImportError("PyTorch and transformers are required for this function")
    
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    device = torch.device(device)
    model.to(device)
    model.eval()
    
    all_outputs = []
    
    with torch.no_grad():
        for batch in batch_iterator(texts, batch_size):
            # Tokenize batch
            inputs = tokenizer(
                batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(device)
            
            # Run inference
            outputs = model(**inputs)
            
            # Move outputs to CPU for easier handling
            if hasattr(outputs, "logits"):
                batch_outputs = outputs.logits.cpu().numpy()
            else:
                batch_outputs = outputs.last_hidden_state.cpu().numpy()
                
            all_outputs.append(batch_outputs)
    
    # Concatenate batches if they're numpy arrays
    if all(isinstance(o, np.ndarray) for o in all_outputs):
        return np.vstack(all_outputs)
    
    # Otherwise return as list of batches
    return all_outputs


def optimized_spacy_inference(nlp, texts: List[str], batch_size: int = 32, 
                             disable: List[str] = None) -> List[Any]:
    """
    Optimized inference for spaCy models.
    
    Args:
        nlp: spaCy model.
        texts (List[str]): Texts to process.
        batch_size (int, optional): Batch size.
        disable (List[str], optional): Pipeline components to disable.
        
    Returns:
        List[Any]: spaCy Doc objects.
    """
    if not SPACY_AVAILABLE:
        raise ImportError("spaCy is required for this function")
    
    # Use spaCy's pipe for batch processing
    return list(nlp.pipe(texts, batch_size=batch_size, disable=disable))


# Model validation and testing utilities

def k_fold_cross_validate(model_factory: Callable, X: List[Any], y: List[Any], 
                         k: int = 5, scoring_func: Callable = None) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation.
    
    Args:
        model_factory (Callable): Function that creates and returns a model.
        X (List[Any]): Features.
        y (List[Any]): Labels.
        k (int, optional): Number of folds.
        scoring_func (Callable, optional): Function to compute score.
        
    Returns:
        Dict[str, Any]: Cross-validation results.
    """
    try:
        from sklearn.model_selection import KFold
    except ImportError:
        raise ImportError("scikit-learn is required for cross-validation")
    
    # Default scoring function
    if scoring_func is None:
        scoring_func = lambda y_true, y_pred: np.mean(np.array(y_true) == np.array(y_pred))
    
    # Initialize k-fold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    scores = []
    fold_results = []
    
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        logger.info(f"Training fold {i+1}/{k}")
        
        # Split data
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]
        
        # Create and train model
        model = model_factory()
        model.fit(X_train, y_train)
        
        # Predict and score
        y_pred = model.predict(X_test)
        score = scoring_func(y_test, y_pred)
        scores.append(score)
        
        fold_results.append({
            "fold": i + 1,
            "score": score,
            "train_size": len(X_train),
            "test_size": len(X_test)
        })
    
    # Compute overall statistics
    return {
        "scores": scores,
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "min_score": np.min(scores),
        "max_score": np.max(scores),
        "fold_results": fold_results,
        "k": k
    }
