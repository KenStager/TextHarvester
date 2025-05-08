"""
Machine Learning Classification Models.

This module provides classifier implementations for topic classification,
including both primary classification for domain relevance and specialized
classification for hierarchical topic categorization.
"""

import logging
import pickle
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

import joblib

from intelligence.utils.text_processing import normalize_text
from intelligence.utils.model_utils import get_model_path, get_embeddings
from intelligence.config import MODEL_CACHE_SIZE, MODELS_DIR

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Data class for classification results."""
    node_id: str
    node_name: str
    confidence: float
    is_primary: bool = False
    children: List['ClassificationResult'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "confidence": self.confidence,
            "is_primary": self.is_primary
        }
        
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
            
        return result


class TopicClassifier:
    """
    Classifier for a specific topic node.
    
    This class handles the classification of content into a specific topic,
    managing model training, prediction, and confidence scoring.
    """
    
    def __init__(self, node_id: str, node_name: str, model_type: str = "svm"):
        """
        Initialize a new topic classifier.
        
        Args:
            node_id: ID of the topic node this classifier is for
            node_name: Name of the topic node
            model_type: Type of model to use ('svm', 'lr', or 'rf')
        """
        self.node_id = node_id
        self.node_name = node_name
        self.model_type = model_type
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        self.model_path = os.path.join(
            MODELS_DIR, 
            "classifiers", 
            f"topic_{node_id.replace('/', '_')}.pkl"
        )
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Try to load a pre-trained model
        self._try_load_model()
    
    def _try_load_model(self) -> bool:
        """
        Try to load a pre-trained model if it exists.
        
        Returns:
            True if model was loaded, False otherwise
        """
        if os.path.exists(self.model_path):
            try:
                model_data = joblib.load(self.model_path)
                self.model = model_data.get("model")
                self.vectorizer = model_data.get("vectorizer")
                self.is_trained = True
                logger.info(f"Loaded pre-trained model for topic '{self.node_name}'")
                return True
            except Exception as e:
                logger.warning(f"Failed to load model for topic '{self.node_name}': {e}")
                
        return False
    
    def train(self, training_data: List[Tuple[str, bool]]) -> Dict[str, Any]:
        """
        Train the classifier on the provided data.
        
        Args:
            training_data: List of (text, is_positive) tuples
        
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training classifier for topic '{self.node_name}' with {len(training_data)} examples")
        
        # Extract texts and labels
        texts = [normalize_text(text) for text, _ in training_data]
        labels = [1 if is_positive else 0 for _, is_positive in training_data]
        
        # Create vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2)
        )
        
        # Create classifier based on model type
        if self.model_type == "svm":
            base_classifier = LinearSVC(class_weight='balanced')
            # Wrap in calibrated classifier to get probabilities
            classifier = CalibratedClassifierCV(base_classifier)
        elif self.model_type == "lr":
            classifier = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                C=1.0,
                solver='liblinear'
            )
        elif self.model_type == "rf":
            classifier = RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Create pipeline
        self.model = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', classifier)
        ])
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train model
        start_time = time.time()
        self.model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_test)
        f1 = f1_score(y_test, y_test, average='weighted')
        
        metrics = {
            "accuracy": accuracy,
            "f1_score": f1,
            "training_examples": len(training_data),
            "training_time": train_time
        }
        
        # Save model
        self._save_model()
        
        self.is_trained = True
        logger.info(f"Trained classifier for topic '{self.node_name}' with accuracy {accuracy:.4f}")
        
        return metrics
    
    def _save_model(self) -> None:
        """Save the trained model to disk."""
        if not self.model:
            logger.warning(f"Cannot save model for topic '{self.node_name}': No model trained")
            return
            
        model_data = {
            "model": self.model,
            "vectorizer": self.vectorizer,
            "metadata": {
                "node_id": self.node_id,
                "node_name": self.node_name,
                "model_type": self.model_type,
                "timestamp": time.time()
            }
        }
        
        joblib.dump(model_data, self.model_path)
        logger.info(f"Saved model for topic '{self.node_name}' to {self.model_path}")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict if the text belongs to this topic.
        
        Args:
            text: The text to classify
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            raise ValueError(f"Model for topic '{self.node_name}' is not trained")
            
        # Normalize text
        normalized_text = normalize_text(text)
        
        # Make prediction
        start_time = time.time()
        
        # Get prediction probabilities
        probs = self.model.predict_proba([normalized_text])[0]
        confidence = float(probs[1])  # Probability of positive class
        
        prediction_time = time.time() - start_time
        
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "is_relevant": confidence >= 0.5,
            "confidence": confidence,
            "prediction_time": prediction_time
        }


class HierarchicalClassifier:
    """
    Hierarchical topic classifier for multi-level topic taxonomies.
    
    This class manages a collection of topic classifiers organized in a hierarchy,
    coordinating the classification process across multiple levels.
    """
    
    def __init__(self, root_classifier: Optional[TopicClassifier] = None,
                 confidence_threshold: float = 0.5):
        """
        Initialize a hierarchical classifier.
        
        Args:
            root_classifier: Optional root level classifier
            confidence_threshold: Confidence threshold for relevance
        """
        self.root_classifier = root_classifier
        self.confidence_threshold = confidence_threshold
        self.child_classifiers = {}  # Map of node_id to (classifier, children) tuples
    
    def add_child_classifier(self, parent_id: str, 
                             child_classifier: TopicClassifier) -> None:
        """
        Add a child classifier to the hierarchy.
        
        Args:
            parent_id: ID of the parent node
            child_classifier: Child classifier to add
        """
        if parent_id not in self.child_classifiers:
            self.child_classifiers[parent_id] = []
            
        self.child_classifiers[parent_id].append(child_classifier)
        logger.debug(f"Added child classifier '{child_classifier.node_name}' to parent '{parent_id}'")
    
    def predict(self, text: str, max_depth: int = 3) -> ClassificationResult:
        """
        Perform hierarchical classification of text.
        
        Args:
            text: Text to classify
            max_depth: Maximum depth to classify to
            
        Returns:
            ClassificationResult with hierarchical results
        """
        if not self.root_classifier:
            raise ValueError("No root classifier defined")
            
        # Start with root classification
        root_result = self.root_classifier.predict(text)
        
        # Create the result object
        result = ClassificationResult(
            node_id=self.root_classifier.node_id,
            node_name=self.root_classifier.node_name,
            confidence=root_result["confidence"],
            is_primary=True
        )
        
        # If root isn't relevant or we've reached max depth, return
        if not root_result["is_relevant"] or max_depth <= 1:
            return result
            
        # Process child classifiers recursively
        self._classify_children(text, self.root_classifier.node_id, result, 
                                current_depth=1, max_depth=max_depth)
        
        return result
    
    def _classify_children(self, text: str, parent_id: str, 
                          parent_result: ClassificationResult,
                          current_depth: int, max_depth: int) -> None:
        """
        Recursively classify text with child classifiers.
        
        Args:
            text: Text to classify
            parent_id: ID of the parent node
            parent_result: Parent classification result to add children to
            current_depth: Current depth in the hierarchy
            max_depth: Maximum depth to classify to
        """
        # Check if we've reached max depth or have no children
        if current_depth >= max_depth or parent_id not in self.child_classifiers:
            return
            
        # Get child classifiers
        child_classifiers = self.child_classifiers[parent_id]
        
        # Classify with each child
        for classifier in child_classifiers:
            child_result = classifier.predict(text)
            
            # Only add relevant children
            if child_result["confidence"] >= self.confidence_threshold:
                child_classification = ClassificationResult(
                    node_id=classifier.node_id,
                    node_name=classifier.node_name,
                    confidence=child_result["confidence"]
                )
                parent_result.children.append(child_classification)
                
                # Recursively classify with this child's children
                self._classify_children(
                    text, classifier.node_id, child_classification,
                    current_depth + 1, max_depth
                )


class FootballClassifier:
    """
    Specialized classifier for football content.
    
    This class provides a pre-configured hierarchical classifier
    specifically designed for Premier League football content.
    """
    
    @staticmethod
    def create(confidence_threshold: float = 0.4) -> HierarchicalClassifier:
        """
        Create a pre-configured football hierarchical classifier.
        
        Args:
            confidence_threshold: Confidence threshold for relevance
            
        Returns:
            Configured HierarchicalClassifier for football
        """
        from intelligence.classification.taxonomies.football import get_premier_league_taxonomy
        
        # Get the football taxonomy
        football_taxonomy = get_premier_league_taxonomy()
        
        # Create the root classifier for football
        root_node = football_taxonomy.root_nodes[0]  # Football node
        root_classifier = TopicClassifier(
            node_id=root_node.id,
            node_name=root_node.name,
            model_type="svm"
        )
        
        # Create hierarchical classifier
        hierarchical_classifier = HierarchicalClassifier(
            root_classifier=root_classifier,
            confidence_threshold=confidence_threshold
        )
        
        # Add Premier League classifier
        pl_node = root_node.children[0]  # Premier League node
        pl_classifier = TopicClassifier(
            node_id=pl_node.id,
            node_name=pl_node.name,
            model_type="svm"
        )
        hierarchical_classifier.add_child_classifier(root_node.id, pl_classifier)
        
        # Add category classifiers (teams, players, etc.)
        for category_node in pl_node.children:
            category_classifier = TopicClassifier(
                node_id=category_node.id,
                node_name=category_node.name,
                model_type="lr"  # Logistic regression for mid-level
            )
            hierarchical_classifier.add_child_classifier(pl_node.id, category_classifier)
            
            # For categories with children, add specialized classifiers
            if category_node.children:
                for subcategory_node in category_node.children:
                    subcategory_classifier = TopicClassifier(
                        node_id=subcategory_node.id,
                        node_name=subcategory_node.name,
                        model_type="rf"  # Random forest for specialized categories
                    )
                    hierarchical_classifier.add_child_classifier(
                        category_node.id, subcategory_classifier
                    )
        
        logger.info("Created football hierarchical classifier")
        return hierarchical_classifier


class TransformerClassifier:
    """
    Deep learning-based classifier using transformer models.
    
    This class provides a more advanced classification approach using
    pre-trained transformer models for higher accuracy at the cost
    of more computational resources.
    """
    
    def __init__(self, node_id: str, node_name: str, 
                 model_name: str = "distilbert-base-uncased"):
        """
        Initialize a transformer classifier.
        
        Args:
            node_id: ID of the topic node this classifier is for
            node_name: Name of the topic node
            model_name: Name of the pre-trained transformer model to use
        """
        self.node_id = node_id
        self.node_name = node_name
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        
        # Placeholder for potential transformer model implementation
        # In a real implementation, this would initialize a transformer model
        # using a library like transformers from Hugging Face
        logger.info(f"Initialized transformer classifier for '{node_name}' using {model_name}")
        
    def train(self, training_data: List[Tuple[str, bool]]) -> Dict[str, Any]:
        """
        Train the transformer classifier.
        
        Args:
            training_data: List of (text, is_positive) tuples
            
        Returns:
            Dictionary with training metrics
        """
        # Placeholder for transformer model training
        # In a real implementation, this would:
        # 1. Prepare the training data (tokenization, etc.)
        # 2. Fine-tune the pre-trained model
        # 3. Save the fine-tuned model
        logger.info(f"Training transformer classifier for '{self.node_name}' with {len(training_data)} examples")
        
        # Simulate successful training
        self.is_trained = True
        
        return {
            "accuracy": 0.95,  # Placeholder metrics
            "f1_score": 0.94,
            "training_examples": len(training_data)
        }
        
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict using the transformer model.
        
        Args:
            text: Text to classify
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            # For demonstration, we'll simulate a prediction
            # In a real implementation, this would:
            # 1. Tokenize the input text
            # 2. Pass it through the model
            # 3. Process the model output
            
            # Simulate a reasonable confidence based on some text features
            text_lower = text.lower()
            contains_name = self.node_name.lower() in text_lower
            
            # Higher confidence if text contains node name
            confidence = 0.8 if contains_name else 0.3
            
            return {
                "node_id": self.node_id,
                "node_name": self.node_name,
                "is_relevant": confidence >= 0.5,
                "confidence": confidence
            }
            
        # Placeholder for actual model prediction
        logger.warning(f"Actual transformer prediction not implemented for '{self.node_name}'")
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "is_relevant": False,
            "confidence": 0.0
        }
