"""
Topic Taxonomy Management System.

This module provides tools for building, maintaining, and interacting with
topic taxonomies used for content classification.
"""

import logging
import json
from typing import Dict, List, Optional, Union, Any, Set, Tuple
import os
import uuid

from db.models.topic_taxonomy import TopicNode as DbTopicNode
from intelligence.config import TAXONOMY_EXPORT_PATH

logger = logging.getLogger(__name__)


class TopicTaxonomy:
    """
    Topic taxonomy management system for content classification.
    
    This class provides methods to build and maintain hierarchical topic taxonomies,
    manage topic nodes, import/export taxonomies, and visualize topic relationships.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize a new topic taxonomy.
        
        Args:
            name: Unique name for the taxonomy
            description: Optional description of the taxonomy
        """
        self.name = name
        self.description = description
        self.root_nodes = []
        self._node_cache = {}  # Cache nodes by ID for faster lookup
        
    def add_root_node(self, node: 'TopicNode') -> None:
        """
        Add a root node to the taxonomy.
        
        Args:
            node: The root node to add
        """
        self.root_nodes.append(node)
        self._update_node_cache(node)
        
    def get_node_by_id(self, node_id: str) -> Optional['TopicNode']:
        """
        Retrieve a node by its ID.
        
        Args:
            node_id: The ID of the node to retrieve
            
        Returns:
            The node if found, None otherwise
        """
        return self._node_cache.get(node_id)
    
    def get_node_by_name(self, name: str) -> Optional['TopicNode']:
        """
        Retrieve a node by its name (first match).
        
        Args:
            name: The name of the node to retrieve
            
        Returns:
            The node if found, None otherwise
        """
        for node_id, node in self._node_cache.items():
            if node.name == name:
                return node
        return None
    
    def get_all_nodes(self) -> List['TopicNode']:
        """
        Get all nodes in the taxonomy.
        
        Returns:
            List of all nodes
        """
        return list(self._node_cache.values())
    
    def get_leaf_nodes(self) -> List['TopicNode']:
        """
        Get all leaf nodes (nodes without children) in the taxonomy.
        
        Returns:
            List of leaf nodes
        """
        return [node for node in self._node_cache.values() if not node.children]
    
    def _update_node_cache(self, node: 'TopicNode') -> None:
        """
        Update the node cache with a node and all its descendants.
        
        Args:
            node: The node to add to the cache
        """
        self._node_cache[node.id] = node
        for child in node.children:
            self._update_node_cache(child)
    
    def get_hierarchy(self) -> List[Dict]:
        """
        Get the full taxonomy hierarchy as a nested dictionary structure.
        
        Returns:
            List of dictionaries representing the taxonomy hierarchy
        """
        return [node.to_dict(include_children=True) for node in self.root_nodes]
    
    def find_path_to_node(self, node_id: str) -> List['TopicNode']:
        """
        Find the path from a root node to the specified node.
        
        Args:
            node_id: ID of the target node
            
        Returns:
            List of nodes forming the path, or empty list if not found
        """
        target_node = self.get_node_by_id(node_id)
        if not target_node:
            return []
            
        path = []
        current = target_node
        
        while current:
            path.insert(0, current)
            current = current.parent
            
        return path
    
    def export_to_json(self, filepath: Optional[str] = None) -> str:
        """
        Export the taxonomy to a JSON file.
        
        Args:
            filepath: Path to save the JSON file, defaults to config path
            
        Returns:
            Path to the exported file
        """
        if not filepath:
            os.makedirs(TAXONOMY_EXPORT_PATH, exist_ok=True)
            filepath = os.path.join(TAXONOMY_EXPORT_PATH, f"{self.name}_taxonomy.json")
            
        taxonomy_dict = {
            "name": self.name,
            "description": self.description,
            "nodes": [node.to_dict(include_children=True) for node in self.root_nodes]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(taxonomy_dict, f, indent=2)
            
        logger.info(f"Exported taxonomy '{self.name}' to {filepath}")
        return filepath
    
    @classmethod
    def import_from_json(cls, filepath: str) -> 'TopicTaxonomy':
        """
        Import a taxonomy from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Imported taxonomy instance
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            taxonomy_dict = json.load(f)
            
        taxonomy = cls(
            name=taxonomy_dict.get("name", "Imported Taxonomy"),
            description=taxonomy_dict.get("description", "")
        )
        
        # Build nodes
        for node_dict in taxonomy_dict.get("nodes", []):
            root_node = TopicNode.from_dict(node_dict)
            taxonomy.add_root_node(root_node)
            
        logger.info(f"Imported taxonomy '{taxonomy.name}' from {filepath}")
        return taxonomy
    
    def save_to_database(self) -> List[DbTopicNode]:
        """
        Save the taxonomy to the database.
        
        Returns:
            List of database TopicNode objects
        """
        db_nodes = []
        
        # Process each root node
        for root_node in self.root_nodes:
            db_nodes.extend(self._save_node_to_db(root_node))
            
        logger.info(f"Saved taxonomy '{self.name}' to database with {len(db_nodes)} nodes")
        return db_nodes
    
    def _save_node_to_db(self, node: 'TopicNode', 
                         parent_db_node: Optional[DbTopicNode] = None) -> List[DbTopicNode]:
        """
        Recursively save a node and its children to the database.
        
        Args:
            node: The node to save
            parent_db_node: Optional parent database node
            
        Returns:
            List of created database nodes
        """
        # Create or update this node
        db_node = DbTopicNode.get_by_id(node.id) if node.id else None
        
        if not db_node:
            db_node = DbTopicNode(
                name=node.name,
                description=node.description,
                keywords=node.keywords,
                parent_id=parent_db_node.id if parent_db_node else None
            )
            db_node.save()
        else:
            # Update existing
            db_node.name = node.name
            db_node.description = node.description
            db_node.keywords = node.keywords
            db_node.parent_id = parent_db_node.id if parent_db_node else None
            db_node.save()
            
        # Update node ID if it was newly created
        if not node.id:
            node.id = db_node.id
            
        result = [db_node]
        
        # Process children
        for child in node.children:
            result.extend(self._save_node_to_db(child, db_node))
            
        return result
    
    @classmethod
    def load_from_database(cls, taxonomy_name: str) -> 'TopicTaxonomy':
        """
        Load a taxonomy from the database based on name.
        
        Args:
            taxonomy_name: Name of the taxonomy to load
            
        Returns:
            Loaded taxonomy instance
        """
        # Find all nodes without parents (roots)
        root_db_nodes = DbTopicNode.get_root_nodes()
        
        taxonomy = cls(name=taxonomy_name)
        
        # Process each root node
        for root_db_node in root_db_nodes:
            root_node = cls._load_node_from_db(root_db_node)
            taxonomy.add_root_node(root_node)
            
        logger.info(f"Loaded taxonomy '{taxonomy_name}' from database with {len(taxonomy.get_all_nodes())} nodes")
        return taxonomy
    
    @staticmethod
    def _load_node_from_db(db_node: DbTopicNode) -> 'TopicNode':
        """
        Recursively load a node and its children from the database.
        
        Args:
            db_node: Database node to load
            
        Returns:
            Loaded TopicNode instance
        """
        # Create the node
        node = TopicNode(
            name=db_node.name,
            description=db_node.description,
            keywords=db_node.keywords,
            node_id=db_node.id
        )
        
        # Load children
        for child_db_node in db_node.get_children():
            child_node = TopicTaxonomy._load_node_from_db(child_db_node)
            node.add_child(child_node)
            
        return node
    
    def visualize(self) -> Dict:
        """
        Generate a visualization-friendly representation of the taxonomy.
        
        Returns:
            Dictionary with visualization data
        """
        # This could be extended to generate actual visualizations using libraries
        # like NetworkX, Graphviz, etc. For now, we'll return a format suitable
        # for D3.js or similar visualization libraries.
        nodes = []
        links = []
        
        # Create nodes
        for node in self.get_all_nodes():
            nodes.append({
                "id": node.id,
                "name": node.name,
                "level": len(self.find_path_to_node(node.id)) - 1
            })
            
            # Create links from parent to children
            if node.parent:
                links.append({
                    "source": node.parent.id,
                    "target": node.id
                })
                
        return {
            "nodes": nodes,
            "links": links
        }
        
    def get_training_data(self) -> Dict[str, List[str]]:
        """
        Generate training data for classifiers from the taxonomy.
        
        Returns:
            Dictionary mapping node IDs to lists of training examples
        """
        training_data = {}
        
        for node in self.get_all_nodes():
            # Start with node's keywords as examples
            examples = node.keywords.copy() if node.keywords else []
            
            # Add node name and description
            examples.append(node.name)
            if node.description:
                examples.append(node.description)
                
            # Add parent context for more specific nodes
            path = self.find_path_to_node(node.id)
            if len(path) > 1:
                context = " > ".join([n.name for n in path])
                examples.append(context)
                
            training_data[node.id] = examples
            
        return training_data


class TopicNode:
    """
    Represents a node in a topic taxonomy.
    
    A topic node can have a parent, children, and metadata like keywords
    that help identify content belonging to this topic.
    """
    
    def __init__(self, name: str, keywords: Optional[List[str]] = None, 
                 description: str = "", node_id: Optional[str] = None,
                 parent: Optional['TopicNode'] = None):
        """
        Initialize a new topic node.
        
        Args:
            name: Name of the topic
            keywords: List of keywords associated with this topic
            description: Optional description of the topic
            node_id: Optional ID for the node (generated if not provided)
            parent: Optional parent node
        """
        self.id = node_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.keywords = keywords or []
        self.parent = parent
        self.children = []
        self.classifier = None  # Will hold a reference to the node's classifier when trained
        
    def add_child(self, child: 'TopicNode') -> None:
        """
        Add a child node to this node.
        
        Args:
            child: The child node to add
        """
        self.children.append(child)
        child.parent = self
        
    def add_keyword(self, keyword: str) -> None:
        """
        Add a keyword to this node.
        
        Args:
            keyword: The keyword to add
        """
        if keyword not in self.keywords:
            self.keywords.append(keyword)
            
    def add_keywords(self, keywords: List[str]) -> None:
        """
        Add multiple keywords to this node.
        
        Args:
            keywords: List of keywords to add
        """
        for keyword in keywords:
            self.add_keyword(keyword)
            
    def to_dict(self, include_children: bool = False) -> Dict:
        """
        Convert the node to a dictionary representation.
        
        Args:
            include_children: Whether to include children recursively
            
        Returns:
            Dictionary representation of the node
        """
        node_dict = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "keywords": self.keywords
        }
        
        if include_children and self.children:
            node_dict["children"] = [
                child.to_dict(include_children=True) for child in self.children
            ]
            
        return node_dict
    
    @classmethod
    def from_dict(cls, node_dict: Dict) -> 'TopicNode':
        """
        Create a node from a dictionary representation.
        
        Args:
            node_dict: Dictionary representation of the node
            
        Returns:
            Created TopicNode instance
        """
        # Create this node
        node = cls(
            name=node_dict["name"],
            keywords=node_dict.get("keywords", []),
            description=node_dict.get("description", ""),
            node_id=node_dict.get("id")
        )
        
        # Create children if present
        for child_dict in node_dict.get("children", []):
            child = cls.from_dict(child_dict)
            node.add_child(child)
            
        return node
            
    def get_full_path(self) -> str:
        """
        Get the full path to this node from the root.
        
        Returns:
            String representation of the path (e.g., "Root > Parent > Child")
        """
        if not self.parent:
            return self.name
            
        return f"{self.parent.get_full_path()} > {self.name}"
    
    def train_classifier(self, training_data: List[Tuple[str, bool]]) -> None:
        """
        Train a classifier for this specific node.
        
        Args:
            training_data: List of (text, is_positive) tuples for training
        """
        from intelligence.classification.classifiers import TopicClassifier
        
        # Initialize and train classifier for this topic node
        self.classifier = TopicClassifier(self.id, self.name)
        self.classifier.train(training_data)
        
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict if the text belongs to this topic.
        
        Args:
            text: The text to classify
            
        Returns:
            Dictionary with prediction results including confidence
        """
        if not self.classifier:
            raise ValueError(f"No trained classifier for node {self.name}")
            
        return self.classifier.predict(text)
