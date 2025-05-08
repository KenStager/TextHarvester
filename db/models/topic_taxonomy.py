"""
Topic Taxonomy System - ORM Models
==================================

This module defines the SQLAlchemy ORM models for the topic taxonomy system,
including the TopicNode class for managing hierarchical topic relationships.
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.ext.mutable import MutableList
import json

# This would be imported from a central configuration
db = SQLAlchemy()

class TopicTaxonomy(db.Model):
    """SQLAlchemy model for the topic_taxonomy table."""
    
    __tablename__ = 'topic_taxonomy'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    parent_id = db.Column(db.Integer, db.ForeignKey('topic_taxonomy.id'))
    description = db.Column(db.Text)
    keywords = db.Column(MutableList.as_mutable(ARRAY(db.String)))
    classifier_model_path = db.Column(db.String(255))
    domain = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to parent and children
    parent = db.relationship('TopicTaxonomy', remote_side=[id], backref=db.backref('children', lazy='dynamic'))
    
    # Relationship to content classifications
    classifications = db.relationship('ContentClassification', backref='topic', lazy='dynamic')
    
    def __repr__(self):
        return f'<TopicTaxonomy {self.name} (domain: {self.domain})>'
    
    def to_dict(self):
        """Convert the TopicTaxonomy instance to a dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'parent_id': self.parent_id,
            'description': self.description,
            'keywords': self.keywords,
            'classifier_model_path': self.classifier_model_path,
            'domain': self.domain,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def to_json(self):
        """Convert the TopicTaxonomy instance to a JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data):
        """Create a TopicTaxonomy instance from a dictionary."""
        return cls(
            name=data.get('name'),
            parent_id=data.get('parent_id'),
            description=data.get('description'),
            keywords=data.get('keywords', []),
            classifier_model_path=data.get('classifier_model_path'),
            domain=data.get('domain')
        )


class ContentClassification(db.Model):
    """SQLAlchemy model for the content_classification table."""
    
    __tablename__ = 'content_classification'
    
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('scraper_content.id'))
    topic_id = db.Column(db.Integer, db.ForeignKey('topic_taxonomy.id'))
    confidence = db.Column(db.Float, nullable=False)
    is_primary = db.Column(db.Boolean, default=False)
    classification_method = db.Column(db.String(50))
    human_verified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ContentClassification content_id={self.content_id} topic_id={self.topic_id} confidence={self.confidence}>'
    
    def to_dict(self):
        """Convert the ContentClassification instance to a dictionary."""
        return {
            'id': self.id,
            'content_id': self.content_id,
            'topic_id': self.topic_id,
            'confidence': self.confidence,
            'is_primary': self.is_primary,
            'classification_method': self.classification_method,
            'human_verified': self.human_verified,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class TopicNode:
    """
    High-level class for working with topic taxonomies as a hierarchical structure.
    This class wraps the TopicTaxonomy model to provide more convenient operations
    for working with topic trees.
    """
    
    def __init__(self, name, keywords=None, parent=None, domain=None, description=None):
        """
        Initialize a TopicNode.
        
        Args:
            name (str): Name of the topic
            keywords (list, optional): List of keywords associated with this topic
            parent (TopicNode, optional): Parent node
            domain (str, optional): Domain this topic belongs to
            description (str, optional): Description of the topic
        """
        self.name = name
        self.keywords = keywords or []
        self.parent = parent
        self.children = []
        self.domain = domain
        self.description = description
        self.classifier = None
        self.db_model = None
    
    def add_child(self, child):
        """
        Add a child node to this node.
        
        Args:
            child (TopicNode): Child node to add
        
        Returns:
            TopicNode: The added child node
        """
        self.children.append(child)
        child.parent = self
        return child
    
    def get_path(self):
        """
        Get the full path from the root to this node.
        
        Returns:
            list: List of TopicNode objects from root to this node
        """
        if self.parent is None:
            return [self]
        return self.parent.get_path() + [self]
    
    def get_path_names(self):
        """
        Get the full path names from the root to this node.
        
        Returns:
            list: List of node names from root to this node
        """
        return [node.name for node in self.get_path()]
    
    def get_full_name(self, separator='.'):
        """
        Get the full path name as a string.
        
        Args:
            separator (str, optional): Separator to use between node names
        
        Returns:
            str: Full path name
        """
        return separator.join(self.get_path_names())
    
    def find_child_by_name(self, name):
        """
        Find a direct child by name.
        
        Args:
            name (str): Name of the child to find
        
        Returns:
            TopicNode: Child node if found, None otherwise
        """
        for child in self.children:
            if child.name.lower() == name.lower():
                return child
        return None
    
    def find_node_by_path(self, path, separator='.'):
        """
        Find a node by its path.
        
        Args:
            path (str): Path to the node using the given separator
            separator (str, optional): Separator used in the path
        
        Returns:
            TopicNode: Node if found, None otherwise
        """
        if not path:
            return None
        
        parts = path.split(separator)
        if parts[0].lower() != self.name.lower():
            return None
        
        if len(parts) == 1:
            return self
        
        next_child = self.find_child_by_name(parts[1])
        if not next_child:
            return None
        
        return next_child.find_node_by_path(separator.join(parts[1:]), separator)
    
    def train_classifier(self, training_data, model_type='distilbert'):
        """
        Train a classifier for this topic node.
        
        Args:
            training_data (list): List of training examples
            model_type (str, optional): Type of model to use
        """
        # This would be implemented with a specific ML framework
        # For now, we just store the fact that we've "trained" a model
        self.classifier = {
            'model_type': model_type,
            'trained_at': datetime.utcnow().isoformat(),
            'examples_count': len(training_data)
        }
        
        # In a real implementation, we would save the model to disk
        # and store the path in classifier_model_path
    
    def save_to_db(self, session=None):
        """
        Save this node and all children to the database.
        
        Args:
            session (SQLAlchemy.Session, optional): Database session to use
        
        Returns:
            TopicTaxonomy: Database model for this node
        """
        if session is None:
            session = db.session
        
        # Create or update this node
        if self.db_model is None:
            # Create new model
            parent_id = self.parent.db_model.id if self.parent and self.parent.db_model else None
            
            self.db_model = TopicTaxonomy(
                name=self.name,
                parent_id=parent_id,
                description=self.description,
                keywords=self.keywords,
                domain=self.domain,
                classifier_model_path=None
            )
            session.add(self.db_model)
        else:
            # Update existing model
            self.db_model.name = self.name
            self.db_model.description = self.description
            self.db_model.keywords = self.keywords
            self.db_model.domain = self.domain
        
        # Save classifier path if we have a classifier
        if self.classifier:
            # In a real implementation, this would be a path to a saved model
            self.db_model.classifier_model_path = f"models/{self.domain}/{self.get_full_name('_').lower()}.pkl"
        
        # Commit to get the ID
        session.commit()
        
        # Save all children
        for child in self.children:
            child.save_to_db(session)
        
        return self.db_model
    
    @classmethod
    def load_from_db(cls, topic_id=None, domain=None, session=None):
        """
        Load a TopicNode tree from the database.
        
        Args:
            topic_id (int, optional): ID of the root topic to load
            domain (str, optional): Domain to load topics for (if no topic_id)
            session (SQLAlchemy.Session, optional): Database session to use
        
        Returns:
            TopicNode: Root node of the loaded tree
        """
        if session is None:
            session = db.session
            
        # Find root nodes
        if topic_id:
            db_root = session.query(TopicTaxonomy).get(topic_id)
            db_roots = [db_root] if db_root else []
        else:
            query = session.query(TopicTaxonomy).filter(TopicTaxonomy.parent_id.is_(None))
            if domain:
                query = query.filter(TopicTaxonomy.domain == domain)
            db_roots = query.all()
        
        if not db_roots:
            return None
        
        # Create nodes for all topics (without building tree yet)
        nodes_by_id = {}
        for db_topic in session.query(TopicTaxonomy).all():
            node = cls(
                name=db_topic.name,
                keywords=db_topic.keywords,
                domain=db_topic.domain,
                description=db_topic.description
            )
            node.db_model = db_topic
            nodes_by_id[db_topic.id] = node
        
        # Build tree structure
        for db_topic in session.query(TopicTaxonomy).all():
            if db_topic.parent_id and db_topic.id in nodes_by_id and db_topic.parent_id in nodes_by_id:
                parent_node = nodes_by_id[db_topic.parent_id]
                node = nodes_by_id[db_topic.id]
                parent_node.add_child(node)
        
        # Return the requested root(s)
        if topic_id:
            return nodes_by_id.get(topic_id)
        
        if domain:
            domain_roots = [node for node in nodes_by_id.values() 
                           if node.domain == domain and node.parent is None]
            return domain_roots
        
        return [node for node in nodes_by_id.values() if node.parent is None]

    @staticmethod
    def get_all_topics_by_domain(domain, session=None):
        """
        Get all topics for a specific domain.
        
        Args:
            domain (str): Domain to get topics for
            session (SQLAlchemy.Session, optional): Database session to use
        
        Returns:
            list: List of TopicTaxonomy instances
        """
        if session is None:
            session = db.session
            
        return session.query(TopicTaxonomy).filter(TopicTaxonomy.domain == domain).all()
