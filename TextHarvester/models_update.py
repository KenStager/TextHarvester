"""
Database model additions for intelligence integration.

This module defines new models for storing intelligence processing results
and updates to existing models to support intelligence features.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship, backref

from app import db

# New models for intelligence integration
class ContentClassification(db.Model):
    """Classification results for scraped content"""
    __tablename__ = 'content_classification'
    
    id = Column(Integer, primary_key=True)
    content_id = Column(Integer, ForeignKey('scraped_content.id'), nullable=False, index=True)
    is_relevant = Column(Boolean, default=False)
    confidence = Column(Float, nullable=False)
    primary_topic = Column(String(255), nullable=True, index=True)
    primary_topic_id = Column(String(100), nullable=True)
    primary_topic_confidence = Column(Float, nullable=True)
    subtopics = Column(JSON, nullable=True)
    processing_time = Column(Float, nullable=True)  # in seconds
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to content
    content = relationship("ScrapedContent", backref=backref("classification", uselist=False))
    
    def __repr__(self):
        return f"<ContentClassification {self.id} - {self.primary_topic}>"


class ContentEntity(db.Model):
    """Extracted entities from scraped content"""
    __tablename__ = 'content_entity'
    
    id = Column(Integer, primary_key=True)
    content_id = Column(Integer, ForeignKey('scraped_content.id'), nullable=False, index=True)
    entity_type = Column(String(100), nullable=False, index=True)
    entity_text = Column(String(1024), nullable=False)
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    entity_id = Column(String(255), nullable=True, index=True)  # For linked entities
    entity_metadata = Column(JSON, nullable=True)  # Renamed from 'metadata' to avoid SQLAlchemy name conflict
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to content
    content = relationship("ScrapedContent", backref=backref("entities", cascade="all, delete-orphan"))
    
    def __repr__(self):
        return f"<ContentEntity {self.id} - {self.entity_type}:{self.entity_text}>"


# Updates to existing models (to be applied via migration)
"""
The following SQL can be used to add intelligence columns to the scraping_configuration table:

ALTER TABLE scraping_configuration 
ADD COLUMN enable_classification BOOLEAN DEFAULT FALSE,
ADD COLUMN enable_entity_extraction BOOLEAN DEFAULT FALSE,
ADD COLUMN intelligence_domain VARCHAR(50) DEFAULT 'football',
ADD COLUMN intelligence_config JSON;
"""

# This class redefines ScrapingConfiguration with the new columns
# It should be used as a reference for updating the actual model
class ScrapingConfigurationUpdate:
    # Intelligence settings
    enable_classification = Column(Boolean, default=False)
    enable_entity_extraction = Column(Boolean, default=False)
    intelligence_domain = Column(String(50), default="football")
    intelligence_config = Column(JSON, nullable=True)  # Additional configuration options
