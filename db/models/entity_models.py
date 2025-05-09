"""
Entity Recognition System - ORM Models
=====================================

This module defines the SQLAlchemy ORM models for the entity recognition system,
including models for entity types, entities, and entity mentions.
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSONB
import json

# This would be imported from a central configuration
db = SQLAlchemy()

class EntityType(db.Model):
    """SQLAlchemy model for the entity_types table."""
    
    __tablename__ = 'entity_types'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    parent_id = db.Column(db.Integer, db.ForeignKey('entity_types.id'))
    domain = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    parent = db.relationship('EntityType', remote_side=[id], backref=db.backref('children', lazy='dynamic'))
    entities = db.relationship('Entity', backref='entity_type', lazy='dynamic')
    
    def __repr__(self):
        return f'<EntityType {self.name}>'
    
    def to_dict(self):
        """Convert the EntityType instance to a dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'parent_id': self.parent_id,
            'domain': self.domain,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def get_subtypes(self):
        """Get all subtypes of this entity type (direct children)."""
        return self.children.all()
    
    def get_all_subtypes(self):
        """Get all subtypes recursively."""
        subtypes = self.get_subtypes()
        for subtype in subtypes:
            subtypes.extend(subtype.get_all_subtypes())
        return subtypes
    
    def get_full_path(self, separator='.'):
        """Get the full type path from root to this type."""
        if self.parent is None:
            return self.name
        return f"{self.parent.get_full_path(separator)}{separator}{self.name}"
    
    @classmethod
    def get_or_create(cls, name, domain, parent=None, description=None, session=None):
        """Get an existing entity type or create a new one."""
        if session is None:
            session = db.session
            
        if parent and isinstance(parent, str):
            # If parent is provided as a string, look it up
            parent_obj = session.query(cls).filter(cls.name == parent, cls.domain == domain).first()
            if parent_obj:
                parent_id = parent_obj.id
            else:
                # Create parent if it doesn't exist
                parent_obj = cls(name=parent, domain=domain)
                session.add(parent_obj)
                session.flush()
                parent_id = parent_obj.id
        elif parent and isinstance(parent, cls):
            parent_id = parent.id
        else:
            parent_id = None
            
        entity_type = session.query(cls).filter(cls.name == name, cls.domain == domain).first()
        
        if not entity_type:
            entity_type = cls(
                name=name,
                domain=domain,
                parent_id=parent_id,
                description=description
            )
            session.add(entity_type)
            session.flush()
            
        return entity_type


class Entity(db.Model):
    """SQLAlchemy model for the entities table."""
    
    __tablename__ = 'entities'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    entity_type_id = db.Column(db.Integer, db.ForeignKey('entity_types.id'))
    canonical_name = db.Column(db.String(255))
    kb_id = db.Column(db.String(100))
    entity_metadata = db.Column(JSONB)  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    mentions = db.relationship('EntityMention', backref='entity', lazy='dynamic')
    
    # Football-specific relationships
    football_team = db.relationship('FootballTeam', uselist=False, backref='entity', lazy='joined')
    football_player = db.relationship('FootballPlayer', uselist=False, backref='entity', lazy='joined')
    
    def __repr__(self):
        return f'<Entity {self.name} (type: {self.entity_type.name if self.entity_type else "None"})>'
    
    def to_dict(self):
        """Convert the Entity instance to a dictionary."""
        entity_dict = {
            'id': self.id,
            'name': self.name,
            'entity_type_id': self.entity_type_id,
            'canonical_name': self.canonical_name,
            'kb_id': self.kb_id,
            'metadata': self.entity_metadata,  # Use entity_metadata but keep key as 'metadata' for compatibility
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
        
        # Add type information if available
        if self.entity_type:
            entity_dict['entity_type'] = self.entity_type.name
            entity_dict['entity_type_path'] = self.entity_type.get_full_path()
        
        # Add domain-specific extensions if available
        if self.football_team:
            entity_dict['football_team'] = {
                'league': self.football_team.league,
                'country': self.football_team.country,
                'city': self.football_team.city,
                'stadium': self.football_team.stadium
            }
        
        if self.football_player:
            entity_dict['football_player'] = {
                'position': self.football_player.position,
                'nationality': self.football_player.nationality,
                'team': self.football_player.current_team.name if self.football_player.current_team else None
            }
            
        return entity_dict
    
    @classmethod
    def get_or_create(cls, name, entity_type, canonical_name=None, kb_id=None, metadata=None, session=None):
        """Get an existing entity or create a new one."""
        if session is None:
            session = db.session
            
        # If entity_type is provided as a string, look it up
        if isinstance(entity_type, str):
            entity_type_obj = EntityType.query.filter_by(name=entity_type).first()
            if not entity_type_obj:
                raise ValueError(f"Entity type '{entity_type}' not found")
            entity_type_id = entity_type_obj.id
        elif isinstance(entity_type, EntityType):
            entity_type_id = entity_type.id
        else:
            entity_type_id = entity_type
            
        # Try to find existing entity by kb_id first if provided
        if kb_id:
            entity = session.query(cls).filter(cls.kb_id == kb_id).first()
            if entity:
                return entity
        
        # Then try by canonical name if provided
        if canonical_name:
            entity = session.query(cls).filter(
                cls.canonical_name == canonical_name,
                cls.entity_type_id == entity_type_id
            ).first()
            if entity:
                return entity
        
        # Finally try by name
        entity = session.query(cls).filter(
            cls.name == name,
            cls.entity_type_id == entity_type_id
        ).first()
        
        if not entity:
            entity = cls(
                name=name,
                entity_type_id=entity_type_id,
                canonical_name=canonical_name or name,
                kb_id=kb_id,
                entity_metadata=metadata or {}  # Use entity_metadata instead of metadata
            )
            session.add(entity)
            session.flush()
            
        return entity
    
    def merge_with(self, other_entity, session=None):
        """Merge another entity into this one."""
        if session is None:
            session = db.session
            
        # Update all mentions of the other entity to point to this one
        session.query(EntityMention).filter(
            EntityMention.entity_id == other_entity.id
        ).update({
            'entity_id': self.id
        })
        
        # Update all relationships
        session.query(EntityRelationship).filter(
            EntityRelationship.source_entity_id == other_entity.id
        ).update({
            'source_entity_id': self.id
        })
        
        session.query(EntityRelationship).filter(
            EntityRelationship.target_entity_id == other_entity.id
        ).update({
            'target_entity_id': self.id
        })
        
        # Transfer any domain-specific information
        if other_entity.football_team and not self.football_team:
            other_entity.football_team.entity_id = self.id
            
        if other_entity.football_player and not self.football_player:
            other_entity.football_player.entity_id = self.id
        
        # Mark the other entity for deletion
        session.delete(other_entity)
        session.flush()
        
        return self


class EntityMention(db.Model):
    """SQLAlchemy model for the entity_mentions table."""
    
    __tablename__ = 'entity_mentions'
    
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('scraper_content.id'))
    entity_id = db.Column(db.Integer, db.ForeignKey('entities.id'))
    start_char = db.Column(db.Integer, nullable=False)
    end_char = db.Column(db.Integer, nullable=False)
    mention_text = db.Column(db.Text, nullable=False)
    confidence = db.Column(db.Float)
    context_before = db.Column(db.Text)
    context_after = db.Column(db.Text)
    human_verified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<EntityMention "{self.mention_text}" (entity_id: {self.entity_id})>'
    
    def to_dict(self):
        """Convert the EntityMention instance to a dictionary."""
        return {
            'id': self.id,
            'content_id': self.content_id,
            'entity_id': self.entity_id,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'mention_text': self.mention_text,
            'confidence': self.confidence,
            'context_before': self.context_before,
            'context_after': self.context_after,
            'human_verified': self.human_verified,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def find_mentions_in_content(cls, content_id, session=None):
        """Find all entity mentions in a specific content item."""
        if session is None:
            session = db.session
            
        return session.query(cls).filter(cls.content_id == content_id).all()
    
    @classmethod
    def find_content_with_entity(cls, entity_id, limit=10, session=None):
        """Find content that mentions a specific entity."""
        if session is None:
            session = db.session
            
        mentions = session.query(cls).filter(cls.entity_id == entity_id).limit(limit).all()
        
        # Get content IDs
        content_ids = [mention.content_id for mention in mentions]
        
        # This assumes there's a Content model imported from the scraper
        from app import Content
        return session.query(Content).filter(Content.id.in_(content_ids)).all()


class EntityRelationship(db.Model):
    """SQLAlchemy model for the entity_relationships table."""
    
    __tablename__ = 'entity_relationships'
    
    id = db.Column(db.Integer, primary_key=True)
    source_entity_id = db.Column(db.Integer, db.ForeignKey('entities.id'))
    target_entity_id = db.Column(db.Integer, db.ForeignKey('entities.id'))
    relationship_type = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float)
    relation_metadata = db.Column(JSONB)  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    source_entity = db.relationship('Entity', foreign_keys=[source_entity_id], backref='outgoing_relationships')
    target_entity = db.relationship('Entity', foreign_keys=[target_entity_id], backref='incoming_relationships')
    
    def __repr__(self):
        return f'<EntityRelationship {self.source_entity_id} -> {self.relationship_type} -> {self.target_entity_id}>'
    
    def to_dict(self):
        """Convert the EntityRelationship instance to a dictionary."""
        return {
            'id': self.id,
            'source_entity_id': self.source_entity_id,
            'target_entity_id': self.target_entity_id,
            'relationship_type': self.relationship_type,
            'confidence': self.confidence,
            'metadata': self.relation_metadata,  # Use relation_metadata but keep key as 'metadata' for compatibility
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def get_or_create(cls, source_entity, target_entity, relationship_type, 
                       confidence=None, metadata=None, start_date=None, end_date=None, 
                       session=None):
        """Get an existing relationship or create a new one."""
        if session is None:
            session = db.session
            
        # Convert entity parameters to IDs if needed
        if isinstance(source_entity, Entity):
            source_entity_id = source_entity.id
        else:
            source_entity_id = source_entity
            
        if isinstance(target_entity, Entity):
            target_entity_id = target_entity.id
        else:
            target_entity_id = target_entity
            
        # Look for existing relationship
        relationship = session.query(cls).filter(
            cls.source_entity_id == source_entity_id,
            cls.target_entity_id == target_entity_id,
            cls.relationship_type == relationship_type
        ).first()
        
        if not relationship:
            relationship = cls(
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relationship_type=relationship_type,
            confidence=confidence,
            relation_metadata=metadata or {},  # Use relation_metadata instead of metadata
            start_date=start_date,
            end_date=end_date
            )
            session.add(relationship)
            session.flush()
            
        return relationship
    
    @classmethod
    def find_related_entities(cls, entity_id, relationship_type=None, session=None):
        """Find entities related to the specified entity."""
        if session is None:
            session = db.session
            
        # Base query for outgoing relationships
        outgoing_query = session.query(Entity).join(
            cls, cls.target_entity_id == Entity.id
        ).filter(cls.source_entity_id == entity_id)
        
        # Base query for incoming relationships
        incoming_query = session.query(Entity).join(
            cls, cls.source_entity_id == Entity.id
        ).filter(cls.target_entity_id == entity_id)
        
        # Apply relationship type filter if specified
        if relationship_type:
            outgoing_query = outgoing_query.filter(cls.relationship_type == relationship_type)
            incoming_query = incoming_query.filter(cls.relationship_type == relationship_type)
        
        # Get results
        outgoing_entities = outgoing_query.all()
        incoming_entities = incoming_query.all()
        
        return {
            'outgoing': outgoing_entities,
            'incoming': incoming_entities,
            'all': outgoing_entities + incoming_entities
        }


# Football-specific entity extension models
class FootballTeam(db.Model):
    """SQLAlchemy model for the football_teams table."""
    
    __tablename__ = 'football_teams'
    
    id = db.Column(db.Integer, primary_key=True)
    entity_id = db.Column(db.Integer, db.ForeignKey('entities.id'))
    league = db.Column(db.String(100))
    country = db.Column(db.String(100))
    city = db.Column(db.String(100))
    stadium = db.Column(db.String(100))
    founded_year = db.Column(db.Integer)
    team_colors = db.Column(db.Text)
    nickname = db.Column(db.Text)
    website = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships with players
    players = db.relationship('FootballPlayer', foreign_keys='FootballPlayer.current_team_id', backref='current_team')
    
    def __repr__(self):
        return f'<FootballTeam {self.entity.name if self.entity else "Unknown"} ({self.league})>'
    
    def to_dict(self):
        """Convert the FootballTeam instance to a dictionary."""
        return {
            'id': self.id,
            'entity_id': self.entity_id,
            'name': self.entity.name if self.entity else None,
            'league': self.league,
            'country': self.country,
            'city': self.city,
            'stadium': self.stadium,
            'founded_year': self.founded_year,
            'team_colors': self.team_colors,
            'nickname': self.nickname,
            'website': self.website,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class FootballPlayer(db.Model):
    """SQLAlchemy model for the football_players table."""
    
    __tablename__ = 'football_players'
    
    id = db.Column(db.Integer, primary_key=True)
    entity_id = db.Column(db.Integer, db.ForeignKey('entities.id'))
    current_team_id = db.Column(db.Integer, db.ForeignKey('football_teams.id'))
    nationality = db.Column(db.String(100))
    birth_date = db.Column(db.Date)
    position = db.Column(db.String(50))
    jersey_number = db.Column(db.Integer)
    height = db.Column(db.Integer)  # in cm
    weight = db.Column(db.Integer)  # in kg
    preferred_foot = db.Column(db.String(10))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<FootballPlayer {self.entity.name if self.entity else "Unknown"} ({self.position})>'
    
    def to_dict(self):
        """Convert the FootballPlayer instance to a dictionary."""
        return {
            'id': self.id,
            'entity_id': self.entity_id,
            'name': self.entity.name if self.entity else None,
            'current_team_id': self.current_team_id,
            'current_team': self.current_team.entity.name if self.current_team and self.current_team.entity else None,
            'nationality': self.nationality,
            'birth_date': self.birth_date.isoformat() if self.birth_date else None,
            'position': self.position,
            'jersey_number': self.jersey_number,
            'height': self.height,
            'weight': self.weight,
            'preferred_foot': self.preferred_foot,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def get_career_history(self, session=None):
        """Get the player's career history (transfers)."""
        if session is None:
            session = db.session
            
        from db.models.football_models import FootballTransfer
        
        transfers = session.query(FootballTransfer).filter(
            FootballTransfer.player_id == self.id
        ).order_by(FootballTransfer.transfer_date).all()
        
        return transfers
