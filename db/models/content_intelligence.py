"""
Content Intelligence Models
==========================

This module defines the SQLAlchemy ORM models for content intelligence,
including temporal analysis, content enrichment, and content relationships.
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSONB
import json

# This would be imported from a central configuration
db = SQLAlchemy()

class TemporalReference(db.Model):
    """SQLAlchemy model for the temporal_references table."""
    
    __tablename__ = 'temporal_references'
    
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('scraper_content.id'))
    reference_date = db.Column(db.Date)
    reference_type = db.Column(db.String(50))  # PUBLICATION, MENTIONED, FUTURE_EVENT, etc.
    confidence = db.Column(db.Float)
    extracted_text = db.Column(db.Text)
    context = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<TemporalReference {self.reference_date} ({self.reference_type})>'
    
    def to_dict(self):
        """Convert the TemporalReference instance to a dictionary."""
        return {
            'id': self.id,
            'content_id': self.content_id,
            'reference_date': self.reference_date.isoformat() if self.reference_date else None,
            'reference_type': self.reference_type,
            'confidence': self.confidence,
            'extracted_text': self.extracted_text,
            'context': self.context,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def get_references_by_content(cls, content_id, session=None):
        """Get all temporal references for a specific content item."""
        if session is None:
            session = db.session
            
        return session.query(cls).filter(cls.content_id == content_id).order_by(cls.reference_date).all()
    
    @classmethod
    def get_references_by_date_range(cls, start_date, end_date, session=None):
        """Get temporal references within a date range."""
        if session is None:
            session = db.session
            
        return session.query(cls).filter(
            cls.reference_date >= start_date,
            cls.reference_date <= end_date
        ).order_by(cls.reference_date).all()


class DomainEvent(db.Model):
    """SQLAlchemy model for the domain_events table."""
    
    __tablename__ = 'domain_events'
    
    id = db.Column(db.Integer, primary_key=True)
    domain = db.Column(db.String(50), nullable=False)
    event_name = db.Column(db.String(255), nullable=False)
    event_date = db.Column(db.Date, nullable=False)
    event_type = db.Column(db.String(50))
    description = db.Column(db.Text)
    entities = db.Column(JSONB)  # Linked entities involved in this event
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<DomainEvent {self.event_name} ({self.event_date})>'
    
    def to_dict(self):
        """Convert the DomainEvent instance to a dictionary."""
        return {
            'id': self.id,
            'domain': self.domain,
            'event_name': self.event_name,
            'event_date': self.event_date.isoformat() if self.event_date else None,
            'event_type': self.event_type,
            'description': self.description,
            'entities': self.entities,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def get_upcoming_events(cls, domain, days_ahead=30, session=None):
        """Get upcoming events for a specific domain."""
        if session is None:
            session = db.session
            
        from datetime import date, timedelta
        today = date.today()
        future_date = today + timedelta(days=days_ahead)
        
        return session.query(cls).filter(
            cls.domain == domain,
            cls.event_date >= today,
            cls.event_date <= future_date
        ).order_by(cls.event_date).all()
    
    @classmethod
    def get_events_by_entity(cls, entity_id, session=None):
        """Get events related to a specific entity."""
        if session is None:
            session = db.session
            
        # This query assumes that entities is a JSONB field with an array of entity IDs
        return session.query(cls).filter(
            cls.entities.contains([{"id": entity_id}])
        ).order_by(cls.event_date).all()


class TemporalRelevanceScore(db.Model):
    """SQLAlchemy model for the temporal_relevance_scores table."""
    
    __tablename__ = 'temporal_relevance_scores'
    
    content_id = db.Column(db.Integer, db.ForeignKey('scraper_content.id'), primary_key=True)
    calculated_at = db.Column(db.DateTime, default=datetime.utcnow, primary_key=True)
    relevance_score = db.Column(db.Float, nullable=False)
    recency_factor = db.Column(db.Float)
    domain_factor = db.Column(db.Float)
    future_event_factor = db.Column(db.Float)
    scoring_factors = db.Column(JSONB)
    
    def __repr__(self):
        return f'<TemporalRelevanceScore content_id={self.content_id} score={self.relevance_score}>'
    
    def to_dict(self):
        """Convert the TemporalRelevanceScore instance to a dictionary."""
        return {
            'content_id': self.content_id,
            'calculated_at': self.calculated_at.isoformat() if self.calculated_at else None,
            'relevance_score': self.relevance_score,
            'recency_factor': self.recency_factor,
            'domain_factor': self.domain_factor,
            'future_event_factor': self.future_event_factor,
            'scoring_factors': self.scoring_factors
        }
    
    @classmethod
    def get_latest_score(cls, content_id, session=None):
        """Get the latest temporal relevance score for a content item."""
        if session is None:
            session = db.session
            
        return session.query(cls).filter(
            cls.content_id == content_id
        ).order_by(cls.calculated_at.desc()).first()
    
    @classmethod
    def get_most_relevant_content(cls, topic_id=None, limit=10, session=None):
        """Get the most temporally relevant content."""
        if session is None:
            session = db.session
            
        query = session.query(cls)
        
        # If topic_id is provided, filter by topic
        if topic_id:
            from db.models.topic_taxonomy import ContentClassification
            query = query.join(
                ContentClassification,
                ContentClassification.content_id == cls.content_id
            ).filter(ContentClassification.topic_id == topic_id)
        
        # Get the latest score for each content item
        subquery = query.distinct(cls.content_id).order_by(
            cls.content_id, 
            cls.calculated_at.desc()
        ).subquery()
        
        # Order by relevance score and limit
        result = session.query(subquery).order_by(
            subquery.c.relevance_score.desc()
        ).limit(limit).all()
        
        return result


class EnhancedContent(db.Model):
    """SQLAlchemy model for the enhanced_content table."""
    
    __tablename__ = 'enhanced_content'
    
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('scraper_content.id'))
    enhanced_metadata = db.Column(JSONB, nullable=False)
    augmented_context = db.Column(JSONB)
    knowledge_links = db.Column(JSONB)
    processing_version = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<EnhancedContent content_id={self.content_id}>'
    
    def to_dict(self):
        """Convert the EnhancedContent instance to a dictionary."""
        return {
            'id': self.id,
            'content_id': self.content_id,
            'enhanced_metadata': self.enhanced_metadata,
            'augmented_context': self.augmented_context,
            'knowledge_links': self.knowledge_links,
            'processing_version': self.processing_version,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def get_by_content_id(cls, content_id, session=None):
        """Get enhanced content by content ID."""
        if session is None:
            session = db.session
            
        return session.query(cls).filter(cls.content_id == content_id).first()
    
    @classmethod
    def get_latest_enhancements(cls, limit=10, session=None):
        """Get the latest enhanced content items."""
        if session is None:
            session = db.session
            
        return session.query(cls).order_by(cls.updated_at.desc()).limit(limit).all()


class ContentQualityMetric(db.Model):
    """SQLAlchemy model for the content_quality_metrics table."""
    
    __tablename__ = 'content_quality_metrics'
    
    content_id = db.Column(db.Integer, db.ForeignKey('scraper_content.id'), primary_key=True)
    readability_score = db.Column(db.Float)
    information_density = db.Column(db.Float)
    sentiment_score = db.Column(db.Float)
    objectivity_score = db.Column(db.Float)
    factual_density = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ContentQualityMetric content_id={self.content_id}>'
    
    def to_dict(self):
        """Convert the ContentQualityMetric instance to a dictionary."""
        return {
            'content_id': self.content_id,
            'readability_score': self.readability_score,
            'information_density': self.information_density,
            'sentiment_score': self.sentiment_score,
            'objectivity_score': self.objectivity_score,
            'factual_density': self.factual_density,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def get_overall_quality_score(self):
        """Calculate an overall quality score based on individual metrics."""
        # This is a simple example - in a real system, you'd have more sophisticated scoring
        metrics = [
            self.readability_score or 0,
            self.information_density or 0,
            self.objectivity_score or 0,
            self.factual_density or 0
        ]
        
        # Return average of available metrics
        available_metrics = [m for m in metrics if m > 0]
        if available_metrics:
            return sum(available_metrics) / len(available_metrics)
        return 0
    

class ContentRelationship(db.Model):
    """SQLAlchemy model for the content_relationships table."""
    
    __tablename__ = 'content_relationships'
    
    id = db.Column(db.Integer, primary_key=True)
    source_content_id = db.Column(db.Integer, db.ForeignKey('scraper_content.id'))
    target_content_id = db.Column(db.Integer, db.ForeignKey('scraper_content.id'))
    relationship_type = db.Column(db.String(50))
    similarity_score = db.Column(db.Float)
    shared_entities = db.Column(JSONB)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ContentRelationship {self.source_content_id} -> {self.target_content_id} ({self.relationship_type})>'
    
    def to_dict(self):
        """Convert the ContentRelationship instance to a dictionary."""
        return {
            'id': self.id,
            'source_content_id': self.source_content_id,
            'target_content_id': self.target_content_id,
            'relationship_type': self.relationship_type,
            'similarity_score': self.similarity_score,
            'shared_entities': self.shared_entities,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def get_related_content(cls, content_id, relationship_type=None, min_similarity=0.5, limit=5, session=None):
        """Get content related to the specified content item."""
        if session is None:
            session = db.session
            
        # Query for relationships where the specified content is the source
        source_query = session.query(cls).filter(
            cls.source_content_id == content_id,
            cls.similarity_score >= min_similarity
        )
        
        # Query for relationships where the specified content is the target
        target_query = session.query(cls).filter(
            cls.target_content_id == content_id,
            cls.similarity_score >= min_similarity
        )
        
        # Apply relationship type filter if specified
        if relationship_type:
            source_query = source_query.filter(cls.relationship_type == relationship_type)
            target_query = target_query.filter(cls.relationship_type == relationship_type)
        
        # Get results
        source_relationships = source_query.order_by(cls.similarity_score.desc()).limit(limit).all()
        target_relationships = target_query.order_by(cls.similarity_score.desc()).limit(limit).all()
        
        # Combine and return results
        related_content_ids = {rel.target_content_id for rel in source_relationships}
        related_content_ids.update({rel.source_content_id for rel in target_relationships})
        
        # Remove the original content ID if it's somehow in the results
        if content_id in related_content_ids:
            related_content_ids.remove(content_id)
        
        # This assumes there's a Content model imported from the scraper
        from app import Content
        return session.query(Content).filter(Content.id.in_(related_content_ids)).all()


# Football-specific models
class FootballMatch(db.Model):
    """SQLAlchemy model for the football_matches table."""
    
    __tablename__ = 'football_matches'
    
    id = db.Column(db.Integer, primary_key=True)
    home_team_id = db.Column(db.Integer, db.ForeignKey('football_teams.id'))
    away_team_id = db.Column(db.Integer, db.ForeignKey('football_teams.id'))
    competition = db.Column(db.String(100))
    match_date = db.Column(db.DateTime)
    venue = db.Column(db.String(100))
    home_score = db.Column(db.Integer)
    away_score = db.Column(db.Integer)
    status = db.Column(db.String(50))  # SCHEDULED, COMPLETED, POSTPONED, etc.
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    home_team = db.relationship('FootballTeam', foreign_keys=[home_team_id])
    away_team = db.relationship('FootballTeam', foreign_keys=[away_team_id])
    
    def __repr__(self):
        home_name = self.home_team.entity.name if self.home_team and self.home_team.entity else "Unknown"
        away_name = self.away_team.entity.name if self.away_team and self.away_team.entity else "Unknown"
        
        if self.status == 'COMPLETED' and self.home_score is not None and self.away_score is not None:
            return f'<FootballMatch {home_name} {self.home_score}-{self.away_score} {away_name} ({self.match_date})>'
        else:
            return f'<FootballMatch {home_name} vs {away_name} ({self.match_date})>'
    
    def to_dict(self):
        """Convert the FootballMatch instance to a dictionary."""
        return {
            'id': self.id,
            'home_team_id': self.home_team_id,
            'home_team': self.home_team.entity.name if self.home_team and self.home_team.entity else None,
            'away_team_id': self.away_team_id,
            'away_team': self.away_team.entity.name if self.away_team and self.away_team.entity else None,
            'competition': self.competition,
            'match_date': self.match_date.isoformat() if self.match_date else None,
            'venue': self.venue,
            'home_score': self.home_score,
            'away_score': self.away_score,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def get_upcoming_matches(cls, team_id=None, limit=10, session=None):
        """Get upcoming matches, optionally filtered by team."""
        if session is None:
            session = db.session
            
        from datetime import datetime
        now = datetime.utcnow()
        
        query = session.query(cls).filter(
            cls.match_date > now,
            cls.status != 'CANCELLED'
        ).order_by(cls.match_date)
        
        if team_id:
            query = query.filter(
                db.or_(
                    cls.home_team_id == team_id,
                    cls.away_team_id == team_id
                )
            )
        
        return query.limit(limit).all()


class FootballTransfer(db.Model):
    """SQLAlchemy model for the football_transfers table."""
    
    __tablename__ = 'football_transfers'
    
    id = db.Column(db.Integer, primary_key=True)
    player_id = db.Column(db.Integer, db.ForeignKey('football_players.id'))
    from_team_id = db.Column(db.Integer, db.ForeignKey('football_teams.id'))
    to_team_id = db.Column(db.Integer, db.ForeignKey('football_teams.id'))
    transfer_date = db.Column(db.Date)
    fee = db.Column(db.Numeric(15, 2))
    fee_currency = db.Column(db.String(10))
    contract_years = db.Column(db.Integer)
    transfer_type = db.Column(db.String(50))  # PERMANENT, LOAN, FREE, etc.
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    player = db.relationship('FootballPlayer')
    from_team = db.relationship('FootballTeam', foreign_keys=[from_team_id])
    to_team = db.relationship('FootballTeam', foreign_keys=[to_team_id])
    
    def __repr__(self):
        player_name = self.player.entity.name if self.player and self.player.entity else "Unknown"
        from_team = self.from_team.entity.name if self.from_team and self.from_team.entity else "Unknown"
        to_team = self.to_team.entity.name if self.to_team and self.to_team.entity else "Unknown"
        
        return f'<FootballTransfer {player_name}: {from_team} -> {to_team} ({self.transfer_date})>'
    
    def to_dict(self):
        """Convert the FootballTransfer instance to a dictionary."""
        return {
            'id': self.id,
            'player_id': self.player_id,
            'player': self.player.entity.name if self.player and self.player.entity else None,
            'from_team_id': self.from_team_id,
            'from_team': self.from_team.entity.name if self.from_team and self.from_team.entity else None,
            'to_team_id': self.to_team_id,
            'to_team': self.to_team.entity.name if self.to_team and self.to_team.entity else None,
            'transfer_date': self.transfer_date.isoformat() if self.transfer_date else None,
            'fee': float(self.fee) if self.fee is not None else None,
            'fee_currency': self.fee_currency,
            'contract_years': self.contract_years,
            'transfer_type': self.transfer_type,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def get_recent_transfers(cls, team_id=None, limit=10, session=None):
        """Get recent transfers, optionally filtered by team."""
        if session is None:
            session = db.session
            
        query = session.query(cls).order_by(cls.transfer_date.desc())
        
        if team_id:
            query = query.filter(
                db.or_(
                    cls.from_team_id == team_id,
                    cls.to_team_id == team_id
                )
            )
        
        return query.limit(limit).all()
