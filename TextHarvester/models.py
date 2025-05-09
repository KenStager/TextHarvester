from datetime import datetime
import enum
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Enum, JSON, Table, Float
from sqlalchemy.orm import relationship, backref

from app import db

# This will be defined after the Source and SourceList models

class ScrapingStatus(enum.Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'

class ScrapingConfiguration(db.Model):
    __tablename__ = 'scraping_configuration'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    base_urls = Column(JSON, nullable=False)  # List of base URLs to scrape
    max_depth = Column(Integer, default=1)  # How deep to follow links
    follow_external_links = Column(Boolean, default=False)
    respect_robots_txt = Column(Boolean, default=True)
    user_agent_rotation = Column(Boolean, default=True)
    rate_limit_seconds = Column(Integer, default=5)  # Rate limiting in seconds
    max_retries = Column(Integer, default=3)
    # Intelligent navigation settings
    enable_intelligent_navigation = Column(Boolean, default=True)
    quality_threshold = Column(Float, default=0.7)  # Quality score threshold for extending depth
    max_extended_depth = Column(Integer, default=2)  # Max levels beyond standard depth to allow
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    jobs = relationship("ScrapingJob", back_populates="configuration", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ScrapingConfiguration {self.name}>"

class ScrapingJob(db.Model):
    __tablename__ = 'scraping_job'
    
    id = Column(Integer, primary_key=True)
    configuration_id = Column(Integer, ForeignKey('scraping_configuration.id'), nullable=False)
    status = Column(Enum(ScrapingStatus), default=ScrapingStatus.PENDING, nullable=False)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    urls_processed = Column(Integer, default=0)
    urls_successful = Column(Integer, default=0)
    urls_failed = Column(Integer, default=0)
    log_output = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    configuration = relationship("ScrapingConfiguration", back_populates="jobs")
    contents = relationship("ScrapedContent", back_populates="job", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ScrapingJob {self.id} - {self.status.value}>"

class ScrapedContent(db.Model):
    __tablename__ = 'scraped_content'
    
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey('scraping_job.id'), nullable=False, index=True)
    url = Column(String(2048), nullable=False)
    title = Column(String(512), nullable=True)
    raw_html = Column(Text, nullable=True)
    extracted_text = Column(Text, nullable=False)
    crawl_depth = Column(Integer, default=0, index=True)
    processing_time = Column(Integer, nullable=True)  # Time in milliseconds
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    job = relationship("ScrapingJob", back_populates="contents")
    content_metadata = relationship("ContentMetadata", back_populates="content", cascade="all, delete-orphan", uselist=False)
    # The quality_metrics relationship is defined in the ContentQualityMetrics model using backref
    
    # Create an index on job_id and created_at for faster pagination queries
    __table_args__ = (
        db.Index('idx_content_job_created', 'job_id', 'created_at'),
    )
    
    def __repr__(self):
        return f"<ScrapedContent {self.id} - {self.url}>"
        
    @property
    def quality_score(self):
        """Get the quality score if quality metrics exist"""
        if hasattr(self, 'quality_metrics') and self.quality_metrics:
            return self.quality_metrics.quality_score
        return None

class ContentMetadata(db.Model):
    __tablename__ = 'content_metadata'
    
    id = Column(Integer, primary_key=True)
    content_id = Column(Integer, ForeignKey('scraped_content.id'), nullable=False)
    word_count = Column(Integer, nullable=True)
    char_count = Column(Integer, nullable=True)
    language = Column(String(10), nullable=True)
    content_type = Column(String(50), nullable=True)
    headers = Column(JSON, nullable=True)  # Store HTTP headers as JSON
    extra_data = Column(JSON, nullable=True)  # Additional metadata as JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    
    content = relationship("ScrapedContent", back_populates="content_metadata")
    
    def __repr__(self):
        return f"<ContentMetadata for Content {self.content_id}>"


class Source(db.Model):
    """
    Represents a single source URL to scrape
    """
    __tablename__ = 'source'
    
    id = Column(Integer, primary_key=True)
    url = Column(String(2048), nullable=False, unique=True)
    name = Column(String(255), nullable=True, index=True)
    description = Column(Text, nullable=True)
    category = Column(String(100), nullable=True, index=True)  # e.g., 'research', 'news', 'blog'
    priority = Column(Integer, default=0, index=True)  # Higher number = higher priority
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Source {self.name or self.url}>"


class SourceList(db.Model):
    """
    Represents a list of sources grouped together
    """
    __tablename__ = 'source_list'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, index=True)
    slug = Column(String(100), nullable=False, unique=True, index=True)  # URL-friendly identifier
    description = Column(Text, nullable=True)
    is_public = Column(Boolean, default=True, index=True) 
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<SourceList {self.name}>"

# Now define the many-to-many relationship with indexes
source_list_sources = Table(
    'source_list_sources',
    db.Model.metadata,
    Column('source_list_id', Integer, ForeignKey('source_list.id'), primary_key=True, index=True),
    Column('source_id', Integer, ForeignKey('source.id'), primary_key=True, index=True),
    # Add combined index for faster lookups
    db.Index('idx_list_source', 'source_list_id', 'source_id')
)

# Add relationships after the table is defined
Source.source_lists = relationship("SourceList", secondary=source_list_sources, back_populates="sources")
SourceList.sources = relationship("Source", secondary=source_list_sources, back_populates="source_lists")

class ContentQualityMetrics(db.Model):
    """Quality metrics for scraped content to support intelligent navigation"""
    __tablename__ = 'content_quality_metrics'
    
    id = Column(Integer, primary_key=True)
    content_id = Column(Integer, ForeignKey('scraped_content.id'), nullable=False, index=True)
    quality_score = Column(Float, nullable=False)
    word_count = Column(Integer, nullable=True)
    paragraph_count = Column(Integer, nullable=True)
    text_ratio = Column(Float, nullable=True)  # Text-to-HTML ratio
    domain = Column(String(255), nullable=True, index=True)
    domain_avg_score = Column(Float, nullable=True)
    parent_url = Column(String(2048), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to content
    content = relationship("ScrapedContent", backref=backref("quality_metrics", uselist=False))
