"""
Patch for integrating intelligence processing into the WebCrawler.

This file contains the necessary modifications to integrate intelligence processing
into the WebCrawler. It can be used to update the crawler.py file or to understand
the changes needed for integration.
"""

# Add import at the top of crawler.py
from scraper.intelligence_integration import IntelligenceProcessor

# Add intelligence processor to WebCrawler.__init__
def __init__(self, job_id):
    """
    Initialize a WebCrawler instance for a specific job
    
    Args:
        job_id (int): The ID of the scraping job
    """
    self.job_id = job_id
    self.job = None
    self.configuration = None
    self.visited_urls = set()
    self.failed_urls = set()
    self.robot_parsers = {}
    self.throttle = {}  # Domain-based throttling
    
    # Intelligence components
    self.link_intelligence = None  # Will be initialized after loading job
    self.page_quality_scores = {}  # Cache for page quality assessment
    self.domain_quality_scores = {}  # Track domain quality for adaptive decisions
    
    # Intelligent navigation enhancements
    self.url_parents = {}  # Store parent URL for each discovered URL
    self.content_html_cache = {}  # Limited cache of page HTML for context analysis
    
    # Intelligence processor for content analysis
    self.intelligence_processor = None  # Will be initialized after loading job
    
    # Register this job as active at initialization
    with self.__class__._lock:
        self.__class__.active_jobs[self.job_id] = False
        active_job_count = len(self.__class__.active_jobs)
        logger.info(f"Initialized crawler for job {job_id}. Active jobs: {active_job_count}, IDs: {list(self.__class__.active_jobs.keys())}")


# Update the load_job method to initialize the intelligence processor
def load_job(self):
    """Load the job and its configuration from the database"""
    from app import app, db
    
    # Make sure we're in an application context
    with app.app_context():
        self.job = ScrapingJob.query.get(self.job_id)
        if not self.job:
            raise ValueError(f"Job with ID {self.job_id} not found")
        
        self.configuration = self.job.configuration
        
        # Initialize link intelligence with base domains from configuration
        base_domains = [urlparse(url).netloc for url in self.configuration.base_urls]
        # Get unique domains
        unique_domains = list(set(base_domains))
        primary_domain = unique_domains[0] if unique_domains else None
        
        # Extract potential keywords from configuration description
        keywords = []
        if self.configuration.description:
            # Extract potential keywords from description
            import re
            words = re.findall(r'\b\w{4,}\b', self.configuration.description.lower())
            # Filter out common words
            stopwords = ['this', 'that', 'with', 'from', 'have', 'will', 'what', 'when', 'where', 'configuration']
            keywords = [w for w in words if w not in stopwords]
        
        # Initialize link intelligence
        self.link_intelligence = LinkIntelligence(primary_domain, keywords)
        logger.info(f"Initialized link intelligence for job {self.job_id} with domain {primary_domain}")
        if keywords:
            logger.info(f"Using content keywords: {', '.join(keywords)}")
        
        # Initialize intelligence processor if intelligence features are enabled
        try:
            # Check if intelligence configuration attributes exist
            enable_classification = getattr(self.configuration, 'enable_classification', False)
            enable_entity_extraction = getattr(self.configuration, 'enable_entity_extraction', False)
            intelligence_domain = getattr(self.configuration, 'intelligence_domain', 'football')
            
            if enable_classification or enable_entity_extraction:
                self.intelligence_processor = IntelligenceProcessor(
                    domain=intelligence_domain,
                    enable_classification=enable_classification,
                    enable_entity_extraction=enable_entity_extraction
                )
                logger.info(f"Initialized intelligence processor for job {self.job_id} with domain {intelligence_domain}")
        except Exception as e:
            logger.warning(f"Failed to initialize intelligence processor: {str(e)}")
            self.intelligence_processor = None
        
        return self.job


# Update the fetch_url method to use the intelligence processor
# This code should be inserted after creating the content object and before database operations
"""
# Create content object
content = ScrapedContent(
    job_id=self.job_id,
    url=url,
    title=title,
    raw_html=raw_html,
    extracted_text=extracted_text,
    crawl_depth=depth,
    processing_time=processing_time
)

# Apply intelligence processing if enabled and processor is available
if self.intelligence_processor:
    try:
        # Intelligence processor needs content.id, so we need to flush first
        db.session.add(content)
        db.session.flush()
        
        # Process content through intelligence pipelines
        intelligence_results = self.intelligence_processor.process_content(content)
        
        if intelligence_results:
            logger.info(f"Applied intelligence processing to content {content.id} in {intelligence_results['processing_time']:.2f}s")
            
            # Store intelligence processing time in metadata
            if content.content_metadata and intelligence_results['processing_time']:
                if not content.content_metadata.extra_data:
                    content.content_metadata.extra_data = {}
                content.content_metadata.extra_data['intelligence_processing_time'] = intelligence_results['processing_time']
                
                # Add information about classification if available
                if intelligence_results.get('classification'):
                    classification = intelligence_results['classification']
                    content.content_metadata.extra_data['has_classification'] = True
                    content.content_metadata.extra_data['primary_topic'] = classification.primary_topic
                
                # Add information about entities if available
                if intelligence_results.get('entities'):
                    entities = intelligence_results['entities']
                    content.content_metadata.extra_data['has_entities'] = True
                    content.content_metadata.extra_data['entity_count'] = len(entities.entities)
    except Exception as e:
        logger.error(f"Error in intelligence processing for {url}: {str(e)}")
        # Continue with normal processing even if intelligence processing fails
"""
