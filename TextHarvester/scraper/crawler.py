import time
import random
import logging
import requests
from datetime import datetime
import threading
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from requests.exceptions import RequestException

from models import ScrapingJob, ScrapedContent, ContentMetadata, ScrapingStatus
from scraper.content_extractor import extract_content
from scraper.utils import get_random_user_agent, get_domain_from_url
from scraper.path_intelligence import LinkIntelligence, evaluate_page_quality

# Thread local storage for app contexts
thread_local = threading.local()

logger = logging.getLogger(__name__)

class WebCrawler:
    # Class-level job status tracking
    # This allows us to stop jobs that might be running in separate threads
    active_jobs = {}  # job_id -> stop_flag
    _lock = threading.RLock()  # Lock for thread-safe access to active_jobs
    
    @classmethod
    def stop_job(cls, job_id):
        """
        Signal a running job to stop processing
        
        Args:
            job_id (int): The ID of the scraping job to stop
            
        Returns:
            bool: True if job was active and flagged to stop, False otherwise
        """
        with cls._lock:
            if job_id in cls.active_jobs:
                cls.active_jobs[job_id] = True
                logger.info(f"Job {job_id} has been flagged to stop")
                return True
            return False
    
    @classmethod
    def check_if_should_stop(cls, job_id):
        """Check if a job has been flagged to stop"""
        with cls._lock:
            return cls.active_jobs.get(job_id, False)
    
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
        
        # Register this job as active at initialization
        with self.__class__._lock:
            self.__class__.active_jobs[self.job_id] = False
            active_job_count = len(self.__class__.active_jobs)
            logger.info(f"Initialized crawler for job {job_id}. Active jobs: {active_job_count}, IDs: {list(self.__class__.active_jobs.keys())}")

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
            
            return self.job

    def update_job_status(self, status, log=None):
        """Update the status of the scraping job using direct SQL for reliability"""
        from app import app, db
        from sqlalchemy import text
        
        # Make sure we're in an application context
        with app.app_context():
            try:
                if not self.job:
                    self.load_job()
                    
                current_time = datetime.utcnow()
                
                # Sanitize log message to remove NUL characters that can cause database errors
                log_entry = None
                if log:
                    log = log.replace('\x00', '')
                    timestamp = f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}]"
                    log_entry = f"{timestamp} {log}"
                    
                    # Update log entry in memory for future references
                    if self.job.log_output:
                        self.job.log_output += f"\n{log_entry}"
                    else:
                        self.job.log_output = log_entry
                
                # Prepare direct SQL update parameters
                params = {
                    'job_id': self.job_id,
                    'status': status.name,  # Use enum name (RUNNING, PENDING, etc.) instead of value
                }
                
                # Set appropriate timestamps based on status
                set_clauses = ["status = :status"]
                
                if log_entry:
                    set_clauses.append("log_output = CASE WHEN log_output IS NULL THEN :log_entry ELSE log_output || '\n' || :log_entry END")
                    params['log_entry'] = log_entry
                
                if status == ScrapingStatus.RUNNING:
                    set_clauses.append("start_time = CASE WHEN start_time IS NULL THEN :current_time ELSE start_time END")
                    params['current_time'] = current_time
                    # Update in-memory job object too
                    if not self.job.start_time:
                        self.job.start_time = current_time
                
                if status in [ScrapingStatus.COMPLETED, ScrapingStatus.FAILED]:
                    set_clauses.append("end_time = :end_time")
                    params['end_time'] = current_time
                    # Update in-memory job object too
                    self.job.end_time = current_time
                
                # Always update timestamp
                set_clauses.append("updated_at = CURRENT_TIMESTAMP")
                
                # Build and execute the SQL query
                sql = text(f"""
                    UPDATE scraping_job 
                    SET {', '.join(set_clauses)}
                    WHERE id = :job_id
                """)
                
                # Execute direct SQL update
                db.session.execute(sql, params)
                
                # Commit the transaction
                db.session.commit()
                
                # Update in-memory status
                self.job.status = status
                
                logger.info(f"Job {self.job_id} status updated to {status.value} via direct SQL" + (f": {log}" if log else ""))
                
            except Exception as e:
                logger.exception(f"Error updating job status: {str(e)}")

    def can_fetch(self, url):
        """Check if the URL can be fetched according to robots.txt"""
        if not self.configuration.respect_robots_txt:
            return True
        
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        if domain not in self.robot_parsers:
            robots_url = f"{parsed_url.scheme}://{domain}/robots.txt"
            rp = RobotFileParser()
            rp.set_url(robots_url)
            try:
                rp.read()
                self.robot_parsers[domain] = rp
            except Exception as e:
                logger.warning(f"Error reading robots.txt for {domain}: {e}")
                # If we can't read robots.txt, we'll allow fetching
                return True
        
        return self.robot_parsers[domain].can_fetch(get_random_user_agent() if self.configuration.user_agent_rotation else '*', url)

    def throttle_request(self, url):
        """Apply rate limiting based on domain"""
        domain = get_domain_from_url(url)
        now = time.time()
        
        if domain in self.throttle:
            last_request_time = self.throttle[domain]
            time_since_last = now - last_request_time
            
            if time_since_last < self.configuration.rate_limit_seconds:
                sleep_time = self.configuration.rate_limit_seconds - time_since_last
                # Add some randomness to avoid detection
                sleep_time += random.uniform(0, 1)
                time.sleep(sleep_time)
        
        self.throttle[domain] = time.time()

    def fetch_url(self, url, depth=0):
        """
        Fetch content from a URL and process it
        
        Args:
            url (str): The URL to fetch
            depth (int): Current crawl depth
        
        Returns:
            tuple: (success, extracted_links)
        """
        # We assume this is called from within an application context
        # since process_domain_urls establishes the context
        from app import db
        
        if url in self.visited_urls or url in self.failed_urls:
            return False, []
        
        if not self.can_fetch(url):
            log_msg = f"Skipping {url} due to robots.txt rules"
            logger.info(log_msg)
            self.failed_urls.add(url)
            return False, []
        
        self.throttle_request(url)
        
        headers = {"User-Agent": get_random_user_agent()} if self.configuration.user_agent_rotation else {}
        start_time = time.time()
        success = False
        extracted_links = []
        
        for attempt in range(self.configuration.max_retries + 1):
            try:
                logger.info(f"Fetching {url} (attempt {attempt+1})")
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                # Extract content
                title, extracted_text, raw_html = extract_content(url, response.content)
                
                # Get links for further crawling
                if depth < self.configuration.max_depth:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    extracted_links = self._extract_links(url, soup)
                
                # Calculate processing time in milliseconds
                processing_time = int((time.time() - start_time) * 1000)
                
                # Store the content
                content = ScrapedContent(
                    job_id=self.job_id,
                    url=url,
                    title=title,
                    raw_html=raw_html,
                    extracted_text=extracted_text,
                    crawl_depth=depth,
                    processing_time=processing_time
                )
                
                # Safely handle database operations with retry mechanism
                try:
                    # Use a single transaction for content and metadata only
                    # Statistics will be updated in batches by the process_domain_urls method
                    from app import app
                    with app.app_context():
                        # Make sure we have the latest job state
                        self.load_job()
                    
                        # Add content
                        db.session.add(content)
                        db.session.flush()  # Get the ID without committing
                        
                        # Create metadata
                        metadata = ContentMetadata(
                            content_id=content.id,
                            word_count=len(extracted_text.split()) if extracted_text else 0,
                            char_count=len(extracted_text) if extracted_text else 0,
                            headers=dict(response.headers),
                            content_type=response.headers.get('Content-Type'),
                            extra_data={}
                        )
                        
                        db.session.add(metadata)
                        db.session.commit()
                except Exception as db_error:
                    logger.warning(f"Database error in fetch_url, attempting rollback: {str(db_error)}")
                    db.session.rollback()
                    # Try once more with a clean session
                    try:
                        # Use a single transaction with a fresh app context
                        from app import app
                        with app.app_context():
                            # Reload job to get fresh data
                            self.load_job()
                            
                            # Recreate the content and metadata objects
                            content = ScrapedContent(
                                job_id=self.job_id,
                                url=url,
                                title=title,
                                raw_html=raw_html,
                                extracted_text=extracted_text,
                                crawl_depth=depth,
                                processing_time=processing_time
                            )
                            
                            db.session.add(content)
                            db.session.flush()
                            
                            metadata = ContentMetadata(
                                content_id=content.id,
                                word_count=len(extracted_text.split()) if extracted_text else 0,
                                char_count=len(extracted_text) if extracted_text else 0,
                                headers=dict(response.headers),
                                content_type=response.headers.get('Content-Type'),
                                extra_data={}
                            )
                            
                            db.session.add(metadata)
                            
                            # Only commit the content and metadata
                            db.session.commit()
                            
                            # Note: Statistics are now managed by the process_domain_urls method
                    except Exception as retry_error:
                        logger.error(f"Failed to recover from database error: {str(retry_error)}")
                        raise
                
                self.visited_urls.add(url)
                success = True
                break
                
            except RequestException as e:
                logger.warning(f"Error fetching {url} (attempt {attempt+1}): {str(e)}")
                if attempt == self.configuration.max_retries:
                    self.failed_urls.add(url)
                    try:
                        # Update statistics using the dedicated method that handles app context
                        self.update_statistics(processed=1, failed=1)
                        
                        # Log the failure separately
                        from app import app
                        with app.app_context():
                            log_msg = f"Failed to fetch {url} after {self.configuration.max_retries + 1} attempts: {str(e)}"
                            self.update_job_status(ScrapingStatus.RUNNING, log=log_msg)
                    except Exception as stats_error:
                        logger.error(f"Error updating statistics for failed URL: {str(stats_error)}")
                        # Try to log the failure without updating statistics
                        try:
                            from app import app
                            with app.app_context():
                                log_msg = f"Failed to fetch {url} after {self.configuration.max_retries + 1} attempts: {str(e)}"
                                self.update_job_status(ScrapingStatus.RUNNING, log=log_msg)
                        except Exception as log_error:
                            logger.error(f"Failed to even log error: {str(log_error)}")
                else:
                    # Wait before retrying
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return success, extracted_links

    def _extract_links(self, base_url, soup):
        """
        Extract and prioritize links from a webpage using intelligent ranking
        
        Args:
            base_url (str): The URL of the page containing the links
            soup (BeautifulSoup): Parsed HTML content
            
        Returns:
            list: Prioritized list of (url, score) tuples
        """
        scored_links = []
        domain = get_domain_from_url(base_url)
        
        # Find all links
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            # Skip empty links
            if not href or href.startswith('#'):
                continue
                
            # Convert relative URLs to absolute
            full_url = urljoin(base_url, href)
            
            # Skip non-HTTP URLs (like mailto:, tel:, etc.)
            if not full_url.startswith(('http://', 'https://')):
                continue
                
            # Skip URLs we've already visited or failed to fetch
            if full_url in self.visited_urls or full_url in self.failed_urls:
                continue
            
            # Check if we should follow external links
            link_domain = get_domain_from_url(full_url)
            if not self.configuration.follow_external_links and domain != link_domain:
                continue
            
            # Score the link using our intelligence engine
            if self.link_intelligence:
                score = self.link_intelligence.score_link(full_url, a_tag, base_url)
                scored_links.append((full_url, score))
            else:
                # Fallback if intelligence engine isn't initialized
                scored_links.append((full_url, 0.5))  # Default neutral score
        
        # Sort links by score (highest first) and extract just the URLs
        sorted_links = sorted(scored_links, key=lambda x: x[1], reverse=True)
        
        # Log high-value links for debugging
        high_value_links = [(url, score) for url, score in sorted_links if score > 0.7]
        if high_value_links:
            logger.debug(f"High-value links from {base_url}: {high_value_links[:5]}")
        
        return sorted_links

    def update_statistics(self, processed=0, successful=0, failed=0):
        """Direct update to job statistics with proper app context handling using SQL"""
        from app import app, db
        from sqlalchemy import text
        
        try:
            if processed > 0 or successful > 0 or failed > 0:
                with app.app_context():
                    # Use direct SQL update instead of ORM to avoid concurrency issues
                    # This guarantees atomic updates even with multiple threads
                    update_sql = text("""
                        UPDATE scraping_job SET 
                            urls_processed = urls_processed + :processed,
                            urls_successful = urls_successful + :successful,
                            urls_failed = urls_failed + :failed,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = :job_id
                    """)
                    
                    # Execute direct SQL update
                    result = db.session.execute(
                        update_sql, 
                        {
                            'processed': processed, 
                            'successful': successful, 
                            'failed': failed,
                            'job_id': self.job_id
                        }
                    )
                    
                    # Commit the transaction
                    db.session.commit()
                    logger.debug(f"Direct SQL update for job statistics - processed: +{processed}, successful: +{successful}, failed: +{failed}")
                    
                    # Reload job to get updated statistics for logging
                    self.load_job()
                    logger.debug(f"Job totals now - processed: {self.job.urls_processed}, successful: {self.job.urls_successful}, failed: {self.job.urls_failed}")
        except Exception as e:
            logger.error(f"Error updating job statistics: {str(e)}")
            # Log detailed error for debugging
            import traceback
            logger.error(traceback.format_exc())
    
    def process_domain_urls(self, domain_queue, results, lock):
        """Process all URLs for a specific domain"""
        from app import app, db
        
        # Create a new Flask application context for this thread
        with app.app_context():
            try:
                # Reload job data in this thread's context
                self.load_job()
                
                domain = domain_queue['domain']
                urls = domain_queue['urls']
                logger.info(f"Starting processing for domain: {domain} with {len(urls)} URLs")
                
                processed = 0
                successful = 0
                failed = 0
                new_urls = []
                
                # Process URLs in smaller batches to update stats more frequently
                batch_size = 5  # Update stats every 5 URLs
                current_batch = {'processed': 0, 'successful': 0, 'failed': 0}
                    
                for url, depth in urls:
                    # Check if job has been flagged to stop
                    if self.__class__.check_if_should_stop(self.job_id):
                        logger.info(f"Job {self.job_id} stop requested during domain processing. Stopping gracefully.")
                        break
                        
                    if url in self.visited_urls or url in self.failed_urls:
                        continue
                        
                    try:
                        success, links = self.fetch_url(url, depth)
                        processed += 1
                        current_batch['processed'] += 1
                        
                        if success:
                            successful += 1
                            current_batch['successful'] += 1
                            # If we haven't reached max depth, collect new links
                            if depth < self.configuration.max_depth:
                                for link in links:
                                    if link not in self.visited_urls and link not in self.failed_urls:
                                        # Only add URLs we haven't processed yet
                                        new_urls.append((link, depth + 1))
                        else:
                            failed += 1
                            current_batch['failed'] += 1
                            
                        # Update statistics in smaller batches
                        if sum(current_batch.values()) >= batch_size:
                            self.update_statistics(
                                processed=current_batch['processed'],
                                successful=current_batch['successful'], 
                                failed=current_batch['failed']
                            )
                            current_batch = {'processed': 0, 'successful': 0, 'failed': 0}
                            
                    except Exception as e:
                        logger.exception(f"Error processing URL {url}: {str(e)}")
                        failed += 1
                        current_batch['failed'] += 1
                
                # Update any remaining stats in the current batch
                if sum(current_batch.values()) > 0:
                    self.update_statistics(
                        processed=current_batch['processed'],
                        successful=current_batch['successful'], 
                        failed=current_batch['failed']
                    )
                
                # Use lock to safely update the shared results
                with lock:
                    results['processed'] += processed
                    results['successful'] += successful
                    results['failed'] += failed
                    results['new_urls'].extend(new_urls)
                    
                logger.info(f"Finished processing domain: {domain}. Processed: {processed}, Successful: {successful}, Failed: {failed}, New URLs: {len(new_urls)}")
            except Exception as e:
                logger.exception(f"Error in domain processing thread for {domain_queue['domain']}: {str(e)}")
        
    def crawl(self):
        """Start the crawling process with parallel domain processing"""
        from app import app, db
        
        try:
            # Ensure we have the job loaded in this context
            with app.app_context():
                self.load_job()
                self.update_job_status(ScrapingStatus.RUNNING, log="Starting parallel crawl job")
                
                # Register job as active
                with self.__class__._lock:
                    self.__class__.active_jobs[self.job_id] = False
                
                # Parse base URLs from configuration
                base_urls = self.configuration.base_urls
                if not base_urls:
                    self.update_job_status(ScrapingStatus.FAILED, log="No base URLs provided")
                    # Clean up active jobs
                    with self.__class__._lock:
                        if self.job_id in self.__class__.active_jobs:
                            del self.__class__.active_jobs[self.job_id]
                    return
                    
                # Group URLs by domain for parallel processing
                domain_groups = {}
                for url in base_urls:
                    domain = get_domain_from_url(url)
                    if domain not in domain_groups:
                        domain_groups[domain] = []
                    domain_groups[domain].append((url, 0))  # (url, depth)
                    
                self.update_job_status(self.job.status, log=f"Found {len(domain_groups)} domains to process in parallel")
                
                # Main crawling loop - process each level of depth before moving to the next
                max_depth = self.configuration.max_depth
                current_depth = 0
                
                while domain_groups and current_depth <= max_depth:
                    # Check if job has been flagged to stop
                    if self.__class__.check_if_should_stop(self.job_id):
                        logger.info(f"Job {self.job_id} has been flagged to stop. Finishing gracefully.")
                        self.update_job_status(ScrapingStatus.COMPLETED, 
                                          log=f"Crawl stopped early by user. Processed {self.job.urls_processed} URLs. "
                                          f"Success: {self.job.urls_successful}, Failed: {self.job.urls_failed}")
                        
                        # Clean up active jobs
                        with self.__class__._lock:
                            if self.job_id in self.__class__.active_jobs:
                                del self.__class__.active_jobs[self.job_id]
                        return
                    self.update_job_status(self.job.status, log=f"Processing depth {current_depth}, domains: {list(domain_groups.keys())}")
                    
                    # Set up shared results with thread synchronization
                    import threading
                    results = {
                        'processed': 0,
                        'successful': 0,
                        'failed': 0,
                        'new_urls': []
                    }
                    lock = threading.Lock()
                    
                    # Create worker threads for each domain
                    threads = []
                    for domain, urls in domain_groups.items():
                        # Only process URLs at current depth
                        urls_at_depth = [url_tuple for url_tuple in urls if url_tuple[1] == current_depth]
                        if not urls_at_depth:
                            continue
                            
                        domain_queue = {
                            'domain': domain,
                            'urls': urls_at_depth
                        }
                        
                        # Create and start a thread for this domain
                        thread = threading.Thread(
                            target=self.process_domain_urls,
                            args=(domain_queue, results, lock)
                        )
                        thread.start()
                        threads.append(thread)
                    
                    # Wait for all domain threads to complete
                    for thread in threads:
                        thread.join()
                    
                    # Update statistics in the database using our direct SQL approach
                    # This is more reliable for concurrent operations
                    self.update_statistics(
                        processed=results['processed'], 
                        successful=results['successful'],
                        failed=results['failed']
                    )
                    
                    # Prepare for next depth level - group new URLs by domain
                    with app.app_context():
                        domain_groups = {}
                        for url, depth in results['new_urls']:
                            domain = get_domain_from_url(url)
                            if domain not in domain_groups:
                                domain_groups[domain] = []
                            domain_groups[domain].append((url, depth))
                    
                    # Move to next depth level
                    current_depth += 1
                
                self.update_job_status(ScrapingStatus.COMPLETED, 
                                    log=f"Crawl completed. Processed {self.job.urls_processed} URLs. "
                                        f"Success: {self.job.urls_successful}, Failed: {self.job.urls_failed}")
                
                # Clean up active jobs
                with self.__class__._lock:
                    if self.job_id in self.__class__.active_jobs:
                        del self.__class__.active_jobs[self.job_id]
            
        except Exception as e:
            logger.exception(f"Error in crawl job: {str(e)}")
            self.update_job_status(ScrapingStatus.FAILED, log=f"Crawl job failed with error: {str(e)}")
            
            # Clean up active jobs even on failure
            with self.__class__._lock:
                if self.job_id in self.__class__.active_jobs:
                    del self.__class__.active_jobs[self.job_id]

    def start_async(self):
        """Start the crawling process in a separate thread without blocking"""
        from app import app
        import threading
        
        def run_in_app_context():
            with app.app_context():
                self.crawl()
        
        # Start in a daemon thread to avoid blocking web requests
        thread = threading.Thread(target=run_in_app_context)
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started background crawling job {self.job_id} in thread {thread.name}")
