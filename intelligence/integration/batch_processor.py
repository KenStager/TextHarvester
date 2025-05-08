"""
Batch Processing System for Content Intelligence.

This module provides batch processing capabilities for the Content Intelligence
Platform, allowing efficient processing of large volumes of scraped content.
"""

import logging
import os
import sys
import threading
import time
import datetime
from typing import Dict, List, Optional, Union, Any, Set, Callable
from queue import Queue, Empty
from dataclasses import dataclass, field
from pathlib import Path

# Add the project root to path to ensure imports work properly
sys.path.append(str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


@dataclass
class BatchJob:
    """A batch processing job."""
    job_id: str
    content_ids: List[int]
    options: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    status: str = "pending"
    progress: int = 0
    total: int = 0
    results: Dict[str, Any] = field(default_factory=dict)
    

class BatchProcessor:
    """
    Batch processor for content intelligence processing.
    
    This class manages batch processing jobs, executing them in the background
    with configurable parallelism and priority.
    """
    
    def __init__(self, components=None, config=None):
        """
        Initialize the batch processor.
        
        Args:
            components: List of processing components to use
            config: Configuration dictionary
        """
        self.components = components or []
        self.config = config or {
            'max_workers': 2,
            'max_queue_size': 100,
            'job_timeout': 3600,  # 1 hour
            'retry_count': 3
        }
        
        self.job_queue = Queue(maxsize=self.config['max_queue_size'])
        self.active_jobs = {}  # job_id -> BatchJob
        self.lock = threading.RLock()
        self.running = False
        self.workers = []
    
    def start(self):
        """Start the batch processor."""
        if self.running:
            return
            
        self.running = True
        
        # Start worker threads
        for i in range(self.config['max_workers']):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"BatchWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            
        logger.info(f"Started batch processor with {len(self.workers)} workers")
    
    def stop(self):
        """Stop the batch processor."""
        if not self.running:
            return
            
        self.running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=5.0)
                
        self.workers = []
        logger.info("Stopped batch processor")
    
    def submit_job(self, content_ids: List[int], options: Dict[str, Any] = None) -> str:
        """
        Submit a new batch job.
        
        Args:
            content_ids: List of content IDs to process
            options: Optional processing options
            
        Returns:
            Job ID
        """
        # Generate a unique job ID
        job_id = f"batch_{int(time.time())}_{len(self.active_jobs)}"
        
        # Create job
        job = BatchJob(
            job_id=job_id,
            content_ids=content_ids,
            options=options or {},
            total=len(content_ids)
        )
        
        # Add to active jobs
        with self.lock:
            self.active_jobs[job_id] = job
        
        # Add to queue
        try:
            self.job_queue.put(job, block=False)
            logger.info(f"Submitted batch job {job_id} with {len(content_ids)} content items")
        except Exception as e:
            logger.error(f"Failed to queue job {job_id}: {str(e)}")
            # Update job status
            with self.lock:
                if job_id in self.active_jobs:
                    self.active_jobs[job_id].status = "failed"
                    self.active_jobs[job_id].results = {"error": str(e)}
        
        # Start processor if not running
        if not self.running:
            self.start()
            
        return job_id
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a batch job.
        
        Args:
            job_id: Job ID to check
            
        Returns:
            Dictionary with job status information
        """
        with self.lock:
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                return {
                    "job_id": job.job_id,
                    "status": job.status,
                    "progress": job.progress,
                    "total": job.total,
                    "percent_complete": (job.progress / job.total * 100) if job.total > 0 else 0,
                    "created_at": job.created_at.isoformat(),
                    "results": job.results
                }
            else:
                return {"error": f"Job {job_id} not found"}
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all active batch jobs.
        
        Returns:
            List of job status dictionaries
        """
        with self.lock:
            return [
                {
                    "job_id": job.job_id,
                    "status": job.status,
                    "progress": job.progress,
                    "total": job.total,
                    "percent_complete": (job.progress / job.total * 100) if job.total > 0 else 0,
                    "created_at": job.created_at.isoformat()
                }
                for job in self.active_jobs.values()
            ]
    
    def process_backlog(self, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process any unprocessed content in the scraper database.
        
        This method is used to process content that may have been scraped
        before the intelligence integration was set up.
        
        Args:
            options: Optional processing options
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Import required components
            from models import ScrapedContent, ContentMetadata
            from app import db
            
            # Query for unprocessed content
            # In a real implementation, we'd have a way to track which content
            # has been processed. For now, we'll process all content.
            query = ScrapedContent.query
            
            # Apply filters if provided
            if options:
                if 'min_word_count' in options:
                    query = query.join(ContentMetadata).filter(
                        ContentMetadata.word_count >= options['min_word_count']
                    )
                    
                if 'job_id' in options:
                    query = query.filter(ScrapedContent.job_id == options['job_id'])
                    
                if 'limit' in options:
                    query = query.limit(options['limit'])
            
            # Execute query and get content IDs
            content_ids = [content.id for content in query.all()]
            
            if not content_ids:
                return {"message": "No unprocessed content found", "count": 0}
            
            # Submit as a batch job
            job_id = self.submit_job(content_ids, options)
            
            return {
                "message": f"Submitted batch job for {len(content_ids)} content items",
                "job_id": job_id,
                "count": len(content_ids)
            }
            
        except Exception as e:
            logger.error(f"Error processing backlog: {str(e)}")
            return {"error": str(e)}
    
    def _worker_loop(self):
        """Background worker thread for processing batch jobs."""
        logger.info(f"Batch worker starting: {threading.current_thread().name}")
        
        while self.running:
            try:
                # Get the next job from the queue
                try:
                    job = self.job_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Update job status
                with self.lock:
                    if job.job_id in self.active_jobs:
                        self.active_jobs[job.job_id].status = "running"
                
                # Process the job
                try:
                    self._process_job(job)
                    
                    # Update job status
                    with self.lock:
                        if job.job_id in self.active_jobs:
                            self.active_jobs[job.job_id].status = "completed"
                            
                except Exception as e:
                    logger.error(f"Error processing job {job.job_id}: {str(e)}")
                    
                    # Update job status
                    with self.lock:
                        if job.job_id in self.active_jobs:
                            self.active_jobs[job.job_id].status = "failed"
                            self.active_jobs[job.job_id].results = {
                                "error": str(e)
                            }
                
                # Mark task as done
                self.job_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in batch worker: {str(e)}")
    
    def _process_job(self, job: BatchJob):
        """
        Process a batch job.
        
        Args:
            job: The job to process
        """
        try:
            # Import the integration manager for processing
            from intelligence.integration.scraper_hooks import process_content_batch
            
            # Process the content in smaller batches to show progress
            batch_size = 10
            for i in range(0, len(job.content_ids), batch_size):
                if not self.running:
                    break
                    
                batch = job.content_ids[i:i+batch_size]
                
                # Process this batch
                batch_results = process_content_batch(batch, job.options)
                
                # Update progress
                with self.lock:
                    if job.job_id in self.active_jobs:
                        self.active_jobs[job.job_id].progress += len(batch)
                        
                        # Merge results
                        if 'successful' in batch_results:
                            self.active_jobs[job.job_id].results['successful'] = \
                                self.active_jobs[job.job_id].results.get('successful', 0) + batch_results['successful']
                                
                        if 'failed' in batch_results:
                            self.active_jobs[job.job_id].results['failed'] = \
                                self.active_jobs[job.job_id].results.get('failed', 0) + batch_results['failed']
                
                # Brief pause to avoid overwhelming the system
                time.sleep(0.1)
            
            logger.info(f"Completed processing job {job.job_id}")
            
        except Exception as e:
            logger.error(f"Error processing job {job.job_id}: {str(e)}")
            raise
