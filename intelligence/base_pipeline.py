"""
Base Pipeline Framework
=====================

This module defines the base pipeline class for all intelligence components, 
providing common functionality for processing state management, error handling, 
performance instrumentation, and parallel processing capabilities.
"""

import os
import logging
import time
import traceback
import json
import threading
import multiprocessing
from queue import Queue, Empty
from typing import List, Dict, Tuple, Optional, Union, Any, Callable, Generator, Iterable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime
import uuid
from pathlib import Path
import signal
import atexit

# Import configuration
from intelligence.config import Config

# Set up logging
logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 300  # seconds
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_THREADS = min(32, (os.cpu_count() or 4) * 2)  # Default to 2x CPU cores, max 32


class PipelineState:
    """Container for pipeline processing state."""
    
    CREATED = "created"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"
    
    def __init__(self, pipeline_id: str):
        """
        Initialize a pipeline state.
        
        Args:
            pipeline_id (str): Unique identifier for the pipeline.
        """
        self.id = pipeline_id
        self.state = self.CREATED
        self.start_time = None
        self.end_time = None
        self.processed_items = 0
        self.failed_items = 0
        self.current_batch = 0
        self.total_batches = 0
        self.last_error = None
        self.last_activity = datetime.now()
        self.statistics = {}
        self.lock = threading.RLock()
    
    def update(self, state: str, **kwargs):
        """
        Update the pipeline state.
        
        Args:
            state (str): New state.
            **kwargs: Additional state attributes to update.
        """
        with self.lock:
            self.state = state
            self.last_activity = datetime.now()
            
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            # Update start/end times based on state
            if state == self.RUNNING and not self.start_time:
                self.start_time = datetime.now()
            elif state in (self.COMPLETED, self.FAILED) and not self.end_time:
                self.end_time = datetime.now()
    
    def increment_processed(self, count: int = 1):
        """Increment the processed items count."""
        with self.lock:
            self.processed_items += count
            self.last_activity = datetime.now()
    
    def increment_failed(self, count: int = 1, error: Optional[Exception] = None):
        """Increment the failed items count and optionally store error."""
        with self.lock:
            self.failed_items += count
            self.last_activity = datetime.now()
            
            if error:
                self.last_error = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc()
                }
    
    def update_statistics(self, stats: Dict[str, Any]):
        """Update the pipeline statistics."""
        with self.lock:
            self.statistics.update(stats)
            self.last_activity = datetime.now()
    
    def is_active(self) -> bool:
        """Check if the pipeline is in an active state."""
        return self.state in (self.INITIALIZING, self.RUNNING)
    
    def can_pause(self) -> bool:
        """Check if the pipeline can be paused."""
        return self.state == self.RUNNING
    
    def can_resume(self) -> bool:
        """Check if the pipeline can be resumed."""
        return self.state == self.PAUSED
    
    def can_stop(self) -> bool:
        """Check if the pipeline can be stopped."""
        return self.state in (self.RUNNING, self.PAUSED)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the state to a dictionary."""
        with self.lock:
            return {
                "id": self.id,
                "state": self.state,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "elapsed_seconds": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                "processed_items": self.processed_items,
                "failed_items": self.failed_items,
                "success_rate": (self.processed_items / max(1, self.processed_items + self.failed_items)) * 100,
                "current_batch": self.current_batch,
                "total_batches": self.total_batches,
                "progress_percent": (self.current_batch / max(1, self.total_batches)) * 100 if self.total_batches else 0,
                "last_error": self.last_error,
                "last_activity": self.last_activity.isoformat(),
                "statistics": self.statistics
            }


class WorkItem:
    """Container for a work item in the pipeline."""
    
    def __init__(self, data: Any, item_id: Optional[str] = None, metadata: Optional[Dict] = None):
        """
        Initialize a work item.
        
        Args:
            data (Any): The data to process.
            item_id (str, optional): Unique identifier for the item.
            metadata (Dict, optional): Additional metadata.
        """
        self.item_id = item_id or str(uuid.uuid4())
        self.data = data
        self.metadata = metadata or {}
        self.result = None
        self.error = None
        self.retries = 0
        self.start_time = None
        self.end_time = None
        self.processing_time = 0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the work item to a dictionary."""
        return {
            "item_id": self.item_id,
            "metadata": self.metadata,
            "result": self.result,
            "error": str(self.error) if self.error else None,
            "retries": self.retries,
            "processing_time": self.processing_time
        }


class PipelineResult:
    """Container for pipeline processing results."""
    
    def __init__(self, pipeline_id: str):
        """
        Initialize pipeline results.
        
        Args:
            pipeline_id (str): Unique identifier for the pipeline.
        """
        self.pipeline_id = pipeline_id
        self.successful_items = []
        self.failed_items = []
        self.state = None
        self.start_time = None
        self.end_time = None
        self.processing_time = 0
        self.statistics = {}
        
    def add_successful_item(self, item: WorkItem):
        """Add a successfully processed item."""
        self.successful_items.append(item)
        
    def add_failed_item(self, item: WorkItem):
        """Add a failed item."""
        self.failed_items.append(item)
        
    def set_state(self, state: PipelineState):
        """Set the final pipeline state."""
        self.state = state
        self.start_time = state.start_time
        self.end_time = state.end_time
        
        if self.start_time and self.end_time:
            self.processing_time = (self.end_time - self.start_time).total_seconds()
            
        self.statistics = state.statistics
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "pipeline_id": self.pipeline_id,
            "successful_count": len(self.successful_items),
            "failed_count": len(self.failed_items),
            "total_count": len(self.successful_items) + len(self.failed_items),
            "success_rate": (len(self.successful_items) / max(1, len(self.successful_items) + len(self.failed_items))) * 100,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "processing_time": self.processing_time,
            "items_per_second": len(self.successful_items) / max(1, self.processing_time),
            "statistics": self.statistics
        }
        
    def save_to_file(self, file_path: str):
        """Save the result to a JSON file."""
        result_dict = self.to_dict()
        
        # Add successful item summaries (but not full data to avoid huge files)
        result_dict["successful_items"] = [
            {
                "item_id": item.item_id,
                "processing_time": item.processing_time,
                "metadata": item.metadata
            } for item in self.successful_items[:100]  # Limit to first 100 items
        ]
        
        # Add failed item details
        result_dict["failed_items"] = [
            {
                "item_id": item.item_id,
                "error": str(item.error) if item.error else None,
                "retries": item.retries,
                "metadata": item.metadata
            } for item in self.failed_items
        ]
        
        # Save to file
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2)


class BasePipeline:
    """Base class for all intelligence pipeline components."""
    
    def __init__(self, pipeline_name: str, config: Optional[Dict] = None):
        """
        Initialize the pipeline.
        
        Args:
            pipeline_name (str): Name of the pipeline.
            config (Dict, optional): Pipeline configuration.
        """
        self.name = pipeline_name
        self.id = f"{pipeline_name}_{str(uuid.uuid4())[:8]}"
        self.config = config or {}
        self.state = PipelineState(self.id)
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.error_queue = Queue()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.workers = []
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        
        # Configure from global config
        self.batch_size = self.config.get('batch_size', DEFAULT_BATCH_SIZE)
        self.max_workers = self.config.get('max_workers', DEFAULT_MAX_THREADS)
        self.timeout = self.config.get('timeout', DEFAULT_TIMEOUT)
        self.max_retries = self.config.get('max_retries', MAX_RETRIES)
        self.use_processes = self.config.get('use_processes', False)
        
        # Register cleanup handlers
        atexit.register(self.cleanup)
        
    def initialize(self):
        """Initialize the pipeline (to be implemented by subclasses)."""
        self.state.update(PipelineState.INITIALIZING)
        # Subclasses should implement specific initialization logic
        pass
        
    def process_item(self, item: WorkItem) -> WorkItem:
        """
        Process a single work item (to be implemented by subclasses).
        
        Args:
            item (WorkItem): Item to process.
            
        Returns:
            WorkItem: Processed item.
        """
        # Subclasses must implement this method
        raise NotImplementedError("Subclasses must implement process_item method")
    
    def process_batch(self, batch: List[WorkItem]) -> List[WorkItem]:
        """
        Process a batch of work items.
        
        Args:
            batch (List[WorkItem]): Batch of items to process.
            
        Returns:
            List[WorkItem]: Processed items.
        """
        # Default implementation processes items individually
        # Subclasses may override for more efficient batch processing
        processed_batch = []
        
        for item in batch:
            try:
                item.start_time = datetime.now()
                processed_item = self.process_item(item)
                item.end_time = datetime.now()
                item.processing_time = (item.end_time - item.start_time).total_seconds()
                processed_batch.append(processed_item)
            except Exception as e:
                self.logger.error(f"Error processing item {item.item_id}: {str(e)}")
                item.error = e
                item.end_time = datetime.now()
                item.processing_time = (item.end_time - item.start_time).total_seconds()
                processed_batch.append(item)
                
        return processed_batch
    
    def preprocess(self, data: List[Any]) -> List[WorkItem]:
        """
        Preprocess data into work items.
        
        Args:
            data (List[Any]): Raw input data.
            
        Returns:
            List[WorkItem]: Work items ready for processing.
        """
        # Default implementation wraps each data item in a WorkItem
        # Subclasses may override for specific preprocessing
        return [WorkItem(item) for item in data]
    
    def postprocess(self, results: List[WorkItem]) -> Any:
        """
        Postprocess work items into final output.
        
        Args:
            results (List[WorkItem]): Processed work items.
            
        Returns:
            Any: Final output.
        """
        # Default implementation extracts result from each work item
        # Subclasses may override for specific postprocessing
        return [item.result for item in results if item.error is None]
    
    def handle_error(self, item: WorkItem) -> bool:
        """
        Handle a processing error.
        
        Args:
            item (WorkItem): Failed work item.
            
        Returns:
            bool: True if the item should be retried, False otherwise.
        """
        # Default implementation retries up to max_retries times
        # Subclasses may override for specific error handling
        if item.retries < self.max_retries:
            item.retries += 1
            self.logger.warning(f"Retrying item {item.item_id} (attempt {item.retries}/{self.max_retries})")
            return True
        else:
            self.logger.error(f"Failed to process item {item.item_id} after {self.max_retries} attempts")
            return False
    
    def worker_thread(self, worker_id: int):
        """
        Worker thread for processing items.
        
        Args:
            worker_id (int): Worker identifier.
        """
        self.logger.debug(f"Worker {worker_id} started")
        
        while not self.stop_event.is_set():
            # Check if paused
            if self.pause_event.is_set():
                time.sleep(1)
                continue
                
            try:
                # Get next item from queue
                item = self.input_queue.get(timeout=1)
                
                try:
                    # Process the item
                    item.start_time = datetime.now()
                    processed_item = self.process_item(item)
                    item.end_time = datetime.now()
                    item.processing_time = (item.end_time - item.start_time).total_seconds()
                    
                    # Put successful result in output queue
                    self.output_queue.put(processed_item)
                    self.state.increment_processed()
                    
                except Exception as e:
                    # Handle processing error
                    item.error = e
                    item.end_time = datetime.now()
                    item.processing_time = (item.end_time - item.start_time).total_seconds()
                    
                    if self.handle_error(item):
                        # Retry the item
                        self.input_queue.put(item)
                    else:
                        # Put failed item in error queue
                        self.error_queue.put(item)
                        self.state.increment_failed(1, e)
                
                finally:
                    # Mark task as done
                    self.input_queue.task_done()
                    
            except Empty:
                # No items in queue, wait a bit
                time.sleep(0.1)
                
            except Exception as e:
                # Unexpected error in worker thread
                self.logger.error(f"Unexpected error in worker {worker_id}: {str(e)}")
                
        self.logger.debug(f"Worker {worker_id} stopped")
    
    def process(self, data: List[Any]) -> Any:
        """
        Process data through the pipeline.
        
        Args:
            data (List[Any]): Input data.
            
        Returns:
            Any: Processed output.
        """
        # Initialize the pipeline
        self.initialize()
        
        # Preprocess data into work items
        work_items = self.preprocess(data)
        
        # Process in batches
        batch_count = (len(work_items) + self.batch_size - 1) // self.batch_size
        self.state.update(PipelineState.RUNNING, total_batches=batch_count)
        
        all_results = []
        
        for i in range(0, len(work_items), self.batch_size):
            # Check if stopped
            if self.stop_event.is_set():
                break
                
            # Get batch
            batch = work_items[i:i + self.batch_size]
            self.state.update(PipelineState.RUNNING, current_batch=i // self.batch_size + 1)
            
            # Process batch
            try:
                batch_start = datetime.now()
                processed_batch = self.process_batch(batch)
                batch_end = datetime.now()
                batch_time = (batch_end - batch_start).total_seconds()
                
                # Update state
                successful = sum(1 for item in processed_batch if item.error is None)
                failed = len(processed_batch) - successful
                self.state.increment_processed(successful)
                self.state.increment_failed(failed)
                
                # Update statistics
                self.state.update_statistics({
                    "last_batch_time": batch_time,
                    "last_batch_size": len(batch),
                    "last_batch_success_rate": (successful / max(1, len(batch))) * 100,
                    "avg_item_time": batch_time / max(1, len(batch))
                })
                
                # Add results
                all_results.extend(processed_batch)
                
            except Exception as e:
                # Batch processing failed
                self.logger.error(f"Error processing batch: {str(e)}")
                self.state.increment_failed(len(batch), e)
                
                # Mark all items in batch as failed
                for item in batch:
                    item.error = e
                    all_results.append(item)
        
        # Update state
        self.state.update(PipelineState.COMPLETED)
        
        # Postprocess results
        return self.postprocess(all_results)
    
    def process_async(self, data: List[Any]) -> None:
        """
        Process data asynchronously using worker threads/processes.
        
        Args:
            data (List[Any]): Input data.
        """
        # Initialize the pipeline
        self.initialize()
        
        # Preprocess data into work items
        work_items = self.preprocess(data)
        
        # Reset queues
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.error_queue = Queue()
        
        # Reset events
        self.stop_event.clear()
        self.pause_event.clear()
        
        # Start worker threads
        worker_count = min(self.max_workers, len(work_items))
        self.workers = []
        
        for i in range(worker_count):
            worker = threading.Thread(target=self.worker_thread, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            
        # Update state
        self.state.update(PipelineState.RUNNING, total_batches=len(work_items))
        
        # Add items to queue
        for item in work_items:
            self.input_queue.put(item)
    
    def process_stream(self, data_stream: Iterable[Any], batch_size: int = None) -> Generator[Any, None, None]:
        """
        Process a stream of data through the pipeline, yielding results as they become available.
        
        Args:
            data_stream (Iterable[Any]): Stream of input data.
            batch_size (int, optional): Size of batches to process.
            
        Yields:
            Any: Processed output items.
        """
        # Initialize the pipeline
        self.initialize()
        
        # Use specified batch size or default
        batch_size = batch_size or self.batch_size
        
        # Initialize batch
        current_batch = []
        
        # Process stream in batches
        for item in data_stream:
            # Add item to current batch
            current_batch.append(item)
            
            # Process batch if it reaches batch_size
            if len(current_batch) >= batch_size:
                # Preprocess batch into work items
                work_items = self.preprocess(current_batch)
                
                # Process batch
                processed_batch = self.process_batch(work_items)
                
                # Update state
                successful = sum(1 for item in processed_batch if item.error is None)
                failed = len(processed_batch) - successful
                self.state.increment_processed(successful)
                self.state.increment_failed(failed)
                
                # Postprocess and yield results
                for result in self.postprocess(processed_batch):
                    yield result
                    
                # Clear batch
                current_batch = []
        
        # Process remaining items in batch
        if current_batch:
            # Preprocess batch into work items
            work_items = self.preprocess(current_batch)
            
            # Process batch
            processed_batch = self.process_batch(work_items)
            
            # Update state
            successful = sum(1 for item in processed_batch if item.error is None)
            failed = len(processed_batch) - successful
            self.state.increment_processed(successful)
            self.state.increment_failed(failed)
            
            # Postprocess and yield results
            for result in self.postprocess(processed_batch):
                yield result
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for asynchronous processing to complete.
        
        Args:
            timeout (float, optional): Maximum time to wait in seconds.
            
        Returns:
            bool: True if processing completed, False if timed out.
        """
        if not self.workers:
            return True
            
        start_time = time.time()
        
        while True:
            # Check if all workers are done
            if all(not worker.is_alive() for worker in self.workers):
                return True
                
            # Check if timeout reached
            if timeout is not None and time.time() - start_time > timeout:
                return False
                
            # Wait a bit
            time.sleep(0.1)
    
    def get_results(self) -> PipelineResult:
        """
        Get the results of asynchronous processing.
        
        Returns:
            PipelineResult: Processing results.
        """
        # Create result container
        result = PipelineResult(self.id)
        
        # Add successful items
        while not self.output_queue.empty():
            try:
                item = self.output_queue.get_nowait()
                result.add_successful_item(item)
            except Empty:
                break
                
        # Add failed items
        while not self.error_queue.empty():
            try:
                item = self.error_queue.get_nowait()
                result.add_failed_item(item)
            except Empty:
                break
                
        # Set final state
        result.set_state(self.state)
        
        return result
    
    def pause(self) -> bool:
        """
        Pause the processing.
        
        Returns:
            bool: True if paused, False otherwise.
        """
        if not self.state.can_pause():
            return False
            
        self.pause_event.set()
        self.state.update(PipelineState.PAUSED)
        self.logger.info(f"Pipeline {self.id} paused")
        return True
    
    def resume(self) -> bool:
        """
        Resume the processing.
        
        Returns:
            bool: True if resumed, False otherwise.
        """
        if not self.state.can_resume():
            return False
            
        self.pause_event.clear()
        self.state.update(PipelineState.RUNNING)
        self.logger.info(f"Pipeline {self.id} resumed")
        return True
    
    def stop(self) -> bool:
        """
        Stop the processing.
        
        Returns:
            bool: True if stopped, False otherwise.
        """
        if not self.state.can_stop():
            return False
            
        self.stop_event.set()
        self.state.update(PipelineState.STOPPING)
        self.logger.info(f"Pipeline {self.id} stopping")
        return True
    
    def cleanup(self):
        """Clean up resources."""
        # Stop processing
        self.stop()
        
        # Wait for workers to finish
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=1)
                
        self.logger.debug(f"Pipeline {self.id} cleaned up")


class ParallelPipeline(BasePipeline):
    """Pipeline that processes items in parallel using multiple workers."""
    
    def __init__(self, pipeline_name: str, process_func: Callable, config: Optional[Dict] = None):
        """
        Initialize the parallel pipeline.
        
        Args:
            pipeline_name (str): Name of the pipeline.
            process_func (Callable): Function to process a single item.
            config (Dict, optional): Pipeline configuration.
        """
        super().__init__(pipeline_name, config)
        self.process_func = process_func
        
    def process_item(self, item: WorkItem) -> WorkItem:
        """
        Process a single work item using the provided function.
        
        Args:
            item (WorkItem): Item to process.
            
        Returns:
            WorkItem: Processed item.
        """
        # Call the provided process function
        try:
            item.result = self.process_func(item.data)
        except Exception as e:
            item.error = e
            self.logger.error(f"Error processing item {item.item_id}: {str(e)}")
            
        return item
    
    def process_parallel(self, data: List[Any], max_workers: Optional[int] = None) -> List[Any]:
        """
        Process data in parallel using thread or process pool.
        
        Args:
            data (List[Any]): Input data.
            max_workers (int, optional): Maximum number of workers.
            
        Returns:
            List[Any]: Processed output.
        """
        # Initialize the pipeline
        self.initialize()
        
        # Use specified max_workers or default
        max_workers = max_workers or self.max_workers
        
        # Preprocess data into work items
        work_items = self.preprocess(data)
        
        # Create executor
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        # Process items in parallel
        self.state.update(PipelineState.RUNNING, total_batches=1)
        
        results = []
        
        with executor_class(max_workers=max_workers) as executor:
            # Submit all items
            future_to_item = {
                executor.submit(self.process_item, item): item
                for item in work_items
            }
            
            # Process completed futures
            for future in as_completed(future_to_item):
                try:
                    item = future.result()
                    results.append(item)
                    
                    if item.error is None:
                        self.state.increment_processed()
                    else:
                        self.state.increment_failed(1, item.error)
                        
                except Exception as e:
                    # This should not happen as exceptions should be caught in process_item
                    self.logger.error(f"Unexpected error processing future: {str(e)}")
                    self.state.increment_failed(1, e)
        
        # Update state
        self.state.update(PipelineState.COMPLETED)
        
        # Postprocess results
        return self.postprocess(results)


class PipelineRegistry:
    """Registry for tracking active pipelines."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PipelineRegistry, cls).__new__(cls)
            cls._instance.pipelines = {}
            cls._instance.lock = threading.RLock()
            
        return cls._instance
    
    def register(self, pipeline: BasePipeline) -> None:
        """
        Register a pipeline.
        
        Args:
            pipeline (BasePipeline): Pipeline to register.
        """
        with self.lock:
            self.pipelines[pipeline.id] = pipeline
            
    def unregister(self, pipeline_id: str) -> None:
        """
        Unregister a pipeline.
        
        Args:
            pipeline_id (str): ID of pipeline to unregister.
        """
        with self.lock:
            if pipeline_id in self.pipelines:
                del self.pipelines[pipeline_id]
                
    def get(self, pipeline_id: str) -> Optional[BasePipeline]:
        """
        Get a pipeline by ID.
        
        Args:
            pipeline_id (str): Pipeline ID.
            
        Returns:
            BasePipeline: Pipeline instance or None if not found.
        """
        with self.lock:
            return self.pipelines.get(pipeline_id)
            
    def get_all(self) -> Dict[str, BasePipeline]:
        """
        Get all registered pipelines.
        
        Returns:
            Dict[str, BasePipeline]: Dictionary of pipeline ID to pipeline instance.
        """
        with self.lock:
            return dict(self.pipelines)
            
    def get_active(self) -> Dict[str, BasePipeline]:
        """
        Get all active pipelines.
        
        Returns:
            Dict[str, BasePipeline]: Dictionary of active pipeline ID to pipeline instance.
        """
        with self.lock:
            return {
                pipeline_id: pipeline
                for pipeline_id, pipeline in self.pipelines.items()
                if pipeline.state.is_active()
            }
            
    def stop_all(self) -> None:
        """Stop all registered pipelines."""
        with self.lock:
            for pipeline in self.pipelines.values():
                pipeline.stop()
                
    def cleanup_completed(self) -> None:
        """Remove completed pipelines from registry."""
        with self.lock:
            completed_ids = [
                pipeline_id
                for pipeline_id, pipeline in self.pipelines.items()
                if pipeline.state.state in (PipelineState.COMPLETED, PipelineState.FAILED)
            ]
            
            for pipeline_id in completed_ids:
                del self.pipelines[pipeline_id]


# Create global registry instance
registry = PipelineRegistry()

# Register signal handlers for graceful shutdown
def signal_handler(signum, frame):
    """Handle termination signals."""
    logger.info(f"Received signal {signum}, stopping all pipelines")
    registry.stop_all()
    
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
