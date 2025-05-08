"""
Rust-integrated export utilities for converting scraped content to formats suitable 
for NER/SpanCat annotation with high performance.
"""
import os
import json
import logging
import subprocess
import tempfile
import requests
from typing import Dict, Any, List, Optional, Generator, Tuple

# Setup logging
logger = logging.getLogger(__name__)

def _get_rust_extractor_url() -> str:
    """Get the URL of the Rust extractor service"""
    return os.environ.get("RUST_EXTRACTOR_URL", "http://127.0.0.1:8888")

def is_rust_export_available() -> bool:
    """Check if Rust export functionality is available"""
    try:
        url = f"{_get_rust_extractor_url()}/api/health"
        response = requests.get(url, timeout=2)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.warning(f"Rust export not available: {str(e)}")
        return False

def export_job_to_jsonl_with_rust(
    job_id: int, 
    output_path: str,
    max_chunk_size: int = 500,
    overlap: int = 50
) -> Tuple[int, int]:
    """
    Export a scraping job's content to JSONL format using the Rust-based exporter.
    
    Args:
        job_id (int): ID of the scraping job
        output_path (str): Path to save the JSONL file
        max_chunk_size (int): Maximum size of each chunk in words
        overlap (int): Number of words to overlap between chunks
        
    Returns:
        Tuple[int, int]: (Number of documents processed, Number of chunks created)
    """
    logger.info(f"Starting Rust-based export for job {job_id} to {output_path}")
    
    try:
        # First validate job and get content count
        url = f"{_get_rust_extractor_url()}/api/export/job"
        payload = {
            "job_id": job_id,
            "chunk_size": max_chunk_size,
            "overlap": overlap,
            "db_url": os.environ.get("DATABASE_URL")
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        if not result.get("success", False):
            raise ValueError(f"Failed to initialize export: {result.get('error', 'Unknown error')}")
        
        content_count = result.get("content_count", 0)
        
        logger.info(f"Validated job {job_id} with {content_count} documents")
        
        # Stream results to file
        stream_url = f"{_get_rust_extractor_url()}/api/export/job/{job_id}/stream"
        params = {
            "chunk_size": max_chunk_size,
            "overlap": overlap
        }
        
        # Since streaming isn't fully implemented, we'll use a simpler approach
        # In the future, we'd use streaming to process large datasets more efficiently
        with open(output_path, "w", encoding="utf-8") as output_file:
            stream_response = requests.get(stream_url, params=params, stream=True)
            stream_response.raise_for_status()
            
            chunk_count = 0
            for chunk in stream_response.iter_content(chunk_size=8192):
                if chunk:
                    output_file.write(chunk.decode("utf-8"))
                    chunk_count += 1
                    
            logger.info(f"Wrote {chunk_count} chunks to {output_path}")
            
        return content_count, chunk_count
        
    except Exception as e:
        logger.exception(f"Error in Rust-based export: {str(e)}")
        raise

def generate_rust_export_records(
    job_id: int,
    max_chunk_size: int = 500,
    overlap: int = 50
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator function to yield JSONL records one by one from the Rust exporter.
    This avoids loading all content into memory at once.
    
    Args:
        job_id (int): ID of the scraping job
        max_chunk_size (int): Maximum size of each chunk in words
        overlap (int): Number of words to overlap between chunks
        
    Yields:
        dict: JSONL record ready to be written to file
    """
    logger.info(f"Starting Rust-based export generator for job {job_id}")
    
    try:
        # Create a temporary file to store the export
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as temp_file:
            temp_path = temp_file.name
            
        # Export to the temporary file
        export_job_to_jsonl_with_rust(job_id, temp_path, max_chunk_size, overlap)
        
        # Read the file line by line to yield records
        with open(temp_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error decoding JSON line: {str(e)}")
                        
        # Clean up the temporary file
        try:
            os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Error removing temporary file: {str(e)}")
            
    except Exception as e:
        logger.exception(f"Error in Rust-based export generator: {str(e)}")
        raise