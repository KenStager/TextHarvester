"""
Export utilities for converting scraped content to formats suitable for NER/SpanCat annotation.
"""

import json
import logging
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from app import db
from models import ScrapedContent, ScrapingJob, ContentMetadata

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Clean and normalize text for annotation.
    Handles problematic characters and encoding issues.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    try:
        # Remove null bytes which can cause database issues
        text = text.replace('\x00', '')
        
        # Handle other problematic control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        # Replace multiple newlines with a single one
        text = re.sub(r'\n+', '\n', text)
        
        # Replace multiple spaces with a single one
        text = re.sub(r'\s+', ' ', text)
        
        # Strip whitespace from start and end
        text = text.strip()
        
        # Additional UTF-8 sanitization - re-encode and decode to catch issues
        # This is a defensive measure to ensure UTF-8 compatibility
        text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        
        return text
    except Exception as e:
        logger.error(f"Error in clean_text: {str(e)}")
        # Return a cautious fallback
        return "" if text is None else str(text).replace('\x00', '')

def split_into_chunks(text: str, max_chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into manageable chunks for annotation.
    Tries to split at sentence boundaries when possible.
    
    Args:
        text (str): Text to split
        max_chunk_size (int): Maximum size of each chunk in words
        overlap (int): Number of words to overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    # Clean the text first
    text = clean_text(text)
    
    # Split into sentences (simple heuristic)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        # Count words in sentence
        sentence_words = len(sentence.split())
        
        # If this sentence alone is too big, we need to split it
        if sentence_words > max_chunk_size:
            # If we have an existing chunk, add it first
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_word_count = 0
            
            # Split the long sentence into parts
            words = sentence.split()
            for i in range(0, len(words), max_chunk_size - overlap):
                if i > 0:
                    start_idx = max(0, i - overlap)
                else:
                    start_idx = 0
                
                chunk = ' '.join(words[start_idx:i + max_chunk_size])
                chunks.append(chunk)
        
        # If adding this sentence exceeds our limit
        elif current_word_count + sentence_words > max_chunk_size:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap from the end of the previous one
            if overlap > 0 and current_chunk:
                # Get the last few words for overlap
                overlap_words = []
                word_count = 0
                
                for i in range(len(current_chunk) - 1, -1, -1):
                    sentence_words = len(current_chunk[i].split())
                    if word_count + sentence_words <= overlap:
                        overlap_words.insert(0, current_chunk[i])
                        word_count += sentence_words
                    else:
                        break
                
                current_chunk = overlap_words
                current_word_count = word_count
            else:
                current_chunk = []
                current_word_count = 0
            
            # Add the new sentence
            current_chunk.append(sentence)
            current_word_count += sentence_words
        else:
            # Add to current chunk
            current_chunk.append(sentence)
            current_word_count += sentence_words
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def create_jsonl_record(text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a JSONL record suitable for Prodigy annotation.
    
    Args:
        text (str): Text content
        metadata (dict): Additional metadata
        
    Returns:
        dict: JSONL record
    """
    # Create record with text
    record: Dict[str, Any] = {"text": text}
    
    # Add metadata if provided
    if metadata:
        record["meta"] = {k: str(v) if isinstance(v, (int, float, bool)) else v for k, v in metadata.items()}
    
    return record

def generate_jsonl_records(job_id: int, max_chunk_size: int = 500, overlap: int = 50):
    """
    Generator function to yield JSONL records one by one.
    This avoids loading all content into memory at once.
    
    Args:
        job_id (int): ID of the scraping job
        max_chunk_size (int): Maximum size of each chunk in words
        overlap (int): Number of words to overlap between chunks
        
    Yields:
        dict: JSONL record ready to be written to file
    """
    # Get job details for metadata
    job = ScrapingJob.query.get(job_id)
    if not job or not job.configuration:
        logger.warning(f"Could not find job or configuration for job {job_id}")
        return
    
    # Get count of content for this job
    total_content = ScrapedContent.query.filter_by(job_id=job_id).count()
    if total_content == 0:
        logger.warning(f"No content found for job {job_id}")
        return
    
    logger.info(f"Generating JSONL records for job {job_id}: {total_content} content items")
    
    # Process in batches to avoid loading everything at once
    batch_size = 50  # Reduced batch size for better memory management
    offset = 0
    processed_count = 0
    error_count = 0
    
    while offset < total_content:
        try:
            # Get batch of content with eager loading of metadata to avoid lazy loading issues
            from sqlalchemy.orm import joinedload
            content_batch = (ScrapedContent.query
                            .filter_by(job_id=job_id)
                            .options(joinedload(ScrapedContent.content_metadata))
                            .order_by(ScrapedContent.id)
                            .limit(batch_size)
                            .offset(offset)
                            .all())
            
            if not content_batch:
                break
            
            logger.debug(f"Processing batch at offset {offset} with {len(content_batch)} items")
                
            for item in content_batch:
                try:
                    # Clean the text
                    text = clean_text(item.extracted_text)
                    
                    if not text:
                        logger.debug(f"Skipping content {item.id} - empty text after cleaning")
                        continue
                    
                    # Split into chunks
                    chunks = split_into_chunks(text, max_chunk_size, overlap)
                    
                    # Build metadata
                    metadata = {
                        "source": "web_scrape",
                        "url": item.url,
                        "title": item.title if item.title else "Untitled",
                        "date": item.created_at.isoformat(),
                        "job_id": job_id,
                        "content_id": item.id,
                        "config_name": job.configuration.name,
                        "crawl_depth": item.crawl_depth
                    }
                    
                    # Add metadata from ContentMetadata if available
                    if hasattr(item, 'content_metadata') and item.content_metadata:
                        meta = item.content_metadata
                        if meta.language:
                            metadata["language"] = meta.language
                        if meta.content_type:
                            metadata["content_type"] = meta.content_type
                        if meta.word_count:
                            metadata["word_count"] = int(meta.word_count)
                    
                    # Yield each chunk as a separate JSONL record
                    for i, chunk in enumerate(chunks):
                        try:
                            # Add chunk index to metadata
                            chunk_metadata = metadata.copy()
                            chunk_metadata["chunk_index"] = i
                            chunk_metadata["chunk_total"] = len(chunks)
                            
                            # Clean any problematic characters
                            clean_chunk = chunk.replace('\x00', '')
                            
                            # Create record
                            record = create_jsonl_record(clean_chunk, chunk_metadata)
                            
                            # Yield the record
                            yield record
                            processed_count += 1
                        except Exception as e:
                            logger.warning(f"Error processing chunk {i} for content {item.id}: {str(e)}")
                            error_count += 1
                            continue
                            
                except Exception as e:
                    logger.warning(f"Error processing content {item.id}: {str(e)}")
                    error_count += 1
                    continue
            
            # Update offset for next batch
            offset += batch_size
            
        except Exception as e:
            logger.warning(f"Error fetching batch at offset {offset}: {str(e)}")
            offset += batch_size  # Still increment to avoid infinite loop
            error_count += 1
    
    logger.info(f"JSONL generation complete: {processed_count} records processed, {error_count} errors")

def export_job_to_jsonl(job_id: int, output_path: str, 
                        max_chunk_size: int = 500, 
                        overlap: int = 50) -> Tuple[int, int]:
    """
    Export a scraping job's content to JSONL format for annotation.
    Uses the generator approach to handle large datasets efficiently.
    
    Args:
        job_id (int): ID of the scraping job
        output_path (str): Path to save the JSONL file
        max_chunk_size (int): Maximum size of each chunk in words
        overlap (int): Number of words to overlap between chunks
        
    Returns:
        Tuple[int, int]: (Number of documents processed, Number of chunks created)
    """
    # Get count for returns
    doc_count = ScrapedContent.query.filter_by(job_id=job_id).count()
    
    # Use a file buffer for better performance
    buffer_size = 50  # Number of records to buffer before writing
    buffer = []
    total_chunks = 0
    processed_docs = 0
    
    try:
        logger.info(f"Starting export of job {job_id} to {output_path}")
        
        # Process in batches directly to file to avoid memory bloat
        batch_size = 25  # Process fewer docs at a time to reduce memory usage
        offset = 0
        
        with open(output_path, 'w', encoding='utf-8') as f:
            while True:
                # Get a batch of content with eager loading
                from sqlalchemy.orm import joinedload
                try:
                    content_batch = (ScrapedContent.query
                                    .filter_by(job_id=job_id)
                                    .options(joinedload(ScrapedContent.content_metadata))
                                    .order_by(ScrapedContent.id)
                                    .limit(batch_size)
                                    .offset(offset)
                                    .all())
                except Exception as e:
                    logger.error(f"Database error during content fetch: {str(e)}")
                    # Skip this batch if there's a database error
                    offset += batch_size
                    continue
                
                if not content_batch:
                    break
                
                logger.info(f"Processing export batch at offset {offset} ({len(content_batch)} documents)")
                
                # Process this batch
                for item in content_batch:
                    try:
                        processed_docs += 1
                        
                        # Clean the text
                        text = clean_text(item.extracted_text)
                        if not text:
                            logger.debug(f"Skipping content {item.id} - empty text after cleaning")
                            continue
                        
                        # Build metadata once per document
                        job = ScrapingJob.query.get(job_id)
                        metadata = {
                            "source": "web_scrape",
                            "url": item.url,
                            "title": item.title if item.title else "Untitled",
                            "date": item.created_at.isoformat(),
                            "job_id": job_id,
                            "content_id": item.id,
                            "config_name": job.configuration.name if job and job.configuration else "Unknown",
                            "crawl_depth": item.crawl_depth
                        }
                        
                        # Add metadata from ContentMetadata if available
                        if hasattr(item, 'content_metadata') and item.content_metadata:
                            meta = item.content_metadata
                            if meta.language:
                                metadata["language"] = meta.language
                            if meta.content_type:
                                metadata["content_type"] = meta.content_type
                            if meta.word_count:
                                metadata["word_count"] = int(meta.word_count)
                                
                        # Split into chunks
                        chunks = split_into_chunks(text, max_chunk_size, overlap)
                        
                        # Process each chunk
                        for i, chunk in enumerate(chunks):
                            try:
                                # Add chunk index to metadata
                                chunk_metadata = metadata.copy()
                                chunk_metadata["chunk_index"] = i
                                chunk_metadata["chunk_total"] = len(chunks)
                                
                                # Clean any problematic characters
                                clean_chunk = chunk.replace('\x00', '')
                                
                                # Create record
                                record = create_jsonl_record(clean_chunk, chunk_metadata)
                                
                                # Sanitize content - handle null bytes and other problematic characters
                                sanitized_record = {
                                    k: v.replace('\x00', '') if isinstance(v, str) else v 
                                    for k, v in record.items()
                                }
                                
                                if isinstance(sanitized_record.get('meta'), dict):
                                    sanitized_record['meta'] = {
                                        k: v.replace('\x00', '') if isinstance(v, str) else v 
                                        for k, v in sanitized_record['meta'].items()
                                    }
                                
                                # Convert to JSON string
                                json_str = json.dumps(sanitized_record, ensure_ascii=False)
                                
                                # Add to buffer
                                buffer.append(json_str)
                                total_chunks += 1
                                
                                # Write buffer if it's full
                                if len(buffer) >= buffer_size:
                                    f.write('\n'.join(buffer) + '\n')
                                    buffer = []  # Clear buffer
                                    
                                    # Flush to disk occasionally
                                    if total_chunks % 500 == 0:
                                        f.flush()
                                        logger.info(f"Export progress: {processed_docs}/{doc_count} docs, {total_chunks} chunks")
                                
                            except Exception as e:
                                logger.warning(f"Error processing chunk {i} for content {item.id}: {str(e)}")
                                # Continue with next chunk instead of failing
                                continue
                    except Exception as e:
                        logger.warning(f"Error processing document {item.id}: {str(e)}")
                        # Continue with next document
                        continue
                
                # Update offset for next batch
                offset += batch_size
                
                # Brief pause to allow other operations
                time.sleep(0.1)
            
            # Write any remaining records in buffer
            if buffer:
                f.write('\n'.join(buffer) + '\n')
                    
        logger.info(f"Successfully exported {processed_docs}/{doc_count} documents with {total_chunks} chunks to {output_path}")
        return doc_count, total_chunks
    except Exception as e:
        logger.exception(f"Failed to export job {job_id}: {str(e)}")
        raise