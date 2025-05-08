import json
import logging
import os
import time
from datetime import datetime
from flask import Blueprint, request, jsonify, render_template, redirect, url_for, flash

from app import db
from models import ScrapingConfiguration, ScrapingJob, ScrapedContent, ScrapingStatus, ContentMetadata
from scraper.crawler import WebCrawler
from scraper.utils import parse_urls_list

api_bp = Blueprint('api', __name__)
logger = logging.getLogger(__name__)

# Web interface routes
@api_bp.route('/', methods=['GET'])
def index():
    """Render the main dashboard page"""
    recent_jobs = ScrapingJob.query.order_by(ScrapingJob.created_at.desc()).limit(10).all()
    configurations = ScrapingConfiguration.query.order_by(ScrapingConfiguration.name).all()
    
    # Get statistics
    total_jobs = ScrapingJob.query.count()
    total_scraped = ScrapedContent.query.count()
    active_jobs = ScrapingJob.query.filter(ScrapingJob.status == ScrapingStatus.RUNNING).count()
    
    return render_template('index.html', 
                          recent_jobs=recent_jobs,
                          configurations=configurations,
                          stats={
                              'total_jobs': total_jobs,
                              'total_scraped': total_scraped,
                              'active_jobs': active_jobs
                          })

@api_bp.route('/config', methods=['GET', 'POST'])
def configuration():
    """Handle scraping configuration creation and editing"""
    from scraper.source_lists import get_all_source_lists, get_source_list

    if request.method == 'POST':
        try:
            name = request.form.get('name')
            description = request.form.get('description', '')
            
            # Check if using a predefined source list
            predefined_list = request.form.get('predefined_source')
            
            if predefined_list and predefined_list != 'custom':
                # Use a predefined source list
                source_list = get_source_list(predefined_list)
                if source_list:
                    base_urls = source_list['sources']
                    logger.info(f"Using predefined source list '{predefined_list}' with {len(base_urls)} URLs")
                else:
                    base_urls = []
                    logger.warning(f"Predefined source list '{predefined_list}' not found")
            else:
                # Parse custom URLs
                base_urls_text = request.form.get('base_urls', '')
                base_urls = parse_urls_list(base_urls_text)
                logger.info(f"Using custom URLs, parsed {len(base_urls)} valid URLs")
            
            # Only validate URLs if using custom source and not predefined list
            if not base_urls and (predefined_list == 'custom' or not predefined_list):
                flash('No valid URLs provided', 'danger')
                return redirect(url_for('api.configuration'))
            
            # Create or update configuration
            config_id = request.form.get('config_id')
            
            if config_id:
                # Update existing configuration
                config = ScrapingConfiguration.query.get(config_id)
                if not config:
                    flash('Configuration not found', 'danger')
                    return redirect(url_for('api.configuration'))
                
                config.name = name
                config.description = description
                config.base_urls = base_urls
                config.max_depth = int(request.form.get('max_depth', 1))
                config.follow_external_links = 'follow_external' in request.form
                config.respect_robots_txt = 'respect_robots' in request.form
                config.user_agent_rotation = 'user_agent_rotation' in request.form
                config.rate_limit_seconds = int(request.form.get('rate_limit', 5))
                config.max_retries = int(request.form.get('max_retries', 3))
                
                db.session.commit()
                flash('Configuration updated successfully', 'success')
            else:
                # Create new configuration
                new_config = ScrapingConfiguration(
                    name=name,
                    description=description,
                    base_urls=base_urls,
                    max_depth=int(request.form.get('max_depth', 1)),
                    follow_external_links='follow_external' in request.form,
                    respect_robots_txt='respect_robots' in request.form,
                    user_agent_rotation='user_agent_rotation' in request.form,
                    rate_limit_seconds=int(request.form.get('rate_limit', 5)),
                    max_retries=int(request.form.get('max_retries', 3))
                )
                
                db.session.add(new_config)
                db.session.commit()
                flash('Configuration created successfully', 'success')
            
            return redirect(url_for('api.index'))
            
        except Exception as e:
            logger.exception(f"Error creating/updating configuration: {str(e)}")
            flash(f'Error: {str(e)}', 'danger')
            return redirect(url_for('api.configuration'))
    
    # GET request - show configuration form
    config_id = request.args.get('id')
    config = None
    
    if config_id:
        config = ScrapingConfiguration.query.get(config_id)
    
    # Get all predefined source lists
    predefined_sources = get_all_source_lists()
    
    return render_template('configuration.html', config=config, predefined_sources=predefined_sources)

@api_bp.route('/status/<int:job_id>', methods=['GET'])
def job_status(job_id):
    """View status of a specific job with enhanced analytics"""
    from urllib.parse import urlparse
    from collections import Counter
    from sqlalchemy import func, desc
    
    job = ScrapingJob.query.get_or_404(job_id)
    
    # Basic statistics for this job
    content_count = ScrapedContent.query.filter_by(job_id=job_id).count()
    
    # Enhanced analytics data
    analytics = {}
    
    # Only gather detailed analytics if content exists
    if content_count > 0:
        # Domain distribution - Which domains have been crawled the most
        # Get a sample of URLs and extract domains in Python for compatibility
        sample_urls = db.session.query(ScrapedContent.url).filter(
            ScrapedContent.job_id == job_id
        ).limit(200).all()
        
        # Extract domains from URLs
        domains = []
        for (url,) in sample_urls:
            try:
                domain = urlparse(url).netloc
                domains.append(domain)
            except:
                continue
        
        # Count occurrences
        domain_counter = Counter(domains)
        analytics['domains'] = dict(domain_counter.most_common(10))
        
        # Get sample of content with metadata for further analysis
        content_with_metadata = db.session.query(
            ScrapedContent, ContentMetadata
        ).join(
            ContentMetadata, ContentMetadata.content_id == ScrapedContent.id
        ).filter(
            ScrapedContent.job_id == job_id
        ).limit(100).all()
        
        # Word count distribution
        word_counts = [m.word_count for _, m in content_with_metadata if m.word_count is not None]
        if word_counts:
            analytics['word_count'] = {
                'min': min(word_counts),
                'max': max(word_counts),
                'avg': sum(word_counts) / len(word_counts),
                'distribution': Counter([
                    '0-100' if wc < 100 else
                    '100-500' if wc < 500 else
                    '500-1000' if wc < 1000 else
                    '1000-5000' if wc < 5000 else
                    '5000+' for wc in word_counts
                ])
            }
        
        # Content type distribution
        content_types = [m.content_type for _, m in content_with_metadata if m.content_type is not None]
        if content_types:
            analytics['content_types'] = Counter(content_types)
        
        # Language distribution
        languages = [m.language for _, m in content_with_metadata if m.language is not None]
        if languages:
            analytics['languages'] = Counter(languages)
        
        # Processing time analysis
        processing_times = [c.processing_time for c, _ in content_with_metadata if c.processing_time is not None]
        if processing_times:
            analytics['processing_time'] = {
                'min': min(processing_times),
                'max': max(processing_times),
                'avg': sum(processing_times) / len(processing_times)
            }
        
        # Crawl depth distribution
        depths = [c.crawl_depth for c, _ in content_with_metadata]
        if depths:
            analytics['crawl_depth'] = Counter(depths)
    
    return render_template('status.html', job=job, content_count=content_count, analytics=analytics)

@api_bp.route('/content/<int:job_id>', methods=['GET'])
def view_job_content(job_id):
    """View content scraped by a specific job with pagination"""
    job = ScrapingJob.query.get_or_404(job_id)
    
    # Get pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 25, type=int)
    
    # Limit per_page to sane values
    if per_page not in [10, 25, 50, 100]:
        per_page = 25
    
    # Query content with pagination
    pagination = ScrapedContent.query.filter_by(job_id=job_id).order_by(
        ScrapedContent.id.desc()
    ).paginate(page=page, per_page=per_page)
    
    return render_template('content.html', job=job, pagination=pagination, per_page=per_page)

@api_bp.route('/jobs/delete/<int:job_id>', methods=['POST'])
def delete_job(job_id):
    """Delete a specific job and its associated content using batched operations with enhanced resilience"""
    job = ScrapingJob.query.get_or_404(job_id)
    
    try:
        logger.info(f"Starting deletion of job {job_id}")
        
        # Get total count of content for progress tracking
        total_content = ScrapedContent.query.filter_by(job_id=job_id).count()
        logger.info(f"Total content to delete: {total_content}")
        
        # Use smaller batch sizes for extremely large jobs
        BATCH_SIZE = 250 if total_content > 10000 else 500
        processed = 0
        error_count = 0
        max_retries = 3
        
        # Start batch deletion process
        while True:
            try:
                # Get a batch of content IDs
                content_ids = db.session.query(ScrapedContent.id) \
                    .filter(ScrapedContent.job_id == job_id) \
                    .limit(BATCH_SIZE) \
                    .all()
                
                if not content_ids:
                    break
                    
                # Convert to flat list
                content_ids = [id[0] for id in content_ids]
                batch_size = len(content_ids)
                
                # Split into smaller chunks for processing if needed
                chunk_size = 100  # Process in chunks of 100 IDs at a time
                for i in range(0, len(content_ids), chunk_size):
                    chunk = content_ids[i:i + chunk_size]
                    
                    # Delete metadata for this chunk in a separate transaction
                    try:
                        result = db.session.execute(
                            db.delete(ContentMetadata)
                            .where(ContentMetadata.content_id.in_(chunk))
                        )
                        metadata_deleted = result.rowcount
                        db.session.commit()
                    except Exception as e:
                        db.session.rollback()
                        logger.warning(f"Error deleting metadata chunk: {str(e)}")
                        # Continue with next operations even if this fails
                    
                    # Brief pause between operations
                    time.sleep(0.05)
                    
                    # Delete content items for this chunk in a separate transaction
                    try:
                        result = db.session.execute(
                            db.delete(ScrapedContent)
                            .where(ScrapedContent.id.in_(chunk))
                        )
                        content_deleted = result.rowcount
                        db.session.commit()
                    except Exception as e:
                        db.session.rollback()
                        logger.warning(f"Error deleting content chunk: {str(e)}")
                        error_count += 1
                        # Continue with next chunk
                
                processed += batch_size
                logger.info(f"Deleted batch: {batch_size} content items, progress: {processed}/{total_content}")
                
                # More significant pause between batches
                time.sleep(0.2)
                
                # Release memory and allow GC
                db.session.expunge_all()
                
            except Exception as e:
                error_count += 1
                logger.warning(f"Error processing batch: {str(e)}")
                if error_count >= max_retries:
                    logger.error("Too many errors, stopping batch deletion")
                    break
                time.sleep(0.5)  # Wait longer after errors
        
        # Update the job status to indicate deletion even if some content remains
        if error_count > 0:
            logger.warning(f"Completed deletion with {error_count} errors. Some content may remain.")
            flash('Job partially deleted. Some content items could not be removed.', 'warning')
        
        try:
            # Finally delete the job itself
            logger.info(f"Deleting job {job_id}")
            db.session.delete(job)
            db.session.commit()
            
            flash('Job deletion completed', 'success')
            logger.info(f"Job {job_id} successfully deleted with {processed} content items")
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error deleting job record: {str(e)}")
            flash(f'Content deleted but job record could not be removed: {str(e)}', 'warning')
            
    except Exception as e:
        db.session.rollback()
        logger.exception(f"Error in deletion process: {str(e)}")
        flash(f'Error deleting job: {str(e)}', 'danger')
    
    return redirect(url_for('api.index'))

@api_bp.route('/config/delete/<int:config_id>', methods=['POST'])
def delete_configuration(config_id):
    """Delete a scraping configuration"""
    config = ScrapingConfiguration.query.get_or_404(config_id)
    
    # Check if the configuration has associated jobs
    job_count = ScrapingJob.query.filter_by(configuration_id=config_id).count()
    
    if job_count > 0:
        flash(f'Cannot delete configuration: {job_count} jobs are using this configuration. Delete the jobs first.', 'danger')
        return redirect(url_for('api.index'))
    
    try:
        # Delete the configuration
        db.session.delete(config)
        db.session.commit()
        
        flash('Configuration deleted successfully', 'success')
    except Exception as e:
        db.session.rollback()
        logger.exception(f"Error deleting configuration: {str(e)}")
        flash(f'Error deleting configuration: {str(e)}', 'danger')
    
    return redirect(url_for('api.index'))

# API endpoints
@api_bp.route('/api/jobs', methods=['POST'])
def create_job():
    """Create a new scraping job"""
    try:
        data = request.get_json()
        config_id = data.get('configuration_id')
        
        if not config_id:
            return jsonify({'error': 'Missing configuration_id'}), 400
        
        # Check if configuration exists
        config = ScrapingConfiguration.query.get(config_id)
        if not config:
            return jsonify({'error': f'Configuration with ID {config_id} not found'}), 404
        
        # Create job
        job = ScrapingJob(configuration_id=config_id)
        db.session.add(job)
        db.session.commit()
        
        # Start crawler in background
        crawler = WebCrawler(job.id)
        crawler.start_async()
        
        return jsonify({
            'status': 'success',
            'job_id': job.id,
            'message': 'Job created and started'
        })
        
    except Exception as e:
        logger.exception(f"Error creating job: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/api/jobs/<int:job_id>', methods=['GET'])
@api_bp.route('/jobs/<int:job_id>', methods=['GET'])  # Add an alternative route to match the JS
def get_job(job_id):
    """Get job details"""
    # Add debug logging
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"API endpoint called: get_job({job_id})")
    
    job = ScrapingJob.query.get(job_id)
    if not job:
        logger.warning(f"Job not found: {job_id}")
        return jsonify({'error': f'Job with ID {job_id} not found'}), 404
    
    # Log job statistics
    logger.debug(f"Job stats: processed={job.urls_processed}, successful={job.urls_successful}, failed={job.urls_failed}")
    
    return jsonify({
        'id': job.id,
        'status': job.status.value,
        'start_time': job.start_time.isoformat() if job.start_time else None,
        'end_time': job.end_time.isoformat() if job.end_time else None,
        'urls_processed': job.urls_processed,
        'urls_successful': job.urls_successful,
        'urls_failed': job.urls_failed,
        'configuration_id': job.configuration_id,
        'created_at': job.created_at.isoformat()
    })

@api_bp.route('/api/jobs', methods=['GET'])
def list_jobs():
    """List all jobs"""
    jobs = ScrapingJob.query.order_by(ScrapingJob.created_at.desc()).all()
    
    return jsonify({
        'jobs': [{
            'id': job.id,
            'status': job.status.value,
            'start_time': job.start_time.isoformat() if job.start_time else None,
            'end_time': job.end_time.isoformat() if job.end_time else None,
            'urls_processed': job.urls_processed,
            'urls_successful': job.urls_successful,
            'urls_failed': job.urls_failed,
            'configuration_id': job.configuration_id,
            'created_at': job.created_at.isoformat()
        } for job in jobs]
    })

@api_bp.route('/api/configurations', methods=['GET'])
def list_configurations():
    """List all configurations"""
    configs = ScrapingConfiguration.query.all()
    
    return jsonify({
        'configurations': [{
            'id': config.id,
            'name': config.name,
            'description': config.description,
            'base_urls': config.base_urls,
            'max_depth': config.max_depth,
            'follow_external_links': config.follow_external_links,
            'respect_robots_txt': config.respect_robots_txt,
            'user_agent_rotation': config.user_agent_rotation,
            'rate_limit_seconds': config.rate_limit_seconds,
            'max_retries': config.max_retries,
            'created_at': config.created_at.isoformat()
        } for config in configs]
    })

@api_bp.route('/api/content/<int:job_id>', methods=['GET'])
def get_job_content(job_id):
    """Get content scraped by a specific job"""
    # Check if job exists
    job = ScrapingJob.query.get(job_id)
    if not job:
        return jsonify({'error': f'Job with ID {job_id} not found'}), 404
    
    # Apply pagination
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    # Query content with pagination
    content_query = ScrapedContent.query.filter_by(job_id=job_id).paginate(page=page, per_page=per_page)
    
    return jsonify({
        'job_id': job_id,
        'total_items': content_query.total,
        'page': content_query.page,
        'per_page': content_query.per_page,
        'total_pages': content_query.pages,
        'has_next': content_query.has_next,
        'has_prev': content_query.has_prev,
        'items': [{
            'id': item.id,
            'url': item.url,
            'title': item.title,
            'extracted_text': item.extracted_text,
            'crawl_depth': item.crawl_depth,
            'processing_time': item.processing_time,
            'created_at': item.created_at.isoformat()
        } for item in content_query.items]
    })

# Start job from web interface
@api_bp.route('/jobs/start', methods=['POST'])
def start_job():
    """Start a new scraping job from the web interface"""
    try:
        config_id = request.form.get('config_id')
        if not config_id:
            flash('No configuration selected', 'danger')
            return redirect(url_for('api.index'))
        
        # First create job with default PENDING status
        job = ScrapingJob(configuration_id=config_id)
        db.session.add(job)
        db.session.commit()
        
        # Update job status to RUNNING immediately
        # This ensures the UI shows the correct status right away
        job.status = ScrapingStatus.RUNNING
        job.start_time = datetime.utcnow()
        db.session.commit()
        
        # Then start crawler in background
        crawler = WebCrawler(job.id)
        crawler.start_async()
        
        flash(f'Job #{job.id} started successfully', 'success')
        return redirect(url_for('api.job_status', job_id=job.id))
        
    except Exception as e:
        logger.exception(f"Error starting job: {str(e)}")
        flash(f'Error: {str(e)}', 'danger')
        return redirect(url_for('api.index'))

# Stop a running job without deleting it
@api_bp.route('/jobs/stop/<int:job_id>', methods=['POST'])
def stop_job(job_id):
    """Stop a running job gracefully without deleting it"""
    job = ScrapingJob.query.get_or_404(job_id)
    
    logger.info(f"Stop job request received for job {job_id}, current status: {job.status.value}")
    
    # Only allow stopping jobs that are in the RUNNING state
    if job.status != ScrapingStatus.RUNNING:
        flash(f'Job #{job_id} is not running (current status: {job.status.value})', 'warning')
        return redirect(url_for('api.job_status', job_id=job_id))
    
    try:
        # Signal the crawler to stop
        from scraper.crawler import WebCrawler
        
        # Debug: Check active jobs before stopping
        with WebCrawler._lock:
            active_jobs = list(WebCrawler.active_jobs.keys())
            logger.debug(f"Current active jobs before stopping: {active_jobs}")
        
        was_stopped = WebCrawler.stop_job(job_id)
        
        if was_stopped:
            flash(f'Job #{job_id} has been signaled to stop gracefully. It may take a moment to finish current tasks.', 'success')
            logger.info(f"User requested to stop job {job_id}")
            
            # As a backup measure, update the job status manually if we couldn't find it in active jobs
            # This ensures the UI shows the correct status even if there's an issue with the job registry
            if job.status == ScrapingStatus.RUNNING:
                logger.info(f"Updating job {job_id} status to completed (early stop)")
                job.status = ScrapingStatus.COMPLETED
                job.end_time = datetime.utcnow()
                job.log_output = f"{job.log_output}\n[{datetime.utcnow()}] Job stopped manually by user"
                db.session.commit()
        else:
            # If we couldn't find the job in active jobs but it's still running,
            # manually update it to prevent it from being stuck in running state
            if job.status == ScrapingStatus.RUNNING:
                logger.info(f"Job {job_id} not found in active jobs registry - manually updating status")
                job.status = ScrapingStatus.COMPLETED
                job.end_time = datetime.utcnow()
                job.log_output = f"{job.log_output}\n[{datetime.utcnow()}] Job marked as completed (not found in active jobs)"
                db.session.commit()
                flash(f'Job #{job_id} was not found in the active jobs registry but has been marked as completed.', 'info')
            else:
                flash(f'Job #{job_id} could not be stopped. It may have already completed or failed.', 'warning')
                logger.warning(f"Failed to stop job {job_id} - not found in active jobs")
            
    except Exception as e:
        logger.exception(f"Error stopping job: {str(e)}")
        flash(f'Error stopping job: {str(e)}', 'danger')
    
    return redirect(url_for('api.job_status', job_id=job_id))

# Export job data to JSONL for annotation
@api_bp.route('/jobs/export/<int:job_id>', methods=['GET', 'POST'])
def export_job(job_id):
    """Export job data to JSONL format for NER/SpanCat annotation"""
    # Check if Python extraction is explicitly enabled via environment variable
    use_python_extraction = os.environ.get("USE_PYTHON_EXTRACTION", "0") == "1"
    
    if use_python_extraction:
        # Use Python-based export as configured
        from scraper.export import export_job_to_jsonl, generate_jsonl_records
        export_func = export_job_to_jsonl
        generate_records = generate_jsonl_records
        logger.info(f"Using Python-based export for job {job_id} (as configured by environment)")
    else:
        # Check if Rust export is available, otherwise use Python
        try:
            from scraper.rust_export import export_job_to_jsonl_with_rust, generate_rust_export_records
            export_func = export_job_to_jsonl_with_rust
            generate_records = generate_rust_export_records
            logger.info(f"Using Rust-based export for job {job_id}")
        except ImportError:
            from scraper.export import export_job_to_jsonl, generate_jsonl_records
            export_func = export_job_to_jsonl
            generate_records = generate_jsonl_records
            logger.info(f"Using Python-based export for job {job_id} (Rust import failed)")
    from flask import send_file, Response, stream_with_context
    import json
    from datetime import datetime
    
    job = ScrapingJob.query.get_or_404(job_id)
    
    if request.method == 'POST':
        try:
            # Set export parameters
            chunk_size = int(request.form.get('chunk_size', 500))
            overlap = int(request.form.get('overlap', 50))
            stream = 'stream' in request.form
            
            # Check if there's any content to export
            content_count = ScrapedContent.query.filter_by(job_id=job_id).count()
            if content_count == 0:
                flash('No content found to export', 'warning')
                return redirect(url_for('api.job_status', job_id=job_id))
                
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Add partial indicator for running jobs
            status_indicator = '_partial' if job.status.value == 'running' else ''
            filename = f"job_{job_id}{status_indicator}_{timestamp}.jsonl"
            
            # Prepare status message based on job status
            job_status_msg = ""
            if job.status.value == 'running':
                job_status_msg = f" (partial export from running job with {job.urls_successful} successfully scraped URLs so far)"
            
            # Always use streaming for reliability with large datasets
            logger.info(f"Streaming export for job {job_id} with {content_count} records")
            
            def generate():
                """Generate records in streaming chunks to avoid timeouts"""
                try:
                    # Yield lines in chunks to avoid timeout
                    chunk_buffer = []
                    batch_size = 100
                    record_count = 0
                    
                    for record in generate_records(job_id, chunk_size, overlap):
                        # Sanitize and convert the record to safe JSON string
                        try:
                            # Handle null bytes and other problematic characters
                            sanitized_record = {}
                            for k, v in record.items():
                                if isinstance(v, str):
                                    sanitized_record[k] = v.replace('\x00', '')
                                else:
                                    sanitized_record[k] = v
                            
                            # Handle nested metadata
                            if isinstance(sanitized_record.get('meta'), dict):
                                sanitized_meta = {}
                                for k, v in sanitized_record['meta'].items():
                                    if isinstance(v, str):
                                        sanitized_meta[k] = v.replace('\x00', '')
                                    else:
                                        sanitized_meta[k] = v
                                sanitized_record['meta'] = sanitized_meta
                            
                            # Convert to JSON and sanitize
                            json_str = json.dumps(sanitized_record, ensure_ascii=False)
                            json_str = json_str.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                            
                            # Add to buffer
                            chunk_buffer.append(json_str)
                            record_count += 1
                            
                            # Yield in batches to avoid excessive HTTP chunking
                            if len(chunk_buffer) >= batch_size:
                                yield '\n'.join(chunk_buffer) + '\n'
                                chunk_buffer = []
                                
                                # Log progress periodically
                                if record_count % 500 == 0:
                                    logger.info(f"Streaming progress: {record_count} records")
                                    
                        except Exception as e:
                            logger.warning(f"Error converting record to JSON: {str(e)}")
                            continue
                    
                    # Yield any remaining records
                    if chunk_buffer:
                        yield '\n'.join(chunk_buffer) + '\n'
                    
                    logger.info(f"Completed streaming export with {record_count} records")
                        
                except Exception as e:
                    logger.exception(f"Error in streaming export: {str(e)}")
            
            # Set content disposition headers for download
            response = Response(
                stream_with_context(generate()),
                mimetype='application/jsonl'
            )
            response.headers.set('Content-Disposition', f'attachment; filename="{filename}"')
            
            # Return streaming response
            flash(f'Starting export of {content_count} documents{job_status_msg}', 'info')
            return response
            
        except Exception as e:
            logger.exception(f"Error initializing export: {str(e)}")
            flash(f'Export error: {str(e)}', 'danger')
            return redirect(url_for('api.job_status', job_id=job_id))
    
    # GET request - show export form
    # Get content count for this job
    content_count = ScrapedContent.query.filter_by(job_id=job_id).count()
    return render_template('export.html', job=job, content_count=content_count)
