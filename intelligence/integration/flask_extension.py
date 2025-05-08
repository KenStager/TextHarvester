"""
Flask Extension for Content Intelligence Integration.

This module provides a Flask extension that integrates the Content Intelligence
Platform with the TextHarvester web interface.
"""

import logging
import os
import sys
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

# Add the project root to path to ensure imports work properly
sys.path.append(str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


class ContentIntelligence:
    """
    Flask extension for Content Intelligence integration.
    
    This class provides a Flask extension that integrates the Content Intelligence
    Platform with the TextHarvester web interface, adding intelligence features
    to the existing scraper UI.
    """
    
    def __init__(self, app=None):
        """
        Initialize the extension.
        
        Args:
            app: Optional Flask app to initialize with
        """
        self.app = app
        
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """
        Initialize the extension with a Flask app.
        
        Args:
            app: Flask app to initialize with
        """
        self.app = app
        
        # Register extension with app
        app.extensions['content_intelligence'] = self
        
        # Set up configuration
        app.config.setdefault('CONTENT_INTELLIGENCE_ENABLED', True)
        app.config.setdefault('CONTENT_INTELLIGENCE_REALTIME', False)
        
        # Register routes and hooks
        self._register_routes(app)
        self._setup_hooks(app)
        
        logger.info("Content Intelligence extension initialized")
    
    def _register_routes(self, app):
        """
        Register routes with the Flask app.
        
        Args:
            app: Flask app to register routes with
        """
        from flask import Blueprint, request, jsonify, render_template, redirect, url_for, flash
        
        # Create a blueprint for the intelligence UI
        intelligence_bp = Blueprint(
            'intelligence_ui', 
            __name__,
            url_prefix='/intelligence',
            template_folder='../templates/intelligence',
            static_folder='../static/intelligence'
        )
        
        @intelligence_bp.route('/', methods=['GET'])
        def index():
            """Intelligence dashboard."""
            # This would normally render an intelligence dashboard template
            return render_template(
                'intelligence/dashboard.html', 
                title="Content Intelligence Dashboard"
            )
        
        @intelligence_bp.route('/content/<int:content_id>', methods=['GET'])
        def view_content(content_id):
            """View content with intelligence enhancements."""
            # Import required models
            from models import ScrapedContent, ContentMetadata
            
            # Get the content
            content = ScrapedContent.query.get_or_404(content_id)
            
            # Check if we have intelligence data for this content
            intelligence_data = self._get_intelligence_data(content_id)
            
            # Render the template
            return render_template(
                'intelligence/content.html',
                content=content,
                intelligence=intelligence_data,
                title=f"Content Intelligence: {content.title or 'Untitled'}"
            )
        
        @intelligence_bp.route('/process/<int:content_id>', methods=['POST'])
        def process_content(content_id):
            """Process a content item with intelligence."""
            try:
                # Import the integration manager
                from intelligence.integration.scraper_hooks import IntegrationManager
                
                # Process the content
                manager = IntegrationManager.get_instance()
                result = manager.process_content_by_id(content_id)
                
                if 'error' in result:
                    flash(f"Error processing content: {result['error']}", 'danger')
                else:
                    flash("Content processed successfully", 'success')
                
                # Redirect to content view
                return redirect(url_for('intelligence_ui.view_content', content_id=content_id))
                
            except Exception as e:
                logger.error(f"Error processing content {content_id}: {str(e)}")
                flash(f"Error: {str(e)}", 'danger')
                return redirect(url_for('api.view_job_content', job_id=content_id))
        
        @intelligence_bp.route('/batch', methods=['GET', 'POST'])
        def batch_processing():
            """Batch processing interface."""
            if request.method == 'POST':
                try:
                    # Get form data
                    job_id = request.form.get('job_id')
                    min_word_count = int(request.form.get('min_word_count', 100))
                    
                    # Import batch processor
                    from intelligence.integration.batch_processor import BatchProcessor
                    
                    # Create processor
                    processor = BatchProcessor()
                    
                    # Build options
                    options = {
                        'min_word_count': min_word_count
                    }
                    
                    if job_id:
                        options['job_id'] = int(job_id)
                    
                    # Process backlog
                    result = processor.process_backlog(options)
                    
                    if 'error' in result:
                        flash(f"Error: {result['error']}", 'danger')
                    else:
                        flash(f"Started batch processing: {result['message']}", 'success')
                        
                        # Redirect to job status
                        if 'job_id' in result:
                            return redirect(url_for('intelligence_ui.batch_status', job_id=result['job_id']))
                    
                    return redirect(url_for('intelligence_ui.batch_processing'))
                    
                except Exception as e:
                    logger.error(f"Error starting batch processing: {str(e)}")
                    flash(f"Error: {str(e)}", 'danger')
                    return redirect(url_for('intelligence_ui.batch_processing'))
            
            # GET request - show batch processing form
            from models import ScrapingJob
            
            # Get recent jobs for form
            recent_jobs = ScrapingJob.query.order_by(ScrapingJob.created_at.desc()).limit(10).all()
            
            return render_template(
                'intelligence/batch.html',
                title="Batch Processing",
                jobs=recent_jobs
            )
        
        @intelligence_bp.route('/batch/<job_id>', methods=['GET'])
        def batch_status(job_id):
            """View status of a batch processing job."""
            # Import batch processor
            from intelligence.integration.batch_processor import BatchProcessor
            
            # Create processor
            processor = BatchProcessor()
            
            # Get job status
            status = processor.get_job_status(job_id)
            
            if 'error' in status:
                flash(f"Error: {status['error']}", 'danger')
                return redirect(url_for('intelligence_ui.batch_processing'))
            
            return render_template(
                'intelligence/batch_status.html',
                title="Batch Job Status",
                status=status
            )
        
        # Register the blueprint with the app
        app.register_blueprint(intelligence_bp)
        
        # Also register API routes
        self._register_api_routes(app)
        
        logger.info("Registered intelligence routes")
    
    def _register_api_routes(self, app):
        """
        Register API routes with the Flask app.
        
        Args:
            app: Flask app to register API routes with
        """
        # Import scraper hooks to register API routes
        from intelligence.integration.scraper_hooks import register_api_routes
        
        # Register API routes
        register_api_routes(app)
    
    def _setup_hooks(self, app):
        """
        Set up hooks with the scraper.
        
        Args:
            app: Flask app
        """
        # Import scraper hooks
        from intelligence.integration.scraper_hooks import setup_scraper_integration
        
        # Set up scraper integration
        if app.config.get('CONTENT_INTELLIGENCE_ENABLED', True):
            integration_success = setup_scraper_integration(app)
            
            if integration_success:
                logger.info("Scraper integration set up successfully")
            else:
                logger.warning("Failed to set up scraper integration")
    
    def _get_intelligence_data(self, content_id):
        """
        Get intelligence data for a content item.
        
        In a real implementation, this would query the database for
        intelligence data. For now, it returns dummy data.
        
        Args:
            content_id: ID of the content item
            
        Returns:
            Dictionary with intelligence data
        """
        try:
            # In a real implementation, this would query the database
            # For now, return an empty dict
            return {}
            
        except Exception as e:
            logger.error(f"Error getting intelligence data for content {content_id}: {str(e)}")
            return {}


def create_templates():
    """
    Create template files for the intelligence interface.
    
    This function creates the necessary template files for the intelligence
    interface if they don't already exist.
    """
    template_dir = Path(__file__).parent.parent.parent / 'templates' / 'intelligence'
    os.makedirs(template_dir, exist_ok=True)
    
    # Dashboard template
    dashboard_template = template_dir / 'dashboard.html'
    if not dashboard_template.exists():
        with open(dashboard_template, 'w') as f:
            f.write("""{% extends 'base.html' %}

{% block title %}Content Intelligence Dashboard{% endblock %}

{% block content %}
<div class="container mt-4">
    <h1>Content Intelligence Dashboard</h1>
    
    <div class="row mt-4">
        <div class="col-md-4">
            <div class="card">
                <div class="card-header">
                    <h5 class="card-title">Classification</h5>
                </div>
                <div class="card-body">
                    <p>Analyze content relevance and topic categorization.</p>
                    <a href="#" class="btn btn-primary">View Topics</a>
                </div>
            </div>
        </div>
        
        <div class="col-md-4">
            <div class="card">
                <div class="card-header">
                    <h5 class="card-title">Entities</h5>
                </div>
                <div class="card-body">
                    <p>Explore extracted entities and relationships.</p>
                    <a href="#" class="btn btn-primary">View Entities</a>
                </div>
            </div>
        </div>
        
        <div class="col-md-4">
            <div class="card">
                <div class="card-header">
                    <h5 class="card-title">Batch Processing</h5>
                </div>
                <div class="card-body">
                    <p>Process multiple content items in batch mode.</p>
                    <a href="{{ url_for('intelligence_ui.batch_processing') }}" class="btn btn-primary">Batch Processing</a>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}""")
    
    # Content template
    content_template = template_dir / 'content.html'
    if not content_template.exists():
        with open(content_template, 'w') as f:
            f.write("""{% extends 'base.html' %}

{% block title %}Content Intelligence: {{ content.title or 'Untitled' }}{% endblock %}

{% block content %}
<div class="container mt-4">
    <h1>Content Intelligence</h1>
    
    <div class="card mb-4">
        <div class="card-header">
            <h2>{{ content.title or 'Untitled Content' }}</h2>
            <small class="text-muted">{{ content.url }}</small>
        </div>
        <div class="card-body">
            <div class="row">
                <div class="col-md-8">
                    <h3>Content Text</h3>
                    <div class="border p-3" style="max-height: 400px; overflow-y: auto;">
                        {{ content.extracted_text|nl2br }}
                    </div>
                </div>
                
                <div class="col-md-4">
                    <h3>Intelligence</h3>
                    {% if intelligence %}
                        <div class="mb-3">
                            <h4>Classification</h4>
                            {% if intelligence.classification %}
                                <p><strong>Primary Topic:</strong> {{ intelligence.classification.primary_topic }}</p>
                                <p><strong>Confidence:</strong> {{ intelligence.classification.confidence|round(2) }}</p>
                                
                                {% if intelligence.classification.subtopics %}
                                    <h5>Subtopics</h5>
                                    <ul>
                                        {% for subtopic in intelligence.classification.subtopics %}
                                            <li>{{ subtopic.topic }} ({{ subtopic.confidence|round(2) }})</li>
                                        {% endfor %}
                                    </ul>
                                {% endif %}
                            {% else %}
                                <p>No classification data available.</p>
                            {% endif %}
                        </div>
                        
                        <div class="mb-3">
                            <h4>Entities</h4>
                            {% if intelligence.entities %}
                                <ul>
                                    {% for entity in intelligence.entities.entities %}
                                        <li>{{ entity.text }} ({{ entity.label }})</li>
                                    {% endfor %}
                                </ul>
                            {% else %}
                                <p>No entity data available.</p>
                            {% endif %}
                        </div>
                    {% else %}
                        <div class="alert alert-info">
                            No intelligence data available for this content.
                        </div>
                        
                        <form action="{{ url_for('intelligence_ui.process_content', content_id=content.id) }}" method="post">
                            <button type="submit" class="btn btn-primary">Process with Intelligence</button>
                        </form>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
    
    <a href="{{ url_for('api.view_job_content', job_id=content.job_id) }}" class="btn btn-secondary">Back to Content List</a>
</div>
{% endblock %}""")
    
    # Batch processing template
    batch_template = template_dir / 'batch.html'
    if not batch_template.exists():
        with open(batch_template, 'w') as f:
            f.write("""{% extends 'base.html' %}

{% block title %}Batch Processing{% endblock %}

{% block content %}
<div class="container mt-4">
    <h1>Batch Processing</h1>
    
    <div class="card mb-4">
        <div class="card-header">
            <h3>Process Content</h3>
        </div>
        <div class="card-body">
            <form action="{{ url_for('intelligence_ui.batch_processing') }}" method="post">
                <div class="mb-3">
                    <label for="job_id" class="form-label">Job ID (Optional)</label>
                    <select name="job_id" id="job_id" class="form-select">
                        <option value="">All Jobs</option>
                        {% for job in jobs %}
                            <option value="{{ job.id }}">Job #{{ job.id }} ({{ job.urls_successful }} URLs)</option>
                        {% endfor %}
                    </select>
                    <div class="form-text">Select a specific job to process or leave blank for all content.</div>
                </div>
                
                <div class="mb-3">
                    <label for="min_word_count" class="form-label">Minimum Word Count</label>
                    <input type="number" name="min_word_count" id="min_word_count" class="form-control" value="100" min="0">
                    <div class="form-text">Only process content with at least this many words.</div>
                </div>
                
                <button type="submit" class="btn btn-primary">Start Batch Processing</button>
            </form>
        </div>
    </div>
    
    <a href="{{ url_for('intelligence_ui.index') }}" class="btn btn-secondary">Back to Dashboard</a>
</div>
{% endblock %}""")
    
    # Batch status template
    batch_status_template = template_dir / 'batch_status.html'
    if not batch_status_template.exists():
        with open(batch_status_template, 'w') as f:
            f.write("""{% extends 'base.html' %}

{% block title %}Batch Job Status{% endblock %}

{% block content %}
<div class="container mt-4">
    <h1>Batch Job Status</h1>
    
    <div class="card mb-4">
        <div class="card-header">
            <h3>Job #{{ status.job_id }}</h3>
        </div>
        <div class="card-body">
            <div class="row">
                <div class="col-md-6">
                    <p><strong>Status:</strong> {{ status.status }}</p>
                    <p><strong>Created:</strong> {{ status.created_at }}</p>
                    <p><strong>Progress:</strong> {{ status.progress }} / {{ status.total }} items ({{ status.percent_complete|round }}%)</p>
                </div>
                
                <div class="col-md-6">
                    <div class="progress mb-3">
                        <div class="progress-bar" role="progressbar" style="width: {{ status.percent_complete }}%;" 
                             aria-valuenow="{{ status.percent_complete|round }}" aria-valuemin="0" aria-valuemax="100">
                            {{ status.percent_complete|round }}%
                        </div>
                    </div>
                    
                    {% if status.status == 'completed' %}
                        <div class="alert alert-success">
                            Job completed successfully.
                        </div>
                    {% elif status.status == 'failed' %}
                        <div class="alert alert-danger">
                            Job failed: {{ status.results.error }}
                        </div>
                    {% else %}
                        <div class="alert alert-info">
                            Job is {{ status.status }}...
                        </div>
                    {% endif %}
                </div>
            </div>
            
            {% if status.results %}
                <div class="mt-4">
                    <h4>Results</h4>
                    <p><strong>Successful:</strong> {{ status.results.successful or 0 }}</p>
                    <p><strong>Failed:</strong> {{ status.results.failed or 0 }}</p>
                </div>
            {% endif %}
        </div>
    </div>
    
    <a href="{{ url_for('intelligence_ui.batch_processing') }}" class="btn btn-secondary">Back to Batch Processing</a>
</div>

{% if status.status in ['pending', 'running'] %}
    <script>
        // Auto-refresh the page every 5 seconds if job is still processing
        setTimeout(function() {
            window.location.reload();
        }, 5000);
    </script>
{% endif %}
{% endblock %}""")
    
    logger.info(f"Created template files in {template_dir}")
