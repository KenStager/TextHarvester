"""
API routes for intelligence features in TextHarvester.

This module provides routes for managing intelligence features,
including configuration, status checks, and content analysis.
"""

import logging
import json
from datetime import datetime
from flask import Blueprint, request, jsonify, render_template, redirect, url_for, flash

from app import db
from models import ScrapingConfiguration, ScrapedContent
from models_update import ContentClassification, ContentEntity
from scraper.intelligence_integration import IntelligenceProcessor, INTELLIGENCE_AVAILABLE

# Create blueprint
intelligence_bp = Blueprint('intelligence', __name__, url_prefix='/api/intelligence')

# Set up logging
logger = logging.getLogger(__name__)

@intelligence_bp.route('/status', methods=['GET'])
def check_status():
    """Check the status of intelligence components."""
    status = {
        'intelligence_available': INTELLIGENCE_AVAILABLE,
        'classification_available': False,
        'classification_message': 'Not initialized',
        'entity_extraction_available': False,
        'entity_extraction_message': 'Not initialized'
    }
    
    if not INTELLIGENCE_AVAILABLE:
        status.update({
            'classification_message': 'Intelligence modules not available',
            'entity_extraction_message': 'Intelligence modules not available'
        })
        return jsonify(status)
    
    # Check classification pipeline
    try:
        processor = IntelligenceProcessor(enable_classification=True, enable_entity_extraction=False)
        classification_pipeline = processor.classification_pipeline
        if classification_pipeline:
            status.update({
                'classification_available': True,
                'classification_message': f'Ready with {processor.domain} domain'
            })
        else:
            status.update({
                'classification_message': 'Failed to initialize pipeline'
            })
    except Exception as e:
        status.update({
            'classification_message': f'Error: {str(e)}'
        })
    
    # Check entity extraction pipeline
    try:
        processor = IntelligenceProcessor(enable_classification=False, enable_entity_extraction=True)
        entity_pipeline = processor.entity_extraction_pipeline
        if entity_pipeline:
            status.update({
                'entity_extraction_available': True,
                'entity_extraction_message': f'Ready with {processor.domain} domain'
            })
        else:
            status.update({
                'entity_extraction_message': 'Failed to initialize pipeline'
            })
    except Exception as e:
        status.update({
            'entity_extraction_message': f'Error: {str(e)}'
        })
    
    return jsonify(status)

@intelligence_bp.route('/config/<int:config_id>', methods=['GET'])
def intelligence_config(config_id):
    """Show intelligence configuration form."""
    config = ScrapingConfiguration.query.get_or_404(config_id)
    return render_template('admin/intelligence_config.html', config=config)

@intelligence_bp.route('/config/<int:config_id>', methods=['POST'])
def update_intelligence_config(config_id):
    """Update intelligence configuration."""
    config = ScrapingConfiguration.query.get_or_404(config_id)
    
    try:
        # Update intelligence settings
        config.enable_classification = 'enable_classification' in request.form
        config.enable_entity_extraction = 'enable_entity_extraction' in request.form
        config.intelligence_domain = request.form.get('intelligence_domain', 'football')
        
        # Update advanced settings
        store_raw = 'store_raw_intelligence' in request.form
        
        # Initialize or update intelligence_config JSON
        if not hasattr(config, 'intelligence_config') or config.intelligence_config is None:
            config.intelligence_config = {}
        
        # Convert to dict if it's a string or other format
        if not isinstance(config.intelligence_config, dict):
            try:
                config.intelligence_config = json.loads(config.intelligence_config)
            except (json.JSONDecodeError, TypeError):
                config.intelligence_config = {}
        
        # Update config
        config.intelligence_config['store_raw_intelligence'] = store_raw
        
        # Save to database
        db.session.commit()
        
        flash('Intelligence configuration updated successfully', 'success')
        return redirect(url_for('api.configuration'))
    
    except Exception as e:
        logger.error(f"Error updating intelligence configuration: {str(e)}")
        db.session.rollback()
        flash(f'Error updating configuration: {str(e)}', 'danger')
        return redirect(url_for('intelligence.intelligence_config', config_id=config_id))

@intelligence_bp.route('/analyze/<int:content_id>', methods=['GET'])
def analyze_content(content_id):
    """Manually analyze a specific content with intelligence pipelines."""
    content = ScrapedContent.query.get_or_404(content_id)
    
    # Check if content already has intelligence processing
    has_classification = db.session.query(ContentClassification).filter_by(content_id=content_id).first() is not None
    has_entities = db.session.query(ContentEntity).filter_by(content_id=content_id).count() > 0
    
    if has_classification and has_entities:
        flash('This content has already been processed by intelligence pipelines', 'info')
        return redirect(url_for('api.view_content', content_id=content_id))
    
    # Initialize intelligence processor with default settings
    processor = IntelligenceProcessor(
        domain="football",  # Default domain
        enable_classification=not has_classification,
        enable_entity_extraction=not has_entities
    )
    
    try:
        # Process content
        results = processor.process_content(content)
        
        # Flash appropriate messages
        if results.get('classification'):
            flash(f'Content classified as "{results["classification"].primary_topic}" with {results["classification"].primary_topic_confidence:.1%} confidence', 'success')
        
        if results.get('entities'):
            entity_count = len(results['entities'].entities)
            flash(f'Extracted {entity_count} entities from content', 'success')
        
        flash(f'Intelligence processing completed in {results["processing_time"]:.2f} seconds', 'info')
        
        return redirect(url_for('api.view_content', content_id=content_id))
    
    except Exception as e:
        logger.error(f"Error during manual intelligence processing: {str(e)}")
        flash(f'Error during intelligence processing: {str(e)}', 'danger')
        return redirect(url_for('api.view_content', content_id=content_id))

@intelligence_bp.route('/content/<int:content_id>/entities', methods=['GET'])
def view_content_entities(content_id):
    """View entities extracted from content."""
    content = ScrapedContent.query.get_or_404(content_id)
    entities = ContentEntity.query.filter_by(content_id=content_id).all()
    
    # Group entities by type
    entity_types = {}
    for entity in entities:
        if entity.entity_type not in entity_types:
            entity_types[entity.entity_type] = []
        entity_types[entity.entity_type].append(entity)
    
    return render_template('content/entities.html', content=content, entities=entities, entity_types=entity_types)

@intelligence_bp.route('/content/<int:content_id>/classification', methods=['GET'])
def view_content_classification(content_id):
    """View classification for content."""
    content = ScrapedContent.query.get_or_404(content_id)
    classification = ContentClassification.query.filter_by(content_id=content_id).first()
    
    if not classification:
        flash('No classification available for this content', 'warning')
        return redirect(url_for('api.view_content', content_id=content_id))
    
    return render_template('content/classification.html', content=content, classification=classification)

@intelligence_bp.route('/overview', methods=['GET'])
def intelligence_overview():
    """Show intelligence overview dashboard."""
    # Calculate intelligence statistics
    total_classification = db.session.query(ContentClassification).count()
    total_entities = db.session.query(ContentEntity).count()
    
    # Get distinct domains
    active_domains = db.session.query(db.func.count(db.func.distinct(ScrapingConfiguration.intelligence_domain)))\
        .filter(ScrapingConfiguration.enable_classification == True)\
        .scalar()
    
    stats = {
        'total_classification': total_classification,
        'total_entities': total_entities,
        'active_domains': active_domains or 0
    }
    
    # Get recent classifications
    recent_classifications = ContentClassification.query\
        .order_by(ContentClassification.created_at.desc())\
        .limit(10).all()
    
    # Get recent entities
    recent_entities = ContentEntity.query\
        .order_by(ContentEntity.created_at.desc())\
        .limit(10).all()
    
    # Get topic distribution data for chart
    topics_query = db.session.query(
        ContentClassification.primary_topic,
        db.func.count(ContentClassification.id)
    ).group_by(ContentClassification.primary_topic)\
        .order_by(db.func.count(ContentClassification.id).desc())\
        .limit(10).all()
    
    topics_data = {
        'labels': [topic for topic, count in topics_query],
        'values': [count for topic, count in topics_query]
    }
    
    # Get entity types distribution data for chart
    entities_query = db.session.query(
        ContentEntity.entity_type,
        db.func.count(ContentEntity.id)
    ).group_by(ContentEntity.entity_type)\
        .order_by(db.func.count(ContentEntity.id).desc())\
        .all()
    
    entities_data = {
        'labels': [entity_type for entity_type, count in entities_query],
        'values': [count for entity_type, count in entities_query]
    }
    
    # Get recent activity
    recent_activity = []
    
    # Add recent classifications to activity
    for classification in recent_classifications[:5]:
        if hasattr(classification, 'content') and classification.content:
            recent_activity.append({
                'title': f"Classification: {classification.primary_topic}",
                'description': f"Document: {classification.content.title or 'Untitled'}",
                'details': f"Confidence: {classification.primary_topic_confidence:.0%}",
                'time_ago': format_time_ago(classification.created_at)
            })
    
    # Add recent entity extractions to activity
    entity_docs = {}
    for entity in recent_entities[:10]:
        if entity.content_id not in entity_docs:
            entity_docs[entity.content_id] = {
                'title': entity.content.title if hasattr(entity, 'content') and entity.content else 'Untitled',
                'count': 1,
                'time': entity.created_at
            }
        else:
            entity_docs[entity.content_id]['count'] += 1
    
    for doc_id, doc_info in list(entity_docs.items())[:5]:
        recent_activity.append({
            'title': f"Entity Extraction",
            'description': f"Document: {doc_info['title']}",
            'details': f"Extracted {doc_info['count']} entities",
            'time_ago': format_time_ago(doc_info['time'])
        })
    
    # Sort activity by time
    recent_activity.sort(key=lambda x: x['time_ago'], reverse=True)
    
    return render_template('intelligence/overview.html',
                          stats=stats,
                          recent_classifications=recent_classifications,
                          recent_entities=recent_entities,
                          topics_data=topics_data,
                          entities_data=entities_data,
                          recent_activity=recent_activity)

def format_time_ago(timestamp):
    """Format a timestamp as a human-readable 'time ago' string."""
    now = datetime.utcnow()
    diff = now - timestamp
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return f"{int(seconds)} seconds ago"
    elif seconds < 3600:
        return f"{int(seconds / 60)} minutes ago"
    elif seconds < 86400:
        return f"{int(seconds / 3600)} hours ago"
    elif seconds < 604800:
        return f"{int(seconds / 86400)} days ago"
    else:
        return timestamp.strftime('%Y-%m-%d')

def register_blueprint(app):
    """Register the intelligence blueprint with the Flask app."""
    app.register_blueprint(intelligence_bp)
    logger.info("Registered intelligence blueprint")
