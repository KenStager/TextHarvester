"""
API routes for source management
"""

import re
import time
import logging
import requests
from urllib.parse import urlparse
from datetime import datetime

from flask import Blueprint, request, render_template, redirect, url_for, flash, jsonify
from sqlalchemy.exc import IntegrityError

from app import db
from models import Source, SourceList
from scraper.source_lists import PREDEFINED_SOURCES
from scraper.utils import get_random_user_agent
from scraper.content_extractor import extract_content

logger = logging.getLogger(__name__)

sources_bp = Blueprint('sources', __name__, url_prefix='/sources')

@sources_bp.route('/')
def index():
    """List all sources and source lists"""
    sources = Source.query.order_by(Source.priority.desc(), Source.name).all()
    source_lists = SourceList.query.order_by(SourceList.name).all()
    return render_template('sources/index.html', sources=sources, source_lists=source_lists)

@sources_bp.route('/lists')
def list_source_lists():
    """List all source lists"""
    source_lists = SourceList.query.order_by(SourceList.name).all()
    return render_template('sources/lists.html', source_lists=source_lists)

@sources_bp.route('/lists/<int:list_id>')
def view_source_list(list_id):
    """View a specific source list"""
    source_list = SourceList.query.get_or_404(list_id)
    return render_template('sources/list_detail.html', source_list=source_list)

@sources_bp.route('/lists/<int:list_id>/test', methods=['GET'])
def test_source_list(list_id):
    """Test all sources in a source list"""
    source_list = SourceList.query.get_or_404(list_id)
    
    # Get all active sources in this list
    sources = source_list.sources
    
    # Track successful and failed sources
    results = {
        'list_id': list_id,
        'list_name': source_list.name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_sources': len(sources),
        'successful': 0,
        'failed': 0,
        'not_tested': 0,
        'source_results': []
    }
    
    for source in sources:
        source_result = {
            'source_id': source.id,
            'name': source.name,
            'url': source.url,
            'status': 'pending',
            'status_code': None,
            'response_time': None,
            'error': None,
            'extracted_text_length': 0
        }
        
        try:
            # Use a random user agent
            headers = {
                'User-Agent': get_random_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml',
                'Accept-Language': 'en-US,en;q=0.9',
                'DNT': '1',
                'Referer': 'https://www.google.com/'
            }
            
            # Fetch the source with a timeout
            start_time = time.time()
            response = requests.get(source.url, headers=headers, timeout=10, allow_redirects=True)
            response_time = time.time() - start_time
            
            # Update results
            source_result['status_code'] = response.status_code
            source_result['response_time'] = round(response_time * 1000)  # Convert to ms
            
            if response.status_code == 200:
                source_result['status'] = 'success'
                results['successful'] += 1
                
                # Try to extract content
                if response.content:
                    _, extracted_text, _ = extract_content(source.url, response.content)
                    if extracted_text:
                        source_result['extracted_text_length'] = len(extracted_text)
                
                # Update source status in database
                try:
                    source.is_active = True
                    db.session.commit()
                except Exception as e:
                    logger.error(f"Error updating source status: {str(e)}")
                    db.session.rollback()  # Rollback to prevent transaction errors
            else:
                source_result['status'] = 'failed'
                source_result['error'] = f"HTTP Error {response.status_code}: {response.reason}"
                results['failed'] += 1
                
        except Exception as e:
            source_result['status'] = 'failed'
            source_result['error'] = str(e)
            results['failed'] += 1
        
        results['source_results'].append(source_result)
    
    return render_template('sources/test_list.html', source_list=source_list, results=results)

@sources_bp.route('/lists/new', methods=['GET', 'POST'])
def new_source_list():
    """Create a new source list"""
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        description = request.form.get('description', '').strip()
        
        if not name:
            flash('Source list name is required', 'danger')
            return redirect(url_for('sources.new_source_list'))
        
        # Generate a slug from the name
        slug = re.sub(r'[^\w]+', '_', name.lower())
        
        try:
            source_list = SourceList(
                name=name,
                slug=slug,
                description=description,
                is_public=True
            )
            db.session.add(source_list)
            db.session.commit()
            flash(f'Source list "{name}" created successfully', 'success')
            return redirect(url_for('sources.view_source_list', list_id=source_list.id))
        except IntegrityError:
            db.session.rollback()
            flash(f'A source list with name or slug "{name}" already exists', 'danger')
            return redirect(url_for('sources.new_source_list'))
    
    return render_template('sources/new_list.html')

@sources_bp.route('/lists/<int:list_id>/edit', methods=['GET', 'POST'])
def edit_source_list(list_id):
    """Edit a source list"""
    source_list = SourceList.query.get_or_404(list_id)
    
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        description = request.form.get('description', '').strip()
        is_public = 'is_public' in request.form
        
        if not name:
            flash('Source list name is required', 'danger')
            return redirect(url_for('sources.edit_source_list', list_id=list_id))
        
        try:
            source_list.name = name
            source_list.description = description
            source_list.is_public = is_public
            db.session.commit()
            flash(f'Source list "{name}" updated successfully', 'success')
            return redirect(url_for('sources.view_source_list', list_id=list_id))
        except IntegrityError:
            db.session.rollback()
            flash(f'Error updating source list: a source list with name "{name}" already exists', 'danger')
    
    return render_template('sources/edit_list.html', source_list=source_list)

@sources_bp.route('/lists/<int:list_id>/delete', methods=['POST'])
def delete_source_list(list_id):
    """Delete a source list"""
    source_list = SourceList.query.get_or_404(list_id)
    
    try:
        db.session.delete(source_list)
        db.session.commit()
        flash(f'Source list "{source_list.name}" deleted successfully', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting source list: {str(e)}', 'danger')
    
    return redirect(url_for('sources.index'))

@sources_bp.route('/new', methods=['GET', 'POST'])
def new_source():
    """Create a new source"""
    source_lists = SourceList.query.order_by(SourceList.name).all()
    
    if request.method == 'POST':
        url = request.form.get('url', '').strip()
        name = request.form.get('name', '').strip()
        description = request.form.get('description', '').strip()
        category = request.form.get('category', '').strip()
        priority = int(request.form.get('priority', 0))
        is_active = 'is_active' in request.form
        source_list_ids = request.form.getlist('source_lists')
        
        if not url:
            flash('Source URL is required', 'danger')
            return redirect(url_for('sources.new_source'))
        
        try:
            source = Source(
                url=url,
                name=name or url,  # Use URL as name if not provided
                description=description,
                category=category,
                priority=priority,
                is_active=is_active
            )
            
            # Add source to selected source lists
            if source_list_ids:
                selected_lists = SourceList.query.filter(SourceList.id.in_(source_list_ids)).all()
                source.source_lists = selected_lists
            
            db.session.add(source)
            db.session.commit()
            flash(f'Source "{name or url}" created successfully', 'success')
            return redirect(url_for('sources.index'))
        except IntegrityError:
            db.session.rollback()
            flash(f'A source with URL "{url}" already exists', 'danger')
            return redirect(url_for('sources.new_source'))
    
    return render_template('sources/new_source.html', source_lists=source_lists)

@sources_bp.route('/<int:source_id>/edit', methods=['GET', 'POST'])
def edit_source(source_id):
    """Edit a source"""
    source = Source.query.get_or_404(source_id)
    source_lists = SourceList.query.order_by(SourceList.name).all()
    
    if request.method == 'POST':
        url = request.form.get('url', '').strip()
        name = request.form.get('name', '').strip()
        description = request.form.get('description', '').strip()
        category = request.form.get('category', '').strip()
        priority = int(request.form.get('priority', 0))
        is_active = 'is_active' in request.form
        source_list_ids = request.form.getlist('source_lists')
        
        if not url:
            flash('Source URL is required', 'danger')
            return redirect(url_for('sources.edit_source', source_id=source_id))
        
        try:
            source.url = url
            source.name = name or url
            source.description = description
            source.category = category
            source.priority = priority
            source.is_active = is_active
            
            # Update source lists
            selected_lists = SourceList.query.filter(SourceList.id.in_(source_list_ids)).all()
            source.source_lists = selected_lists
            
            db.session.commit()
            flash(f'Source "{name or url}" updated successfully', 'success')
            return redirect(url_for('sources.index'))
        except IntegrityError:
            db.session.rollback()
            flash(f'Error updating source: a source with URL "{url}" already exists', 'danger')
    
    return render_template('sources/edit_source.html', source=source, source_lists=source_lists)

@sources_bp.route('/<int:source_id>/delete', methods=['POST'])
def delete_source(source_id):
    """Delete a source"""
    source = Source.query.get_or_404(source_id)
    
    try:
        db.session.delete(source)
        db.session.commit()
        flash(f'Source "{source.name}" deleted successfully', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting source: {str(e)}', 'danger')
    
    return redirect(url_for('sources.index'))

@sources_bp.route('/lists/<int:list_id>/add_source', methods=['GET', 'POST'])
def add_source_to_list(list_id):
    """Add a source to a list"""
    source_list = SourceList.query.get_or_404(list_id)
    
    # Get sources not already in this list
    existing_source_ids = [source.id for source in source_list.sources]
    available_sources = Source.query.filter(~Source.id.in_(existing_source_ids) if existing_source_ids else True).all()
    
    if request.method == 'POST':
        source_ids = request.form.getlist('sources')
        if not source_ids:
            flash('No sources selected', 'warning')
            return redirect(url_for('sources.add_source_to_list', list_id=list_id))
        
        sources = Source.query.filter(Source.id.in_(source_ids)).all()
        
        try:
            source_list.sources.extend(sources)
            db.session.commit()
            flash(f'Added {len(sources)} sources to "{source_list.name}"', 'success')
            return redirect(url_for('sources.view_source_list', list_id=list_id))
        except Exception as e:
            db.session.rollback()
            flash(f'Error adding sources: {str(e)}', 'danger')
    
    return render_template('sources/add_source_to_list.html', source_list=source_list, available_sources=available_sources)

@sources_bp.route('/lists/<int:list_id>/remove_source/<int:source_id>', methods=['POST'])
def remove_source_from_list(list_id, source_id):
    """Remove a source from a list"""
    source_list = SourceList.query.get_or_404(list_id)
    source = Source.query.get_or_404(source_id)
    
    try:
        source_list.sources.remove(source)
        db.session.commit()
        flash(f'Removed "{source.name}" from "{source_list.name}"', 'success')
    except ValueError:
        flash(f'Source is not in the list', 'warning')
    except Exception as e:
        db.session.rollback()
        flash(f'Error removing source: {str(e)}', 'danger')
    
    return redirect(url_for('sources.view_source_list', list_id=list_id))

@sources_bp.route('/test/<int:source_id>', methods=['GET'])
def test_source(source_id):
    """Test a source by attempting to fetch and extract content"""
    source = Source.query.get_or_404(source_id)
    url = source.url
    
    results = {
        'url': url,
        'source_id': source_id, 
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'success': False,
        'status_code': None,
        'response_time': None,
        'headers': None,
        'content_preview': None,
        'extracted_text_preview': None,
        'extracted_text_length': 0,
        'error': None,
        'recommendations': []
    }
    
    try:
        # Use a random user agent
        headers = {
            'User-Agent': get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9',
            'DNT': '1',  # Do Not Track
            'Referer': 'https://www.google.com/'  # Common referer to avoid some blocks
        }
        
        # Measure response time
        start_time = time.time()
        response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        response_time = time.time() - start_time
        
        # Update results
        results['status_code'] = response.status_code
        results['response_time'] = round(response_time * 1000)  # Convert to ms
        results['headers'] = dict(response.headers)
        
        # Check if request was successful
        if response.status_code == 200:
            results['success'] = True
            
            # Get a preview of the content
            results['content_preview'] = response.text[:1000] + '...' if len(response.text) > 1000 else response.text
            
            # Try to extract the content
            if response.content:
                title, extracted_text, _ = extract_content(url, response.content)
                if extracted_text:
                    results['extracted_text_preview'] = extracted_text[:500] + '...' if len(extracted_text) > 500 else extracted_text
                    results['extracted_text_length'] = len(extracted_text)
        else:
            # Add error information
            results['error'] = f"HTTP Error {response.status_code}: {response.reason}"
            
            # Add recommendations based on status code
            if response.status_code == 403:
                results['recommendations'].append("The site is blocking scraping requests. Consider these solutions:")
                results['recommendations'].append("- Add a delay between requests (set higher rate limit)")
                results['recommendations'].append("- Use more varied user agents")
                results['recommendations'].append("- Add referrer headers")
                results['recommendations'].append("- Consider using a proxy service or rotating IP addresses")
            elif response.status_code == 404:
                results['recommendations'].append("The URL doesn't exist. Check if the URL is correct.")
            elif response.status_code == 429:
                results['recommendations'].append("Too many requests. Reduce request frequency and implement proper rate limiting.")
            elif response.status_code >= 500:
                results['recommendations'].append("Server error. Try again later or reduce request frequency.")
                
    except requests.exceptions.Timeout:
        results['error'] = "Connection timed out"
        results['recommendations'].append("The site took too long to respond. It might be temporarily down or blocking requests.")
    except requests.exceptions.ConnectionError as e:
        results['error'] = f"Connection error: {str(e)}"
        if "Name or service not known" in str(e):
            results['recommendations'].append("Domain name could not be resolved. Check if the URL is correct.")
        else:
            results['recommendations'].append("Connection was refused or reset. The site might be blocking requests or temporarily down.")
    except Exception as e:
        results['error'] = f"Error: {str(e)}"
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify(results)
    
    # Mark the source as valid or invalid based on test results
    try:
        if results['success']:
            source.is_active = True
            db.session.commit()
    except Exception as e:
        logger.error(f"Error updating source status: {str(e)}")
        # Don't let database errors affect the test results display
        flash(f"Warning: Could not update source status in database: {str(e)}", "warning")
    
    return render_template('sources/test_source.html', source=source, results=results)

@sources_bp.route('/import_predefined', methods=['GET', 'POST'])
def import_predefined_sources():
    """Import sources from predefined lists"""
    if request.method == 'POST':
        list_id = request.form.get('predefined_list')
        if not list_id or list_id not in PREDEFINED_SOURCES:
            flash('Invalid predefined source list selected', 'danger')
            return redirect(url_for('sources.import_predefined_sources'))
        
        predefined_list = PREDEFINED_SOURCES[list_id]
        
        # Create a new source list if it doesn't exist
        slug = re.sub(r'[^\w]+', '_', predefined_list['name'].lower())
        source_list = SourceList.query.filter_by(slug=slug).first()
        
        if not source_list:
            source_list = SourceList(
                name=predefined_list['name'],
                slug=slug,
                description=predefined_list['description'],
                is_public=True
            )
            db.session.add(source_list)
        
        # Add each source from the predefined list
        count_added = 0
        for url in predefined_list['sources']:
            existing_source = Source.query.filter_by(url=url).first()
            
            if existing_source:
                # If the source exists but isn't in the list, add it
                if existing_source not in source_list.sources:
                    source_list.sources.append(existing_source)
                    count_added += 1
            else:
                # Create new source
                name = url.replace('https://', '').replace('http://', '').split('/')[0]
                source = Source(
                    url=url,
                    name=name,
                    category=list_id,
                    is_active=True
                )
                db.session.add(source)
                source_list.sources.append(source)
                count_added += 1
        
        try:
            db.session.commit()
            flash(f'Imported {count_added} sources into "{predefined_list["name"]}" list', 'success')
            return redirect(url_for('sources.view_source_list', list_id=source_list.id))
        except Exception as e:
            db.session.rollback()
            flash(f'Error importing sources: {str(e)}', 'danger')
    
    return render_template('sources/import.html', predefined_sources=PREDEFINED_SOURCES)