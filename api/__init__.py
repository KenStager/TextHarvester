"""
API Package Initialization

This module initializes the API package and registers all route blueprints.
"""

from flask import Blueprint

# Initialize API blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Import and register route modules
from .routes import knowledge, sources, intelligence

# Register blueprints
api_bp.register_blueprint(knowledge.bp)
api_bp.register_blueprint(sources.bp)
api_bp.register_blueprint(intelligence.bp)

# Import API initialization function
def init_api(app):
    """
    Initialize the API with the Flask application.
    
    Args:
        app: Flask application instance
    """
    app.register_blueprint(api_bp)
