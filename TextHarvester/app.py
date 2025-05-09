import os
import logging
from pathlib import Path

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Try to load .env file if it exists (for local development)
env_path = Path('.') / '.env'
if env_path.exists():
    logger.info("Loading environment from .env file")
    from dotenv import load_dotenv
    load_dotenv()

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)

# Set default secret key if not provided in environment
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")

# Add custom template filters
@app.template_filter('has_attr')
def has_attr_filter(obj, attr):
    """Jinja2 filter to check if an object has an attribute"""
    return hasattr(obj, attr)

@app.template_filter('attr_value')
def attr_value_filter(obj, attr, default=None):
    """Jinja2 filter to get attribute value with a default"""
    return getattr(obj, attr, default)

# Configure the database - fallback to SQLite for local development if no PostgreSQL URL is provided
database_url = os.environ.get("DATABASE_URL")
if not database_url:
    logger.warning("No DATABASE_URL found in environment, using SQLite for local development")
    # Create absolute path for data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    db_path = os.path.join(data_dir, 'web_scraper.db')
    logger.info(f"Using SQLite database at: {db_path}")
    # Use absolute path with 4 slashes for Windows compatibility
    database_url = f"sqlite:///{db_path}"

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize the app with the extension
db.init_app(app)

with app.app_context():
    # Import models here so their tables are created
    from models import ScrapedContent, ScrapingJob, ScrapingConfiguration, ContentMetadata, Source, SourceList

    logger.info("Creating database tables...")
    db.create_all()
    logger.info("Database tables created")

    # Register API routes
    from api.routes import api_bp
    app.register_blueprint(api_bp, url_prefix='')
    
    # Register sources routes
    from api.sources import sources_bp
    app.register_blueprint(sources_bp)

    logger.info("Application initialized")
