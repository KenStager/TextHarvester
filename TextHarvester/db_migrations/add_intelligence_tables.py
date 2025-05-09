"""
Database migration to add intelligence-related tables and columns.

This script adds the necessary tables and columns to support intelligence features
in the TextHarvester application. It should be run once to update the database schema.
"""

import os
import sys
import logging
from datetime import datetime

# Add parent directory to path so we can import app
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from app import app, db
from sqlalchemy import text, Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Float, JSON, MetaData, Table

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_migration():
    """Run the database migration to add intelligence-related tables and columns."""
    try:
        with app.app_context():
            logger.info("Starting intelligence database migration")
            
            # Use raw SQL for modifying existing tables to avoid model conflicts
            update_scraping_configuration()
            
            # Create new tables using models
            create_intelligence_tables()
            
            logger.info("Intelligence database migration completed successfully")
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        raise

def update_scraping_configuration():
    """Add intelligence columns to the scraping_configuration table."""
    logger.info("Adding intelligence columns to scraping_configuration table")
    
    try:
        # Check if columns already exist
        inspector = db.inspect(db.engine)
        existing_columns = [col['name'] for col in inspector.get_columns('scraping_configuration')]
        
        # Only add columns that don't already exist
        columns_to_add = []
        if 'enable_classification' not in existing_columns:
            columns_to_add.append('ADD COLUMN enable_classification BOOLEAN DEFAULT FALSE')
        if 'enable_entity_extraction' not in existing_columns:
            columns_to_add.append('ADD COLUMN enable_entity_extraction BOOLEAN DEFAULT FALSE')
        if 'intelligence_domain' not in existing_columns:
            columns_to_add.append("ADD COLUMN intelligence_domain VARCHAR(50) DEFAULT 'football'")
        if 'intelligence_config' not in existing_columns:
            columns_to_add.append('ADD COLUMN intelligence_config JSON')
        
        if columns_to_add:
            # Build and execute SQL statement
            alter_sql = f"ALTER TABLE scraping_configuration {', '.join(columns_to_add)}"
            db.session.execute(text(alter_sql))
            db.session.commit()
            logger.info("Added intelligence columns to scraping_configuration table")
        else:
            logger.info("Intelligence columns already exist in scraping_configuration table")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating scraping_configuration table: {str(e)}")
        raise

def create_intelligence_tables():
    """Create the intelligence-related tables if they don't already exist."""
    # Import models from models_update
    from models_update import ContentClassification, ContentEntity
    
    logger.info("Creating intelligence tables")
    
    try:
        # Check if tables already exist
        inspector = db.inspect(db.engine)
        existing_tables = inspector.get_table_names()
        
        # Create tables if they don't exist
        if 'content_classification' not in existing_tables:
            ContentClassification.__table__.create(db.engine)
            logger.info("Created content_classification table")
        else:
            logger.info("content_classification table already exists")
            
        if 'content_entity' not in existing_tables:
            ContentEntity.__table__.create(db.engine)
            logger.info("Created content_entity table")
        else:
            logger.info("content_entity table already exists")
            
    except Exception as e:
        logger.error(f"Error creating intelligence tables: {str(e)}")
        raise

if __name__ == "__main__":
    run_migration()
    print("Migration completed successfully!")
