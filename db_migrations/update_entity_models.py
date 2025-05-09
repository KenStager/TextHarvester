"""
Migration script to fix SQLAlchemy reserved attribute name conflicts in entity models.

This script updates the entities and entity_relationships tables to rename
the 'metadata' column to 'entity_metadata' and 'relation_metadata' respectively,
which avoids conflicts with SQLAlchemy's reserved attribute names.
"""

import os
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("db_migration")

# Add parent directory to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

try:
    from TextHarvester.app import app, db
    logger.info("Successfully imported app and db")
except ImportError as e:
    logger.error(f"Error importing app: {e}")
    try:
        # Try alternative import path
        from app import app, db
        logger.info("Successfully imported app and db (alternative path)")
    except ImportError as e:
        logger.error(f"Error importing app (alternative path): {e}")
        sys.exit(1)

def migrate():
    """
    Perform migration to rename metadata columns.
    """
    logger.info("Starting migration to fix SQLAlchemy reserved attribute name conflicts...")
    
    with app.app_context():
        try:
            # Check if the entities table exists
            tables = db.engine.table_names()
            if 'entities' not in tables:
                logger.warning("Entities table not found, skipping migration")
                return
                
            # Check if the entity_metadata column already exists
            connection = db.engine.connect()
            columns_query = """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'entities' 
                AND table_schema = current_schema()
            """
            result = connection.execute(columns_query)
            columns = [row[0] for row in result]
            
            # Rename metadata column in entities table if it exists and entity_metadata doesn't
            if 'metadata' in columns and 'entity_metadata' not in columns:
                logger.info("Renaming 'metadata' to 'entity_metadata' in entities table")
                db.engine.execute("ALTER TABLE entities RENAME COLUMN metadata TO entity_metadata")
            elif 'entity_metadata' in columns:
                logger.info("entity_metadata column already exists in entities table")
            else:
                logger.warning("metadata column not found in entities table")
            
            # Check if the entity_relationships table exists
            if 'entity_relationships' not in tables:
                logger.warning("Entity_relationships table not found, skipping migration")
                return
                
            # Check if the relation_metadata column already exists
            columns_query = """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'entity_relationships' 
                AND table_schema = current_schema()
            """
            result = connection.execute(columns_query)
            columns = [row[0] for row in result]
            
            # Rename metadata column in entity_relationships table if it exists and relation_metadata doesn't
            if 'metadata' in columns and 'relation_metadata' not in columns:
                logger.info("Renaming 'metadata' to 'relation_metadata' in entity_relationships table")
                db.engine.execute("ALTER TABLE entity_relationships RENAME COLUMN metadata TO relation_metadata")
            elif 'relation_metadata' in columns:
                logger.info("relation_metadata column already exists in entity_relationships table")
            else:
                logger.warning("metadata column not found in entity_relationships table")
                
            logger.info("Migration completed successfully")
            
        except Exception as e:
            logger.error(f"Error during migration: {e}")
            raise

if __name__ == "__main__":
    try:
        migrate()
        logger.info("Migration script completed successfully")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)
