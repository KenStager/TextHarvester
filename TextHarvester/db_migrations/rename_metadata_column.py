"""
Database migration to rename the metadata column to entity_metadata in the content_entity table.

This script updates the content_entity table to resolve the naming conflict with SQLAlchemy's
reserved 'metadata' attribute name.
"""

import os
import sys
import logging
import sqlite3
from datetime import datetime

# Add parent directory to path so we can import app
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from app import app, db
from sqlalchemy import text, inspect

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_migration():
    """Run the database migration to rename the metadata column in content_entity table."""
    try:
        with app.app_context():
            logger.info("Starting metadata column rename migration")
            
            # Check if table and column exist
            inspector = inspect(db.engine)
            if 'content_entity' not in inspector.get_table_names():
                logger.info("content_entity table doesn't exist, no migration needed")
                return
                
            existing_columns = [col['name'] for col in inspector.get_columns('content_entity')]
            
            if 'metadata' in existing_columns and 'entity_metadata' not in existing_columns:
                # SQLite doesn't support ALTER TABLE RENAME COLUMN directly before version 3.25.0
                # We need to check which approach to use
                
                # Determine SQLite version
                sqlite_version = db.session.execute(text("SELECT sqlite_version()")).scalar()
                logger.info(f"SQLite version: {sqlite_version}")
                
                # Convert version string to tuple of integers for comparison
                version_parts = tuple(map(int, sqlite_version.split('.')))
                
                if version_parts >= (3, 25, 0):
                    # Use direct rename column if SQLite version is 3.25.0 or newer
                    logger.info("Using ALTER TABLE RENAME COLUMN approach")
                    sql = "ALTER TABLE content_entity RENAME COLUMN metadata TO entity_metadata"
                    db.session.execute(text(sql))
                else:
                    # Use the create new table approach for older SQLite versions
                    logger.info("Using table recreation approach for older SQLite")
                    
                    # 1. Create new table with desired schema
                    db.session.execute(text("""
                    CREATE TABLE content_entity_new (
                        id INTEGER PRIMARY KEY,
                        content_id INTEGER NOT NULL,
                        entity_type VARCHAR(100) NOT NULL,
                        entity_text VARCHAR(1024) NOT NULL,
                        start_char INTEGER NOT NULL,
                        end_char INTEGER NOT NULL,
                        confidence REAL NOT NULL,
                        entity_id VARCHAR(255),
                        entity_metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (content_id) REFERENCES scraped_content (id)
                    )
                    """))
                    
                    # 2. Copy data from old table to new table
                    db.session.execute(text("""
                    INSERT INTO content_entity_new (
                        id, content_id, entity_type, entity_text, start_char, end_char, 
                        confidence, entity_id, entity_metadata, created_at
                    )
                    SELECT
                        id, content_id, entity_type, entity_text, start_char, end_char,
                        confidence, entity_id, metadata, created_at
                    FROM content_entity
                    """))
                    
                    # 3. Drop old table
                    db.session.execute(text("DROP TABLE content_entity"))
                    
                    # 4. Rename new table to old table name
                    db.session.execute(text("ALTER TABLE content_entity_new RENAME TO content_entity"))
                    
                    # 5. Create indexes that existed on the original table
                    db.session.execute(text("CREATE INDEX ix_content_entity_content_id ON content_entity (content_id)"))
                    db.session.execute(text("CREATE INDEX ix_content_entity_entity_type ON content_entity (entity_type)"))
                    db.session.execute(text("CREATE INDEX ix_content_entity_entity_id ON content_entity (entity_id)"))
                
                # Commit transaction
                db.session.commit()
                logger.info("Successfully renamed metadata column to entity_metadata")
            elif 'entity_metadata' in existing_columns:
                logger.info("Column is already named entity_metadata, no migration needed")
            elif 'metadata' not in existing_columns:
                logger.info("metadata column doesn't exist, no migration needed")
            
            logger.info("Metadata column rename migration completed successfully")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Migration failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_migration()
    print("Migration completed successfully!")
