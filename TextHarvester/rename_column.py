"""
Simple script to rename the metadata column to entity_metadata in the content_entity table.
"""

import os
import sqlite3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_migration():
    """Run the database migration to rename the metadata column in content_entity table."""
    try:
        # Connect directly to SQLite database
        db_path = os.path.join('TextHarvester', 'data', 'web_scraper.db')
        
        # Log the full path for debugging
        abs_path = os.path.abspath(db_path)
        logger.info(f"Trying to connect to database at: {abs_path}")
        
        if not os.path.exists(abs_path):
            # Try alternate path
            db_path = os.path.join('data', 'web_scraper.db')
            abs_path = os.path.abspath(db_path)
            logger.info(f"Alternate path: {abs_path}")
            
            if not os.path.exists(abs_path):
                logger.error(f"Database file not found at {abs_path}")
                return

        conn = sqlite3.connect(abs_path)
        cursor = conn.cursor()
        logger.info("Connected to SQLite database")
        
        # Check if table and column exist
        cursor.execute("PRAGMA table_info(content_entity)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        if 'content_entity' not in [table[0] for table in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]:
            logger.info("content_entity table doesn't exist, no migration needed")
            conn.close()
            return
            
        if 'metadata' in column_names and 'entity_metadata' not in column_names:
            # Use the table recreation approach which works in all SQLite versions
            logger.info("Renaming metadata column to entity_metadata")
            
            # 1. Create new table with desired schema
            cursor.execute("""
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
            """)
            
            # 2. Copy data from old table to new table
            cursor.execute("""
            INSERT INTO content_entity_new (
                id, content_id, entity_type, entity_text, start_char, end_char, 
                confidence, entity_id, entity_metadata, created_at
            )
            SELECT
                id, content_id, entity_type, entity_text, start_char, end_char,
                confidence, entity_id, metadata, created_at
            FROM content_entity
            """)
            
            # 3. Drop old table
            cursor.execute("DROP TABLE content_entity")
            
            # 4. Rename new table to old table name
            cursor.execute("ALTER TABLE content_entity_new RENAME TO content_entity")
            
            # 5. Recreate indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS ix_content_entity_content_id ON content_entity (content_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS ix_content_entity_entity_type ON content_entity (entity_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS ix_content_entity_entity_id ON content_entity (entity_id)")
            
            # Commit changes
            conn.commit()
            logger.info("Successfully renamed metadata column to entity_metadata")
        elif 'entity_metadata' in column_names:
            logger.info("Column is already named entity_metadata, no migration needed")
        elif 'metadata' not in column_names:
            logger.info("metadata column doesn't exist, no migration needed")
        
        # Close connection
        conn.close()
        logger.info("Migration completed successfully")
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_migration()
    print("Migration completed successfully!")
