#!/usr/bin/env python
"""
Script to add additional indices to the intelligence tables
"""
import os
import sys
import logging
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("add_indices")

def add_indices():
    """Add additional indices to the intelligence tables"""
    try:
        # Define database path
        db_path = os.path.join(os.path.dirname(__file__), 'data', 'web_scraper.db')
        logger.info(f"Using SQLite database at: {db_path}")
        
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Add indices to content_classification
        logger.info("Adding additional indices to content_classification...")
        
        # Index for confidence
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_classification_confidence ON content_classification(confidence)")
        
        # Index for created_at (for chronological queries)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_classification_created_at ON content_classification(created_at)")
        
        # Add indices to content_entity
        logger.info("Adding additional indices to content_entity...")
        
        # Composite index for entity_type and entity_text
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_type_text ON content_entity(entity_type, entity_text)")
        
        # Index for confidence
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_confidence ON content_entity(confidence)")
        
        # Index for created_at
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_created_at ON content_entity(created_at)")
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        logger.info("Additional indices created successfully!")
        return True
        
    except Exception as e:
        logger.exception(f"Error adding indices: {e}")
        return False

if __name__ == "__main__":
    success = add_indices()
    sys.exit(0 if success else 1)
