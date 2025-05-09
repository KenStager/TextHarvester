#!/usr/bin/env python
"""
Direct database migration script using SQLite for adding intelligence tables.
"""
import os
import sys
import logging
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("direct_migration")

def run_migration():
    """Run direct SQLite migration to add intelligence tables for content analysis"""
    try:
        # Define database path
        db_path = os.path.join(os.path.dirname(__file__), 'data', 'web_scraper.db')
        logger.info(f"Using SQLite database at: {db_path}")
        
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='content_classification'")
        has_classification = cursor.fetchone() is not None
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='content_entity'")
        has_entity = cursor.fetchone() is not None
        
        # Add columns to scraping_configuration if needed
        cursor.execute("PRAGMA table_info(scraping_configuration)")
        existing_columns = [col[1] for col in cursor.fetchall()]
        
        columns_to_add = []
        if 'enable_classification' not in existing_columns:
            columns_to_add.append(('enable_classification', 'BOOLEAN DEFAULT 0'))
        if 'enable_entity_extraction' not in existing_columns:
            columns_to_add.append(('enable_entity_extraction', 'BOOLEAN DEFAULT 0'))
        if 'intelligence_domain' not in existing_columns:
            columns_to_add.append(('intelligence_domain', "VARCHAR(50) DEFAULT 'football'"))
        if 'intelligence_config' not in existing_columns:
            columns_to_add.append(('intelligence_config', 'TEXT'))
        
        # Add columns to scraping_configuration
        for column_name, column_def in columns_to_add:
            logger.info(f"Adding column to scraping_configuration: {column_name}")
            cursor.execute(f"ALTER TABLE scraping_configuration ADD COLUMN {column_name} {column_def}")
        
        # Create content_classification table if needed
        if not has_classification:
            logger.info("Creating content_classification table...")
            cursor.execute("""
            CREATE TABLE content_classification (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id INTEGER NOT NULL,
                is_relevant BOOLEAN DEFAULT 0,
                confidence REAL NOT NULL,
                primary_topic VARCHAR(255),
                primary_topic_id VARCHAR(100),
                primary_topic_confidence REAL,
                subtopics TEXT,
                processing_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (content_id) REFERENCES scraped_content (id) ON DELETE CASCADE
            )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX idx_classification_content_id ON content_classification(content_id)")
            cursor.execute("CREATE INDEX idx_classification_primary_topic ON content_classification(primary_topic)")
        
        # Create content_entity table if needed
        if not has_entity:
            logger.info("Creating content_entity table...")
            cursor.execute("""
            CREATE TABLE content_entity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id INTEGER NOT NULL,
                entity_type VARCHAR(100) NOT NULL,
                entity_text VARCHAR(1024) NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                confidence REAL NOT NULL,
                entity_id VARCHAR(255),
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (content_id) REFERENCES scraped_content (id) ON DELETE CASCADE
            )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX idx_entity_content_id ON content_entity(content_id)")
            cursor.execute("CREATE INDEX idx_entity_type ON content_entity(entity_type)")
            cursor.execute("CREATE INDEX idx_entity_id ON content_entity(entity_id)")
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        logger.info("Intelligence migration completed successfully!")
        return True
        
    except Exception as e:
        logger.exception(f"Error during migration: {e}")
        return False

if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)
