#!/usr/bin/env python
"""
Database migration script to add intelligent navigation columns
"""
import os
import sys
import logging
from sqlalchemy import text

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("db_migration")

def run_migration():
    """Run database migration to add new columns for intelligent navigation"""
    try:
        # Import Flask app with database connection
        from app import app, db
        
        # Run migrations in app context
        with app.app_context():
            logger.info("Starting database migration for intelligent navigation...")
            
            # Check if we're using SQLite
            is_sqlite = 'sqlite' in db.engine.url.drivername
            
            # Check if the columns already exist
            inspector = db.inspect(db.engine)
            existing_columns = [col['name'] for col in inspector.get_columns('scraping_configuration')]
            
            # Add new columns if they don't exist
            columns_to_add = []
            if 'enable_intelligent_navigation' not in existing_columns:
                columns_to_add.append(('enable_intelligent_navigation', 'BOOLEAN DEFAULT 1'))
            if 'quality_threshold' not in existing_columns:
                columns_to_add.append(('quality_threshold', 'FLOAT DEFAULT 0.7'))
            if 'max_extended_depth' not in existing_columns:
                columns_to_add.append(('max_extended_depth', 'INTEGER DEFAULT 2'))
            
            if not columns_to_add:
                logger.info("All columns already exist. No migration needed.")
                return
            
            # Run the ALTER TABLE statements
            for column_name, column_def in columns_to_add:
                if is_sqlite:
                    # SQLite syntax
                    sql = f"ALTER TABLE scraping_configuration ADD COLUMN {column_name} {column_def}"
                else:
                    # PostgreSQL syntax
                    sql = f"ALTER TABLE scraping_configuration ADD COLUMN IF NOT EXISTS {column_name} {column_def}"
                
                logger.info(f"Adding column: {column_name}")
                db.session.execute(text(sql))
            
            # Create the content_quality_metrics table if it doesn't exist
            if not inspector.has_table("content_quality_metrics"):
                logger.info("Creating content_quality_metrics table...")
                
                if is_sqlite:
                    sql = """
                    CREATE TABLE content_quality_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content_id INTEGER NOT NULL,
                        quality_score FLOAT NOT NULL,
                        word_count INTEGER,
                        paragraph_count INTEGER,
                        text_ratio FLOAT,
                        domain VARCHAR(255),
                        domain_avg_score FLOAT,
                        parent_url VARCHAR(2048),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (content_id) REFERENCES scraped_content (id) ON DELETE CASCADE
                    )
                    """
                else:
                    sql = """
                    CREATE TABLE IF NOT EXISTS content_quality_metrics (
                        id SERIAL PRIMARY KEY,
                        content_id INTEGER NOT NULL REFERENCES scraped_content(id) ON DELETE CASCADE,
                        quality_score FLOAT NOT NULL,
                        word_count INTEGER,
                        paragraph_count INTEGER,
                        text_ratio FLOAT,
                        domain VARCHAR(255),
                        domain_avg_score FLOAT,
                        parent_url VARCHAR(2048),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                
                db.session.execute(text(sql))
                
                # Create index on content_id for better performance
                db.session.execute(text("CREATE INDEX idx_quality_metrics_content_id ON content_quality_metrics(content_id)"))
                
                # Create index on domain for queries that filter by domain
                db.session.execute(text("CREATE INDEX idx_quality_metrics_domain ON content_quality_metrics(domain)"))
            
            # Commit all changes
            db.session.commit()
            
            logger.info("Database migration completed successfully!")

    except Exception as e:
        logger.exception(f"Error during migration: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)
