use chrono::Utc;
use log::{info, error, debug};
use serde::{Serialize, Deserialize};
use std::error::Error;
use tokio_postgres::{Config, NoTls, Row};
use futures::stream::TryStreamExt;

use crate::models::DatabaseConfig;
use crate::utils::text::clean_text;

/// Export options for job data
#[derive(Debug, Serialize, Deserialize)]
pub struct ExportOptions {
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    
    #[serde(default = "default_overlap")]
    pub overlap: usize,
}

fn default_chunk_size() -> usize {
    500
}

fn default_overlap() -> usize {
    50
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            chunk_size: default_chunk_size(),
            overlap: default_overlap(),
        }
    }
}

/// Job exporter for converting scraped content to JSONL
pub struct JobExporter {
    db_config: DatabaseConfig,
}

impl JobExporter {
    /// Create a new job exporter with the given database configuration
    pub fn new(db_config: DatabaseConfig) -> Self {
        Self {
            db_config,
        }
    }
    
    /// Validate that a job exists and return the content count
    pub fn validate_job(&self, job_id: i32) -> Result<i64, Box<dyn Error>> {
        // Parse database URL
        let config = self.db_config.url.parse::<Config>()?;
        
        // Connect to the database
        let rt = tokio::runtime::Runtime::new()?;
        let (client, connection) = rt.block_on(async {
            let (client, connection) = config.connect(NoTls).await?;
            Ok::<_, tokio_postgres::Error>((client, connection))
        })?;
        
        // Spawn the connection task in the background
        rt.spawn(async move {
            if let Err(e) = connection.await {
                error!("Database connection error: {}", e);
            }
        });
        
        // Check if job exists
        let row = rt.block_on(async {
            client.query_one(
                "SELECT COUNT(*) FROM scraped_content WHERE job_id = $1",
                &[&job_id],
            ).await
        })?;
        
        let content_count: i64 = row.get(0);
        
        if content_count == 0 {
            return Err("No content found for job".into());
        }
        
        Ok(content_count)
    }
    
    /// Export job data to JSONL format
    pub fn export_to_jsonl(
        &self,
        job_id: i32,
        options: &ExportOptions,
    ) -> Result<String, Box<dyn Error>> {
        // Parse database URL
        let config = self.db_config.url.parse::<Config>()?;
        
        // Connect to the database
        let rt = tokio::runtime::Runtime::new()?;
        let (client, connection) = rt.block_on(async {
            let (client, connection) = config.connect(NoTls).await?;
            Ok::<_, tokio_postgres::Error>((client, connection))
        })?;
        
        // Spawn the connection task in the background
        rt.spawn(async move {
            if let Err(e) = connection.await {
                error!("Database connection error: {}", e);
            }
        });
        
        // Get job information
        let job_row = rt.block_on(async {
            client.query_one(
                "SELECT status, urls_successful FROM scraping_job WHERE id = $1",
                &[&job_id],
            ).await
        })?;
        
        // Change column indices to match the query:
        // Using String::from_utf8_lossy to handle potential non-UTF8 data
        let _job_status: String = String::from_utf8_lossy(job_row.get::<_, &[u8]>(0)).to_string();
        let _urls_successful: i32 = job_row.get(1);
        
        // Get content for job
        let rows = rt.block_on(async {
            client.query(
                "SELECT sc.id, sc.url, sc.title, sc.extracted_text, sc.crawl_depth, sc.processing_time, sc.created_at,
                cm.word_count, cm.char_count, cm.language
                FROM scraped_content sc
                LEFT JOIN content_metadata cm ON sc.id = cm.content_id
                WHERE sc.job_id = $1
                ORDER BY sc.id ASC",
                &[&job_id],
            ).await
        })?;
        
        // Generate JSONL
        let mut jsonl = String::new();
        
        // Process rows into JSONL
        for row in rows {
            if let Ok(json_record) = self.process_row_to_json(&row, options) {
                jsonl.push_str(&json_record);
                jsonl.push('\n');
            }
        }
        
        Ok(jsonl)
    }
    
    /// Process a database row into a JSON record
    fn process_row_to_json(&self, row: &Row, options: &ExportOptions) -> Result<String, Box<dyn Error>> {
        let id: i32 = row.get(0);
        let url: String = row.get(1);
        let title: Option<String> = row.get(2);
        let text: String = row.get(3);
        let depth: i32 = row.get(4);
        let processing_time: Option<i32> = row.get(5);
        
        // Parse the timestamp string instead of directly getting DateTime
        let created_at_str: String = row.get(6);
        let created_at = chrono::DateTime::parse_from_rfc3339(&created_at_str)
            .unwrap_or_else(|_| Utc::now().into());
            
        let word_count: Option<i32> = row.get(7);
        let char_count: Option<i32> = row.get(8);
        let language: Option<String> = row.get(9);
        
        // Clean text
        let cleaned_text = clean_text(&text);
        
        // Create metadata
        let metadata = serde_json::json!({
            "id": id,
            "url": url,
            "title": title,
            "depth": depth,
            "processing_time_ms": processing_time,
            "created_at": created_at.to_rfc3339(),
            "word_count": word_count,
            "char_count": char_count,
            "language": language,
        });
        
        // Create record
        let record = serde_json::json!({
            "text": cleaned_text,
            "meta": metadata,
        });
        
        // Convert to JSON string
        let json_str = serde_json::to_string(&record)?;
        
        Ok(json_str)
    }
    
    /// Stream job content in chunks suitable for NER/SpanCat annotation
    pub async fn stream_chunks(
        &self,
        job_id: i32,
        options: &ExportOptions,
    ) -> Result<Box<dyn futures::Stream<Item = Result<String, Box<dyn Error + Send + Sync>>> + Unpin>, Box<dyn Error>> {
        // Implementation for streaming would go here
        // This is a placeholder until we implement proper streaming
        
        // Create an error with specific Box type
        Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, "Streaming not implemented yet")))
    }
}