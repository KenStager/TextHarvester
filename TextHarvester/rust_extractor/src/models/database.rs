use serde::{Serialize, Deserialize};

/// Database configuration
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    /// Database URL (postgres://user:password@host:port/dbname)
    pub url: String,
}

/// Represents a scraping job from the database
#[derive(Debug, Serialize, Deserialize)]
pub struct ScrapingJob {
    pub id: i32,
    pub status: String,
    pub start_time: Option<chrono::DateTime<chrono::Utc>>,
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
    pub urls_processed: i32,
    pub urls_successful: i32,
    pub urls_failed: i32,
}

/// Represents scraped content from the database
#[derive(Debug, Serialize, Deserialize)]
pub struct ScrapedContent {
    pub id: i32,
    pub job_id: i32,
    pub url: String,
    pub title: Option<String>,
    pub extracted_text: String,
    pub crawl_depth: i32,
    pub processing_time: Option<i32>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Represents content metadata from the database
#[derive(Debug, Serialize, Deserialize)]
pub struct ContentMetadata {
    pub id: i32,
    pub content_id: i32,
    pub word_count: Option<i32>,
    pub char_count: Option<i32>,
    pub language: Option<String>,
    pub content_type: Option<String>,
}

/// Processing result for exported content
#[derive(Debug, Serialize, Deserialize)]
pub struct ExportRecord {
    pub text: String,
    pub meta: serde_json::Value,
}