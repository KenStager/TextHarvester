use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Response model for content extraction
#[derive(Debug, Serialize, Deserialize)]
pub struct ExtractionResponse {
    /// URL that was processed
    pub url: String,
    
    /// Title of the page
    pub title: Option<String>,
    
    /// Main extracted text content
    pub text: String,
    
    /// Publication date if detected
    #[serde(skip_serializing_if = "Option::is_none")]
    pub date: Option<String>,
    
    /// Author information if detected
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    
    /// Detected language of the content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    
    /// Statistics about the extracted content
    pub stats: ExtractionStats,
    
    /// Any additional metadata extracted
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
    
    /// Success of the extraction
    pub success: bool,
    
    /// Error message if extraction failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Statistics about the extracted content
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct ExtractionStats {
    /// Number of words in the extracted text
    pub word_count: usize,
    
    /// Number of characters in the extracted text
    pub char_count: usize,
    
    /// Number of paragraphs identified
    pub paragraph_count: usize,
    
    /// Number of images found (if included)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_count: Option<usize>,
    
    /// Number of links found (if included)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub link_count: Option<usize>,
    
    /// Number of tables found (if included)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub table_count: Option<usize>,
}