use serde::{Deserialize, Serialize};

/// Request model for content extraction
#[derive(Debug, Serialize, Deserialize)]
pub struct ExtractionRequest {
    /// URL of the page to extract content from
    pub url: String,
    
    /// Optional raw HTML content. If provided, the extractor won't fetch the URL
    #[serde(default)]
    pub html_content: Option<String>,
    
    /// Extraction options
    #[serde(default)]
    pub options: ExtractionOptions,
}

/// Options for content extraction
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct ExtractionOptions {
    /// Whether to include images in the extracted content
    #[serde(default)]
    pub include_images: bool,
    
    /// Whether to include links in the extracted content
    #[serde(default)]
    pub include_links: bool,
    
    /// Whether to include tables in the extracted content
    #[serde(default)]
    pub include_tables: bool,
    
    /// Whether to clean the extracted text (remove excessive whitespace, etc.)
    #[serde(default = "default_true")]
    pub clean_text: bool,
    
    /// Minimum date for content publication (for filtering old content)
    #[serde(default)]
    pub min_date: Option<String>,
}

fn default_true() -> bool {
    true
}