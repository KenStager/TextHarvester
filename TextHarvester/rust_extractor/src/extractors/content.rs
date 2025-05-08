use crate::extractors::{HtmlExtractor, TextCleaner};
use crate::models::{ExtractionOptions, ExtractionResponse, ExtractionStats};
use anyhow::{Context, Result};
use reqwest::blocking::Client;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use url::Url;

/// Main content extractor that orchestrates the extraction process
pub struct ContentExtractor {
    html_extractor: HtmlExtractor,
    text_cleaner: TextCleaner,
    client: Client,
}

impl Default for ContentExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl ContentExtractor {
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("RustExtractor/0.1.0 (https://github.com/yourusername/rust-extractor)")
            .build()
            .expect("Failed to create HTTP client");
            
        ContentExtractor {
            html_extractor: HtmlExtractor::new(),
            text_cleaner: TextCleaner::new(),
            client,
        }
    }
    
    /// Extract content from a URL
    pub fn extract_from_url(&self, url: &str, options: &ExtractionOptions) -> Result<ExtractionResponse> {
        let start_time = Instant::now();
        
        // Fetch HTML content from URL
        let response = self.client.get(url)
            .send()
            .context("Failed to fetch URL")?;
            
        let html_content = response.text()
            .context("Failed to get response text")?;
            
        self.process_html(url, &html_content, options, start_time)
    }
    
    /// Extract content from provided HTML
    pub fn extract_from_html(&self, url: &str, html_content: &str, options: &ExtractionOptions) -> Result<ExtractionResponse> {
        let start_time = Instant::now();
        self.process_html(url, html_content, options, start_time)
    }
    
    /// Process HTML content and extract information
    fn process_html(&self, url: &str, html_content: &str, options: &ExtractionOptions, start_time: Instant) -> Result<ExtractionResponse> {
        // Extract title
        let title = self.html_extractor.extract_title(html_content);
        
        // Extract main content
        let raw_content = self.html_extractor.extract_main_content(html_content)
            .context("Failed to extract main content")?;
            
        // Clean the text if requested
        let text = if options.clean_text {
            self.text_cleaner.clean(&raw_content)
        } else {
            raw_content
        };
        
        // Extract metadata
        let metadata_map = self.html_extractor.extract_metadata(html_content, url);
        
        // Compute statistics
        let word_count = self.text_cleaner.count_words(&text);
        let char_count = self.text_cleaner.count_chars(&text);
        let paragraph_count = self.text_cleaner.count_paragraphs(&text);
        
        // Get author and date from metadata
        let author = metadata_map.get("author").cloned();
        let date = metadata_map.get("date").cloned()
            .or_else(|| metadata_map.get("article:published_time").cloned())
            .or_else(|| metadata_map.get("og:published_time").cloned());
            
        // Detect language (simple implementation)
        let language = self.text_cleaner.detect_language(&text);
        
        // Calculate processing time
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        // Convert metadata to output format
        let metadata: HashMap<String, String> = metadata_map.into_iter()
            .filter(|(k, _)| !matches!(k.as_str(), "author" | "date" | "article:published_time" | "og:published_time"))
            .collect();
            
        // Create stats object
        let stats = ExtractionStats {
            word_count,
            char_count,
            paragraph_count,
            image_count: None,  // Would require HTML parsing to count images
            link_count: None,   // Would require HTML parsing to count links
            table_count: None,  // Would require HTML parsing to count tables
        };
        
        // Create response
        let response = ExtractionResponse {
            url: url.to_string(),
            title,
            text,
            date,
            author,
            language,
            processing_time_ms: processing_time,
            stats,
            metadata,
            success: true,
            error: None,
        };
        
        Ok(response)
    }
    
    /// Create an error response
    pub fn create_error_response(&self, url: &str, error_message: &str) -> ExtractionResponse {
        ExtractionResponse {
            url: url.to_string(),
            title: None,
            text: String::new(),
            date: None,
            author: None,
            language: None,
            processing_time_ms: 0,
            stats: ExtractionStats::default(),
            metadata: HashMap::new(),
            success: false,
            error: Some(error_message.to_string()),
        }
    }
}