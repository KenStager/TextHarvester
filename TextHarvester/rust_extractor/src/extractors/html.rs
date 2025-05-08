use scraper::{Html, Selector};
use select::document::Document;
use select::predicate::{Class, Name, Predicate};
use std::collections::HashMap;
use url::Url;

pub struct HtmlExtractor {
    // Configuration options could be added here
}

impl Default for HtmlExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl HtmlExtractor {
    pub fn new() -> Self {
        HtmlExtractor {}
    }

    /// Extract title from HTML document
    pub fn extract_title(&self, html: &str) -> Option<String> {
        let document = Html::parse_document(html);
        
        // Try to get title from the title tag
        let title_selector = Selector::parse("title").ok()?;
        if let Some(title_element) = document.select(&title_selector).next() {
            let title = title_element.inner_html();
            if !title.trim().is_empty() {
                return Some(title.trim().to_string());
            }
        }
        
        // Fallback to h1 if title tag is empty
        let h1_selector = Selector::parse("h1").ok()?;
        if let Some(h1_element) = document.select(&h1_selector).next() {
            let h1 = h1_element.inner_html();
            if !h1.trim().is_empty() {
                return Some(h1.trim().to_string());
            }
        }
        
        None
    }
    
    /// Extract metadata from HTML document
    pub fn extract_metadata(&self, html: &str, url: &str) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        let document = Html::parse_document(html);
        
        // Extract common metadata
        self.extract_meta_tags(&document, &mut metadata);
        
        // Try to extract author information
        if let Some(author) = self.extract_author(&document) {
            metadata.insert("author".to_string(), author);
        }
        
        // Try to extract publication date
        if let Some(date) = self.extract_date(&document, url) {
            metadata.insert("date".to_string(), date);
        }
        
        // Extract canonical URL if available
        if let Some(canonical) = self.extract_canonical_url(&document) {
            metadata.insert("canonical_url".to_string(), canonical);
        }
        
        metadata
    }
    
    /// Extract content from HTML according to various heuristics
    pub fn extract_main_content(&self, html: &str) -> Option<String> {
        // Use select for more control over the extraction process
        let document = Document::from(html);
        
        // Try different strategies to find the main content
        let main_content = self.extract_by_main_element(&document)
            .or_else(|| self.extract_by_article_element(&document))
            .or_else(|| self.extract_by_content_id(&document))
            .or_else(|| self.extract_by_content_class(&document));
            
        main_content.map(|s| s.trim().to_string())
    }
    
    /// Attempt to extract the main element's text
    fn extract_by_main_element(&self, document: &Document) -> Option<String> {
        let main_elements: Vec<_> = document.find(Name("main")).collect();
        if !main_elements.is_empty() {
            let content = main_elements
                .iter()
                .map(|node| node.text())
                .collect::<Vec<_>>()
                .join("\n\n");
            
            if !content.trim().is_empty() {
                return Some(content);
            }
        }
        None
    }
    
    /// Attempt to extract article element text
    fn extract_by_article_element(&self, document: &Document) -> Option<String> {
        let article_elements: Vec<_> = document.find(Name("article")).collect();
        if !article_elements.is_empty() {
            let content = article_elements
                .iter()
                .map(|node| node.text())
                .collect::<Vec<_>>()
                .join("\n\n");
            
            if !content.trim().is_empty() {
                return Some(content);
            }
        }
        None
    }
    
    /// Attempt to extract by content ID
    fn extract_by_content_id(&self, document: &Document) -> Option<String> {
        // Common content ID patterns
        for id in &["content", "main-content", "article-content", "post-content"] {
            // Use direct matching instead of predicates
            for node in document.find(Name("*")) {
                if let Some(attr_val) = node.attr("id") {
                    if attr_val == *id {
                        let content = node.text();
                        if !content.trim().is_empty() {
                            return Some(content);
                        }
                    }
                }
            }
        }
        None
    }
    
    /// Attempt to extract by content class
    fn extract_by_content_class(&self, document: &Document) -> Option<String> {
        // Common content class patterns
        for class_name in &["content", "main-content", "article", "post", "entry"] {
            // Use direct matching instead of predicates
            for node in document.find(Name("*")) {
                if let Some(attr_val) = node.attr("class") {
                    // Check if the class attribute contains our target class
                    let classes: Vec<&str> = attr_val.split_whitespace().collect();
                    if classes.contains(class_name) {
                        let content = node.text();
                        if !content.trim().is_empty() {
                            return Some(content);
                        }
                    }
                }
            }
        }
        None
    }
    
    /// Extract meta tags from HTML document
    fn extract_meta_tags(&self, document: &Html, metadata: &mut HashMap<String, String>) {
        if let Ok(meta_selector) = Selector::parse("meta") {
            for meta in document.select(&meta_selector) {
                let name = meta.value().attr("name").or_else(|| meta.value().attr("property"));
                let content = meta.value().attr("content");
                
                if let (Some(name), Some(content)) = (name, content) {
                    if !content.trim().is_empty() {
                        metadata.insert(name.to_string(), content.trim().to_string());
                    }
                }
            }
        }
    }
    
    /// Extract author information from HTML
    fn extract_author(&self, document: &Html) -> Option<String> {
        // Try meta tags first
        if let Ok(author_selector) = Selector::parse("meta[name='author'], meta[property='og:author'], meta[property='article:author']") {
            if let Some(author_element) = document.select(&author_selector).next() {
                if let Some(content) = author_element.value().attr("content") {
                    if !content.trim().is_empty() {
                        return Some(content.trim().to_string());
                    }
                }
            }
        }
        
        // Try common author elements
        for selector_str in &[
            ".author", ".byline", ".meta-author", "[itemprop='author']",
            "[rel='author']", ".post-author", ".entry-author"
        ] {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(author_element) = document.select(&selector).next() {
                    let author = author_element.text().collect::<String>();
                    if !author.trim().is_empty() {
                        return Some(author.trim().to_string());
                    }
                }
            }
        }
        
        None
    }
    
    /// Extract publication date from HTML
    fn extract_date(&self, document: &Html, _url: &str) -> Option<String> {
        // Try meta tags first
        for selector_str in &[
            "meta[name='date']",
            "meta[property='article:published_time']",
            "meta[property='og:published_time']",
            "meta[name='DC.date']",
            "meta[name='pubdate']"
        ] {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(date_element) = document.select(&selector).next() {
                    if let Some(content) = date_element.value().attr("content") {
                        if !content.trim().is_empty() {
                            return Some(content.trim().to_string());
                        }
                    }
                }
            }
        }
        
        // Try common date elements
        for selector_str in &[
            "[itemprop='datePublished']",
            ".published", ".post-date", ".entry-date",
            ".meta-date", ".date", "time"
        ] {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(date_element) = document.select(&selector).next() {
                    // Check for datetime attribute in time elements
                    if let Some(datetime) = date_element.value().attr("datetime") {
                        if !datetime.trim().is_empty() {
                            return Some(datetime.trim().to_string());
                        }
                    }
                    
                    // Otherwise use text content
                    let date = date_element.text().collect::<String>();
                    if !date.trim().is_empty() {
                        return Some(date.trim().to_string());
                    }
                }
            }
        }
        
        None
    }
    
    /// Extract canonical URL if specified
    fn extract_canonical_url(&self, document: &Html) -> Option<String> {
        if let Ok(link_selector) = Selector::parse("link[rel='canonical']") {
            if let Some(link_element) = document.select(&link_selector).next() {
                if let Some(href) = link_element.value().attr("href") {
                    if !href.trim().is_empty() {
                        return Some(href.trim().to_string());
                    }
                }
            }
        }
        None
    }
}