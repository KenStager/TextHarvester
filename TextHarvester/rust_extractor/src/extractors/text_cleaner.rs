use regex::Regex;
use std::borrow::Cow;
use unicode_segmentation::UnicodeSegmentation;

pub struct TextCleaner {
    whitespace_regex: Regex,
    newline_regex: Regex,
    special_chars_regex: Regex,
}

impl Default for TextCleaner {
    fn default() -> Self {
        Self::new()
    }
}

impl TextCleaner {
    pub fn new() -> Self {
        TextCleaner {
            whitespace_regex: Regex::new(r"\s+").unwrap(),
            newline_regex: Regex::new(r"\n{3,}").unwrap(),
            special_chars_regex: Regex::new(r"[\u{0000}-\u{0008}\u{000B}\u{000C}\u{000E}-\u{001F}\u{007F}-\u{009F}\u{FEFF}]").unwrap(),
        }
    }

    /// Clean and normalize text content
    pub fn clean(&self, text: &str) -> String {
        let mut result = text.trim().to_string();
        
        // Replace problematic invisible characters
        result = self.special_chars_regex.replace_all(&result, "").to_string();
        
        // Normalize whitespace (replace multiple spaces with single space)
        result = self.whitespace_regex.replace_all(&result, " ").to_string();
        
        // Normalize newlines (replace 3+ consecutive newlines with double newline)
        result = self.newline_regex.replace_all(&result, "\n\n").to_string();
        
        // Final trim
        result.trim().to_string()
    }
    
    /// Count words in text using Unicode-aware word boundaries
    pub fn count_words(&self, text: &str) -> usize {
        text.unicode_words().count()
    }
    
    /// Count characters in text (excluding whitespace)
    pub fn count_chars(&self, text: &str) -> usize {
        text.chars().filter(|c| !c.is_whitespace()).count()
    }
    
    /// Count paragraphs (text separated by blank lines)
    pub fn count_paragraphs(&self, text: &str) -> usize {
        text.split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .count()
    }
    
    /// Detect language of text (simple implementation, to be enhanced)
    pub fn detect_language(&self, _text: &str) -> Option<String> {
        // Placeholder for language detection
        // In a real implementation, this would use a language detection library
        None
    }
    
    /// Sanitize text for safe use in different contexts
    pub fn sanitize_for_output<'a>(&self, text: &'a str) -> Cow<'a, str> {
        // Replace any problematic characters for output
        let sanitized = self.special_chars_regex.replace_all(text, "");
        
        // Convert other problematic Unicode characters to their ASCII equivalents or spaces
        sanitized
    }
}