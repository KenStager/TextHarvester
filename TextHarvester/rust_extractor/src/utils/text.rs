use unicode_normalization::UnicodeNormalization;
use regex::Regex;
use lazy_static::lazy_static;

/// Clean and normalize text for better quality
///
/// This function performs various text cleaning operations:
/// - Remove control characters
/// - Normalize Unicode (NFC)
/// - Replace multiple spaces with a single space
/// - Trim whitespace
///
/// Args:
///     text: The text to clean
///
/// Returns:
///     The cleaned text
pub fn clean_text(text: &str) -> String {
    lazy_static! {
        static ref CONTROL_CHAR_REGEX: Regex = Regex::new(r"[\p{Cc}\p{Cf}\p{Zl}\p{Zp}]").unwrap();
        static ref MULTI_SPACE_REGEX: Regex = Regex::new(r"\s+").unwrap();
    }
    
    // Process text
    let mut cleaned = text.to_string();
    
    // Remove null bytes and control characters
    cleaned = CONTROL_CHAR_REGEX.replace_all(&cleaned, "").to_string();
    
    // Normalize Unicode (NFC)
    cleaned = cleaned.nfc().collect::<String>();
    
    // Replace multiple spaces with a single space
    cleaned = MULTI_SPACE_REGEX.replace_all(&cleaned, " ").to_string();
    
    // Trim whitespace
    cleaned = cleaned.trim().to_string();
    
    cleaned
}

/// Split text into manageable chunks for annotation
///
/// This function splits text into chunks of a specified maximum size,
/// with optional overlap between chunks. It attempts to split at
/// sentence boundaries when possible.
///
/// Args:
///     text: The text to split
///     max_chunk_size: Maximum size of each chunk in words
///     overlap: Number of words to overlap between chunks
///
/// Returns:
///     A vector of text chunks
pub fn split_into_chunks(text: &str, max_chunk_size: usize, overlap: usize) -> Vec<String> {
    lazy_static! {
        static ref SENTENCE_END_REGEX: Regex = Regex::new(r"[.!?]\s+").unwrap();
        static ref WORD_REGEX: Regex = Regex::new(r"\s+").unwrap();
    }
    
    // If text is short enough, return it as a single chunk
    let words: Vec<&str> = WORD_REGEX.split(text).collect();
    if words.len() <= max_chunk_size {
        return vec![text.to_string()];
    }
    
    // Find sentence boundaries
    let mut sentence_boundaries = Vec::new();
    let mut last_index = 0;
    
    for cap in SENTENCE_END_REGEX.find_iter(text) {
        sentence_boundaries.push(cap.end());
        last_index = cap.end();
    }
    
    // Add the end of text if it doesn't end with a sentence boundary
    if last_index < text.len() {
        sentence_boundaries.push(text.len());
    }
    
    // Create chunks based on sentence boundaries and max_chunk_size
    let mut chunks = Vec::new();
    let mut start_idx = 0;
    let mut end_idx;
    let mut last_chunk_end = 0;
    
    while start_idx < text.len() {
        // Find the furthest sentence boundary within max_chunk_size
        end_idx = find_best_boundary(text, start_idx, max_chunk_size, &sentence_boundaries);
        
        // Extract chunk
        let chunk = text[start_idx..end_idx].trim().to_string();
        if !chunk.is_empty() {
            chunks.push(chunk);
        }
        
        // Move start index for next chunk (with overlap)
        if end_idx > start_idx + overlap {
            start_idx = if end_idx > overlap { end_idx - overlap } else { 0 };
        } else {
            start_idx = end_idx;
        }
        
        // Prevent infinite loops
        if end_idx <= last_chunk_end {
            start_idx = end_idx + 1;
        }
        last_chunk_end = end_idx;
        
        // Break if we've reached the end
        if end_idx >= text.len() {
            break;
        }
    }
    
    chunks
}

/// Find the best sentence boundary for chunking
///
/// Finds the furthest sentence boundary that's within the max_chunk_size limit.
fn find_best_boundary(
    text: &str,
    start_idx: usize,
    max_size: usize,
    boundaries: &[usize],
) -> usize {
    let max_text_len = (text.len() - start_idx).min(start_idx + max_size * 10); // Rough character estimation
    let target_idx = start_idx + max_text_len;
    
    // Find the closest boundary
    let mut best_boundary = start_idx + 1;
    
    for &boundary in boundaries {
        if boundary > start_idx && boundary <= target_idx {
            best_boundary = boundary;
        } else if boundary > target_idx {
            break;
        }
    }
    
    best_boundary.min(text.len())
}