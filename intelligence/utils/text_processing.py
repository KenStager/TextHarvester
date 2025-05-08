"""
Text Processing Utilities
========================

This module provides text normalization and preprocessing utilities for
the Content Intelligence Platform, including text cleaning, language detection,
tokenization, and domain-specific text preprocessing.
"""

import re
import unicodedata
import html
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from html.parser import HTMLParser
import string
import json

# Set up logging
logger = logging.getLogger(__name__)

# Try to import spaCy - will be used if available
try:
    import spacy
    from spacy.language import Language
    SPACY_AVAILABLE = True
    
    # Load English model for basic processing
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.warning("Default spaCy model not found. Some functions may not work.")
        nlp = None
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None
    logger.warning("spaCy not installed. Some text processing features will be unavailable.")


# HTML Cleaner
class MLStripper(HTMLParser):
    """Simple HTML tag stripper."""
    
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []
        
    def handle_data(self, d):
        self.fed.append(d)
        
    def get_data(self):
        return ''.join(self.fed)


def strip_html(text: str) -> str:
    """
    Remove HTML tags from text.
    
    Args:
        text (str): HTML text to clean.
        
    Returns:
        str: Plain text with HTML tags removed.
    """
    if not text:
        return ""
        
    s = MLStripper()
    s.feed(html.unescape(text))
    return s.get_data()


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    Args:
        text (str): Text to normalize.
        
    Returns:
        str: Text with normalized whitespace.
    """
    if not text:
        return ""
        
    # Replace multiple whitespace characters with a single space
    text = re.sub(r'\s+', ' ', text)
    # Trim leading/trailing whitespace
    return text.strip()


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters to their closest ASCII representation.
    
    Args:
        text (str): Text to normalize.
        
    Returns:
        str: Text with normalized Unicode.
    """
    if not text:
        return ""
        
    # Normalize to NFKD form and encode as ASCII, ignoring non-ASCII characters
    normalized = unicodedata.normalize('NFKD', text)
    return normalized


def remove_special_chars(text: str, keep_punct: bool = True) -> str:
    """
    Remove special characters from text.
    
    Args:
        text (str): Text to clean.
        keep_punct (bool, optional): Whether to keep punctuation.
        
    Returns:
        str: Cleaned text.
    """
    if not text:
        return ""
        
    if keep_punct:
        # Only remove non-printable characters
        return ''.join(c for c in text if c.isprintable())
    else:
        # Remove all non-alphanumeric characters except spaces
        return re.sub(r'[^\w\s]', '', text)


def clean_text(text: str, strip_html_tags: bool = True, normalize_spaces: bool = True,
               normalize_chars: bool = True, remove_specials: bool = True,
               lowercase: bool = False) -> str:
    """
    Perform comprehensive text cleaning.
    
    Args:
        text (str): Text to clean.
        strip_html_tags (bool, optional): Whether to remove HTML tags.
        normalize_spaces (bool, optional): Whether to normalize whitespace.
        normalize_chars (bool, optional): Whether to normalize Unicode characters.
        remove_specials (bool, optional): Whether to remove special characters.
        lowercase (bool, optional): Whether to convert text to lowercase.
        
    Returns:
        str: Cleaned text.
    """
    if not text:
        return ""
    
    # Apply cleaning steps in sequence
    if strip_html_tags:
        text = strip_html(text)
    
    if normalize_chars:
        text = normalize_unicode(text)
    
    if remove_specials:
        text = remove_special_chars(text)
    
    if normalize_spaces:
        text = normalize_whitespace(text)
    
    if lowercase:
        text = text.lower()
    
    return text


def detect_language(text: str) -> str:
    """
    Detect the language of a text.
    
    Args:
        text (str): Text to analyze.
        
    Returns:
        str: ISO 639-1 language code.
    """
    # Try to use spaCy's language detection if available
    if SPACY_AVAILABLE and nlp:
        # Process a small sample of the text
        sample = text[:1000]
        doc = nlp(sample)
        
        # Use document-level language prediction
        return doc.lang_
    
    # Fallback to a simple heuristic based on character frequency
    # This is a very basic approximation and not reliable
    text = text.lower()
    
    # Check for common English words
    english_common_words = ['the', 'and', 'that', 'have', 'for', 'not', 'with', 'you', 'this', 'but']
    english_count = sum(1 for word in english_common_words if f' {word} ' in f' {text} ')
    
    if english_count >= 3:
        return 'en'
    
    # Default to English if no clear detection
    return 'en'


def get_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text (str): Text to split.
        
    Returns:
        List[str]: List of sentences.
    """
    if not text:
        return []
    
    if SPACY_AVAILABLE and nlp:
        # Use spaCy for better sentence segmentation
        doc = nlp(text)
        return [sent.text for sent in doc.sents]
    else:
        # Fallback to simple regex-based splitting
        # This is not very accurate but works as a fallback
        pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
        return re.split(pattern, text)


def get_tokens(text: str, lowercase: bool = False, 
               remove_punct: bool = False, remove_stopwords: bool = False) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text (str): Text to tokenize.
        lowercase (bool, optional): Whether to lowercase tokens.
        remove_punct (bool, optional): Whether to remove punctuation.
        remove_stopwords (bool, optional): Whether to remove stopwords.
        
    Returns:
        List[str]: List of tokens.
    """
    if not text:
        return []
    
    if SPACY_AVAILABLE and nlp:
        # Use spaCy for better tokenization
        doc = nlp(text)
        
        tokens = []
        for token in doc:
            # Apply filters
            if remove_punct and token.is_punct:
                continue
            if remove_stopwords and token.is_stop:
                continue
                
            # Get the token text
            token_text = token.text
            if lowercase:
                token_text = token_text.lower()
                
            tokens.append(token_text)
            
        return tokens
    else:
        # Fallback to simple whitespace tokenization
        if lowercase:
            text = text.lower()
            
        if remove_punct:
            text = remove_special_chars(text, keep_punct=False)
            
        return text.split()


def get_text_stats(text: str) -> Dict[str, Any]:
    """
    Get basic statistics about a text.
    
    Args:
        text (str): Text to analyze.
        
    Returns:
        Dict[str, Any]: Dictionary of text statistics.
    """
    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "avg_word_length": 0,
            "avg_sentence_length": 0
        }
    
    # Get sentences and words
    sentences = get_sentences(text)
    words = get_tokens(text)
    
    # Calculate statistics
    char_count = len(text)
    word_count = len(words)
    sentence_count = len(sentences)
    
    avg_word_length = sum(len(word) for word in words) / max(1, word_count)
    avg_sentence_length = word_count / max(1, sentence_count)
    
    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_length,
        "avg_sentence_length": avg_sentence_length
    }


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100, 
               respect_sentences: bool = True) -> List[str]:
    """
    Split text into overlapping chunks of specified size.
    
    Args:
        text (str): Text to chunk.
        chunk_size (int, optional): Maximum size of each chunk.
        overlap (int, optional): Overlap between chunks.
        respect_sentences (bool, optional): Whether to avoid splitting sentences.
        
    Returns:
        List[str]: List of text chunks.
    """
    if not text:
        return []
    
    # Clean the text first
    text = clean_text(text)
    
    if respect_sentences:
        # Get sentences to use as building blocks
        sentences = get_sentences(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed the chunk size,
            # save the current chunk and start a new one
            if current_size + sentence_size > chunk_size and current_size > 0:
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_size = 0
                
                # Add sentences from the end of previous chunk for overlap
                for prev_sentence in reversed(current_chunk):
                    if overlap_size + len(prev_sentence) <= overlap:
                        overlap_sentences.insert(0, prev_sentence)
                        overlap_size += len(prev_sentence)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            # Add the current sentence to the chunk
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
    else:
        # Simple character-based chunking
        chunks = []
        
        for i in range(0, len(text), chunk_size - overlap):
            # Ensure we don't go beyond the end of the text
            end = min(i + chunk_size, len(text))
            chunks.append(text[i:end])
            
            # Stop if we've reached the end of the text
            if end == len(text):
                break
    
    return chunks


# Domain-specific text processing functions

def normalize_football_teams(text: str, team_mapping: Dict[str, str] = None) -> str:
    """
    Normalize football team names in text.
    
    Args:
        text (str): Text to process.
        team_mapping (Dict[str, str], optional): Mapping of team name variants to canonical names.
        
    Returns:
        str: Text with normalized team names.
    """
    if not text:
        return ""
    
    # Default mapping of common Premier League team name variants
    if team_mapping is None:
        team_mapping = {
            "Man Utd": "Manchester United",
            "Man United": "Manchester United",
            "Manchester Utd": "Manchester United",
            "MUFC": "Manchester United",
            "Man City": "Manchester City",
            "MCFC": "Manchester City",
            "Arsenal FC": "Arsenal",
            "The Gunners": "Arsenal",
            "Chelsea FC": "Chelsea",
            "The Blues": "Chelsea",
            "Liverpool FC": "Liverpool",
            "LFC": "Liverpool",
            "The Reds": "Liverpool",
            "Spurs": "Tottenham Hotspur",
            "Tottenham": "Tottenham Hotspur",
            # Add more mappings as needed
        }
    
    # Create a pattern to match all team variants
    pattern = r'\b(' + '|'.join(re.escape(key) for key in team_mapping.keys()) + r')\b'
    
    # Replace team name variants with canonical names
    return re.sub(pattern, lambda m: team_mapping[m.group(0)], text)


def extract_football_scores(text: str) -> List[Dict[str, Any]]:
    """
    Extract football match scores from text.
    
    Args:
        text (str): Text to process.
        
    Returns:
        List[Dict[str, Any]]: List of extracted scores.
    """
    if not text:
        return []
    
    # Pattern for score formats like "Team1 2-1 Team2" or "Team1 2 - 1 Team2"
    score_pattern = r'([A-Za-z ]+)\s+(\d+)\s*-\s*(\d+)\s+([A-Za-z ]+)'
    matches = re.finditer(score_pattern, text)
    
    scores = []
    for match in matches:
        home_team = match.group(1).strip()
        home_score = int(match.group(2))
        away_score = int(match.group(3))
        away_team = match.group(4).strip()
        
        scores.append({
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "text": match.group(0)
        })
    
    return scores


def normalize_player_names(text: str, player_mapping: Dict[str, str] = None) -> str:
    """
    Normalize player names in text.
    
    Args:
        text (str): Text to process.
        player_mapping (Dict[str, str], optional): Mapping of player name variants to canonical names.
        
    Returns:
        str: Text with normalized player names.
    """
    if not text or not player_mapping:
        return text
    
    # Create a pattern to match all player variants
    pattern = r'\b(' + '|'.join(re.escape(key) for key in player_mapping.keys()) + r')\b'
    
    # Replace player name variants with canonical names
    return re.sub(pattern, lambda m: player_mapping[m.group(0)], text)


def extract_temporal_expressions(text: str) -> List[Dict[str, Any]]:
    """
    Extract temporal expressions from text.
    
    Args:
        text (str): Text to process.
        
    Returns:
        List[Dict[str, Any]]: List of extracted temporal expressions.
    """
    if not text:
        return []
    
    # Patterns for various date formats
    patterns = [
        # ISO dates (2022-01-31)
        (r'\b(\d{4}-\d{2}-\d{2})\b', 'ISO_DATE'),
        
        # Common date formats (31/01/2022, 31-01-2022, etc.)
        (r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b', 'DATE'),
        
        # Written dates (January 31, 2022)
        (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}\b', 'WRITTEN_DATE'),
        
        # Relative dates (yesterday, today, tomorrow)
        (r'\b(yesterday|today|tomorrow)\b', 'RELATIVE_DAY'),
        
        # Day references (Monday, Tuesday, etc.)
        (r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b', 'DAY_OF_WEEK'),
        
        # Time expressions (3:00 PM, 15:00, etc.)
        (r'\b(\d{1,2}:\d{2}(?:\s*[AP]M)?)\b', 'TIME'),
        
        # Relative time (last week, next month, etc.)
        (r'\b(last|this|next)\s+(week|month|year|season)\b', 'RELATIVE_TIME')
    ]
    
    expressions = []
    
    for pattern, expr_type in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            expressions.append({
                "text": match.group(0),
                "type": expr_type,
                "start": match.start(),
                "end": match.end()
            })
    
    return expressions


# Function to create a text preprocessor for a specific domain
def create_domain_preprocessor(domain: str) -> callable:
    """
    Create a text preprocessor for a specific domain.
    
    Args:
        domain (str): Domain name.
        
    Returns:
        callable: Domain-specific text preprocessing function.
    """
    if domain == 'football':
        def football_preprocessor(text: str, team_mapping: Dict[str, str] = None,
                                 player_mapping: Dict[str, str] = None) -> str:
            """Preprocess text for football domain."""
            if not text:
                return ""
                
            # Apply general text cleaning
            text = clean_text(text)
            
            # Apply football-specific normalization
            text = normalize_football_teams(text, team_mapping)
            
            if player_mapping:
                text = normalize_player_names(text, player_mapping)
                
            return text
            
        return football_preprocessor
    else:
        # Default preprocessor just does general text cleaning
        return lambda text: clean_text(text)
