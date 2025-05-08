import random
from urllib.parse import urlparse

def get_random_user_agent():
    """
    Return a random user agent from a list of common user agents
    
    Returns:
        str: Random user agent string
    """
    user_agents = [
        # Chrome
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
        # Firefox
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
        # Safari
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
        # Edge
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59"
    ]
    return random.choice(user_agents)

def get_domain_from_url(url):
    """
    Extract the domain from a URL
    
    Args:
        url (str): URL to parse
        
    Returns:
        str: Domain name
    """
    parsed_url = urlparse(url)
    return parsed_url.netloc

def is_valid_url(url):
    """
    Check if a URL is valid
    
    Args:
        url (str): URL to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
    except:
        return False

def parse_urls_list(urls_text):
    """
    Parse a list of URLs from text input (one URL per line)
    
    Args:
        urls_text (str): Text containing URLs
        
    Returns:
        list: List of valid URLs
    """
    if not urls_text:
        return []
        
    lines = urls_text.strip().split('\n')
    valid_urls = []
    
    for line in lines:
        url = line.strip()
        if is_valid_url(url):
            valid_urls.append(url)
    
    return valid_urls
