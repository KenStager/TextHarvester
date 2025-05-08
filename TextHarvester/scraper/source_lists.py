"""
Predefined source lists for common scraping tasks across various domains.
These can be imported and used in the web interface or directly via the API.
Add new source lists or modify existing ones as needed for your specific domain.
"""

TECH_NEWS_SOURCES = [
    "https://techcrunch.com",
    "https://www.theverge.com",
    "https://arstechnica.com",
    "https://www.wired.com",
    "https://venturebeat.com",
    "https://www.cnet.com",
    "https://www.engadget.com",
    "https://www.zdnet.com"
]

RESEARCH_PUBLICATION_SOURCES = [
    "https://arxiv.org",
    "https://www.sciencedirect.com",
    "https://www.nature.com",
    "https://www.science.org",
    "https://scholar.google.com",
    "https://www.researchgate.net",
    "https://pubmed.ncbi.nlm.nih.gov",
    "https://www.frontiersin.org"
]

INDUSTRY_BLOG_SOURCES = [
    "https://hbr.org",
    "https://www.mckinsey.com/insights",
    "https://www.forrester.com/blogs",
    "https://www.gartner.com/en/publications",
    "https://sloanreview.mit.edu",
    "https://www.forbes.com",
    "https://www.bloomberg.com",
    "https://www.businessinsider.com"
]

AI_RESEARCH_SOURCES = [
    "https://arxiv.org/list/cs.AI/recent",
    "https://paperswithcode.com",
    "https://openai.com/research",
    "https://ai.googleblog.com",
    "https://www.microsoft.com/en-us/research/blog"
]

# Dictionary of all predefined source lists for easy access
PREDEFINED_SOURCES = {
    "tech_news": {
        "name": "Technology News Sites",
        "description": "Major technology news and information websites",
        "sources": TECH_NEWS_SOURCES
    },
    "research_publications": {
        "name": "Academic & Research Publications",
        "description": "Scientific journals and research publication platforms",
        "sources": RESEARCH_PUBLICATION_SOURCES
    },
    "industry_blogs": {
        "name": "Industry Analysis & Business Insights",
        "description": "Business analysis, market trends, and industry reports",
        "sources": INDUSTRY_BLOG_SOURCES
    },
    "ai_research": {
        "name": "AI Research & Development",
        "description": "Artificial intelligence research and development sources",
        "sources": AI_RESEARCH_SOURCES
    }
}

def get_source_list(list_id):
    """
    Get a predefined source list by ID
    
    Args:
        list_id (str): ID of the predefined list
        
    Returns:
        dict: The source list details or None if not found
    """
    return PREDEFINED_SOURCES.get(list_id)

def get_all_source_lists():
    """
    Get all available predefined source lists
    
    Returns:
        dict: Dictionary of all predefined source lists
    """
    return PREDEFINED_SOURCES

def combine_source_lists(list_ids):
    """
    Combine multiple predefined source lists into one list
    
    Args:
        list_ids (list): List of predefined source list IDs
        
    Returns:
        list: Combined list of sources
    """
    combined_sources = []
    for list_id in list_ids:
        source_list = get_source_list(list_id)
        if source_list:
            combined_sources.extend(source_list["sources"])
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(combined_sources))