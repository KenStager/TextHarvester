# TextHarvester Intelligent Navigation System

## Overview

The intelligent navigation system enhances TextHarvester's web scraping capabilities by making smarter decisions about which links to follow and how deep to crawl. This system improves content discovery efficiency and quality by:

1. **Tracking URL parent-child relationships** to provide context for navigation decisions
2. **Evaluating page quality** using metrics like content length, paragraph density, and text-to-HTML ratio
3. **Adaptively adjusting crawl depth** based on content quality rather than using uniform maximum depth
4. **Scoring links** based on URL structure, anchor text, surrounding context, and page position

## Key Components

### 1. URL Parent Tracking

The crawler maintains a `url_parents` dictionary that stores the parent URL for each discovered link. This provides context for making intelligent depth decisions and understanding content relationships.

### 2. Page Quality Assessment

When a page is fetched, the system evaluates its quality using the `evaluate_page_quality()` function, which:
- Analyzes word count, paragraph density, and content structure
- Produces a quality score (0.0-1.0) and detailed metrics
- Stores scores in `page_quality_scores` for future reference
- Updates domain quality averages in `domain_quality_scores`

### 3. Dynamic Depth Extension

The `should_extend_depth()` method determines whether to crawl beyond the standard maximum depth based on:
- Parent page quality (high-quality parents suggest valuable children)
- Domain average quality (consistently good domains get deeper crawling)
- Link promise (links with high intelligence scores get extended depth)

The absolute maximum depth is capped at `max_depth + 2` to prevent unbounded crawling.

### 4. HTML Content Caching

A limited cache of HTML content (`content_html_cache`) is maintained to:
- Provide context for link scoring and analysis
- Enable more accurate link quality predictions
- Support parent-child relationship evaluation

The cache has a size limit with an eviction policy to prevent memory issues.

## Using the Intelligent Navigation System

The intelligent navigation system works automatically with no additional configuration required. It enhances the standard web crawler behavior in these ways:

1. **More efficient crawling**: By prioritizing high-quality content sources
2. **Deeper exploration of valuable content**: Even beyond the configured maximum depth
3. **Less time wasted on low-value pages**: By limiting crawl depth for low-quality sources

## Example

Consider a website with this structure:

```
Homepage (depth 0)
└── Category Page (depth 1)
    └── High-Quality Article (depth 2)
        └── Related Article (depth 3, beyond standard max_depth=2)
            └── Reference Document (depth 4, beyond standard max_depth=2)
```

With a standard crawler (max_depth=2), only the Homepage, Category Page, and High-Quality Article would be crawled.

With intelligent navigation, if the High-Quality Article has a high quality score, the crawler will:
1. Extend depth to include the Related Article (depth 3)
2. If the Related Article also has high quality, possibly include the Reference Document (depth 4)

## Implementation Details

The intelligent navigation system is implemented with these key features:

1. **Thread safety**: All shared data structures use proper locking mechanisms
2. **Memory management**: HTML cache has size limits and eviction policies
3. **Error resilience**: Core crawler functionality continues even if intelligence features fail
4. **Configurable decisions**: Quality thresholds determine depth extension

## Testing

Unit tests for the intelligent navigation system are available in `tests/test_intelligent_navigation.py`.
