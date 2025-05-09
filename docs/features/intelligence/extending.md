# Extending TextHarvester Intelligence Features

This guide describes how to extend and customize the TextHarvester intelligence features for new domains or enhanced capabilities.

## Adding a New Domain

TextHarvester's intelligence features are designed to be domain-adaptive. Here's how to add support for a new domain:

### 1. Create Domain-Specific Pattern Files

Pattern files help the entity extraction system recognize domain-specific entities.

```bash
# Create a new pattern file for your domain
mkdir -p models/patterns
touch models/patterns/your_domain_patterns.jsonl
```

Add patterns in JSONL format:
```json
{"label":"ORGANIZATION","pattern":"Domain-Specific Organization"}
{"label":"CUSTOM_ENTITY","pattern":"Special Entity Name"}
```

### 2. Add Default Classification Results

Update the `create_default_for_domain` method in `intelligence/classification/classifiers.py` to include your domain:

```python
@classmethod
def create_default_for_domain(cls, domain: str = "general", confidence: float = 0.7) -> 'ClassificationResult':
    """Create a default classification result for a domain."""
    domain_defaults = {
        # Existing domains...
        
        "your_domain": {
            "id": "your_domain_main",
            "name": "Your Domain Main Topic",
            "children": [
                {"id": "subtopic1", "name": "Subtopic 1", "confidence": 0.6},
                {"id": "subtopic2", "name": "Subtopic 2", "confidence": 0.5}
            ]
        },
        
        # Always keep "general" as fallback...
    }
    
    # Rest of the method remains the same
}
```

### 3. Create a Domain Taxonomy (Optional)

For more advanced classification, create a domain-specific taxonomy:

```python
# Create file: intelligence/classification/taxonomies/your_domain.py

from intelligence.classification.topic_taxonomy import TopicTaxonomy, TopicNode

def get_your_domain_taxonomy() -> TopicTaxonomy:
    """Create and return your domain taxonomy."""
    
    # Create the root taxonomy
    taxonomy = TopicTaxonomy(
        name="your_domain_taxonomy",
        description="Taxonomy for your domain content"
    )
    
    # Create the main domain node
    main_node = TopicNode(
        name="Your Domain", 
        description="Your domain content",
        keywords=[
            "keyword1", "keyword2", "keyword3"
        ]
    )
    taxonomy.add_root_node(main_node)
    
    # Add subcategories
    subcategory1 = TopicNode(
        name="Subcategory 1",
        description="First subcategory of your domain",
        keywords=["sub1_keyword1", "sub1_keyword2"]
    )
    main_node.add_child(subcategory1)
    
    subcategory2 = TopicNode(
        name="Subcategory 2",
        description="Second subcategory of your domain",
        keywords=["sub2_keyword1", "sub2_keyword2"]
    )
    main_node.add_child(subcategory2)
    
    # Add deeper levels if needed
    subitem1 = TopicNode(
        name="Subitem 1",
        description="Item within subcategory 1",
        keywords=["subitem1_keyword1", "subitem1_keyword2"]
    )
    subcategory1.add_child(subitem1)
    
    return taxonomy
```

### 4. Add Domain-Specific Entity Types (Optional)

For custom entity types in your domain:

```python
# Create file: intelligence/entities/taxonomies/your_domain_entities.py

from typing import Dict, List
from intelligence.entities.entity_types import (
    EntityTypeRegistry, EntityType, EntityAttribute
)

def create_your_domain_entity_registry() -> EntityTypeRegistry:
    """Create entity type registry for your domain."""
    
    registry = EntityTypeRegistry(domain="your_domain")
    
    # Create custom entity type
    custom_type = EntityType(
        "CUSTOM_TYPE", 
        "A custom entity type for your domain", 
        domain="your_domain"
    )
    custom_type.add_attribute(EntityAttribute("name", "Name of the entity"))
    custom_type.add_attribute(EntityAttribute("identifier", "Unique identifier"))
    custom_type.add_attribute(EntityAttribute("category", "Category within your domain"))
    
    registry.add_root_type(custom_type)
    
    # Create subtypes if needed
    subtype1 = EntityType(
        "SUBTYPE1", 
        "First subtype of custom entity", 
        parent=custom_type
    )
    subtype1.add_attribute(EntityAttribute("special_property", "Property specific to this subtype"))
    
    return registry
```

### 5. Update Fast Filter Keywords

To improve classification performance, add domain-specific keywords to the fast filter:

```python
# In intelligence/classification/fast_filter.py

def create_domain_fast_filter() -> FastFilter:
    """Create a fast filter for your domain."""
    filter_instance = FastFilter(threshold=0.25, strategy="hybrid")
    
    # Add domain-specific keywords
    filter_instance.add_domain_keywords("your_domain", [
        "keyword1", "keyword2", "keyword3", "keyword4", 
        "keyword5", "keyword6", "keyword7"
    ])
    
    return filter_instance
```

### 6. Register in Configuration (Optional)

For more complete integration, update the configuration in `intelligence/config.py`:

```python
# Add domain configuration
YOUR_DOMAIN_CONFIG = {
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "importance": 1.0,
    "entity_salience": {
        "CUSTOM_TYPE": 1.0,
        "SUBTYPE1": 0.9
    },
    "content_half_life": {
        "main_category": 30,
        "specialty": 60
    }
}

# Register domain config
DOMAIN_CONFIGS["your_domain"] = YOUR_DOMAIN_CONFIG
```

## Enhancing Classification

### Improving Default Classification

To make default classification more nuanced:

1. **Update the domain detection logic**:

```python
def detect_domain(text: str) -> str:
    """Try to detect the appropriate domain from text content."""
    text_lower = text.lower()
    
    # Check for football indicators
    football_terms = ["goal", "team", "match", "player", "league", "football", "soccer"]
    football_count = sum(1 for term in football_terms if term in text_lower)
    
    # Check for your domain indicators
    your_domain_terms = ["term1", "term2", "term3", "term4", "term5"]
    your_domain_count = sum(1 for term in your_domain_terms if term in text_lower)
    
    # Return the most likely domain based on term counts
    if football_count > your_domain_count:
        return "football"
    elif your_domain_count > 0:
        return "your_domain"
    else:
        return "general"
```

2. **Add confidence calculation based on term density**:

```python
def calculate_confidence(text: str, domain: str) -> float:
    """Calculate classification confidence based on term density."""
    word_count = len(text.split())
    if word_count < 20:
        return 0.5  # Low confidence for very short texts
    
    text_lower = text.lower()
    
    domain_terms = {
        "football": ["goal", "team", "match", "player", "league", "football", "soccer"],
        "your_domain": ["term1", "term2", "term3", "term4", "term5"],
        "general": ["news", "report", "information", "article", "update"]
    }
    
    terms = domain_terms.get(domain, domain_terms["general"])
    matches = sum(1 for term in terms if term in text_lower)
    
    # Calculate term density
    density = matches / word_count
    
    # Convert to confidence value between 0.5 and 0.95
    confidence = min(0.95, 0.5 + density * 100)
    return confidence
```

### Creating Mock Classifiers

For testing without full ML models:

```python
class MockClassifier:
    """Mock classifier that uses keyword matching for testing."""
    
    def __init__(self, domain: str = "general"):
        self.domain = domain
        
        # Define keywords for each topic
        self.topic_keywords = {
            "news": ["report", "update", "published", "announced", "news"],
            "product": ["release", "version", "product", "feature", "launch"],
            "review": ["review", "rating", "recommend", "opinion", "tested"],
            "guide": ["how", "guide", "tutorial", "steps", "learn"],
            "analysis": ["analysis", "research", "study", "examined", "findings"]
        }
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict topics based on keyword presence."""
        text_lower = text.lower()
        
        # Count keyword matches for each topic
        topic_scores = {}
        for topic, keywords in self.topic_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                # Calculate score based on match count
                score = min(0.95, 0.5 + matches * 0.1)
                topic_scores[topic] = score
        
        # Default if no matches
        if not topic_scores:
            topic_scores["general"] = 0.7
            
        # Sort by score
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return result in expected format
        primary_topic = sorted_topics[0][0]
        primary_score = sorted_topics[0][1]
        
        return {
            "is_relevant": True,
            "confidence": primary_score,
            "primary_topic": primary_topic.capitalize(),
            "primary_topic_id": primary_topic,
            "primary_topic_confidence": primary_score
        }
```

## Enhancing Entity Extraction

### Adding Custom Entity Rules

Create more sophisticated entity matching rules:

```python
# In your domain pattern file

# Regex-based patterns
{"label": "CUSTOM_ID", "pattern": [{"REGEX": "[A-Z]{2}-\\d{4}-[A-Z]{2}"}]}

# Token-based patterns
{"label": "KEY_PHRASE", "pattern": [{"LOWER": "key"}, {"LOWER": "phrase"}]}

# Multi-pattern entities
{"label": "PROCESS", "pattern": [
  [{"LOWER": "first"}, {"LOWER": "step"}],
  [{"LOWER": "phase"}, {"IS_DIGIT": true}],
  [{"LOWER": "procedure"}, {"LOWER": "number"}, {"IS_DIGIT": true}]
]}
```

### Implementing Entity Linking

For more advanced entity recognition with linking:

```python
class CustomEntityLinker:
    """Custom entity linker for your domain."""
    
    def __init__(self):
        # Load entity knowledge base
        self.entities = self._load_entities()
        
    def _load_entities(self):
        """Load entity knowledge base from file or database."""
        # Example implementation
        entities = {
            "entity1": {
                "id": "ENT001",
                "name": "Entity One",
                "aliases": ["E1", "Entity 1", "First Entity"],
                "type": "CUSTOM_TYPE"
            },
            "entity2": {
                "id": "ENT002",
                "name": "Entity Two",
                "aliases": ["E2", "Entity 2", "Second Entity"],
                "type": "CUSTOM_TYPE"
            }
        }
        return entities
    
    def link_entity(self, text: str, entity_type: str) -> Optional[Dict]:
        """Try to link text to a known entity."""
        # Normalize text for comparison
        text_norm = text.lower().strip()
        
        # Check direct matches
        for entity_id, entity in self.entities.items():
            if entity['type'] != entity_type:
                continue
                
            if text_norm == entity['name'].lower() or text_norm in [a.lower() for a in entity['aliases']]:
                return {
                    "id": entity['id'],
                    "name": entity['name'],
                    "confidence": 1.0
                }
        
        # Check fuzzy matches if no direct match
        best_match = None
        best_score = 0.0
        
        for entity_id, entity in self.entities.items():
            if entity['type'] != entity_type:
                continue
                
            # Simple character overlap similarity (for demonstration)
            similarity = self._calculate_similarity(text_norm, entity['name'].lower())
            
            if similarity > 0.8 and similarity > best_score:
                best_match = entity
                best_score = similarity
        
        if best_match:
            return {
                "id": best_match['id'],
                "name": best_match['name'],
                "confidence": best_score
            }
            
        return None
        
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (simplified for demonstration)."""
        # Character bigram similarity
        bigrams1 = set(text1[i:i+2] for i in range(len(text1)-1))
        bigrams2 = set(text2[i:i+2] for i in range(len(text2)-1))
        
        # Jaccard similarity
        intersection = len(bigrams1.intersection(bigrams2))
        union = len(bigrams1.union(bigrams2))
        
        if union == 0:
            return 0.0
            
        return intersection / union
```

## Enhancing Integration

### Adding Smart Content Processing

For more intelligent integration with the scraper:

```python
def process_content_batch(contents: List[Any], processor: IntelligenceProcessor) -> Dict[int, Dict]:
    """Process a batch of content items with intelligent prioritization."""
    # Sort contents by potential value
    contents = sorted(contents, key=lambda c: _estimate_content_value(c))
    
    # Process in sorted order
    results = {}
    for content in contents:
        result = processor.process_content(content)
        results[content.id] = result
        
    return results

def _estimate_content_value(content: Any) -> float:
    """Estimate the potential value of content for prioritization."""
    # Simple heuristics for prioritization
    value = 0.0
    
    # Length-based value (longer content often has more value)
    if hasattr(content, 'word_count') and content.word_count:
        value += min(1.0, content.word_count / 1000)
    elif hasattr(content, 'extracted_text'):
        word_count = len(content.extracted_text.split())
        value += min(1.0, word_count / 1000)
    
    # Depth-based value (often deeper pages are more specific)
    if hasattr(content, 'crawl_depth'):
        # Value peaks at depth 2-3, then decreases
        depth_value = 1.0 - abs(content.crawl_depth - 2.5) / 5
        value += max(0, depth_value)
    
    # Title-based value
    if hasattr(content, 'title') and content.title:
        # Check for high-value indicators in title
        indicators = ["guide", "review", "analysis", "report", "exclusive"]
        for indicator in indicators:
            if indicator in content.title.lower():
                value += 0.5
                break
    
    return value
```

### Adding Memory-Efficient Processing

For handling large datasets efficiently:

```python
def stream_process_content(content_iterator, processor: IntelligenceProcessor,
                          batch_size: int = 10) -> Iterator[Tuple[Any, Dict]]:
    """
    Stream-process content to minimize memory usage.
    
    Args:
        content_iterator: Iterator yielding content objects
        processor: IntelligenceProcessor instance
        batch_size: Number of items to process in each batch
        
    Yields:
        Tuples of (content, result)
    """
    batch = []
    
    for content in content_iterator:
        batch.append(content)
        
        if len(batch) >= batch_size:
            # Process batch
            for item in batch:
                result = processor.process_content(item)
                yield (item, result)
                
            # Clear batch
            batch = []
    
    # Process any remaining items
    for item in batch:
        result = processor.process_content(item)
        yield (item, result)
```

## Best Practices for Extensions

1. **Maintain Backward Compatibility**: Ensure your extensions don't break existing functionality.

2. **Implement Graceful Fallbacks**: All components should work reasonably even when dependencies are missing.

3. **Add Unit Tests**: Create tests for your extensions:
   ```bash
   # Add test file for your domain
   touch tests/test_your_domain_intelligence.py
   ```

4. **Document Your Extensions**: Update relevant documentation to describe your additions.

5. **Use Progressive Enhancement**: Add complex features in layers, with each layer working without requiring the next.

6. **Performance Testing**: Check resource usage with your extensions using the test suite.
