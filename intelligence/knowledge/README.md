# Knowledge Management System

## Overview

The Knowledge Management System (KMS) is a core component of the Content Intelligence Platform that organizes extracted information into a structured, queryable knowledge base. It connects entities, concepts, events, and claims with their relationships and sources, enabling sophisticated content intelligence applications.

## Key Features

- **Knowledge Graph Management**: Organize information into a graph structure of nodes and edges
- **Entity Extraction**: Extract and store entities from content with context and relationships
- **Contradiction Detection**: Identify and resolve contradictory information
- **Source Credibility Assessment**: Weight information based on source reliability
- **Temporal Analysis**: Manage time-sensitive knowledge and historical context
- **Knowledge Querying**: Find paths, patterns, and relationships in the knowledge graph

## Components

### Knowledge Graph (`graph.py`)

Manages operations on the knowledge graph, including creation, querying, and manipulation of nodes and relationships.

```python
from intelligence.knowledge.graph import KnowledgeGraph

# Initialize with optional domain filter
graph = KnowledgeGraph(domain="football")

# Create entities, concepts, events, and claims
player_node = graph.create_entity_node(
    name="Mohamed Salah",
    entity_id=123,  # Optional reference to an Entity in entity_models
    attributes={"position": "forward", "nationality": "Egyptian"},
    tags=["player", "premier_league"],
    confidence=0.95
)

# Create relationships between nodes
edge = graph.create_edge(
    source_id=player_node.id,
    target_id=team_node.id,
    relationship_type="plays_for",
    confidence=0.95
)

# Query the graph
neighbors = graph.get_node_neighbors(player_node.id)
```

### Knowledge Extraction (`extraction.py`)

Extracts structured knowledge from processed content and converts it into knowledge graph nodes and edges.

```python
from intelligence.knowledge.extraction import KnowledgeExtractor

# Initialize with optional domain filter
extractor = KnowledgeExtractor(domain="football")

# Extract knowledge from processed content
results = extractor.extract_from_content(content_id=123)
```

### Knowledge Storage (`storage.py`)

Manages efficient storage and retrieval of knowledge graph data, including batch operations and persistence.

```python
from intelligence.knowledge.storage import KnowledgeStorage

# Initialize with optional domain filter
storage = KnowledgeStorage(domain="football")

# Get storage statistics
stats = storage.get_storage_stats()

# Export the knowledge graph
export_data = storage.export_knowledge_graph(
    include_nodes=True,
    include_edges=True,
    node_types=["entity", "event"],
    min_confidence=0.7
)

# Export to a file
storage.export_to_json_file("football_knowledge.json")
```

### Contradiction Detection (`conflict.py`)

Identifies and manages contradictions in the knowledge graph, enabling conflict resolution and uncertainty handling.

```python
from intelligence.knowledge.conflict import ContradictionDetector

# Initialize with optional domain filter
detector = ContradictionDetector(domain="football")

# Detect contradictions in a new claim
contradictions = detector.detect_claim_contradictions(new_claim_id=456)

# Resolve a contradiction
detector.resolve_contradiction(
    contradiction_id=789,
    resolution_status="resolved_primary",
    resolution_notes="More recent information confirmed"
)
```

### Credibility Scoring (`credibility.py`)

Evaluates and manages the credibility of content sources to weight information appropriately.

```python
from intelligence.knowledge.credibility import CredibilityScorer

# Initialize with optional domain filter
scorer = CredibilityScorer(domain="football")

# Evaluate a source
credibility = scorer.evaluate_source("https://example.com/football-news")

# Recalculate node confidence based on source credibility
update_result = scorer.recalculate_confidence_with_credibility(node_id=123)
```

### Knowledge Querying (`queries.py`)

Provides utilities for querying and exploring the knowledge graph, including path finding, pattern matching, and aggregation.

```python
from intelligence.knowledge.queries import KnowledgeQuerier

# Initialize with optional domain filter
querier = KnowledgeQuerier(domain="football")

# Find paths between nodes
paths = querier.find_paths(
    start_node_id=123,
    end_node_id=456,
    max_length=3
)

# Get a subgraph around a node
subgraph = querier.get_subgraph(
    node_id=123,
    max_depth=2,
    max_nodes=50
)

# Query claims about an entity
claims = querier.query_claims_by_entity(entity_id=123)
```

### Knowledge Pipeline (`pipeline.py`)

Manages the complete knowledge processing workflow, coordinating extraction, storage, contradiction detection, and credibility assessment.

```python
from intelligence.knowledge.pipeline import KnowledgePipeline

# Initialize with optional domain filter
pipeline = KnowledgePipeline(domain="football")

# Process a single content item
result = pipeline.process_content(content_id=123)

# Process multiple content items in batch
batch_result = pipeline.batch_process(content_ids=[123, 456, 789])

# Process unprocessed content
unprocessed_result = pipeline.process_unprocessed_content(limit=100)
```

### Visualization (`visualization.py`)

Generates visualization data for knowledge graph components, supporting various visualization formats and libraries.

```python
from intelligence.knowledge.visualization import GraphVisualizationGenerator

# Initialize with optional domain filter
generator = GraphVisualizationGenerator(domain="football")

# Generate graph data
graph_data = generator.generate_graph_data(
    node_ids=[123, 456, 789],
    max_nodes=100
)

# Export to Vis.js format
vis_data = generator.export_to_vis_js(graph_data)
```

## API Integration

The Knowledge Management System exposes RESTful API endpoints for interacting with the knowledge graph, including:

- Node management (`/api/knowledge/nodes`)
- Edge management (`/api/knowledge/edges`)
- Querying (`/api/knowledge/query/*`)
- Processing (`/api/knowledge/process`)
- Visualizations (`/api/knowledge/visualize/*`)

See `api/routes/knowledge.py` for the complete API implementation.

## Database Models

The Knowledge Management System uses the following database models defined in `db/models/knowledge_base.py`:

- `KnowledgeNode`: Base class for all knowledge nodes
- `EntityNode`, `ConceptNode`, `EventNode`, `ClaimNode`: Specialized node types
- `KnowledgeEdge`: Relationships between nodes
- `KnowledgeSource`: Sources of information for nodes and edges
- `KnowledgeContradiction`: Contradictory information in the knowledge base
- `SourceCredibility`: Credibility scores for content sources

## Usage in the Premier League Football Domain

For the Premier League football domain, the Knowledge Management System can:

1. Extract and store information about teams, players, managers, matches, and tournaments
2. Track player transfers, team performance, and match results
3. Identify contradictions in transfer rumors and injury reports
4. Weight information based on source credibility (official team websites vs. rumor sites)
5. Enable sophisticated queries about player careers, team history, and match statistics
6. Generate visualizations of team relationships, player networks, and season timelines

## Testing

The Knowledge Management System includes comprehensive tests in the `tests/intelligence/knowledge` directory.
