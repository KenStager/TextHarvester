# TextHarvester Intelligence Visualization

This document describes the intelligence visualization features in the TextHarvester dashboard, including how to view and interpret classification and entity extraction results.

## Intelligence Overview Dashboard

The Intelligence Overview dashboard provides a centralized view of all intelligence processing results across jobs and content items.

### Accessing the Intelligence Dashboard

1. From the main TextHarvester dashboard, click on the "Intelligence" link in the navigation menu.
2. The overview page shows key statistics, component status, and recent intelligence results.

### Dashboard Components

- **Statistics Cards**: Shows counts of classified documents, extracted entities, and active domains.
- **Component Status**: Displays the current status of classification and entity extraction pipelines.
- **Topic Distribution Chart**: Visualizes the distribution of primary topics across all classified content.
- **Entity Type Chart**: Shows the frequency of different entity types across all processed content.
- **Recent Activity**: Lists recent classification and entity extraction activities.
- **Recent Results**: Tables of recent classification and entity extraction results with links to detailed views.

## Content Intelligence Views

### Classification Results

Each classified document has a dedicated classification view that shows:

1. **Primary Topic**: The main topic assigned to the content with confidence score.
2. **Relevance**: Whether the content is considered relevant to the intelligence domain.
3. **Subtopics**: Additional topics detected in the content with their confidence scores.

To access the classification results:
- From the content list, click "View" for a specific content item
- In the content view modal, click "View Classification"

### Entity Extraction Results

Each document with extracted entities has a dedicated entity view that shows:

1. **Entity List**: A table of all extracted entities with their types and confidence scores.
2. **Highlighted Text**: The original text with entities highlighted by type.
3. **Entity Type Sections**: Groupings of entities by type for easier analysis.

To access the entity extraction results:
- From the content list, click "View" for a specific content item
- In the content view modal, click "View Entities"

## Job Status Intelligence

The job status page includes intelligence-related statistics:

1. **Topic Distribution**: Chart showing the distribution of topics across the job's content.
2. **Entity Type Distribution**: Chart showing the types of entities extracted from the job's content.
3. **Quality Metrics**: Content quality scores correlated with intelligence results.

## Configuration Options

Intelligence features can be configured for each scraping job:

1. Navigate to the configuration page for a job
2. Click "Intelligence Configuration"
3. Enable/disable classification and entity extraction
4. Select the intelligence domain (e.g., football, general, finance, technology)
5. Configure advanced settings like raw data storage

## Interpreting Results

### Classification Confidence

- **High confidence (>80%)**: Strong indication that the content belongs to the assigned topic
- **Medium confidence (50-80%)**: Reasonable indication, but may contain content from multiple topics
- **Low confidence (<50%)**: Tentative classification, content may be ambiguous or outside known domains

### Entity Confidence

- **High confidence (>80%)**: Entity is likely correctly identified
- **Medium confidence (50-80%)**: Entity is probably correctly identified but may have ambiguities
- **Low confidence (<50%)**: Entity identification is tentative and should be verified

## Troubleshooting

If intelligence results are not appearing:

1. Check if intelligence components are available on the Intelligence Overview page
2. Verify that intelligence features are enabled in the job configuration
3. Check that the job has completed processing
4. Look for any error messages in the job logs