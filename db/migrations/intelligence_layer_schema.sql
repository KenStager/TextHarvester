-- intelligence_layer_schema.sql
-- Database schema for the Content Intelligence Platform
-- Version: 1.0.0
-- Date: 2025-05-06

-- Schema versioning
CREATE TABLE IF NOT EXISTS schema_versions (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) NOT NULL,
    description TEXT,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO schema_versions (version, description) 
VALUES ('1.0.0', 'Initial intelligence layer schema');

-- ===============================
-- Topic Taxonomy System Tables
-- ===============================

-- Table for topic taxonomy nodes
CREATE TABLE IF NOT EXISTS topic_taxonomy (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES topic_taxonomy(id),
    description TEXT,
    keywords TEXT[],
    classifier_model_path VARCHAR(255),
    domain VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for content classification results
CREATE TABLE IF NOT EXISTS content_classification (
    id SERIAL PRIMARY KEY,
    content_id INTEGER REFERENCES scraper_content(id),
    topic_id INTEGER REFERENCES topic_taxonomy(id),
    confidence FLOAT NOT NULL,
    is_primary BOOLEAN DEFAULT FALSE,
    classification_method VARCHAR(50),
    human_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for topic classification
CREATE INDEX IF NOT EXISTS idx_content_classification_content_id ON content_classification(content_id);
CREATE INDEX IF NOT EXISTS idx_content_classification_topic_id ON content_classification(topic_id);
CREATE INDEX IF NOT EXISTS idx_content_classification_confidence ON content_classification(confidence);
CREATE INDEX IF NOT EXISTS idx_topic_taxonomy_parent_id ON topic_taxonomy(parent_id);
CREATE INDEX IF NOT EXISTS idx_topic_taxonomy_domain ON topic_taxonomy(domain);

-- ===============================
-- Entity Recognition System Tables
-- ===============================

-- Table for entity types
CREATE TABLE IF NOT EXISTS entity_types (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    parent_id INTEGER REFERENCES entity_types(id),
    domain VARCHAR(50) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for entities (canonical records)
CREATE TABLE IF NOT EXISTS entities (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    entity_type_id INTEGER REFERENCES entity_types(id),
    canonical_name VARCHAR(255),
    kb_id VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for entity mentions in content
CREATE TABLE IF NOT EXISTS entity_mentions (
    id SERIAL PRIMARY KEY,
    content_id INTEGER REFERENCES scraper_content(id),
    entity_id INTEGER REFERENCES entities(id),
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    mention_text TEXT NOT NULL,
    confidence FLOAT,
    context_before TEXT,
    context_after TEXT,
    human_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for entity relationships
CREATE TABLE IF NOT EXISTS entity_relationships (
    id SERIAL PRIMARY KEY,
    source_entity_id INTEGER REFERENCES entities(id),
    target_entity_id INTEGER REFERENCES entities(id),
    relationship_type VARCHAR(100) NOT NULL,
    confidence FLOAT,
    metadata JSONB,
    start_date DATE,
    end_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for entity tables
CREATE INDEX IF NOT EXISTS idx_entity_types_parent_id ON entity_types(parent_id);
CREATE INDEX IF NOT EXISTS idx_entity_types_domain ON entity_types(domain);
CREATE INDEX IF NOT EXISTS idx_entities_entity_type_id ON entities(entity_type_id);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_content_id ON entity_mentions(content_id);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity_id ON entity_mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_relationships_source ON entity_relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_relationships_target ON entity_relationships(target_entity_id);

-- ===============================
-- Temporal Analysis System Tables
-- ===============================

-- Table for temporal references in content
CREATE TABLE IF NOT EXISTS temporal_references (
    id SERIAL PRIMARY KEY,
    content_id INTEGER REFERENCES scraper_content(id),
    reference_date DATE,
    reference_type VARCHAR(50), -- PUBLICATION, MENTIONED, FUTURE_EVENT, etc.
    confidence FLOAT,
    extracted_text TEXT,
    context TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for domain-specific events
CREATE TABLE IF NOT EXISTS domain_events (
    id SERIAL PRIMARY KEY,
    domain VARCHAR(50) NOT NULL,
    event_name VARCHAR(255) NOT NULL,
    event_date DATE NOT NULL,
    event_type VARCHAR(50),
    description TEXT,
    entities JSONB, -- Linked entities involved in this event
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for temporal relevance scores
CREATE TABLE IF NOT EXISTS temporal_relevance_scores (
    content_id INTEGER REFERENCES scraper_content(id),
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    relevance_score FLOAT NOT NULL,
    recency_factor FLOAT,
    domain_factor FLOAT,
    future_event_factor FLOAT,
    scoring_factors JSONB,
    PRIMARY KEY (content_id, calculated_at)
);

-- Create indexes for temporal tables
CREATE INDEX IF NOT EXISTS idx_temporal_references_content_id ON temporal_references(content_id);
CREATE INDEX IF NOT EXISTS idx_temporal_references_reference_date ON temporal_references(reference_date);
CREATE INDEX IF NOT EXISTS idx_domain_events_domain ON domain_events(domain);
CREATE INDEX IF NOT EXISTS idx_domain_events_event_date ON domain_events(event_date);
CREATE INDEX IF NOT EXISTS idx_domain_events_event_type ON domain_events(event_type);

-- ===============================
-- Content Enrichment System Tables
-- ===============================

-- Table for enhanced content
CREATE TABLE IF NOT EXISTS enhanced_content (
    id SERIAL PRIMARY KEY,
    content_id INTEGER REFERENCES scraper_content(id),
    enhanced_metadata JSONB NOT NULL,
    augmented_context JSONB,
    knowledge_links JSONB,
    processing_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for content quality metrics
CREATE TABLE IF NOT EXISTS content_quality_metrics (
    content_id INTEGER REFERENCES scraper_content(id),
    readability_score FLOAT,
    information_density FLOAT,
    sentiment_score FLOAT,
    objectivity_score FLOAT,
    factual_density FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (content_id)
);

-- Table for content relationships
CREATE TABLE IF NOT EXISTS content_relationships (
    id SERIAL PRIMARY KEY,
    source_content_id INTEGER REFERENCES scraper_content(id),
    target_content_id INTEGER REFERENCES scraper_content(id),
    relationship_type VARCHAR(50),
    similarity_score FLOAT,
    shared_entities JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for enrichment tables
CREATE INDEX IF NOT EXISTS idx_enhanced_content_content_id ON enhanced_content(content_id);
CREATE INDEX IF NOT EXISTS idx_content_relationships_source ON content_relationships(source_content_id);
CREATE INDEX IF NOT EXISTS idx_content_relationships_target ON content_relationships(target_content_id);

-- ===============================
-- Football-Specific Extensions
-- ===============================

-- Table for football teams
CREATE TABLE IF NOT EXISTS football_teams (
    id SERIAL PRIMARY KEY,
    entity_id INTEGER REFERENCES entities(id),
    league VARCHAR(100),
    country VARCHAR(100),
    city VARCHAR(100),
    stadium VARCHAR(100),
    founded_year INTEGER,
    team_colors TEXT,
    nickname TEXT,
    website VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for football players
CREATE TABLE IF NOT EXISTS football_players (
    id SERIAL PRIMARY KEY,
    entity_id INTEGER REFERENCES entities(id),
    current_team_id INTEGER REFERENCES football_teams(id),
    nationality VARCHAR(100),
    birth_date DATE,
    position VARCHAR(50),
    jersey_number INTEGER,
    height INTEGER, -- in cm
    weight INTEGER, -- in kg
    preferred_foot VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for player transfers
CREATE TABLE IF NOT EXISTS football_transfers (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES football_players(id),
    from_team_id INTEGER REFERENCES football_teams(id),
    to_team_id INTEGER REFERENCES football_teams(id),
    transfer_date DATE,
    fee DECIMAL(15, 2),
    fee_currency VARCHAR(10),
    contract_years INTEGER,
    transfer_type VARCHAR(50), -- PERMANENT, LOAN, FREE, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for football matches
CREATE TABLE IF NOT EXISTS football_matches (
    id SERIAL PRIMARY KEY,
    home_team_id INTEGER REFERENCES football_teams(id),
    away_team_id INTEGER REFERENCES football_teams(id),
    competition VARCHAR(100),
    match_date TIMESTAMP,
    venue VARCHAR(100),
    home_score INTEGER,
    away_score INTEGER,
    status VARCHAR(50), -- SCHEDULED, COMPLETED, POSTPONED, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for football tables
CREATE INDEX IF NOT EXISTS idx_football_teams_entity_id ON football_teams(entity_id);
CREATE INDEX IF NOT EXISTS idx_football_players_entity_id ON football_players(entity_id);
CREATE INDEX IF NOT EXISTS idx_football_players_current_team_id ON football_players(current_team_id);
CREATE INDEX IF NOT EXISTS idx_football_transfers_player_id ON football_transfers(player_id);
CREATE INDEX IF NOT EXISTS idx_football_matches_home_team_id ON football_matches(home_team_id);
CREATE INDEX IF NOT EXISTS idx_football_matches_away_team_id ON football_matches(away_team_id);
CREATE INDEX IF NOT EXISTS idx_football_matches_match_date ON football_matches(match_date);

-- Record schema version application
INSERT INTO schema_versions (version, description) 
VALUES ('1.0.0-football', 'Premier League football extensions');