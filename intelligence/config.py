"""
Configuration Settings for Content Intelligence Platform
=======================================================

This module contains configuration settings for all intelligence components,
including model paths, processing thresholds, and domain-specific settings.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Base directory paths
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
EXPORTS_DIR = ROOT_DIR / "exports"

# Ensure directories exist
for directory in [MODELS_DIR, DATA_DIR, LOGS_DIR, EXPORTS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Environment settings
ENV = os.getenv("TEXTHARVESTER_ENV", "development")  # development, test, production

# Database settings
DB_CONFIG = {
    "development": {
        "uri": os.getenv("DATABASE_URI", "postgresql://localhost/textharvester_dev")
    },
    "test": {
        "uri": os.getenv("TEST_DATABASE_URI", "postgresql://localhost/textharvester_test")
    },
    "production": {
        "uri": os.getenv("DATABASE_URI", "postgresql://localhost/textharvester")
    }
}

# Redis cache settings (for model caching and job queue)
REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", 6379)),
    "db": int(os.getenv("REDIS_DB", 0))
}

# Processing limits and thresholds
PROCESSING_CONFIG = {
    # Classification settings
    "classification": {
        "min_confidence_threshold": 0.5,  # Minimum confidence for topic classification
        "max_topics_per_content": 5,      # Maximum number of topics to assign per content
        "primary_topic_boost": 1.2        # Boost factor for primary topics
    },
    
    # Entity recognition settings
    "entity_recognition": {
        "min_confidence_threshold": 0.6,   # Minimum confidence for entity detection
        "max_entities_per_content": 100,   # Maximum entities to extract per content
        "context_window_chars": 100,       # Character window for entity context
        "linking_threshold": 0.7           # Minimum score for entity linking
    },
    
    # Temporal analysis settings
    "temporal_analysis": {
        "recency_half_life_days": 30,      # Half-life for content recency decay
        "future_event_boost": 1.3,         # Boost for content mentioning future events
        "min_date_confidence": 0.7         # Minimum confidence for date extraction
    },
    
    # Content enrichment settings
    "enrichment": {
        "max_context_blocks": 5,           # Maximum context blocks to add
        "max_related_content": 10,         # Maximum related content items to link
        "min_relatedness_score": 0.6       # Minimum score for content relationship
    },
    
    # General processing settings
    "general": {
        "max_content_length": 100000,      # Maximum content length to process
        "min_content_length": 100,         # Minimum content length to process
        "parallel_workers": 4,             # Number of parallel workers
        "batch_size": 50                   # Batch size for processing
    }
}

# ML Model settings
MODEL_CONFIG = {
    # Classification models
    "classification": {
        "fast_filter": {
            "type": "keybert",
            "path": str(MODELS_DIR / "classification" / "fast_filter"),
            "batch_size": 32
        },
        "primary_classifier": {
            "type": "distilbert",
            "path": str(MODELS_DIR / "classification" / "primary"),
            "batch_size": 16
        },
        "specialized_classifiers": {
            "base_path": str(MODELS_DIR / "classification" / "specialized"),
            "batch_size": 16
        }
    },
    
    # NER models
    "ner": {
        "spacy_model": "en_core_web_trf",
        "custom_models_path": str(MODELS_DIR / "ner"),
        "batch_size": 10
    },
    
    # Entity linking models
    "entity_linking": {
        "model_type": "bert",
        "model_path": str(MODELS_DIR / "entity_linking"),
        "batch_size": 16
    },
    
    # Temporal analysis models
    "temporal": {
        "date_extraction_model": str(MODELS_DIR / "temporal" / "date_extraction"),
        "relative_date_model": str(MODELS_DIR / "temporal" / "relative_dates"),
        "batch_size": 20
    },
    
    # Content quality models
    "quality": {
        "readability_model": str(MODELS_DIR / "quality" / "readability"),
        "sentiment_model": str(MODELS_DIR / "quality" / "sentiment"),
        "objectivity_model": str(MODELS_DIR / "quality" / "objectivity"),
        "batch_size": 20
    }
}

# Prodigy annotation settings
PRODIGY_CONFIG = {
    "host": os.getenv("PRODIGY_HOST", "localhost"),
    "port": int(os.getenv("PRODIGY_PORT", 8080)),
    "base_dataset": "textharvester",
    "label_colors": {
        "TEAM": "#85C1E9",
        "PERSON": "#F5B7B1",
        "COMPETITION": "#AED6F1",
        "VENUE": "#D7BDE2",
        "EVENT": "#F9E79F",
        "STATISTIC": "#A3E4D7"
    }
}

# Domain-specific settings
DOMAIN_CONFIGS = {}

# Football domain settings
FOOTBALL_CONFIG = {
    "league_ids": {
        "premier_league": 39,
        "la_liga": 140,
        "bundesliga": 78,
        "serie_a": 135,
        "ligue_1": 61
    },
    "season": "2024/2025",
    "transfer_windows": {
        "summer": {
            "start": "2024-06-10",
            "end": "2024-08-31"
        },
        "winter": {
            "start": "2024-01-01",
            "end": "2024-01-31"
        }
    },
    "competitions": {
        "premier_league": {
            "name": "Premier League",
            "country": "England",
            "teams": 20,
            "importance": 1.0
        },
        "fa_cup": {
            "name": "FA Cup",
            "country": "England",
            "importance": 0.8
        },
        "efl_cup": {
            "name": "EFL Cup",
            "country": "England",
            "importance": 0.7
        },
        "champions_league": {
            "name": "UEFA Champions League",
            "country": "Europe",
            "importance": 0.9
        }
    },
    "player_positions": [
        "Goalkeeper", "Right Back", "Center Back", "Left Back", 
        "Defensive Midfielder", "Central Midfielder", "Attacking Midfielder",
        "Right Winger", "Left Winger", "Forward", "Striker"
    ],
    "team_importance": {
        "Manchester City": 1.0,
        "Liverpool": 1.0,
        "Arsenal": 0.9,
        "Manchester United": 0.9,
        "Chelsea": 0.9,
        "Tottenham Hotspur": 0.8
        # More teams would be added here
    },
    # Entity salience adjustments for domain
    "entity_salience": {
        "TEAM": 1.0,
        "PERSON.PLAYER": 0.9,
        "PERSON.MANAGER": 0.8,
        "COMPETITION": 0.7,
        "VENUE": 0.6,
        "EVENT.MATCH": 1.0,
        "EVENT.TRANSFER": 0.9,
        "STATISTIC": 0.7
    },
    # Content type half-life (in days)
    "content_half_life": {
        "match_report": 3,
        "transfer_news": 14,
        "injury_update": 7,
        "season_preview": 90,
        "historical": 1825  # 5 years
    }
}

# Register domain configs
DOMAIN_CONFIGS["football"] = FOOTBALL_CONFIG

class Config:
    """Configuration manager for the Content Intelligence Platform."""
    
    @staticmethod
    def get_db_uri() -> str:
        """Get database URI for the current environment."""
        return DB_CONFIG[ENV]["uri"]
    
    @staticmethod
    def get_processing_config(component: str = None) -> Dict:
        """
        Get processing configuration.
        
        Args:
            component (str, optional): Specific component configuration to get.
        
        Returns:
            Dict: The requested configuration.
        """
        if component and component in PROCESSING_CONFIG:
            return PROCESSING_CONFIG[component]
        return PROCESSING_CONFIG
    
    @staticmethod
    def get_model_config(model_type: str = None) -> Dict:
        """
        Get model configuration.
        
        Args:
            model_type (str, optional): Specific model type configuration to get.
        
        Returns:
            Dict: The requested model configuration.
        """
        if model_type and model_type in MODEL_CONFIG:
            return MODEL_CONFIG[model_type]
        return MODEL_CONFIG
    
    @staticmethod
    def get_domain_config(domain: str) -> Dict:
        """
        Get domain-specific configuration.
        
        Args:
            domain (str): Domain to get configuration for.
        
        Returns:
            Dict: The domain configuration or empty dict if not found.
        """
        return DOMAIN_CONFIGS.get(domain, {})
    
    @staticmethod
    def get_football_config() -> Dict:
        """Get football domain configuration."""
        return DOMAIN_CONFIGS["football"]
    
    @staticmethod
    def get_prodigy_config() -> Dict:
        """Get Prodigy configuration."""
        return PRODIGY_CONFIG
    
    @staticmethod
    def get_path(path_type: str) -> Path:
        """
        Get a path from the configuration.
        
        Args:
            path_type (str): Type of path to get (root, models, data, logs, exports).
        
        Returns:
            Path: The requested path.
        """
        paths = {
            "root": ROOT_DIR,
            "models": MODELS_DIR,
            "data": DATA_DIR,
            "logs": LOGS_DIR,
            "exports": EXPORTS_DIR
        }
        return paths.get(path_type.lower(), ROOT_DIR)
    
    @staticmethod
    def save_domain_config(domain: str, config: Dict) -> None:
        """
        Save updated domain configuration.
        
        Args:
            domain (str): Domain name.
            config (Dict): Updated configuration.
        """
        DOMAIN_CONFIGS[domain] = config
        
        # Save to JSON file for persistence
        domain_config_path = DATA_DIR / "configs" / f"{domain}_config.json"
        domain_config_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(domain_config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    @staticmethod
    def load_domain_config(domain: str) -> Dict:
        """
        Load domain configuration from file.
        
        Args:
            domain (str): Domain name.
        
        Returns:
            Dict: The loaded configuration or existing one if file not found.
        """
        domain_config_path = DATA_DIR / "configs" / f"{domain}_config.json"
        
        if domain_config_path.exists():
            with open(domain_config_path, 'r') as f:
                config = json.load(f)
                DOMAIN_CONFIGS[domain] = config
                return config
        
        return DOMAIN_CONFIGS.get(domain, {})
