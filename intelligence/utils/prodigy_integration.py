"""
Prodigy Integration Utilities
===========================

This module provides utilities for integrating with Prodigy, including stream generators,
functions to convert between Prodigy formats and internal models, and annotation result processing.
"""

import os
import json
import logging
import random
from typing import List, Dict, Tuple, Optional, Union, Any, Callable, Generator, Iterable
from pathlib import Path
import hashlib
from datetime import datetime

# Import configuration
from intelligence.config import Config

# Set up logging
logger = logging.getLogger(__name__)

# Try to import prodigy - will be used if available
try:
    import prodigy
    from prodigy.components.db import connect
    PRODIGY_AVAILABLE = True
except ImportError:
    PRODIGY_AVAILABLE = False
    logger.warning("Prodigy not installed. Integration features will be unavailable.")


# Helper functions for Prodigy tasks

def create_text_classification_task(text: str, options: List[str], 
                                   meta: Optional[Dict] = None) -> Dict:
    """
    Create a text classification task for Prodigy.
    
    Args:
        text (str): Text to classify.
        options (List[str]): Classification options.
        meta (Dict, optional): Metadata for the task.
        
    Returns:
        Dict: Prodigy task.
    """
    # Generate a stable ID based on content
    task_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    
    return {
        "text": text,
        "options": options,
        "meta": meta or {},
        "_task_hash": task_hash,
        "_input_hash": task_hash,
        "_view_id": "classification"
    }


def create_ner_task(text: str, spans: Optional[List[Dict]] = None, 
                   meta: Optional[Dict] = None) -> Dict:
    """
    Create a named entity recognition task for Prodigy.
    
    Args:
        text (str): Text for entity annotation.
        spans (List[Dict], optional): Pre-annotated spans.
        meta (Dict, optional): Metadata for the task.
        
    Returns:
        Dict: Prodigy task.
    """
    # Generate a stable ID based on content
    task_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    
    return {
        "text": text,
        "spans": spans or [],
        "meta": meta or {},
        "_task_hash": task_hash,
        "_input_hash": task_hash,
        "_view_id": "ner"
    }


def create_span_categorization_task(text: str, spans: Optional[List[Dict]] = None,
                                  meta: Optional[Dict] = None) -> Dict:
    """
    Create a span categorization task for Prodigy.
    
    Args:
        text (str): Text for span annotation.
        spans (List[Dict], optional): Pre-annotated spans.
        meta (Dict, optional): Metadata for the task.
        
    Returns:
        Dict: Prodigy task.
    """
    # Generate a stable ID based on content
    task_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    
    return {
        "text": text,
        "spans": spans or [],
        "meta": meta or {},
        "_task_hash": task_hash,
        "_input_hash": task_hash,
        "_view_id": "spans"
    }


def create_entity_linking_task(text: str, spans: List[Dict], kb_ids: List[str],
                              meta: Optional[Dict] = None) -> Dict:
    """
    Create an entity linking task for Prodigy.
    
    Args:
        text (str): Text containing entities to link.
        spans (List[Dict]): Entity spans to link.
        kb_ids (List[str]): Knowledge base IDs for linking.
        meta (Dict, optional): Metadata for the task.
        
    Returns:
        Dict: Prodigy task.
    """
    # Generate a stable ID based on content
    task_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    
    return {
        "text": text,
        "spans": spans,
        "kb_ids": kb_ids,
        "meta": meta or {},
        "_task_hash": task_hash,
        "_input_hash": task_hash,
        "_view_id": "entity_linker"
    }


def create_temporal_annotation_task(text: str, spans: Optional[List[Dict]] = None,
                                  meta: Optional[Dict] = None) -> Dict:
    """
    Create a temporal expression annotation task for Prodigy.
    
    Args:
        text (str): Text for temporal annotation.
        spans (List[Dict], optional): Pre-annotated temporal spans.
        meta (Dict, optional): Metadata for the task.
        
    Returns:
        Dict: Prodigy task.
    """
    # Generate a stable ID based on content
    task_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    
    return {
        "text": text,
        "spans": spans or [],
        "meta": meta or {},
        "_task_hash": task_hash,
        "_input_hash": task_hash,
        "_view_id": "spans"
    }


# Stream generators for Prodigy

def create_content_stream(content_items: List[Dict], task_type: str = "ner",
                         preprocessor: Optional[Callable] = None) -> Generator[Dict, None, None]:
    """
    Create a stream of annotation tasks from content items.
    
    Args:
        content_items (List[Dict]): Content items to annotate.
        task_type (str, optional): Type of annotation task.
        preprocessor (Callable, optional): Function to preprocess content.
        
    Yields:
        Dict: Annotation tasks.
    """
    for item in content_items:
        text = item.get("text", "")
        
        # Skip empty content
        if not text:
            continue
            
        # Apply preprocessor if provided
        if preprocessor:
            text = preprocessor(text)
        
        # Create metadata
        meta = {
            "content_id": item.get("id"),
            "source_url": item.get("source_url"),
            "title": item.get("title"),
            "date": item.get("date"),
            "domain": item.get("domain")
        }
        
        # Create task based on type
        if task_type == "classification":
            options = item.get("options", [])
            yield create_text_classification_task(text, options, meta)
            
        elif task_type == "ner":
            spans = item.get("spans", [])
            yield create_ner_task(text, spans, meta)
            
        elif task_type == "spans":
            spans = item.get("spans", [])
            yield create_span_categorization_task(text, spans, meta)
            
        elif task_type == "entity_linker":
            spans = item.get("spans", [])
            kb_ids = item.get("kb_ids", [])
            yield create_entity_linking_task(text, spans, kb_ids, meta)
            
        elif task_type == "temporal":
            spans = item.get("spans", [])
            yield create_temporal_annotation_task(text, spans, meta)
            
        else:
            logger.warning(f"Unknown task type: {task_type}")


def create_classification_stream_from_db(domain: str, count: int = 100, 
                                        confidence_below: float = 0.8) -> Generator[Dict, None, None]:
    """
    Create a stream of classification tasks from database content.
    
    Args:
        domain (str): Domain to get content for.
        count (int, optional): Maximum number of items.
        confidence_below (float, optional): Filter items with confidence below this value.
        
    Yields:
        Dict: Classification tasks.
    """
    try:
        from db.models.topic_taxonomy import ContentClassification
        from app import db, Content

        # Query for content items with low classification confidence
        query = db.session.query(Content, ContentClassification).join(
            ContentClassification,
            Content.id == ContentClassification.content_id
        ).filter(
            ContentClassification.confidence < confidence_below
        ).limit(count)
        
        for content, classification in query:
            # Get topic options
            from db.models.topic_taxonomy import TopicTaxonomy
            topics = db.session.query(TopicTaxonomy).filter(
                TopicTaxonomy.domain == domain
            ).all()
            
            options = [topic.name for topic in topics]
            
            # Create metadata
            meta = {
                "content_id": content.id,
                "source_url": content.source_url,
                "title": content.title,
                "date": content.created_at,
                "domain": domain,
                "current_classification": classification.topic_id,
                "current_confidence": classification.confidence
            }
            
            yield create_text_classification_task(content.text, options, meta)
            
    except ImportError:
        logger.error("Database models not available")
        return


def create_ner_stream_from_db(domain: str, entity_types: List[str], 
                             count: int = 100) -> Generator[Dict, None, None]:
    """
    Create a stream of NER tasks from database content.
    
    Args:
        domain (str): Domain to get content for.
        entity_types (List[str]): Entity types to annotate.
        count (int, optional): Maximum number of items.
        
    Yields:
        Dict: NER tasks.
    """
    try:
        from db.models.topic_taxonomy import ContentClassification
        from app import db, Content

        # Query for classified content in the domain
        query = db.session.query(Content, ContentClassification).join(
            ContentClassification,
            Content.id == ContentClassification.content_id
        ).filter(
            ContentClassification.is_primary == True
        ).limit(count)
        
        for content, classification in query:
            # Get any existing entity mentions
            from db.models.entity_models import EntityMention
            mentions = db.session.query(EntityMention).filter(
                EntityMention.content_id == content.id
            ).all()
            
            # Convert to spans format
            spans = []
            for mention in mentions:
                # Get entity type
                from db.models.entity_models import Entity, EntityType
                entity = db.session.query(Entity).filter(
                    Entity.id == mention.entity_id
                ).first()
                
                if entity and entity.entity_type:
                    entity_type = db.session.query(EntityType).filter(
                        EntityType.id == entity.entity_type_id
                    ).first()
                    
                    if entity_type and entity_type.name in entity_types:
                        spans.append({
                            "start": mention.start_char,
                            "end": mention.end_char,
                            "label": entity_type.name,
                            "text": mention.mention_text
                        })
            
            # Create metadata
            meta = {
                "content_id": content.id,
                "source_url": content.source_url,
                "title": content.title,
                "date": content.created_at,
                "domain": domain,
                "topic_id": classification.topic_id
            }
            
            yield create_ner_task(content.text, spans, meta)
            
    except ImportError:
        logger.error("Database models not available")
        return


# Prodigy dataset management

def create_dataset(dataset_name: str, description: Optional[str] = None) -> bool:
    """
    Create a new Prodigy dataset.
    
    Args:
        dataset_name (str): Name of the dataset.
        description (str, optional): Dataset description.
        
    Returns:
        bool: True if dataset was created, False otherwise.
    """
    if not PRODIGY_AVAILABLE:
        logger.error("Prodigy not installed")
        return False
    
    try:
        db = connect()
        if not db.get_dataset(dataset_name):
            db.add_dataset(dataset_name, meta={"desc": description})
            logger.info(f"Created dataset: {dataset_name}")
            return True
        else:
            logger.info(f"Dataset already exists: {dataset_name}")
            return False
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        return False


def get_dataset_stats(dataset_name: str) -> Dict[str, Any]:
    """
    Get statistics for a Prodigy dataset.
    
    Args:
        dataset_name (str): Name of the dataset.
        
    Returns:
        Dict[str, Any]: Dataset statistics.
    """
    if not PRODIGY_AVAILABLE:
        logger.error("Prodigy not installed")
        return {}
    
    try:
        db = connect()
        examples = db.get_dataset(dataset_name)
        
        if not examples:
            logger.warning(f"Dataset not found: {dataset_name}")
            return {}
        
        # Count annotations
        total = len(examples)
        accepted = len([ex for ex in examples if ex.get("answer") == "accept"])
        rejected = len([ex for ex in examples if ex.get("answer") == "reject"])
        unanswered = total - accepted - rejected
        
        # Get annotation types
        view_ids = set(ex.get("_view_id") for ex in examples if "_view_id" in ex)
        
        # Get entity/span statistics if applicable
        entity_counts = {}
        if "ner" in view_ids or "spans" in view_ids:
            for ex in examples:
                if ex.get("spans") and ex.get("answer") == "accept":
                    for span in ex["spans"]:
                        label = span.get("label")
                        if label:
                            entity_counts[label] = entity_counts.get(label, 0) + 1
        
        return {
            "dataset": dataset_name,
            "total": total,
            "accepted": accepted,
            "rejected": rejected,
            "unanswered": unanswered,
            "completion": f"{(accepted + rejected) / max(1, total):.1%}",
            "view_ids": list(view_ids),
            "entity_counts": entity_counts,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting dataset stats: {str(e)}")
        return {}


def export_dataset(dataset_name: str, output_path: str, 
                  format: str = "jsonl") -> bool:
    """
    Export a Prodigy dataset to file.
    
    Args:
        dataset_name (str): Name of the dataset.
        output_path (str): Path to save the export.
        format (str, optional): Export format (jsonl, spacy).
        
    Returns:
        bool: True if export succeeded, False otherwise.
    """
    if not PRODIGY_AVAILABLE:
        logger.error("Prodigy not installed")
        return False
    
    try:
        db = connect()
        examples = db.get_dataset(dataset_name)
        
        if not examples:
            logger.warning(f"Dataset not found or empty: {dataset_name}")
            return False
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for ex in examples:
                    json.dump(ex, f, ensure_ascii=False)
                    f.write('\n')
                    
        elif format == "spacy":
            if "ner" not in [ex.get("_view_id") for ex in examples if "_view_id" in ex]:
                logger.error("Only NER datasets can be exported to spaCy format")
                return False
                
            try:
                from prodigy.components.db import get_examples
                from prodigy.components.loaders import get_stream
                from prodigy.components.exporters import export_spacy
                
                # Export to spaCy format
                stream = get_examples(db, [dataset_name])
                export_spacy(output_path, get_stream(stream))
                
            except ImportError:
                logger.error("Prodigy exporters not available")
                return False
                
        else:
            logger.error(f"Unsupported export format: {format}")
            return False
        
        logger.info(f"Exported dataset {dataset_name} to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting dataset: {str(e)}")
        return False


# Functions for converting between Prodigy and internal models

def prodigy_to_content_classification(example: Dict) -> Dict:
    """
    Convert Prodigy classification annotation to internal format.
    
    Args:
        example (Dict): Prodigy annotation example.
        
    Returns:
        Dict: Internal format classification.
    """
    if example.get("answer") != "accept":
        return None
    
    meta = example.get("meta", {})
    content_id = meta.get("content_id")
    
    if not content_id:
        logger.warning("Missing content_id in annotation")
        return None
    
    # Get selected option
    option = example.get("accept", [None])[0]
    
    if not option:
        logger.warning("No selected option in annotation")
        return None
    
    return {
        "content_id": content_id,
        "topic_name": option,
        "confidence": 1.0,  # Human annotation has max confidence
        "is_primary": True,
        "classification_method": "human_annotation",
        "human_verified": True,
        "prodigy_task_hash": example.get("_task_hash")
    }


def prodigy_to_entity_mentions(example: Dict) -> List[Dict]:
    """
    Convert Prodigy NER annotation to internal entity mentions.
    
    Args:
        example (Dict): Prodigy annotation example.
        
    Returns:
        List[Dict]: Internal format entity mentions.
    """
    if example.get("answer") != "accept":
        return []
    
    meta = example.get("meta", {})
    content_id = meta.get("content_id")
    
    if not content_id:
        logger.warning("Missing content_id in annotation")
        return []
    
    # Get spans
    spans = example.get("spans", [])
    
    mentions = []
    for span in spans:
        start = span.get("start")
        end = span.get("end")
        text = span.get("text")
        label = span.get("label")
        
        if None in (start, end, text, label):
            logger.warning(f"Incomplete span data: {span}")
            continue
        
        # Get context before and after
        full_text = example.get("text", "")
        context_window = 50  # Characters
        
        context_before = full_text[max(0, start - context_window):start]
        context_after = full_text[end:min(len(full_text), end + context_window)]
        
        mentions.append({
            "content_id": content_id,
            "entity_type": label,
            "start_char": start,
            "end_char": end,
            "mention_text": text,
            "context_before": context_before,
            "context_after": context_after,
            "confidence": 1.0,  # Human annotation has max confidence
            "human_verified": True,
            "prodigy_task_hash": example.get("_task_hash")
        })
    
    return mentions


def prodigy_to_entity_links(example: Dict) -> List[Dict]:
    """
    Convert Prodigy entity linking annotation to internal format.
    
    Args:
        example (Dict): Prodigy annotation example.
        
    Returns:
        List[Dict]: Internal format entity links.
    """
    if example.get("answer") != "accept":
        return []
    
    meta = example.get("meta", {})
    content_id = meta.get("content_id")
    
    if not content_id:
        logger.warning("Missing content_id in annotation")
        return []
    
    # Get spans and their KB links
    spans = example.get("spans", [])
    links = []
    
    for span in spans:
        start = span.get("start")
        end = span.get("end")
        text = span.get("text")
        kb_id = span.get("kb_id")
        label = span.get("label")
        
        if None in (start, end, text, kb_id):
            # This may be a span without linking
            continue
        
        links.append({
            "content_id": content_id,
            "entity_type": label,
            "start_char": start,
            "end_char": end,
            "mention_text": text,
            "kb_id": kb_id,
            "confidence": 1.0,  # Human annotation has max confidence
            "human_verified": True,
            "prodigy_task_hash": example.get("_task_hash")
        })
    
    return links


def prodigy_to_temporal_references(example: Dict) -> List[Dict]:
    """
    Convert Prodigy temporal annotation to internal format.
    
    Args:
        example (Dict): Prodigy annotation example.
        
    Returns:
        List[Dict]: Internal format temporal references.
    """
    if example.get("answer") != "accept":
        return []
    
    meta = example.get("meta", {})
    content_id = meta.get("content_id")
    
    if not content_id:
        logger.warning("Missing content_id in annotation")
        return []
    
    # Get spans
    spans = example.get("spans", [])
    references = []
    
    for span in spans:
        start = span.get("start")
        end = span.get("end")
        text = span.get("text")
        label = span.get("label")
        
        if None in (start, end, text, label):
            logger.warning(f"Incomplete span data: {span}")
            continue
        
        # Get context
        full_text = example.get("text", "")
        context_window = 50  # Characters
        context = full_text[max(0, start - context_window):min(len(full_text), end + context_window)]
        
        references.append({
            "content_id": content_id,
            "reference_type": label,
            "extracted_text": text,
            "context": context,
            "confidence": 1.0,  # Human annotation has max confidence
            "prodigy_task_hash": example.get("_task_hash")
        })
    
    return references


# Functions for processing annotation results

def process_classification_annotations(dataset_name: str) -> Dict[str, Any]:
    """
    Process classification annotations and update database.
    
    Args:
        dataset_name (str): Name of the Prodigy dataset.
        
    Returns:
        Dict[str, Any]: Processing results.
    """
    if not PRODIGY_AVAILABLE:
        logger.error("Prodigy not installed")
        return {"error": "Prodigy not installed"}
    
    try:
        # Get annotations
        db = connect()
        examples = db.get_dataset(dataset_name)
        
        if not examples:
            logger.warning(f"Dataset not found or empty: {dataset_name}")
            return {"error": "Dataset not found or empty"}
        
        # Process classification annotations
        classifications = []
        for ex in examples:
            if ex.get("answer") == "accept":
                classification = prodigy_to_content_classification(ex)
                if classification:
                    classifications.append(classification)
        
        # Update database
        try:
            from db.models.topic_taxonomy import ContentClassification, TopicTaxonomy
            from app import db
            
            updated = 0
            created = 0
            
            for classification in classifications:
                # Get topic ID from name
                topic = db.session.query(TopicTaxonomy).filter(
                    TopicTaxonomy.name == classification["topic_name"]
                ).first()
                
                if not topic:
                    logger.warning(f"Topic not found: {classification['topic_name']}")
                    continue
                
                # Check if classification exists
                existing = db.session.query(ContentClassification).filter(
                    ContentClassification.content_id == classification["content_id"],
                    ContentClassification.topic_id == topic.id,
                    ContentClassification.is_primary == True
                ).first()
                
                if existing:
                    # Update existing classification
                    existing.confidence = classification["confidence"]
                    existing.human_verified = True
                    updated += 1
                else:
                    # Create new classification
                    new_classification = ContentClassification(
                        content_id=classification["content_id"],
                        topic_id=topic.id,
                        confidence=classification["confidence"],
                        is_primary=classification["is_primary"],
                        classification_method=classification["classification_method"],
                        human_verified=classification["human_verified"]
                    )
                    db.session.add(new_classification)
                    created += 1
            
            db.session.commit()
            
            return {
                "dataset": dataset_name,
                "processed": len(classifications),
                "updated": updated,
                "created": created
            }
            
        except ImportError:
            logger.error("Database models not available")
            return {"error": "Database models not available"}
            
    except Exception as e:
        logger.error(f"Error processing annotations: {str(e)}")
        return {"error": str(e)}


def process_ner_annotations(dataset_name: str) -> Dict[str, Any]:
    """
    Process NER annotations and update database.
    
    Args:
        dataset_name (str): Name of the Prodigy dataset.
        
    Returns:
        Dict[str, Any]: Processing results.
    """
    if not PRODIGY_AVAILABLE:
        logger.error("Prodigy not installed")
        return {"error": "Prodigy not installed"}
    
    try:
        # Get annotations
        db = connect()
        examples = db.get_dataset(dataset_name)
        
        if not examples:
            logger.warning(f"Dataset not found or empty: {dataset_name}")
            return {"error": "Dataset not found or empty"}
        
        # Process NER annotations
        all_mentions = []
        for ex in examples:
            if ex.get("answer") == "accept":
                mentions = prodigy_to_entity_mentions(ex)
                all_mentions.extend(mentions)
        
        # Update database
        try:
            from db.models.entity_models import EntityType, Entity, EntityMention
            from app import db
            
            updated = 0
            created = 0
            entity_types_created = 0
            entities_created = 0
            
            for mention in all_mentions:
                # Get or create entity type
                entity_type = db.session.query(EntityType).filter(
                    EntityType.name == mention["entity_type"]
                ).first()
                
                if not entity_type:
                    # Extract domain from dataset name
                    domain = dataset_name.split('_')[0] if '_' in dataset_name else 'general'
                    
                    entity_type = EntityType(
                        name=mention["entity_type"],
                        domain=domain,
                        description=f"Entity type for {mention['entity_type']}"
                    )
                    db.session.add(entity_type)
                    db.session.flush()
                    entity_types_created += 1
                
                # Get or create entity (simplified, would normally involve more complex logic)
                entity = db.session.query(Entity).filter(
                    Entity.name == mention["mention_text"],
                    Entity.entity_type_id == entity_type.id
                ).first()
                
                if not entity:
                    entity = Entity(
                        name=mention["mention_text"],
                        entity_type_id=entity_type.id,
                        canonical_name=mention["mention_text"]
                    )
                    db.session.add(entity)
                    db.session.flush()
                    entities_created += 1
                
                # Check if mention exists
                existing = db.session.query(EntityMention).filter(
                    EntityMention.content_id == mention["content_id"],
                    EntityMention.start_char == mention["start_char"],
                    EntityMention.end_char == mention["end_char"]
                ).first()
                
                if existing:
                    # Update existing mention
                    existing.entity_id = entity.id
                    existing.confidence = mention["confidence"]
                    existing.human_verified = True
                    updated += 1
                else:
                    # Create new mention
                    new_mention = EntityMention(
                        content_id=mention["content_id"],
                        entity_id=entity.id,
                        start_char=mention["start_char"],
                        end_char=mention["end_char"],
                        mention_text=mention["mention_text"],
                        confidence=mention["confidence"],
                        context_before=mention.get("context_before"),
                        context_after=mention.get("context_after"),
                        human_verified=True
                    )
                    db.session.add(new_mention)
                    created += 1
            
            db.session.commit()
            
            return {
                "dataset": dataset_name,
                "processed": len(all_mentions),
                "mentions_updated": updated,
                "mentions_created": created,
                "entity_types_created": entity_types_created,
                "entities_created": entities_created
            }
            
        except ImportError:
            logger.error("Database models not available")
            return {"error": "Database models not available"}
            
    except Exception as e:
        logger.error(f"Error processing annotations: {str(e)}")
        return {"error": str(e)}


# Functions for creating custom Prodigy recipes

def create_football_ner_recipe(dataset: str = None, source: str = None) -> Dict[str, Any]:
    """
    Create a custom Prodigy recipe for football NER.
    
    Args:
        dataset (str, optional): Dataset name.
        source (str, optional): Source of content (file path or database).
        
    Returns:
        Dict[str, Any]: Prodigy recipe configuration.
    """
    if not PRODIGY_AVAILABLE:
        logger.error("Prodigy not installed")
        return {}
    
    # Get football entity types from config
    football_config = Config.get_football_config()
    
    # Define labels based on domain configuration
    labels = []
    
    # Add team labels
    labels.append("TEAM")
    
    # Add person labels
    labels.append("PERSON.PLAYER")
    labels.append("PERSON.MANAGER")
    
    # Add venue label
    labels.append("VENUE")
    
    # Add competition label
    labels.append("COMPETITION")
    
    # Add event labels
    labels.append("EVENT.MATCH")
    labels.append("EVENT.TRANSFER")
    
    # Add statistic labels
    labels.append("STATISTIC.GOAL")
    labels.append("STATISTIC.ASSIST")
    
    # Determine stream source
    if source and source.endswith('.jsonl'):
        stream_source = "jsonl_file"
    elif source and os.path.isdir(source):
        stream_source = "directory"
    else:
        stream_source = "database"
    
    # Create recipe config
    recipe = {
        "dataset": dataset or "football_ner",
        "labels": labels,
        "view_id": "ner",
        "stream": stream_source,
        "source": source,
        "exclude": [],
        "patterns": [
            # Pattern for teams (examples)
            {"label": "TEAM", "pattern": "Manchester United"},
            {"label": "TEAM", "pattern": "Liverpool"},
            {"label": "TEAM", "pattern": "Arsenal"},
            {"label": "TEAM", "pattern": "Chelsea"},
            {"label": "TEAM", "pattern": "Manchester City"},
            
            # Pattern for players (examples)
            {"label": "PERSON.PLAYER", "pattern": [{"LOWER": "mohamed"}, {"LOWER": "salah"}]},
            {"label": "PERSON.PLAYER", "pattern": [{"LOWER": "kevin"}, {"LOWER": "de"}, {"LOWER": "bruyne"}]},
            
            # Pattern for competitions
            {"label": "COMPETITION", "pattern": "Premier League"},
            {"label": "COMPETITION", "pattern": "FA Cup"},
            {"label": "COMPETITION", "pattern": "Champions League"}
        ]
    }
    
    return recipe


def create_football_classification_recipe(dataset: str = None, source: str = None) -> Dict[str, Any]:
    """
    Create a custom Prodigy recipe for football content classification.
    
    Args:
        dataset (str, optional): Dataset name.
        source (str, optional): Source of content (file path or database).
        
    Returns:
        Dict[str, Any]: Prodigy recipe configuration.
    """
    if not PRODIGY_AVAILABLE:
        logger.error("Prodigy not installed")
        return {}
    
    # Define options for classification
    options = [
        {"id": "match_report", "text": "Match Report"},
        {"id": "transfer_news", "text": "Transfer News"},
        {"id": "injury_update", "text": "Injury Update"},
        {"id": "team_news", "text": "Team News"},
        {"id": "opinion", "text": "Opinion/Analysis"},
        {"id": "preview", "text": "Match Preview"},
        {"id": "interview", "text": "Interview"},
        {"id": "other", "text": "Other Football Content"},
        {"id": "not_football", "text": "Not Football Related"}
    ]
    
    # Determine stream source
    if source and source.endswith('.jsonl'):
        stream_source = "jsonl_file"
    elif source and os.path.isdir(source):
        stream_source = "directory"
    else:
        stream_source = "database"
    
    # Create recipe config
    recipe = {
        "dataset": dataset or "football_classification",
        "view_id": "classification",
        "stream": stream_source,
        "source": source,
        "exclude": [],
        "options": options,
        "choice_style": "single"  # Only allow one selection
    }
    
    return recipe


# Annotation workflow management

def create_annotation_workflow(domain: str, workflow_type: str, count: int = 100) -> Dict[str, Any]:
    """
    Create a complete annotation workflow for a domain.
    
    Args:
        domain (str): Domain for the workflow.
        workflow_type (str): Type of annotation workflow.
        count (int, optional): Number of items to include.
        
    Returns:
        Dict[str, Any]: Workflow configuration.
    """
    if not PRODIGY_AVAILABLE:
        logger.error("Prodigy not installed")
        return {"error": "Prodigy not installed"}
    
    # Create dataset names
    dataset_name = f"{domain}_{workflow_type}_{datetime.now().strftime('%Y%m%d')}"
    
    # Create datasets based on workflow type
    if workflow_type == "ner":
        # Define entity types based on domain
        if domain == "football":
            entity_types = ["TEAM", "PERSON.PLAYER", "PERSON.MANAGER", "VENUE", 
                           "COMPETITION", "EVENT.MATCH", "EVENT.TRANSFER"]
            
            # Create dataset
            create_dataset(dataset_name, f"Football NER annotation dataset")
            
            # Create recipe
            recipe = create_football_ner_recipe(dataset_name)
            
            return {
                "domain": domain,
                "type": workflow_type,
                "dataset": dataset_name,
                "recipe": "ner.manual",
                "config": recipe,
                "entity_types": entity_types,
                "created_at": datetime.now().isoformat()
            }
            
    elif workflow_type == "classification":
        # Create dataset
        create_dataset(dataset_name, f"{domain.capitalize()} content classification dataset")
        
        # Create recipe based on domain
        if domain == "football":
            recipe = create_football_classification_recipe(dataset_name)
            
            return {
                "domain": domain,
                "type": workflow_type,
                "dataset": dataset_name,
                "recipe": "textcat.manual",
                "config": recipe,
                "created_at": datetime.now().isoformat()
            }
    
    # Default response
    return {
        "error": f"Unsupported workflow type for domain: {workflow_type} - {domain}"
    }


def get_annotation_progress(dataset_name: str) -> Dict[str, Any]:
    """
    Get progress statistics for an annotation workflow.
    
    Args:
        dataset_name (str): Name of the dataset.
        
    Returns:
        Dict[str, Any]: Progress statistics.
    """
    # Get dataset stats
    stats = get_dataset_stats(dataset_name)
    
    if not stats:
        return {"error": f"Dataset not found: {dataset_name}"}
    
    # Calculate additional metrics
    completion_rate = (stats["accepted"] + stats["rejected"]) / max(1, stats["total"])
    
    # Calculate estimated time to completion
    # Assuming 10 seconds per annotation
    remaining = stats["unanswered"]
    estimated_seconds = remaining * 10
    
    # Format as hours and minutes
    hours = estimated_seconds // 3600
    minutes = (estimated_seconds % 3600) // 60
    
    return {
        "dataset": dataset_name,
        "total_examples": stats["total"],
        "annotated": stats["accepted"] + stats["rejected"],
        "accepted": stats["accepted"],
        "rejected": stats["rejected"],
        "remaining": remaining,
        "completion_rate": f"{completion_rate:.1%}",
        "estimated_time": f"{int(hours)}h {int(minutes)}m",
        "entity_counts": stats.get("entity_counts", {}),
        "last_updated": stats["last_updated"]
    }
