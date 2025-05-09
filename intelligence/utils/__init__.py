"""
Intelligence Utilities
====================

This package provides utility functions and classes for the intelligence layer.
"""

from .text_processing import (
    clean_text, 
    normalize_whitespace, 
    normalize_unicode,
    remove_special_chars,
    strip_html,
    detect_language,
    get_sentences,
    get_tokens,
    get_text_stats,
    chunk_text,
    normalize_football_teams,
    extract_football_scores,
    normalize_player_names,
    extract_temporal_expressions,
    create_domain_preprocessor
)

from .model_utils import (
    load_model,
    save_model,
    clear_model_cache,
    get_model_cache_info,
    with_model,
    get_embeddings,
    get_tfidf_vectors,
    compute_similarity_matrix,
    ModelPerformanceTracker,
    tracked_inference,
    batch_iterator,
    batched_inference,
    optimized_huggingface_inference,
    optimized_spacy_inference,
    k_fold_cross_validate,
    get_model_path
)

# Conditionally import Prodigy integration if available
try:
    from .prodigy_integration import (
        create_text_classification_task,
        create_ner_task,
        create_span_categorization_task,
        create_entity_linking_task,
        create_temporal_annotation_task,
        create_content_stream,
        create_classification_stream_from_db,
        create_ner_stream_from_db,
        create_dataset,
        get_dataset_stats,
        export_dataset,
        prodigy_to_content_classification,
        prodigy_to_entity_mentions,
        prodigy_to_entity_links,
        prodigy_to_temporal_references,
        process_classification_annotations,
        process_ner_annotations,
        create_football_ner_recipe,
        create_football_classification_recipe,
        create_annotation_workflow,
        get_annotation_progress
    )
except ImportError:
    pass
