"""
Knowledge Processing Pipeline

This module provides the KnowledgePipeline class for processing content
through the complete knowledge management workflow, including extraction,
storage, contradiction detection, and credibility assessment.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from app import db
from db.models.knowledge_base import KnowledgeNode, KnowledgeEdge
from intelligence.base_pipeline import BasePipeline

logger = logging.getLogger(__name__)


class KnowledgePipeline(BasePipeline):
    """
    Manages the complete knowledge processing workflow,
    coordinating extraction, storage, contradiction detection,
    and credibility assessment.
    """
    
    def __init__(self, domain: str = None):
        """
        Initialize the knowledge pipeline.
        
        Args:
            domain: Optional domain to filter operations by (e.g., 'football')
        """
        super().__init__(name="knowledge_pipeline")
        self.domain = domain
        
        # Initialize components
        from .extraction import KnowledgeExtractor
        from .storage import KnowledgeStorage
        from .conflict import ContradictionDetector
        from .credibility import CredibilityScorer
        from .graph import KnowledgeGraph
        
        self.extractor = KnowledgeExtractor(domain=domain)
        self.storage = KnowledgeStorage(domain=domain)
        self.conflict_detector = ContradictionDetector(domain=domain)
        self.credibility_scorer = CredibilityScorer(domain=domain)
        self.graph = KnowledgeGraph(domain=domain)
        
        logger.info(f"Initialized knowledge pipeline for domain: {domain}")
    
    def process_content(self, content_id: int) -> Dict[str, Any]:
        """
        Process a content item through the complete knowledge pipeline.
        
        Args:
            content_id: ID of the content to process
            
        Returns:
            Dictionary with processing results
        """
        self.start_processing()
        
        try:
            # Check if content exists in the scraper database
            from db.models.scraper import ScraperContent
            content = ScraperContent.query.get(content_id)
            
            if not content:
                logger.warning(f"Content not found: {content_id}")
                self.finish_processing()
                return {
                    "status": "error",
                    "message": "Content not found"
                }
            
            # Check if content has been enhanced
            from db.models.content_intelligence import EnhancedContent
            enhanced_content = EnhancedContent.query.filter_by(content_id=content_id).first()
            
            if not enhanced_content:
                logger.warning(f"Content has not been enhanced: {content_id}")
                self.finish_processing()
                return {
                    "status": "error",
                    "message": "Content has not been enhanced for intelligence extraction"
                }
            
            results = {
                "content_id": content_id,
                "title": content.title,
                "url": content.url,
                "steps": {}
            }
            
            # Step 1: Extract knowledge
            self.update_progress(10, "Extracting knowledge")
            extraction_results = self.extractor.extract_from_content(content_id)
            results["steps"]["extraction"] = {
                "status": "success",
                "entities": len(extraction_results.get("extracted_entities", [])),
                "relationships": len(extraction_results.get("extracted_relationships", [])),
                "claims": len(extraction_results.get("extracted_claims", [])),
                "events": len(extraction_results.get("extracted_events", []))
            }
            
            # Step 2: Evaluate source credibility
            self.update_progress(30, "Evaluating source credibility")
            if content.url:
                credibility_results = self.credibility_scorer.evaluate_source(content.url)
                results["steps"]["credibility"] = {
                    "status": "success",
                    "source_url": content.url,
                    "overall_score": credibility_results.get("overall_score", 0),
                    "domain_expertise": credibility_results.get("domain_expertise", {})
                }
            else:
                results["steps"]["credibility"] = {
                    "status": "skipped",
                    "reason": "No source URL available"
                }
            
            # Step 3: Detect contradictions in new knowledge
            self.update_progress(50, "Detecting contradictions")
            contradictions = []
            
            # Check for contradictions in each new claim
            for claim in extraction_results.get("extracted_claims", []):
                claim_contradictions = self.conflict_detector.detect_claim_contradictions(
                    new_claim_id=claim.get("node_id")
                )
                contradictions.extend(claim_contradictions)
            
            # Check for contradictions in each new entity
            for entity in extraction_results.get("extracted_entities", []):
                entity_contradictions = self.conflict_detector.detect_entity_contradictions(
                    entity_id=entity.get("node_id")
                )
                contradictions.extend(entity_contradictions)
            
            results["steps"]["contradiction_detection"] = {
                "status": "success",
                "contradictions_found": len(contradictions),
                "contradictions": contradictions
            }
            
            # Step 4: Update confidence based on source credibility
            self.update_progress(70, "Updating confidence scores")
            confidence_updates = []
            
            # Update confidence for nodes
            for entity in extraction_results.get("extracted_entities", []):
                update_result = self.credibility_scorer.recalculate_confidence_with_credibility(
                    node_id=entity.get("node_id")
                )
                if update_result.get("status") == "success":
                    confidence_updates.append(update_result)
            
            # Update confidence for edges
            for relationship in extraction_results.get("extracted_relationships", []):
                update_result = self.credibility_scorer.recalculate_edge_confidence(
                    edge_id=relationship.get("edge_id")
                )
                if update_result.get("status") == "success":
                    confidence_updates.append(update_result)
            
            results["steps"]["confidence_update"] = {
                "status": "success",
                "updates": len(confidence_updates)
            }
            
            # Step 5: Update knowledge statistics
            self.update_progress(90, "Updating knowledge statistics")
            graph_stats = self.storage.get_storage_stats()
            
            results["steps"]["statistics"] = {
                "status": "success",
                "nodes_total": graph_stats["nodes"]["total"],
                "edges_total": graph_stats["edges"]["total"],
                "contradictions": graph_stats["contradictions"]["unresolved"]
            }
            
            # Complete processing
            self.update_progress(100, "Processing complete")
            results["status"] = "success"
            results["processing_time"] = self.get_processing_time()
            
            self.finish_processing()
            logger.info(f"Completed knowledge processing for content {content_id}")
            
            return results
            
        except Exception as e:
            logger.exception(f"Error processing content {content_id}: {e}")
            self.finish_processing(error=str(e))
            
            return {
                "status": "error",
                "message": f"Processing failed: {str(e)}",
                "content_id": content_id
            }
    
    def batch_process(self, content_ids: List[int]) -> Dict[str, Any]:
        """
        Process multiple content items in batch.
        
        Args:
            content_ids: List of content IDs to process
            
        Returns:
            Dictionary with batch processing results
        """
        results = {
            "total": len(content_ids),
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "errors": [],
            "content_results": {}
        }
        
        for content_id in content_ids:
            try:
                # Process the content
                content_result = self.process_content(content_id)
                
                # Store the result
                results["content_results"][content_id] = content_result
                
                # Update counters
                if content_result.get("status") == "success":
                    results["successful"] += 1
                elif content_result.get("status") == "skipped":
                    results["skipped"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append({
                        "content_id": content_id,
                        "message": content_result.get("message", "Unknown error")
                    })
                
            except Exception as e:
                logger.exception(f"Error in batch processing for content {content_id}: {e}")
                results["failed"] += 1
                results["errors"].append({
                    "content_id": content_id,
                    "message": str(e)
                })
        
        return results
    
    def process_unprocessed_content(self, limit: int = 100) -> Dict[str, Any]:
        """
        Find and process content that hasn't been processed through the knowledge pipeline.
        
        Args:
            limit: Maximum number of content items to process
            
        Returns:
            Dictionary with processing results
        """
        # Find content that has been enhanced but not processed through the knowledge pipeline
        from db.models.content_intelligence import EnhancedContent
        from db.models.knowledge_base import KnowledgeSource
        
        # Get enhanced content IDs
        enhanced_content_ids = db.session.query(EnhancedContent.content_id).all()
        enhanced_content_ids = [id[0] for id in enhanced_content_ids]
        
        # Get knowledge source content IDs
        knowledge_source_content_ids = db.session.query(KnowledgeSource.content_id).distinct().all()
        knowledge_source_content_ids = [id[0] for id in knowledge_source_content_ids]
        
        # Find content that has been enhanced but not processed
        unprocessed_ids = list(set(enhanced_content_ids) - set(knowledge_source_content_ids))
        
        # Apply domain filter if necessary
        if self.domain:
            from db.models.content_intelligence import ContentClassification
            
            domain_content_ids = db.session.query(ContentClassification.content_id).filter(
                ContentClassification.topic_id.in_(
                    db.session.query(db.models.topic_taxonomy.TopicTaxonomy.id).filter(
                        db.models.topic_taxonomy.TopicTaxonomy.name.ilike(f"%{self.domain}%")
                    )
                )
            ).all()
            domain_content_ids = [id[0] for id in domain_content_ids]
            
            unprocessed_ids = list(set(unprocessed_ids).intersection(set(domain_content_ids)))
        
        # Limit the number of items to process
        unprocessed_ids = unprocessed_ids[:limit]
        
        # Process the unprocessed content
        return self.batch_process(unprocessed_ids)
    
    def refresh_knowledge_for_entity(self, entity_id: int) -> Dict[str, Any]:
        """
        Refresh knowledge for a specific entity by reprocessing related content.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            Dictionary with refresh results
        """
        # Find content mentioning this entity
        from db.models.entity_models import EntityMention
        
        content_ids = db.session.query(EntityMention.content_id).filter(
            EntityMention.entity_id == entity_id
        ).distinct().all()
        content_ids = [id[0] for id in content_ids]
        
        # Reprocess the content
        results = self.batch_process(content_ids)
        results["entity_id"] = entity_id
        
        return results
