"""
Content Intelligence Platform
===========================

This package provides the intelligence layer for the TextHarvester platform,
transforming raw scraped content into structured, actionable insights.
"""

__version__ = '0.1.0'

from .config import Config
from .base_pipeline import (
    BasePipeline, 
    ParallelPipeline, 
    PipelineState, 
    WorkItem, 
    PipelineResult,
    PipelineRegistry,
    registry
)
