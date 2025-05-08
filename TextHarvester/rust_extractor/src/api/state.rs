use std::sync::Arc;
use crate::extractors::ContentExtractor;

/// Shared application state
pub struct AppState {
    pub extractor: Arc<ContentExtractor>,
}