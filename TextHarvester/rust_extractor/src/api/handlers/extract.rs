use actix_web::{web, HttpResponse, Responder};
use chrono::Utc;
use log::info;
use serde::{Serialize, Deserialize};

use crate::extractors::ContentExtractor;
use crate::models::{ExtractionRequest, ExtractionOptions};

/// Extract content from a URL or HTML
pub async fn extract_content(
    req: web::Json<ExtractionRequest>,
) -> impl Responder {
    let url = &req.url;
    let html_content = req.html_content.clone();
    
    info!("Extracting content from URL: {}", url);
    
    // Create extractor
    let extractor = ContentExtractor::new();
    
    // Create options
    let options = ExtractionOptions {
        // No need to unwrap as clean_text has a default value
        clean_text: req.options.clean_text,
        ..Default::default()
    };
    
    // Extract content
    let result = if let Some(html) = html_content.as_deref() {
        extractor.extract_from_html(url, html, &options)
    } else {
        extractor.extract_from_url(url, &options)
    };
    
    match result {
        Ok(result) => {
            HttpResponse::Ok().json(result)
        },
        Err(e) => {
            HttpResponse::InternalServerError().json(serde_json::json!({
                "success": false,
                "error": e.to_string(),
                "timestamp": Utc::now().to_rfc3339(),
            }))
        }
    }
}

/// Health check endpoint
pub async fn health() -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({
        "status": "ok",
        "message": "API is operational",
        "timestamp": Utc::now().to_rfc3339(),
    }))
}

/// Configure API routes
pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api")
            .route("/health", web::get().to(health))
            .route("/extract", web::post().to(extract_content))
    );
}