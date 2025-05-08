use actix_web::{web, HttpResponse, Responder};
use chrono::Utc;
use log::{info, error};
use serde::{Serialize, Deserialize};
use std::sync::Arc;

use crate::exporters::job_exporter::{JobExporter, ExportOptions};
use crate::api::state::AppState;
use crate::models::DatabaseConfig;

#[derive(Debug, Serialize, Deserialize)]
pub struct ExportRequest {
    pub job_id: i32,
    pub chunk_size: Option<usize>,
    pub overlap: Option<usize>,
    pub db_url: Option<String>,
}

/// Export job data to JSONL format
pub async fn export_job(
    req: web::Json<ExportRequest>,
    _state: web::Data<AppState>,
) -> impl Responder {
    let job_id = req.job_id;
    let chunk_size = req.chunk_size.unwrap_or(500);
    let overlap = req.overlap.unwrap_or(50);
    
    info!("Starting Rust-based export for job {}", job_id);
    
    // Get database URL from request or environment
    let db_url = match &req.db_url {
        Some(url) => url.clone(),
        None => match std::env::var("DATABASE_URL") {
            Ok(url) => url,
            Err(_) => {
                error!("Database URL not provided and not found in environment");
                return HttpResponse::BadRequest().json(serde_json::json!({
                    "success": false,
                    "error": "Database URL not provided and not found in environment",
                    "timestamp": Utc::now().to_rfc3339(),
                }));
            }
        }
    };
    
    // Create database config
    let db_config = DatabaseConfig {
        url: db_url,
    };
    
    // Create exporter
    let exporter = JobExporter::new(db_config);
    
    // Create export options
    let _options = ExportOptions {
        chunk_size,
        overlap,
    };
    
    // Start export in background and return job ID
    match exporter.validate_job(job_id) {
        Ok(content_count) => {
            // Return job information
            HttpResponse::Ok().json(serde_json::json!({
                "success": true,
                "job_id": job_id,
                "content_count": content_count,
                "message": format!("Export initialized for job {} with {} documents", job_id, content_count),
                "timestamp": Utc::now().to_rfc3339(),
            }))
        },
        Err(e) => {
            error!("Error validating job {}: {}", job_id, e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "success": false,
                "error": e.to_string(),
                "timestamp": Utc::now().to_rfc3339(),
            }))
        }
    }
}

/// Stream export results for a job
pub async fn stream_export(
    path: web::Path<i32>,
    query: web::Query<ExportOptions>,
    _state: web::Data<AppState>,
) -> impl Responder {
    let job_id = path.into_inner();
    let chunk_size = query.chunk_size;
    let overlap = query.overlap;
    
    info!("Streaming export for job {} with chunk_size={}, overlap={}", 
         job_id, chunk_size, overlap);
    
    // Get database URL from environment
    let db_url = match std::env::var("DATABASE_URL") {
        Ok(url) => url,
        Err(_) => {
            error!("Database URL not found in environment");
            return HttpResponse::BadRequest().json(serde_json::json!({
                "success": false,
                "error": "Database URL not found in environment",
                "timestamp": Utc::now().to_rfc3339(),
            }));
        }
    };
    
    // Create database config
    let db_config = DatabaseConfig {
        url: db_url,
    };
    
    // Create exporter
    let exporter = JobExporter::new(db_config);
    
    // Create export options
    let options = ExportOptions {
        chunk_size,
        overlap,
    };
    
    // Create a streaming response with chunked transfer encoding
    HttpResponse::Ok()
        .content_type("application/x-jsonl")  // Use JSONL content type
        .streaming(exporter.export_job_stream(job_id, &options))
}

/// Configure API routes
pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/export")
            .route("/job", web::post().to(export_job))
            .route("/job/{job_id}/stream", web::get().to(stream_export))
    );
}