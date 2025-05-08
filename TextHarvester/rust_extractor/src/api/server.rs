use crate::api::handlers;
use crate::api::state::AppState;
use crate::models::DatabaseConfig;
use crate::extractors::ContentExtractor;
use actix_cors::Cors;
use actix_web::{middleware, web, App, HttpServer};
use log::{info, warn};
use std::sync::Arc;

pub struct Server {
    host: String,
    port: u16,
}

impl Default for Server {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8888,
        }
    }
}

impl Server {
    pub fn new(host: String, port: u16) -> Self {
        Self { host, port }
    }
    
    pub async fn run(&self) -> std::io::Result<()> {
        let addr = format!("{}:{}", self.host, self.port);
        info!("Starting content extractor server at {}", addr);
        
        // Create shared application state
        let extractor = Arc::new(ContentExtractor::new());
        let app_state = web::Data::new(AppState {
            extractor,
        });
        
        // Configure and start the HTTP server
        HttpServer::new(move || {
            // Configure CORS
            let cors = Cors::default()
                .allow_any_origin()
                .allow_any_method()
                .allow_any_header()
                .max_age(3600);
                
            App::new()
                .wrap(middleware::Logger::default())
                .wrap(cors)
                .app_data(app_state.clone())
                // API routes
                .configure(handlers::extract::configure_routes)
                .configure(handlers::export::configure_routes)
        })
        .bind(&addr)?
        .run()
        .await
    }
}