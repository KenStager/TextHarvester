mod api;
mod extractors;
mod exporters;
mod models;
mod utils;

use api::Server;
use clap::{Args, Parser, Subcommand};
use log::{error, info, LevelFilter};

/// Rust content extractor for web pages - a high-performance alternative to Python-based extractors
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the API server
    Server(ServerArgs),
    
    /// Extract content from a URL and print to stdout
    Extract(ExtractArgs),
}

#[derive(Args)]
struct ServerArgs {
    /// Host to bind to
    #[arg(short, long, default_value = "0.0.0.0")]
    host: String,
    
    /// Port to listen on
    #[arg(short, long, default_value_t = 8888)]
    port: u16,
    
    /// Enable debug logging
    #[arg(short, long, default_value_t = false)]
    debug: bool,
}

#[derive(Args)]
struct ExtractArgs {
    /// URL to extract content from
    #[arg(required = true)]
    url: String,
    
    /// Output format (text or json)
    #[arg(short, long, default_value = "text")]
    format: String,
    
    /// Whether to clean the text
    #[arg(short, long, default_value_t = true)]
    clean: bool,
}

/// Simple logger initialization
fn init_logger() {
    env_logger::Builder::new()
        .filter_level(LevelFilter::Info)
        .format_timestamp(None)
        .format_module_path(false)
        .init();
}

/// Set log level
fn set_log_level(level: LevelFilter) {
    log::set_max_level(level);
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    // Initialize logger
    init_logger();
    
    // Parse command line arguments
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Server(args) => {
            // Set debug logging if requested
            if args.debug {
                set_log_level(LevelFilter::Debug);
                info!("Debug logging enabled");
            }
            
            // Start API server
            let server = Server::new(args.host, args.port);
            server.run().await?;
        },
        Commands::Extract(args) => {
            // Create content extractor
            let extractor = extractors::content::ContentExtractor::new();
            
            // Create extraction options
            let options = models::ExtractionOptions {
                clean_text: args.clean,
                ..Default::default()
            };
            
            // Extract content
            match extractor.extract_from_url(&args.url, &options) {
                Ok(result) => {
                    // Output according to format
                    if args.format == "json" {
                        println!("{}", serde_json::to_string_pretty(&result).unwrap());
                    } else {
                        // Text format - print title and content
                        if let Some(title) = &result.title {
                            println!("# {}\n", title);
                        }
                        println!("{}", result.text);
                        println!("\n---");
                        println!("Words: {}", result.stats.word_count);
                        println!("Characters: {}", result.stats.char_count);
                        println!("Paragraphs: {}", result.stats.paragraph_count);
                        if let Some(author) = &result.author {
                            println!("Author: {}", author);
                        }
                        if let Some(date) = &result.date {
                            println!("Date: {}", date);
                        }
                    }
                },
                Err(err) => {
                    error!("Failed to extract content: {}", err);
                    std::process::exit(1);
                }
            }
        }
    }
    
    Ok(())
}