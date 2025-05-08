use env_logger::{Builder, Env};
use log::LevelFilter;
use std::io::Write;

/// Initialize the logger with a default configuration
pub fn init_logger() {
    let env = Env::default()
        .filter_or("RUST_LOG", "info")
        .write_style_or("RUST_LOG_STYLE", "always");
        
    Builder::from_env(env)
        .format(|buf, record| {
            let ts = buf.timestamp();
            writeln!(
                buf,
                "[{} {}] {}",
                ts,
                record.level(),
                record.args()
            )
        })
        .init();
}

/// Initialize a more verbose logger for debugging
pub fn init_debug_logger() {
    let env = Env::default()
        .filter_or("RUST_LOG", "debug")
        .write_style_or("RUST_LOG_STYLE", "always");
        
    Builder::from_env(env)
        .format(|buf, record| {
            let ts = buf.timestamp();
            writeln!(
                buf,
                "[{} {} {}:{}] {}",
                ts,
                record.level(),
                record.file().unwrap_or("unknown"),
                record.line().unwrap_or(0),
                record.args()
            )
        })
        .init();
}

/// Set the global log level
pub fn set_log_level(level: LevelFilter) {
    log::set_max_level(level);
}