#!/bin/bash
# Build script for the Rust content extractor

set -e

echo "Building Rust content extractor..."
cd "$(dirname "$0")"

# Build the release version
cargo build --release

# Copy the binary to the expected location
mkdir -p target/release
echo "Build complete! Binary available at target/release/rust_extractor"
echo ""
echo "To use the Rust extractor:"
echo "1. Run it directly: ./target/release/rust_extractor extract https://example.com"
echo "2. Start the API server: ./target/release/rust_extractor server"
echo "3. Set USE_RUST_EXTRACTOR=true in your environment to use it from Python"