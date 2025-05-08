@echo off
echo Starting Rust extractor service...
cd %~dp0\TextHarvester\rust_extractor
start /b .\target\release\rust_extractor.exe server --host 127.0.0.1 --port 8888
echo Rust extractor service started on http://127.0.0.1:8888
echo Press CTRL+C to stop this script, but the Rust extractor will continue running in the background.
pause
