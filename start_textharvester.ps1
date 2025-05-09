# PowerShell script to start TextHarvester
Write-Host "Starting TextHarvester application..." -ForegroundColor Green

# Change to the TextHarvester directory
Set-Location -Path "$PSScriptRoot\TextHarvester"

# Try to activate the conda environment if it exists
if (Test-Path "$env:USERPROFILE\anaconda3\condabin\conda.bat") {
    Write-Host "Activating Anaconda environment..." -ForegroundColor Yellow
    & "$env:USERPROFILE\anaconda3\condabin\conda.bat" activate prodigy_env
} elseif (Test-Path "$env:USERPROFILE\miniconda3\condabin\conda.bat") {
    Write-Host "Activating Miniconda environment..." -ForegroundColor Yellow
    & "$env:USERPROFILE\miniconda3\condabin\conda.bat" activate prodigy_env
}

# Check if python is available
try {
    $pythonVersion = python --version
    Write-Host "Using Python: $pythonVersion" -ForegroundColor Cyan
    
    # Start the application
    Write-Host "Starting main.py..." -ForegroundColor Green
    python main.py
} catch {
    Write-Host "Python command not found, trying python3..." -ForegroundColor Yellow
    try {
        $python3Version = python3 --version
        Write-Host "Using Python3: $python3Version" -ForegroundColor Cyan
        
        # Start the application with python3
        python3 main.py
    } catch {
        Write-Host "ERROR: Python not found. Please make sure Python is installed and in your PATH." -ForegroundColor Red
        Read-Host "Press Enter to exit"
    }
}

Read-Host "Press Enter to exit"
