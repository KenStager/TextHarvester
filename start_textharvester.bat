@echo off
echo Starting TextHarvester application...
cd %~dp0\TextHarvester

:: Try to use the conda environment if available
IF EXIST %USERPROFILE%\anaconda3\condabin\conda.bat (
    call %USERPROFILE%\anaconda3\condabin\conda.bat activate prodigy_env
) ELSE IF EXIST %USERPROFILE%\miniconda3\condabin\conda.bat (
    call %USERPROFILE%\miniconda3\condabin\conda.bat activate prodigy_env
)

:: Try different Python commands
where python >nul 2>nul
IF %ERRORLEVEL% EQU 0 (
    echo Using system Python...
    python main.py
) ELSE (
    echo Python command not found, trying python3...
    where python3 >nul 2>nul
    IF %ERRORLEVEL% EQU 0 (
        python3 main.py
    ) ELSE (
        echo ERROR: Python not found. Please make sure Python is installed and in your PATH.
        pause
    )
)

pause
