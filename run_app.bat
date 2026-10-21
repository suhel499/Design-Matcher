@echo off
REM ==============================================================================
REM Design Matcher - Desktop Application Launcher (Windows)
REM ==============================================================================
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    echo Launching Design Matcher GUI...
    start "" ".venv\Scripts\python.exe" run.py %*
) else (
    echo [ERROR] Virtual environment not found at .venv!
    pause
)
