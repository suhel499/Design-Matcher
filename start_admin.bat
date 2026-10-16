@echo off
REM ==============================================================================
REM Design Matcher - Admin Portal Launcher (Windows)
REM ==============================================================================
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo ==========================================================
echo  Starting Design Matcher Licensing Admin Server...
echo  Web Portal URL:   http://localhost:8000/admin
echo  API Docs URL:     http://localhost:8000/docs
echo  Default Username: admin
echo  Default Password: AdminPass@2026
echo ==========================================================

python -m uvicorn licensing_server.main:app --host 0.0.0.0 --port 8000 --reload
pause
