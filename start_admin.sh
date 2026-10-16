#!/usr/bin/env bash
# ==============================================================================
# Design Matcher - Admin Portal Launcher (macOS / Linux)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "=========================================================="
echo " Starting Design Matcher Licensing Admin Server..."
echo " Web Portal URL:   http://localhost:8000/admin"
echo " API Docs URL:     http://localhost:8000/docs"
echo " Default Username: admin"
echo " Default Password: AdminPass@2026"
echo "=========================================================="

python3 -m uvicorn licensing_server.main:app --host 0.0.0.0 --port 8000 --reload
