#!/bin/bash
# FastAPI Development Server Launcher
# This script starts the FastAPI server using the virtual environment

cd "$(dirname "$0")/.."
.venv/bin/python -m fastapi_cli dev backend/app/main.py
