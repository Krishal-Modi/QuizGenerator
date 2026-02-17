# Quick Start Script for Quiz Generator

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Quiz Generator - Starting Server" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
& .\.venv\Scripts\Activate.ps1

# Run the Flask application
python app.py
