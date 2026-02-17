@echo off
REM Quick start script for Quiz Generator

echo.
echo ========================================
echo   Quiz Generator - Starting Server
echo ========================================
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run the Flask application
python app.py

pause
