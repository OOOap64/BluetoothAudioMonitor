@echo off
chcp 65001 > nul
cd /d "%~dp0"
set "PY=python"
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PY=venv\Scripts\python.exe"
)
echo Starting Bluetooth Audio Monitor...
"%PY%" main.py
pause
