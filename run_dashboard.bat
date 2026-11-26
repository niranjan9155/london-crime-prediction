@echo off
rem Sets up the project (first time only) and opens the dashboard in the browser.
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.11 or newer from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)

if not exist ".venv\setup_done.txt" (
    echo Setting up the project for the first time. This takes a few minutes ...
    python -m venv .venv || goto failed
    ".venv\Scripts\python.exe" -m pip install --upgrade pip || goto failed
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto failed
    ".venv\Scripts\python.exe" -m pip install -e . || goto failed
    echo done> ".venv\setup_done.txt"
)

echo Opening the dashboard. Keep this window open; press Ctrl+C here to stop it.
".venv\Scripts\python.exe" -m streamlit run dashboard.py
exit /b 0

:failed
echo.
echo Setup did not finish. Check your internet connection and run this file again.
pause
exit /b 1
