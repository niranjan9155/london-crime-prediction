#!/usr/bin/env bash
# Sets up the project (first time only) and opens the dashboard in the browser.
set -e
cd "$(dirname "$0")"

if [ ! -f .venv/setup_done.txt ]; then
    echo "Setting up the project for the first time. This takes a few minutes ..."
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python -m pip install -e .
    echo done > .venv/setup_done.txt
fi

echo "Opening the dashboard. Press Ctrl+C to stop it."
.venv/bin/python -m streamlit run dashboard.py
