#!/usr/bin/env bash
# Starts the API and the chat page together inside one container.
set -e

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Wait until the API answers before opening the page.
until python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" 2>/dev/null; do
  sleep 1
done

exec python -m streamlit run ui/streamlit_app.py \
  --server.port "${PORT:-8501}" --server.address 0.0.0.0 \
  --browser.gatherUsageStats false
