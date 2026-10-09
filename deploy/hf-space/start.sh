#!/bin/bash
# Two processes, one container. The API (private, 127.0.0.1) starts first; the frontend (the public port) starts
# once the API is ready, so a visitor arriving during a cold start never sees "API not reachable".
# If either process stops, stop the container so the platform restarts it.
cd /app/backend && uvicorn ecg_agent.api.main:app --host 127.0.0.1 --port 8000 --workers 1 &
for _ in $(seq 1 120); do
  python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz')" 2>/dev/null && break
  sleep 1
done
cd /app/frontend && streamlit run app.py --server.port "${PORT:-7860}" --server.address 0.0.0.0 \
    --server.headless true --browser.gatherUsageStats false &
wait -n
exit 1
