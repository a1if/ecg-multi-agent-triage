# Hugging Face Space (free CPU hardware): API + frontend in one container, replay mode.
# The API listens on 127.0.0.1 only; the frontend is the one public port (7860). Built from the folder that
# backend/scripts/build_hf_space.py assembles (backend/ and frontend/ next to this file).
# Same base image and the same pinned PyTorch as backend/Dockerfile.api, the image the Gemma answers were recorded
# in: recorded answers are keyed by exact numbers, which match only within the same environment.
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    RECEIVER=replay DEMO_PRESETS=1 CLASSIFIER=gemma ALLOW_UPLOADS=0 RUNS_PER_MINUTE=6 QUESTIONS_PER_MINUTE=30 \
    RECORDS_DIR=/app/backend/data/records MODELS_DIR=/app/backend/artifacts/models \
    REPLAY_STORE=/app/backend/data/replay/answers.jsonl API_URL=http://127.0.0.1:8000

RUN pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
WORKDIR /app
COPY backend/pyproject.toml backend/
COPY backend/src backend/src
RUN pip install "./backend[api]"
COPY frontend/requirements.txt frontend/
RUN pip install -r frontend/requirements.txt
COPY backend/artifacts backend/artifacts
COPY backend/data backend/data
COPY frontend frontend
COPY --chmod=755 start.sh /app/start.sh

# Spaces run the container as uid 1000
RUN useradd -m -u 1000 user && chown -R user /app/backend/data
USER user
EXPOSE 7860
CMD ["/app/start.sh"]
