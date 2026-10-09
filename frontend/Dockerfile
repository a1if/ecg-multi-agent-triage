# Frontend (Streamlit): a thin client of the API. No ML libraries.
# Build:  docker build -t ecg-frontend frontend
# The base image is a build argument: Docker Hub by default; a mirror where Docker Hub is unreachable
# (e.g. BASE_IMAGE=mirror.gcr.io/library/python:3.12-slim, see docker-compose.yml).
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 API_URL=http://api:8000
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY app.py api.py ui.py ./
COPY views ./views
RUN useradd --create-home --uid 10001 app
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false"]
