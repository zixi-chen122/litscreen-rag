FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # All state lives on the /data volume, so it survives container upgrades.
    LITSCREEN_DB_PATH=/data/litscreen.db \
    CHROMA_PERSIST_DIR=/data/chroma \
    REVIEW_QUEUE_PATH=/data/review_queue.jsonl \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=50

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN useradd --create-home --uid 1000 litscreen \
    && mkdir /data && chown litscreen:litscreen /data
COPY --chown=litscreen:litscreen . .
USER litscreen

VOLUME /data
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
