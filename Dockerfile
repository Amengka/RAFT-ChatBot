FROM python:3.12-slim

WORKDIR /app

# Build deps for chromadb / sentence-transformers native wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Fail fast if Streamlit is not actually serving
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "chatbot-modified.py", "--server.port=8501", "--server.address=0.0.0.0"]
