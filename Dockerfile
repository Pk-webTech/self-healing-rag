FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer caching)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e "."

# Copy source
COPY . .

# Create data directories
RUN mkdir -p data/raw data/processed data/chroma_db data/logs

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]