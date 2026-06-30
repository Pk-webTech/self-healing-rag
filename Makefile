.PHONY: install run test lint format clean ingest demo

# ── Setup ─────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"
	cp -n .env.example .env || true
	mkdir -p data/raw data/processed data/chroma_db data/logs

# ── Run ───────────────────────────────────────────────────────
run:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

run-prod:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2

# ── Testing ───────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-cov:
	pytest tests/ --cov=. --cov-report=term-missing --cov-report=html

# ── Code quality ──────────────────────────────────────────────
lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy . --ignore-missing-imports

# ── Ingest ────────────────────────────────────────────────────
ingest:
	@echo "Usage: make ingest DIR=data/raw/"
	python -c "from ingestion import IngestionPipeline; \
	           p = IngestionPipeline(); \
	           import sys; p.ingest_directory('$(DIR)')"

# ── Docker ────────────────────────────────────────────────────
docker-build:
	docker build -t self-healing-rag:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

# ── Clean ─────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov

clean-data:
	rm -rf data/chroma_db/* data/logs/* data/processed/*
	@echo "⚠️  Vector store and logs cleared"