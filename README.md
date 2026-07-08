# Self-Healing RAG

> **Adaptive Retrieval-Augmented Generation with autonomous quality control, self-healing loops, and full observability.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-77%20passing-brightgreen.svg)]()

---

## What It Does

Self-Healing RAG is a production-grade RAG pipeline that doesn't just retrieve and generate — it **evaluates its own answers, heals failures automatically, and adapts over time**.

When a query comes in:

1. **Retrieves** relevant chunks using hybrid dense + BM25 search
2. **Generates** a grounded answer via GPT-4o / Claude / Llama
3. **Evaluates** the answer using LLM-as-judge (faithfulness, relevance, grounding)
4. **Heals** poor answers automatically — expands queries, re-retrieves, quarantines bad chunks
5. **Adapts** — updates chunk quality scores, tunes retrieval parameters, refreshes few-shot examples
6. **Observes** — Prometheus metrics, structured traces, threshold-based alerts

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                       │
│  DocLoader → Chunker → Embedder → MetadataTagger → VectorStore  │
└───────────────────────────────┬─────────────────────────────────┘
                                │ chunks + embeddings
┌───────────────────────────────▼─────────────────────────────────┐
│                         QUERY ENGINE                            │
│  QueryProcessor → HybridRetriever → Reranker → ContextBuilder   │
│         (HyDE / Multi-Query)   (BM25 + Dense ANN)  (CE model)  │
└───────────────────────────────┬─────────────────────────────────┘
                                │ ranked context
┌───────────────────────────────▼─────────────────────────────────┐
│                        LLM GENERATOR                            │
│         GPT-4o / Claude 3.5 / Llama 3.1 (via Ollama)           │
└───────────────────────────────┬─────────────────────────────────┘
                                │ answer
┌───────────────────────────────▼─────────────────────────────────┐
│                    SELF-HEALING ENGINE                          │
│  FaithfulnessJudge + RelevanceJudge + GroundingJudge            │
│        → Verdict Aggregator (PASS / SOFT_FAIL / HARD_FAIL)      │
│        → Healing Dispatcher → Actions (expand / re-retrieve /   │
│                                        re-embed / quarantine)   │
└───────────────────────────────┬─────────────────────────────────┘
                                │ healed result
┌───────────────────────────────▼─────────────────────────────────┐
│                     ADAPTIVE LEARNING                           │
│  QualityTracker (EWM) → ChunkEvolver → RetrievalTuner →        │
│  PromptOptimizer (few-shot selection)                           │
└───────────────────────────────┬─────────────────────────────────┘
                                │ telemetry
┌───────────────────────────────▼─────────────────────────────────┐
│                       OBSERVABILITY                             │
│  Prometheus Metrics · JSONL Traces · Alert Engine               │
│  /metrics · /traces · /alerts/history · /alerts/window-stats    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
self-healing-rag/
├── core/                    # Shared config, models, logger, LLM client
├── ingestion/               # Document loading, chunking, embedding, vector store
├── retrieval/               # Hybrid retrieval, re-ranking, context building
├── generation/              # LLM generation with structured prompt templates
├── evaluation/              # LLM-as-judge (faithfulness, relevance, grounding)
│   └── judges/
├── healing/                 # Self-healing loop, dispatcher, actions, DB logging
├── adaptation/              # EWM quality tracker, chunk evolver, retrieval tuner
├── observability/           # Prometheus metrics, JSONL tracer, alert engine
├── api/
│   └── routers/             # FastAPI routers: query, ingest, heal, adapt, observability
├── db/                      # SQLAlchemy ORM models, async session factory
├── tests/
│   ├── unit/                # 57 unit tests
│   └── integration/         # 20 integration tests
├── configs/
│   ├── config.yaml          # All tunable parameters
│   └── prompts.yaml         # Externalised prompt templates
├── data/                    # Raw docs, vector store, logs (git-ignored)
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```
---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-username/self-healing-rag.git
cd self-healing-rag
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` and set at least one LLM provider key:

```env
# For OpenAI (default)
OPENAI_API_KEY=sk-...

# OR for Anthropic
ANTHROPIC_API_KEY=sk-ant-...
GENERATION_PROVIDER=anthropic
EMBEDDING_PROVIDER=openai   # OpenAI embeddings work with Anthropic generation

# OR for fully local (no API key needed)
GENERATION_PROVIDER=ollama
EMBEDDING_PROVIDER=huggingface
OLLAMA_MODEL=llama3.1:8b
```

### 3. Start the API

```bash
make run
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

### 4. Ingest documents

```bash
# Upload a PDF
curl -X POST http://localhost:8000/ingest/files \
  -F "files=@your_document.pdf"

# Ingest a URL
curl -X POST http://localhost:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'

# Ingest raw text
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Your content here...", "source": "manual"}'
```

### 5. Query with self-healing

```bash
# Basic query (no evaluation)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is self-healing RAG?"}'

# Query with full evaluation + auto-healing
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is self-healing RAG?", "evaluate": true}'
```

---

## All API Endpoints

| Method   | Endpoint                    | Description                             |
| -------- | --------------------------- | --------------------------------------- |
| `POST`   | `/query`                    | Full RAG pipeline with optional healing |
| `POST`   | `/ingest/files`             | Upload PDF/TXT/MD/HTML files            |
| `POST`   | `/ingest/url`               | Fetch and ingest a URL                  |
| `POST`   | `/ingest/text`              | Ingest raw text                         |
| `GET`    | `/ingest/stats`             | Vector store statistics                 |
| `DELETE` | `/ingest/reset`             | Clear the vector store                  |
| `GET`    | `/heal/events`              | List recent heal events                 |
| `GET`    | `/heal/query-logs`          | Query history with verdicts             |
| `GET`    | `/heal/stats`               | Healing statistics                      |
| `POST`   | `/adapt/run`                | Manually trigger adaptation cycle       |
| `GET`    | `/adapt/stats`              | Chunk quality history stats             |
| `GET`    | `/adapt/chunk-history/{id}` | Quality timeline for a chunk            |
| `GET`    | `/adapt/flagged-chunks`     | Currently flagged chunks                |
| `GET`    | `/metrics`                  | Prometheus metrics scrape endpoint      |
| `GET`    | `/traces`                   | Recent request traces                   |
| `GET`    | `/alerts/history`           | Fired alerts                            |
| `GET`    | `/alerts/window-stats`      | Live rolling window stats               |
| `GET`    | `/health`                   | Health check                            |

---

## Configuration Reference

All parameters live in `configs/config.yaml`. Key settings:

```yaml
ingestion:
  chunk_size: 512 # tokens per chunk
  chunk_overlap: 64 # overlap between chunks
  chunking_strategy: recursive # recursive | semantic

embedding:
  provider: openai # openai | huggingface
  openai_model: text-embedding-3-small

retrieval:
  k: 10 # candidates to retrieve
  final_k: 4 # top-k after re-ranking
  alpha: 0.7 # dense weight in BM25+dense fusion
  use_reranker: true

generation:
  provider: openai # openai | anthropic | ollama
  temperature: 0.2
  max_tokens: 1024

evaluation:
  faithfulness_threshold: 0.75
  relevance_threshold: 0.70
  grounding_threshold: 0.65
  max_heal_rounds: 3

adaptation:
  ewm_alpha: 0.3 # EWM smoothing factor
  drop_threshold: 0.25 # drop chunks below this quality
  tuner_min_samples: 20 # queries needed before tuning

observability:
  metrics_enabled: true
  alerting_enabled: true
  alert_heal_rate_threshold: 0.5
  alert_latency_p95_ms: 5000
```

---

## Running with Ollama (fully local, no API keys)

```bash
# 1. Install Ollama: https://ollama.ai
ollama pull llama3.1:8b

# 2. Configure .env
echo "GENERATION_PROVIDER=ollama" >> .env
echo "EMBEDDING_PROVIDER=huggingface" >> .env
echo "OLLAMA_MODEL=llama3.1:8b" >> .env

# 3. Run
make run
```

---

## Docker

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## Development

```bash
# Run all tests
make test

# Unit tests only
make test-unit

# Integration tests only
make test-integration

# With coverage report
make test-cov

# Lint and format
make lint
make format

# Clean build artefacts
make clean
```

---

## Monitoring

**Prometheus** — scrape `/metrics`:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: self-healing-rag
    static_configs:
      - targets: ["localhost:8000"]
```

**Key metrics:**

| Metric                          | Type      | Description                    |
| ------------------------------- | --------- | ------------------------------ |
| `shr_queries_total`             | Counter   | Total queries by verdict label |
| `shr_query_latency_ms`          | Histogram | End-to-end latency             |
| `shr_heal_events_total`         | Counter   | Healing actions by type        |
| `shr_weighted_score_latest`     | Gauge     | Latest evaluation score        |
| `shr_vector_store_chunks_total` | Gauge     | Chunks in vector store         |
| `shr_tokens_total`              | Counter   | LLM tokens by type             |

**Alerting** — configure in `config.yaml`:

```yaml
observability:
  alert_backend: webhook # log | webhook | slack
  alert_webhook_url: https://...
  alert_score_drop_threshold: 0.15
  alert_heal_rate_threshold: 0.5
```

---

## Tech Stack

| Layer         | Technology                                            |
| ------------- | ----------------------------------------------------- |
| API           | FastAPI 0.111+, Pydantic v2, uvicorn                  |
| LLM           | OpenAI GPT-4o, Anthropic Claude 3.5, Ollama (local)   |
| Embeddings    | OpenAI text-embedding-3-small, HuggingFace BGE        |
| Vector Store  | ChromaDB (persistent), FAISS (in-memory)              |
| Retrieval     | BM25 (rank-bm25) + dense ANN, cross-encoder reranking |
| Database      | SQLAlchemy 2.x async, aiosqlite, SQLite/PostgreSQL    |
| Observability | prometheus-client, JSONL traces, custom alert engine  |
| Testing       | pytest, pytest-asyncio, unittest.mock                 |

---

## License

MIT — see [LICENSE](LICENSE).

---

_Built as a 5-phase research and engineering project demonstrating production-grade RAG system design with autonomous quality control._
