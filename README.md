# Self-Healing RAG

> Adaptive Retrieval-Augmented Generation with autonomous quality control and self-healing loops.

---

## Architecture

```
Ingest → Chunk → Embed → VectorStore
                  ↓
Query → Hybrid Retrieve (BM25 + Dense) → Re-rank → Context Builder
                  ↓
            LLM Generator
                  ↓
         [Phase 2] RAGAS Judges → Verdict
                  ↓
         [Phase 3] Healing Dispatcher → Actions
                  ↓
         [Phase 4] Adaptive Learning (Optuna / DSPy)
                  ↓
         [Phase 5] Observability (Prometheus / Streamlit)
```

---

## Phases

| Phase | Scope                                                    | Status      |
| ----- | -------------------------------------------------------- | ----------- |
| 1     | Foundation Pipeline (Ingest + Retrieve + Generate + API) | ✅ Complete |
| 2     | Hallucination Detector (RAGAS judges + Verdict)          | 🔜 Next     |
| 3     | Self-Healing Loop (Dispatcher + Actions)                 | 🔜          |
| 4     | Adaptive Learning (Chunk Evolver + HPO)                  | 🔜          |
| 5     | Observability (Metrics + Dashboard)                      | 🔜          |

---

## Quick Start

### 1. Install

```bash
make install
cp .env.example .env
# Edit .env — add OPENAI_API_KEY (or set GENERATION_PROVIDER=ollama)
```

### 2. Start API

```bash
make run
# → http://localhost:8000/docs
```

### 3. Ingest documents

```bash
# Via API
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "RAG systems combine retrieval with generation...", "source": "intro"}'

# Or upload files
curl -X POST http://localhost:8000/ingest/files \
  -F "files=@docs/paper.pdf"
```

### 4. Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is self-healing RAG?"}'
```

---

## Configuration

All tuning parameters live in `configs/config.yaml`:

| Key                       | Default | Description                       |
| ------------------------- | ------- | --------------------------------- |
| `ingestion.chunk_size`    | 512     | Tokens per chunk                  |
| `ingestion.chunk_overlap` | 64      | Overlap between chunks            |
| `retrieval.k`             | 10      | Candidates retrieved              |
| `retrieval.final_k`       | 4       | Top-k after re-ranking            |
| `retrieval.alpha`         | 0.7     | Dense weight in RRF fusion        |
| `generation.provider`     | openai  | `openai` / `anthropic` / `ollama` |

---

## Testing

```bash
make test          # all tests
make test-unit     # unit only
make test-cov      # with coverage report
```

---

## Project Structure

```
self-healing-rag/
├── core/           # config, logger, shared models
├── ingestion/      # loader → chunker → embedder → metadata → vector_store
├── retrieval/      # query_processor → retriever → reranker → context_builder
├── generation/     # prompt_templates → generator
├── api/            # FastAPI routers, schemas, deps
├── tests/          # unit + integration
├── configs/        # config.yaml, prompts.yaml
└── data/           # raw docs, chroma_db, logs
```
