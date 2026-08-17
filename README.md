# RAG Pipeline with Hybrid Search Over Internal Docs

Production-style RAG system: multi-format ingestion, three switchable chunking
strategies, dense (Chroma) + sparse (BM25) indexing kept in sync, cosine-
similarity dedup, hybrid retrieval (dense + BM25 + RRF fusion + cross-encoder
reranking), and grounded answer generation with inline citations + citation
verification, a FastAPI layer, and a Dockerfile for deployment. **Everything
in the original tech stack is built.**

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env
```

By default embeddings run **locally and free** via `sentence-transformers`
(`EMBEDDING_PROVIDER=local`, model `BAAI/bge-small-en-v1.5`) — no API key
needed, just a one-time model download (~130MB) on first run. To use OpenAI's
`text-embedding-3-small` instead, set `EMBEDDING_PROVIDER=openai` and fill in
`OPENAI_API_KEY` in `.env`.

## Generate sample docs (optional)

Five placeholder "internal docs" (markdown, txt, html) covering an API
reference, config reference, onboarding guide, incident runbook, and
deployment notes. They share a couple of copy-pasted paragraphs on purpose,
to exercise the dedup path, and contain specific error codes / config keys /
function names, to give BM25 something exact-match-able later.

```bash
python scripts/generate_sample_docs.py
```

## Ingest

```bash
# Single strategy (default: structure_aware)
python scripts/ingest.py --strategy structure_aware --reset

# Build all three strategies as separate, comparable indexes
python scripts/ingest.py --strategy all --reset

# Re-index from data/processed/ without re-parsing the raw files
python scripts/ingest.py --strategy structure_aware --source processed
```

Each strategy gets its **own** Chroma collection, manifest, and BM25 pickle
(suffixed `_fixed` / `_structure_aware` / `_semantic` under `data/index/`).
Dedup only makes sense within one coherent index — the same paragraph
chunked two different ways will legitimately look near-identical, and
merging strategies into one index would falsely trigger the duplicate
filter. `chunking_strategy` is still recorded on every chunk's metadata for
inspection.

## Inspect / verify

```bash
python scripts/inspect_index.py --strategy structure_aware --query "rate limit"
```

Prints manifest vs. BM25 corpus size (to confirm they're in sync) and the
top BM25 matches for a keyword query.

## Query (Phase 2: hybrid retrieval)

```bash
python scripts/query.py --strategy structure_aware --query "What happens when a client exceeds the rate limit?"
```

Runs, and prints the output of, every stage:

1. **Dense retrieval** — embeds the query, top `DENSE_TOP_K` (default 10) chunks from Chroma by cosine similarity.
2. **Sparse retrieval** — same query tokenized and scored against the BM25 corpus, top `SPARSE_TOP_K` (default 10). This is what catches an exact `ERR_RATE_LIMITED` or `gateway.rate_limit.rps` even when the wording doesn't semantically match.
3. **RRF fusion** — merges both ranked lists: `score(chunk) = dense_weight / (rrf_k + dense_rank) + sparse_weight / (rrf_k + sparse_rank)`, each term 0 if the chunk wasn't in that list. Weights default to 0.7 dense / 0.3 sparse (`DENSE_WEIGHT` / `SPARSE_WEIGHT` in `.env`); a chunk found by both lists outranks one found by only one. Top `RRF_TOP_N` (default 20) survive.
4. **Cross-encoder rerank** — `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, free, via `sentence-transformers`) scores each `(query, chunk_text)` pair jointly rather than comparing precomputed vectors, which is slower but substantially more precise. Final `FINAL_TOP_K` (default 5) become the answer set, each tagged with its source file and section heading for citation.

All the knobs (`*_TOP_K`, `*_WEIGHT`, `RRF_K`, `RERANKER_MODEL`) live in `.env` — see `.env.example`.

## Answer (Phase 3: grounded generation + citation verification)

Generation runs via **Ollama** — free, local, no API key, but it's a separate
install:

```bash
# one-time setup
# 1. install from https://ollama.com
# 2. pull a model (small enough to run on CPU):
ollama pull llama3.2
# 3. Ollama usually starts its server automatically after install;
#    if not: ollama serve
```

Then:

```bash
python scripts/answer.py --strategy structure_aware --query "What happens when a client exceeds the rate limit?"
```

This chains Phase 2's `hybrid_retrieve()` output straight into generation:

1. The final top-5 reranked chunks are numbered `[1]`..`[5]` and dropped into a prompt that instructs the model to answer **using only those sources** and cite every claim inline (`[1]`, `[1][3]`, etc.).
2. `verify_citations()` then parses the model's own answer and checks two failure modes that matter in a compliance-sensitive internal-docs setting, without a second LLM call:
   - **Invalid citations** — the model cited `[7]` but only 5 sources were given (an invented reference).
   - **Weak grounding** — a sentence cites `[2]`, but shares almost no vocabulary with source 2's actual text (word-overlap heuristic — cheap, not perfect, but catches the obvious cases).
   - **Uncited sentences** — a claim with no `[n]` marker at all (excluding hedges like "I don't have enough information").

The verification report prints alongside the answer so you can see exactly which claims are trustworthy.

To use a different LLM, edit `.env`: `GENERATION_PROVIDER` currently only implements `ollama`; `OpenAIGenerator`/`AnthropicGenerator` would slot in next to `OllamaGenerator` in `src/generation.py` behind the same `create_generator()` factory pattern used for embeddings.

## API

```bash
uvicorn src.api:app --reload
```

| Endpoint | Method | Body | Returns |
|---|---|---|---|
| `/health` | GET | — | `{"status": "ok"}` |
| `/strategies` | GET | — | Which chunking-strategy indexes exist and their chunk counts. |
| `/query` | POST | `{"query": "...", "strategy": "structure_aware"}` | Phase 2 hybrid-retrieval results with the full score breakdown (`dense_score`, `sparse_score`, `rrf_score`, `rerank_score`, `matched_by`) per chunk. |
| `/answer` | POST | `{"query": "...", "strategy": "structure_aware"}` | Phase 3 grounded answer plus its sources and citation-verification report. |

There's deliberately no `/ingest` endpoint — building an index is a slow,
offline batch job (`scripts/ingest.py`), not something an HTTP request should
trigger. Embedder and reranker models load once at process startup (FastAPI
`lifespan`), not per-request.

Verified working end to end against a live server: `/health`, `/strategies`,
`/query` (correct ranking and score breakdown), and `/answer` (correct
grounded answer, `"is_clean": true` citation report) all returned correctly.

## Docker

```bash
docker build -t ragforge-hybrid-search .
docker run -p 8000:8000 \
  -v "$(pwd)/data/index:/app/data/index" \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  ragforge-hybrid-search
```

Two things are intentionally **not** baked into the image:

- **Ollama** — a separate long-running service (normally on the host or its own container). Point `OLLAMA_HOST` at wherever it's actually running; `host.docker.internal` reaches the host from inside a container on Docker Desktop (Windows/Mac).
- **The index** (`data/index/`) — built by `scripts/ingest.py`, which you run once against a mounted volume so it persists across container restarts instead of being baked into an image layer:
  ```bash
  docker run --rm -v "$(pwd)/data/index:/app/data/index" ragforge-hybrid-search python scripts/ingest.py --strategy all --reset
  ```

(Docker build itself wasn't verified in this environment — Docker Desktop's
daemon wasn't running — but the image follows the same install/copy/run steps
already verified working outside a container.)

## How it fits together

```
data/raw/*.{md,txt,html,pdf}
        │  loaders.py (format-specific parsing -> Document/Section)
        ▼
data/processed/<doc_id>.json     (normalized text + metadata; re-indexable without re-upload)
        │  chunking.py (fixed | structure_aware | semantic)
        ▼
Chunk objects (text, source_file, section_heading, chunking_strategy, char_count, page_number)
        │  indexing.py
        ├── embeddings.py -> local bge-small-en-v1.5 (or OpenAI text-embedding-3-small)
        ├── dedup check: cosine similarity > 0.95 against existing Chroma entries -> skip
        ├── Chroma collection (dense)         data/index/chroma_<strategy>/
        ├── chunks_manifest_<strategy>.jsonl  (source of truth for BM25 rebuild)
        └── bm25_index_<strategy>.pkl (sparse)
```

## Module reference

| File | Responsibility |
|---|---|
| `src/config.py` | Env-driven config; `scoped_for_strategy` isolates per-strategy index paths. |
| `src/models.py` | `Document`, `Section`, `Chunk` dataclasses + JSON (de)serialization. |
| `src/loaders.py` | Markdown / text / HTML / PDF -> normalized `Document`. |
| `src/chunking.py` | Three strategies, all returning `list[Chunk]`. |
| `src/embeddings.py` | `create_embedder()` factory: local sentence-transformers (free) or OpenAI (retry/backoff), same interface either way; also used for semantic-chunking sentence similarity. |
| `src/indexing.py` | `ChunkIndex`: embed, dedup, write to Chroma, append manifest, rebuild BM25. Also exports `read_manifest()`, shared with retrieval. |
| `src/retrieval.py` | `dense_retrieve`, `sparse_retrieve`, `fuse_rankings` (RRF), `Reranker` (cross-encoder), `hybrid_retrieve` (wires all four). |
| `src/generation.py` | `build_prompt()`, `OllamaGenerator` / `create_generator()`, `verify_citations()`. |
| `src/api.py` | FastAPI app: `/health`, `/strategies`, `/query`, `/answer`. Loads embedder/reranker once at startup. |
| `scripts/ingest.py` | CLI: load -> chunk -> index. |
| `scripts/inspect_index.py` | Sanity-check counts and run a raw BM25 query. |
| `scripts/query.py` | CLI: full hybrid retrieval, printing dense / sparse / fused / reranked results at every stage. |
| `scripts/answer.py` | CLI: retrieve -> generate grounded answer -> citation verification report. |
| `scripts/generate_sample_docs.py` | Placeholder internal docs for testing. |
| `Dockerfile` / `.dockerignore` | Containerizes the API layer; Ollama and the index are external/mounted, not baked in. |
