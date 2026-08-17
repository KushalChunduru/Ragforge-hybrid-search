# Serves the FastAPI layer (Phase 2 hybrid retrieval + Phase 3 generation).
#
# Ollama and any index data are NOT bundled in this image on purpose:
# - Ollama is a separate long-running service, normally on the host or its
#   own container -- point OLLAMA_HOST at it (e.g. http://host.docker.internal:11434
#   on Docker Desktop for Windows/Mac).
# - The Chroma/BM25 index under data/index/ is built by scripts/ingest.py,
#   which is a deliberate offline step, not something that belongs in image
#   build or container startup. Mount data/index as a volume so the index
#   built via `docker run ... python scripts/ingest.py ...` persists across
#   container restarts, instead of baking a snapshot into the image.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
COPY data/raw/ data/raw/
COPY .env.example .env.example

# Local embedding/reranker models download to this cache on first use --
# mount it as a volume too if you want to avoid re-downloading on rebuilds.
ENV HF_HOME=/app/.cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
