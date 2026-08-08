"""Environment-driven configuration for the ingestion pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path(env_key: str, default: str) -> Path:
    value = os.getenv(env_key, default)
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class Config:
    embedding_provider: str  # "local" | "openai"
    embedding_model: str
    openai_api_key: str

    chunk_size: int
    chunk_overlap: int
    dedup_similarity_threshold: float
    semantic_similarity_threshold: float
    semantic_max_chars: int

    chroma_persist_dir: Path
    chroma_collection: str
    manifest_path: Path
    bm25_index_path: Path

    raw_docs_dir: Path
    processed_docs_dir: Path

    dense_top_k: int
    sparse_top_k: int
    rrf_top_n: int
    final_top_k: int
    dense_weight: float
    sparse_weight: float
    rrf_k: int
    reranker_model: str

    generation_provider: str  # "ollama" (only one implemented so far)
    ollama_host: str
    ollama_model: str


_DEFAULT_MODEL_BY_PROVIDER = {
    "local": "BAAI/bge-small-en-v1.5",
    "openai": "text-embedding-3-small",
}


def load_config() -> Config:
    provider = os.getenv("EMBEDDING_PROVIDER", "local")
    default_model = _DEFAULT_MODEL_BY_PROVIDER.get(provider, "BAAI/bge-small-en-v1.5")

    return Config(
        embedding_provider=provider,
        embedding_model=os.getenv("EMBEDDING_MODEL") or default_model,
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        dedup_similarity_threshold=float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", "0.95")),
        semantic_similarity_threshold=float(os.getenv("SEMANTIC_CHUNK_SIMILARITY_THRESHOLD", "0.75")),
        semantic_max_chars=int(os.getenv("SEMANTIC_CHUNK_MAX_CHARS", "1500")),
        chroma_persist_dir=_path("CHROMA_PERSIST_DIR", "data/index/chroma"),
        chroma_collection=os.getenv("CHROMA_COLLECTION", "chunks"),
        manifest_path=_path("MANIFEST_PATH", "data/index/chunks_manifest.jsonl"),
        bm25_index_path=_path("BM25_INDEX_PATH", "data/index/bm25_index.pkl"),
        raw_docs_dir=_path("RAW_DOCS_DIR", "data/raw"),
        processed_docs_dir=_path("PROCESSED_DOCS_DIR", "data/processed"),
        dense_top_k=int(os.getenv("DENSE_TOP_K", "10")),
        sparse_top_k=int(os.getenv("SPARSE_TOP_K", "10")),
        rrf_top_n=int(os.getenv("RRF_TOP_N", "20")),
        final_top_k=int(os.getenv("FINAL_TOP_K", "5")),
        dense_weight=float(os.getenv("DENSE_WEIGHT", "0.7")),
        sparse_weight=float(os.getenv("SPARSE_WEIGHT", "0.3")),
        rrf_k=int(os.getenv("RRF_K", "60")),
        reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        generation_provider=os.getenv("GENERATION_PROVIDER", "ollama"),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2"),
    )


def scoped_for_strategy(config: Config, strategy: str) -> Config:
    """Return a Config pointing at an isolated index (own Chroma collection,
    manifest, and BM25 pickle) for a single chunking strategy.

    Dedup only makes sense *within* one coherent index: chunking the same
    document with two different strategies produces chunks that legitimately
    overlap a lot, and would falsely trip the near-duplicate filter if they
    shared an index. Keeping each strategy's index isolated lets you build
    and compare all three side by side without that collision, while
    `chunking_strategy` is still recorded on every chunk's metadata.
    """
    suffix = f"_{strategy}"
    return Config(
        embedding_provider=config.embedding_provider,
        embedding_model=config.embedding_model,
        openai_api_key=config.openai_api_key,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        dedup_similarity_threshold=config.dedup_similarity_threshold,
        semantic_similarity_threshold=config.semantic_similarity_threshold,
        semantic_max_chars=config.semantic_max_chars,
        chroma_persist_dir=config.chroma_persist_dir.parent / f"{config.chroma_persist_dir.name}{suffix}",
        chroma_collection=f"{config.chroma_collection}{suffix}",
        manifest_path=config.manifest_path.with_name(f"{config.manifest_path.stem}{suffix}{config.manifest_path.suffix}"),
        bm25_index_path=config.bm25_index_path.with_name(f"{config.bm25_index_path.stem}{suffix}{config.bm25_index_path.suffix}"),
        raw_docs_dir=config.raw_docs_dir,
        processed_docs_dir=config.processed_docs_dir,
        dense_top_k=config.dense_top_k,
        sparse_top_k=config.sparse_top_k,
        rrf_top_n=config.rrf_top_n,
        final_top_k=config.final_top_k,
        dense_weight=config.dense_weight,
        sparse_weight=config.sparse_weight,
        rrf_k=config.rrf_k,
        reranker_model=config.reranker_model,
        generation_provider=config.generation_provider,
        ollama_host=config.ollama_host,
        ollama_model=config.ollama_model,
    )
