"""Phase 2: hybrid retrieval.

    dense_retrieve()   -> Chroma top-k by cosine similarity
    sparse_retrieve()  -> BM25 top-k by keyword score
    fuse_rankings()    -> weighted Reciprocal Rank Fusion over both lists
    Reranker           -> cross-encoder re-scores the fused top-N against
                           the actual query, kept as a distinct second pass
    hybrid_retrieve()  -> wires all four steps together

Dense and sparse catch different things: dense finds paraphrases and
conceptually related chunks; sparse finds exact tokens (error codes, config
keys, function names) that an embedding can blur together. RRF fusion
before reranking means both signals get a vote before the (slower, more
precise) reranker narrows down to a final answer set.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Optional

import chromadb

from .config import Config
from .embeddings import Embedder
from .indexing import read_manifest, tokenize
from .models import Chunk


@dataclass
class ScoredChunk:
    chunk: Optional[Chunk] = None
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_rank: Optional[int] = None
    sparse_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None

    @property
    def matched_by(self) -> str:
        if self.dense_rank is not None and self.sparse_rank is not None:
            return "dense+sparse"
        if self.dense_rank is not None:
            return "dense"
        if self.sparse_rank is not None:
            return "sparse"
        return "none"


def dense_retrieve(query: str, config: Config, embedder: Embedder, top_k: Optional[int] = None) -> list[tuple[str, float]]:
    """Returns [(chunk_id, cosine_similarity)] ranked best first."""
    top_k = top_k or config.dense_top_k

    client = chromadb.PersistentClient(path=str(config.chroma_persist_dir))
    collection = client.get_or_create_collection(
        name=config.chroma_collection, metadata={"hnsw:space": "cosine"}
    )
    if collection.count() == 0:
        return []

    query_embedding = embedder.embed_one(query)
    result = collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)

    ids = (result.get("ids") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    # Cosine space in Chroma reports distance = 1 - cosine_similarity.
    return [(chunk_id, 1.0 - dist) for chunk_id, dist in zip(ids, distances)]


def sparse_retrieve(query: str, config: Config, top_k: Optional[int] = None) -> list[tuple[str, float]]:
    """Returns [(chunk_id, bm25_score)] ranked best first. Zero-score chunks
    (no keyword overlap at all) are dropped rather than padding the list.
    """
    top_k = top_k or config.sparse_top_k

    if not config.bm25_index_path.exists():
        return []
    with open(config.bm25_index_path, "rb") as f:
        payload = pickle.load(f)

    bm25 = payload.get("bm25")
    chunk_ids = payload.get("chunk_ids") or []
    if bm25 is None or not chunk_ids:
        return []

    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(zip(chunk_ids, scores), key=lambda x: x[1], reverse=True)
    return [(chunk_id, float(score)) for chunk_id, score in ranked[:top_k] if score > 0]


def fuse_rankings(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    dense_weight: float,
    sparse_weight: float,
    rrf_k: int,
) -> dict[str, ScoredChunk]:
    """Weighted Reciprocal Rank Fusion: each list contributes
    weight / (rrf_k + rank) to a chunk's score, rank being 1-indexed.
    A chunk found by both lists accumulates both contributions -- that's
    the whole fusion mechanism, no separate merge step needed.
    """
    scored: dict[str, ScoredChunk] = {}

    for rank, (chunk_id, score) in enumerate(dense_results, start=1):
        entry = scored.setdefault(chunk_id, ScoredChunk())
        entry.dense_rank = rank
        entry.dense_score = score

    for rank, (chunk_id, score) in enumerate(sparse_results, start=1):
        entry = scored.setdefault(chunk_id, ScoredChunk())
        entry.sparse_rank = rank
        entry.sparse_score = score

    for entry in scored.values():
        dense_component = dense_weight / (rrf_k + entry.dense_rank) if entry.dense_rank else 0.0
        sparse_component = sparse_weight / (rrf_k + entry.sparse_rank) if entry.sparse_rank else 0.0
        entry.rrf_score = dense_component + sparse_component

    return scored


def hydrate_and_rank(scored: dict[str, ScoredChunk], config: Config) -> list[ScoredChunk]:
    """Attach the full Chunk (text, metadata) to each fused result and sort
    by rrf_score descending. Chunk lookup uses the manifest, since Chroma
    query results only carry back what was asked for (ids/distances here).
    """
    manifest = {c.chunk_id: c for c in read_manifest(config.manifest_path)}

    hydrated = []
    for chunk_id, entry in scored.items():
        chunk = manifest.get(chunk_id)
        if chunk is None:
            continue
        entry.chunk = chunk
        hydrated.append(entry)

    hydrated.sort(key=lambda e: e.rrf_score or 0.0, reverse=True)
    return hydrated


class Reranker:
    """Cross-encoder second pass: scores (query, chunk_text) pairs directly,
    which is slower but far more precise than the bi-encoder similarity
    used for dense retrieval, since it lets the model attend across both
    texts jointly instead of comparing two independently-computed vectors.
    """

    def __init__(self, config: Config):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(config.reranker_model)

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []

        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._model.predict(pairs)
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)

        candidates.sort(key=lambda c: c.rerank_score, reverse=True)
        return candidates[:top_k]


def hybrid_retrieve(
    query: str,
    config: Config,
    embedder: Embedder,
    reranker: Reranker,
) -> list[ScoredChunk]:
    """Full Phase 2 pipeline: dense + sparse -> RRF fusion -> hydrate ->
    cross-encoder rerank -> final top-k.
    """
    dense_results = dense_retrieve(query, config, embedder, top_k=config.dense_top_k)
    sparse_results = sparse_retrieve(query, config, top_k=config.sparse_top_k)

    fused = fuse_rankings(dense_results, sparse_results, config.dense_weight, config.sparse_weight, config.rrf_k)
    ranked = hydrate_and_rank(fused, config)

    candidates = ranked[: config.rrf_top_n]
    return reranker.rerank(query, candidates, top_k=config.final_top_k)
