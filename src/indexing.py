"""ChunkIndex: keeps a Chroma (dense) collection and a BM25 (sparse) index in
sync over the same set of chunks, with cosine-similarity dedup on insert.

Both indexes are derived from one source of truth: a JSONL manifest
(`chunks_manifest.jsonl`) written to disk. BM25 has no incremental update
API in rank_bm25, so on every ingest run it is rebuilt from the full
manifest -- guaranteeing it can never drift out of sync with Chroma.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from .config import Config
from .embeddings import Embedder
from .models import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def read_manifest(manifest_path: Path) -> list[Chunk]:
    """Read all chunks from a manifest JSONL file. Shared by ChunkIndex and
    by retrieval/inspection code that needs to hydrate chunk_ids -> Chunk.
    """
    import json

    if not manifest_path.exists():
        return []
    chunks = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(Chunk.from_dict(json.loads(line)))
    return chunks


@dataclass
class DuplicateMatch:
    chunk: Chunk
    similarity: float
    matched_chunk_id: str


@dataclass
class IngestReport:
    added: list[Chunk] = field(default_factory=list)
    duplicates: list[DuplicateMatch] = field(default_factory=list)

    @property
    def added_count(self) -> int:
        return len(self.added)

    @property
    def duplicate_count(self) -> int:
        return len(self.duplicates)


class ChunkIndex:
    def __init__(self, config: Config, embedder: Embedder):
        self.config = config
        self.embedder = embedder

        config.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        config.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(config.chroma_persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=config.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        try:
            self._client.delete_collection(self.config.chroma_collection)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.config.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        if self.config.manifest_path.exists():
            self.config.manifest_path.unlink()
        if self.config.bm25_index_path.exists():
            self.config.bm25_index_path.unlink()

    def add_chunks(self, chunks: list[Chunk]) -> IngestReport:
        report = IngestReport()
        if not chunks:
            return report

        embeddings = self.embedder.embed_batch([c.text for c in chunks])

        for chunk, embedding in zip(chunks, embeddings):
            duplicate = self._find_duplicate(chunk, embedding)
            if duplicate is not None:
                report.duplicates.append(duplicate)
                continue

            self._collection.add(
                ids=[chunk.chunk_id],
                embeddings=[embedding.tolist()],
                documents=[chunk.text],
                metadatas=[chunk.to_metadata()],
            )
            report.added.append(chunk)

        if report.added:
            self._append_manifest(report.added)
            self._rebuild_bm25()

        return report

    def _find_duplicate(self, chunk: Chunk, embedding) -> DuplicateMatch | None:
        if self._collection.count() == 0:
            return None

        result = self._collection.query(query_embeddings=[embedding.tolist()], n_results=1)
        ids = result.get("ids") or [[]]
        distances = result.get("distances") or [[]]
        if not ids[0] or not distances[0]:
            return None

        # Cosine space in Chroma reports distance = 1 - cosine_similarity.
        similarity = 1.0 - distances[0][0]
        if similarity > self.config.dedup_similarity_threshold:
            return DuplicateMatch(chunk=chunk, similarity=similarity, matched_chunk_id=ids[0][0])
        return None

    def _append_manifest(self, chunks: list[Chunk]) -> None:
        import json

        with open(self.config.manifest_path, "a", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict()) + "\n")

    def _read_manifest(self) -> list[Chunk]:
        return read_manifest(self.config.manifest_path)

    def _rebuild_bm25(self) -> None:
        chunks = self._read_manifest()
        corpus_ids = [c.chunk_id for c in chunks]
        tokenized_corpus = [tokenize(c.text) for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

        with open(self.config.bm25_index_path, "wb") as f:
            pickle.dump({"bm25": bm25, "chunk_ids": corpus_ids}, f)

    def stats(self) -> dict:
        manifest_count = len(self._read_manifest())
        return {
            "chroma_count": self._collection.count(),
            "manifest_count": manifest_count,
            "bm25_index_exists": self.config.bm25_index_path.exists(),
        }
