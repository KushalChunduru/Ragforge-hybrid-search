"""Embedding clients: a free local backend (sentence-transformers) and an
OpenAI backend, both behind the same embed_batch/embed_one interface so
chunking.py and indexing.py don't care which one is in use.

Used both for chunk-level embeddings (indexing) and sentence-level
embeddings (semantic chunking boundary detection).
"""
from __future__ import annotations

import time
from typing import Protocol

import numpy as np

from .config import Config

_MAX_RETRIES = 3
_BACKOFF_SECONDS = 2.0


class Embedder(Protocol):
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]: ...
    def embed_one(self, text: str) -> np.ndarray: ...


class LocalEmbeddingClient:
    """Runs a sentence-transformers model on-device. Free, no API key, no
    network call at embed time (aside from the one-time model download).
    """

    def __init__(self, config: Config, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(config.embedding_model)
        self._batch_size = batch_size

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [np.asarray(v, dtype=np.float32) for v in vectors]

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]


class OpenAIEmbeddingClient:
    def __init__(self, config: Config, batch_size: int = 100):
        from openai import OpenAI

        if not config.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in, "
                "or set EMBEDDING_PROVIDER=local to use a free local model instead."
            )
        self._client = OpenAI(api_key=config.openai_api_key)
        self._model = config.embedding_model
        self._batch_size = batch_size

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []

        vectors: list[np.ndarray] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            vectors.extend(self._embed_with_retry(batch))
        return vectors

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def _embed_with_retry(self, batch: list[str]) -> list[np.ndarray]:
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.embeddings.create(model=self._model, input=batch)
                return [np.array(item.embedding, dtype=np.float32) for item in response.data]
            except Exception as exc:  # openai raises various transient errors here
                last_error = exc
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_BACKOFF_SECONDS * (2**attempt))
        raise RuntimeError(f"Embedding request failed after {_MAX_RETRIES} attempts") from last_error


def create_embedder(config: Config) -> Embedder:
    if config.embedding_provider == "local":
        return LocalEmbeddingClient(config)
    if config.embedding_provider == "openai":
        return OpenAIEmbeddingClient(config)
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER {config.embedding_provider!r}. Use 'local' or 'openai'."
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
    return float(np.dot(a, b) / denom)
