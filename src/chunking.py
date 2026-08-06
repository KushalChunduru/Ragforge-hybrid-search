"""Three switchable chunking strategies, all producing a common Chunk shape.

- fixed:            baseline fixed-size character splitting with overlap, structure-blind.
- structure_aware:  splits on the document's own section headings first, and only
                     falls back to recursive character splitting inside sections
                     that are still too large.
- semantic:         splits within each section at sentence boundaries where embedding
                     similarity between consecutive sentences drops, i.e. topic shifts.
"""
from __future__ import annotations

import re
from typing import Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Config
from .embeddings import Embedder, cosine_similarity
from .models import Chunk, Document

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def fixed_size_chunking(document: Document, config: Config) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    pieces = splitter.split_text(document.full_text)

    chunks = []
    for i, text in enumerate(pieces):
        text = text.strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                chunk_id=Chunk.make_id(document.doc_id, "fixed", i, text),
                document_id=document.doc_id,
                source_file=document.source_path,
                section_heading=None,
                chunking_strategy="fixed",
                chunk_index=i,
                text=text,
            )
        )
    return chunks


def structure_aware_chunking(document: Document, config: Config) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    chunks = []
    global_index = 0
    for section in document.sections:
        text = section.text.strip()
        if not text:
            continue

        pieces = [text] if len(text) <= config.chunk_size else splitter.split_text(text)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    chunk_id=Chunk.make_id(document.doc_id, "structure_aware", global_index, piece),
                    document_id=document.doc_id,
                    source_file=document.source_path,
                    section_heading=section.heading,
                    chunking_strategy="structure_aware",
                    chunk_index=global_index,
                    text=piece,
                    page_number=section.page_number,
                )
            )
            global_index += 1
    return chunks


def semantic_chunking(document: Document, config: Config, embedder: Embedder) -> list[Chunk]:
    chunks = []
    global_index = 0

    for section in document.sections:
        sentences = _split_sentences(section.text)
        if not sentences:
            continue

        if len(sentences) == 1:
            groups = [sentences]
        else:
            embeddings = embedder.embed_batch(sentences)
            groups = []
            current_group = [sentences[0]]
            current_chars = len(sentences[0])
            for i in range(1, len(sentences)):
                sentence = sentences[i]
                similarity = cosine_similarity(embeddings[i - 1], embeddings[i])
                would_exceed = current_chars + len(sentence) > config.semantic_max_chars
                if similarity < config.semantic_similarity_threshold or would_exceed:
                    groups.append(current_group)
                    current_group = [sentence]
                    current_chars = len(sentence)
                else:
                    current_group.append(sentence)
                    current_chars += len(sentence)
            groups.append(current_group)

        for group in groups:
            text = " ".join(group).strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    chunk_id=Chunk.make_id(document.doc_id, "semantic", global_index, text),
                    document_id=document.doc_id,
                    source_file=document.source_path,
                    section_heading=section.heading,
                    chunking_strategy="semantic",
                    chunk_index=global_index,
                    text=text,
                    page_number=section.page_number,
                )
            )
            global_index += 1
    return chunks


STRATEGIES: dict[str, str] = {
    "fixed": "Fixed-size character splitting with overlap (baseline)",
    "structure_aware": "Recursive splitting anchored to section headings",
    "semantic": "Embedding-similarity sentence grouping by topic boundary",
}


def get_chunker(strategy: str) -> Callable[..., list[Chunk]]:
    if strategy == "fixed":
        return fixed_size_chunking
    if strategy == "structure_aware":
        return structure_aware_chunking
    if strategy == "semantic":
        return semantic_chunking
    raise ValueError(f"Unknown chunking strategy: {strategy!r}. Choose from {list(STRATEGIES)}")


def chunk_document(document: Document, strategy: str, config: Config, embedder: Embedder | None = None) -> list[Chunk]:
    if strategy == "semantic":
        if embedder is None:
            raise ValueError("semantic chunking requires an Embedder")
        return semantic_chunking(document, config, embedder)
    chunker = get_chunker(strategy)
    return chunker(document, config)
