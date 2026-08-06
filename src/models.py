"""Core data models shared across loading, chunking, and indexing."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class Section:
    """A logical section of a document (e.g. content under one markdown heading, or one PDF page)."""

    heading: Optional[str]
    text: str
    page_number: Optional[int] = None


@dataclass
class Document:
    """A normalized, format-agnostic representation of an ingested file."""

    doc_id: str
    source_path: str
    file_format: str  # "markdown" | "text" | "html" | "pdf"
    sections: list[Section] = field(default_factory=list)
    title: Optional[str] = None
    ingested_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.sections if s.text.strip())

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        sections = [Section(**s) for s in d.get("sections", [])]
        return cls(
            doc_id=d["doc_id"],
            source_path=d["source_path"],
            file_format=d["file_format"],
            sections=sections,
            title=d.get("title"),
            ingested_at=d.get("ingested_at", datetime.now(timezone.utc).isoformat()),
        )

    def save(self, processed_dir: Path) -> Path:
        processed_dir.mkdir(parents=True, exist_ok=True)
        out_path = processed_dir / f"{self.doc_id}.json"
        out_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out_path

    @classmethod
    def load(cls, path: Path) -> "Document":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass
class Chunk:
    """A single retrievable unit produced by a chunking strategy."""

    chunk_id: str
    document_id: str
    source_file: str
    section_heading: Optional[str]
    chunking_strategy: str  # "fixed" | "structure_aware" | "semantic"
    chunk_index: int
    text: str
    char_count: int = 0
    page_number: Optional[int] = None

    def __post_init__(self):
        if not self.char_count:
            self.char_count = len(self.text)

    @staticmethod
    def make_id(document_id: str, strategy: str, chunk_index: int, text: str) -> str:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        return f"{document_id}::{strategy}::{chunk_index}::{digest}"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_metadata(self) -> dict:
        """Metadata payload stored alongside the embedding in the vector store."""
        return {
            "document_id": self.document_id,
            "source_file": self.source_file,
            "section_heading": self.section_heading or "",
            "chunking_strategy": self.chunking_strategy,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
            "page_number": self.page_number if self.page_number is not None else -1,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(
            chunk_id=d["chunk_id"],
            document_id=d["document_id"],
            source_file=d["source_file"],
            section_heading=d.get("section_heading"),
            chunking_strategy=d["chunking_strategy"],
            chunk_index=d["chunk_index"],
            text=d["text"],
            char_count=d.get("char_count", 0),
            page_number=d.get("page_number"),
        )
