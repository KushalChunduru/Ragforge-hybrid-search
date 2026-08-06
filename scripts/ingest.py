"""CLI entrypoint for Phase 1: load -> chunk -> embed -> index (Chroma + BM25).

Examples:
    python scripts/ingest.py --strategy structure_aware
    python scripts/ingest.py --strategy all --reset
    python scripts/ingest.py --strategy semantic --source processed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import STRATEGIES, chunk_document
from src.config import load_config, scoped_for_strategy
from src.embeddings import Embedder, create_embedder
from src.indexing import ChunkIndex
from src.loaders import iter_raw_documents
from src.models import Document


def load_documents(source: str, config) -> list[Document]:
    if source == "raw":
        documents = list(iter_raw_documents(config.raw_docs_dir))
        for doc in documents:
            path = doc.save(config.processed_docs_dir)
            print(f"  processed -> {path.relative_to(config.processed_docs_dir.parent)}")
        return documents

    # source == "processed"
    if not config.processed_docs_dir.exists():
        raise SystemExit(f"No processed docs found at {config.processed_docs_dir}. Run with --source raw first.")
    return [Document.load(p) for p in sorted(config.processed_docs_dir.glob("*.json"))]


def run_for_strategy(strategy: str, documents: list[Document], base_config, embedder: Embedder, reset: bool) -> None:
    config = scoped_for_strategy(base_config, strategy)
    index = ChunkIndex(config, embedder)

    if reset:
        index.reset()

    print(f"\n=== strategy: {strategy} ({STRATEGIES[strategy]}) ===")
    print(f"index: {config.chroma_collection}  manifest: {config.manifest_path.name}")

    total_added = 0
    total_dupes = 0
    for doc in documents:
        chunker_embedder = embedder if strategy == "semantic" else None
        chunks = chunk_document(doc, strategy, config, embedder=chunker_embedder)
        if not chunks:
            print(f"  {doc.doc_id}: 0 chunks (empty document?)")
            continue

        report = index.add_chunks(chunks)
        total_added += report.added_count
        total_dupes += report.duplicate_count

        dupe_note = f", {report.duplicate_count} duplicate(s) skipped" if report.duplicate_count else ""
        print(f"  {doc.doc_id}: {len(chunks)} chunks -> {report.added_count} indexed{dupe_note}")

        for dup in report.duplicates:
            print(
                f"    [dedup] {dup.chunk.chunk_id} ~ {dup.matched_chunk_id} "
                f"(cosine similarity {dup.similarity:.3f})"
            )

    stats = index.stats()
    print(f"  totals: {total_added} indexed, {total_dupes} duplicates skipped")
    print(f"  index stats: {stats}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        choices=[*STRATEGIES.keys(), "all"],
        default="structure_aware",
        help="Chunking strategy to index with. 'all' builds one isolated index per strategy for comparison.",
    )
    parser.add_argument(
        "--source",
        choices=["raw", "processed"],
        default="raw",
        help="Load from data/raw (re-parses files) or data/processed (re-uses normalized JSON, no re-upload needed).",
    )
    parser.add_argument("--reset", action="store_true", help="Clear the target index(es) before ingesting.")
    args = parser.parse_args()

    config = load_config()
    print(f"Embedding provider: {config.embedding_provider} ({config.embedding_model})")
    embedder = create_embedder(config)

    print(f"Loading documents from {args.source}...")
    documents = load_documents(args.source, config)
    print(f"Loaded {len(documents)} document(s).")

    strategies = list(STRATEGIES.keys()) if args.strategy == "all" else [args.strategy]
    for strategy in strategies:
        run_for_strategy(strategy, documents, config, embedder, args.reset)


if __name__ == "__main__":
    main()
