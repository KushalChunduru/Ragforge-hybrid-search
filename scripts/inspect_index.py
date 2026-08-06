"""Sanity-check an index built by ingest.py: counts, a sample chunk, and a
manual BM25 keyword query (no embedding call needed) to confirm dense/sparse
are actually in sync.

Usage:
    python scripts/inspect_index.py --strategy structure_aware --query "rate limit"
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, scoped_for_strategy
from src.indexing import tokenize
from src.models import Chunk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="structure_aware")
    parser.add_argument("--query", default="rate limit")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    base_config = load_config()
    config = scoped_for_strategy(base_config, args.strategy)

    if not config.manifest_path.exists():
        raise SystemExit(f"No manifest at {config.manifest_path}. Run scripts/ingest.py first.")

    chunks = []
    with open(config.manifest_path, "r", encoding="utf-8") as f:
        import json

        for line in f:
            if line.strip():
                chunks.append(Chunk.from_dict(json.loads(line)))

    print(f"manifest: {len(chunks)} chunks")

    with open(config.bm25_index_path, "rb") as f:
        payload = pickle.load(f)
    bm25 = payload["bm25"]
    chunk_ids = payload["chunk_ids"]
    print(f"bm25 corpus size: {len(chunk_ids)} (in sync: {len(chunk_ids) == len(chunks)})")

    by_id = {c.chunk_id: c for c in chunks}
    scores = bm25.get_scores(tokenize(args.query))
    ranked = sorted(zip(chunk_ids, scores), key=lambda x: x[1], reverse=True)[: args.top_k]

    print(f"\nBM25 top {args.top_k} for query: {args.query!r}")
    for chunk_id, score in ranked:
        chunk = by_id[chunk_id]
        preview = chunk.text.replace("\n", " ")[:100]
        print(f"  [{score:.3f}] {chunk.source_file} :: {chunk.section_heading} :: {preview}...")


if __name__ == "__main__":
    main()
