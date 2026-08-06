"""CLI entrypoint for Phase 2: hybrid retrieval end to end, with every stage
printed so you can see dense vs. sparse vs. fused vs. reranked results.

Example:
    python scripts/query.py --strategy structure_aware --query "What happens when a client exceeds the rate limit?"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, scoped_for_strategy
from src.embeddings import create_embedder
from src.indexing import read_manifest
from src.retrieval import (
    Reranker,
    dense_retrieve,
    fuse_rankings,
    hydrate_and_rank,
    sparse_retrieve,
)


def preview(text: str, width: int = 90) -> str:
    text = text.replace("\n", " ")
    return text[:width] + ("..." if len(text) > width else "")


def cite(chunk) -> str:
    loc = f"{chunk.source_file}"
    if chunk.section_heading:
        loc += f" :: {chunk.section_heading}"
    if chunk.page_number is not None:
        loc += f" (p.{chunk.page_number})"
    return loc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="structure_aware", help="Which chunking-strategy index to query.")
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    base_config = load_config()
    config = scoped_for_strategy(base_config, args.strategy)

    manifest = {c.chunk_id: c for c in read_manifest(config.manifest_path)}
    if not manifest:
        raise SystemExit(f"No index found for strategy {args.strategy!r}. Run scripts/ingest.py first.")

    print(f"Embedding provider: {config.embedding_provider} ({config.embedding_model})")
    embedder = create_embedder(config)

    print(f"\nQuery: {args.query!r}  (index: {args.strategy}, {len(manifest)} chunks)\n")

    # --- Stage 1: dense ---
    dense_results = dense_retrieve(args.query, config, embedder)
    print(f"=== Dense retrieval (top {len(dense_results)}, cosine similarity) ===")
    for rank, (chunk_id, score) in enumerate(dense_results, start=1):
        c = manifest.get(chunk_id)
        if c:
            print(f"  {rank:2d}. [{score:.3f}] {cite(c)} :: {preview(c.text)}")

    # --- Stage 2: sparse ---
    sparse_results = sparse_retrieve(args.query, config)
    print(f"\n=== Sparse retrieval / BM25 (top {len(sparse_results)}, keyword score) ===")
    for rank, (chunk_id, score) in enumerate(sparse_results, start=1):
        c = manifest.get(chunk_id)
        if c:
            print(f"  {rank:2d}. [{score:.3f}] {cite(c)} :: {preview(c.text)}")

    # --- Stage 3: RRF fusion ---
    fused = fuse_rankings(dense_results, sparse_results, config.dense_weight, config.sparse_weight, config.rrf_k)
    ranked = hydrate_and_rank(fused, config)
    top_fused = ranked[: config.rrf_top_n]
    print(
        f"\n=== RRF fusion (dense_weight={config.dense_weight}, sparse_weight={config.sparse_weight}, "
        f"k={config.rrf_k}) -> top {len(top_fused)} ==="
    )
    for rank, entry in enumerate(top_fused, start=1):
        print(
            f"  {rank:2d}. [rrf={entry.rrf_score:.4f}] ({entry.matched_by:11s}) "
            f"{cite(entry.chunk)} :: {preview(entry.chunk.text)}"
        )

    # --- Stage 4: cross-encoder rerank ---
    print(f"\nLoading reranker ({config.reranker_model})...")
    reranker = Reranker(config)
    final = reranker.rerank(args.query, top_fused, top_k=config.final_top_k)

    print(f"\n=== Final answer set after reranking (top {len(final)}) ===")
    for rank, entry in enumerate(final, start=1):
        print(f"  {rank}. [rerank={entry.rerank_score:.3f}] {cite(entry.chunk)}")
        print(f"     {preview(entry.chunk.text, width=160)}")


if __name__ == "__main__":
    main()
