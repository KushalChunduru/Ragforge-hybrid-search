"""CLI entrypoint for Phase 3: retrieve (Phase 2) -> generate a grounded
answer with inline citations -> verify those citations.

Requires Ollama running locally with a model pulled:
    https://ollama.com
    ollama pull llama3.2
    ollama serve   (usually starts automatically after install)

Example:
    python scripts/answer.py --strategy structure_aware --query "What happens when a client exceeds the rate limit?"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, scoped_for_strategy
from src.embeddings import create_embedder
from src.generation import build_prompt, create_generator, verify_citations
from src.retrieval import Reranker, hybrid_retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="structure_aware", help="Which chunking-strategy index to query.")
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    base_config = load_config()
    config = scoped_for_strategy(base_config, args.strategy)

    print(f"Embedding provider: {config.embedding_provider} ({config.embedding_model})")
    embedder = create_embedder(config)
    reranker = Reranker(config)

    print(f"Retrieving for: {args.query!r} (index: {args.strategy})")
    sources = hybrid_retrieve(args.query, config, embedder, reranker)
    if not sources:
        raise SystemExit(
            f"No chunks retrieved for strategy {args.strategy!r}. Run scripts/ingest.py first."
        )

    print(f"\nSources handed to the LLM ({len(sources)}):")
    for i, s in enumerate(sources, start=1):
        heading = f" :: {s.chunk.section_heading}" if s.chunk.section_heading else ""
        print(f"  [{i}] {s.chunk.source_file}{heading}")

    prompt = build_prompt(args.query, sources)

    print(f"\nGenerating answer via {config.generation_provider} ({config.ollama_model})...")
    generator = create_generator(config)
    try:
        answer = generator.generate(prompt)
    except RuntimeError as exc:
        raise SystemExit(f"\n{exc}") from None

    print("\n=== Answer ===")
    print(answer)

    report = verify_citations(answer, sources)
    print("\n=== Citation verification ===")
    if report.is_clean:
        print("  clean: every claim is cited, every citation number is valid, all grounded in their source.")
    else:
        if report.invalid_citations:
            print(f"  INVALID citations (source number doesn't exist): {sorted(set(report.invalid_citations))}")
        if report.uncited_sentences:
            print(f"  UNCITED sentences ({len(report.uncited_sentences)}):")
            for s in report.uncited_sentences:
                print(f"    - {s}")
        if report.weak_grounding:
            print("  WEAK grounding (cited, but low word overlap with the claimed source -- check for hallucination):")
            for sentence, n, overlap in report.weak_grounding:
                print(f"    - [{n}] overlap={overlap:.2f} :: {sentence}")

    print(f"\n  citations used: {sorted(report.cited_source_numbers)} of {len(sources)} sources provided")


if __name__ == "__main__":
    main()
