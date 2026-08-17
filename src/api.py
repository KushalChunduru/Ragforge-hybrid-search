"""FastAPI layer over Phase 2 (hybrid retrieval) and Phase 3 (grounded
generation). Ingestion stays a CLI/batch concern (`scripts/ingest.py`) --
building an index is a slow, deliberate operation you run offline, not
something an HTTP request should trigger, so there's no /ingest endpoint
here on purpose.

Run with:
    uvicorn src.api:app --reload
"""
from __future__ import annotations

import re
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import load_config, scoped_for_strategy
from .embeddings import create_embedder
from .generation import build_prompt, create_generator, verify_citations
from .indexing import read_manifest
from .retrieval import Reranker, ScoredChunk, hybrid_retrieve

_STRATEGY_RE = re.compile(r"^chunks_manifest_(.+)\.jsonl$")


def list_available_strategies() -> list[dict]:
    base_config = load_config()
    manifest_dir = base_config.manifest_path.parent
    if not manifest_dir.exists():
        return []

    strategies = []
    for path in sorted(manifest_dir.glob("chunks_manifest_*.jsonl")):
        match = _STRATEGY_RE.match(path.name)
        if not match:
            continue
        strategy = match.group(1)
        strategies.append({"strategy": strategy, "chunk_count": len(read_manifest(path))})
    return strategies


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    app.state.config = config
    app.state.embedder = create_embedder(config)
    app.state.reranker = Reranker(config)
    app.state.generator = create_generator(config)
    yield


app = FastAPI(title="RAG Hybrid Search API", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str
    strategy: str = "structure_aware"


class SourceResult(BaseModel):
    rank: int
    source_file: str
    section_heading: Optional[str] = None
    text: str
    matched_by: str
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


def _to_source_result(rank: int, scored: ScoredChunk) -> SourceResult:
    return SourceResult(
        rank=rank,
        source_file=scored.chunk.source_file,
        section_heading=scored.chunk.section_heading,
        text=scored.chunk.text,
        matched_by=scored.matched_by,
        dense_score=scored.dense_score,
        sparse_score=scored.sparse_score,
        rrf_score=scored.rrf_score,
        rerank_score=scored.rerank_score,
    )


class QueryResponse(BaseModel):
    query: str
    strategy: str
    results: list[SourceResult]


class CitationReportResponse(BaseModel):
    is_clean: bool
    cited_source_numbers: list[int]
    invalid_citations: list[int]
    uncited_sentences: list[str]
    weak_grounding: list[dict]


class AnswerResponse(BaseModel):
    query: str
    strategy: str
    answer: str
    sources: list[SourceResult]
    citation_report: CitationReportResponse


def _require_strategy_index(strategy: str) -> None:
    available = {s["strategy"] for s in list_available_strategies()}
    if strategy not in available:
        raise HTTPException(
            status_code=404,
            detail=f"No index for strategy {strategy!r}. Available: {sorted(available) or 'none -- run scripts/ingest.py first'}",
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/strategies")
def strategies() -> list[dict]:
    return list_available_strategies()


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    _require_strategy_index(request.strategy)
    base_config = app.state.config
    config = scoped_for_strategy(base_config, request.strategy)

    results = hybrid_retrieve(request.query, config, app.state.embedder, app.state.reranker)
    return QueryResponse(
        query=request.query,
        strategy=request.strategy,
        results=[_to_source_result(i, r) for i, r in enumerate(results, start=1)],
    )


@app.post("/answer", response_model=AnswerResponse)
def answer(request: QueryRequest) -> AnswerResponse:
    _require_strategy_index(request.strategy)
    base_config = app.state.config
    config = scoped_for_strategy(base_config, request.strategy)

    sources = hybrid_retrieve(request.query, config, app.state.embedder, app.state.reranker)
    if not sources:
        raise HTTPException(status_code=404, detail="No chunks retrieved for this query/strategy.")

    prompt = build_prompt(request.query, sources)
    try:
        generated = app.state.generator.generate(prompt)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None

    report = verify_citations(generated, sources)

    return AnswerResponse(
        query=request.query,
        strategy=request.strategy,
        answer=generated,
        sources=[_to_source_result(i, r) for i, r in enumerate(sources, start=1)],
        citation_report=CitationReportResponse(
            is_clean=report.is_clean,
            cited_source_numbers=sorted(report.cited_source_numbers),
            invalid_citations=sorted(set(report.invalid_citations)),
            uncited_sentences=report.uncited_sentences,
            weak_grounding=[
                {"sentence": s, "source_number": n, "overlap": round(o, 3)} for s, n, o in report.weak_grounding
            ],
        ),
    )
