"""Phase 3: grounded answer generation with inline citations, plus a
lightweight citation-verification pass.

Retrieved sources are numbered [1..N] in the order Phase 2 (hybrid
retrieval + rerank) returned them. The LLM is instructed to cite every
claim with [n]. After generation, verify_citations() checks two common
grounding failures without needing a second LLM call:

  1. Invented source numbers (the model cites [7] when only 5 sources exist).
  2. Weak grounding (a sentence cites [2] but shares almost no vocabulary
     with source 2's actual text -- a cheap proxy for "this citation
     doesn't really support this claim").

It also flags sentences that make a claim with no citation at all, since
in a compliance-sensitive internal-docs setting an uncited assertion is
exactly the kind of thing a reviewer needs to catch.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import Config
from .retrieval import ScoredChunk

_CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_HEDGE_RE = re.compile(
    r"\b(don't have|do not have|not (enough|sufficient) information|cannot answer|no information|unable to)\b",
    re.IGNORECASE,
)

_SYSTEM_INSTRUCTIONS = (
    "You are a documentation assistant. Answer the question using ONLY the "
    "numbered sources below. Cite every factual claim inline with the "
    "bracketed source number(s) it came from, e.g. [1] or [1][3]. If the "
    "sources don't contain enough information to answer, say so explicitly "
    "instead of guessing."
)


def build_prompt(query: str, sources: list[ScoredChunk]) -> str:
    blocks = []
    for i, s in enumerate(sources, start=1):
        heading = f" — {s.chunk.section_heading}" if s.chunk.section_heading else ""
        blocks.append(f"[{i}] Source: {s.chunk.source_file}{heading}\n{s.chunk.text}")
    sources_block = "\n\n".join(blocks)

    return (
        f"{_SYSTEM_INSTRUCTIONS}\n\n"
        f"{sources_block}\n\n"
        f"Question: {query}\n\n"
        "Answer (with inline [n] citations):"
    )


class OllamaGenerator:
    """Calls a locally-running Ollama server. Free, no API key -- but
    requires Ollama installed and a model pulled (`ollama pull <model>`).
    """

    def __init__(self, config: Config):
        import requests

        self._requests = requests
        self._host = config.ollama_host.rstrip("/")
        self._model = config.ollama_model

    def generate(self, prompt: str) -> str:
        try:
            response = self._requests.post(
                f"{self._host}/api/generate",
                json={"model": self._model, "prompt": prompt, "stream": False},
                timeout=180,
            )
        except self._requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self._host}. Is it running? "
                "Install from https://ollama.com, then `ollama pull "
                f"{self._model}` and leave `ollama serve` running."
            ) from exc

        if response.status_code == 404:
            raise RuntimeError(
                f"Ollama doesn't have model '{self._model}' pulled yet. Run: ollama pull {self._model}"
            )
        response.raise_for_status()
        return response.json()["response"].strip()


def create_generator(config: Config):
    if config.generation_provider == "ollama":
        return OllamaGenerator(config)
    raise ValueError(
        f"Unknown GENERATION_PROVIDER {config.generation_provider!r}. Only 'ollama' is implemented so far."
    )


@dataclass
class CitationReport:
    cited_source_numbers: set[int] = field(default_factory=set)
    invalid_citations: list[int] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)
    weak_grounding: list[tuple[str, int, float]] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.invalid_citations and not self.uncited_sentences and not self.weak_grounding


def _token_overlap(a: str, b: str) -> float:
    """Fraction of a's tokens that also appear in b -- a cheap, dependency-free
    proxy for 'does this sentence actually draw on this source'.
    """
    tokens_a = set(re.findall(r"[a-z0-9]+", a.lower()))
    tokens_b = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


def verify_citations(answer: str, sources: list[ScoredChunk], overlap_threshold: float = 0.25) -> CitationReport:
    report = CitationReport()
    n_sources = len(sources)

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(answer.strip()) if s.strip()]

    for sentence in sentences:
        citation_numbers = [int(n) for n in _CITATION_RE.findall(sentence)]

        if not citation_numbers:
            if not _HEDGE_RE.search(sentence):
                report.uncited_sentences.append(sentence)
            continue

        for n in citation_numbers:
            if n < 1 or n > n_sources:
                report.invalid_citations.append(n)
                continue
            report.cited_source_numbers.add(n)
            overlap = _token_overlap(sentence, sources[n - 1].chunk.text)
            if overlap < overlap_threshold:
                report.weak_grounding.append((sentence, n, overlap))

    return report
