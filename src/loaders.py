"""Multi-format document loading. Normalizes markdown / text / html / pdf into
a Document made of (heading, text) Sections plus metadata.
"""
from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from .models import Document, Section

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf"}

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def make_doc_id(path: Path) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", path.stem).lower()


def load_markdown(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    matches = list(_MD_HEADING_RE.finditer(raw))

    sections: list[Section] = []
    title = None

    if not matches:
        sections.append(Section(heading=None, text=raw.strip()))
    else:
        if matches[0].start() > 0:
            preamble = raw[: matches[0].start()].strip()
            if preamble:
                sections.append(Section(heading=None, text=preamble))

        for i, m in enumerate(matches):
            heading = m.group(2).strip()
            if title is None and len(m.group(1)) == 1:
                title = heading
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
            body = raw[start:end].strip()
            sections.append(Section(heading=heading, text=f"{heading}\n{body}" if body else heading))

    return Document(
        doc_id=make_doc_id(path),
        source_path=str(path),
        file_format="markdown",
        sections=sections,
        title=title or path.stem,
    )


def load_text(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    return Document(
        doc_id=make_doc_id(path),
        source_path=str(path),
        file_format="text",
        sections=[Section(heading=None, text=raw.strip())],
        title=path.stem,
    )


def load_html(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else path.stem

    headings = soup.find_all(re.compile(r"^h[1-6]$"))
    sections: list[Section] = []

    if not headings:
        text = soup.get_text("\n", strip=True)
        sections.append(Section(heading=None, text=text))
    else:
        for h in headings:
            heading_text = h.get_text(strip=True)
            parts = []
            for sib in h.find_next_siblings():
                if re.match(r"^h[1-6]$", sib.name or ""):
                    break
                parts.append(sib.get_text("\n", strip=True))
            body = "\n".join(p for p in parts if p)
            sections.append(Section(heading=heading_text, text=f"{heading_text}\n{body}" if body else heading_text))

    return Document(
        doc_id=make_doc_id(path),
        source_path=str(path),
        file_format="html",
        sections=sections,
        title=title,
    )


def load_pdf(path: Path) -> Document:
    reader = PdfReader(str(path))
    sections: list[Section] = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append(Section(heading=None, text=text, page_number=i + 1))

    meta_title = None
    if reader.metadata and reader.metadata.title:
        meta_title = str(reader.metadata.title)

    return Document(
        doc_id=make_doc_id(path),
        source_path=str(path),
        file_format="pdf",
        sections=sections,
        title=meta_title or path.stem,
    )


_LOADERS = {
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_text,
    ".html": load_html,
    ".htm": load_html,
    ".pdf": load_pdf,
}


def load_document(path: Path) -> Document:
    ext = path.suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file extension: {ext} ({path})")
    return loader(path)


def iter_raw_documents(raw_dir: Path):
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield load_document(path)
