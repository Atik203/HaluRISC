"""T3: document chunking + file parsing.

Chunks are ~CHUNK_SIZE chars with OVERLAP chars of context; chunk boundaries
prefer sentence boundaries when available. PDF/DOCX/TXT parsing is lazy so
tests never need the optional packages.
"""

import re

CHUNK_SIZE = 1500
OVERLAP = 200
MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB per upload

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def extract_text_from_bytes(filename: str, data: bytes) -> str:
    """Parse a PDF/DOCX/TXT/MD upload into plain text."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        import io

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if lower.endswith(".docx"):
        import io

        from docx import Document

        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text)
    # txt / md / anything else: decode text
    return data.decode("utf-8", errors="replace")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list:
    """Fixed-size chunks preferring sentence boundaries; overlap for continuity."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    # pre-split into sentence tokens so boundaries are preferred
    sentences = [s for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
    chunks: list = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) <= chunk_size:
            current = f"{current} {sentence}".strip()
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail} {sentence}".strip()
        else:
            # single sentence longer than chunk_size: hard cut
            chunks.append(sentence[:chunk_size])
            current = sentence[chunk_size - overlap :] if overlap else ""
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]
