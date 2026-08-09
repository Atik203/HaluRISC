"""T2: claim-level decomposition of an answer into atomic claims.

Deterministic sentence/clause splitter (no LLM dependency in the core path):
  - sentences split on [.!?] boundaries
  - long sentences (>= 200 chars) are refined into clauses on , ; — :
    boundaries, keeping clauses >= 40 chars (short fragments stay merged)
  - empty/whitespace claims dropped, duplicates removed, capped per answer

Run (repo root):
  python -c "from src.claims.decompose import split_claims; print(split_claims('...'))"
"""

import re

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
CLAUSE_SPLIT = re.compile(r"(?:,|;|—|:)\s+")

MIN_CLAUSE_CHARS = 40
LONG_SENTENCE_CHARS = 200
MAX_CLAIM_CHARS = 500
MAX_CLAIMS = 12


def split_sentences(text: str) -> list:
    """Split prose into sentences (keeps trailing punctuation on the sentence)."""
    return [s.strip() for s in SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def _clauses_of(sentence: str) -> list:
    """Clause-level refinement for long sentences."""
    parts = [p.strip() for p in CLAUSE_SPLIT.split(sentence)]
    parts = [p for p in parts if p]
    # merge trailing fragments that are too short to stand alone
    merged: list = []
    for p in parts:
        if merged and len(p) < MIN_CLAUSE_CHARS:
            merged[-1] = f"{merged[-1]}, {p}"
        else:
            merged.append(p)
    return merged


def split_claims(text: str, max_claims: int = MAX_CLAIMS) -> list:
    """Split an answer into atomic claims (deterministic)."""
    out: list = []
    for sentence in split_sentences(text):
        if len(sentence) >= LONG_SENTENCE_CHARS:
            candidates = _clauses_of(sentence)
        else:
            candidates = [sentence]
        for c in candidates:
            claim = c[:MAX_CLAIM_CHARS].strip()
            if claim and claim not in out:
                out.append(claim)
    return out[:max_claims]
