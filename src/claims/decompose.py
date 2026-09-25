"""T2: claim-level decomposition of an answer into atomic claims.

Deterministic sentence/clause splitter (no LLM dependency in the core path):

  - sentences split on [.!?] boundaries
  - every sentence is refined into atomic clauses on conjunction boundaries
    (", and", ", but", ", while", ", whereas", ", which") and on , ; — :
    boundaries, then leading conjunctions are stripped from each clause
  - clauses shorter than 40 characters are merged back into the previous
    clause, so appositions and short lists stay intact
  - empty/whitespace claims dropped, duplicates removed, capped per answer

Atomic claims matter for verification: a compound claim ("X, and Y") cannot
be entailed by a single evidence sentence, and the cross-encoder reacts with a
false contradiction. Splitting conjunctions removes that failure mode.

Run (repo root):
  python -c "from src.claims.decompose import split_claims; print(split_claims('...'))"
"""

import re

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
CLAUSE_SPLIT = re.compile(r"(?:,|;|—|:)\s+")
CONJUNCTION_SPLIT = re.compile(r",\s+(?:and|but|while|whereas|which)\s+", re.IGNORECASE)
LEADING_CONJUNCTION = re.compile(r"^(?:and|but|while|whereas|which|so)\s+", re.IGNORECASE)

MIN_CLAUSE_CHARS = 40
MAX_CLAIM_CHARS = 500
MAX_CLAIMS = 12


def split_sentences(text: str) -> list:
    """Split prose into sentences (keeps trailing punctuation on the sentence)."""
    return [s.strip() for s in SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def _clauses_of(sentence: str) -> list:
    """Atomic clause refinement: conjunction split, punctuation clauses, min-length merge."""
    parts = CONJUNCTION_SPLIT.split(sentence)
    refined: list = []
    for part in parts:
        refined.extend(CLAUSE_SPLIT.split(part))
    parts = [LEADING_CONJUNCTION.sub("", p.strip()).strip() for p in refined]
    parts = [p for p in parts if p]
    # merge fragments that are too short to stand alone as claims
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
        for c in _clauses_of(sentence):
            claim = c[:MAX_CLAIM_CHARS].strip()
            if claim and claim not in out:
                out.append(claim)
    return out[:max_claims]
