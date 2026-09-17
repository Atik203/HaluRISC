"""T3: lazy cross-encoder reranker.

Loaded once (HF cache), ~90 MB. Falls back to the fused order when the model
is unavailable (e.g., offline). Model configurable via HALU_RERANKER_MODEL.
"""

import os
import threading

RERANKER_MODEL = os.environ.get("HALU_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

_lock = threading.Lock()
_model = None


def _load():
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            from sentence_transformers import CrossEncoder

            _model = CrossEncoder(RERANKER_MODEL)
    return _model


def rerank(query: str, candidates: list, top_k: int) -> list:
    """Reorder candidates by cross-encoder relevance; returns top_k."""
    try:
        model = _load()
    except Exception:
        return candidates[:top_k]
    try:
        pairs = [[query, c["text"]] for c in candidates]
        scores = model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda t: float(t[1]), reverse=True)
        return [{**c, "score": float(s)} for c, s in ranked[:top_k]]
    except Exception:
        return candidates[:top_k]
