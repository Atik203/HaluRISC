"""
Semantic similarity feature extraction for HaluRISC.

Group 7 features (roadmap §6):
  cosine_ctx_ans, cosine_q_ans

Uses sentence-transformers `all-MiniLM-L6-v2` embeddings.
Empty context -> cosine_ctx_ans = 0.0 (no evidence available).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _safe_device(device: Optional[str]) -> Optional[str]:
    """Return the device to use; cuda is only honored when torch supports it."""
    if device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        logger.warning("device=cuda requested but torch has no CUDA support; using CPU")
        return None
    return device


def load_embedding_model(device: Optional[str] = None, model_kwargs: Optional[dict] = None):
    """Load the SBERT embedding model (lazy, cached at call site).

    device=None -> library default (GPU if available); set "cpu" for stability.
    model_kwargs -> extra kwargs for the model loader (e.g. torch_dtype=float16).
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL, device=_safe_device(device), model_kwargs=model_kwargs)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def extract_semantic_features(question: str, context: str, answer: str, model) -> dict:
    """Group 7: cosine similarity between answer and context/question."""
    texts = [answer]
    if context.strip():
        texts.append(context)
    texts.append(question)
    vecs = model.encode(texts, convert_to_numpy=True)

    ans_vec = vecs[0]
    offset = 1
    cosine_ctx_ans = _cosine(ans_vec, vecs[offset]) if context.strip() else 0.0
    if context.strip():
        offset += 1
    cosine_q_ans = _cosine(ans_vec, vecs[offset])

    return {
        "cosine_ctx_ans": round(cosine_ctx_ans, 6),
        "cosine_q_ans": round(cosine_q_ans, 6),
    }


def extract_semantic_features_df(df: pd.DataFrame, model, batch_size: int = 64) -> pd.DataFrame:
    """Batch semantic features; encodes each column once and vectorizes."""
    logger.info(f"Extracting semantic (SBERT) features for {len(df)} samples...")
    answers = df["answer"].astype(str).tolist()
    contexts = df["context"].astype(str).tolist()
    questions = df["question"].astype(str).tolist()

    ans_vecs = model.encode(answers, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
    q_vecs = model.encode(questions, batch_size=batch_size, convert_to_numpy=True)
    ctx_vecs = (
        model.encode([c for c in contexts if c.strip()], batch_size=batch_size, convert_to_numpy=True)
        if any(c.strip() for c in contexts)
        else np.zeros((0, ans_vecs.shape[1]))
    )

    ctx_iter = iter(ctx_vecs)
    rows = []
    for i, c in enumerate(contexts):
        ans_vec, q_vec = ans_vecs[i], q_vecs[i]
        ctx_vec = next(ctx_iter) if c.strip() else None
        rows.append({
            "cosine_ctx_ans": round(_cosine(ans_vec, ctx_vec), 6) if ctx_vec is not None else 0.0,
            "cosine_q_ans": round(_cosine(ans_vec, q_vec), 6),
        })
    return pd.DataFrame(rows, index=df.index)
