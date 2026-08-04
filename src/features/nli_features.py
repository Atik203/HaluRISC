"""
NLI consistency feature extraction for HaluRISC.

Group 4 features (roadmap §6, mandatory per blueprint A8):
  nli_ctx_entails_ans, nli_ctx_contradicts_ans, nli_ctx_neutral_ans
  nli_ans_entails_ctx, nli_ans_contradicts_ctx, nli_ans_neutral_ctx

Primary model: cross-encoder/nli-deberta-v3-base (blueprint A8).
Fallback if download fails: cross-encoder/nli-MiniLM2-L6.

CrossEncoder output is a 3-class softmax in order
[contradiction, entailment, neutral] (SNLI/MultiNLI schema).

Empty context -> all three probs set to 1/3 (neutral), per prepare.py rule.
"""

import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

NLI_MODEL_PRIMARY = "cross-encoder/nli-deberta-v3-base"
NLI_MODEL_FALLBACK = "cross-encoder/nli-MiniLM2-L6"
NLI_MODEL_ENV = "HALU_NLI_MODEL"

NEUTRAL = 1.0 / 3.0

# CrossEncoder label order for NLI checkpoints
LABELS = ["contradiction", "entailment", "neutral"]


def load_nli_model(model_name: Optional[str] = None):
    """Load the NLI CrossEncoder (falls back to MiniLM2 on failure)."""
    from sentence_transformers import CrossEncoder

    chosen = model_name or os.environ.get(NLI_MODEL_ENV, NLI_MODEL_PRIMARY)
    try:
        logger.info(f"Loading NLI CrossEncoder: {chosen} ...")
        model = CrossEncoder(chosen)
        logger.info("NLI CrossEncoder loaded.")
        return model, chosen
    except Exception as e:
        if chosen != NLI_MODEL_FALLBACK:
            logger.warning(f"NLI model {chosen} failed ({e}); falling back to {NLI_MODEL_FALLBACK}")
            return load_nli_model(NLI_MODEL_FALLBACK)
        raise


def _neutral_row() -> dict:
    return {
        "nli_ctx_entails_ans": NEUTRAL,
        "nli_ctx_contradicts_ans": NEUTRAL,
        "nli_ctx_neutral_ans": NEUTRAL,
        "nli_ans_entails_ctx": NEUTRAL,
        "nli_ans_contradicts_ctx": NEUTRAL,
        "nli_ans_neutral_ctx": NEUTRAL,
    }


def extract_nli_features(question: str, context: str, answer: str, model) -> dict:
    """Group 4: entailment/contradiction probabilities, both directions."""
    if not context.strip():
        return _neutral_row()

    logits = model.predict([[context, answer], [answer, context]])
    probs = logits if logits.shape[1] == 3 else None
    if probs is None:
        raise ValueError(f"Unexpected NLI output shape {logits.shape}; expected (n, 3)")

    p_ctx = {LABELS[i]: float(probs[0, i]) for i in range(3)}
    p_ans = {LABELS[i]: float(probs[1, i]) for i in range(3)}

    return {
        "nli_ctx_entails_ans": round(p_ctx["entailment"], 6),
        "nli_ctx_contradicts_ans": round(p_ctx["contradiction"], 6),
        "nli_ctx_neutral_ans": round(p_ctx["neutral"], 6),
        "nli_ans_entails_ctx": round(p_ans["entailment"], 6),
        "nli_ans_contradicts_ctx": round(p_ans["contradiction"], 6),
        "nli_ans_neutral_ctx": round(p_ans["neutral"], 6),
    }


def extract_nli_features_df(df: pd.DataFrame, model, batch_size: int = 64) -> pd.DataFrame:
    """Batch NLI features; processes both (ctx, ans) and (ans, ctx) directions."""
    logger.info(f"Extracting NLI features for {len(df)} samples (2 directions each)...")
    rows = []
    batch_ctx_ans, batch_ans_ctx, batch_idx = [], [], []

    def flush():
        nonlocal batch_ctx_ans, batch_ans_ctx, batch_idx
        if not batch_idx:
            return
        all_probs = model.predict(batch_ctx_ans + batch_ans_ctx, batch_size=batch_size)
        n = len(batch_idx)
        for k, idx in enumerate(batch_idx):
            p_ctx = {LABELS[i]: float(all_probs[k, i]) for i in range(3)}
            p_ans = {LABELS[i]: float(all_probs[n + k, i]) for i in range(3)}
            rows.append((idx, {
                "nli_ctx_entails_ans": round(p_ctx["entailment"], 6),
                "nli_ctx_contradicts_ans": round(p_ctx["contradiction"], 6),
                "nli_ctx_neutral_ans": round(p_ctx["neutral"], 6),
                "nli_ans_entails_ctx": round(p_ans["entailment"], 6),
                "nli_ans_contradicts_ctx": round(p_ans["contradiction"], 6),
                "nli_ans_neutral_ctx": round(p_ans["neutral"], 6),
            }))
        batch_ctx_ans, batch_ans_ctx, batch_idx = [], [], []

    for idx, row in df.iterrows():
        context, answer = str(row["context"]), str(row["answer"])
        if not context.strip():
            rows.append((idx, _neutral_row()))
            continue
        batch_ctx_ans.append((context, answer))
        batch_ans_ctx.append((answer, context))
        batch_idx.append(idx)
        if len(batch_idx) >= batch_size * 4:
            flush()
    flush()

    rows.sort(key=lambda t: t[0])
    return pd.DataFrame([r for _, r in rows], index=[i for i, _ in rows])
