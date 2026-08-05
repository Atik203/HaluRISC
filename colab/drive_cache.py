"""
Drive-cache helpers for the Colab notebook (cells 5b / 6b / 7d.5 / 7e).

Deterministic heavy artifacts (HaluEval features, B3 external features) are
cached on Google Drive between sessions so extraction does not repeat on every
run. Every restore is VERIFIED against freshly prepared data; a mismatch falls
back to re-extraction automatically.

No google.colab imports here, so these functions are unit-testable locally.
"""

import hashlib
import json
from pathlib import Path

EXPECTED_ROWS = 20000
EXPECTED_LABELS = {0: 10000, 1: 10000}
EXPECTED_SPLITS = {"train": 14000, "val": 3000, "test": 3000}
# Sanity feature names that must be present (full 26 are checked by the runners)
SANITY_FEATURES = ["n_chars", "n_words", "overlap_answer_context",
                   "nli_ctx_entails_ans", "cosine_ctx_ans"]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_halueval_features(features_path, qa_path) -> dict:
    """Verify a cached features_full.parquet against the freshly built qa_clean.parquet.

    Returns {"ok": bool, "checks": [reasons]}.
    """
    import pandas as pd

    try:
        feats = pd.read_parquet(features_path)
        qa = pd.read_parquet(qa_path)
    except Exception as e:  # corrupt or unreadable cache
        return {"ok": False, "checks": [f"read failed: {e}"]}

    checks = []
    if len(feats) != EXPECTED_ROWS:
        checks.append(f"rows {len(feats)} != {EXPECTED_ROWS}")
    if set(feats["sample_id"]) != set(qa["sample_id"]):
        checks.append("sample_id sets differ from qa_clean")
    if feats["label"].value_counts().to_dict() != EXPECTED_LABELS:
        checks.append(f"label balance {feats['label'].value_counts().to_dict()} != {EXPECTED_LABELS}")
    counts = feats.groupby("split").size().to_dict()
    if counts != EXPECTED_SPLITS:
        checks.append(f"split counts {counts} != {EXPECTED_SPLITS}")
    if "item_idx" in feats.columns and feats.groupby("item_idx")["split"].nunique().max() != 1:
        checks.append("item_idx groups span multiple splits (leakage)")
    missing_feats = [c for c in SANITY_FEATURES if c not in feats.columns]
    if missing_feats:
        checks.append(f"missing feature columns {missing_feats}")
    return {"ok": not checks, "checks": checks}


def read_cache_meta(meta_path) -> dict:
    try:
        return json.loads(Path(meta_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def cache_key_matches(meta: dict, unified_parquet_path) -> bool:
    """B3 external-feature cache is reusable only when the unified parquet hash matches."""
    if not meta.get("input_sha256"):
        return False
    try:
        return meta["input_sha256"] == sha256_file(unified_parquet_path)
    except OSError:
        return False
