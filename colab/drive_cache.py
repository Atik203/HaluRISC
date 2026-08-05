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


def restore_config_hash(config_json_path, key: str, file_path) -> bool:
    """Generic: cached run-config records input hashes; restore only when they match.

    key is the dotted path into the config, e.g. "inputs.features_full.parquet"
    or "unified_parquet_sha256". Returns False on any mismatch/missing file.
    """
    import json

    try:
        cfg = json.loads(Path(config_json_path).read_text(encoding="utf-8"))
        node = cfg
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        return node == sha256_file(file_path)
    except (OSError, ValueError, TypeError):
        return False


def b2_restore_valid(b2_config_path, features_path) -> bool:
    """B2 artifacts are reusable when the cached run consumed THIS features_full.parquet."""
    return restore_config_hash(b2_config_path, "inputs.features_full.parquet", features_path)


def b3_restore_valid(b3_config_path, unified_path, b2_models_dir) -> bool:
    """B3 results are reusable when unified parquet AND B2 models match the cached run."""
    import json

    try:
        cfg = json.loads(Path(b3_config_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    try:
        if cfg.get("unified_parquet_sha256") != sha256_file(unified_path):
            return False
        hashes = cfg.get("b2_model_hashes") or {}
        return all(
            hashes.get(f"seed_{s}") == sha256_file(Path(b2_models_dir) / f"xgboost_seed_{s}.joblib")
            for s in (42, 123, 456)
        )
    except OSError:
        return False


def b4_restore_valid(b4_config_path, b3_predictions_path) -> bool:
    """B4 artifacts are reusable when they were produced from THIS b3 predictions file."""
    return restore_config_hash(b4_config_path, "inputs.b3_predictions.parquet", b3_predictions_path)


def version_a_restore_valid(marker_path, features_path, qa_path, split_report_path, cache_dir) -> bool:
    """Validate the cached root Version A artifacts against current inputs."""
    try:
        marker = json.loads(Path(marker_path).read_text(encoding="utf-8"))
        expected = {
            "features_full.parquet": sha256_file(features_path),
            "qa_clean.parquet": sha256_file(qa_path),
            "split_integrity_report.json": sha256_file(split_report_path),
        }
        if marker.get("inputs") != expected:
            return False
        return all((Path(cache_dir) / rel).exists() for rel in marker.get("artifacts", []))
    except (OSError, ValueError, TypeError):
        return False


def b3_feature_cache_safe(cache_path) -> bool:
    """Reject old B3 caches that accidentally contain raw external text."""
    try:
        import pandas as pd

        columns = set(pd.read_parquet(cache_path, columns=None).columns)
        forbidden = {"question", "context", "answer", "span_annotations"}
        return not (columns & forbidden) and "sample_id" in columns
    except (OSError, ValueError, ImportError):
        return False


def b3_results_safe(results_dir) -> bool:
    """Reject cached B3 error cases containing unredacted FaithBench text."""
    import json

    error_path = Path(results_dir) / "b3_error_cases.json"
    if not error_path.exists():
        return True
    try:
        cases = json.loads(error_path.read_text(encoding="utf-8"))
        for case in cases:
            if case.get("source_dataset") == "faithbench":
                if any(case.get(field, "") for field in ("question", "context", "answer", "span_annotations")):
                    return False
        return True
    except (OSError, ValueError, TypeError):
        return False
