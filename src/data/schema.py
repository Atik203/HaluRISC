"""
HaluRISC Version B unified dataset schema (roadmap §14 B1).

Canonical, versioned row contract shared by HaluEval, RAGTruth, and FaithBench.

Design rules:
  - Version A preprocessing (`src/data/prepare.py`) is frozen and unchanged.
  - This module is additive: it defines the B1 contract and validation only.
  - Original annotations and metadata are preserved losslessly in JSON columns;
    the binary `label` is a derived, documented interface, never a replacement.
  - `sample_id` is globally unique; `source_group_id` is the leakage-control
    group key (HaluEval: item_idx, RAGTruth: source_id, FaithBench: raw id).
"""

import hashlib
import json

SCHEMA_VERSION = "b1-schema-v1"
LABEL_MAPPING_VERSION = "b1-labels-v1"

SOURCE_DATASETS = ("halueval", "ragtruth", "faithbench")
TASKS = ("qa", "summarization", "data_to_text")
SPLITS = ("train", "val", "test")

UNIFIED_COLUMNS = [
    "sample_id",          # globally unique row id
    "source_dataset",     # halueval | ragtruth | faithbench
    "source_group_id",    # leakage-control group key
    "task",               # qa | summarization | data_to_text
    "domain",             # stable domain/source category
    "question",           # user question; "" for summarization/data-to-text
    "context",            # evidence / source text
    "answer",             # model response / summary (must match span offsets)
    "label",              # unified binary label (0 = faithful, 1 = hallucinated)
    "span_annotations",   # JSON string, original annotations preserved verbatim
    "generator_model",    # original model when available
    "official_split",     # native dataset split (RAGTruth train/test) or ""
    "experiment_split",   # HaluEval grouped split (train/val/test) or ""
    "quality",            # RAGTruth quality flag (good/truncated/...) or ""
    "native_record_id",   # original dataset row id
    "native_metadata",    # JSON string, other source-specific metadata
    "label_mapping_version",
]

EMPTY_SPANS = "[]"
EMPTY_META = "{}"


def json_dumps(obj) -> str:
    """Deterministic JSON serialization for metadata columns."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def frame_fingerprint(df) -> str:
    """Stable content hash of a canonical frame (determinism check)."""
    return sha256_text(json_dumps(df.to_dict(orient="records")))


def _all_json_strings(df, col: str) -> bool:
    def ok(v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return True
        if not isinstance(v, str):
            return False
        try:
            json.loads(v)
        except (ValueError, TypeError):
            return False
        return True

    return df[col].map(ok).all()


def validate_unified_df(df):
    """Raise ValueError with a precise message on any contract violation."""
    missing = [c for c in UNIFIED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing canonical columns: {missing}")
    extra = [c for c in df.columns if c not in UNIFIED_COLUMNS]
    if extra:
        raise ValueError(f"unexpected columns: {extra}")

    if df["sample_id"].duplicated().any():
        raise ValueError("sample_id must be globally unique")
    blank = df["sample_id"].isna() | (df["sample_id"].astype(str).str.strip() == "")
    if blank.any():
        raise ValueError("sample_id must be non-empty")
    blank = df["source_group_id"].isna() | (df["source_group_id"].astype(str).str.strip() == "")
    if blank.any():
        raise ValueError("source_group_id must be non-empty")

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("label must be 0 or 1")
    bad_ds = set(df["source_dataset"].unique()) - set(SOURCE_DATASETS)
    if bad_ds:
        raise ValueError(f"unknown source_dataset values: {sorted(bad_ds)}")
    bad_task = set(df["task"].unique()) - set(TASKS)
    if bad_task:
        raise ValueError(f"unknown task values: {sorted(bad_task)}")

    empty_ans = df["answer"].isna() | (df["answer"].astype(str).str.strip() == "")
    if empty_ans.any():
        raise ValueError("answer must be non-empty")

    exp_values = set(df["experiment_split"][df["experiment_split"] != ""].unique())
    bad_exp = exp_values - set(SPLITS)
    if bad_exp:
        raise ValueError(f"experiment_split must be in {SPLITS} or empty, got: {sorted(bad_exp)}")
    off_values = set(df["official_split"][df["official_split"] != ""].unique())
    for v in off_values:
        if not isinstance(v, str):
            raise ValueError(f"official_split must be a string, got {v!r}")

    for col in ("span_annotations", "native_metadata"):
        if not _all_json_strings(df, col):
            raise ValueError(f"{col} must be a JSON string (or empty)")
    return True
