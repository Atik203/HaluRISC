"""Drive-cache verification tests (pure functions; no google.colab imports)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colab.drive_cache import (  # noqa: E402
    b3_feature_cache_safe,
    b3_results_safe,
    cache_key_matches,
    read_cache_meta,
    sha256_file,
    version_a_restore_valid,
    verify_halueval_features,
)


def _synthetic_qa(n_groups: int = 10000) -> pd.DataFrame:
    rows = []
    for i in range(n_groups):
        r = i % 20
        split = "train" if r < 14 else ("val" if r < 17 else "test")
        rows.append({"sample_id": f"q_{i}_correct", "item_idx": i, "question": f"q{i}",
                     "context": f"c{i}", "answer": f"a{i}", "label": 0, "split": split})
        rows.append({"sample_id": f"q_{i}_hallucinated", "item_idx": i, "question": f"q{i}",
                     "context": f"c{i}", "answer": f"b{i}", "label": 1, "split": split})
    return pd.DataFrame(rows)


def _synthetic_features(qa: pd.DataFrame) -> pd.DataFrame:
    feats = {c: 0.5 for c in ("n_chars", "n_words", "overlap_answer_context",
                              "nli_ctx_entails_ans", "cosine_ctx_ans")}
    out = qa.copy()
    for c, v in feats.items():
        out[c] = v
    return out


def test_verify_matching_cache(tmp_path):
    qa = _synthetic_qa()
    feats = _synthetic_features(qa)
    qa_path, feat_path = tmp_path / "qa.parquet", tmp_path / "feats.parquet"
    qa.to_parquet(qa_path)
    feats.to_parquet(feat_path)
    v = verify_halueval_features(str(feat_path), str(qa_path))
    assert v["ok"] is True
    assert v["checks"] == []


def test_verify_detects_sample_mismatch(tmp_path):
    qa = _synthetic_qa()
    feats = _synthetic_features(qa)
    feats.loc[0, "sample_id"] = "q_99999_correct"
    qa_path, feat_path = tmp_path / "qa.parquet", tmp_path / "feats.parquet"
    qa.to_parquet(qa_path)
    feats.to_parquet(feat_path)
    v = verify_halueval_features(str(feat_path), str(qa_path))
    assert v["ok"] is False
    assert any("sample_id" in c for c in v["checks"])


def test_verify_detects_split_mismatch(tmp_path):
    qa = _synthetic_qa()
    feats = _synthetic_features(qa)
    feats.loc[feats["split"] == "val", "split"] = "test"
    qa_path, feat_path = tmp_path / "qa.parquet", tmp_path / "feats.parquet"
    qa.to_parquet(qa_path)
    feats.to_parquet(feat_path)
    v = verify_halueval_features(str(feat_path), str(qa_path))
    assert v["ok"] is False
    assert any("split" in c for c in v["checks"])


def test_verify_unreadable_cache(tmp_path):
    qa = _synthetic_qa(n_groups=50)
    qa_path = tmp_path / "qa.parquet"
    qa.to_parquet(qa_path)
    v = verify_halueval_features(str(tmp_path / "missing.parquet"), str(qa_path))
    assert v["ok"] is False


def test_cache_key_matches(tmp_path):
    meta = {"input_sha256": "abc"}
    unified = tmp_path / "u.parquet"
    unified.write_bytes(b"data")
    assert cache_key_matches(meta, str(unified)) is False  # hash mismatch
    good_meta = {"input_sha256": sha256_file(str(unified))}
    assert cache_key_matches(good_meta, str(unified)) is True
    assert cache_key_matches({}, str(unified)) is False


def test_read_cache_meta(tmp_path):
    p = tmp_path / "meta.json"
    p.write_text('{"input_sha256": "x"}')
    assert read_cache_meta(str(p)) == {"input_sha256": "x"}
    assert read_cache_meta(str(tmp_path / "nope.json")) == {}


def test_version_a_restore_validates_inputs_and_artifacts(tmp_path):
    features = tmp_path / "features.parquet"
    qa = tmp_path / "qa.parquet"
    split = tmp_path / "split.json"
    features.write_bytes(b"features")
    qa.write_bytes(b"qa")
    split.write_bytes(b"split")
    cache = tmp_path / "version_a"
    (cache / "models").mkdir(parents=True)
    (cache / "models" / "params.json").write_text("{}")
    marker = cache / "marker.json"
    import json

    marker.write_text(json.dumps({
        "inputs": {
            "features_full.parquet": sha256_file(features),
            "qa_clean.parquet": sha256_file(qa),
            "split_integrity_report.json": sha256_file(split),
        },
        "artifacts": ["models/params.json"],
    }))
    assert version_a_restore_valid(marker, features, qa, split, cache) is True
    features.write_bytes(b"changed")
    assert version_a_restore_valid(marker, features, qa, split, cache) is False


def test_b3_feature_cache_rejects_raw_text(tmp_path):
    safe = pd.DataFrame({"sample_id": ["x"], "n_chars": [1.0]})
    unsafe = safe.assign(context=["restricted text"])
    safe_path, unsafe_path = tmp_path / "safe.parquet", tmp_path / "unsafe.parquet"
    safe.to_parquet(safe_path)
    unsafe.to_parquet(unsafe_path)
    assert b3_feature_cache_safe(safe_path) is True
    assert b3_feature_cache_safe(unsafe_path) is False


def test_b3_results_reject_unredacted_faithbench(tmp_path):
    p = tmp_path / "b3_error_cases.json"
    p.write_text('[{"source_dataset":"faithbench","context":"raw"}]')
    assert b3_results_safe(tmp_path) is False
    p.write_text('[{"source_dataset":"faithbench","context":"","answer":"","question":"","span_annotations":""}]')
    assert b3_results_safe(tmp_path) is True
