"""Test: B3 runner reuses a COMPLETE local cache without --skip-features."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.models.run_b3_cross_domain as b3  # noqa: E402

FEATURE_COLS = ["n_chars", "overlap_answer_context", "cosine_ctx_ans"]


def _make_unified(tmp_path, n=8) -> Path:
    rows = []
    for i in range(n):
        rows.append({
            "sample_id": f"rt:{i}", "source_dataset": "ragtruth",
            "source_group_id": f"ragtruth:{i}", "task": "qa", "domain": "marco",
            "official_split": "test", "quality": "good", "generator_model": "gpt-3.5-turbo-0613",
            "label": i % 2, "context": "c", "answer": "a", "question": "q",
        })
    path = tmp_path / "unified.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


class FakeExtractor:
    def __init__(self, raise_on_call=False):
        self.raise_on_call = raise_on_call
        self.calls = 0

    def __call__(self, df, models, batch_size=128):
        self.calls += 1
        if self.raise_on_call:
            raise AssertionError("extractor must not be called when a complete cache exists")
        rows = [{"sample_id": r["sample_id"], "n_chars": 1.0,
                 "overlap_answer_context": 0.5, "cosine_ctx_ans": 0.25} for _, r in df.iterrows()]
        return pd.DataFrame(rows)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    unified = _make_unified(tmp_path)
    monkeypatch.setattr(b3, "UNIFIED", unified)
    monkeypatch.setattr(b3, "FEATURES_CACHE", tmp_path / "b3_external_features.parquet")
    monkeypatch.setattr(b3, "FEATURES_CACHE_META", tmp_path / "b3_external_features.meta.json")
    return pd.read_parquet(unified)


def test_complete_cache_reused_without_flag(setup):
    df = setup
    b3.extract_or_load_external_features(df, FEATURE_COLS, "cuda", 128,
                                         extract_fn=FakeExtractor(), models={}, chunk_size=2)
    # second call WITHOUT --skip-features must not call the extractor at all
    spy = FakeExtractor(raise_on_call=True)
    merged = b3.extract_or_load_external_features(df, FEATURE_COLS, "cuda", 128,
                                                  extract_fn=spy, models={}, chunk_size=2)
    assert spy.calls == 0
    assert len(merged) == len(df)
    assert merged[FEATURE_COLS].notna().all().all()


def test_partial_cache_still_resumes(setup):
    df = setup
    meta = {"input_sha256": b3.sha256(b3.UNIFIED), "complete": False, "n_rows": 2}
    partial = pd.DataFrame([
        {"sample_id": "rt:0", "n_chars": 1.0, "overlap_answer_context": 0.5, "cosine_ctx_ans": 0.25},
        {"sample_id": "rt:1", "n_chars": 1.0, "overlap_answer_context": 0.5, "cosine_ctx_ans": 0.25},
    ])
    partial.to_parquet(b3.FEATURES_CACHE)
    b3.FEATURES_CACHE_META.write_text(json.dumps(meta))
    fake = FakeExtractor()
    merged = b3.extract_or_load_external_features(df, FEATURE_COLS, "cuda", 128,
                                                  extract_fn=fake, models={}, chunk_size=2)
    assert fake.calls == 3  # 6 remaining rows -> 3 chunks
    assert len(merged) == len(df)
