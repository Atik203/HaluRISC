"""Chunked B3 extraction resume tests (no torch; fake extractor + tmp cache paths)."""

import json
import sys
from pathlib import Path

import numpy as np
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
    """In-memory fake of extract_full_feature_set; can be told to crash."""

    def __init__(self, crash_after_chunks=None):
        self.crash_after_chunks = crash_after_chunks
        self.calls = 0

    def __call__(self, df, models, batch_size=128):
        self.calls += 1
        if self.crash_after_chunks is not None and self.calls > self.crash_after_chunks:
            raise RuntimeError("simulated runtime death during extraction")
        rows = []
        for _, r in df.iterrows():
            rows.append({"sample_id": r["sample_id"],
                         "n_chars": float(len(r["answer"])),
                         "overlap_answer_context": 0.5,
                         "cosine_ctx_ans": 0.25})
        return pd.DataFrame(rows)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    unified = _make_unified(tmp_path)
    monkeypatch.setattr(b3, "UNIFIED", unified)
    monkeypatch.setattr(b3, "FEATURES_CACHE", tmp_path / "b3_external_features.parquet")
    monkeypatch.setattr(b3, "FEATURES_CACHE_META", tmp_path / "b3_external_features.meta.json")
    df = pd.read_parquet(unified)
    return df


def test_partial_cache_resumes(setup):
    df = setup
    fake = FakeExtractor(crash_after_chunks=1)
    with pytest.raises(RuntimeError, match="simulated runtime death"):
        b3.extract_or_load_external_features(df, FEATURE_COLS, "cuda", 128,
                                             extract_fn=fake, models={}, chunk_size=2)
    meta = json.loads(b3.FEATURES_CACHE_META.read_text())
    assert meta["complete"] is False
    assert meta["n_rows"] == 2  # one chunk of 2 done before the crash

    # resume: a fresh call continues from the partial cache
    fake2 = FakeExtractor()
    merged = b3.extract_or_load_external_features(df, FEATURE_COLS, "cuda", 128,
                                                  extract_fn=fake2, models={}, chunk_size=2)
    assert len(merged) == len(df)
    assert merged[FEATURE_COLS].notna().all().all()
    meta2 = json.loads(b3.FEATURES_CACHE_META.read_text())
    assert meta2["complete"] is True
    assert meta2["n_rows"] == len(df)


def test_skip_features_rejects_partial(setup):
    df = setup
    fake = FakeExtractor(crash_after_chunks=1)
    with pytest.raises(RuntimeError):
        b3.extract_or_load_external_features(df, FEATURE_COLS, "cuda", 128,
                                             extract_fn=fake, models={}, chunk_size=2)
    with pytest.raises(ValueError, match="PARTIAL"):
        b3.extract_or_load_external_features(df, FEATURE_COLS, "cuda", 128, skip_features=True)


def test_skip_features_accepts_complete(setup):
    df = setup
    fake = FakeExtractor()
    b3.extract_or_load_external_features(df, FEATURE_COLS, "cuda", 128,
                                         extract_fn=fake, models={}, chunk_size=2)
    merged = b3.extract_or_load_external_features(df, FEATURE_COLS, "cuda", 128, skip_features=True)
    assert len(merged) == len(df)
    assert merged[FEATURE_COLS].notna().all().all()


def test_no_redundant_work_on_resume(setup):
    df = setup
    fake = FakeExtractor(crash_after_chunks=1)
    with pytest.raises(RuntimeError):
        b3.extract_or_load_external_features(df, FEATURE_COLS, "cuda", 128,
                                             extract_fn=fake, models={}, chunk_size=2)
    calls_before = fake.calls
    fake2 = FakeExtractor()
    b3.extract_or_load_external_features(df, FEATURE_COLS, "cuda", 128,
                                         extract_fn=fake2, models={}, chunk_size=2)
    # resumed run must not re-extract the 2 already-done rows: 6 remaining -> 3 chunks of 2
    assert fake2.calls == 3
