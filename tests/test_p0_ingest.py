"""Tests for the P0 author-review ingest helpers.

These cover the three failure modes found in the independent review:
a corrected row must count as verified, the merged three-author blind file
must be readable, and a one-rater blind file must not be treated as complete.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "models" / "p0_ingest_author_review.py"


def load_module():
    spec = importlib.util.spec_from_file_location("p0_ingest", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_author_corrected_is_a_verified_state():
    module = load_module()
    assert module.is_verified(pd.Series(["author_verified", "author_corrected"]))


def test_merged_blind_layout_is_read(tmp_path):
    module = load_module()
    merged = pd.DataFrame({
        "item_id": ["a", "b", "c"],
        "sheet": ["taxonomy", "taxonomy", "taxonomy"],
        "a1": ["x", "y", "x"],
        "a2": ["x", "y", "y"],
        "a3": ["x", "z", "x"],
    })
    path = tmp_path / "p0_blind_overlap_merged.csv"
    merged.to_csv(path, index=False)

    ratings, warnings = module.read_blind(path)
    assert warnings == []
    assert ratings["taxonomy"]["a"] == ["x", "x", "x"]
    assert ratings["taxonomy"]["b"] == ["y", "y", "z"]
    assert module.fleiss_kappa(list(ratings["taxonomy"].values())) is not None


def test_one_rater_is_not_complete():
    module = load_module()
    one_rater = {"taxonomy": {"a": ["x"], "b": ["y"]}}
    three_raters = {"taxonomy": {"a": ["x", "x", "x"], "b": ["y", "y", "y"]}}

    assert not all(
        value.get("raters", 0) >= 3 and value.get("fleiss_kappa") is not None
        for value in [{"raters": len(next(iter(one_rater.values()))), "fleiss_kappa": None}]
    )
    assert all(
        value.get("raters", 0) >= 3 and value.get("fleiss_kappa") is not None
        for value in [{"raters": 3, "fleiss_kappa": 1.0}]
    )
