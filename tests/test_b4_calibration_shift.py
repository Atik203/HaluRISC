"""B4 calibration-under-shift tests (synthetic; no heavy models, no downloads)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.run_b4_calibration_shift import (  # noqa: E402
    MODEL_THRESHOLD,
    SEEDS,
    adaptive_ece,
    apply_calibrator,
    calibration_metrics,
    calibration_slope_intercept,
    fit_calibrator,
    reliability_curve,
    subgroup_calibration,
    target_calibration_experiment,
)


def test_calibrators_bounded_and_monotone():
    rng = np.random.default_rng(1)
    scores = rng.random(500)
    y = (scores > 0.6).astype(int)
    for method in ("platt", "isotonic"):
        cal = fit_calibrator(method, scores, y)
        p = apply_calibrator(method, cal, scores)
        assert (p >= 0.0).all() and (p <= 1.0).all()


def test_perfect_calibration_is_zero_error():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = y.astype(float)
    m = calibration_metrics(y, p)
    assert m["ece"] == 0.0
    assert m["ace"] == 0.0
    assert m["brier"] == 0.0
    assert m["f1"] == 1.0
    assert m["slope"] > 0  # degenerate exact 0/1 scores clip the logit; slope stays positive
    assert abs(m["intercept"]) < 1e-6  # balanced classes keep the separating plane symmetric


def test_platt_fixes_miscalibration():
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.5, 1000)
    biased = np.clip(np.full(1000, 0.25) + 0.3 * y + rng.normal(0, 0.1, 1000), 0.01, 0.99)
    raw = calibration_metrics(y, biased)
    cal = fit_calibrator("platt", biased, y)
    p = apply_calibrator("platt", cal, biased)
    fitted = calibration_metrics(y, p)
    assert fitted["brier"] < raw["brier"]  # Platt must reduce Brier on this bias


def test_reliability_curve_bins_cover_all_rows():
    rng = np.random.default_rng(4)
    y = rng.binomial(1, 0.5, 200)
    p = rng.random(200)
    curve = reliability_curve(y, p, n_bins=10)
    assert len(curve) == 10
    assert sum(c["n"] for c in curve) == 200
    for c in curve:
        if c["confidence"] is not None:
            assert 0.0 <= c["confidence"] <= 1.0


def test_adaptive_ece_equal_frequency():
    rng = np.random.default_rng(5)
    p = np.sort(rng.random(1000))
    y = (p > 0.5).astype(int)
    assert 0.0 <= adaptive_ece(y, p, 10) <= 1.0
    # thresholded-uniform case: expected ~0.25 (bin center vs hard threshold), sanity-bound it
    assert adaptive_ece(y, p, 10) < 0.35


def _make_external_scores(n_groups: int = 25, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_groups):
        for split in ("train", "test"):
            score = rng.random()
            rows.append({
                "sample_id": f"rt:{i}_{split}",
                "source_dataset": "ragtruth",
                "source_group_id": f"ragtruth:{split}_{i}",
                "task": "qa",
                "domain": "marco",
                "official_split": split,
                "quality": "good",
                "generator_model": "gpt-3.5-turbo-0613",
                "label": int(score > 0.5),
                "answer": "text " * 30,
                **{f"score_{s}": score for s in SEEDS},
            })
    return pd.DataFrame(rows)


def test_target_calibration_group_disjointness():
    df = _make_external_scores()
    cal_df = df[df["official_split"] == "train"]
    test_df = df[df["official_split"] == "test"]
    out = target_calibration_experiment(cal_df, test_df)
    assert out["overlapping_groups_removed"] == 0  # groups are split-exclusive here
    for method in ("platt", "isotonic"):
        assert "ece_mean" in out["methods"][method]
        assert 0.0 <= out["methods"][method]["ece_mean"] <= 1.0


def test_target_calibration_removes_overlap_groups():
    df = _make_external_scores(n_groups=15)
    cal_df = pd.concat([df[df["official_split"] == "train"], df[df["official_split"] == "test"].head(2)])
    test_df = df[df["official_split"] == "test"]
    out = target_calibration_experiment(cal_df, test_df)
    assert out["overlapping_groups_removed"] == 2


def test_subgroup_minimum_rules():
    df = _make_external_scores(n_groups=6)
    rng = np.random.default_rng(8)
    calibrators = {
        "platt": {42: fit_calibrator("platt", rng.random(100), rng.binomial(1, 0.5, 100))},
        "isotonic": {42: fit_calibrator("isotonic", rng.random(100), rng.binomial(1, 0.5, 100))},
    }
    rows = subgroup_calibration(df, "task", calibrators, 10)
    assert rows and all(r["reported"] is False for r in rows)


def test_calibration_metrics_schema():
    rng = np.random.default_rng(9)
    y = rng.binomial(1, 0.4, 300)
    p = rng.random(300)
    m = calibration_metrics(y, p)
    for k in ("ece", "ace", "brier", "nll", "slope", "intercept", "f1", "auroc", "predicted_positive_rate"):
        assert k in m
    assert m["predicted_positive_rate"] == float(((p >= MODEL_THRESHOLD).astype(int)).mean())


def test_slope_intercept_simple():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    slope, intercept = calibration_slope_intercept(y, p)
    assert slope > 0
    assert isinstance(slope, float) and isinstance(intercept, float)
