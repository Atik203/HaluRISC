"""WP2/WP3 regression tests: leakage checks, prior-only sanity, stable schemas."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import rebuild_calibration as wp2  # noqa: E402
from src.models import rebuild_intervals as wp3  # noqa: E402
from src.models.config import SEEDS  # noqa: E402
from src.models.rebuild_common import (  # noqa: E402
    CALIBRATION_DIR,
    CALIBRATION_FEATURES,
    CALIBRATION_PREVALENCE,
    FEATURES_FULL,
    FIXED_HP_PRED,
    INTERVALS_DIR,
    TEST_PREVALENCE,
    metric_bundle,
)
from src.models.train_pipeline import ece  # noqa: E402


# --------------------------------------------------------------------------
# Leakage and split integrity
# --------------------------------------------------------------------------
def test_calibration_and_test_sample_ids_are_disjoint():
    calibration = pd.read_parquet(CALIBRATION_FEATURES)
    preds = pd.read_parquet(FIXED_HP_PRED)
    test = preds[
        (preds["variant"] == "m3")
        & (preds["seed"] == SEEDS[0])
        & (preds["dataset"] == "ragtruth_qa_test")
    ]
    overlap = set(calibration["sample_id"]) & set(test["sample_id"])
    assert overlap == set(), f"calibration/test overlap: {sorted(overlap)[:5]}"
    assert len(calibration) == 5034
    assert len(test) == 900


def test_halueval_val_and_test_are_disjoint():
    features = pd.read_parquet(FEATURES_FULL)
    val = features[features["split"] == "val"]
    test = features[features["split"] == "test"]
    assert set(val["sample_id"]).isdisjoint(set(test["sample_id"]))
    # Splits are grouped by item, not just by row.
    assert set(val["item_idx"]).isdisjoint(set(test["item_idx"]))
    assert len(val) == 3000
    assert len(test) == 3000


def test_prior_prevalences_match_the_reported_constants():
    calibration = pd.read_parquet(CALIBRATION_FEATURES)
    features = pd.read_parquet(FEATURES_FULL)
    predictions = pd.read_parquet(FIXED_HP_PRED)
    test = predictions[
        (predictions["variant"] == "m3")
        & (predictions["seed"] == SEEDS[0])
        & (predictions["dataset"] == "ragtruth_qa_test")
    ]
    assert abs(calibration["label"].mean() - CALIBRATION_PREVALENCE) < 1e-12
    assert abs(test["label"].mean() - TEST_PREVALENCE) < 1e-12
    assert len(features[features["split"] == "val"]) == 3000


# --------------------------------------------------------------------------
# Prior-only ECE is zero by construction
# --------------------------------------------------------------------------
def test_prior_only_ece_is_zero_by_construction():
    predictions = pd.read_parquet(FIXED_HP_PRED)
    test = predictions[
        (predictions["variant"] == "m3")
        & (predictions["seed"] == SEEDS[0])
        & (predictions["dataset"] == "ragtruth_qa_test")
    ]
    y = test["label"].to_numpy(dtype=int)
    constant = np.full(len(y), TEST_PREVALENCE)
    assert ece(y, constant) < 1e-12
    bundle = metric_bundle(y, constant)
    assert bundle["ece"] < 1e-12
    assert abs(bundle["brier"] - TEST_PREVALENCE * (1.0 - TEST_PREVALENCE)) < 1e-12


# --------------------------------------------------------------------------
# Output schemas
# --------------------------------------------------------------------------
def test_wp2_output_schemas():
    calibration_path = CALIBRATION_DIR / "calibration.csv"
    if not calibration_path.exists():
        pytest.skip("run src/models/rebuild_calibration.py first")

    calibration = pd.read_csv(calibration_path)
    slopes = pd.read_csv(CALIBRATION_DIR / "calibration_slope.csv")
    prior = pd.read_csv(CALIBRATION_DIR / "prior_only.csv")
    budget = pd.read_csv(CALIBRATION_DIR / "label_budget.csv")

    assert list(calibration.columns) == wp2.CALIBRATION_COLUMNS
    assert list(slopes.columns) == wp2.SLOPE_COLUMNS
    assert list(prior.columns) == wp2.PRIOR_COLUMNS
    assert list(budget.columns) == wp2.LABEL_BUDGET_COLUMNS

    assert set(calibration["model"]) == {"fixed_hp", "frozen_b6_tuned"}
    assert set(calibration["dataset"]) == {"halueval_test", "ragtruth_qa_test"}
    assert set(calibration["seed"]) == {str(seed) for seed in SEEDS} | {"pooled"}
    assert (calibration["n_boot"] == wp2.N_BOOT_CALIBRATION).all()
    assert calibration["n_boot_valid"].between(900, wp2.N_BOOT_CALIBRATION).all()
    assert calibration["prevalence"].notna().all()
    assert calibration["prevalence"].between(0.0, 1.0).all()
    assert (calibration["threshold"] == wp2.THRESHOLD).all()
    for column in ("ece", "brier", "auroc", "pr_auc", "f1"):
        assert calibration[column].between(0.0, 1.0).all()
        assert (calibration[f"{column}_ci_low"] <= calibration[f"{column}_ci_high"]).all()

    assert set(slopes["fit_split"]) == {"halueval_val", "ragtruth_qa_calibration"}
    assert (slopes["n_fit"].isin([3000, 5034])).all()
    assert (slopes["cal_slope_ci_low"] <= slopes["cal_slope_ci_high"]).all()

    zero_rows = prior[prior["ece_zero_by_construction"]]
    assert len(zero_rows) == 2
    assert (zero_rows["ece"].abs() < 1e-12).all()
    assert set(prior["predictor"]) == {"calibration_prior", "test_prior"}

    assert set(budget["size"]) == set(wp2.LABEL_BUDGET_SIZES)
    assert (budget["n_draws"] == wp2.LABEL_BUDGET_DRAWS).all()
    assert (budget["n_draws_used"] > 0).all()
    assert set(budget["seed"]) == {str(seed) for seed in SEEDS} | {"pooled"}

    report = json.loads((CALIBRATION_DIR / "calibration_report.json").read_text())
    assert report["schema"] == "rebuild-wp2-calibration-v1"
    assert set(report["headline"]) >= {
        "source_platt_halueval_test",
        "target_platt_ragtruth_qa_test",
        "raw_ragtruth_qa_test",
    }
    assert report["protocol"]["n_boot"] == wp2.N_BOOT_CALIBRATION


def test_wp3_output_schemas():
    intervals_path = INTERVALS_DIR / "intervals.csv"
    if not intervals_path.exists():
        pytest.skip("run src/models/rebuild_intervals.py first")

    intervals = pd.read_csv(intervals_path)
    contrast = pd.read_csv(INTERVALS_DIR / "intervals_contrast.csv")

    assert list(intervals.columns) == wp3.INTERVAL_COLUMNS
    assert list(contrast.columns) == wp3.CONTRAST_COLUMNS
    assert set(intervals["source"]) == set(wp3.SOURCES)
    assert set(intervals["dataset"]) == set(wp3.DATASET_ORDER)
    assert set(intervals["seed"]) == {str(seed) for seed in SEEDS} | {"pooled"}
    assert (intervals["n_boot"] == wp3.N_BOOT_INTERVALS).all()
    assert (intervals["ci_low"] <= intervals["ci_high"]).all()
    assert intervals["point_delta"].notna().all()

    expected_groups = {
        "halueval_test": 1500,
        "ragtruth_qa_test": 150,
        "ragtruth_all_test": 450,
    }
    for dataset, n_groups in expected_groups.items():
        sub = intervals[intervals["dataset"] == dataset]
        assert (sub["n_groups"] == n_groups).all()

    headline = {
        ("halueval_test", "m3-m0", "f1"),
        ("halueval_test", "m3-m0", "auroc"),
        ("halueval_test", "m2-m1", "f1"),
        ("halueval_test", "m3-m1", "f1"),
        ("ragtruth_qa_test", "m3-m0", "auroc"),
        ("ragtruth_all_test", "m3-m0", "f1"),
    }
    fixed_pooled = intervals[
        (intervals["source"] == "fixed_hp") & (intervals["seed"] == "pooled")
    ]
    present = set(zip(fixed_pooled["dataset"], fixed_pooled["comparison"], fixed_pooled["metric"]))
    assert headline <= present

    report = json.loads((INTERVALS_DIR / "intervals_report.json").read_text())
    assert report["schema"] == "rebuild-wp3-intervals-v1"
    assert report["protocol"]["n_boot"] == wp3.N_BOOT_INTERVALS
    assert len(report["headline"]) == len(wp3.HEADLINE)
    for entry in report["headline"]:
        assert set(entry["fixed_hp"]["per_seed"]) == {str(seed) for seed in SEEDS}
        assert entry["fixed_hp"]["ci"][0] <= entry["fixed_hp"]["ci"][1]
        assert entry["frozen_b6_tuned"]["ci"][0] <= entry["frozen_b6_tuned"]["ci"][1]
        assert entry["contrast_fixed_minus_tuned"]["ci"][0] <= entry["contrast_fixed_minus_tuned"]["ci"][1]
