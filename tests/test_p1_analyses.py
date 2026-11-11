"""Regression tests for the P1 external baseline and conformal analyses."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import p1_conformal as p1c  # noqa: E402
from src.models import p1_external_baseline as p1e  # noqa: E402


P1_DIR = ROOT / "artifacts" / "results" / "rebuild" / "p1"


# --------------------------------------------------------------------------
# ROUGE-L
# --------------------------------------------------------------------------
def test_rouge_l_identical_is_one():
    precision, recall, f1 = p1e.rouge_l_scores("The cat sat down", "The cat sat down")
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)


def test_rouge_l_disjoint_is_zero():
    precision, recall, f1 = p1e.rouge_l_scores("alpha beta", "gamma delta")
    assert precision == 0.0
    assert recall == 0.0
    assert f1 == 0.0


def test_rouge_l_known_lcs():
    # LCS is "the quick fox" (3 tokens); answer and context each have 4.
    precision, recall, f1 = p1e.rouge_l_scores("the quick brown fox", "the quick fox jumps")
    assert precision == pytest.approx(3 / 4)
    assert recall == pytest.approx(3 / 4)
    assert f1 == pytest.approx(3 / 4)


def test_lcs_handles_empty_tokens():
    assert p1e.lcs_length([], ["a"]) == 0
    assert p1e.lcs_length(["a"], []) == 0
    assert p1e.lcs_length(["a", "b", "c"], ["a", "c"]) == 2


# --------------------------------------------------------------------------
# Conformal machinery
# --------------------------------------------------------------------------
def test_conformal_quantile_uses_finite_sample_rank():
    scores = np.arange(1, 101, dtype=float)
    q, rank = p1c.conformal_quantile(scores, alpha=0.10)
    assert rank == 91
    assert q == 91.0
    q95, rank95 = p1c.conformal_quantile(scores, alpha=0.05)
    assert rank95 == 96
    assert q95 == 96.0


def test_conformal_quantile_clips_when_rank_exceeds_n():
    q, rank = p1c.conformal_quantile(np.array([0.1, 0.2]), alpha=0.001)
    assert q == 1.0
    assert rank > 2


def test_prediction_set_metrics_simple_case():
    y = np.array([1, 0, 1, 0])
    # q = 0.7: include positive at p >= 0.3, include negative at p <= 0.7.
    p = np.array([0.9, 0.1, 0.5, 0.8])
    metrics = p1c.prediction_set_metrics(y, p, q=0.7)
    # Row 0: {1}; row 1: {0}; row 2: {0,1}; row 3: {1}, so the negative in
    # row 3 is not covered.
    assert metrics["coverage"] == pytest.approx(0.75)
    assert metrics["coverage_pos"] == pytest.approx(1.0)
    assert metrics["coverage_neg"] == pytest.approx(0.5)
    assert metrics["avg_set_size"] == pytest.approx(1.25)
    assert metrics["frac_empty"] == 0.0
    assert metrics["frac_singleton_positive"] == pytest.approx(0.5)
    assert metrics["frac_singleton_negative"] == pytest.approx(0.25)
    assert metrics["frac_both"] == pytest.approx(0.25)


def test_prediction_set_metrics_empty_sets_when_q_below_half():
    y = np.array([1, 0])
    p = np.array([0.5, 0.5])
    metrics = p1c.prediction_set_metrics(y, p, q=0.4)
    # include positive at p >= 0.6 (false), include negative at p <= 0.4
    # (false): both rows get empty sets.
    assert metrics["coverage"] == 0.0
    assert metrics["avg_set_size"] == 0.0
    assert metrics["frac_empty"] == 1.0


def test_conformal_q_is_shared_across_datasets():
    """Regression: the quantile must come from calibration, not the eval set."""
    path = P1_DIR / "conformal.csv"
    if not path.exists():
        pytest.skip("run src/models/p1_conformal.py first")
    table = pd.read_csv(path)
    for (alpha, seed), block in table.groupby(["alpha", "seed"]):
        values = block["q"].to_numpy()
        assert np.allclose(values, values[0], atol=1e-12), (alpha, seed, values)


# --------------------------------------------------------------------------
# Output schemas and leakage checks
# --------------------------------------------------------------------------
def test_external_baseline_output_schema():
    path = P1_DIR / "external_baseline.csv"
    if not path.exists():
        pytest.skip("run src/models/p1_external_baseline.py first")
    table = pd.read_csv(path)
    assert list(table.columns) == p1e.RESULT_COLUMNS
    assert set(table["analysis"]) == {"reimplementation", "reimplementation_train_fit", "reference"}
    assert {"external_lr", "external_lr_train", "ec_xgb_m3"} <= set(table["model"])
    assert (table["n"] == 3000).all()
    assert table["f1"].between(0.0, 1.0).all()
    assert (table["f1_ci_low"] <= table["f1_ci_high"]).all()
    assert (table["auroc_ci_low"] <= table["auroc_ci_high"]).all()
    report = json.loads((P1_DIR / "external_baseline.json").read_text())
    assert report["schema"] == "rebuild-p1-external-baseline-v1"
    assert report["protocol"]["test_used_for_fit_or_selection"] is False
    assert report["protocol"]["n_fit"] == 3000
    assert report["protocol"]["n_fit_train_fit"] == 14000
    assert set(report["features"]) == set(p1e.FEATURE_COLUMNS)


def test_conformal_output_schema():
    path = P1_DIR / "conformal.csv"
    if not path.exists():
        pytest.skip("run src/models/p1_conformal.py first")
    table = pd.read_csv(path)
    assert list(table.columns) == p1c.CONFORMAL_COLUMNS
    assert set(table["dataset"]) == set(p1c.DATASETS)
    assert set(table["alpha"]) == set(p1c.ALPHAS)
    assert set(table["seed"]) == {str(seed) for seed in (42, 123, 456)} | {"pooled"}
    assert (table["coverage"] >= 0.0).all() and (table["coverage"] <= 1.0).all()
    assert (table["avg_set_size"] >= 0.0).all() and (table["avg_set_size"] <= 2.0).all()
    assert (table["coverage_ci_low"] <= table["coverage_ci_high"]).all()

    calibration = table[table["dataset"] == "ragtruth_qa_calibration"]
    for alpha in p1c.ALPHAS:
        block = calibration[(calibration["alpha"] == alpha) & (calibration["seed"] == "pooled")]
        assert abs(float(block["coverage"].iloc[0]) - (1.0 - alpha)) < 0.01

    report = json.loads((P1_DIR / "conformal.json").read_text())
    assert report["schema"] == "rebuild-p1-conformal-v1"
    assert report["protocol"]["n_boot"] == p1c.N_BOOT
    assert report["protocol"]["test_used_for_fit_or_selection"] is False
    assert {"disjoint_platt", "class_conditional"} <= set(report["sensitivity"])


def test_multicalibration_output_schema():
    path = P1_DIR / "multicalibration.csv"
    if not path.exists():
        pytest.skip("run src/models/p1_conformal.py first")
    table = pd.read_csv(path)
    assert list(table.columns) == p1c.MULTICALIBRATION_COLUMNS
    assert set(table["dataset"]) == {"ragtruth_qa_test", "faithbench", "halueval_test"}
    assert set(table["score"]) == {"raw", "target_platt"}
    assert {"overall", "length_words_short", "length_words_natural", "claim_count"} <= set(
        table["binning"]
    )
    assert set(table["seed"]) == {str(seed) for seed in (42, 123, 456)} | {"pooled"}
    assert (table["ece"] >= 0.0).all()
    assert (table["brier"] >= 0.0).all() and (table["brier"] <= 1.0).all()
    overall = table[(table["binning"] == "overall") & (table["bin"] == "all")]
    assert not overall.empty
    for dataset in ("ragtruth_qa_test", "faithbench", "halueval_test"):
        assert (overall["dataset"] == dataset).any()


def test_calibration_and_qa_test_are_disjoint():
    frames = p1c.load_frames()
    calibration = set(frames["ragtruth_qa_calibration"]["sample_id"])
    qa_test = set(frames["ragtruth_qa_test"]["sample_id"])
    assert calibration.isdisjoint(qa_test)
    assert len(calibration) == 5034
    assert len(qa_test) == 900


def test_combined_report_contains_both_sections():
    path = P1_DIR / "p1_external_baseline_and_conformal_REPORT.md"
    if not path.exists():
        pytest.skip("run both P1 analyses first")
    text = path.read_text(encoding="utf-8")
    assert "P1.3" in text
    assert "P1.4" in text
    assert "External Baseline" in text
    assert "Conformal" in text


def test_individual_reports_exist():
    for name in ("external_baseline_REPORT.md", "conformal_REPORT.md"):
        path = P1_DIR / name
        if not path.exists():
            pytest.skip(f"run the analysis that writes {name} first")
        assert path.stat().st_size > 1000
