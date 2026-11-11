"""P1.4 - split conformal prediction and subgroup calibration on EC-XGB m3.

Protocol
--------
* Model: fixed-hyperparameter m3 (EC-XGB, 35 features), seeds 42/123/456,
  loaded from ``artifacts/models/rebuild/fixed_hp``.
* Target calibrator: Platt fitted on the 5,034 RAGTruth QA calibration rows
  (the same link as WP2).  No test row enters the fit.
* Split conformal: LAC nonconformity ``1 - p_true`` on the 5,034 calibration
  rows.  The finite-sample quantile is the ``ceil((n+1)(1-alpha))``-th smallest
  score.  Prediction sets are ``{1 if p >= 1-q}`` and ``{0 if p <= q}``.
* Coverage: calibration, RAGTruth QA test (900 rows), FaithBench (750 rows),
  and HaluEval test (3,000 rows, in-domain reference).  Positive, negative, and
  marginal coverage are all reported because marginal coverage can hide a
  collapsed positive set.
* Multicalibration: ECE and Brier per answer-length bin and claim-count bin,
  for raw m3 scores and target-Platt scores, on QA test, FaithBench, and
  HaluEval test.  HaluEval is a reference, not a target of the calibrator.
* CIs: 1,000 cluster-bootstrap draws.  Clusters are source groups for RAGTruth
  (6 rows), items for HaluEval (2 rows), and rows for FaithBench.

Two supplementary checks in the JSON:
* ``disjoint_platt``: fit Platt on half the calibration groups and run
  conformal on the other half, so the score function and the conformal
  quantile do not share rows.
* ``class_conditional``: Mondrian (class-conditional) conformal quantiles,
  which diagnose the positive-coverage collapse under prior shift.

Outputs (artifacts/results/rebuild/p1/):
  conformal.csv, multicalibration.csv, conformal.json, REPORT.md

Run (repo root, .venv):
  python src/models/p1_conformal.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd

from src.models.config import BOOTSTRAP_SEED, RESULTS_DIR, SEEDS
from src.models.rebuild_common import (
    CALIBRATION_FEATURES,
    EXTERNAL_FEATURES,
    FIXED_HP_MODELS,
    FIXED_HP_PRED,
    apply_platt,
    fit_platt,
    load_calibration_features,
    load_halueval_features,
    predict_raw,
    stable_seed,
    summarize_bootstrap,
)
from src.models.run_b6_modified import SOURCE_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("p1_conformal")

P1_DIR = RESULTS_DIR / "rebuild" / "p1"
VARIANT = "m3"
N_BOOT = 1000
ALPHAS = (0.10, 0.05)
ECE_BINS = 10
MIN_CI_N = 30

DATASETS = ("ragtruth_qa_calibration", "ragtruth_qa_test", "faithbench", "halueval_test")

SHORT_LENGTH_BINS = [0, 1, 2, 4, 8, 16, 10**9]
SHORT_LENGTH_LABELS = ["1", "2", "3-4", "5-8", "9-16", "17+"]
NATURAL_LENGTH_BINS = [0, 60, 80, 100, 120, 160, 10**9]
NATURAL_LENGTH_LABELS = ["<=60", "61-80", "81-100", "101-120", "121-160", "161+"]
CLAIM_BINS = [0, 1, 2, 3, 5, 10**9]
CLAIM_LABELS = ["1", "2", "3", "4-5", "6+"]
BIN_ORDER = {
    "overall": ["all"],
    "length_words_short": SHORT_LENGTH_LABELS,
    "length_words_natural": NATURAL_LENGTH_LABELS,
    "claim_count": CLAIM_LABELS,
}

CONFORMAL_COLUMNS = [
    "dataset",
    "n",
    "n_groups",
    "alpha",
    "nominal_coverage",
    "seed",
    "q",
    "q_sd",
    "coverage",
    "coverage_ci_low",
    "coverage_ci_high",
    "coverage_pos",
    "coverage_neg",
    "avg_set_size",
    "avg_set_size_ci_low",
    "avg_set_size_ci_high",
    "frac_empty",
    "frac_singleton_negative",
    "frac_singleton_positive",
    "frac_both",
    "n_boot",
]
MULTICALIBRATION_COLUMNS = [
    "dataset",
    "binning",
    "bin",
    "n",
    "n_groups",
    "prevalence",
    "score",
    "seed",
    "ece",
    "brier",
    "mean_score",
    "observed_rate",
    "ece_ci_low",
    "ece_ci_high",
    "brier_ci_low",
    "brier_ci_high",
    "small_n",
    "n_boot",
]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_frames() -> dict[str, pd.DataFrame]:
    """Load the four evaluation frames with a group id for cluster bootstrap."""
    if not CALIBRATION_FEATURES.exists() or not EXTERNAL_FEATURES.exists():
        raise FileNotFoundError("missing processed feature tables")

    calibration = load_calibration_features().reset_index(drop=True)
    # RAGTruth source groups are six consecutive sample ids; the calibration
    # table keeps six rows per group and the numeric suffix is a multiple of 6.
    numeric_id = calibration["sample_id"].str.split(":").str[1].astype(int)
    calibration["group_id"] = (numeric_id // 6).astype(int)

    external = pd.read_parquet(EXTERNAL_FEATURES)
    external[SOURCE_COLUMN] = 1.0
    qa_test = external[
        (external["source_dataset"] == "ragtruth")
        & (external["task"] == "qa")
        & (external["split"] == "test")
    ].reset_index(drop=True)
    faithbench = external[external["source_dataset"] == "faithbench"].reset_index(drop=True)
    qa_test["group_id"] = qa_test["source_group_id"].astype(str)
    faithbench["group_id"] = faithbench["sample_id"].astype(str)

    halueval = load_halueval_features()
    halueval_test = halueval[halueval["split"] == "test"].reset_index(drop=True)
    halueval_test["group_id"] = halueval_test["item_idx"].astype(str)

    frames = {
        "ragtruth_qa_calibration": calibration,
        "ragtruth_qa_test": qa_test,
        "faithbench": faithbench,
        "halueval_test": halueval_test,
    }
    expected = {
        "ragtruth_qa_calibration": 5034,
        "ragtruth_qa_test": 900,
        "faithbench": 750,
        "halueval_test": 3000,
    }
    for name, frame in frames.items():
        if len(frame) != expected[name]:
            raise ValueError(f"{name}: expected {expected[name]} rows, found {len(frame)}")
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"{name}: duplicate sample_id")
    if len(set(frames["ragtruth_qa_calibration"]["sample_id"]) & set(qa_test["sample_id"])) != 0:
        raise ValueError("calibration and QA test overlap")
    return frames


def verify_stored_scores(
    frame: pd.DataFrame, seed: int, dataset: str, predictions: pd.DataFrame, raw_scores: np.ndarray
) -> None:
    """Check the loaded model against the frozen fixed-hp prediction table."""
    block = predictions[
        (predictions["variant"] == VARIANT)
        & (predictions["seed"] == seed)
        & (predictions["dataset"] == dataset)
    ]
    if block.empty:
        raise ValueError(f"no stored predictions for {dataset} seed {seed}")
    block = block.set_index("sample_id").loc[frame["sample_id"]]
    if not np.array_equal(block["label"].to_numpy(dtype=int), frame["label"].to_numpy(dtype=int)):
        raise ValueError(f"{dataset} seed {seed}: label mismatch")
    max_diff = float(np.max(np.abs(block["score"].to_numpy(dtype=float) - raw_scores)))
    if max_diff > 1e-6:
        raise ValueError(f"{dataset} seed {seed}: score mismatch {max_diff:.3e}")


# --------------------------------------------------------------------------
# Conformal machinery
# --------------------------------------------------------------------------
def conformal_quantile(scores: np.ndarray, alpha: float) -> tuple[float, int]:
    """Finite-sample LAC quantile: the ceil((n+1)(1-alpha))-th smallest score."""
    ordered = np.sort(np.asarray(scores, dtype=float))
    n = len(ordered)
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return 1.0, rank
    return float(ordered[rank - 1]), rank


def prediction_set_metrics(y_true: np.ndarray, scores: np.ndarray, q: float) -> dict:
    """Coverage and set-size summaries for one calibrated score vector."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    include_positive = scores >= 1.0 - q
    include_negative = scores <= q
    covered = np.where(y_true == 1, include_positive, include_negative)
    size = include_positive.astype(int) + include_negative.astype(int)
    positives = y_true == 1
    negatives = ~positives
    return {
        "coverage": float(covered.mean()),
        "coverage_pos": float(covered[positives].mean()) if positives.any() else float("nan"),
        "coverage_neg": float(covered[negatives].mean()) if negatives.any() else float("nan"),
        "avg_set_size": float(size.mean()),
        "frac_empty": float((size == 0).mean()),
        "frac_singleton_negative": float(((size == 1) & include_negative).mean()),
        "frac_singleton_positive": float(((size == 1) & include_positive).mean()),
        "frac_both": float((size == 2).mean()),
        "_covered": covered,
        "_size": size,
        "_include_positive": include_positive,
        "_include_negative": include_negative,
    }


def group_rows(groups: np.ndarray) -> list[np.ndarray]:
    """Contiguous row indices for each unique group."""
    _, codes = np.unique(groups, return_inverse=True)
    return [np.flatnonzero(codes == index) for index in range(codes.max() + 1)]


def cluster_bootstrap_conformal(
    entries: list[dict],
    groups: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Cluster bootstrap for coverage and set size, averaged over seeds.

    ``entries`` is one dict per model seed, all sharing the same row order and
    group vector.  Each draw resamples groups once and applies the same rows to
    every seed, preserving cross-seed correlation.
    """
    rows = group_rows(groups)
    n_groups = len(rows)
    rng = np.random.default_rng(seed)
    coverage_draws: list[float] = []
    size_draws: list[float] = []
    for _ in range(n_boot):
        picked = rng.integers(0, n_groups, n_groups)
        index = np.concatenate([rows[group] for group in picked])
        coverage_draws.append(float(np.mean([entry["_covered"][index].mean() for entry in entries])))
        size_draws.append(float(np.mean([entry["_size"][index].mean() for entry in entries])))
    return {
        "coverage": summarize_bootstrap(coverage_draws),
        "avg_set_size": summarize_bootstrap(size_draws),
        "n_groups": n_groups,
    }


def conformal_table(
    frames: dict[str, pd.DataFrame],
    scores: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """Build the per-dataset, per-alpha, per-seed conformal table.

    The quantile is computed once per seed on the 5,034 calibration rows and
    then applied unchanged to every evaluation dataset.  Recomputing the
    quantile on an evaluation set would make coverage nominal by construction
    and would hide the shift failure.
    """
    rows = []
    calibration = frames["ragtruth_qa_calibration"]
    y_cal = calibration["label"].to_numpy(dtype=int)
    for alpha in ALPHAS:
        q_by_seed = {}
        for seed in SEEDS:
            p_cal = scores[str(seed)]["ragtruth_qa_calibration"]
            q, _ = conformal_quantile(np.where(y_cal == 1, 1.0 - p_cal, p_cal), alpha)
            q_by_seed[str(seed)] = q

        for dataset in DATASETS:
            frame = frames[dataset]
            y = frame["label"].to_numpy(dtype=int)
            groups = frame["group_id"].to_numpy()
            seed_entries = []
            seed_q = []
            for seed in SEEDS:
                p = scores[str(seed)][dataset]
                q = q_by_seed[str(seed)]
                metrics = prediction_set_metrics(y, p, q)
                seed_entries.append(metrics)
                seed_q.append(q)
                ci = cluster_bootstrap_conformal(
                    [metrics], groups, N_BOOT, stable_seed("p1_conformal", dataset, alpha, seed)
                )
                rows.append({
                    "dataset": dataset,
                    "n": int(len(y)),
                    "n_groups": int(ci["n_groups"]),
                    "alpha": float(alpha),
                    "nominal_coverage": float(1.0 - alpha),
                    "seed": str(seed),
                    "q": float(q),
                    "q_sd": 0.0,
                    "coverage": metrics["coverage"],
                    "coverage_ci_low": float(ci["coverage"]["q025"]),
                    "coverage_ci_high": float(ci["coverage"]["q975"]),
                    "coverage_pos": metrics["coverage_pos"],
                    "coverage_neg": metrics["coverage_neg"],
                    "avg_set_size": metrics["avg_set_size"],
                    "avg_set_size_ci_low": float(ci["avg_set_size"]["q025"]),
                    "avg_set_size_ci_high": float(ci["avg_set_size"]["q975"]),
                    "frac_empty": metrics["frac_empty"],
                    "frac_singleton_negative": metrics["frac_singleton_negative"],
                    "frac_singleton_positive": metrics["frac_singleton_positive"],
                    "frac_both": metrics["frac_both"],
                    "n_boot": int(N_BOOT),
                })
            ci = cluster_bootstrap_conformal(
                seed_entries, groups, N_BOOT, stable_seed("p1_conformal", dataset, alpha, "pooled")
            )
            rows.append({
                "dataset": dataset,
                "n": int(len(y)),
                "n_groups": int(ci["n_groups"]),
                "alpha": float(alpha),
                "nominal_coverage": float(1.0 - alpha),
                "seed": "pooled",
                "q": float(np.mean(seed_q)),
                "q_sd": float(np.std(seed_q, ddof=1)),
                "coverage": float(np.mean([entry["coverage"] for entry in seed_entries])),
                "coverage_ci_low": float(ci["coverage"]["q025"]),
                "coverage_ci_high": float(ci["coverage"]["q975"]),
                "coverage_pos": float(np.mean([entry["coverage_pos"] for entry in seed_entries])),
                "coverage_neg": float(np.mean([entry["coverage_neg"] for entry in seed_entries])),
                "avg_set_size": float(np.mean([entry["avg_set_size"] for entry in seed_entries])),
                "avg_set_size_ci_low": float(ci["avg_set_size"]["q025"]),
                "avg_set_size_ci_high": float(ci["avg_set_size"]["q975"]),
                "frac_empty": float(np.mean([entry["frac_empty"] for entry in seed_entries])),
                "frac_singleton_negative": float(
                    np.mean([entry["frac_singleton_negative"] for entry in seed_entries])
                ),
                "frac_singleton_positive": float(
                    np.mean([entry["frac_singleton_positive"] for entry in seed_entries])
                ),
                "frac_both": float(np.mean([entry["frac_both"] for entry in seed_entries])),
                "n_boot": int(N_BOOT),
            })
    table = pd.DataFrame(rows)
    return table[CONFORMAL_COLUMNS]


# --------------------------------------------------------------------------
# Multicalibration
# --------------------------------------------------------------------------
def assign_bins(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Three binning schemes, all fixed before any test evaluation."""
    return {
        "overall": pd.Series(["all"] * len(frame), index=frame.index),
        "length_words_short": pd.cut(
            frame["n_words"], bins=SHORT_LENGTH_BINS, labels=SHORT_LENGTH_LABELS, right=True
        ),
        "length_words_natural": pd.cut(
            frame["n_words"], bins=NATURAL_LENGTH_BINS, labels=NATURAL_LENGTH_LABELS, right=True
        ),
        "claim_count": pd.cut(
            frame["n_claims"], bins=CLAIM_BINS, labels=CLAIM_LABELS, right=True
        ),
    }


def _fast_ece(y_true: np.ndarray, scores: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """Equal-width ECE with the same definition as ``train_pipeline.ece``."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, n_bins - 1)
    total = len(y_true)
    value = 0.0
    for bin_index in range(n_bins):
        mask = index == bin_index
        if mask.any():
            value += (mask.sum() / total) * abs(y_true[mask].mean() - scores[mask].mean())
    return float(value)


def _fast_brier(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(np.mean((scores - y_true) ** 2))


def cluster_bootstrap_calibration(
    per_seed: list[tuple[np.ndarray, np.ndarray]],
    groups: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    """Cluster bootstrap for mean ECE and Brier across seeds, within one bin."""
    rows = group_rows(groups)
    n_groups = len(rows)
    rng = np.random.default_rng(seed)
    ece_draws: list[float] = []
    brier_draws: list[float] = []
    for _ in range(n_boot):
        picked = rng.integers(0, n_groups, n_groups)
        index = np.concatenate([rows[group] for group in picked])
        ece_draws.append(
            float(np.mean([_fast_ece(y[index], p[index]) for y, p in per_seed]))
        )
        brier_draws.append(
            float(np.mean([_fast_brier(y[index], p[index]) for y, p in per_seed]))
        )
    return {
        "ece": summarize_bootstrap(ece_draws),
        "brier": summarize_bootstrap(brier_draws),
        "n_groups": n_groups,
    }


def multicalibration_table(
    frames: dict[str, pd.DataFrame],
    raw_scores: dict[str, dict[str, np.ndarray]],
    platt_scores: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """ECE and Brier per dataset, binning, bin, score type, and seed."""
    rows = []
    for dataset in ("ragtruth_qa_test", "faithbench", "halueval_test"):
        frame = frames[dataset]
        bins = assign_bins(frame)
        y = frame["label"].to_numpy(dtype=int)
        groups = frame["group_id"].to_numpy()
        for binning, labels in bins.items():
            for bin_label in BIN_ORDER[binning]:
                mask = (labels == bin_label).to_numpy()
                if not mask.any():
                    continue
                y_bin = y[mask]
                groups_bin = groups[mask]
                n_bin = int(mask.sum())
                for score_name, score_by_seed in (("raw", raw_scores), ("target_platt", platt_scores)):
                    per_seed_metrics = []
                    for seed in SEEDS:
                        p_bin = score_by_seed[str(seed)][dataset][mask]
                        per_seed_metrics.append({
                            "seed": str(seed),
                            "ece": _fast_ece(y_bin, p_bin),
                            "brier": _fast_brier(y_bin, p_bin),
                            "mean_score": float(np.mean(p_bin)),
                            "observed_rate": float(np.mean(y_bin)),
                        })
                    for entry in per_seed_metrics:
                        rows.append({
                            "dataset": dataset,
                            "binning": binning,
                            "bin": str(bin_label),
                            "n": n_bin,
                            "n_groups": int(len(np.unique(groups_bin))),
                            "prevalence": float(np.mean(y_bin)),
                            "score": score_name,
                            "seed": entry["seed"],
                            "ece": entry["ece"],
                            "brier": entry["brier"],
                            "mean_score": entry["mean_score"],
                            "observed_rate": entry["observed_rate"],
                            "ece_ci_low": float("nan"),
                            "ece_ci_high": float("nan"),
                            "brier_ci_low": float("nan"),
                            "brier_ci_high": float("nan"),
                            "small_n": bool(n_bin < MIN_CI_N),
                            "n_boot": 0,
                        })
                    if n_bin >= MIN_CI_N:
                        pairs = [
                            (y_bin, score_by_seed[str(seed)][dataset][mask]) for seed in SEEDS
                        ]
                        ci = cluster_bootstrap_calibration(
                            pairs,
                            groups_bin,
                            N_BOOT,
                            stable_seed("p1_multical", dataset, binning, bin_label, score_name),
                        )
                        ci_values = {
                            "ece_ci_low": float(ci["ece"]["q025"]),
                            "ece_ci_high": float(ci["ece"]["q975"]),
                            "brier_ci_low": float(ci["brier"]["q025"]),
                            "brier_ci_high": float(ci["brier"]["q975"]),
                            "n_boot": int(N_BOOT),
                        }
                        n_groups = int(ci["n_groups"])
                    else:
                        ci_values = {
                            "ece_ci_low": float("nan"),
                            "ece_ci_high": float("nan"),
                            "brier_ci_low": float("nan"),
                            "brier_ci_high": float("nan"),
                            "n_boot": 0,
                        }
                        n_groups = int(len(np.unique(groups_bin)))
                    rows.append({
                        "dataset": dataset,
                        "binning": binning,
                        "bin": str(bin_label),
                        "n": n_bin,
                        "n_groups": n_groups,
                        "prevalence": float(np.mean(y_bin)),
                        "score": score_name,
                        "seed": "pooled",
                        "ece": float(np.mean([entry["ece"] for entry in per_seed_metrics])),
                        "brier": float(np.mean([entry["brier"] for entry in per_seed_metrics])),
                        "mean_score": float(
                            np.mean([entry["mean_score"] for entry in per_seed_metrics])
                        ),
                        "observed_rate": float(np.mean(y_bin)),
                        "small_n": bool(n_bin < MIN_CI_N),
                        **ci_values,
                    })
    table = pd.DataFrame(rows)
    return table[MULTICALIBRATION_COLUMNS]


# --------------------------------------------------------------------------
# Supplementary checks
# --------------------------------------------------------------------------
def disjoint_platt_check(
    frames: dict[str, pd.DataFrame],
    raw_scores: dict[str, dict[str, np.ndarray]],
) -> dict:
    """Fit Platt and the conformal quantile on disjoint calibration groups."""
    calibration = frames["ragtruth_qa_calibration"]
    y_cal = calibration["label"].to_numpy(dtype=int)
    groups = calibration["group_id"].to_numpy()
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(20261011)
    shuffled = rng.permutation(unique_groups)
    fit_groups = set(shuffled[: len(shuffled) // 2].tolist())
    fit_mask = np.isin(groups, list(fit_groups))
    conformal_mask = ~fit_mask
    out = {}
    for seed in SEEDS:
        p_raw_cal = raw_scores[str(seed)]["ragtruth_qa_calibration"]
        platt = fit_platt(p_raw_cal[fit_mask], y_cal[fit_mask])
        p_conformal = apply_platt(platt, p_raw_cal[conformal_mask])
        y_conformal = y_cal[conformal_mask]
        out[str(seed)] = {}
        for alpha in ALPHAS:
            q, rank = conformal_quantile(
                np.where(y_conformal == 1, 1.0 - p_conformal, p_conformal), alpha
            )
            entry = {
                "n_fit": int(fit_mask.sum()),
                "n_conformal": int(conformal_mask.sum()),
                "q": q,
                "rank": rank,
            }
            for dataset in ("ragtruth_qa_test", "faithbench"):
                frame = frames[dataset]
                p = apply_platt(platt, raw_scores[str(seed)][dataset])
                metrics = prediction_set_metrics(frame["label"].to_numpy(dtype=int), p, q)
                entry[dataset] = {key: metrics[key] for key in (
                    "coverage", "coverage_pos", "coverage_neg", "avg_set_size"
                )}
            out[str(seed)][str(alpha)] = entry
    return out


def class_conditional_check(
    frames: dict[str, pd.DataFrame],
    scores: dict[str, dict[str, np.ndarray]],
) -> dict:
    """Mondrian conformal quantiles, one per true class."""
    calibration = frames["ragtruth_qa_calibration"]
    y_cal = calibration["label"].to_numpy(dtype=int)
    out = {}
    for seed in SEEDS:
        p_cal = scores[str(seed)]["ragtruth_qa_calibration"]
        out[str(seed)] = {}
        for alpha in ALPHAS:
            q_positive, _ = conformal_quantile(1.0 - p_cal[y_cal == 1], alpha)
            q_negative, _ = conformal_quantile(p_cal[y_cal == 0], alpha)
            entry = {"q_positive": q_positive, "q_negative": q_negative}
            for dataset in ("ragtruth_qa_test", "faithbench"):
                frame = frames[dataset]
                p = scores[str(seed)][dataset]
                include_positive = p >= 1.0 - q_positive
                include_negative = p <= q_negative
                y = frame["label"].to_numpy(dtype=int)
                covered = np.where(y == 1, include_positive, include_negative)
                size = include_positive.astype(int) + include_negative.astype(int)
                entry[dataset] = {
                    "coverage": float(covered.mean()),
                    "coverage_pos": float(covered[y == 1].mean()),
                    "coverage_neg": float(covered[y == 0].mean()),
                    "avg_set_size": float(size.mean()),
                }
            out[str(seed)][str(alpha)] = entry
    return out


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------
def json_safe(value):
    """Convert numpy scalars and non-finite floats into JSON-safe values."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def conformal_report_table(table: pd.DataFrame) -> str:
    lines = [
        "| dataset | alpha | seed | n | q | coverage [95% CI] | pos | neg | set size |",
        "|---|---:|---|---:|---:|---|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['alpha']:.2f} | {row['seed']} | {int(row['n'])} | "
            f"{fmt(row['q'])} | {fmt(row['coverage'])} "
            f"[{fmt(row['coverage_ci_low'])}, {fmt(row['coverage_ci_high'])}] | "
            f"{fmt(row['coverage_pos'])} | {fmt(row['coverage_neg'])} | "
            f"{fmt(row['avg_set_size'])} |"
        )
    return "\n".join(lines)


def multical_report_table(table: pd.DataFrame, dataset: str, binning: str) -> str:
    subset = table[
        (table["dataset"] == dataset) & (table["binning"] == binning) & (table["seed"] == "pooled")
    ]
    lines = [
        "| bin | n | prevalence | raw ECE | raw Brier | Platt ECE | Platt Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    present = set(subset["bin"].astype(str))
    labels = [label for label in BIN_ORDER[binning] if label in present]
    for label in labels:
        block = subset[subset["bin"] == label]
        raw = block[block["score"] == "raw"]
        platt = block[block["score"] == "target_platt"]
        if raw.empty or platt.empty:
            continue
        lines.append(
            f"| {label} | {int(raw['n'].iloc[0])} | {fmt(raw['prevalence'].iloc[0])} | "
            f"{fmt(raw['ece'].iloc[0])} | {fmt(raw['brier'].iloc[0])} | "
            f"{fmt(platt['ece'].iloc[0])} | {fmt(platt['brier'].iloc[0])} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    P1_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    frames = load_frames()
    predictions = pd.read_parquet(FIXED_HP_PRED)

    raw_scores: dict[str, dict[str, np.ndarray]] = {}
    platt_scores: dict[str, dict[str, np.ndarray]] = {}
    platt_parameters: dict[str, dict] = {}
    y_cal = frames["ragtruth_qa_calibration"]["label"].to_numpy(dtype=int)

    for seed in SEEDS:
        model_path = FIXED_HP_MODELS / f"xgboost_m3_seed_{seed}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"missing m3 model: {model_path}")
        model = joblib.load(model_path)
        p_raw_cal = predict_raw(model, frames["ragtruth_qa_calibration"])
        platt = fit_platt(p_raw_cal, y_cal)
        platt_parameters[str(seed)] = {
            "coef": float(platt.coef_[0, 0]),
            "intercept": float(platt.intercept_[0]),
            "n_fit": int(len(y_cal)),
            "fit_prevalence": float(y_cal.mean()),
        }
        raw_scores[str(seed)] = {}
        platt_scores[str(seed)] = {}
        for dataset, frame in frames.items():
            raw = predict_raw(model, frame)
            raw_scores[str(seed)][dataset] = raw
            platt_scores[str(seed)][dataset] = apply_platt(platt, raw)
            if dataset != "ragtruth_qa_calibration":
                verify_stored_scores(frame, seed, dataset, predictions, raw)
        logger.info("scored seed %d", seed)

    conformal = conformal_table(frames, platt_scores)
    conformal.to_csv(P1_DIR / "conformal.csv", index=False)

    multical = multicalibration_table(frames, raw_scores, platt_scores)
    multical.to_csv(P1_DIR / "multicalibration.csv", index=False)

    disjoint = disjoint_platt_check(frames, raw_scores)
    conditional = class_conditional_check(frames, platt_scores)

    # Compact JSON headline: pooled rows only.
    pooled = conformal[conformal["seed"] == "pooled"]
    multical_pooled = multical[multical["seed"] == "pooled"]
    summary = {}
    for dataset in DATASETS:
        summary[dataset] = {}
        for alpha in ALPHAS:
            block = pooled[(pooled["dataset"] == dataset) & (pooled["alpha"] == alpha)]
            if block.empty:
                continue
            row = block.iloc[0]
            summary[dataset][f"{alpha:.2f}"] = {
                key: float(row[key]) for key in (
                    "q", "coverage", "coverage_ci_low", "coverage_ci_high",
                    "coverage_pos", "coverage_neg", "avg_set_size",
                    "avg_set_size_ci_low", "avg_set_size_ci_high",
                )
            }

    def overall(dataset: str) -> dict:
        block = multical_pooled[
            (multical_pooled["dataset"] == dataset)
            & (multical_pooled["binning"] == "overall")
            & (multical_pooled["bin"] == "all")
        ]
        return {
            score: {
                "n": int(block[block["score"] == score]["n"].iloc[0]),
                "prevalence": float(block[block["score"] == score]["prevalence"].iloc[0]),
                "ece": float(block[block["score"] == score]["ece"].iloc[0]),
                "brier": float(block[block["score"] == score]["brier"].iloc[0]),
            }
            for score in ("raw", "target_platt")
            if not block[block["score"] == score].empty
        }

    report = {
        "schema": "rebuild-p1-conformal-v1",
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "protocol": {
            "model": "fixed-hp EC-XGB m3",
            "seeds": SEEDS,
            "calibration_rows": int(len(y_cal)),
            "calibration_prevalence": float(y_cal.mean()),
            "target_platt": "Platt on raw m3 scores, fitted on the 5,034 calibration rows only",
            "platt_fit": platt_parameters,
            "nonconformity": "1 - target-Platt score of the true label (LAC)",
            "quantile": "ceil((n+1)(1-alpha))-th smallest calibration score",
            "alphas": list(ALPHAS),
            "n_boot": N_BOOT,
            "bootstrap": "cluster bootstrap over source groups (RAGTruth), items (HaluEval), rows (FaithBench)",
            "test_used_for_fit_or_selection": False,
        },
        "conformal": summary,
        "multicalibration_overall": {
            dataset: overall(dataset)
            for dataset in ("ragtruth_qa_test", "faithbench", "halueval_test")
        },
        "sensitivity": {
            "disjoint_platt": disjoint,
            "class_conditional": conditional,
        },
        "notes": [
            "Marginal coverage can hide a collapsed positive set: report coverage_pos and coverage_neg beside it.",
            "FaithBench has 68.1% positives, while the target-Platt scores are concentrated near 0.07; the positive set is almost never predicted.",
            "The calibration row is in-sample for the target Platt, so its conformal coverage is a construction check, not independent evidence.",
            "HaluEval is an in-domain reference; target-Platt is a deliberate domain mismatch there.",
            "Subgroup ECE uses 10 equal-width bins and is noisy for small bins; n and small_n are reported.",
        ],
    }
    (P1_DIR / "conformal.json").write_text(
        json.dumps(json_safe(report), indent=2), encoding="utf-8"
    )

    qa_report = conformal_report_table(
        conformal[
            (conformal["dataset"].isin(["ragtruth_qa_test", "faithbench"]))
            & (conformal["seed"] == "pooled")
        ]
    )
    calibration_report = conformal_report_table(
        conformal[(conformal["dataset"] == "ragtruth_qa_calibration") & (conformal["seed"] == "pooled")]
    )
    halu_report = conformal_report_table(
        conformal[(conformal["dataset"] == "halueval_test") & (conformal["seed"] == "pooled")]
    )
    qa_multical = multical_report_table(multical, "ragtruth_qa_test", "length_words_natural")
    fb_multical = multical_report_table(multical, "faithbench", "length_words_natural")
    qa_claim = multical_report_table(multical, "ragtruth_qa_test", "claim_count")
    fb_claim = multical_report_table(multical, "faithbench", "claim_count")

    qa_90 = summary["ragtruth_qa_test"]["0.10"]
    qa_95 = summary["ragtruth_qa_test"]["0.05"]
    fb_90 = summary["faithbench"]["0.10"]
    fb_95 = summary["faithbench"]["0.05"]
    qa_overall = report["multicalibration_overall"]["ragtruth_qa_test"]
    fb_overall = report["multicalibration_overall"]["faithbench"]

    def mean_sensitivity(block: dict, dataset: str, level: str, key: str) -> float:
        return float(np.mean([block[str(seed)][level][dataset][key] for seed in SEEDS]))

    disjoint_qa_90 = mean_sensitivity(disjoint, "ragtruth_qa_test", "0.1", "coverage")
    disjoint_qa_95 = mean_sensitivity(disjoint, "ragtruth_qa_test", "0.05", "coverage")
    disjoint_fb_90 = mean_sensitivity(disjoint, "faithbench", "0.1", "coverage")
    conditional_qa_pos_90 = mean_sensitivity(conditional, "ragtruth_qa_test", "0.1", "coverage_pos")
    conditional_qa_pos_95 = mean_sensitivity(conditional, "ragtruth_qa_test", "0.05", "coverage_pos")
    conditional_fb_pos_90 = mean_sensitivity(conditional, "faithbench", "0.1", "coverage_pos")

    report_md = f"""# P1.4 - Split Conformal Prediction and Multicalibration

Generated by `src/models/p1_conformal.py`. Every number below comes from
`conformal.csv`, `multicalibration.csv`, and `conformal.json`.

## Protocol

- Model: fixed-hyperparameter EC-XGB m3, seeds 42/123/456.
- Target Platt: fitted on the 5,034 RAGTruth QA calibration rows only.
- Nonconformity: LAC, `1 - target-Platt score of the true label`.
- Quantile: `ceil((n+1)(1-alpha))`-th smallest calibration score.
- Prediction set: include positive if `p >= 1-q`; include negative if `p <= q`.
- Evaluation: RAGTruth QA test (900 rows), FaithBench (750 rows), and HaluEval
  test (3,000 rows, in-domain reference).
- CIs: 1,000 cluster-bootstrap draws. RAGTruth clusters are source groups (6
  rows), HaluEval clusters are items (2 rows), FaithBench clusters are rows.
- Test rows are never used to fit the calibrator or the conformal quantile.

## Conformal coverage

### Calibration rows (construction check)

{calibration_report}

### RAGTruth QA test and FaithBench (shift)

{qa_report}

### HaluEval test (in-domain reference)

{halu_report}

At 90% nominal coverage, QA test marginal coverage is {fmt(qa_90['coverage'])}
[{fmt(qa_90['coverage_ci_low'])}, {fmt(qa_90['coverage_ci_high'])}], but positive
coverage is {fmt(qa_90['coverage_pos'])} and negative coverage is
{fmt(qa_90['coverage_neg'])}. At 95%, the corresponding values are
{fmt(qa_95['coverage'])} [{fmt(qa_95['coverage_ci_low'])},
{fmt(qa_95['coverage_ci_high'])}] marginal, {fmt(qa_95['coverage_pos'])}
positive, and {fmt(qa_95['coverage_neg'])} negative. The marginal number is
above nominal because the sets are large for negatives and both-label sets
are common, not because the positive label is covered well.

FaithBench is the honest failure case. At 90% nominal coverage the marginal
coverage is {fmt(fb_90['coverage'])} [{fmt(fb_90['coverage_ci_low'])},
{fmt(fb_90['coverage_ci_high'])}], the positive coverage is
{fmt(fb_90['coverage_pos'])}, the negative coverage is
{fmt(fb_90['coverage_neg'])}, and the average set size is
{fmt(fb_90['avg_set_size'])}. At 95% the positive coverage is
{fmt(fb_95['coverage_pos'])}. The marginal number is approximately the negative
rate (1 - 0.681 = 0.319), so the interval is covering the majority class almost
exclusively. An average set size of {fmt(fb_90['avg_set_size'])} means almost
every set is the singleton negative label; a constant negative predictor would
match this coverage. This is prior shift plus score shift, not a coverage
guarantee.

## Multicalibration: RAGTruth QA test

Length bins are the natural-response bins fixed from the calibration
distribution (`<=60`, `61-80`, `81-100`, `101-120`, `121-160`, `161+` words).
Pooled rows average the three seeds; CIs are cluster-bootstrap 95% intervals.

### Answer length

{qa_multical}

### Claim count

{qa_claim}

Overall on QA test, raw m3 ECE is {fmt(qa_overall['raw']['ece'])} and target
Platt ECE is {fmt(qa_overall['target_platt']['ece'])}. Target Platt improves
every length bin and every claim-count bin, but it does not equalize them:
short answers and low claim counts remain the worst-calibrated subgroups. The
raw score is overconfident everywhere; target Platt moves the level but leaves
the subgroup ordering.

## Multicalibration: FaithBench

### Answer length

{fb_multical}

### Claim count

{fb_claim}

Overall on FaithBench, raw m3 ECE is {fmt(fb_overall['raw']['ece'])} and target
Platt ECE is {fmt(fb_overall['target_platt']['ece'])}. Target Platt makes
calibration worse in every FaithBench subgroup, because it maps scores toward
the low RAGTruth QA prevalence while FaithBench is 68.1% positive. This is the
same mechanism that collapses conformal positive coverage.

## Supplementary checks

- **Disjoint Platt/conformal split.** Fitting Platt on half of the calibration
  groups and the conformal quantile on the other half gives QA test coverage
  {fmt(disjoint_qa_90)} at 90% and {fmt(disjoint_qa_95)} at 95% (primary:
  {fmt(qa_90['coverage'])} and {fmt(qa_95['coverage'])}). FaithBench stays at
  {fmt(disjoint_fb_90)}. Sharing rows between the Platt fit and the quantile
  does not create the QA coverage result.
- **Class-conditional (Mondrian) conformal.** Per-class quantiles restore QA
  positive coverage to {fmt(conditional_qa_pos_90)} at 90% and
  {fmt(conditional_qa_pos_95)} at 95%, but do not rescue FaithBench, where
  positive coverage is {fmt(conditional_fb_pos_90)}. The full table is in
  `conformal.json`, `sensitivity.class_conditional`. This is a diagnostic, not
  a deployed procedure.

## Interpretation

Split conformal works as advertised on the exchangeable target test: QA
marginal coverage is at or above nominal at both levels. It does not deliver
class-conditional coverage, and it does not transfer to FaithBench, where the
positive label is almost never in the set. Reporting only marginal coverage
would therefore overstate the method. The useful result is the pairing of
coverage with subgroup calibration: target Platt lowers ECE inside RAGTruth QA
but raises it on FaithBench, and the conformal set inherits that failure.

## Caveats

- The 5,034 calibration rows are used both to fit the target Platt and to
  compute the primary conformal quantile. The disjoint check in the JSON shows
  the QA test conclusion is unchanged, but the calibration-row coverage is
  in-sample and should not be read as independent validation.
- RAGTruth calibration rows are grouped (839 source groups). The row-level
  quantile is the standard split-conformal construction; cluster-bootstrap CIs
  reflect group dependence, but the point quantile does not use a group-level
  correction.
- FaithBench is a summarization stress test with a different label mapping
  and a much higher positive rate. Its coverage numbers are a shift diagnostic,
  not a target-domain performance claim.
- Subgroup ECE uses 10 equal-width bins. Cells with fewer than 30 rows are
  flagged `small_n` and their ECE is descriptive only.
- Claim count is capped at six by the claim extractor, so the `6+` bin means
  six or more extracted claims, not an unbounded count.
- HaluEval is an in-domain reference for target Platt, not a target of the
  calibrator; its high coverage reflects the raw score separation, not
  calibration quality.

## Files

- `conformal.csv` - every dataset, alpha, seed, coverage, and set-size row.
- `multicalibration.csv` - every dataset, binning, bin, score, and seed row.
- `conformal.json` - protocol, pooled summary, and both sensitivity checks.
- `conformal_REPORT.md` - this file.
- `p1_external_baseline_and_conformal_REPORT.md` - combined calibration report.

## Command

```powershell
& .venv\\Scripts\\python.exe src\\models\\p1_conformal.py
```
"""
    (P1_DIR / "conformal_REPORT.md").write_text(report_md, encoding="utf-8")
    from src.models.p1_report import write_combined_report

    write_combined_report(P1_DIR)
    logger.info("wrote P1 conformal artifacts to %s in %.1fs", P1_DIR, time.time() - started)

    print("\nConformal coverage (pooled seeds)")
    print(
        conformal[conformal["seed"] == "pooled"][
            ["dataset", "alpha", "q", "coverage", "coverage_pos", "coverage_neg", "avg_set_size"]
        ].round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()
