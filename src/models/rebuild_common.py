"""Shared helpers for the WP2/WP3 rebuild (calibration and intervals).

The rebuild keeps every new artifact under ``artifacts/results/rebuild`` and
never touches the frozen ``b2``/``b4``/``b6``/``review`` outputs.  This module
holds only the pieces both scripts need: canonical paths, the m3 feature
matrix, metric definitions, Platt/isotonic/Cox calibrators, and bootstrap
summaries.
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.config import (  # noqa: E402
    BOOTSTRAP_SEED,
    DATA_PROCESSED,
    MODELS_DIR,
    RESULTS_DIR,
    SEEDS,
)
from src.models.run_b6_modified import (  # noqa: E402
    SOURCE_COLUMN,
    feature_matrix,
    variant_features,
)
from src.models.train_pipeline import ece  # noqa: E402

# --------------------------------------------------------------------------
# Canonical paths and protocol constants
# --------------------------------------------------------------------------
REBUILD_RESULTS = RESULTS_DIR / "rebuild"
CALIBRATION_DIR = REBUILD_RESULTS / "calibration"
INTERVALS_DIR = REBUILD_RESULTS / "intervals"
FIXED_HP_DIR = REBUILD_RESULTS / "fixed_hp"
FIXED_HP_PRED = FIXED_HP_DIR / "fixed_hp_predictions.parquet"
FIXED_HP_MODELS = MODELS_DIR / "rebuild" / "fixed_hp"

B6_PRED = RESULTS_DIR / "b6" / "b6_predictions.parquet"
B6_MODELS = MODELS_DIR / "b6"

FEATURES_FULL = DATA_PROCESSED / "features_full.parquet"
CLAIM_FEATURES = DATA_PROCESSED / "claim_features.parquet"
CALIBRATION_FEATURES = DATA_PROCESSED / "calibration_features.parquet"
EXTERNAL_FEATURES = DATA_PROCESSED / "external_features.parquet"

VARIANT = "m3"
THRESHOLD = 0.5
ECE_BINS = 10
FPR_TARGET = 0.05

N_BOOT_CALIBRATION = 1000
N_BOOT_INTERVALS = 2000

# Fit-split prevalences, reported to four decimals in the rebuild plan.
CALIBRATION_PREVALENCE = 0.31068732618196265  # 5,034 RAGTruth QA calibration rows
TEST_PREVALENCE = 0.17777777777777778  # 900 RAGTruth QA test rows (160 positives)

BOOTSTRAP_SEED_CALIBRATION = BOOTSTRAP_SEED  # 777
BOOTSTRAP_SEED_INTERVALS = BOOTSTRAP_SEED + 1  # 778
BOOTSTRAP_SEED_LABEL_BUDGET = BOOTSTRAP_SEED + 2  # 779

DATASETS = ("halueval_test", "ragtruth_qa_test", "ragtruth_all_test")


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def m3_feature_names() -> list[str]:
    """Ordered 35-feature m3 matrix (26 base + 8 claim + source indicator)."""
    return list(variant_features(VARIANT))


def load_halueval_features() -> pd.DataFrame:
    """HaluEval rows with claim features merged and ``source_ragtruth = 0``."""
    base = pd.read_parquet(FEATURES_FULL)
    claims = pd.read_parquet(CLAIM_FEATURES)
    df = base.merge(claims, on="sample_id", how="left", validate="one_to_one")
    claim_columns = [c for c in m3_feature_names() if c in claims.columns]
    if df[claim_columns].isna().any().any():
        raise ValueError("claim features missing after merge")
    df[SOURCE_COLUMN] = 0.0
    return df


def load_calibration_features() -> pd.DataFrame:
    """RAGTruth QA calibration rows (5,034) with ``source_ragtruth = 1``."""
    df = pd.read_parquet(CALIBRATION_FEATURES)
    df[SOURCE_COLUMN] = 1.0
    return df


def predict_raw(model, frame: pd.DataFrame, names: list[str] | None = None) -> np.ndarray:
    """Raw positive-class probability from a fitted m3 model."""
    names = names or m3_feature_names()
    return model.predict_proba(feature_matrix(frame, names, log_lengths=True))[:, 1]


def load_predictions(path: Path = FIXED_HP_PRED) -> pd.DataFrame:
    """Prediction table with variant/seed/dataset/sample_id/label/score."""
    df = pd.read_parquet(path)
    required = {"variant", "seed", "dataset", "sample_id", "label", "score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def group_frame(dataset: str) -> pd.DataFrame:
    """Return ``sample_id, label, group_id`` for an evaluation dataset.

    Group ids follow the rebuild plan: HaluEval uses ``item_idx`` (the two
    prompt variants of one item stay together); external datasets use
    ``source_group_id`` from ``external_features.parquet``.
    """
    if dataset == "halueval_test":
        base = pd.read_parquet(FEATURES_FULL)
        frame = base.loc[base["split"] == "test", ["sample_id", "item_idx", "label"]].copy()
        frame = frame.rename(columns={"item_idx": "group_id"})
    elif dataset == "ragtruth_qa_test":
        ext = pd.read_parquet(EXTERNAL_FEATURES)
        frame = ext.loc[
            (ext["source_dataset"] == "ragtruth")
            & (ext["task"] == "qa")
            & (ext["split"] == "test"),
            ["sample_id", "source_group_id", "label"],
        ].rename(columns={"source_group_id": "group_id"})
    elif dataset == "ragtruth_all_test":
        ext = pd.read_parquet(EXTERNAL_FEATURES)
        frame = ext.loc[
            (ext["source_dataset"] == "ragtruth") & (ext["split"] == "test"),
            ["sample_id", "source_group_id", "label"],
        ].rename(columns={"source_group_id": "group_id"})
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    frame = frame.copy()
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{dataset}: duplicate sample_id in group frame")
    if frame["group_id"].isna().any():
        raise ValueError(f"{dataset}: missing group ids")
    frame["group_id"] = frame["group_id"].astype(str)
    frame["label"] = frame["label"].astype(int)
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def clean_scores(scores) -> np.ndarray:
    """Clip probabilities into (0, 1) and replace non-finite values with 0.5."""
    arr = np.asarray(scores, dtype=float)
    arr = np.nan_to_num(arr, nan=0.5, posinf=1.0 - 1e-12, neginf=1e-12)
    return np.clip(arr, 1e-12, 1.0 - 1e-12)


def recall_at_fpr(y_true, y_score, max_fpr: float = FPR_TARGET) -> float:
    """Maximum recall at a false-positive rate no larger than ``max_fpr``."""
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, clean_scores(y_score))
    allowed = fpr <= max_fpr
    if not allowed.any():
        return 0.0
    return float(tpr[allowed].max())


def metric_bundle(y_true, y_score, threshold: float = THRESHOLD) -> dict:
    """Discrimination, calibration, and operating-point metrics at one score."""
    y_true = np.asarray(y_true, dtype=int)
    scores = clean_scores(y_score)
    predicted = scores >= threshold
    prevalence = float(y_true.mean())
    brier = float(brier_score_loss(y_true, scores))
    prior_brier = prevalence * (1.0 - prevalence)
    binary = len(np.unique(y_true)) > 1
    return {
        "n": int(len(y_true)),
        "prevalence": prevalence,
        "threshold": float(threshold),
        "ece": float(ece(y_true, scores, ECE_BINS)),
        "brier": brier,
        "brier_skill_score": (
            float(1.0 - brier / prior_brier) if prior_brier > 0 else float("nan")
        ),
        "auroc": float(roc_auc_score(y_true, scores)) if binary else float("nan"),
        "pr_auc": float(average_precision_score(y_true, scores)) if binary else float("nan"),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "predicted_positive_rate": float(predicted.mean()),
        "recall_at_5pct_fpr": recall_at_fpr(y_true, scores, FPR_TARGET),
    }


METRIC_KEYS = (
    "ece",
    "brier",
    "brier_skill_score",
    "auroc",
    "pr_auc",
    "f1",
    "precision",
    "recall",
    "predicted_positive_rate",
    "recall_at_5pct_fpr",
)

CI_KEYS = ("ece", "brier", "brier_skill_score", "auroc", "pr_auc", "f1")


# --------------------------------------------------------------------------
# Calibrators
# --------------------------------------------------------------------------
def fit_platt(scores, labels) -> LogisticRegression:
    """Platt scaling: logistic regression of the label on the raw score.

    ``C=1.0`` matches the frozen B6 calibration protocol, so the rebuilt
    table stays comparable with the earlier artifacts.  The calibrator is
    always fit on a held-out split, never on the evaluation rows.
    """
    return LogisticRegression(max_iter=2000).fit(
        clean_scores(scores).reshape(-1, 1), np.asarray(labels, dtype=int)
    )


def apply_platt(model: LogisticRegression, scores) -> np.ndarray:
    return model.predict_proba(clean_scores(scores).reshape(-1, 1))[:, 1]


def fit_isotonic(scores, labels) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip").fit(
        clean_scores(scores), np.asarray(labels, dtype=int)
    )


def apply_isotonic(model: IsotonicRegression, scores) -> np.ndarray:
    return np.clip(model.predict(clean_scores(scores)), 0.0, 1.0)


def logit(scores) -> np.ndarray:
    scores = clean_scores(scores)
    return np.log(scores / (1.0 - scores))


def fit_cox(scores, labels) -> LogisticRegression:
    """Cox calibration regression of the label on logit(raw score).

    A perfectly calibrated score has slope 1 and intercept 0.
    """
    return LogisticRegression(max_iter=2000).fit(
        logit(scores).reshape(-1, 1), np.asarray(labels, dtype=int)
    )


def calibration_coefficients(scores, labels) -> dict:
    """Point estimates for the applied Platt fit and the Cox slope."""
    platt = fit_platt(scores, labels)
    cox = fit_cox(scores, labels)
    return {
        "platt_coef": float(platt.coef_[0, 0]),
        "platt_intercept": float(platt.intercept_[0]),
        "cal_slope": float(cox.coef_[0, 0]),
        "cal_intercept": float(cox.intercept_[0]),
    }


def bootstrap_calibration_coefficients(
    scores, labels, n_boot: int = N_BOOT_CALIBRATION, seed: int = BOOTSTRAP_SEED_CALIBRATION
) -> dict:
    """Row-bootstrap percentile CIs for Platt and Cox coefficients."""
    scores = clean_scores(scores)
    labels = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {
        "platt_coef": [],
        "platt_intercept": [],
        "cal_slope": [],
        "cal_intercept": [],
    }
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        y_boot = labels[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        values = calibration_coefficients(scores[idx], y_boot)
        for key, value in values.items():
            draws[key].append(value)
    return {key: summarize_bootstrap(values) for key, values in draws.items()}


# --------------------------------------------------------------------------
# Bootstrap helpers
# --------------------------------------------------------------------------
def stable_seed(*parts) -> int:
    """Deterministic 32-bit seed derived from readable parts (stable across runs)."""
    text = "|".join(str(part) for part in parts)
    return int(zlib.crc32(text.encode("utf-8")))


def summarize_bootstrap(values) -> dict:
    """Mean, SD, and 2.5/97.5 percentiles of a bootstrap distribution."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "sd": float("nan"),
            "q025": float("nan"),
            "q975": float("nan"),
            "n_valid": 0,
        }
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "q025": float(np.percentile(arr, 2.5)),
        "q975": float(np.percentile(arr, 97.5)),
        "n_valid": int(arr.size),
    }


def bootstrap_metric_cis(
    y_true,
    y_score,
    n_boot: int = N_BOOT_CALIBRATION,
    seed: int = BOOTSTRAP_SEED_CALIBRATION,
    threshold: float = THRESHOLD,
) -> dict:
    """Row-bootstrap (i.i.d. rows) percentile CIs for the calibration table.

    Rows are resampled independently, as requested for WP2.  The grouped
    two-level bootstrap in WP3 handles the HaluEval item pairing.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = clean_scores(y_score)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {key: [] for key in CI_KEYS}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        y_boot = y_true[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        bundle = metric_bundle(y_boot, scores[idx], threshold)
        for key in CI_KEYS:
            draws[key].append(bundle[key])
    return {key: summarize_bootstrap(values) for key, values in draws.items()}


def bootstrap_metric_cis_pooled(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    n_boot: int = N_BOOT_CALIBRATION,
    seed: int = BOOTSTRAP_SEED_CALIBRATION,
    threshold: float = THRESHOLD,
) -> dict:
    """Row-bootstrap CIs for a seed-averaged metric.

    All pairs must share the same row order (the three seeds predict the same
    evaluation rows).  Each draw uses one row index set for every seed so the
    pooled interval keeps the cross-seed correlation.
    """
    if not pairs:
        raise ValueError("pairs is empty")
    lengths = {len(y) for y, _ in pairs}
    if len(lengths) != 1:
        raise ValueError(f"pairs have different lengths: {lengths}")
    n = lengths.pop()
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {key: [] for key in CI_KEYS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(pairs[0][0][idx])) < 2:
            continue
        bundles = [metric_bundle(y[idx], score[idx], threshold) for y, score in pairs]
        for key in CI_KEYS:
            draws[key].append(float(np.mean([b[key] for b in bundles])))
    return {key: summarize_bootstrap(values) for key, values in draws.items()}


def draw_group_row_indices(group_sizes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cluster-bootstrap row indices for contiguous groups.

    ``group_sizes`` is the number of rows in each group, in the same order as
    the sorted evaluation frame.  Groups are drawn with replacement, then each
    row of a drawn group is repeated as many times as the group was drawn.
    """
    group_sizes = np.asarray(group_sizes, dtype=int)
    n_groups = len(group_sizes)
    counts = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups))
    row_multiplicity = np.repeat(counts, group_sizes)
    return np.repeat(np.arange(int(group_sizes.sum())), row_multiplicity)


def f1_and_auroc(y_true, y_score, threshold: float = THRESHOLD) -> tuple[float, float]:
    """Fast F1/AUROC pair used inside the 2,000-draw interval bootstrap.

    The implementation avoids sklearn's per-call validation overhead.  AUROC is
    the Mann-Whitney statistic with average ranks for tied scores, which is the
    same definition as ``sklearn.metrics.roc_auc_score``.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(y_score, dtype=np.float64)
    predicted = scores >= threshold
    tp = int(np.count_nonzero(predicted & (y_true == 1)))
    fp = int(np.count_nonzero(predicted & (y_true == 0)))
    fn = int(np.count_nonzero(~predicted & (y_true == 1)))
    denominator = 2 * tp + fp + fn
    f1 = float(2 * tp / denominator) if denominator else 0.0

    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return f1, float("nan")

    order = np.argsort(scores, kind="mergesort")
    ordered = scores[order]
    boundaries = np.flatnonzero(ordered[1:] != ordered[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n]))
    average_ranks = (starts + ends + 1) / 2.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.repeat(average_ranks, ends - starts)
    auroc = float(
        (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )
    return f1, auroc
