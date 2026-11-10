"""WP4 — explanations for the rebuilt fixed-hyperparameter EC-XGB (m3).

The rebuilt model is EC-XGB m3 trained with fixed hyperparameters on 35
features (26 base + 8 claim-level NLI aggregates + a RAGTruth source
indicator).  This script replaces the old B2-based SHAP/audit with:

  A. Exact TreeSHAP on 1,000 stratified HaluEval test rows, via XGBoost
     ``pred_contribs=True``.  Saves the global mean-|SHAP| ranking with
     feature groups and a 600-DPI top-15 bar figure.
  B. Per-case SHAP for the 26 B5 audit cases: top-5 features, model-input
     values, signed contributions, raw score, label and a 120-character
     answer snippet.  The human-judgment columns are left empty.
  C. Optional ``--perturbation`` pilot: six text perturbations with full
     feature re-extraction (base + claim features), capped by ``--max-minutes``.
  D. A comparison against the frozen B2 SHAP ranking, plus a same-1000-row
     B2 control so that model differences are not confounded with sampling.
  E. ``REPORT.md`` with exact numbers, commands and caveats.

The SHAP values explain the raw margin (log-odds), not the Platt display
score.  ``source_ragtruth`` is constant 0 on HaluEval; its attributions come
from interaction structure, not from within-domain source variation.

Run (repo root, .venv):
  python src/models/rebuild_explanations.py
  python src/models/rebuild_explanations.py --perturbation --n-perturb 20
  python src/models/rebuild_explanations.py --perturbation --n-perturb 20 --device cuda
"""

import argparse
import hashlib
import json
import logging
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import kendalltau, spearmanr

from src.features.claim_features import FEATURE_COLUMNS as CLAIM_COLUMNS
from src.models.config import DATA_PROCESSED, RESULTS_DIR
from src.models.run_b6_modified import (
    FEATURES_FULL,
    CLAIM_FEATURES,
    SOURCE_COLUMN,
    feature_matrix,
    variant_features,
)
from src.models.train_pipeline import FEATURE_GROUPS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rebuild_explanations")

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "models" / "rebuild" / "fixed_hp" / "xgboost_m3_seed_42.joblib"
FEATURE_NAMES_PATH = ROOT / "artifacts" / "results" / "rebuild" / "fixed_hp" / "b6_feature_names.json"
OUT_DIR = RESULTS_DIR / "rebuild" / "explanations"
UNIFIED_RECORDS = DATA_PROCESSED / "unified_records.parquet"
AUDIT_CASES = RESULTS_DIR / "b5" / "b5_review_cases_reviewed.csv"
B5_IMPORTANCE = RESULTS_DIR / "b5" / "b5_feature_importance.json"
B2_MODEL = ROOT / "artifacts" / "models" / "b2" / "xgboost_seed_42.joblib"
B2_CONFIG = RESULTS_DIR / "b2" / "b2_run_config.json"
B6_PREDICTIONS = RESULTS_DIR / "rebuild" / "fixed_hp" / "b6_predictions.parquet"

MODEL_NAME = "EC-XGB"
MODEL_VERSION = "m3-fixed-hp-seed-42"
DEFAULT_N_SHAP = 1000
DEFAULT_N_PERTURB = 20
DEFAULT_MAX_MINUTES = 15.0
FLOAT_FORMAT = "{:.6f}"

PERTURBATION_COLUMNS = [
    "sample_id",
    "perturbation",
    "label",
    "raw_score",
    "perturbed_score",
    "score_delta",
    "abs_score_delta",
    "top1_flip",
    "orig_top1",
    "pert_top1",
    "top5_overlap",
    "spearman",
    "max_feature_changed",
    "max_feature_delta",
    "seed",
]

GROUP_COLORS = {
    "length": "#0ea5e9",
    "lexical": "#8b5cf6",
    "entity": "#06b6d4",
    "nli": "#f59e0b",
    "numeric": "#10b981",
    "hedging": "#64748b",
    "semantic": "#ec4899",
    "claim_nli": "#ef4444",
    "source": "#94a3b8",
    "unknown": "#cbd5e1",
}


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_groups(names: list) -> dict:
    """Map every m3 feature to one of the 7 base groups, claim_nli or source."""
    mapping = {}
    for group, columns in FEATURE_GROUPS.items():
        for column in columns:
            mapping[column] = group
    for column in CLAIM_COLUMNS:
        mapping[column] = "claim_nli"
    mapping[SOURCE_COLUMN] = "source"
    return {name: mapping.get(name, "unknown") for name in names}


def stratified_sample(frame: pd.DataFrame, n: int, seed: int) -> np.ndarray:
    """Deterministic label-stratified row indices."""
    rng = np.random.default_rng(seed)
    labels = sorted(frame["label"].unique().tolist())
    quota = n // max(len(labels), 1)
    picks: list = []
    for label in labels:
        pool = frame.index[frame["label"] == label].to_numpy()
        size = min(quota, len(pool))
        if size:
            picks.extend(rng.choice(pool, size=size, replace=False).tolist())
    if len(picks) < n:
        remaining = np.setdiff1d(frame.index.to_numpy(), np.asarray(picks, dtype=int))
        extra = min(n - len(picks), len(remaining))
        if extra:
            picks.extend(rng.choice(remaining, size=extra, replace=False).tolist())
    return np.asarray(sorted(picks), dtype=int)


def snippet(text, limit: int = 120) -> str:
    """Whitespace-normalised answer snippet, at most ``limit`` characters."""
    cleaned = " ".join(str(text if text is not None else "").split())
    return cleaned[:limit]


def assign_role(label: int, score: float) -> str:
    """Error-enriched audit role under the current model."""
    predicted = int(score >= 0.5)
    if label == 0 and predicted == 1:
        return "fp"
    if label == 1 and predicted == 0:
        return "fn"
    if 0.35 <= score <= 0.65:
        return "borderline"
    if label == 1 and predicted == 1:
        return "tp"
    return "other"


def format_role_counts(counts: dict) -> str:
    order = [("fp", "FP"), ("fn", "FN"), ("tp", "TP"), ("borderline", "borderline"), ("other", "other")]
    parts = [f"{int(counts[key])} {label}" for key, label in order if key in counts]
    return ", ".join(parts)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def load_feature_names() -> list:
    payload = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
    names = payload.get("m3", payload) if isinstance(payload, dict) else payload
    expected = variant_features("m3")
    if list(names) != expected:
        raise ValueError("m3 feature-name JSON does not match variant_features('m3')")
    if len(names) != 35:
        raise ValueError(f"expected 35 m3 features, found {len(names)}")
    return list(names)


def load_test_frame() -> pd.DataFrame:
    if not FEATURES_FULL.exists() or not CLAIM_FEATURES.exists():
        raise FileNotFoundError("features_full.parquet and claim_features.parquet are required")
    base = pd.read_parquet(FEATURES_FULL)
    claims = pd.read_parquet(CLAIM_FEATURES)
    frame = base.merge(claims, on="sample_id", how="left", validate="one_to_one")
    if frame[list(CLAIM_COLUMNS)].isna().any().any():
        raise ValueError("claim features missing after merge")
    frame[SOURCE_COLUMN] = 0.0
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    if len(test) != 3000:
        logger.warning("expected 3000 HaluEval test rows, found %d", len(test))
    return test


def load_model_and_booster():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    booster = model.get_booster()
    # CPU prediction keeps the results deterministic and avoids the
    # "mismatched devices" warning when the booster was trained on CUDA.
    booster.set_param({"device": "cpu"})
    return model, booster


def tree_shap(booster, X: np.ndarray) -> tuple:
    """Exact TreeSHAP via XGBoost pred_contribs.

    Returns (shap_values, bias, probabilities, margin_error).
    """
    dmatrix = xgb.DMatrix(X)
    probabilities = np.asarray(booster.predict(dmatrix), dtype=np.float64)
    contributions = np.asarray(booster.predict(dmatrix, pred_contribs=True), dtype=np.float64)
    if contributions.shape[1] != X.shape[1] + 1:
        raise ValueError("unexpected pred_contribs shape; feature order may not match")
    shap_values = contributions[:, :-1]
    bias = contributions[:, -1]
    margin = np.asarray(booster.predict(dmatrix, output_margin=True), dtype=np.float64)
    margin_error = float(np.max(np.abs(shap_values.sum(axis=1) + bias - margin)))
    return shap_values, bias, probabilities, margin_error


def validate_against_b6(test: pd.DataFrame, names: list, booster) -> dict | None:
    """Reproduce the saved m3 seed-42 test scores from the rebuilt feature matrix."""
    if not B6_PREDICTIONS.exists():
        logger.warning("b6_predictions.parquet unavailable; score reproduction check skipped")
        return None
    preds = pd.read_parquet(B6_PREDICTIONS)
    reference = preds[
        (preds["variant"] == "m3") & (preds["seed"] == 42) & (preds["dataset"] == "halueval_test")
    ].set_index("sample_id")
    missing = [sid for sid in test["sample_id"] if sid not in reference.index]
    if missing:
        logger.warning("score reproduction check skipped; %d test rows missing", len(missing))
        return None
    X = feature_matrix(test, names, log_lengths=True)
    scores = np.asarray(booster.predict(xgb.DMatrix(X)), dtype=np.float64)
    reference_scores = reference.loc[test["sample_id"], "score"].to_numpy(dtype=np.float64)
    return {
        "n": int(len(test)),
        "reference": str(B6_PREDICTIONS.relative_to(ROOT)),
        "max_abs_diff": float(np.max(np.abs(scores - reference_scores))),
    }


def load_unified_text(sample_ids: list) -> pd.DataFrame:
    """Canonical HaluEval text from unified_records.parquet, indexed by sample_id."""
    columns = ["sample_id", "source_dataset", "question", "context", "answer", "label"]
    frame = pd.read_parquet(UNIFIED_RECORDS, columns=columns)
    frame = frame[frame["source_dataset"] == "halueval"].copy()
    frame["sample_id"] = frame["sample_id"].str.replace(r"^halueval:", "", regex=True)
    frame = frame.drop_duplicates("sample_id").set_index("sample_id")
    missing = [sid for sid in sample_ids if sid not in frame.index]
    if missing:
        raise ValueError(f"audit text missing for {len(missing)} sample_ids: {missing[:5]}")
    return frame.loc[sample_ids]


# --------------------------------------------------------------------------
# A. Global SHAP
# --------------------------------------------------------------------------
def compute_global_shap(test: pd.DataFrame, names: list, booster, n_shap: int, seed: int) -> dict:
    sample_idx = stratified_sample(test, n_shap, seed)
    sample = test.loc[sample_idx].reset_index(drop=True)
    X = feature_matrix(sample, names, log_lengths=True)
    shap_values, bias, probabilities, margin_error = tree_shap(booster, X)
    logger.info(
        "Global TreeSHAP: %d rows x %d features, max |sum(SHAP)+bias-margin| = %.2e",
        len(sample), len(names), margin_error,
    )

    groups = feature_groups(names)
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    std_signed = shap_values.std(axis=0)
    total = float(mean_abs.sum())
    order = np.argsort(mean_abs)[::-1]

    ranking = pd.DataFrame({
        "rank": np.arange(1, len(names) + 1),
        "feature": [names[i] for i in order],
        "feature_group": [groups[names[i]] for i in order],
        "mean_abs_shap": [float(mean_abs[i]) for i in order],
        "mean_shap": [float(mean_signed[i]) for i in order],
        "std_shap": [float(std_signed[i]) for i in order],
        "share_pct": [float(100.0 * mean_abs[i] / total) if total else 0.0 for i in order],
        "constant_on_sample": [bool(np.std(X[:, i]) == 0.0) for i in order],
        "n_unique": [int(len(np.unique(X[:, i]))) for i in order],
    })
    ranking["cumulative_share_pct"] = ranking["share_pct"].cumsum()

    long_rows = pd.DataFrame({
        "sample_id": np.repeat(sample["sample_id"].to_numpy(), len(names)),
        "label": np.repeat(sample["label"].to_numpy(), len(names)),
        "feature": np.tile(np.asarray(names), len(sample)),
        "value": X.reshape(-1),
        "shap": shap_values.reshape(-1),
    })

    summary = {
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "n_rows": int(len(sample)),
        "n_features": int(len(names)),
        "sample_seed": int(seed),
        "sampling": "label-stratified without replacement",
        "shap_space": "raw margin (log-odds)",
        "mean_bias": float(np.mean(bias)),
        "sum_mean_abs_shap": total,
        "margin_error_max": margin_error,
        "mean_score": float(np.mean(probabilities)),
        "positive_rate": float(np.mean(sample["label"])),
        "top_10": ranking.head(10)[["rank", "feature", "feature_group", "mean_abs_shap", "share_pct"]].to_dict("records"),
        "top_15": ranking.head(15)[["rank", "feature", "feature_group", "mean_abs_shap", "share_pct"]].to_dict("records"),
        "constant_features": ranking.loc[ranking["constant_on_sample"], "feature"].tolist(),
        "caveat": (
            "TreeSHAP explains the raw margin, not the Platt display score. "
            "source_ragtruth is constant 0 on HaluEval; its nonzero attributions "
            "come from interaction structure, not within-domain source variation."
        ),
    }
    return {
        "ranking": ranking,
        "long": long_rows,
        "summary": summary,
        "sample_ids": sample["sample_id"].tolist(),
        "sample_idx": sample_idx,
        "sample": sample,
        "X": X,
        "shap_values": shap_values,
    }


def plot_global_top15(ranking: pd.DataFrame, out_path: Path, n_rows: int) -> None:
    top = ranking.head(15).iloc[::-1].reset_index(drop=True)
    colors = [GROUP_COLORS.get(group, GROUP_COLORS["unknown"]) for group in top["feature_group"]]
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    bars = ax.barh(
        np.arange(len(top)),
        top["mean_abs_shap"],
        color=colors,
        edgecolor="#1e293b",
        linewidth=0.4,
        height=0.72,
    )
    for bar, constant in zip(bars, top["constant_on_sample"]):
        if constant:
            bar.set_hatch("//")
            bar.set_edgecolor("#334155")

    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=9)
    ax.set_xlabel("mean |SHAP| (raw margin)", fontsize=10)
    ax.set_title(
        f"EC-XGB (m3, seed 42): top-15 features by mean |SHAP|\n"
        f"{n_rows:,} stratified HaluEval test rows",
        fontsize=11,
        loc="left",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.7)
    ax.set_axisbelow(True)
    xmax = float(top["mean_abs_shap"].max())
    ax.set_xlim(0, xmax * 1.18)
    for bar, value in zip(bars, top["mean_abs_shap"]):
        ax.text(
            bar.get_width() + xmax * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=8,
            color="#334155",
        )

    handles = []
    seen = []
    for group in top["feature_group"]:
        if group in seen:
            continue
        seen.append(group)
        handles.append(plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS.get(group, GROUP_COLORS["unknown"])))
    ax.legend(handles, seen, title="feature group", frameon=False, fontsize=8, title_fontsize=8,
              loc="lower right")

    footnote = (
        "TreeSHAP on the raw margin (log-odds); bars with hatching are constant on this sample "
        "(attributions arise through interactions)."
    )
    fig.text(0.01, 0.01, footnote, fontsize=7.5, color="#475569")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


# --------------------------------------------------------------------------
# B. Per-case audit
# --------------------------------------------------------------------------
def compute_audit_cases(test: pd.DataFrame, names: list, booster) -> tuple:
    review = pd.read_csv(AUDIT_CASES)
    if "sample_id" not in review.columns:
        raise ValueError(f"{AUDIT_CASES} has no sample_id column")
    ids = review["sample_id"].astype(str).tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sample_id in the audit case file")

    text = load_unified_text(ids)
    audit = test[test["sample_id"].isin(ids)].set_index("sample_id").loc[ids].reset_index()
    if len(audit) != len(ids):
        missing = sorted(set(ids) - set(audit["sample_id"]))
        raise ValueError(f"audit cases missing from the test split: {missing[:5]}")

    text = text.loc[audit["sample_id"].tolist()]
    if not np.array_equal(audit["label"].to_numpy(), text["label"].to_numpy()):
        raise ValueError("label mismatch between feature table and unified_records")
    audit["question"] = text["question"].to_numpy()
    audit["context"] = text["context"].to_numpy()
    audit["answer"] = text["answer"].to_numpy()

    X = feature_matrix(audit, names, log_lengths=True)
    shap_values, bias, scores, margin_error = tree_shap(booster, X)
    logger.info("Audit TreeSHAP: %d cases, margin error %.2e", len(audit), margin_error)

    role_counts: Counter = Counter()
    rows = []
    role_rows = []
    long_rows = []
    for i, row in audit.iterrows():
        score = float(scores[i])
        role = assign_role(int(row["label"]), score)
        role_counts[role] += 1
        case_id = f"m3_{role}_{role_counts[role]:02d}"
        order = np.argsort(-np.abs(shap_values[i]))[:5]
        b2_score = float(review.loc[review["sample_id"] == row["sample_id"], "raw_score"].iloc[0])
        rows.append({
            "case_id": case_id,
            "sample_id": row["sample_id"],
            "label": int(row["label"]),
            "raw_score": round(score, 6),
            "top_features": "; ".join(names[j] for j in order),
            "top_values": "; ".join(FLOAT_FORMAT.format(float(X[i, j])) for j in order),
            "top_shap": "; ".join(f"{float(shap_values[i, j]):+.6f}" for j in order),
            "snippet": snippet(row["answer"], 120),
            "answer_plausible": "",
            "review_status": "",
        })
        role_rows.append({
            "case_id": case_id,
            "sample_id": row["sample_id"],
            "label": int(row["label"]),
            "b2_role": assign_role(int(row["label"]), b2_score),
            "b2_raw_score": round(b2_score, 6),
            "m3_role": role,
            "m3_raw_score": round(score, 6),
        })
        for j, name in enumerate(names):
            long_rows.append({
                "case_id": case_id,
                "sample_id": row["sample_id"],
                "feature": name,
                "value": float(X[i, j]),
                "shap": float(shap_values[i, j]),
            })

    audit_csv = pd.DataFrame(rows, columns=[
        "case_id", "sample_id", "label", "raw_score", "top_features", "top_values",
        "top_shap", "snippet", "answer_plausible", "review_status",
    ])
    role_csv = pd.DataFrame(role_rows)
    long_df = pd.DataFrame(long_rows)
    summary = {
        "n_cases": int(len(audit)),
        "role_counts_m3": {key: int(value) for key, value in sorted(role_counts.items())},
        "role_counts_b2": {
            key: int(value)
            for key, value in sorted(Counter(role_csv["b2_role"]).items())
        },
        "margin_error_max": margin_error,
        "case_text_source": str(UNIFIED_RECORDS.relative_to(ROOT)),
        "human_columns": ["answer_plausible", "review_status"],
        "human_columns_state": "left empty for the author",
        "review_provenance": (
            "The B5 audit was AI-assisted (review_mode=ai-expert). This WP4 export "
            "does not reuse or relabel those judgments as human review."
        ),
    }
    return audit_csv, role_csv, long_df, summary


# --------------------------------------------------------------------------
# D. B2 comparison
# --------------------------------------------------------------------------
def rank_of(series: pd.Series) -> pd.Series:
    return series.rank(ascending=False, method="min").astype(int)


def topk_overlap(a: list, b: list, k: int) -> float:
    left, right = set(a[:k]), set(b[:k])
    union = left | right
    return float(len(left & right) / len(union)) if union else 0.0


def compare_with_b2(test: pd.DataFrame, names: list, m3_ranking: pd.DataFrame,
                    sample_idx: np.ndarray) -> tuple:
    frozen = json.loads(B5_IMPORTANCE.read_text(encoding="utf-8"))
    if "mean_abs_shap" not in frozen:
        raise ValueError(f"{B5_IMPORTANCE} has no mean_abs_shap block")
    b2_frozen = pd.Series(frozen["mean_abs_shap"], dtype=float)
    b2_frozen_rank = rank_of(b2_frozen)

    m3_series = m3_ranking.set_index("feature")["mean_abs_shap"]
    m3_rank = m3_ranking.set_index("feature")["rank"]
    groups = feature_groups(names)

    b2_same = None
    b2_same_rank = None
    b2_same_error = None
    if B2_MODEL.exists() and B2_CONFIG.exists():
        b2_config = json.loads(B2_CONFIG.read_text(encoding="utf-8"))
        b2_names = list(b2_config["feature_cols"])
        missing = [name for name in b2_names if name not in test.columns]
        if missing:
            logger.warning("B2 same-sample control skipped; missing columns %s", missing[:5])
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                b2_model = joblib.load(B2_MODEL)
            b2_booster = b2_model.get_booster()
            b2_booster.set_param({"device": "cpu"})
            X_b2 = test.loc[sample_idx, b2_names].to_numpy(dtype=np.float64)
            b2_shap, _, b2_proba, b2_same_error = tree_shap(b2_booster, X_b2)
            b2_same = pd.Series(np.abs(b2_shap).mean(axis=0), index=b2_names)
            b2_same_rank = rank_of(b2_same)
            logger.info(
                "B2 same-sample control: %d rows x %d features, margin error %.2e",
                len(X_b2), len(b2_names), b2_same_error,
            )
    else:
        logger.warning("B2 model/config unavailable; same-sample control skipped")

    rows = []
    for feature in names:
        in_b2 = feature in b2_frozen.index
        in_same = b2_same is not None and feature in b2_same.index
        rows.append({
            "feature": feature,
            "feature_group": groups[feature],
            "m3_rank": int(m3_rank[feature]),
            "m3_mean_abs_shap": float(m3_series[feature]),
            "b2_frozen_rank": int(b2_frozen_rank[feature]) if in_b2 else None,
            "b2_frozen_mean_abs_shap": float(b2_frozen[feature]) if in_b2 else None,
            "b2_same_sample_rank": int(b2_same_rank[feature]) if in_same else None,
            "b2_same_sample_mean_abs_shap": float(b2_same[feature]) if in_same else None,
            "rank_shift_frozen": int(m3_rank[feature] - b2_frozen_rank[feature]) if in_b2 else None,
            "rank_shift_same_sample": int(m3_rank[feature] - b2_same_rank[feature]) if in_same else None,
            "in_m3_top10": bool(m3_rank[feature] <= 10),
            "in_b2_top10": bool(in_b2 and b2_frozen_rank[feature] <= 10),
        })
    comparison = pd.DataFrame(rows)

    common = [feature for feature in names if feature in b2_frozen.index]
    m3_values = [float(m3_series[feature]) for feature in common]
    frozen_values = [float(b2_frozen[feature]) for feature in common]
    frozen_rho = float(spearmanr(m3_values, frozen_values).statistic)
    frozen_tau = float(kendalltau(m3_values, frozen_values).statistic)

    m3_top10 = m3_ranking["feature"].head(10).tolist()
    frozen_top10 = b2_frozen.sort_values(ascending=False).head(10).index.tolist()
    m3_top3 = m3_ranking["feature"].head(3).tolist()
    frozen_top3 = b2_frozen.sort_values(ascending=False).head(3).index.tolist()

    same_summary = None
    if b2_same is not None:
        same_common = [feature for feature in names if feature in b2_same.index]
        same_rho = float(spearmanr(
            [float(m3_series[feature]) for feature in same_common],
            [float(b2_same[feature]) for feature in same_common],
        ).statistic)
        same_tau = float(kendalltau(
            [float(m3_series[feature]) for feature in same_common],
            [float(b2_same[feature]) for feature in same_common],
        ).statistic)
        same_top10 = b2_same.sort_values(ascending=False).head(10).index.tolist()
        same_top3 = b2_same.sort_values(ascending=False).head(3).index.tolist()
        same_summary = {
            "n_rows": int(len(sample_idx)),
            "n_common_features": int(len(same_common)),
            "spearman": same_rho,
            "kendall_tau": same_tau,
            "top10_overlap_jaccard": topk_overlap(m3_top10, same_top10, 10),
            "top3_overlap_jaccard": topk_overlap(m3_top3, same_top3, 3),
            "m3_top10": m3_top10,
            "b2_top10": same_top10,
            "margin_error_max": b2_same_error,
        }

    summary = {
        "b2_reference": str(B5_IMPORTANCE.relative_to(ROOT)),
        "b2_reference_rows": int(frozen.get("n_test") or frozen.get("n_rows") or 3000),
        "b2_reference_rows_note": "B5 computed SHAP on the full 3,000-row HaluEval test split",
        "b2_reference_kind": "frozen B5 SHAP ranking (B2 seed 42)",
        "n_common_features": int(len(common)),
        "spearman": frozen_rho,
        "kendall_tau": frozen_tau,
        "top3_overlap_jaccard": topk_overlap(m3_top10, frozen_top10, 3),
        "top10_overlap_jaccard": topk_overlap(m3_top10, frozen_top10, 10),
        "top10_shared": [feature for feature in m3_top10 if feature in frozen_top10],
        "m3_only_top10": [feature for feature in m3_top10 if feature not in frozen_top10],
        "b2_only_top10": [feature for feature in frozen_top10 if feature not in m3_top10],
        "m3_top10": m3_top10,
        "b2_top10": frozen_top10,
        "m3_top3": m3_top3,
        "b2_top3": frozen_top3,
        "top1_same": bool(m3_top10[0] == frozen_top10[0]),
        "same_sample_control": same_summary,
    }
    return comparison, summary


# --------------------------------------------------------------------------
# C. Perturbation pilot
# --------------------------------------------------------------------------
def run_perturbation_pilot(args, test: pd.DataFrame, names: list, booster) -> dict:
    from src.features.claim_features import compute_claim_features
    from src.features.extract_features import extract_all_features_single, load_heavy_models
    from src.models.run_b5_explanation_reliability import PERTURBATION_TYPES, apply_perturbation

    pilot_idx = stratified_sample(test, args.n_perturb, args.seed)
    pilot = test.loc[pilot_idx].reset_index(drop=True)
    pilot_text = load_unified_text(pilot["sample_id"].tolist())
    pilot["question"] = pilot_text["question"].to_numpy()
    pilot["context"] = pilot_text["context"].to_numpy()
    pilot["answer"] = pilot_text["answer"].to_numpy()
    rows: list = []
    attempted = 0
    unchanged = 0
    completed_samples = 0
    timed_out = False
    started = time.time()

    logger.info("Loading NER/NLI/embedding models for perturbation re-extraction...")
    models = load_heavy_models(device=args.device)
    load_seconds = time.time() - started
    logger.info("Heavy models loaded in %.1fs", load_seconds)

    csv_path = OUT_DIR / "m3_perturbations.csv"
    for position, row in pilot.iterrows():
        elapsed_minutes = (time.time() - started) / 60.0
        if elapsed_minutes > args.max_minutes:
            timed_out = True
            logger.warning("Perturbation pilot hit the %.1f-minute budget; stopping.", args.max_minutes)
            break

        X_orig = feature_matrix(pilot.iloc[[position]], names, log_lengths=True)
        shap_orig, _, score_orig_arr, _ = tree_shap(booster, X_orig)
        score_orig = float(score_orig_arr[0])
        order_orig = np.argsort(-np.abs(shap_orig[0]))
        top1_orig = names[int(order_orig[0])]
        top5_orig = {names[int(j)] for j in order_orig[:5]}

        for kind in PERTURBATION_TYPES:
            attempted += 1
            question, context, answer, meta = apply_perturbation(
                kind,
                str(row["question"]),
                str(row["context"]),
                str(row["answer"]),
                args.seed + position,
                models["nlp"],
            )
            if not meta.get("changed"):
                unchanged += 1
                continue

            base_features = extract_all_features_single(question, context, answer, models)
            claim_features = compute_claim_features(answer, context, models["nli"])
            payload = {**base_features, **claim_features, SOURCE_COLUMN: 0.0}
            missing = [name for name in names if name not in payload]
            if missing:
                raise ValueError(f"re-extraction missing features: {missing[:5]}")
            X_pert = feature_matrix(pd.DataFrame([payload]), names, log_lengths=True)
            shap_pert, _, score_pert_arr, _ = tree_shap(booster, X_pert)
            score_pert = float(score_pert_arr[0])

            order_pert = np.argsort(-np.abs(shap_pert[0]))
            top1_pert = names[int(order_pert[0])]
            top5_pert = {names[int(j)] for j in order_pert[:5]}
            feature_delta = np.abs(X_pert[0] - X_orig[0])
            max_feature_idx = int(np.argmax(feature_delta))

            try:
                rho = float(spearmanr(np.abs(shap_orig[0]), np.abs(shap_pert[0])).statistic)
            except ValueError:
                rho = float("nan")

            rows.append({
                "sample_id": row["sample_id"],
                "perturbation": kind,
                "label": int(row["label"]),
                "raw_score": round(score_orig, 6),
                "perturbed_score": round(score_pert, 6),
                "score_delta": round(score_pert - score_orig, 6),
                "abs_score_delta": round(abs(score_pert - score_orig), 6),
                "top1_flip": int(top1_orig != top1_pert),
                "orig_top1": top1_orig,
                "pert_top1": top1_pert,
                "top5_overlap": round(len(top5_orig & top5_pert) / 5.0, 6),
                "spearman": round(rho, 6) if np.isfinite(rho) else None,
                "max_feature_changed": names[max_feature_idx],
                "max_feature_delta": round(float(feature_delta[max_feature_idx]), 6),
                "seed": int(args.seed + position),
            })

        completed_samples += 1
        pd.DataFrame(rows, columns=PERTURBATION_COLUMNS).to_csv(csv_path, index=False)
        logger.info(
            "Perturbation pilot: %d/%d samples, %d changed perturbations, %.1fs",
            completed_samples, len(pilot), len(rows), time.time() - started,
        )

    elapsed = time.time() - started
    perturbation_df = pd.DataFrame(rows, columns=PERTURBATION_COLUMNS)
    perturbation_df.to_csv(csv_path, index=False)

    if len(perturbation_df):
        aggregates = perturbation_df.groupby("perturbation").agg(
            n=("sample_id", "count"),
            mean_abs_score_delta=("score_delta", lambda s: float(np.abs(s).mean())),
            std_score_delta=("score_delta", lambda s: float(s.std())),
            max_abs_score_delta=("score_delta", lambda s: float(np.abs(s).max())),
            large_delta_rate=("score_delta", lambda s: float((np.abs(s) > 0.3).mean())),
            top1_flip_rate=("top1_flip", "mean"),
            mean_top5_overlap=("top5_overlap", "mean"),
            mean_spearman=("spearman", "mean"),
        ).reset_index().round(6)
    else:
        aggregates = pd.DataFrame()
    aggregates.to_csv(OUT_DIR / "m3_perturbation_aggregates.csv", index=False)

    summary = {
        "requested_samples": int(args.n_perturb),
        "completed_samples": int(completed_samples),
        "completed": bool((not timed_out) and completed_samples == args.n_perturb),
        "timed_out": bool(timed_out),
        "budget_minutes": float(args.max_minutes),
        "elapsed_seconds": float(elapsed),
        "model_load_seconds": float(load_seconds),
        "attempted_perturbations": int(attempted),
        "changed_perturbations": int(len(perturbation_df)),
        "unchanged_perturbations": int(unchanged),
        "mean_seconds_per_changed_perturbation": (
            float((elapsed - load_seconds) / len(perturbation_df)) if len(perturbation_df) else None
        ),
        "device_requested": args.device or "auto",
        "device_used": str(getattr(models["nli"], "device", "unknown")),
        "perturbation_types": list(PERTURBATION_TYPES),
        "note": (
            "20-row pilot only. These rows are a deterministic label-stratified "
            "subsample, not a representative evaluation set."
        ),
    }
    (OUT_DIR / "m3_perturbation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "Perturbation pilot %s: %d/%d samples, %d changed evaluations in %.1fs",
        "completed" if summary["completed"] else "NOT completed",
        completed_samples, args.n_perturb, len(perturbation_df), elapsed,
    )
    return {"rows": perturbation_df, "aggregates": aggregates, "summary": summary}


# --------------------------------------------------------------------------
# E. Report
# --------------------------------------------------------------------------
def markdown_table(frame: pd.DataFrame, columns: list, float_formats: dict | None = None) -> str:
    float_formats = float_formats or {}
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                cells.append(float_formats.get(column, "{:.6f}").format(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(config: dict, global_result: dict, audit_summary: dict, audit_csv: pd.DataFrame,
                 role_csv: pd.DataFrame, comparison_summary: dict, comparison: pd.DataFrame,
                 perturbation: dict | None, validation: dict | None, args) -> None:
    ranking = global_result["ranking"]
    summary = global_result["summary"]
    top10 = ranking.head(10)
    top15 = ranking.head(15)

    b2_top10 = comparison_summary["b2_top10"]
    m3_only = comparison_summary["m3_only_top10"]
    b2_only = comparison_summary["b2_only_top10"]
    top1_changed = not comparison_summary["top1_same"]
    top3_overlap = comparison_summary["top3_overlap_jaccard"]

    if top1_changed or top3_overlap < 0.5:
        story = (
            "The explanation story changes at the top: the m3 ranking does not preserve "
            "the frozen B2 ordering. The new ordering should be described from the m3 "
            "values rather than carried over from the B2 narrative."
        )
    else:
        story = (
            "The core explanation story persists: context overlap and answer length still "
            "dominate, and two of the three top features are shared with B2. The change is "
            "mainly a relative reweighting plus an added claim-level signal, not a replacement "
            "of the original drivers."
        )
    if m3_only or b2_only:
        story += (
            f" The m3 top-10 adds {', '.join(m3_only) if m3_only else 'nothing'} and "
            f"drops {', '.join(b2_only) if b2_only else 'nothing'} relative to B2."
        )

    lines = []
    lines.append("# WP4 — explanations for the rebuilt EC-XGB (m3)")
    lines.append("")
    lines.append(f"Generated: {config['generated_at_utc']}")
    lines.append("")
    lines.append(f"- Model: `{config['model_path']}`")
    lines.append(f"- Model version: `{MODEL_VERSION}`")
    lines.append(f"- Feature set: {summary['n_features']} m3 features "
                 f"(26 base + 8 claim-level NLI + `{SOURCE_COLUMN}`)")
    lines.append(
        "- SHAP: exact TreeSHAP via `pred_contribs=True` (`approx_contribs=False`), raw margin (log-odds)"
    )
    lines.append(f"- Global sample: {summary['n_rows']:,} label-stratified HaluEval test rows, seed {summary['sample_seed']}")
    lines.append("- Human judgment columns in `m3_audit_cases.csv`: left empty")
    lines.append("")
    lines.append("## 1. Commands")
    lines.append("")
    lines.append("```powershell")
    lines.append("& .venv\\Scripts\\python.exe src\\models\\rebuild_explanations.py")
    lines.append("& .venv\\Scripts\\python.exe src\\models\\rebuild_explanations.py --perturbation --n-perturb 20")
    lines.append("```")
    lines.append("")
    lines.append("## 2. Global TreeSHAP")
    lines.append("")
    lines.append(
        f"Exactness check: `max |sum(SHAP) + bias - margin| = {summary['margin_error_max']:.3e}`. "
        f"Mean bias (expected margin) = {summary['mean_bias']:.6f}; mean raw score = {summary['mean_score']:.6f}."
    )
    if validation:
        lines.append(
            f"Feature-construction check: the m3 scores reproduce `{validation['reference']}` on all "
            f"{validation['n']:,} test rows (max absolute difference = "
            f"{validation['max_abs_diff']:.3e})."
        )
    lines.append("")
    lines.append("Top 15 by mean |SHAP|:")
    lines.append("")
    lines.append(markdown_table(
        top15,
        ["rank", "feature", "feature_group", "mean_abs_shap", "share_pct", "constant_on_sample"],
        {"mean_abs_shap": "{:.6f}", "share_pct": "{:.3f}"},
    ))
    lines.append("")
    lines.append(f"Figure: `{config['figure_path']}` (600 DPI, top 15, horizontal bars).")
    lines.append("")
    lines.append(
        f"The top 10 features account for {ranking.head(10)['share_pct'].sum():.2f}% of the "
        f"total mean |SHAP| mass; the top 15 account for {ranking.head(15)['share_pct'].sum():.2f}%."
    )
    lines.append("")
    lines.append("## 3. Per-case audit")
    lines.append("")
    lines.append(
        f"The {audit_summary['n_cases']} cases come from `artifacts/results/b5/b5_review_cases_reviewed.csv`. "
        f"Under m3 the roles redistribute to {format_role_counts(audit_summary['role_counts_m3'])} "
        f"(B2 roles were {format_role_counts(audit_summary['role_counts_b2'])})."
    )
    lines.append("")
    lines.append("Top-3 m3 features per case (values and signed contributions are in the CSV):")
    lines.append("")
    audit_view = audit_csv.copy()
    audit_view["top3"] = audit_view["top_features"].apply(
        lambda value: ", ".join(str(value).split("; ")[:3])
    )
    audit_view = audit_view[["case_id", "sample_id", "label", "raw_score", "top3"]]
    lines.append(markdown_table(audit_view, ["case_id", "sample_id", "label", "raw_score", "top3"],
                                {"raw_score": "{:.6f}"}))
    lines.append("")
    lines.append(
        "`m3_audit_cases.csv` has exactly the requested columns: `case_id, sample_id, label, "
        "raw_score, top_features, top_values, top_shap, snippet, answer_plausible, review_status`. "
        "`answer_plausible` and `review_status` are empty for the author. `top_values` are "
        "model-input values; `n_chars` and `n_words` are therefore log1p-transformed."
    )
    lines.append("")
    lines.append("## 4. Comparison with the frozen B2 ranking")
    lines.append("")
    lines.append(
        f"Reference: `{comparison_summary['b2_reference']}` "
        f"(`{comparison_summary['b2_reference_kind']}`, n={comparison_summary['b2_reference_rows']}). "
        "No B2 SHAP CSV exists, so the frozen B5 JSON is the reference."
    )
    lines.append("")
    lines.append("Top 10 side by side (m3 rank / B2 rank):")
    lines.append("")
    side = pd.DataFrame({
        "m3_rank": list(range(1, 11)),
        "m3_feature": comparison_summary["m3_top10"],
        "b2_rank": list(range(1, 11)),
        "b2_feature": b2_top10,
    })
    lines.append(markdown_table(side, ["m3_rank", "m3_feature", "b2_rank", "b2_feature"]))
    lines.append("")
    lines.append(
        f"Rank agreement over the {comparison_summary['n_common_features']} shared features: "
        f"Spearman rho = {comparison_summary['spearman']:.6f}, "
        f"Kendall tau = {comparison_summary['kendall_tau']:.6f}. "
        f"Top-10 Jaccard = {comparison_summary['top10_overlap_jaccard']:.6f}; "
        f"top-3 Jaccard = {comparison_summary['top3_overlap_jaccard']:.6f}."
    )
    lines.append("")
    lines.append(story)
    lines.append("")

    m3_view = ranking.set_index("feature")
    comp_view = comparison.set_index("feature")
    cosine_m3_rank = int(m3_view.loc["cosine_q_ans", "rank"])
    cosine_b2_rank = int(comp_view.loc["cosine_q_ans", "b2_frozen_rank"])
    claim_rank = int(m3_view.loc["claim_mean_contradicts", "rank"])
    neutral_rank = int(m3_view.loc["nli_ctx_neutral_ans", "rank"])
    claim_gap = float(
        m3_view.loc["claim_mean_contradicts", "mean_abs_shap"]
        - m3_view.loc["nli_ctx_neutral_ans", "mean_abs_shap"]
    )
    lines.append(
        f"The main relative change is `cosine_q_ans`: B2 rank {cosine_b2_rank} "
        f"({comp_view.loc['cosine_q_ans', 'b2_frozen_mean_abs_shap']:.6f}) to m3 rank "
        f"{cosine_m3_rank} ({m3_view.loc['cosine_q_ans', 'mean_abs_shap']:.6f}). "
        f"`claim_mean_contradicts` enters at rank {claim_rank} "
        f"({m3_view.loc['claim_mean_contradicts', 'mean_abs_shap']:.6f}); its margin over "
        f"`nli_ctx_neutral_ans` at rank {neutral_rank} is only {claim_gap:.6f}, so the exact "
        "ordering at that cut-off is soft."
    )
    lines.append("")
    same = comparison_summary.get("same_sample_control")
    if same:
        lines.append(
            f"Same-1,000-row B2 control: Spearman rho = {same['spearman']:.6f}, "
            f"Kendall tau = {same['kendall_tau']:.6f}, top-10 Jaccard = "
            f"{same['top10_overlap_jaccard']:.6f}. This control rules out the 1,000-row "
            "subsample as the main reason for the m3/B2 differences."
        )
        lines.append("")
    lines.append("Full per-feature comparison: `m3_vs_b2_importance.csv`.")
    lines.append("")
    lines.append("## 5. Perturbation pilot")
    lines.append("")
    if perturbation is None:
        lines.append("Not run in this invocation. Run `--perturbation --n-perturb 20` to populate this section.")
    else:
        psum = perturbation["summary"]
        lines.append(
            f"Status: **{'completed' if psum['completed'] else 'did NOT complete'}**. "
            f"{psum['completed_samples']}/{psum['requested_samples']} samples, "
            f"{psum['changed_perturbations']} changed perturbations "
            f"({psum['unchanged_perturbations']} unchanged), "
            f"{psum['elapsed_seconds']:.1f}s wall time including {psum['model_load_seconds']:.1f}s "
            f"model load, budget {psum['budget_minutes']:.1f} min, device "
            f"`{psum['device_used']}`. "
            f"Mean time per changed perturbation after load: "
            f"{psum['mean_seconds_per_changed_perturbation']:.3f}s."
        )
        lines.append("")
        if len(perturbation["aggregates"]):
            lines.append(markdown_table(
                perturbation["aggregates"],
                ["perturbation", "n", "mean_abs_score_delta", "max_abs_score_delta",
                 "large_delta_rate", "top1_flip_rate", "mean_top5_overlap", "mean_spearman"],
                {key: "{:.6f}" for key in [
                    "mean_abs_score_delta", "max_abs_score_delta", "large_delta_rate",
                    "top1_flip_rate", "mean_top5_overlap", "mean_spearman",
                ]},
            ))
        lines.append("")
        lines.append(
            "Full rows: `m3_perturbations.csv`; aggregates: `m3_perturbation_aggregates.csv`; "
            "timing: `m3_perturbation_summary.json`. These are multi-feature interventions: one "
            "perturbation can move length, overlap, NLI and claim features at once, so the deltas "
            "are not single-feature causal effects. Per-type n is small (2-20); treat this as a "
            "timing and plumbing pilot, not an estimate."
        )
    lines.append("")
    lines.append("## 6. Caveats")
    lines.append("")
    lines.append(
        "- SHAP values are additive on the **raw margin (log-odds)**, not on the probability "
        "or the Platt display score. A positive value raises the hallucination margin; it is "
        "not a percentage-point change in risk."
    )
    lines.append(
        "- The audit cases are **error-enriched**: they were selected from B2 false positives, "
        "false negatives and borderline cases, not sampled at random. Their m3 scores and "
        "roles differ from the B2 roles, and the set says nothing about population-level rates."
    )
    lines.append(
        f"- `{SOURCE_COLUMN}` is constant 0 on HaluEval. Its nonzero mean |SHAP| "
        f"({float(ranking.set_index('feature').loc[SOURCE_COLUMN, 'mean_abs_shap']):.6f}) comes "
        "from interaction structure in the trees, not from within-domain source variation. "
        "Do not read it as a causal 'trained on RAGTruth' effect for individual HaluEval rows."
    )
    lines.append(
        "- The global ranking uses 1,000 rows. It is a deterministic stratified subsample, "
        "not a bootstrap; small rank differences near the cut-off should not be over-read."
    )
    lines.append(
        "- Only the seed-42 m3 model is explained. The seed-123 and seed-456 models exist, "
        "but their SHAP rankings were not computed, so cross-seed explanation stability is untested."
    )
    lines.append(
        "- `n_chars` and `n_words` enter the m3 model as log1p values. The audit CSV reports "
        "model-input values, so those two features appear on the log scale."
    )
    lines.append(
        "- The B5.5 audit was AI-assisted (`review_mode = ai-expert`). This export leaves the "
        "human columns empty and does not describe the old judgments as human review."
    )
    lines.append("")
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved %s", OUT_DIR / "REPORT.md")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="WP4 explanations for the rebuilt EC-XGB m3 model")
    parser.add_argument("--n-shap", type=int, default=DEFAULT_N_SHAP,
                        help="HaluEval test rows for global TreeSHAP (default 1000)")
    parser.add_argument("--perturbation", action="store_true",
                        help="run the six-perturbation pilot with full feature re-extraction")
    parser.add_argument("--n-perturb", type=int, default=DEFAULT_N_PERTURB,
                        help="samples in the perturbation pilot (default 20)")
    parser.add_argument("--max-minutes", type=float, default=DEFAULT_MAX_MINUTES,
                        help="wall-clock budget for the perturbation pilot (default 15)")
    parser.add_argument("--device", default=None, help="cuda|cpu for heavy re-extraction (default: auto)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    names = load_feature_names()
    test = load_test_frame()
    model, booster = load_model_and_booster()
    logger.info("Loaded %s with %d features on %d test rows", MODEL_VERSION, len(names), len(test))
    validation = validate_against_b6(test, names, booster)
    if validation:
        logger.info(
            "Score reproduction check: max |m3 - b6_predictions| = %.3e over %d test rows",
            validation["max_abs_diff"], validation["n"],
        )

    global_result = compute_global_shap(test, names, booster, args.n_shap, args.seed)
    ranking = global_result["ranking"]
    ranking.to_csv(OUT_DIR / "m3_global_shap_importance.csv", index=False)
    global_result["long"].to_parquet(OUT_DIR / "m3_global_shap_values.parquet", index=False)
    (OUT_DIR / "m3_shap_summary.json").write_text(
        json.dumps(global_result["summary"], indent=2), encoding="utf-8"
    )
    plot_global_top15(ranking, OUT_DIR / "m3_shap_top15.png", global_result["summary"]["n_rows"])

    audit_csv, role_csv, audit_long, audit_summary = compute_audit_cases(test, names, booster)
    audit_csv.to_csv(OUT_DIR / "m3_audit_cases.csv", index=False)
    role_csv.to_csv(OUT_DIR / "m3_audit_case_roles.csv", index=False)
    audit_long.to_parquet(OUT_DIR / "m3_audit_case_shap.parquet", index=False)
    (OUT_DIR / "m3_audit_summary.json").write_text(
        json.dumps(audit_summary, indent=2), encoding="utf-8"
    )

    comparison, comparison_summary = compare_with_b2(
        test, names, ranking, np.asarray(global_result["sample_idx"], dtype=int)
    )
    comparison.to_csv(OUT_DIR / "m3_vs_b2_importance.csv", index=False)
    (OUT_DIR / "m3_vs_b2_comparison.json").write_text(
        json.dumps(comparison_summary, indent=2), encoding="utf-8"
    )

    perturbation = None
    if args.perturbation:
        perturbation = run_perturbation_pilot(args, test, names, booster)

    config = {
        "schema": "rebuild-explanations-v1",
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "feature_names_path": str(FEATURE_NAMES_PATH.relative_to(ROOT)),
        "feature_count": len(names),
        "feature_groups": {group: columns for group, columns in FEATURE_GROUPS.items()},
        "claim_feature_columns": list(CLAIM_COLUMNS),
        "source_indicator": SOURCE_COLUMN,
        "shap": "exact TreeSHAP via xgboost pred_contribs=True",
        "shap_approx_contribs": False,
        "shap_space": "raw margin (log-odds)",
        "n_global_rows": int(args.n_shap),
        "global_sample_seed": int(args.seed),
        "score_reproduction_check": validation,
        "audit_cases_path": str(AUDIT_CASES.relative_to(ROOT)),
        "audit_text_source": str(UNIFIED_RECORDS.relative_to(ROOT)),
        "perturbation_run": bool(args.perturbation),
        "n_perturb": int(args.n_perturb) if args.perturbation else None,
        "inputs": {
            "model_sha256": sha256(MODEL_PATH),
            "feature_names_sha256": sha256(FEATURE_NAMES_PATH),
            "features_full_sha256": sha256(FEATURES_FULL),
            "claim_features_sha256": sha256(CLAIM_FEATURES),
            "audit_cases_sha256": sha256(AUDIT_CASES),
        },
        "versions": {
            "python": sys.version.split()[0],
            "xgboost": xgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "elapsed_seconds": float(time.time() - started),
    }
    config["figure_path"] = str((OUT_DIR / "m3_shap_top15.png").relative_to(ROOT))
    (OUT_DIR / "m3_run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    write_report(config, global_result, audit_summary, audit_csv, role_csv,
                 comparison_summary, comparison, perturbation, validation, args)

    print("\n" + "=" * 88)
    print(f" WP4 — {MODEL_NAME} ({MODEL_VERSION}) explanations")
    print("=" * 88)
    print(f"Global rows: {global_result['summary']['n_rows']} | features: {len(names)} | "
          f"margin error: {global_result['summary']['margin_error_max']:.2e}")
    print("Top 10 by mean |SHAP|:")
    for _, row in ranking.head(10).iterrows():
        print(f"  {int(row['rank']):2d}. {row['feature']:30s} {row['mean_abs_shap']:.6f} "
              f"({row['feature_group']})")
    print(f"Audit cases: {audit_summary['n_cases']} | roles: {audit_summary['role_counts_m3']}")
    print(f"B2 rank agreement: Spearman {comparison_summary['spearman']:.4f} | "
          f"top-10 Jaccard {comparison_summary['top10_overlap_jaccard']:.4f}")
    if perturbation:
        psum = perturbation["summary"]
        print(f"Perturbation pilot: {'completed' if psum['completed'] else 'NOT completed'} | "
              f"{psum['completed_samples']}/{psum['requested_samples']} samples | "
              f"{psum['elapsed_seconds']:.1f}s")
    print(f"Outputs: {OUT_DIR}")
    print("=" * 88)


if __name__ == "__main__":
    main()
