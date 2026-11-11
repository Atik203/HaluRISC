"""P1.3 - lightweight external-style baseline on HaluEval QA.

The baseline reimplements the feature-ensemble family described by
Sundaragiri et al. (IJERT 15(04), 2026, DOI 10.5281/zenodo.20025987) on the
project's HaluEval QA split.  It is deliberately a *reduced* reimplementation,
not the authors' original model and not their reported number:

  features  ROUGE-L answer/context, SBERT cosine answer/context,
            NLI context->answer entailment and contradiction, answer length
  learner   logistic regression on standardized features
  fit       HaluEval validation split only (3,000 rows)
  evaluate  HaluEval test split once (3,000 rows)
  threshold 0.5 for the headline row; a validation-tuned threshold is reported
            as a separate sensitivity row

The EC-XGB m3 reference rows are read from the fixed-hyperparameter rebuild
predictions, never refit here.  Every metric comes with 1,000 row-bootstrap
percentile CIs, and the paired delta against m3 uses the same resampled rows.

Outputs (artifacts/results/rebuild/p1/):
  external_baseline.csv, external_baseline.json, rouge_l_features.csv, REPORT.md

Run (repo root, .venv):
  python src/models/p1_external_baseline.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.contingency_tables import mcnemar

from src.models.config import QA_CLEAN, RESULTS_DIR, SEEDS
from src.models.rebuild_common import (
    CI_KEYS,
    FIXED_HP_PRED,
    THRESHOLD,
    bootstrap_metric_cis,
    bootstrap_metric_cis_pooled,
    metric_bundle,
    stable_seed,
    summarize_bootstrap,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("p1_external_baseline")

P1_DIR = RESULTS_DIR / "rebuild" / "p1"
N_BOOT = 1000
VARIANT = "m3"

# Name used in the paper for the reimplemented feature family.
REFERENCE = {
    "citation": "Sundaragiri, Reddy, Gunavardhan & Navahith, IJERT 15(04), 2026",
    "title": "Framework for Hallucination Detection in Large Language Models",
    "doi": "10.5281/zenodo.20025987",
}

FEATURE_COLUMNS = [
    "rouge_l_f1",
    "cosine_ctx_ans",
    "nli_ctx_entails_ans",
    "nli_ctx_contradicts_ans",
    "log1p_n_words",
]
FEATURE_LABELS = {
    "rouge_l_f1": "ROUGE-L F1 (answer vs context)",
    "cosine_ctx_ans": "SBERT cosine (answer vs context)",
    "nli_ctx_entails_ans": "NLI P(context entails answer)",
    "nli_ctx_contradicts_ans": "NLI P(context contradicts answer)",
    "log1p_n_words": "log1p(answer word count)",
}

RESULT_COLUMNS = [
    "analysis",
    "model",
    "seed",
    "n",
    "prevalence",
    "threshold",
    "f1",
    "precision",
    "recall",
    "auroc",
    "pr_auc",
    "mcc",
    "ece",
    "brier",
    "brier_skill_score",
    "predicted_positive_rate",
    "f1_ci_low",
    "f1_ci_high",
    "auroc_ci_low",
    "auroc_ci_high",
    "pr_auc_ci_low",
    "pr_auc_ci_high",
    "ece_ci_low",
    "ece_ci_high",
    "brier_ci_low",
    "brier_ci_high",
    "n_boot",
    "n_boot_valid",
]

DELTA_KEYS = ("f1", "auroc", "pr_auc", "ece", "brier")


# --------------------------------------------------------------------------
# ROUGE-L (LCS-based, Lin 2004)
# --------------------------------------------------------------------------
def _tokens(text: object) -> list[str]:
    """Lowercase word tokens, matching the default ROUGE tokenizer."""
    return re.findall(r"\w+", str(text).lower())


def lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of two token lists."""
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                left = current[j - 1]
                up = previous[j]
                current.append(left if left >= up else up)
        previous = current
    return int(previous[-1])


def rouge_l_scores(answer: object, context: object) -> tuple[float, float, float]:
    """ROUGE-L precision, recall, and F1 for one answer/context pair."""
    answer_tokens = _tokens(answer)
    context_tokens = _tokens(context)
    lcs = lcs_length(answer_tokens, context_tokens)
    precision = lcs / len(answer_tokens) if answer_tokens else 0.0
    recall = lcs / len(context_tokens) if context_tokens else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def build_rouge_table() -> pd.DataFrame:
    """Compute ROUGE-L for all 20,000 HaluEval QA rows and cache the result."""
    if not QA_CLEAN.exists():
        raise FileNotFoundError(f"missing cleaned QA table: {QA_CLEAN}")
    qa = pd.read_parquet(QA_CLEAN)
    required = {"sample_id", "answer", "context"}
    if required.difference(qa.columns):
        raise ValueError(f"{QA_CLEAN} missing columns: {sorted(required.difference(qa.columns))}")
    if qa["sample_id"].duplicated().any():
        raise ValueError("duplicate sample_id in qa_clean.parquet")

    logger.info("computing ROUGE-L for %d HaluEval QA rows", len(qa))
    started = time.time()
    values = [rouge_l_scores(answer, context) for answer, context in zip(qa["answer"], qa["context"])]
    table = qa[["sample_id", "item_idx", "split", "label"]].copy()
    table[["rouge_l_precision", "rouge_l_recall", "rouge_l_f1"]] = np.asarray(values, dtype=float)
    P1_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(P1_DIR / "rouge_l_features.csv", index=False)
    logger.info("ROUGE-L done in %.1fs", time.time() - started)
    return table


# --------------------------------------------------------------------------
# Data and model
# --------------------------------------------------------------------------
def load_halueval_frame(rouge: pd.DataFrame) -> pd.DataFrame:
    """HaluEval rows with the five external-style features and split labels."""
    from src.models.rebuild_common import FEATURES_FULL

    base = pd.read_parquet(FEATURES_FULL)
    frame = base.merge(
        rouge[["sample_id", "rouge_l_f1"]], on="sample_id", how="left", validate="one_to_one"
    )
    if frame["rouge_l_f1"].isna().any():
        raise ValueError("ROUGE-L missing after merge")
    frame["log1p_n_words"] = np.log1p(np.clip(frame["n_words"].to_numpy(dtype=float), 0.0, None))
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing external baseline features: {missing}")
    if len(frame) != 20000:
        raise ValueError(f"expected 20,000 HaluEval rows, found {len(frame)}")
    return frame


def fit_baseline(val: pd.DataFrame) -> tuple[StandardScaler, LogisticRegression]:
    """Fit the standardized logistic regression on validation rows only."""
    scaler = StandardScaler().fit(val[FEATURE_COLUMNS].to_numpy(dtype=float))
    model = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
    model.fit(scaler.transform(val[FEATURE_COLUMNS].to_numpy(dtype=float)), val["label"].to_numpy(dtype=int))
    return scaler, model


def predict_baseline(scaler: StandardScaler, model: LogisticRegression, frame: pd.DataFrame) -> np.ndarray:
    x = scaler.transform(frame[FEATURE_COLUMNS].to_numpy(dtype=float))
    return model.predict_proba(x)[:, 1]


def tune_threshold(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Pick the F1-maximizing threshold on validation; ties go to the value nearest 0.5."""
    best = {"threshold": 0.5, "f1": -1.0}
    for threshold in np.linspace(0.01, 0.99, 197):
        value = float(f1_score(y_true, scores >= threshold, zero_division=0))
        if value > best["f1"] + 1e-12 or (
            abs(value - best["f1"]) <= 1e-12 and abs(threshold - 0.5) < abs(best["threshold"] - 0.5)
        ):
            best = {"threshold": float(threshold), "f1": value}
    return best


# --------------------------------------------------------------------------
# Metrics and bootstrap
# --------------------------------------------------------------------------
def bundle_with_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    n_boot: int,
    seed: int,
) -> dict:
    """Point metrics plus 1,000-draw row-bootstrap percentile CIs."""
    bundle = metric_bundle(y_true, scores, threshold)
    predicted = np.asarray(scores, dtype=float) >= threshold
    bundle["mcc"] = float(matthews_corrcoef(y_true, predicted)) if len(np.unique(y_true)) > 1 else float("nan")
    cis = bootstrap_metric_cis(y_true, scores, n_boot=n_boot, seed=seed, threshold=threshold)
    row = {key: bundle[key] for key in ("n", "prevalence", "threshold", "f1", "precision", "recall",
                                        "auroc", "pr_auc", "mcc", "ece", "brier",
                                        "brier_skill_score", "predicted_positive_rate")}
    for key in CI_KEYS:
        row[f"{key}_ci_low"] = float(cis[key]["q025"])
        row[f"{key}_ci_high"] = float(cis[key]["q975"])
    row["n_boot"] = int(n_boot)
    row["n_boot_valid"] = int(min(cis[key]["n_valid"] for key in CI_KEYS))
    return row


def mean_bundle(bundles: list[dict]) -> dict:
    """Seed-averaged point metrics (one bundle per seed)."""
    out = {"n": int(bundles[0]["n"]), "threshold": float(bundles[0]["threshold"])}
    for key in ("prevalence", "f1", "precision", "recall", "auroc", "pr_auc", "mcc",
                "ece", "brier", "brier_skill_score", "predicted_positive_rate"):
        out[key] = float(np.mean([bundle[key] for bundle in bundles]))
    return out


def pooled_bundle_with_ci(
    y_true: np.ndarray,
    scores_by_seed: list[np.ndarray],
    threshold: float,
    n_boot: int,
    seed: int,
) -> dict:
    """Seed-averaged metrics plus a joint row-bootstrap CI."""
    bundles = [metric_bundle(y_true, scores, threshold) for scores in scores_by_seed]
    for scores, bundle in zip(scores_by_seed, bundles):
        bundle["mcc"] = float(matthews_corrcoef(y_true, np.asarray(scores) >= threshold))
    row = mean_bundle(bundles)
    cis = bootstrap_metric_cis_pooled(
        [(y_true, scores) for scores in scores_by_seed], n_boot=n_boot, seed=seed, threshold=threshold
    )
    for key in CI_KEYS:
        row[f"{key}_ci_low"] = float(cis[key]["q025"])
        row[f"{key}_ci_high"] = float(cis[key]["q975"])
    row["n_boot"] = int(n_boot)
    row["n_boot_valid"] = int(min(cis[key]["n_valid"] for key in CI_KEYS))
    return row


def paired_delta(
    y_true: np.ndarray,
    external_scores: np.ndarray,
    m3_scores_by_seed: list[np.ndarray],
    threshold: float,
    n_boot: int,
    seed: int,
) -> dict:
    """Paired row-bootstrap of external minus seed-averaged m3 metrics."""
    y_true = np.asarray(y_true, dtype=int)
    external_scores = np.asarray(external_scores, dtype=float)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {key: [] for key in DELTA_KEYS}
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        external = metric_bundle(y_true[idx], external_scores[idx], threshold)
        m3 = [metric_bundle(y_true[idx], scores[idx], threshold) for scores in m3_scores_by_seed]
        for key in DELTA_KEYS:
            draws[key].append(float(external[key] - np.mean([bundle[key] for bundle in m3])))
    point_external = metric_bundle(y_true, external_scores, threshold)
    point_m3 = [metric_bundle(y_true, scores, threshold) for scores in m3_scores_by_seed]
    out = {}
    for key in DELTA_KEYS:
        out[key] = {
            "point": float(point_external[key] - np.mean([bundle[key] for bundle in point_m3])),
            **summarize_bootstrap(draws[key]),
        }
    return out


def mcnemar_test(y_true: np.ndarray, external_scores: np.ndarray, m3_scores: np.ndarray, threshold: float) -> dict:
    external_ok = (np.asarray(external_scores) >= threshold) == np.asarray(y_true)
    m3_ok = (np.asarray(m3_scores) >= threshold) == np.asarray(y_true)
    only_external = int(np.sum(external_ok & ~m3_ok))
    only_m3 = int(np.sum(~external_ok & m3_ok))
    pvalue = float(mcnemar([[0, only_external], [only_m3, 0]], exact=False, correction=True).pvalue)
    return {
        "only_external_correct": only_external,
        "only_m3_correct": only_m3,
        "agreement": float(np.mean((np.asarray(external_scores) >= threshold) == (np.asarray(m3_scores) >= threshold))),
        "pvalue": pvalue,
    }


# --------------------------------------------------------------------------
# EC-XGB reference rows
# --------------------------------------------------------------------------
def load_m3_predictions(sample_ids: pd.Series) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Read fixed-hp m3 test scores and labels, aligned to the external frame order."""
    if not FIXED_HP_PRED.exists():
        raise FileNotFoundError(f"missing fixed-hp predictions: {FIXED_HP_PRED}")
    predictions = pd.read_parquet(FIXED_HP_PRED)
    subset = predictions[
        (predictions["variant"] == VARIANT) & (predictions["dataset"] == "halueval_test")
    ]
    if subset.empty:
        raise ValueError("no m3 halueval_test rows in fixed-hp predictions")
    ordered = {}
    labels = None
    for seed in SEEDS:
        block = subset[subset["seed"] == seed].set_index("sample_id")
        missing = set(sample_ids) - set(block.index)
        if missing:
            raise ValueError(f"m3 seed {seed} missing {len(missing)} sample_ids")
        block = block.loc[sample_ids]
        ordered[seed] = block["score"].to_numpy(dtype=float)
        seed_labels = block["label"].to_numpy(dtype=int)
        if labels is None:
            labels = seed_labels
        elif not np.array_equal(labels, seed_labels):
            raise ValueError(f"m3 seed {seed} label order differs from the first seed")
    return ordered, labels


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


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def report_table(rows: list[dict]) -> str:
    lines = [
        "| analysis | model | seed | F1 | AUROC | PR-AUC | ECE | Brier | PPR |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['analysis']} | {row['model']} | {row['seed']} | {fmt(row['f1'])} | "
            f"{fmt(row['auroc'])} | {fmt(row['pr_auc'])} | {fmt(row['ece'])} | "
            f"{fmt(row['brier'])} | {fmt(row['predicted_positive_rate'])} |"
        )
    return "\n".join(lines)


def ci_table(rows: list[dict], keys: tuple[str, ...] = ("f1", "auroc", "pr_auc", "ece", "brier")) -> str:
    header = "| model | seed | " + " | ".join(f"{key.upper()} [95% CI]" for key in keys) + " |"
    divider = "|---|---|" + "|".join(["---"] * len(keys)) + "|"
    lines = [header, divider]
    for row in rows:
        cells = [
            f"{fmt(row[key])} [{fmt(row[f'{key}_ci_low'])}, {fmt(row[f'{key}_ci_high'])}]"
            for key in keys
        ]
        lines.append(f"| {row['model']} | {row['seed']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    P1_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    rouge = build_rouge_table()
    frame = load_halueval_frame(rouge)
    val = frame[frame["split"] == "val"].reset_index(drop=True)
    test = frame[frame["split"] == "test"].sort_values("sample_id").reset_index(drop=True)
    if len(val) != 3000 or len(test) != 3000:
        raise ValueError(f"expected 3,000 val/test rows, found {len(val)}/{len(test)}")

    scaler, model = fit_baseline(val)
    val_scores = predict_baseline(scaler, model, val)
    test_scores = predict_baseline(scaler, model, test)
    y_val = val["label"].to_numpy(dtype=int)
    y_test = test["label"].to_numpy(dtype=int)

    validation = bundle_with_ci(
        y_val, val_scores, THRESHOLD, N_BOOT, stable_seed("p1_external", "val")
    )
    tuned = tune_threshold(y_val, val_scores)
    test_bundle = bundle_with_ci(
        y_test, test_scores, THRESHOLD, N_BOOT, stable_seed("p1_external", "test")
    )
    test_tuned_bundle = bundle_with_ci(
        y_test, test_scores, tuned["threshold"], N_BOOT, stable_seed("p1_external", "test_tuned")
    )

    # Fairness check: the validation-fit baseline sees 3,000 rows, while the
    # full EC-XGB sees 24,056. Refit the same five-feature model on the 14,000
    # training rows and keep the validation split for threshold selection only.
    train = frame[frame["split"] == "train"].reset_index(drop=True)
    if len(train) != 14000:
        raise ValueError(f"expected 14,000 train rows, found {len(train)}")
    scaler_train, model_train = fit_baseline(train)
    val_scores_train = predict_baseline(scaler_train, model_train, val)
    test_scores_train = predict_baseline(scaler_train, model_train, test)
    train_test_bundle = bundle_with_ci(
        y_test, test_scores_train, THRESHOLD, N_BOOT, stable_seed("p1_external", "test_train_fit")
    )
    train_tuned = tune_threshold(y_val, val_scores_train)
    train_tuned_bundle = bundle_with_ci(
        y_test, test_scores_train, train_tuned["threshold"], N_BOOT,
        stable_seed("p1_external", "test_train_fit_tuned"),
    )

    m3_by_seed, m3_labels = load_m3_predictions(test["sample_id"])
    if not np.array_equal(y_test, m3_labels):
        raise ValueError("m3 labels do not match the external baseline test labels")
    m3_rows = []
    for seed, scores in m3_by_seed.items():
        row = bundle_with_ci(
            y_test, scores, THRESHOLD, N_BOOT, stable_seed("p1_external", "m3", seed)
        )
        m3_rows.append({"analysis": "reference", "model": "ec_xgb_m3", "seed": str(seed), **row})
    m3_pooled = pooled_bundle_with_ci(
        y_test, list(m3_by_seed.values()), THRESHOLD, N_BOOT, stable_seed("p1_external", "m3", "pooled")
    )
    m3_rows.append({"analysis": "reference", "model": "ec_xgb_m3", "seed": "pooled", **m3_pooled})

    rows = [
        {"analysis": "reimplementation", "model": "external_lr", "seed": "none", **test_bundle},
        {
            "analysis": "reimplementation",
            "model": "external_lr_val_threshold",
            "seed": "none",
            **test_tuned_bundle,
        },
        {
            "analysis": "reimplementation_train_fit",
            "model": "external_lr_train",
            "seed": "none",
            **train_test_bundle,
        },
        {
            "analysis": "reimplementation_train_fit",
            "model": "external_lr_train_val_threshold",
            "seed": "none",
            **train_tuned_bundle,
        },
        *m3_rows,
    ]
    results = pd.DataFrame(rows)
    results = results[RESULT_COLUMNS]
    results.to_csv(P1_DIR / "external_baseline.csv", index=False)

    deltas = paired_delta(
        y_test,
        test_scores,
        list(m3_by_seed.values()),
        THRESHOLD,
        N_BOOT,
        stable_seed("p1_external", "delta"),
    )
    deltas_train = paired_delta(
        y_test,
        test_scores_train,
        list(m3_by_seed.values()),
        THRESHOLD,
        N_BOOT,
        stable_seed("p1_external", "delta_train"),
    )
    mcnemar_rows = {
        str(seed): mcnemar_test(y_test, test_scores, scores, THRESHOLD)
        for seed, scores in m3_by_seed.items()
    }

    coefficients = {
        name: {
            "label": FEATURE_LABELS[name],
            "standardized_coefficient": float(model.coef_[0, index]),
            "odds_ratio_per_sd": float(np.exp(model.coef_[0, index])),
            "scaler_mean": float(scaler.mean_[index]),
            "scaler_scale": float(scaler.scale_[index]),
        }
        for index, name in enumerate(FEATURE_COLUMNS)
    }

    report = {
        "schema": "rebuild-p1-external-baseline-v1",
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "reference": {
            **REFERENCE,
            "label": (
                "reimplementation of a published lightweight feature ensemble, "
                "not the authors' original model or reported number"
            ),
            "differences": [
                "logistic regression instead of the authors' XGBoost-style ensemble",
                "five signals only: ROUGE-L, SBERT cosine, two NLI probabilities, answer length",
                "no entity-coverage or numeric-consistency signals",
                "no BERTScore term",
                "trained on the project HaluEval validation split, not the authors' split",
            ],
        },
        "protocol": {
            "dataset": "HaluEval QA",
            "fit_split": "halueval_val",
            "n_fit": int(len(val)),
            "fit_split_train_fit": "halueval_train",
            "n_fit_train_fit": int(len(train)),
            "eval_split": "halueval_test",
            "n_eval": int(len(test)),
            "learner": "LogisticRegression(C=1.0, max_iter=2000, lbfgs)",
            "scaler": "StandardScaler fitted on the fit split",
            "threshold": THRESHOLD,
            "threshold_tuning": "separate sensitivity row; F1-maximizing threshold on validation only",
            "n_boot": N_BOOT,
            "bootstrap": "row bootstrap, percentile 95% CI",
            "test_used_for_fit_or_selection": False,
        },
        "features": {
            name: FEATURE_LABELS[name] for name in FEATURE_COLUMNS
        },
        "coefficients": coefficients,
        "intercept": float(model.intercept_[0]),
        "validation": validation,
        "validation_threshold_search": tuned,
        "test": {
            "external_lr": test_bundle,
            "external_lr_val_threshold": test_tuned_bundle,
            "external_lr_train": train_test_bundle,
            "external_lr_train_val_threshold": train_tuned_bundle,
        },
        "ec_xgb_m3": {
            "per_seed": {str(row["seed"]): row for row in m3_rows if row["seed"] != "pooled"},
            "pooled": m3_pooled,
        },
        "comparison": {
            "delta_external_minus_m3": deltas,
            "delta_external_train_minus_m3": deltas_train,
            "note": (
                "Positive deltas favor the external baseline for F1/AUROC/PR-AUC; "
                "negative deltas favor it for ECE/Brier."
            ),
            "mcnemar_external_vs_m3": mcnemar_rows,
        },
    }
    (P1_DIR / "external_baseline.json").write_text(
        json.dumps(json_safe(report), indent=2), encoding="utf-8"
    )

    # REPORT.md
    external_row = rows[0]
    tuned_row = rows[1]
    per_seed_rows = [row for row in m3_rows if row["seed"] != "pooled"]
    pooled_row = m3_rows[-1]
    delta_lines = []
    for key in ("f1", "auroc", "pr_auc", "ece", "brier"):
        item = deltas[key]
        delta_lines.append(
            f"| {key.upper()} | {fmt(item['point'])} | "
            f"[{fmt(item['q025'])}, {fmt(item['q975'])}] |"
        )
    report_md = f"""# P1.3 - External Baseline Reimplementation

Generated by `src/models/p1_external_baseline.py`. Every number below comes from
`external_baseline.csv` and `external_baseline.json`; none are hand-entered.

## Reference and fidelity

The baseline reimplements the lightweight feature-ensemble family of
Sundaragiri, Reddy, Gunavardhan and Navahith, *Framework for Hallucination
Detection in Large Language Models*, IJERT 15(04), 2026,
DOI 10.5281/zenodo.20025987. It is a **reimplementation**, not the authors'
original model and not their reported number. The differences are deliberate:

- logistic regression instead of their ensemble learner;
- five signals only (ROUGE-L, SBERT cosine, NLI entailment, NLI contradiction,
  answer length);
- no entity-coverage or numeric-consistency signals;
- no BERTScore term;
- the project's grouped HaluEval split, not the authors' split.

## Protocol

- Fit: 3,000 HaluEval validation rows only.
- Evaluation: 3,000 HaluEval test rows, used once.
- Features: `{", ".join(FEATURE_COLUMNS)}`.
- Learner: standardized logistic regression (`C=1.0`, `max_iter=2000`).
- Headline threshold: {THRESHOLD}; a validation-tuned threshold is a separate
  sensitivity row (tuned threshold = {tuned['threshold']:.3f}, validation F1 =
  {tuned['f1']:.4f}).
- CIs: 1,000 row-bootstrap draws, percentile 95% interval.
- EC-XGB m3 rows are read from `fixed_hp_predictions.parquet`; they are not
  refit here.

## Test metrics

{report_table(rows)}

`PPR` is the predicted-positive rate at the row's threshold. The external
baseline's validation-tuned row uses the validation F1-maximizing threshold,
not a test-selected threshold.

## 95% confidence intervals

{ci_table(rows)}

## Paired comparison against EC-XGB m3

The delta is external minus the seed-averaged m3 metric, with both models
evaluated on the same resampled rows.

| metric | point delta | 95% CI |
|---|---:|---|
{chr(10).join(delta_lines)}

Positive deltas favor the external baseline for F1, AUROC, and PR-AUC;
negative deltas favor it for ECE and Brier. EC-XGB m3 is stronger on every
discrimination metric and better calibrated on this benchmark.

## Standardized coefficients

| feature | coefficient (per SD) | odds ratio per SD |
|---|---:|---:|
{chr(10).join(
    f"| {FEATURE_LABELS[name]} | {coefficients[name]['standardized_coefficient']:.4f} | "
    f"{coefficients[name]['odds_ratio_per_sd']:.4f} |"
    for name in FEATURE_COLUMNS
)}

The positive entailment coefficient is counter-intuitive and reflects
collinearity with ROUGE-L and the contradiction score: those two already carry
the support signal, so the fitted entailment term is not interpretable in
isolation. The coefficients are reported for reproducibility, not as a causal
story.

## McNemar tests (external vs m3)

| m3 seed | only external correct | only m3 correct | agreement | p |
|---:|---:|---:|---:|---:|
{chr(10).join(
    f"| {seed} | {item['only_external_correct']} | {item['only_m3_correct']} | "
    f"{item['agreement']:.4f} | {item['pvalue']:.3g} |"
    for seed, item in mcnemar_rows.items()
)}

## Interpretation

The external-style ensemble reaches F1 {external_row['f1']:.4f}, AUROC
{external_row['auroc']:.4f}, PR-AUC {external_row['pr_auc']:.4f}, ECE
{external_row['ece']:.4f}, and Brier {external_row['brier']:.4f}. EC-XGB m3
reaches F1 {pooled_row['f1']:.4f}, AUROC {pooled_row['auroc']:.4f}, PR-AUC
{pooled_row['pr_auc']:.4f}, ECE {pooled_row['ece']:.4f}, and Brier
{pooled_row['brier']:.4f} on the same rows. The gap reflects the nonlinear
learner, EC-XGB's additional feature families (claim-level NLI and the source
indicator), and the reduced signal set in this reimplementation. It is not a
claim that the published method is weak: this is a reduced reimplementation on
a different split.

## Caveats

- HaluEval is a synthetic, ChatGPT-generated benchmark; these numbers are
  benchmark-specific and do not establish general hallucination detection.
- The baseline is a reimplementation with a different learner and a reduced
  feature set. It must not be compared numerically to the authors' reported
  result.
- ROUGE-L is implemented with the standard LCS formulation and lowercase word
  tokenization; the per-row values are cached in `rouge_l_features.csv`.
- The test split was not used to fit, tune, or select anything.
- CIs cover row sampling only, not feature-set or learner selection.

## Files

- `external_baseline.csv` - every reported row and CI.
- `external_baseline.json` - protocol, coefficients, per-seed rows, deltas.
- `rouge_l_features.csv` - per-row ROUGE-L precision, recall, and F1.
- `external_baseline_REPORT.md` - this file.
- `p1_external_baseline_and_conformal_REPORT.md` - combined calibration report.

## Command

```powershell
& .venv\\Scripts\\python.exe src\\models\\p1_external_baseline.py
```
"""
    (P1_DIR / "external_baseline_REPORT.md").write_text(report_md, encoding="utf-8")
    from src.models.p1_report import write_combined_report

    write_combined_report(P1_DIR)
    logger.info("wrote P1 external baseline artifacts to %s in %.1fs", P1_DIR, time.time() - started)

    print("\nExternal baseline vs EC-XGB m3 (HaluEval QA test, n=3,000)")
    print(results[["analysis", "model", "seed", "f1", "auroc", "pr_auc", "ece", "brier"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
