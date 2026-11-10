"""WP2 - calibration rebuild on the fixed-hyperparameter ablation.

Protocol
--------
* Source calibrator: Platt fitted on the 3,000 HaluEval validation rows only.
* Target calibrator: Platt fitted on the 5,034 RAGTruth QA calibration rows.
* Evaluation: HaluEval test (3,000 rows, in-domain) and RAGTruth QA test
  (900 rows, held out).  Test rows are never used to fit or select anything.
* Isotonic fits are reported as fixed reference rows, not selected by test ECE.
* Prior-only constants at the calibration prevalence (0.3107) and the test
  prevalence (0.1778), with ECE and Brier.
* Label budget: 100 draws at 50, 100, 252, 500, 1,000, and 5,034 calibration
  rows; Platt refit per draw and evaluated on the 900-row QA test.
* CIs: 1,000 row-bootstrap draws for every reported metric.

Outputs (artifacts/results/rebuild/calibration/):
  calibration.csv, prior_only.csv, calibration_slope.csv, label_budget.csv,
  calibration_report.json, REPORT.md

Run (repo root, .venv):
  python src/models/rebuild_calibration.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd

from src.models.config import SEEDS
from src.models.rebuild_common import (
    B6_MODELS,
    B6_PRED,
    CALIBRATION_DIR,
    CALIBRATION_PREVALENCE,
    CI_KEYS,
    ECE_BINS,
    FIXED_HP_MODELS,
    FIXED_HP_PRED,
    FPR_TARGET,
    N_BOOT_CALIBRATION,
    TEST_PREVALENCE,
    THRESHOLD,
    VARIANT,
    apply_isotonic,
    apply_platt,
    bootstrap_calibration_coefficients,
    bootstrap_metric_cis,
    bootstrap_metric_cis_pooled,
    calibration_coefficients,
    fit_isotonic,
    fit_platt,
    load_calibration_features,
    load_halueval_features,
    metric_bundle,
    predict_raw,
    stable_seed,
    summarize_bootstrap,
)
from src.models.run_b6_modified import SOURCE_COLUMN  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rebuild_calibration")

MODEL_SOURCES = {
    "fixed_hp": {"models": FIXED_HP_MODELS, "predictions": FIXED_HP_PRED},
    "frozen_b6_tuned": {"models": B6_MODELS, "predictions": B6_PRED},
}
EVAL_DATASETS = ("halueval_test", "ragtruth_qa_test")
CALIBRATORS = {
    "fixed_hp": ("raw", "source_platt", "target_platt", "source_isotonic", "target_isotonic"),
    "frozen_b6_tuned": ("raw", "source_platt", "target_platt"),
}
LABEL_BUDGET_SIZES = (50, 100, 252, 500, 1000, 5034)
LABEL_BUDGET_DRAWS = 100

CALIBRATION_COLUMNS = [
    "model", "variant", "seed", "dataset", "calibrator", "n", "n_seeds",
    "prevalence", "threshold", "ece", "brier", "brier_skill_score", "auroc",
    "pr_auc", "f1", "precision", "recall", "predicted_positive_rate",
    "recall_at_5pct_fpr", "ece_ci_low", "ece_ci_high", "brier_ci_low",
    "brier_ci_high", "brier_skill_score_ci_low", "brier_skill_score_ci_high",
    "auroc_ci_low", "auroc_ci_high", "pr_auc_ci_low", "pr_auc_ci_high",
    "f1_ci_low", "f1_ci_high", "n_boot", "n_boot_valid",
]
SLOPE_COLUMNS = [
    "model", "variant", "seed", "fit_split", "n_fit", "fit_prevalence",
    "platt_coef", "platt_coef_ci_low", "platt_coef_ci_high", "platt_intercept",
    "platt_intercept_ci_low", "platt_intercept_ci_high", "cal_slope",
    "cal_slope_ci_low", "cal_slope_ci_high", "cal_intercept",
    "cal_intercept_ci_low", "cal_intercept_ci_high", "n_boot", "n_boot_valid",
]
PRIOR_COLUMNS = [
    "model", "variant", "predictor", "constant", "eval_split", "n",
    "prevalence", "ece", "brier", "ece_zero_by_construction", "note",
]
LABEL_BUDGET_COLUMNS = [
    "model", "variant", "seed", "size", "n_draws", "n_draws_used", "n_test",
    "test_prevalence", "ece_mean", "ece_sd", "ece_q025", "ece_q975",
    "brier_mean", "brier_sd", "brier_q025", "brier_q975",
]

POINT_KEYS = (
    "ece", "brier", "brier_skill_score", "auroc", "pr_auc", "f1",
    "precision", "recall", "predicted_positive_rate", "recall_at_5pct_fpr",
)


# --------------------------------------------------------------------------
# Setup and sanity checks
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


def evaluation_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (HaluEval val, RAGTruth QA calibration, external QA test) frames."""
    halo = load_halueval_features()
    cal = load_calibration_features()
    external = pd.read_parquet(
        Path(__file__).resolve().parents[2] / "data" / "processed" / "external_features.parquet"
    )
    external[SOURCE_COLUMN] = 1.0
    qa = external[
        (external["source_dataset"] == "ragtruth")
        & (external["task"] == "qa")
        & (external["split"] == "test")
    ].copy()
    if len(qa) != 900:
        raise ValueError(f"expected 900 RAGTruth QA test rows, found {len(qa)}")
    return (
        halo[halo["split"] == "val"].reset_index(drop=True),
        cal.reset_index(drop=True),
        qa.reset_index(drop=True),
    )


def test_predictions(source: str, seed: int, dataset: str) -> pd.DataFrame:
    preds = pd.read_parquet(MODEL_SOURCES[source]["predictions"])
    sub = preds[
        (preds["variant"] == VARIANT)
        & (preds["seed"] == seed)
        & (preds["dataset"] == dataset)
    ]
    if sub.empty:
        raise ValueError(f"no predictions for {source} seed {seed} {dataset}")
    return sub.sort_values("sample_id").reset_index(drop=True)


def build_seed_fits(
    source: str,
    seed: int,
    val: pd.DataFrame,
    cal: pd.DataFrame,
    qa: pd.DataFrame,
    halo_test: pd.DataFrame,
) -> dict:
    """Fit source/target calibrators for one model seed and collect scores."""
    model = joblib.load(MODEL_SOURCES[source]["models"] / f"xgboost_m3_seed_{seed}.joblib")
    val_scores = predict_raw(model, val)
    cal_scores = predict_raw(model, cal)
    y_val = val["label"].to_numpy(dtype=int)
    y_cal = cal["label"].to_numpy(dtype=int)

    frames = {"halueval_test": halo_test, "ragtruth_qa_test": qa}
    test = {}
    for dataset in EVAL_DATASETS:
        frame = frames[dataset].sort_values("sample_id").reset_index(drop=True)
        stored = test_predictions(source, seed, dataset)
        if not np.array_equal(frame["sample_id"].to_numpy(), stored["sample_id"].to_numpy()):
            raise ValueError(f"{dataset}: sample_id order mismatch for {source} seed {seed}")
        if not np.array_equal(frame["label"].to_numpy(dtype=int), stored["label"].to_numpy(dtype=int)):
            raise ValueError(f"{dataset}: label mismatch for {source} seed {seed}")
        model_scores = predict_raw(model, frame)
        max_diff = float(np.max(np.abs(model_scores - stored["score"].to_numpy())))
        if max_diff > 1e-6:
            raise ValueError(
                f"{source} seed {seed} {dataset}: stored scores differ from model by {max_diff:.3e}"
            )
        test[dataset] = {
            "y": stored["label"].to_numpy(dtype=int),
            "raw": stored["score"].to_numpy(dtype=float),
        }

    return {
        "val_scores": val_scores,
        "y_val": y_val,
        "cal_scores": cal_scores,
        "y_cal": y_cal,
        "test": test,
    }


def calibrator_scores(fits: dict) -> dict:
    """Fit the four calibrators once, then apply them to both test sets."""
    source_platt = fit_platt(fits["val_scores"], fits["y_val"])
    target_platt = fit_platt(fits["cal_scores"], fits["y_cal"])
    source_isotonic = fit_isotonic(fits["val_scores"], fits["y_val"])
    target_isotonic = fit_isotonic(fits["cal_scores"], fits["y_cal"])

    scores = {}
    for dataset, block in fits["test"].items():
        raw = block["raw"]
        scores[dataset] = {
            "raw": raw,
            "source_platt": apply_platt(source_platt, raw),
            "target_platt": apply_platt(target_platt, raw),
            "source_isotonic": apply_isotonic(source_isotonic, raw),
            "target_isotonic": apply_isotonic(target_isotonic, raw),
        }
    return scores


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def metric_row(
    base: dict, bundle: dict, cis: dict, n_boot: int, n_seeds: int
) -> dict:
    row = dict(base)
    row["n"] = int(bundle["n"])
    row["n_seeds"] = int(n_seeds)
    row["prevalence"] = float(bundle["prevalence"])
    row["threshold"] = float(bundle["threshold"])
    for key in POINT_KEYS:
        row[key] = float(bundle[key])
    for key in CI_KEYS:
        row[f"{key}_ci_low"] = float(cis[key]["q025"])
        row[f"{key}_ci_high"] = float(cis[key]["q975"])
    row["n_boot"] = int(n_boot)
    row["n_boot_valid"] = int(min(cis[key]["n_valid"] for key in CI_KEYS))
    return row


def mean_bundle(bundles: list[dict]) -> dict:
    out = {"n": int(bundles[0]["n"]), "threshold": THRESHOLD}
    for key in POINT_KEYS + ("prevalence",):
        out[key] = float(np.mean([bundle[key] for bundle in bundles]))
    return out


def build_calibration_table(
    all_fits: dict, scores: dict
) -> pd.DataFrame:
    rows = []
    for source, seeds in all_fits.items():
        for seed in SEEDS:
            for dataset in EVAL_DATASETS:
                y = seeds[seed]["test"][dataset]["y"]
                for calibrator in CALIBRATORS[source]:
                    p = scores[source][seed][dataset][calibrator]
                    bundle = metric_bundle(y, p, THRESHOLD)
                    cis = bootstrap_metric_cis(
                        y,
                        p,
                        n_boot=N_BOOT_CALIBRATION,
                        seed=stable_seed("calibration", source, seed, dataset, calibrator),
                    )
                    rows.append(
                        metric_row(
                            {
                                "model": source,
                                "variant": VARIANT,
                                "seed": str(seed),
                                "dataset": dataset,
                                "calibrator": calibrator,
                            },
                            bundle,
                            cis,
                            N_BOOT_CALIBRATION,
                            n_seeds=1,
                        )
                    )

        # Seed-averaged rows: one shared row resample per draw.
        for dataset in EVAL_DATASETS:
            for calibrator in CALIBRATORS[source]:
                pairs = [
                    (seeds[seed]["test"][dataset]["y"], scores[source][seed][dataset][calibrator])
                    for seed in SEEDS
                ]
                bundles = [metric_bundle(y, p, THRESHOLD) for y, p in pairs]
                cis = bootstrap_metric_cis_pooled(
                    pairs,
                    n_boot=N_BOOT_CALIBRATION,
                    seed=stable_seed("calibration_pooled", source, dataset, calibrator),
                )
                rows.append(
                    metric_row(
                        {
                            "model": source,
                            "variant": VARIANT,
                            "seed": "pooled",
                            "dataset": dataset,
                            "calibrator": calibrator,
                        },
                        mean_bundle(bundles),
                        cis,
                        N_BOOT_CALIBRATION,
                        n_seeds=len(SEEDS),
                    )
                )
    return pd.DataFrame(rows, columns=CALIBRATION_COLUMNS)


def build_slope_table(all_fits: dict) -> pd.DataFrame:
    rows = []
    for source, seeds in all_fits.items():
        for seed in SEEDS:
            fits = seeds[seed]
            for fit_split, scores, labels in (
                ("halueval_val", fits["val_scores"], fits["y_val"]),
                ("ragtruth_qa_calibration", fits["cal_scores"], fits["y_cal"]),
            ):
                point = calibration_coefficients(scores, labels)
                boot = bootstrap_calibration_coefficients(
                    scores,
                    labels,
                    n_boot=N_BOOT_CALIBRATION,
                    seed=stable_seed("slope", source, seed, fit_split),
                )
                row = {
                    "model": source,
                    "variant": VARIANT,
                    "seed": str(seed),
                    "fit_split": fit_split,
                    "n_fit": int(len(labels)),
                    "fit_prevalence": float(np.mean(labels)),
                    "n_boot": N_BOOT_CALIBRATION,
                    "n_boot_valid": int(min(item["n_valid"] for item in boot.values())),
                }
                for key, value in point.items():
                    row[key] = float(value)
                    row[f"{key}_ci_low"] = float(boot[key]["q025"])
                    row[f"{key}_ci_high"] = float(boot[key]["q975"])
                rows.append(row)
    return pd.DataFrame(rows, columns=SLOPE_COLUMNS)


def build_prior_table(cal: pd.DataFrame, qa: pd.DataFrame) -> pd.DataFrame:
    splits = {
        "ragtruth_qa_calibration": cal["label"].to_numpy(dtype=int),
        "ragtruth_qa_test": qa["label"].to_numpy(dtype=int),
    }
    priors = (
        ("calibration_prior", CALIBRATION_PREVALENCE),
        ("test_prior", TEST_PREVALENCE),
    )
    rows = []
    for predictor, constant in priors:
        for split, labels in splits.items():
            bundle = metric_bundle(labels, np.full(len(labels), constant), THRESHOLD)
            zero_by_construction = bool(abs(constant - float(labels.mean())) < 1e-12)
            rows.append(
                {
                    "model": "prior_only",
                    "variant": "constant",
                    "predictor": predictor,
                    "constant": float(constant),
                    "eval_split": split,
                    "n": int(len(labels)),
                    "prevalence": float(labels.mean()),
                    "ece": float(bundle["ece"]),
                    "brier": float(bundle["brier"]),
                    "ece_zero_by_construction": zero_by_construction,
                    "note": (
                        "ECE is exactly zero by construction when the constant equals "
                        "the evaluated prevalence; it is a base-rate check, not evidence "
                        "of calibration skill. Brier remains the informative term."
                    ),
                }
            )
    return pd.DataFrame(rows, columns=PRIOR_COLUMNS)


def build_label_budget_table(all_fits: dict) -> tuple[pd.DataFrame, dict]:
    cal_scores = {seed: all_fits["fixed_hp"][seed]["cal_scores"] for seed in SEEDS}
    y_cal = all_fits["fixed_hp"][SEEDS[0]]["y_cal"]
    test = {
        seed: all_fits["fixed_hp"][seed]["test"]["ragtruth_qa_test"] for seed in SEEDS
    }
    rows = []
    summary = {}
    for size in LABEL_BUDGET_SIZES:
        rng = np.random.default_rng(stable_seed("label_budget", size))
        draws = [rng.choice(len(y_cal), size=size, replace=False) for _ in range(LABEL_BUDGET_DRAWS)]
        per_seed = {seed: {"ece": [], "brier": []} for seed in SEEDS}
        pooled = {"ece": [], "brier": []}
        for idx in draws:
            if len(np.unique(y_cal[idx])) < 2:
                continue
            values = {}
            for seed in SEEDS:
                model = fit_platt(cal_scores[seed][idx], y_cal[idx])
                scores = apply_platt(model, test[seed]["raw"])
                bundle = metric_bundle(test[seed]["y"], scores, THRESHOLD)
                values[seed] = (bundle["ece"], bundle["brier"])
                per_seed[seed]["ece"].append(bundle["ece"])
                per_seed[seed]["brier"].append(bundle["brier"])
            pooled["ece"].append(float(np.mean([values[seed][0] for seed in SEEDS])))
            pooled["brier"].append(float(np.mean([values[seed][1] for seed in SEEDS])))

        for seed in SEEDS:
            ece_summary = summarize_bootstrap(per_seed[seed]["ece"])
            brier_summary = summarize_bootstrap(per_seed[seed]["brier"])
            rows.append(
                {
                    "model": "fixed_hp",
                    "variant": VARIANT,
                    "seed": str(seed),
                    "size": int(size),
                    "n_draws": LABEL_BUDGET_DRAWS,
                    "n_draws_used": int(ece_summary["n_valid"]),
                    "n_test": int(len(test[seed]["y"])),
                    "test_prevalence": float(np.mean(test[seed]["y"])),
                    "ece_mean": ece_summary["mean"],
                    "ece_sd": ece_summary["sd"],
                    "ece_q025": ece_summary["q025"],
                    "ece_q975": ece_summary["q975"],
                    "brier_mean": brier_summary["mean"],
                    "brier_sd": brier_summary["sd"],
                    "brier_q025": brier_summary["q025"],
                    "brier_q975": brier_summary["q975"],
                }
            )
        ece_summary = summarize_bootstrap(pooled["ece"])
        brier_summary = summarize_bootstrap(pooled["brier"])
        rows.append(
            {
                "model": "fixed_hp",
                "variant": VARIANT,
                "seed": "pooled",
                "size": int(size),
                "n_draws": LABEL_BUDGET_DRAWS,
                "n_draws_used": int(ece_summary["n_valid"]),
                "n_test": int(len(test[SEEDS[0]]["y"])),
                "test_prevalence": float(np.mean(test[SEEDS[0]]["y"])),
                "ece_mean": ece_summary["mean"],
                "ece_sd": ece_summary["sd"],
                "ece_q025": ece_summary["q025"],
                "ece_q975": ece_summary["q975"],
                "brier_mean": brier_summary["mean"],
                "brier_sd": brier_summary["sd"],
                "brier_q025": brier_summary["q025"],
                "brier_q975": brier_summary["q975"],
            }
        )
        summary[str(size)] = {
            "pooled": {
                "ece_mean": ece_summary["mean"],
                "ece_sd": ece_summary["sd"],
                "ece_q025": ece_summary["q025"],
                "ece_q975": ece_summary["q975"],
                "brier_mean": brier_summary["mean"],
                "brier_sd": brier_summary["sd"],
                "brier_q025": brier_summary["q025"],
                "brier_q975": brier_summary["q975"],
            },
            "per_seed": {str(seed): {
                "ece_mean": float(np.mean(per_seed[seed]["ece"])),
                "ece_sd": float(np.std(per_seed[seed]["ece"], ddof=1)),
                "brier_mean": float(np.mean(per_seed[seed]["brier"])),
                "brier_sd": float(np.std(per_seed[seed]["brier"], ddof=1)),
            } for seed in SEEDS},
        }
    return pd.DataFrame(rows, columns=LABEL_BUDGET_COLUMNS), summary


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def select_rows(
    table: pd.DataFrame, model: str, dataset: str, calibrator: str, seed: str | None = None
) -> pd.DataFrame:
    sub = table[
        (table["model"] == model)
        & (table["dataset"] == dataset)
        & (table["calibrator"] == calibrator)
    ]
    if seed is not None:
        sub = sub[sub["seed"] == seed]
    return sub


def fmt(value: float, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def build_report(
    table: pd.DataFrame,
    slopes: pd.DataFrame,
    prior: pd.DataFrame,
    budget: pd.DataFrame,
    budget_summary: dict,
    all_fits: dict,
) -> dict:
    def block(model: str, dataset: str, calibrator: str) -> dict:
        pooled = select_rows(table, model, dataset, calibrator, "pooled").iloc[0]
        per_seed = select_rows(table, model, dataset, calibrator)
        per_seed = per_seed[per_seed["seed"] != "pooled"]
        return {
            "dataset": dataset,
            "calibrator": calibrator,
            "pooled": {
                key: float(pooled[key])
                for key in POINT_KEYS
                if key in pooled and np.isfinite(pooled[key])
            },
            "pooled_ci": {
                f"{key}_ci": [float(pooled[f"{key}_ci_low"]), float(pooled[f"{key}_ci_high"])]
                for key in CI_KEYS
            },
            "per_seed": {
                str(row["seed"]): {
                    key: float(row[key]) for key in POINT_KEYS if np.isfinite(row[key])
                }
                for _, row in per_seed.iterrows()
            },
        }

    slope_block = {}
    for model in MODEL_SOURCES:
        slope_block[model] = {}
        for split in ("halueval_val", "ragtruth_qa_calibration"):
            sub = slopes[(slopes["model"] == model) & (slopes["fit_split"] == split)]
            slope_block[model][split] = {
                "n_fit": int(sub["n_fit"].iloc[0]),
                "fit_prevalence": float(sub["fit_prevalence"].iloc[0]),
                "mean": {
                    key: float(sub[key].mean())
                    for key in ("platt_coef", "platt_intercept", "cal_slope", "cal_intercept")
                },
                "per_seed": {
                    str(row["seed"]): {
                        key: float(row[key])
                        for key in ("platt_coef", "platt_intercept", "cal_slope", "cal_intercept")
                    }
                    for _, row in sub.iterrows()
                },
            }

    return {
        "schema": "rebuild-wp2-calibration-v1",
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "script": "src/models/rebuild_calibration.py",
        "protocol": {
            "variant": VARIANT,
            "seeds": list(SEEDS),
            "threshold": THRESHOLD,
            "ece_bins": ECE_BINS,
            "fpr_target": FPR_TARGET,
            "n_boot": N_BOOT_CALIBRATION,
            "bootstrap": "row bootstrap, percentile interval",
            "source_calibration_split": "HaluEval validation (3,000 rows)",
            "target_calibration_split": "RAGTruth QA calibration (5,034 rows)",
            "evaluation_splits": ["HaluEval test (3,000 rows)", "RAGTruth QA test (900 rows)"],
            "calibrators": list(CALIBRATORS["fixed_hp"]),
            "selection_rule": "Platt is fixed a priori; test ECE is never used for selection",
            "prior_constants": [CALIBRATION_PREVALENCE, TEST_PREVALENCE],
            "label_budget": {
                "sizes": list(LABEL_BUDGET_SIZES),
                "draws_per_size": LABEL_BUDGET_DRAWS,
                "sampling": "without replacement from the 5,034 calibration rows",
                "evaluation": "900-row RAGTruth QA test, one Platt fit per draw",
            },
        },
        "split_sizes": {
            "halueval_val": int(len(all_fits["fixed_hp"][SEEDS[0]]["y_val"])),
            "ragtruth_qa_calibration": int(len(all_fits["fixed_hp"][SEEDS[0]]["y_cal"])),
            "halueval_test": int(len(all_fits["fixed_hp"][SEEDS[0]]["test"]["halueval_test"]["y"])),
            "ragtruth_qa_test": int(len(all_fits["fixed_hp"][SEEDS[0]]["test"]["ragtruth_qa_test"]["y"])),
        },
        "prevalences": {
            "ragtruth_qa_calibration": float(np.mean(all_fits["fixed_hp"][SEEDS[0]]["y_cal"])),
            "ragtruth_qa_test": float(np.mean(all_fits["fixed_hp"][SEEDS[0]]["test"]["ragtruth_qa_test"]["y"])),
            "halueval_val": float(np.mean(all_fits["fixed_hp"][SEEDS[0]]["y_val"])),
            "halueval_test": float(np.mean(all_fits["fixed_hp"][SEEDS[0]]["test"]["halueval_test"]["y"])),
        },
        "slopes": slope_block,
        "headline": {
            "source_platt_halueval_test": block("fixed_hp", "halueval_test", "source_platt"),
            "raw_ragtruth_qa_test": block("fixed_hp", "ragtruth_qa_test", "raw"),
            "source_platt_ragtruth_qa_test": block("fixed_hp", "ragtruth_qa_test", "source_platt"),
            "target_platt_ragtruth_qa_test": block("fixed_hp", "ragtruth_qa_test", "target_platt"),
            "target_isotonic_ragtruth_qa_test": block("fixed_hp", "ragtruth_qa_test", "target_isotonic"),
        },
        "prior_only": prior.to_dict(orient="records"),
        "label_budget": budget_summary,
        "caveats": [
            "Test rows are never used to fit or select a calibrator. The source calibrator uses the 3,000 HaluEval validation rows; the target calibrator uses the 5,034 RAGTruth QA calibration rows.",
            "ECE uses 10 equal-width bins and is sensitive to binning. A prior-only constant has ECE zero by construction on the split whose prevalence defines it.",
            "RAGTruth QA test discrimination is near chance in this rebuild; a calibrator can reduce ECE and Brier without adding ranking skill, and the Brier skill score can remain negative.",
            "WP2 intervals resample rows i.i.d.; the paired HaluEval items and external source groups are handled by the WP3 two-level bootstrap.",
            "Isotonic rows are fixed reference fits, not selected by test ECE. The reported headline and the protocol choice are Platt.",
            "The RAGTruth QA calibration split and the 900-row test split are disjoint at the sample_id level.",
        ],
    }


def build_markdown(
    table: pd.DataFrame,
    slopes: pd.DataFrame,
    prior: pd.DataFrame,
    budget: pd.DataFrame,
) -> str:
    lines = [
        "# WP2 - Calibration Rebuild",
        "",
        "Generated by `src/models/rebuild_calibration.py`. All numbers below come from",
        "the CSV files in this directory; none are hand-entered.",
        "",
        "## Protocol",
        "",
        "- Model: m3 (EC-XGB, 35 features) under the fixed WP1 hyperparameters.",
        "- Source Platt: fitted on the 3,000 HaluEval validation rows only.",
        "- Target Platt: fitted on the 5,034 RAGTruth QA calibration rows.",
        "- Evaluation: 3,000 HaluEval test rows and the disjoint 900-row RAGTruth QA test.",
        "- Test rows are never used to fit or select a calibrator. Headline choice is Platt.",
        f"- Intervals: {N_BOOT_CALIBRATION:,} row-bootstrap draws, percentile 95% CI.",
        f"- ECE: {ECE_BINS} equal-width bins. Decision threshold: {THRESHOLD}.",
        "",
        "## Commands",
        "",
        "```powershell",
        "& .venv\\Scripts\\python.exe src\\models\\rebuild_calibration.py",
        "& .venv\\Scripts\\python.exe -m pytest tests\\test_rebuild_calibration.py -v",
        "```",
        "",
        "## Calibration slope and intercept",
        "",
        "`cal_slope` is the Cox slope from a logistic regression of the label on",
        "logit(raw score); a perfectly calibrated score has slope 1 and intercept 0.",
        "`platt_coef` is the coefficient of the applied Platt fit on the raw score.",
        "",
        "| model | fit split | n | prevalence | Cox slope (mean) | Cox intercept (mean) | Platt coef (mean) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_SOURCES:
        for split in ("halueval_val", "ragtruth_qa_calibration"):
            sub = slopes[(slopes["model"] == model) & (slopes["fit_split"] == split)]
            lines.append(
                f"| {model} | {split} | {int(sub['n_fit'].iloc[0]):,} | "
                f"{fmt(sub['fit_prevalence'].iloc[0])} | {fmt(sub['cal_slope'].mean())} | "
                f"{fmt(sub['cal_intercept'].mean())} | {fmt(sub['platt_coef'].mean())} |"
            )
    lines += [
        "",
        "Per-seed slope values and 95% bootstrap CIs are in `calibration_slope.csv`.",
        "",
        "## Test metrics (seed-averaged)",
        "",
        "| dataset | calibrator | ECE | Brier | Brier skill | AUROC | PR-AUC | F1@0.5 | PPR | Recall@5%FPR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("fixed_hp",):
        for dataset in EVAL_DATASETS:
            for calibrator in CALIBRATORS[model]:
                row = select_rows(table, model, dataset, calibrator, "pooled").iloc[0]
                lines.append(
                    f"| {dataset} | {calibrator} | {fmt(row['ece'])} | {fmt(row['brier'])} | "
                    f"{fmt(row['brier_skill_score'])} | {fmt(row['auroc'])} | {fmt(row['pr_auc'])} | "
                    f"{fmt(row['f1'])} | {fmt(row['predicted_positive_rate'])} | "
                    f"{fmt(row['recall_at_5pct_fpr'])} |"
                )
    lines += [
        "",
        "95% CIs for every column are in `calibration.csv` (`*_ci_low`, `*_ci_high`).",
        "`predicted_positive_rate` is the flag rate at threshold 0.5.",
        "",
        "## Key 95% intervals (seed-averaged)",
        "",
        "| dataset | calibrator | ECE [95% CI] | Brier [95% CI] | AUROC [95% CI] | F1@0.5 [95% CI] |",
        "|---|---|---|---|---|---|",
    ]
    key_rows = (
        ("halueval_test", "raw"),
        ("halueval_test", "source_platt"),
        ("ragtruth_qa_test", "raw"),
        ("ragtruth_qa_test", "source_platt"),
        ("ragtruth_qa_test", "target_platt"),
    )

    def interval(row: pd.Series, key: str) -> str:
        return (
            f"{fmt(row[key])} [{fmt(row[f'{key}_ci_low'])}, "
            f"{fmt(row[f'{key}_ci_high'])}]"
        )

    for dataset, calibrator in key_rows:
        row = select_rows(table, "fixed_hp", dataset, calibrator, "pooled").iloc[0]
        lines.append(
            f"| {dataset} | {calibrator} | {interval(row, 'ece')} | "
            f"{interval(row, 'brier')} | {interval(row, 'auroc')} | "
            f"{interval(row, 'f1')} |"
        )
    lines += [
        "",
        "## Prior-only predictors",
        "",
        "| predictor | constant | evaluated on | n | prevalence | ECE | Brier | ECE zero by construction |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for _, row in prior.iterrows():
        lines.append(
            f"| {row['predictor']} | {fmt(row['constant'])} | {row['eval_split']} | "
            f"{int(row['n']):,} | {fmt(row['prevalence'])} | {fmt(row['ece'])} | "
            f"{fmt(row['brier'])} | {str(bool(row['ece_zero_by_construction'])).lower()} |"
        )
    lines += [
        "",
        "ECE is exactly zero by construction when the constant equals the evaluated",
        "prevalence. It is a base-rate check, not evidence of calibration skill; the",
        "Brier score is the informative prior-only term.",
        "",
        "## Label budget",
        "",
        f"{LABEL_BUDGET_DRAWS} draws per size, sampled without replacement from the",
        "5,034 calibration rows. Each draw refits Platt and is scored on the 900-row",
        "RAGTruth QA test. The pooled row averages the three model seeds per draw.",
        "",
        "| calibration rows | ECE mean | ECE SD | ECE 2.5-97.5% | Brier mean | Brier SD |",
        "|---:|---:|---:|---|---:|---:|",
    ]
    for size in LABEL_BUDGET_SIZES:
        row = budget[(budget["seed"] == "pooled") & (budget["size"] == size)].iloc[0]
        lines.append(
            f"| {size:,} | {fmt(row['ece_mean'])} | {fmt(row['ece_sd'])} | "
            f"{fmt(row['ece_q025'])}-{fmt(row['ece_q975'])} | "
            f"{fmt(row['brier_mean'])} | {fmt(row['brier_sd'])} |"
        )
    lines += [
        "",
        "Per-seed means, SDs, and percentiles are in `label_budget.csv`. At 5,034",
        "rows every draw is the full split, so the SD is zero by construction.",
        "",
        "## Notes and risks",
        "",
        "- RAGTruth QA test AUROC is near chance (about 0.55). Target calibration",
        "  lowers ECE but does not create ranking skill, and the Brier skill score",
        "  against the test prior can stay negative.",
        "- ECE is bin-dependent. The report keeps Brier and AUROC beside it.",
        "- WP2 intervals resample rows i.i.d.; the paired HaluEval items and the",
        "  external source groups are handled by the WP3 two-level bootstrap.",
        "- Isotonic rows are fixed reference fits, not selected on test ECE.",
        "- The frozen tuned-hp B6 rows are shown for context only; the rebuild",
        "  model is `fixed_hp`.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

    val, cal, qa = evaluation_frames()
    halo = load_halueval_features()
    halo_test = halo[halo["split"] == "test"].copy()
    logger.info(
        "splits: val=%d calibration=%d halueval_test=%d ragtruth_qa_test=%d",
        len(val), len(cal), len(halo_test), len(qa),
    )

    all_fits = {}
    scores = {}
    for source in MODEL_SOURCES:
        all_fits[source] = {}
        scores[source] = {}
        for seed in SEEDS:
            logger.info("fitting calibrators: %s seed %s", source, seed)
            fits = build_seed_fits(source, seed, val, cal, qa, halo_test)
            all_fits[source][seed] = fits
            scores[source][seed] = calibrator_scores(fits)

    table = build_calibration_table(all_fits, scores)
    slopes = build_slope_table(all_fits)
    prior = build_prior_table(cal, qa)
    budget, budget_summary = build_label_budget_table(all_fits)

    table.to_csv(CALIBRATION_DIR / "calibration.csv", index=False)
    slopes.to_csv(CALIBRATION_DIR / "calibration_slope.csv", index=False)
    prior.to_csv(CALIBRATION_DIR / "prior_only.csv", index=False)
    budget.to_csv(CALIBRATION_DIR / "label_budget.csv", index=False)

    report = build_report(table, slopes, prior, budget, budget_summary, all_fits)
    (CALIBRATION_DIR / "calibration_report.json").write_text(
        json.dumps(json_safe(report), indent=2), encoding="utf-8"
    )
    (CALIBRATION_DIR / "REPORT.md").write_text(
        build_markdown(table, slopes, prior, budget), encoding="utf-8"
    )

    print("\nCalibration table (seed-averaged, fixed_hp):")
    view = table[(table["model"] == "fixed_hp") & (table["seed"] == "pooled")]
    print(
        view[
            ["dataset", "calibrator", "ece", "brier", "brier_skill_score", "auroc", "f1"]
        ].round(4).to_string(index=False)
    )
    print("\nSlopes (mean across seeds):")
    print(
        slopes.groupby(["model", "fit_split"])[
            ["platt_coef", "platt_intercept", "cal_slope", "cal_intercept"]
        ].mean().round(4).to_string()
    )
    print("\nLabel budget (pooled ECE):")
    print(
        budget[budget["seed"] == "pooled"][["size", "ece_mean", "ece_sd", "brier_mean"]]
        .round(4)
        .to_string(index=False)
    )
    print(f"\nSaved WP2 outputs to {CALIBRATION_DIR}")


if __name__ == "__main__":
    main()
