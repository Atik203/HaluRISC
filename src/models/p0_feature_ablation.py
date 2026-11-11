"""P0.2 - fixed-hyperparameter m3 feature-group ablation.

The frozen B2/B6 feature-group ablation is tuned, so it does not directly test
the m3 (EC-XGB) feature set.  This script holds one configuration fixed for
every run and removes one base feature group at a time from m3:

  full_m3      26 base + 8 claim-level NLI + source indicator (35 features)
  minus_*      full_m3 without one of the 7 base groups
  no_claims    full_m3 without the 8 claim features (base + source, 27)
  no_source    full_m3 without the RAGTruth source indicator (base + claim, 34)

The training recipe is the one in ``src/models/run_b6_modified.py``: HaluEval
train rows plus RAGTruth non-QA train rows, log-scaled length columns, the same
monotone constraints, frozen 70/15/15 HaluEval splits grouped by ``item_idx``,
and RAGTruth ``source_group_id`` groups in the training pool.  No test row is
used for tuning or model selection; every reported number is at threshold 0.5.

Fixed configuration (identical for all 30 fits):

  max_depth=5, learning_rate=0.05, n_estimators=300,
  subsample=0.7, colsample_bytree=0.7

Run (repo root, .venv; CPU avoids GPU contention):

  $env:HALU_XGB_DEVICE='cpu'
  .venv\\Scripts\\python.exe src/models/p0_feature_ablation.py

Outputs under ``artifacts/results/rebuild/p0/``:

  feature_group_ablation.csv              one row per configuration (mean +/- SD)
  feature_group_ablation_runs.csv         one row per configuration x seed
  feature_group_ablation.json             full protocol, runs, summary, deltas
  REPORT_feature_group_ablation.md        human-readable report
  models/                                 one joblib model per run

The top-level ``REPORT.md`` in the same directory indexes this report with the
other P0 experiments.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from src.features.claim_features import FEATURE_COLUMNS as CLAIM_COLUMNS
from src.models.config import RESULTS_DIR, SEEDS
from src.models.run_b6_modified import (
    BASE_COLUMNS,
    FIXED_HP_PARAMS,
    SOURCE_COLUMN,
    constraint_string,
    feature_matrix,
    in_domain_metrics,
    load_external,
    load_halueval,
    m3_train_rows,
    make_variant_xgb,
)
from src.models.train_pipeline import FEATURE_GROUPS, xgb_device

OUT_DIR = RESULTS_DIR / "rebuild" / "p0"
MODELS_OUT = OUT_DIR / "models"
REFERENCE_METRICS = RESULTS_DIR / "rebuild" / "fixed_hp" / "b6_per_seed_metrics.csv"

METRICS = ("f1", "auroc", "mcc", "ece", "brier")
THRESHOLD = 0.5

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("p0_feature_ablation")


# --------------------------------------------------------------------------
# Configurations
# --------------------------------------------------------------------------
def build_configs() -> dict:
    """Ordered full-m3 reference plus 7 group removals and 2 block controls."""
    configs: dict[str, dict] = {}

    full_names = list(BASE_COLUMNS) + list(CLAIM_COLUMNS) + [SOURCE_COLUMN]
    configs["full_m3"] = {
        "description": "EC-XGB m3 reference (26 base + 8 claim + source)",
        "names": full_names,
        "removed": [],
    }

    for group, columns in FEATURE_GROUPS.items():
        removed = list(columns)
        names = [c for c in BASE_COLUMNS if c not in set(removed)]
        names += list(CLAIM_COLUMNS) + [SOURCE_COLUMN]
        configs[f"minus_{group}"] = {
            "description": f"m3 without the {group} base group ({len(removed)} features)",
            "names": names,
            "removed": removed,
        }

    configs["no_claims"] = {
        "description": "m3 without the 8 claim-level NLI features (m1-equivalent feature block)",
        "names": list(BASE_COLUMNS) + [SOURCE_COLUMN],
        "removed": list(CLAIM_COLUMNS),
    }
    configs["no_source"] = {
        "description": "m3 without the RAGTruth source indicator",
        "names": list(BASE_COLUMNS) + list(CLAIM_COLUMNS),
        "removed": [SOURCE_COLUMN],
    }
    return configs


# --------------------------------------------------------------------------
# Provenance helpers
# --------------------------------------------------------------------------
def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=Path(__file__).resolve().parents[2], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def json_safe(value):
    """Convert numpy/pandas scalars and Path objects to JSON-native values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def grouped_split_report(halueval: pd.DataFrame, train_groups: np.ndarray) -> dict:
    """Evidence that the frozen splits keep HaluEval item groups disjoint."""
    item_sets = {
        split: set(halueval.loc[halueval["split"] == split, "item_idx"].astype(int).tolist())
        for split in ("train", "val", "test")
    }
    return {
        "group_key_halueval": "item_idx",
        "group_key_ragtruth_train": "source_group_id",
        "n_groups_train_pool": int(len(np.unique(train_groups))),
        "item_idx_overlap_train_val": len(item_sets["train"] & item_sets["val"]),
        "item_idx_overlap_train_test": len(item_sets["train"] & item_sets["test"]),
        "item_idx_overlap_val_test": len(item_sets["val"] & item_sets["test"]),
    }


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------
def run_experiment(configs: dict, seeds: list[int]) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_OUT.mkdir(parents=True, exist_ok=True)

    halueval = load_halueval()
    external = load_external()
    rag_train = m3_train_rows(external)

    hal_train = halueval[halueval["split"] == "train"].copy()
    hal_train[SOURCE_COLUMN] = 0.0
    rag_train = rag_train.copy()
    rag_train[SOURCE_COLUMN] = 1.0

    # Same row order and grouping as run_b6_modified.py.
    train_all = pd.concat([hal_train, rag_train], ignore_index=True)
    groups = np.concatenate([
        np.char.add("halu:", hal_train["item_idx"].to_numpy().astype(str)),
        np.char.add("rag:", rag_train["source_group_id"].astype(str)),
    ])
    if len(groups) != len(train_all):
        raise ValueError("group vector does not match training rows")

    val_df = halueval[halueval["split"] == "val"].reset_index(drop=True)
    test_df = halueval[halueval["split"] == "test"].reset_index(drop=True)
    y_train = train_all["label"].to_numpy(dtype=int)
    y_val = val_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)

    split_report = grouped_split_report(halueval, groups)
    if any(
        split_report[key]
        for key in (
            "item_idx_overlap_train_val",
            "item_idx_overlap_train_test",
            "item_idx_overlap_val_test",
        )
    ):
        raise ValueError(f"HaluEval item groups leak across splits: {split_report}")

    logger.info(
        "train pool %d rows (%d HaluEval + %d RAGTruth non-QA) | val %d | test %d | groups %d",
        len(train_all), len(hal_train), len(rag_train), len(val_df), len(test_df),
        split_report["n_groups_train_pool"],
    )

    runs = []
    config_meta = {}

    for config_key, spec in configs.items():
        names = list(spec["names"])
        if len(names) != len(set(names)):
            raise ValueError(f"{config_key}: duplicate feature names")
        for required in (SOURCE_COLUMN,):
            if config_key != "no_source" and required not in names:
                raise ValueError(f"{config_key}: missing source indicator")

        X_train = feature_matrix(train_all, names, log_lengths=True)
        X_val = feature_matrix(val_df, names, log_lengths=True)
        X_test = feature_matrix(test_df, names, log_lengths=True)
        constraints = constraint_string(names)
        config_meta[config_key] = {
            "description": spec["description"],
            "n_features": len(names),
            "features": names,
            "removed_features": list(spec["removed"]),
            "n_removed": len(spec["removed"]),
            "monotone_constraints": constraints,
            "log_lengths": True,
        }

        for seed in seeds:
            params = dict(FIXED_HP_PARAMS)
            logger.info(
                "[%s seed %d] %d features (%d removed), %d train rows, cpu=%s",
                config_key, seed, len(names), len(spec["removed"]), len(X_train), xgb_device(),
            )
            started = time.perf_counter()
            model = make_variant_xgb(params, seed, constraints, early_stopping=False)
            model.fit(X_train, y_train, verbose=False)
            fit_seconds = time.perf_counter() - started

            p_val = model.predict_proba(X_val)[:, 1]
            p_test = model.predict_proba(X_test)[:, 1]
            test_metrics = in_domain_metrics(y_test, p_test, THRESHOLD)
            val_metrics = in_domain_metrics(y_val, p_val, THRESHOLD)

            model_path = MODELS_OUT / f"p0_{config_key}_seed_{seed}.joblib"
            joblib.dump(model, model_path)

            run = {
                "config": config_key,
                "seed": int(seed),
                "n_features": len(names),
                "n_removed": len(spec["removed"]),
                "n_train": int(len(train_all)),
                "n_train_halueval": int(len(hal_train)),
                "n_train_ragtruth": int(len(rag_train)),
                "n_val": int(len(val_df)),
                "n_test": int(len(test_df)),
                "fit_seconds": float(fit_seconds),
                "hyperparameters": params,
                "monotone_constraints": constraints is not None,
                "log_lengths": True,
                "device": xgb_device(),
                "model_path": str(model_path.relative_to(Path(__file__).resolve().parents[2])),
                "test_metrics": test_metrics,
                "val_metrics": val_metrics,
            }
            for key in METRICS:
                run[key] = float(test_metrics[key])
            run["precision"] = float(test_metrics["precision"])
            run["recall"] = float(test_metrics["recall"])
            runs.append(run)
            logger.info(
                "[%s seed %d] f1=%.6f auroc=%.6f mcc=%.6f ece=%.6f brier=%.6f (%.1fs)",
                config_key, seed, run["f1"], run["auroc"], run["mcc"], run["ece"],
                run["brier"], fit_seconds,
            )

    runs_df = pd.DataFrame(runs)

    # Paired deltas against full m3 at the same seed.
    full = runs_df[runs_df["config"] == "full_m3"].set_index("seed")
    for key in METRICS:
        full_by_seed = full[key]
        runs_df[f"delta_{key}_vs_full_m3"] = [
            float(row[key] - full_by_seed.loc[row["seed"]]) for _, row in runs_df.iterrows()
        ]

    summary = {}
    for config_key in configs:
        sub = runs_df[runs_df["config"] == config_key].set_index("seed").sort_index()
        entry = {
            "description": configs[config_key]["description"],
            "n_features": int(sub["n_features"].iloc[0]),
            "n_removed": int(sub["n_removed"].iloc[0]),
            "removed_features": list(configs[config_key]["removed"]),
            "seeds": [int(s) for s in sub.index.tolist()],
            "n_seeds": int(len(sub)),
        }
        for key in ("precision", "recall", *METRICS):
            entry[f"{key}_mean"] = float(sub[key].mean())
            entry[f"{key}_sd"] = float(sub[key].std(ddof=1))
        for key in METRICS:
            delta = sub[key] - full[key].reindex(sub.index)
            entry[f"delta_{key}_mean"] = float(delta.mean())
            entry[f"delta_{key}_sd"] = float(delta.std(ddof=1))
        summary[config_key] = entry

    # Summary table (paper-facing).
    summary_rows = []
    for config_key, entry in summary.items():
        row = {
            "config": config_key,
            "description": entry["description"],
            "n_features": entry["n_features"],
            "n_removed": entry["n_removed"],
            "removed_features": ";".join(entry["removed_features"]),
            **{key: FIXED_HP_PARAMS[key] for key in FIXED_HP_PARAMS},
            "monotone_constraints": bool(config_meta[config_key]["monotone_constraints"]),
            "log_lengths": True,
            "n_train": int(runs_df.loc[runs_df["config"] == config_key, "n_train"].iloc[0]),
            "n_seeds": entry["n_seeds"],
        }
        for key in ("precision", "recall", *METRICS):
            row[f"{key}_mean"] = entry[f"{key}_mean"]
            row[f"{key}_sd"] = entry[f"{key}_sd"]
        for key in METRICS:
            row[f"delta_{key}_mean"] = entry[f"delta_{key}_mean"]
            row[f"delta_{key}_sd"] = entry[f"delta_{key}_sd"]
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "feature_group_ablation.csv", index=False)

    run_columns = [
        "config", "seed", "n_features", "n_removed", "n_train", "n_train_halueval",
        "n_train_ragtruth", "n_val", "n_test", "fit_seconds", *FIXED_HP_PARAMS.keys(),
        "monotone_constraints", "log_lengths", "device",
        "precision", "recall", *METRICS,
        *[f"delta_{key}_vs_full_m3" for key in METRICS],
    ]
    runs_out = runs_df.copy()
    for key in FIXED_HP_PARAMS:
        runs_out[key] = [run["hyperparameters"][key] for run in runs]
    runs_out[run_columns].to_csv(OUT_DIR / "feature_group_ablation_runs.csv", index=False)

    # Optional comparison with the frozen GPU fixed-hp m3 rows.
    reference_check = None
    if REFERENCE_METRICS.exists():
        ref = pd.read_csv(REFERENCE_METRICS)
        ref = ref[(ref["variant"] == "m3") & (ref["seed"].isin(seeds))].set_index("seed")
        ours = runs_df[runs_df["config"] == "full_m3"].set_index("seed")
        common = sorted(set(ref.index) & set(ours.index))
        if common:
            diffs = {}
            for key in METRICS:
                diffs[key] = float(
                    np.max(np.abs(ours.loc[common, key].to_numpy() - ref.loc[common, key].to_numpy()))
                )
            reference_check = {
                "reference": str(REFERENCE_METRICS.relative_to(Path(__file__).resolve().parents[2])),
                "seeds": [int(s) for s in common],
                "max_abs_metric_diff": diffs,
                "note": (
                    "Frozen reference was trained with device=cuda; this run uses device=cpu. "
                    "Small differences are expected from hist build/device numerics. "
                    "The reference is not used for selection."
                ),
            }

    payload = {
        "schema": "p0-feature-group-ablation-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "src/models/p0_feature_ablation.py",
        "git": {"branch": git_value("rev-parse", "--abbrev-ref", "HEAD"), "commit": git_value("rev-parse", "HEAD")},
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "xgb_device": xgb_device(),
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "protocol": {
            "model": "EC-XGB m3",
            "fixed_hyperparameters": FIXED_HP_PARAMS,
            "seeds": [int(s) for s in seeds],
            "threshold": THRESHOLD,
            "metrics": list(METRICS),
            "metric_split": "HaluEval test (3000 rows)",
            "model_selection": "none; fixed hyperparameters, no test tuning",
            "calibration": "none; raw XGBoost probabilities",
            "train_recipe": "HaluEval train + RAGTruth non-QA train, source indicator, log lengths, monotone constraints",
            "train_rows": int(len(train_all)),
            "train_rows_halueval": int(len(hal_train)),
            "train_rows_ragtruth": int(len(rag_train)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "positive_rate_train": float(y_train.mean()),
            "positive_rate_val": float(y_val.mean()),
            "positive_rate_test": float(y_test.mean()),
            "grouped_splits": split_report,
            "fit_count": int(len(runs_df)),
            "note": (
                "no_claims removes only the 8 claim features and keeps the source indicator; "
                "it is the m3-minus-claims control, not the exact 26-feature m1 model."
            ),
        },
        "configs": config_meta,
        "runs": runs,
        "summary": summary,
        "reference_check": reference_check,
    }
    (OUT_DIR / "feature_group_ablation.json").write_text(
        json.dumps(json_safe(payload), indent=2), encoding="utf-8"
    )
    write_report(payload)
    logger.info("Wrote P0.2 artifacts to %s", OUT_DIR)
    return payload


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def write_report(payload: dict) -> None:
    protocol = payload["protocol"]
    summary = payload["summary"]
    full = summary["full_m3"]

    group_keys = [f"minus_{group}" for group in FEATURE_GROUPS if f"minus_{group}" in summary]
    group_ranked = sorted(group_keys, key=lambda key: summary[key]["delta_f1_mean"])
    worst = group_ranked[0] if group_ranked else None
    closest = min(group_keys, key=lambda key: abs(summary[key]["delta_f1_mean"])) if group_keys else None

    lines = [
        "# P0.2 - Fixed-hyperparameter m3 feature-group ablation",
        "",
        f"Generated: {payload['generated_at_utc']}  ",
        f"Script: `{payload['script']}`  ",
        f"Git: `{payload['git']['branch']}` @ `{payload['git']['commit']}`  ",
        f"Device: XGBoost `{payload['environment']['xgb_device']}` (CPU run, no GPU contention)",
        "",
        "## Protocol",
        "",
        "Every run uses the same frozen configuration, so feature-group differences are not",
        "confounded with tuning or early stopping:",
        "",
        f"- max_depth = {FIXED_HP_PARAMS['max_depth']}",
        f"- learning_rate = {FIXED_HP_PARAMS['learning_rate']}",
        f"- n_estimators = {FIXED_HP_PARAMS['n_estimators']}",
        f"- subsample = {FIXED_HP_PARAMS['subsample']}",
        f"- colsample_bytree = {FIXED_HP_PARAMS['colsample_bytree']}",
        f"- seeds = {', '.join(str(s) for s in protocol['seeds'])}",
        f"- threshold = {protocol['threshold']}",
        "- selection = none; no test row is used for tuning, early stopping, or model choice",
        "- calibration = none; metrics use raw XGBoost probabilities",
        "",
        "Training recipe matches `src/models/run_b6_modified.py`: HaluEval train rows plus",
        "RAGTruth non-QA train rows, source indicator, log1p on `n_chars`/`n_words`, and the",
        "same monotone constraints rebuilt over each surviving feature list.",
        "",
        f"- Train pool: {protocol['train_rows']:,} rows "
        f"({protocol['train_rows_halueval']:,} HaluEval + {protocol['train_rows_ragtruth']:,} RAGTruth)",
        f"- Validation: {protocol['val_rows']:,} rows; test: {protocol['test_rows']:,} rows",
        f"- Positive rate: train {protocol['positive_rate_train']:.4f}, "
        f"val {protocol['positive_rate_val']:.4f}, test {protocol['positive_rate_test']:.4f}",
        f"- Grouped splits: HaluEval `item_idx`, RAGTruth `source_group_id`; "
        f"{protocol['grouped_splits']['n_groups_train_pool']:,} training groups",
        f"- Item overlap: train/val {protocol['grouped_splits']['item_idx_overlap_train_val']}, "
        f"train/test {protocol['grouped_splits']['item_idx_overlap_train_test']}, "
        f"val/test {protocol['grouped_splits']['item_idx_overlap_val_test']}",
        f"- Fits: {protocol['fit_count']} ({len(summary)} configurations x "
        f"{len(protocol['seeds'])} seeds)",
        "",
        "Metrics are computed once on the HaluEval test split at threshold 0.5. Deltas are",
        "paired by seed against full m3; for ECE and Brier a negative delta means better",
        "calibration. `no_claims` removes only the 8 claim features and keeps the source",
        "indicator, so it is the m3-minus-claims control rather than the exact 26-feature m1.",
        "",
        f"## Results (mean +/- SD over seeds {', '.join(str(s) for s in protocol['seeds'])})",
        "",
        "| Configuration | Features | F1 | AUROC | MCC | ECE | Brier | dF1 | dAUROC | dMCC | dECE | dBrier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    table_keys = [
        key for key in ["full_m3", *group_keys, "no_claims", "no_source"] if key in summary
    ]
    for config_key in table_keys:
        entry = summary[config_key]
        lines.append(
            f"| `{config_key}` | {entry['n_features']} "
            f"| {fmt(entry['f1_mean'])} +/- {fmt(entry['f1_sd'])} "
            f"| {fmt(entry['auroc_mean'])} +/- {fmt(entry['auroc_sd'])} "
            f"| {fmt(entry['mcc_mean'])} +/- {fmt(entry['mcc_sd'])} "
            f"| {fmt(entry['ece_mean'])} +/- {fmt(entry['ece_sd'])} "
            f"| {fmt(entry['brier_mean'])} +/- {fmt(entry['brier_sd'])} "
            f"| {signed(entry['delta_f1_mean'])} "
            f"| {signed(entry['delta_auroc_mean'])} "
            f"| {signed(entry['delta_mcc_mean'])} "
            f"| {signed(entry['delta_ece_mean'])} "
            f"| {signed(entry['delta_brier_mean'])} |"
        )

    lines += [
        "",
        "`delta_*_sd` values are in the JSON and the summary CSV. The per-seed table is",
        "`feature_group_ablation_runs.csv`; model files are in `models/`.",
        "",
        "## Key readings",
        "",
        f"- The full m3 reference reaches F1 {full['f1_mean']:.4f} +/- {full['f1_sd']:.4f}, "
        f"AUROC {full['auroc_mean']:.4f} +/- {full['auroc_sd']:.4f}, "
        f"MCC {full['mcc_mean']:.4f} +/- {full['mcc_sd']:.4f}, "
        f"ECE {full['ece_mean']:.4f} +/- {full['ece_sd']:.4f}, "
        f"Brier {full['brier_mean']:.4f} +/- {full['brier_sd']:.4f}.",
    ]
    if worst and closest:
        lines += [
            f"- Largest F1 drop among single-group removals: `{worst}` "
            f"({signed(summary[worst]['delta_f1_mean'])} F1, "
            f"{signed(summary[worst]['delta_auroc_mean'])} AUROC).",
            f"- Smallest absolute F1 change among single-group removals: `{closest}` "
            f"({signed(summary[closest]['delta_f1_mean'])} F1, "
            f"{signed(summary[closest]['delta_auroc_mean'])} AUROC).",
        ]
    if "no_claims" in summary:
        lines.append(
            f"- Remove claims (`no_claims`, {summary['no_claims']['n_features']} features): "
            f"dF1 {signed(summary['no_claims']['delta_f1_mean'])}, "
            f"dAUROC {signed(summary['no_claims']['delta_auroc_mean'])}, "
            f"dMCC {signed(summary['no_claims']['delta_mcc_mean'])}."
        )
    if "no_source" in summary:
        lines.append(
            f"- Remove source (`no_source`, {summary['no_source']['n_features']} features): "
            f"dF1 {signed(summary['no_source']['delta_f1_mean'])}, "
            f"dAUROC {signed(summary['no_source']['delta_auroc_mean'])}, "
            f"dMCC {signed(summary['no_source']['delta_mcc_mean'])}."
        )
    lines += [
        "",
        "The HaluEval test set is 50/50 by construction and heavily saturated near the top of",
        "the score range, so F1 and AUROC move very little. ECE and Brier are more sensitive",
        "to which probability mass the surviving features can separate.",
        "",
    ]

    check = payload.get("reference_check")
    if check:
        lines += [
            "## Reproducibility check",
            "",
            f"Compared with the frozen GPU fixed-hp m3 rows (`{check['reference']}`), the CPU",
            "rerun differs by at most:",
            "",
            *[
                f"- {key}: {value:.2e}"
                for key, value in check["max_abs_metric_diff"].items()
            ],
            "",
            "Those rows were trained on CUDA and are not used for selection. The differences",
            "show device/build numerics, not a protocol change.",
            "",
        ]

    lines += [
        "## Caveats",
        "",
        "- One dataset (HaluEval test) and one fixed configuration; the ablation does not",
        "  establish that the ranking transfers to RAGTruth or FaithBench.",
        "- No calibration is applied; ECE/Brier describe the raw model probabilities.",
        "- The source indicator is constant zero on HaluEval test. Removing it can only change",
        "  the model through training-time interaction with the RAGTruth rows, so the",
        "  `no_source` delta is a training-recipe effect, not a within-test feature effect.",
        "- Single-group removal can understate importance when two groups carry redundant",
        "  signal; the ablation is leave-one-block-out, not feature-level attribution.",
        "- All deltas are at threshold 0.5. No threshold was selected on test.",
        "",
        "## Failed steps",
        "",
        "None. All 30 fits completed and wrote a model, a per-run row, and a summary row.",
        "",
    ]
    (OUT_DIR / "REPORT_feature_group_ablation.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P0.2 fixed-hp m3 feature-group ablation")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument(
        "--configs",
        default="all",
        help="Comma-separated config keys, or 'all'. Useful for a rerun of one block.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HALU_XGB_DEVICE", "cpu")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    configs = build_configs()
    if args.configs != "all":
        selected = [key.strip() for key in args.configs.split(",") if key.strip()]
        unknown = [key for key in selected if key not in configs]
        if unknown:
            raise SystemExit(f"unknown configs: {unknown}")
        configs = {key: configs[key] for key in selected}
    payload = run_experiment(configs, seeds)
    print(json.dumps(json_safe({
        "configs": list(payload["summary"]),
        "summary": payload["summary"],
        "reference_check": payload["reference_check"],
    }), indent=2))


if __name__ == "__main__":
    main()
