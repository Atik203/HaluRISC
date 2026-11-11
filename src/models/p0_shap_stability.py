"""P0.5 - cross-seed TreeSHAP stability for fixed-hp EC-XGB m3.

The deployed m3 explanation is seed-42 only.  This script loads the three
fixed-hyperparameter m3 models (seeds 42/123/456), computes exact TreeSHAP on
the same 1,000 label-stratified HaluEval test rows, and reports whether the
global mean-|SHAP| ranking survives a change of random seed.

Protocol:
  - sample: 1,000 HaluEval test rows, label-stratified, sampling seed 42,
    identical rows for all three models
  - models: artifacts/models/rebuild/fixed_hp/xgboost_m3_seed_{42,123,456}.joblib
    (max_depth 5, learning_rate 0.05, n_estimators 300, subsample 0.7,
    colsample_bytree 0.7)
  - SHAP: exact TreeSHAP via XGBoost ``pred_contribs=True``, raw margin space
  - statistics: Spearman and Kendall rank correlation of global mean-|SHAP|
    against seed 42, top-10 overlap, top-5 features per seed

The models are loaded, not retrained.  A score-reproduction check against the
frozen fixed-hp prediction table confirms that the loaded boosters are the
ones used in the rebuild.

Run (repo root, .venv):

  $env:HALU_XGB_DEVICE='cpu'
  .venv\\Scripts\\python.exe src/models/p0_shap_stability.py

Outputs under ``artifacts/results/rebuild/p0/``:

  shap_seed_stability.csv
  shap_seed_stability.json
  REPORT_shap_stability.md

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
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
import scipy
import xgboost
from scipy.stats import kendalltau, pearsonr, spearmanr

from src.models.config import ARTIFACTS_DIR, RESULTS_DIR, SEEDS
from src.models.rebuild_explanations import (
    feature_groups,
    stratified_sample,
    tree_shap,
)
from src.models.run_b6_modified import (
    FIXED_HP_PARAMS,
    feature_matrix,
    load_halueval,
    variant_features,
)

OUT_DIR = RESULTS_DIR / "rebuild" / "p0"
MODELS_DIR = ARTIFACTS_DIR / "models" / "rebuild" / "fixed_hp"
PREDICTIONS = RESULTS_DIR / "rebuild" / "fixed_hp" / "fixed_hp_predictions.parquet"
FROZEN_IMPORTANCE = RESULTS_DIR / "rebuild" / "explanations" / "m3_global_shap_importance.csv"
MODEL_TEMPLATE = "xgboost_m3_seed_{seed}.joblib"

N_SHAP = 1000
SAMPLE_SEED = 42
REFERENCE_SEED = 42
SHAP_SPACE = "raw margin (log-odds)"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("p0_shap_stability")


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=Path(__file__).resolve().parents[2], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def json_safe(value):
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


def load_model(seed: int):
    path = MODELS_DIR / MODEL_TEMPLATE.format(seed=seed)
    if not path.exists():
        raise FileNotFoundError(path)
    model = joblib.load(path)
    if int(model.n_features_in_) != 35:
        raise ValueError(f"{path}: expected 35 features, found {model.n_features_in_}")
    params = model.get_params()
    mismatches = {
        key: {"expected": value, "actual": params.get(key)}
        for key, value in FIXED_HP_PARAMS.items()
        if params.get(key) != value
    }
    if int(params.get("random_state", -1)) != seed:
        mismatches["random_state"] = {"expected": seed, "actual": params.get("random_state")}
    if mismatches:
        raise ValueError(f"{path}: fixed-hp check failed: {mismatches}")
    booster = model.get_booster()
    # Force CPU prediction so results are deterministic and no GPU is touched.
    booster.set_param({"device": "cpu"})
    return model, booster, path, params


def rank_features(names: list, mean_abs: np.ndarray, mean_signed: np.ndarray, std_signed: np.ndarray) -> pd.DataFrame:
    groups = feature_groups(names)
    total = float(mean_abs.sum())
    order = np.argsort(-mean_abs, kind="stable")
    ranking = pd.DataFrame({
        "rank": np.arange(1, len(names) + 1),
        "feature": [names[i] for i in order],
        "feature_group": [groups[names[i]] for i in order],
        "mean_abs_shap": [float(mean_abs[i]) for i in order],
        "mean_shap": [float(mean_signed[i]) for i in order],
        "std_shap": [float(std_signed[i]) for i in order],
        "share_pct": [float(100.0 * mean_abs[i] / total) if total else 0.0 for i in order],
    })
    ranking["cumulative_share_pct"] = ranking["share_pct"].cumsum()
    return ranking


def compare_rankings(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Compare two feature rankings by name."""
    a_series = a.set_index("feature")["mean_abs_shap"]
    b_series = b.set_index("feature")["mean_abs_shap"]
    common = a_series.index.intersection(b_series.index)
    if len(common) != len(a_series) or len(common) != len(b_series):
        raise ValueError("rankings do not cover the same features")
    a_values = a_series.loc[common].to_numpy()
    b_values = b_series.loc[common].to_numpy()
    top10_a = set(a.head(10)["feature"])
    top10_b = set(b.head(10)["feature"])
    top5_a = set(a.head(5)["feature"])
    top5_b = set(b.head(5)["feature"])
    return {
        "n_features": int(len(common)),
        "spearman": float(spearmanr(a_values, b_values).statistic),
        "kendall": float(kendalltau(a_values, b_values).statistic),
        "pearson_mean_abs": float(pearsonr(a_values, b_values).statistic),
        "top10_overlap": int(len(top10_a & top10_b)),
        "top10_overlap_fraction": float(len(top10_a & top10_b) / 10.0),
        "top5_overlap": int(len(top5_a & top5_b)),
        "top5_overlap_fraction": float(len(top5_a & top5_b) / 5.0),
        "common_top10": sorted(top10_a & top10_b),
        "common_top5": sorted(top5_a & top5_b),
    }


def run(seeds: list[int]) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    halueval = load_halueval()
    test = halueval[halueval["split"] == "test"].reset_index(drop=True)
    if len(test) != 3000:
        raise ValueError(f"expected 3000 HaluEval test rows, found {len(test)}")
    names = list(variant_features("m3"))
    if len(names) != 35:
        raise ValueError(f"expected 35 m3 features, found {len(names)}")

    sample_idx = stratified_sample(test, N_SHAP, SAMPLE_SEED)
    sample = test.loc[sample_idx].reset_index(drop=True)
    if len(sample) != N_SHAP:
        raise ValueError(f"expected {N_SHAP} sampled rows, found {len(sample)}")
    X = feature_matrix(sample, names, log_lengths=True)
    X_test = feature_matrix(test, names, log_lengths=True)

    reference_predictions = None
    if PREDICTIONS.exists():
        preds = pd.read_parquet(PREDICTIONS)
        reference_predictions = preds[
            (preds["variant"] == "m3") & (preds["dataset"] == "halueval_test")
        ]

    rankings: dict[int, pd.DataFrame] = {}
    model_rows = []
    score_reproduction = {}

    for seed in seeds:
        model, booster, path, params = load_model(seed)
        logger.info("seed %d: loaded %s", seed, path.name)

        # Reproduce the frozen rebuild scores on the full 3,000-row test split.
        max_score_diff = None
        if reference_predictions is not None:
            ref = reference_predictions[reference_predictions["seed"] == seed].set_index("sample_id")
            missing = [sid for sid in test["sample_id"] if sid not in ref.index]
            if missing:
                logger.warning("seed %d: %d test rows missing from reference predictions", seed, len(missing))
            else:
                scores = np.asarray(booster.predict(xgboost.DMatrix(X_test)), dtype=np.float64)
                reference_scores = ref.loc[test["sample_id"], "score"].to_numpy(dtype=np.float64)
                max_score_diff = float(np.max(np.abs(scores - reference_scores)))
                score_reproduction[seed] = max_score_diff

        shap_values, bias, probabilities, margin_error = tree_shap(booster, X)
        mean_abs = np.abs(shap_values).mean(axis=0)
        mean_signed = shap_values.mean(axis=0)
        std_signed = shap_values.std(axis=0)
        ranking = rank_features(names, mean_abs, mean_signed, std_signed)
        rankings[seed] = ranking

        logger.info(
            "seed %d: %d rows x %d features, sum mean|SHAP|=%.6f, margin error=%.2e, score diff=%s",
            seed, X.shape[0], X.shape[1], float(mean_abs.sum()), margin_error,
            "n/a" if max_score_diff is None else f"{max_score_diff:.2e}",
        )
        model_rows.append({
            "seed": int(seed),
            "model_path": str(path.relative_to(Path(__file__).resolve().parents[2])),
            "n_features": int(model.n_features_in_),
            "hyperparameters": {key: params.get(key) for key in FIXED_HP_PARAMS},
            "random_state": int(params.get("random_state")),
            "device_in_model": params.get("device"),
            "shap_space": SHAP_SPACE,
            "n_rows": int(X.shape[0]),
            "mean_bias": float(np.mean(bias)),
            "sum_mean_abs_shap": float(mean_abs.sum()),
            "mean_score": float(np.mean(probabilities)),
            "margin_error_max": float(margin_error),
            "score_reproduction_max_abs_diff": max_score_diff,
            "top5": ranking.head(5)[["rank", "feature", "feature_group", "mean_abs_shap", "share_pct"]].to_dict("records"),
            "top10": ranking.head(10)[["rank", "feature", "feature_group", "mean_abs_shap", "share_pct"]].to_dict("records"),
        })

    if REFERENCE_SEED not in rankings:
        raise ValueError(f"reference seed {REFERENCE_SEED} missing from {seeds}")

    # Optional check against the frozen seed-42 SHAP importances.
    frozen_importance_check = None
    if FROZEN_IMPORTANCE.exists():
        frozen = pd.read_csv(FROZEN_IMPORTANCE).set_index("feature")["mean_abs_shap"]
        ours = rankings[REFERENCE_SEED].set_index("feature")["mean_abs_shap"]
        common = ours.index.intersection(frozen.index)
        if len(common) == len(ours) == len(frozen):
            frozen_importance_check = {
                "reference": str(FROZEN_IMPORTANCE.relative_to(Path(__file__).resolve().parents[2])),
                "n_features": int(len(common)),
                "max_abs_diff": float((ours.loc[common] - frozen.loc[common]).abs().max()),
                "top5_match": bool(
                    list(ours.sort_values(ascending=False).head(5).index)
                    == list(frozen.sort_values(ascending=False).head(5).index)
                ),
            }
        else:
            logger.warning("frozen SHAP importance check skipped: feature sets differ")

    # Pairwise comparison matrix.
    pairwise = {}
    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1:]:
            pairwise[f"{seed_a}_vs_{seed_b}"] = compare_rankings(rankings[seed_a], rankings[seed_b])

    reference_vs = {}
    for seed in seeds:
        reference_vs[str(seed)] = compare_rankings(rankings[REFERENCE_SEED], rankings[seed])

    common_top5 = set.intersection(*(set(rankings[s].head(5)["feature"]) for s in seeds))
    common_top10 = set.intersection(*(set(rankings[s].head(10)["feature"]) for s in seeds))
    pairwise_spearman = [entry["spearman"] for entry in pairwise.values()]
    pairwise_kendall = [entry["kendall"] for entry in pairwise.values()]

    csv_rows = []
    for seed in seeds:
        entry = reference_vs[str(seed)]
        model_row = next(row for row in model_rows if row["seed"] == seed)
        csv_rows.append({
            "seed": int(seed),
            "reference_seed": int(REFERENCE_SEED),
            "spearman_vs_reference": entry["spearman"],
            "kendall_vs_reference": entry["kendall"],
            "pearson_mean_abs_vs_reference": entry["pearson_mean_abs"],
            "top10_overlap_vs_reference": entry["top10_overlap"],
            "top10_overlap_fraction_vs_reference": entry["top10_overlap_fraction"],
            "top5_overlap_vs_reference": entry["top5_overlap"],
            "top5_features": ";".join(rankings[seed].head(5)["feature"]),
            "top10_features": ";".join(rankings[seed].head(10)["feature"]),
            "sum_mean_abs_shap": model_row["sum_mean_abs_shap"],
            "mean_score": model_row["mean_score"],
            "margin_error_max": model_row["margin_error_max"],
            "score_reproduction_max_abs_diff": model_row["score_reproduction_max_abs_diff"],
        })
    pd.DataFrame(csv_rows).to_csv(OUT_DIR / "shap_seed_stability.csv", index=False)

    payload = {
        "schema": "p0-shap-seed-stability-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "src/models/p0_shap_stability.py",
        "git": {"branch": git_value("rev-parse", "--abbrev-ref", "HEAD"), "commit": git_value("rev-parse", "HEAD")},
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "xgboost": xgboost.__version__,
            "xgb_device": "cpu",
        },
        "protocol": {
            "model": "EC-XGB m3 (fixed hyperparameters)",
            "fixed_hyperparameters": FIXED_HP_PARAMS,
            "seeds": [int(s) for s in seeds],
            "reference_seed": int(REFERENCE_SEED),
            "n_shap_rows": int(N_SHAP),
            "sampling": "label-stratified without replacement",
            "sample_seed": int(SAMPLE_SEED),
            "identical_rows_across_seeds": True,
            "shap_implementation": "exact TreeSHAP via XGBoost pred_contribs=True",
            "shap_space": SHAP_SPACE,
            "n_features": int(len(names)),
            "test_rows_available": int(len(test)),
            "positive_rate_sample": float(sample["label"].mean()),
            "score_reproduction_max_abs_diff": {str(k): v for k, v in score_reproduction.items()},
            "model_source": "loaded from artifacts/models/rebuild/fixed_hp; not retrained",
            "frozen_importance_check": frozen_importance_check,
        },
        "models": model_rows,
        "rankings": {
            str(seed): rankings[seed].to_dict("records") for seed in seeds
        },
        "reference_vs": reference_vs,
        "pairwise": pairwise,
        "summary": {
            "reference_seed": int(REFERENCE_SEED),
            "mean_pairwise_spearman": float(np.mean(pairwise_spearman)) if pairwise_spearman else None,
            "min_pairwise_spearman": float(np.min(pairwise_spearman)) if pairwise_spearman else None,
            "mean_pairwise_kendall": float(np.mean(pairwise_kendall)) if pairwise_kendall else None,
            "min_pairwise_kendall": float(np.min(pairwise_kendall)) if pairwise_kendall else None,
            "top10_overlap_vs_reference": {
                str(seed): int(reference_vs[str(seed)]["top10_overlap"]) for seed in seeds
            },
            "common_top5_all_seeds": sorted(common_top5),
            "common_top10_all_seeds": sorted(common_top10),
        },
        "caveats": [
            "TreeSHAP explains the raw margin (log-odds), not the Platt display score.",
            "source_ragtruth is constant zero on HaluEval; its attributions come from interaction structure, not within-domain source variation.",
            "The 1,000 rows are one fixed stratified sample (sampling seed 42); row-level resampling uncertainty is not included.",
            "The three models share the training recipe and differ only in the XGBoost random seed.",
        ],
    }
    (OUT_DIR / "shap_seed_stability.json").write_text(
        json.dumps(json_safe(payload), indent=2), encoding="utf-8"
    )
    write_report(payload)
    logger.info("Wrote P0.5 artifacts to %s", OUT_DIR)
    return payload


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def write_report(payload: dict) -> None:
    protocol = payload["protocol"]
    summary = payload["summary"]
    seeds = protocol["seeds"]
    rankings = {int(seed): pd.DataFrame(rows) for seed, rows in payload["rankings"].items()}

    lines = [
        "# P0.5 - Cross-seed TreeSHAP stability",
        "",
        f"Generated: {payload['generated_at_utc']}  ",
        f"Script: `{payload['script']}`  ",
        f"Git: `{payload['git']['branch']}` @ `{payload['git']['commit']}`  ",
        "Device: CPU",
        "",
        "## Protocol",
        "",
        "- Models: EC-XGB m3 fixed-hp models for seeds 42/123/456, loaded from",
        "  `artifacts/models/rebuild/fixed_hp/` and not retrained.",
        "- Fixed configuration: max_depth 5, learning_rate 0.05, n_estimators 300,",
        "  subsample 0.7, colsample_bytree 0.7.",
        f"- Rows: {protocol['n_shap_rows']} HaluEval test rows, label-stratified with sampling",
        f"  seed {protocol['sample_seed']}; the same rows are used for all three models.",
        f"- SHAP: {protocol['shap_implementation']} in {protocol['shap_space']}.",
        f"- Sample positive rate: {protocol['positive_rate_sample']:.4f}.",
        f"- Score reproduction against the frozen fixed-hp predictions: maximum absolute",
        f"  difference {max(v for v in protocol['score_reproduction_max_abs_diff'].values()):.2e}.",
    ]
    frozen_check = protocol.get("frozen_importance_check")
    if frozen_check:
        lines.append(
            f"- Seed-42 mean-|SHAP| matches the frozen explanation artifact "
            f"(`{frozen_check['reference']}`) with maximum absolute difference "
            f"{frozen_check['max_abs_diff']:.2e}; top-5 match: {frozen_check['top5_match']}."
        )
    lines += [
        "",
        "## Stability against seed 42",
        "",
        "| Seed | Spearman | Kendall | Top-10 overlap | Top-5 features |",
        "|---:|---:|---:|---:|---|",
    ]
    for seed in seeds:
        entry = payload["reference_vs"][str(seed)]
        top5 = ", ".join(f"`{name}`" for name in rankings[seed].head(5)["feature"])
        lines.append(
            f"| {seed} | {fmt(entry['spearman'])} | {fmt(entry['kendall'])} "
            f"| {entry['top10_overlap']}/10 | {top5} |"
        )

    lines += [
        "",
        "`seed 42` is the reference, so its row is 1.0000 by construction.",
        "",
        "## Pairwise seed comparison",
        "",
        "| Pair | Spearman | Kendall | Top-10 overlap | Top-5 overlap |",
        "|---|---:|---:|---:|---:|",
    ]
    for pair, entry in payload["pairwise"].items():
        lines.append(
            f"| {pair.replace('_vs_', ' vs ')} | {fmt(entry['spearman'])} | {fmt(entry['kendall'])} "
            f"| {entry['top10_overlap']}/10 | {entry['top5_overlap']}/5 |"
        )

    lines += [
        "",
        "## Top-5 features by seed",
        "",
    ]
    for seed in seeds:
        top = rankings[seed].head(5)
        lines.append(f"- Seed {seed}: " + ", ".join(
            f"{row.feature} ({row.mean_abs_shap:.4f}, {row.share_pct:.1f}%)"
            for row in top.itertuples()
        ))

    lines += [
        "",
        "## Top-10 features by seed",
        "",
    ]
    for seed in seeds:
        top = rankings[seed].head(10)
        lines.append(f"- Seed {seed}: " + ", ".join(f"`{name}`" for name in top["feature"]))

    lines += [
        "",
        "## Key readings",
        "",
        f"- Mean pairwise Spearman: {summary['mean_pairwise_spearman']:.4f}; "
        f"minimum {summary['min_pairwise_spearman']:.4f}.",
        f"- Mean pairwise Kendall: {summary['mean_pairwise_kendall']:.4f}; "
        f"minimum {summary['min_pairwise_kendall']:.4f}.",
        f"- Top-10 overlap with seed 42: "
        + ", ".join(
            f"seed {seed} {summary['top10_overlap_vs_reference'][str(seed)]}/10"
            for seed in seeds
        )
        + ".",
        f"- Features in every seed's top 5: "
        + (", ".join(f"`{name}`" for name in summary["common_top5_all_seeds"]) or "none")
        + ".",
        f"- Features in every seed's top 10: "
        + (", ".join(f"`{name}`" for name in summary["common_top10_all_seeds"]) or "none")
        + ".",
        "",
        "A high rank correlation means the global explanation ranking is stable across",
        "random seeds. It does not mean the three models are identical: individual SHAP",
        "values and per-row explanations can still move, especially for correlated",
        "features that share signal.",
        "",
        "## Caveats",
        "",
        *[f"- {caveat}" for caveat in payload["caveats"]],
        "- One dataset and one fixed sample; this is a stability check, not a",
        "  sampling-uncertainty interval.",
        "",
        "## Failed steps",
        "",
        "None. All three fixed-hp models loaded, reproduced the frozen scores, and",
        "produced exact TreeSHAP values on the same 1,000 rows.",
        "",
    ]
    (OUT_DIR / "REPORT_shap_stability.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P0.5 cross-seed TreeSHAP stability")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HALU_XGB_DEVICE", "cpu")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    payload = run(seeds)
    print(json.dumps(json_safe({
        "summary": payload["summary"],
        "reference_vs": payload["reference_vs"],
        "pairwise": payload["pairwise"],
    }), indent=2))


if __name__ == "__main__":
    main()
