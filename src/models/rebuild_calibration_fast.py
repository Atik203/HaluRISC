"""Fast WP2 calibration table (fallback).

Computes the same headline calibration rows as `rebuild_calibration.py` without
the 1,000-draw coefficient bootstrap. Use only if the full run is unavailable.
Outputs go to `artifacts/results/rebuild/calibration_fast/` so the full WP2
directory is never overwritten.

Run:
  python src/models/rebuild_calibration_fast.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.config import MODELS_DIR, RESULTS_DIR
from src.models.rebuild_common import calibration_coefficients, clean_scores, ece, fit_platt
from src.models.run_b6_modified import feature_matrix

SEEDS = [42, 123, 456]
OUT = RESULTS_DIR / "rebuild" / "calibration_fast"
FIXED = RESULTS_DIR / "rebuild" / "fixed_hp"
MODELS = MODELS_DIR / "rebuild" / "fixed_hp"
SIZES = [50, 100, 252, 500, 1000, 5034]
DRAWS = 100


def load_frames():
    features = pd.read_parquet("data/processed/features_full.parquet")
    claims = pd.read_parquet("data/processed/claim_features.parquet")
    val = features[features["split"] == "val"].merge(claims, on="sample_id", how="left")
    val = val.fillna(0.0)
    val["source_ragtruth"] = 0.0
    cal = pd.read_parquet("data/processed/calibration_features.parquet").copy()
    cal["source_ragtruth"] = 1.0
    names = json.loads((FIXED / "b6_feature_names.json").read_text(encoding="utf-8"))["m3"]
    pred = pd.read_parquet(FIXED / "fixed_hp_predictions.parquet")
    return val, cal, names, pred


def metrics(y, p, threshold=0.5):
    p = clean_scores(p)
    y = np.asarray(y, dtype=int)
    pred = (p >= threshold).astype(int)
    return {
        "ece": float(ece(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "ppr": float(pred.mean()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    val, cal, names, pred = load_frames()
    X_val = feature_matrix(val, names, True)
    X_cal = feature_matrix(cal, names, True)
    y_val = val["label"].to_numpy()
    y_cal = cal["label"].to_numpy()

    rows, slopes, per_seed_scores = [], [], {}
    for seed in SEEDS:
        model = joblib.load(MODELS / f"xgboost_m3_seed_{seed}.joblib")
        raw_val = model.predict_proba(X_val)[:, 1]
        raw_cal = model.predict_proba(X_cal)[:, 1]
        test = pred[(pred["dataset"] == "ragtruth_qa_test") & (pred["seed"] == seed) & (pred["variant"] == "m3")]
        y_test = test["label"].to_numpy()
        raw_test = test["score"].to_numpy()

        platt_src = fit_platt(raw_val, y_val)
        platt_tgt = fit_platt(raw_cal, y_cal)
        scores = {
            "raw": raw_test,
            "source_platt": platt_src.predict_proba(raw_test.reshape(-1, 1))[:, 1],
            "target_platt": platt_tgt.predict_proba(raw_test.reshape(-1, 1))[:, 1],
        }
        for method, p in scores.items():
            row = {"model": "fixed_hp", "seed": seed, "dataset": "ragtruth_qa_test", "method": method}
            row.update(metrics(y_test, p))
            rows.append(row)

        coef = calibration_coefficients(raw_cal, y_cal)
        slopes.append({"seed": seed, **coef})
        per_seed_scores[seed] = {"cal": raw_cal, "test": raw_test, "y_test": y_test, "y_cal": y_cal}

    frame = pd.DataFrame(rows)
    mean_rows = []
    for method in ("raw", "source_platt", "target_platt"):
        sub = frame[frame["method"] == method]
        mean_rows.append({"model": "fixed_hp", "seed": "mean", "dataset": "ragtruth_qa_test", "method": method,
                          **{c: float(sub[c].mean()) for c in ("ece", "brier", "auroc", "f1", "precision", "recall", "ppr")}})
    frame = pd.concat([frame, pd.DataFrame(mean_rows)], ignore_index=True)
    frame.to_csv(OUT / "calibration.csv", index=False)

    slope_frame = pd.DataFrame(slopes)
    slope_frame.to_csv(OUT / "slope.csv", index=False)
    print("slope mean:", slope_frame[["cal_slope", "cal_intercept"]].mean().round(4).to_dict())

    # Prior only
    prior = []
    for name, constant, split, y in (
        ("test_prior", 0.1778, "ragtruth_qa_test", per_seed_scores[42]["y_test"]),
        ("cal_prior", 0.3107, "ragtruth_qa_calibration", per_seed_scores[42]["y_cal"]),
    ):
        p = np.full(len(y), constant)
        prior.append({"predictor": name, "constant": constant, "eval_split": split, "n": len(y),
                      "prevalence": float(np.mean(y)), "ece": float(ece(y, p)),
                      "brier": float(brier_score_loss(y, p))})
    prior_frame = pd.DataFrame(prior)
    prior_frame.to_csv(OUT / "prior.csv", index=False)

    # Label budget, seed 42, 100 draws
    budget = []
    y_cal = per_seed_scores[42]["y_cal"]
    cal_scores = per_seed_scores[42]["cal"]
    y_test = per_seed_scores[42]["y_test"]
    test_scores = per_seed_scores[42]["test"]
    rng = np.random.default_rng(20261111)
    for size in SIZES:
        eces, briers = [], []
        for _ in range(DRAWS):
            idx = rng.choice(len(y_cal), size=size, replace=False)
            if len(np.unique(y_cal[idx])) < 2:
                continue
            cal_fit = fit_platt(cal_scores[idx], y_cal[idx])
            p = cal_fit.predict_proba(test_scores.reshape(-1, 1))[:, 1]
            eces.append(float(ece(y_test, p)))
            briers.append(float(brier_score_loss(y_test, p)))
        budget.append({"size": size, "draws": len(eces), "ece_mean": float(np.mean(eces)),
                       "ece_sd": float(np.std(eces, ddof=1)), "ece_q025": float(np.percentile(eces, 2.5)),
                       "ece_q975": float(np.percentile(eces, 97.5)), "brier_mean": float(np.mean(briers)),
                       "brier_sd": float(np.std(briers, ddof=1))})
    budget_frame = pd.DataFrame(budget)
    budget_frame.to_csv(OUT / "label_budget.csv", index=False)
    print(budget_frame.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
