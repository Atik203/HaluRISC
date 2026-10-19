"""Reconcile the calibration protocols for the 2026-10-20 review.

Builds one comparable table for the standard B2 XGBoost and EC-XGB on the
same 900-row RAGTruth QA test set:

  raw -> source Platt (HaluEval validation) -> target Platt/isotonic
  (RAGTruth QA calibration split).

Standard-model values are read from the frozen B4 metrics. EC-XGB values are
recomputed per seed from the frozen m3 models and the cached calibration
features, so the table has a consistent seed aggregation and a clear source
for 0.726 vs 0.7433 and 0.126 vs 0.1335/0.1316.

Run (repo root):
  python src/models/review_calibration_protocol.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.config import DATA_PROCESSED, MODELS_DIR, RESULTS_DIR
from src.models.run_b6_modified import feature_matrix
from src.models.train_pipeline import ece

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("review_calibration_protocol")

FEATURES = DATA_PROCESSED / "features_full.parquet"
CLAIMS = DATA_PROCESSED / "claim_features.parquet"
CAL_FEATURES = DATA_PROCESSED / "calibration_features.parquet"
B6_PRED = RESULTS_DIR / "b6" / "b6_predictions.parquet"
B4_TARGET = RESULTS_DIR / "b4" / "b4_target_calibration.json"
B6_CAL = RESULTS_DIR / "b6" / "b6_calibration.json"
NAMES = RESULTS_DIR / "b6" / "b6_feature_names.json"
MODELS = MODELS_DIR / "b6"
OUT_DIR = RESULTS_DIR / "review"

SEEDS = [42, 123, 456]
SOURCE = "source_ragtruth"
METHODS = ["raw", "source_platt", "source_isotonic", "target_platt", "target_isotonic"]


def nll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = np.nan_to_num(p, nan=0.5, posinf=1 - 1e-12, neginf=1e-12)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.asarray(p, dtype=float)
    p = np.nan_to_num(p, nan=0.5, posinf=1 - 1e-12, neginf=1e-12)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    pred = (p >= 0.5).astype(int)
    return {
        "ece": float(ece(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "nll": nll(y, p),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "predicted_positive_rate": float(pred.mean()),
    }


def build_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    names = json.loads(NAMES.read_text(encoding="utf-8"))["m3"]
    claim_cols = [
        "claim_max_contradicts", "claim_mean_contradicts", "claim_contradicted_ratio",
        "claim_supported_ratio", "claim_unsupported_ratio", "claim_min_entails",
        "claim_mean_entails", "n_claims",
    ]

    features = pd.read_parquet(FEATURES)
    claims = pd.read_parquet(CLAIMS)
    val = features[features["split"] == "val"].merge(
        claims, on="sample_id", how="left", validate="one_to_one"
    )
    for col in claim_cols:
        val[col] = val[col].fillna(0.0)
    val[SOURCE] = 0.0

    cal = pd.read_parquet(CAL_FEATURES)
    cal[SOURCE] = 1.0

    missing_val = [c for c in names if c not in val.columns]
    missing_cal = [c for c in names if c not in cal.columns]
    if missing_val or missing_cal:
        raise KeyError(f"missing val={missing_val} cal={missing_cal}")

    return val, cal


def main() -> None:
    names = json.loads(NAMES.read_text(encoding="utf-8"))["m3"]
    val, cal = build_frames()
    X_val = feature_matrix(val, names, log_lengths=True)
    X_cal = feature_matrix(cal, names, log_lengths=True)
    y_val = val["label"].to_numpy()
    y_cal = cal["label"].to_numpy()

    preds = pd.read_parquet(B6_PRED)
    preds = preds[(preds["variant"] == "m3") & (preds["dataset"] == "ragtruth_qa_test")]

    per_seed = {}
    for seed in SEEDS:
        model = joblib.load(MODELS / f"xgboost_m3_seed_{seed}.joblib")
        raw_val = model.predict_proba(X_val)[:, 1]
        raw_cal = model.predict_proba(X_cal)[:, 1]

        test = preds[preds["seed"] == seed]
        if len(test) != 900:
            raise ValueError(f"seed {seed}: expected 900 test rows, got {len(test)}")
        y_test = test["label"].to_numpy()
        raw_test = test["score"].to_numpy()

        platt_src = LogisticRegression(max_iter=2000).fit(raw_val.reshape(-1, 1), y_val)
        iso_src = IsotonicRegression(out_of_bounds="clip").fit(raw_val, y_val)
        platt_tgt = LogisticRegression(max_iter=2000).fit(raw_cal.reshape(-1, 1), y_cal)
        iso_tgt = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, y_cal)

        scores = {
            "raw": raw_test,
            "source_platt": platt_src.predict_proba(raw_test.reshape(-1, 1))[:, 1],
            "source_isotonic": np.clip(iso_src.predict(raw_test), 0.0, 1.0),
            "target_platt": platt_tgt.predict_proba(raw_test.reshape(-1, 1))[:, 1],
            "target_isotonic": np.clip(iso_tgt.predict(raw_test), 0.0, 1.0),
        }
        per_seed[str(seed)] = {m: metrics(y_test, p) for m, p in scores.items()}
        logger.info(
            "seed %s raw ECE %.4f | source Platt %.4f | target Platt %.4f | target iso %.4f",
            seed,
            per_seed[str(seed)]["raw"]["ece"],
            per_seed[str(seed)]["source_platt"]["ece"],
            per_seed[str(seed)]["target_platt"]["ece"],
            per_seed[str(seed)]["target_isotonic"]["ece"],
        )

    mean = {m: {} for m in METHODS}
    std = {m: {} for m in METHODS}
    for m in METHODS:
        for k in per_seed["42"][m]:
            vals = [per_seed[s][m][k] for s in per_seed]
            mean[m][k] = float(np.mean(vals))
            std[m][k] = float(np.std(vals, ddof=0))

    b4 = json.loads(B4_TARGET.read_text(encoding="utf-8"))["methods"]

    def b4_row(d: dict) -> dict:
        return {
            "ece": d["ece_mean"],
            "brier": d["brier_mean"],
            "nll": d["nll_mean"],
            "auroc": d["auroc_mean"],
            "f1": d["f1_mean"],
            "precision": float("nan"),
            "recall": float("nan"),
            "predicted_positive_rate": d["predicted_positive_rate_mean"],
        }

    standard = {
        "raw": b4_row(b4["raw_reference"]),
        "source_platt": b4_row(b4["source_platt_reference"]),
        "source_isotonic": b4_row(b4["source_isotonic_reference"]),
        "target_platt": b4_row(b4["platt"]),
        "target_isotonic": b4_row(b4["isotonic"]),
    }

    bundle = {
        "meta": {
            "source": "review_calibration_protocol.py",
            "n_val": int(len(val)),
            "n_calibration": int(len(cal)),
            "n_test": int(len(preds[preds["seed"] == 42])),
            "seeds": SEEDS,
            "ec_xgb_model": "xgboost_m3_seed_{seed}",
            "standard_model": "B2 tuned XGBoost (B4 metrics)",
            "source_calibration_split": "HaluEval validation",
            "target_calibration_split": "RAGTruth QA calibration (5,034 rows)",
            "test_split": "RAGTruth QA test (900 rows, disjoint groups)",
            "positive_prevalence_test": float(
                preds[preds["seed"] == 42]["label"].mean()
            ),
            "threshold": 0.5,
        },
        "standard": standard,
        "ec_xgb": {"per_seed": per_seed, "mean": mean, "std": std},
        "deployed_display": json.loads(B6_CAL.read_text(encoding="utf-8")),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "calibration_protocol.json").write_text(
        json.dumps(bundle, indent=2), encoding="utf-8"
    )

    rows = []
    for model_name, block in (("standard", standard), ("ec_xgb", mean)):
        for m in METHODS:
            if m not in block:
                continue
            r = {"model": model_name, "method": m}
            r.update({k: round(v, 6) for k, v in block[m].items() if isinstance(v, (int, float))})
            rows.append(r)
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "calibration_protocol.csv", index=False)

    cols = ["model", "method", "ece", "brier", "nll", "auroc", "f1", "predicted_positive_rate"]
    print(table[cols].to_string(index=False))
    print("\nSaved:", OUT_DIR / "calibration_protocol.json")


if __name__ == "__main__":
    main()
