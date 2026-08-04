"""
HaluRISC full experiment protocol (blueprint A9-A10, roadmap Phases 4-5).

Mandatory rules implemented here:
  - 5-fold stratified CV + randomized search tuning for XGBoost (30 iters)
  - Baselines: heuristic (1 - overlap, threshold tuned on val), LR (scaled), RF
  - Every experiment repeated with seeds 42, 123, 456 -> mean +/- std
  - Calibration: Platt (sigmoid) fit on VALIDATION only; isotonic compared on TEST
  - Metrics: P/R/F1/AUROC/PR-AUC/MCC + ECE (10 bins) + Brier
  - Statistics: McNemar (XGBoost vs best baseline), bootstrap 95% CIs (1000),
    Wilcoxon signed-rank across seeds
  - Ablations: remove each of the 7 feature groups one at a time (3 seeds)
  - Artifacts: model_xgb_calibrated.joblib, scaler, params.json, result tables

Run (repo root, .venv):
  python src/models/train_pipeline.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.contingency_tables import mcnemar
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_pipeline")

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "processed" / "features_full.parquet"
FEATURES_FALLBACK = ROOT / "data" / "processed" / "features_core.parquet"
SPLIT_JSON = ROOT / "artifacts" / "split_indices.json"
MODELS_DIR = ROOT / "artifacts" / "models"
RESULTS_DIR = ROOT / "artifacts" / "results"
FIGURES_DIR = ROOT / "artifacts" / "figures"

SEEDS = [42, 123, 456]
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 777

FEATURE_GROUPS: Dict[str, List[str]] = {
    "length": ["n_chars", "n_words", "n_sentences", "avg_word_len"],
    "lexical": ["overlap_answer_context", "overlap_answer_question", "jaccard_ans_ctx", "jaccard_ans_q"],
    "entity": ["n_entities_answer", "n_entities_context", "entity_overlap_ratio", "novel_entity_ratio"],
    "nli": [
        "nli_ctx_entails_ans", "nli_ctx_contradicts_ans", "nli_ctx_neutral_ans",
        "nli_ans_entails_ctx", "nli_ans_contradicts_ctx", "nli_ans_neutral_ctx",
    ],
    "numeric": ["n_numbers_answer", "n_numbers_context", "number_overlap_ratio", "novel_numbers"],
    "hedging": ["hedge_count", "hedge_density"],
    "semantic": ["cosine_ctx_ans", "cosine_q_ans"],
}

TUNING_GRID = {
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "n_estimators": [100, 200, 300, 500],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
}


def xgb_device() -> str:
    """cuda if available else cpu (XGBoost 3.4 device parameter)."""
    try:
        import xgboost as xgb

        if xgb.__version__.split(".")[0] >= "3":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
    except ImportError:
        pass
    return "cpu"


def classification_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def ece(y_true, y_prob, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idxs = np.clip(np.searchsorted(bins, y_prob, side="right") - 1, 0, n_bins - 1)
    total = len(y_true)
    ece_val = 0.0
    for b in range(n_bins):
        mask = idxs == b
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece_val += (mask.sum() / total) * abs(acc - conf)
    return float(ece_val)


def bootstrap_ci(y_true, y_pred, y_prob, n: int = N_BOOTSTRAP) -> dict:
    """Bootstrap 95% CIs for F1 and AUROC."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    m = len(y_true)
    f1s, aucs = [], []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        if len(np.unique(y_true[idx])) < 2:
            continue
        f1s.append(f1_score(y_true[idx], y_pred[idx], zero_division=0))
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return {
        "f1_ci": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
        "auroc_ci": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
    }


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    path = FEATURES_PATH if FEATURES_PATH.exists() else FEATURES_FALLBACK
    logger.info(f"Loading features from {path.name}")
    df = pd.read_parquet(path)
    if "split" not in df.columns:
        raise ValueError("feature matrix has no 'split' column; run src/data/prepare.py first")

    feature_cols = []
    for group, cols in FEATURE_GROUPS.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            logger.warning(f"Group '{group}' missing columns {missing} -> skipped")
            continue
        feature_cols.extend(cols)
    if not feature_cols:
        raise ValueError("no known feature columns found")

    meta_cols = ["sample_id", "item_idx", "label", "split"]
    df = df.dropna(subset=["label"]).reset_index(drop=True)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    logger.info(
        f"Split sizes - train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)} | features: {len(feature_cols)}"
    )
    return train_df, val_df, test_df, feature_cols


def heuristic_baseline(val_df: pd.DataFrame, test_df: pd.DataFrame, col: str = "overlap_answer_context"):
    """Rule: risk = 1 - overlap_answer_context; threshold tuned on validation."""
    thresholds = np.linspace(0, 1, 101)
    best_thresh, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(val_df["label"], (val_df[col] < t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    test_probs = 1.0 - test_df[col]
    test_preds = (test_df[col] < best_thresh).astype(int)
    return test_preds, test_probs, {"threshold": float(best_thresh), "val_f1": float(best_f1)}


def make_xgb(params: dict, seed: int, scale_pos_weight: float) -> XGBClassifier:
    base = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
        device=xgb_device(),
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
    )
    base.update(params)
    return XGBClassifier(**base)


def tune_xgboost(X_train, y_train, scale_pos_weight: float, seed: int = 42) -> dict:
    logger.info("Tuning XGBoost via RandomizedSearchCV (5-fold, 30 iters)...")
    t0 = time.time()
    xgb = make_xgb({}, seed, scale_pos_weight)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rs = RandomizedSearchCV(
        xgb, TUNING_GRID, n_iter=30, cv=cv, scoring="roc_auc", n_jobs=1, random_state=seed, verbose=0
    )
    rs.fit(X_train, y_train)
    best = rs.best_params_
    logger.info(f"Best params ({time.time() - t0:.0f}s): {best}  cv_auc={rs.best_score_:.4f}")
    return best, float(rs.best_score_)


def train_seed_models(
    X_train, y_train, X_val, y_val, X_test, params: dict, scale_pos_weight: float
) -> List[dict]:
    """Train XGBoost for each seed with early stopping on validation."""
    results = []
    for seed in SEEDS:
        xgb = make_xgb(params, seed, scale_pos_weight)
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_prob = xgb.predict_proba(X_test)[:, 1]
        results.append({"seed": seed, "model": xgb, "y_prob": y_prob})
    return results


def mean_std_table(rows: List[dict], metric_keys: List[str]) -> dict:
    vals = {k: [r[k] for r in rows] for k in metric_keys}
    out = {}
    for k, v in vals.items():
        out[f"{k}_mean"] = float(np.mean(v))
        out[f"{k}_std"] = float(np.std(v))
        out[f"{k}_all"] = [float(x) for x in v]
    return out


class CalibratedXGBoost:
    """Deployable artifact (blueprint A18): raw XGBoost + Platt calibrator, sklearn-compatible."""

    def __init__(self, model, calibrator):
        self.model = model
        self.calibrator = calibrator

    def predict_proba(self, X):
        p = self.model.predict_proba(X)[:, 1]
        return self.calibrator.predict_proba(p.reshape(-1, 1))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    train_df, val_df, test_df, feature_cols = load_data()
    X_train, y_train = train_df[feature_cols].values, train_df["label"].values
    X_val, y_val = val_df[feature_cols].values, val_df["label"].values
    X_test, y_test = test_df[feature_cols].values, test_df["label"].values

    pos_ratio = float(y_train.sum() / max(1, (len(y_train) - y_train.sum())))
    logger.info(f"Positive ratio (train): {y_train.mean():.4f} -> scale_pos_weight={pos_ratio:.3f}")

    # ---- 1. Heuristic baseline ----
    h_pred, h_prob, h_info = heuristic_baseline(val_df, test_df)
    h_metrics = classification_metrics(y_test, h_pred, h_prob)
    logger.info(f"Heuristic baseline on test: f1={h_metrics['f1']:.4f} (thresh={h_info['threshold']:.2f})")

    # ---- 2. Tuning ----
    best_params, best_cv_auc = tune_xgboost(pd.DataFrame(X_train, columns=feature_cols), y_train, pos_ratio)

    # ---- 3. Baselines (LR, RF) with 3 seeds ----
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    baseline_rows = {"lr": [], "rf": []}
    for seed in SEEDS:
        lr = LogisticRegression(max_iter=2000, random_state=seed)
        lr.fit(X_train_s, y_train)
        p = lr.predict_proba(X_test_s)[:, 1]
        baseline_rows["lr"].append(classification_metrics(y_test, (p >= 0.5).astype(int), p))

        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed)
        rf.fit(X_train, y_train)
        p = rf.predict_proba(X_test)[:, 1]
        baseline_rows["rf"].append(classification_metrics(y_test, (p >= 0.5).astype(int), p))

    # ---- 4. XGBoost per seed + calibration on val ----
    xgb_models = train_seed_models(X_train, y_train, X_val, y_val, X_test, best_params, pos_ratio)
    xgb_rows = []
    for r in xgb_models:
        m = classification_metrics(y_test, (r["y_prob"] >= 0.5).astype(int), r["y_prob"])
        m["seed"] = r["seed"]
        xgb_rows.append(m)

    # ---- 5. Calibration (Platt on val, compare isotonic on test) ----
    # sklearn >= 1.9 dropped CalibratedClassifierCV(cv="prefit"); manual Platt
    # (logistic regression on raw scores) and isotonic are equivalent and version-proof.
    calibrators = {}
    calibration_results = {"platt": {}, "isotonic": {}}
    for method in ["sigmoid", "isotonic"]:
        label = "platt" if method == "sigmoid" else "isotonic"
        row_metrics, row_ece, row_brier = [], [], []
        for r in xgb_models:
            p_val = r["model"].predict_proba(X_val)[:, 1]
            p_test = r["model"].predict_proba(X_test)[:, 1]
            if method == "sigmoid":
                lr = LogisticRegression(max_iter=2000)
                lr.fit(p_val.reshape(-1, 1), y_val)
                p_cal = lr.predict_proba(p_test.reshape(-1, 1))[:, 1]
                if r["seed"] == SEEDS[0]:
                    calibrators[r["seed"]] = lr
            else:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(p_val, y_val)
                p_cal = iso.predict(p_test)
            row_metrics.append(classification_metrics(y_test, (p_cal >= 0.5).astype(int), p_cal))
            row_ece.append(ece(y_test, p_cal))
            row_brier.append(brier_score_loss(y_test, p_cal))
        calibration_results[label] = {
            "f1_mean": float(np.mean([m["f1"] for m in row_metrics])),
            "ece_mean": float(np.mean(row_ece)),
            "brier_mean": float(np.mean(row_brier)),
            "ece_all": [float(x) for x in row_ece],
            "brier_all": [float(x) for x in row_brier],
        }
        logger.info(f"{label} calibration: f1={calibration_results[label]['f1_mean']:.4f} "
                    f"ece={calibration_results[label]['ece_mean']:.4f} brier={calibration_results[label]['brier_mean']:.4f}")

    # ---- 6. Statistics ----
    # McNemar: XGBoost (seed 42) vs best baseline (RF seed 42)
    rf_42 = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf_42.fit(X_train, y_train)
    rf_pred_42 = rf_42.predict(X_test)
    xgb_pred_42 = (xgb_models[0]["y_prob"] >= 0.5).astype(int)
    table = np.array([
        [(xgb_pred_42 == 1).sum(), ((xgb_pred_42 == 1) & (rf_pred_42 == 0)).sum()],
        [((xgb_pred_42 == 0) & (rf_pred_42 == 1)).sum(), (xgb_pred_42 == 0).sum()],
    ])
    mcn = mcnemar([[table[1, 1], table[1, 0]], [table[0, 1], table[0, 0]]], exact=False, correction=True)
    stats_tests = {"mcnemar_p_value": float(mcn.pvalue), "mcnemar_statistic": float(mcn.statistic)}

    boot = bootstrap_ci(y_test, (xgb_models[0]["y_prob"] >= 0.5).astype(int), xgb_models[0]["y_prob"])
    stats_tests["bootstrap_f1_ci"] = boot["f1_ci"]
    stats_tests["bootstrap_auroc_ci"] = boot["auroc_ci"]

    # Wilcoxon across seeds: XGB F1 vs RF F1 (3 paired values)
    xgb_f1 = [m["f1"] for m in xgb_rows]
    rf_f1 = [m["f1"] for m in baseline_rows["rf"]]
    if np.std(xgb_f1 - np.array(rf_f1)) > 0 or xgb_f1 != rf_f1:
        try:
            w = stats.wilcoxon(xgb_f1, rf_f1)
            stats_tests["wilcoxon_xgb_vs_rf_p"] = float(w.pvalue)
        except ValueError:
            stats_tests["wilcoxon_xgb_vs_rf_p"] = None
    else:
        stats_tests["wilcoxon_xgb_vs_rf_p"] = None
    logger.info(f"Statistics: {json.dumps(stats_tests, indent=2)}")

    # ---- 7. Ablations (7 groups x 3 seeds) ----
    ablation_rows = []
    for group in FEATURE_GROUPS:
        kept = [c for c in feature_cols if c not in FEATURE_GROUPS[group]]
        if len(kept) == len(feature_cols):
            continue
        Xa_train, Xa_test = train_df[kept].values, test_df[kept].values
        f1s, aucs = [], []
        for seed in SEEDS:
            m = make_xgb(best_params, seed, pos_ratio)
            m.fit(Xa_train, y_train, eval_set=[(val_df[kept].values, y_val)], verbose=False)
            p = m.predict_proba(Xa_test)[:, 1]
            f1s.append(f1_score(y_test, (p >= 0.5).astype(int), zero_division=0))
            aucs.append(roc_auc_score(y_test, p))
        ablation_rows.append({
            "removed_group": group,
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
            "auroc_mean": float(np.mean(aucs)), "auroc_std": float(np.std(aucs)),
        })
        logger.info(f"Ablation -{group}: f1={np.mean(f1s):.4f} auroc={np.mean(aucs):.4f}")

    # ---- 8. Save artifacts ----
    final_seed = 42
    final_model = xgb_models[SEEDS.index(final_seed)]["model"]
    final_cal = calibrators[final_seed]

    joblib.dump(final_model, MODELS_DIR / "model_xgboost_raw.joblib")
    joblib.dump(
        {"kind": "xgb+platt", "model": final_model, "calibrator": calibrators[final_seed]},
        MODELS_DIR / "model_xgboost_calibrated.joblib",
    )
    joblib.dump(calibrators[final_seed], MODELS_DIR / "calibrator_platt.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    with open(MODELS_DIR / "params.json", "w") as f:
        json.dump({
            "best_params": best_params, "best_cv_auc": best_cv_auc,
            "seeds": SEEDS, "scale_pos_weight": pos_ratio,
            "feature_groups": FEATURE_GROUPS, "feature_cols": feature_cols,
            "n_features": len(feature_cols), "model_version": "xgboost-v1.0",
            "device": xgb_device(), "n_train": int(len(X_train)), "n_val": int(len(X_val)), "n_test": int(len(X_test)),
        }, f, indent=2)
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    logger.info(f"Saved model artifacts to {MODELS_DIR}")

    # ---- 9. Results tables ----
    def summarize(rows, name):
        return {"model": name, **{k: float(np.mean([r[k] for r in rows])) for k in
                                   ["precision", "recall", "f1", "auroc", "pr_auc", "mcc"]},
                **{f"{k}_std": float(np.std([r[k] for r in rows])) for k in
                   ["precision", "recall", "f1", "auroc", "pr_auc", "mcc"]}}

    results = {
        "heuristic": {**h_metrics, **h_info},
        "logistic_regression": summarize(baseline_rows["lr"], "Logistic Regression"),
        "random_forest": summarize(baseline_rows["rf"], "Random Forest"),
        "xgboost": summarize(xgb_rows, "XGBoost"),
        "calibration": calibration_results,
        "statistics": stats_tests,
        "ablation": ablation_rows,
        "bootstrap": boot,
    }

    with open(RESULTS_DIR / "final_results.json", "w") as f:
        json.dump(results, f, indent=2)

    summary_df = pd.DataFrame([results["heuristic"], results["logistic_regression"],
                               results["random_forest"], results["xgboost"]]).set_index("model")
    summary_df.to_csv(RESULTS_DIR / "model_comparison.csv")
    pd.DataFrame(ablation_rows).to_csv(RESULTS_DIR / "ablation_results.csv", index=False)

    print("\n" + "=" * 90)
    print(" HaluRISC Final Model Comparison (test set, mean over seeds 42/123/456)")
    print("=" * 90)
    print(summary_df[["precision", "recall", "f1", "auroc", "pr_auc", "mcc"]].round(4).to_string())
    print("=" * 90)
    logger.info(f"Saved final results to {RESULTS_DIR}")


if __name__ == "__main__":
    import argparse
    import joblib

    sys.path.insert(0, str(ROOT))
    main()
