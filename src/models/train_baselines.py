"""
Baseline modeling script for HaluRISC.
Trains and evaluates Heuristic Rule, Logistic Regression, Random Forest, and XGBoost models
on extracted features, printing a performance comparison table.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, matthews_corrcoef
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FEATURES_PATH = os.path.join("data", "processed", "features_core.parquet")
RESULTS_DIR = os.path.join("artifacts", "results")
MODELS_DIR = os.path.join("artifacts", "models")

def evaluate_predictions(y_true, y_pred, y_prob) -> dict:
    """Calculates evaluation metrics for binary classification."""
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "mcc": float(matthews_corrcoef(y_true, y_pred))
    }

def run_heuristic_baseline(val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Rule-based heuristic: high overlap_answer_context -> low risk (0), low overlap -> high risk (1)."""
    # Threshold tuned on val set
    thresholds = np.linspace(0, 1, 101)
    best_thresh = 0.5
    best_val_f1 = 0.0

    for t in thresholds:
        val_preds = (val_df["overlap_answer_context"] < t).astype(int)
        f1 = f1_score(val_df["label"], val_preds, zero_division=0)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_thresh = t

    logging.info(f"Heuristic best overlap threshold on Val: {best_thresh:.2f} (F1: {best_val_f1:.4f})")

    # Predict on test
    test_probs = 1.0 - test_df["overlap_answer_context"]
    test_preds = (test_df["overlap_answer_context"] < best_thresh).astype(int)

    metrics = evaluate_predictions(test_df["label"], test_preds, test_probs)
    return metrics, best_thresh

def train_and_eval_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"{FEATURES_PATH} not found. Run src/features/extract_features.py first.")

    df = pd.read_parquet(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in ["sample_id", "item_idx", "label", "split"]]

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    logging.info(f"Train samples: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logging.info(f"Feature list ({len(feature_cols)}): {feature_cols}")

    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_val, y_val = val_df[feature_cols], val_df["label"]
    X_test, y_test = test_df[feature_cols], test_df["label"]

    results = {}

    # 1. Heuristic Baseline
    heur_metrics, heur_thresh = run_heuristic_baseline(val_df, test_df)
    results["Heuristic (Overlap)"] = heur_metrics

    # 2. Logistic Regression (Scaled)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    lr_preds = (lr_probs >= 0.5).astype(int)
    results["Logistic Regression"] = evaluate_predictions(y_test, lr_preds, lr_probs)

    # 3. Random Forest
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_preds = (rf_probs >= 0.5).astype(int)
    results["Random Forest"] = evaluate_predictions(y_test, rf_preds, rf_probs)

    # 4. XGBoost
    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05, 
        eval_metric="logloss", random_state=42, n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    xgb_preds = (xgb_probs >= 0.5).astype(int)
    results["XGBoost (Default)"] = evaluate_predictions(y_test, xgb_preds, xgb_probs)

    # Convert results to DataFrame and display
    results_df = pd.DataFrame(results).T
    results_df = results_df[["precision", "recall", "f1", "auroc", "pr_auc", "mcc"]]
    
    print("\n" + "="*80)
    print(" HaluRISC Baseline Model Comparison on Test Set (Core Features)")
    print("="*80)
    print(results_df.to_string())
    print("="*80 + "\n")

    # Save results to JSON and CSV
    results_df.to_csv(os.path.join(RESULTS_DIR, "baseline_results.csv"))
    with open(os.path.join(RESULTS_DIR, "baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logging.info(f"Saved baseline evaluation results to {RESULTS_DIR}")

if __name__ == "__main__":
    train_and_eval_all()
