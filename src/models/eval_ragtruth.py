"""
Zero-shot external validation of the HaluRISC model on RAGTruth QA (blueprint §8.4,
roadmap Phase 5 "External comparison").

Runs the final calibrated XGBoost model with NO training on RAGTruth data:
  features are extracted with the same pipeline, then predict.

Run (repo root, .venv):
  python src/models/eval_ragtruth.py
"""

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("eval_ragtruth")

ROOT = Path(__file__).resolve().parents[2]
RAGTRUTH_PATH = ROOT / "data" / "raw" / "ragtruth" / "ragtruth_qa.parquet"
UNIFIED_PATH = ROOT / "data" / "processed" / "unified_records.parquet"
MODELS_DIR = ROOT / "artifacts" / "models"
RESULTS_DIR = ROOT / "artifacts" / "results"

N_SAMPLES = 2000


def load_ragtruth_frame(n: int = N_SAMPLES) -> pd.DataFrame:
    """RAGTruth QA rows for zero-shot validation.

    Priority: (1) legacy Version A parquet if present, (2) the B1 unified
    dataset built by cell 7d (data/processed/unified_records.parquet), which
    is written to the legacy path on first use so repeat runs skip the merge.
    """
    if RAGTRUTH_PATH.exists():
        return pd.read_parquet(RAGTRUTH_PATH).head(n)
    if not UNIFIED_PATH.exists():
        raise FileNotFoundError(
            f"RAGTruth data missing: run cell 7d (B1 unified build) or place "
            f"{RAGTRUTH_PATH}"
        )
    unified = pd.read_parquet(
        UNIFIED_PATH, columns=["source_dataset", "task", "question", "context", "answer", "label"]
    )
    qa = unified[(unified["source_dataset"] == "ragtruth") & (unified["task"] == "qa")]
    qa = qa[["question", "context", "answer", "label"]].reset_index(drop=True)
    RAGTRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    qa.to_parquet(RAGTRUTH_PATH, index=False)
    logger.info(f"Built {RAGTRUTH_PATH} from the unified dataset ({len(qa)} QA rows)")
    return qa.head(n)


def ece(y_true, y_prob, n_bins: int = 10) -> float:
    from src.models.train_pipeline import ece as ece_fn

    return ece_fn(y_true, y_prob, n_bins)


def main():
    sys.path.insert(0, str(ROOT))
    from src.features.extract_features import extract_all_features_single, load_heavy_models

    df = load_ragtruth_frame(N_SAMPLES)
    logger.info(f"RAGTruth QA holdout: {len(df)} samples (label balance: {df['label'].value_counts().to_dict()})")

    bundle = joblib.load(MODELS_DIR / "model_xgboost_calibrated.joblib")
    if isinstance(bundle, dict) and bundle.get("kind") == "xgb+platt":
        raw, platt = bundle["model"], bundle["calibrator"]

        def predict_proba(X):
            p = raw.predict_proba(X)[:, 1]
            return platt.predict_proba(p.reshape(-1, 1))

    else:
        predict_proba = bundle.predict_proba
    feature_cols = json.loads((MODELS_DIR / "feature_names.json").read_text())
    models = load_heavy_models()

    logger.info("Extracting features on RAGTruth (zero-shot)...")
    rows = []
    for _, r in df.iterrows():
        rows.append(extract_all_features_single(r["question"], r["context"], r["answer"], models))
    X = pd.DataFrame(rows)[feature_cols].values
    y = df["label"].values

    y_prob = predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    results = {
        "n_samples": int(len(y)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, y_prob)),
        "pr_auc": float(average_precision_score(y, y_prob)),
        "mcc": float(matthews_corrcoef(y, y_pred)),
        "ece": float(ece(y, y_prob)),
        "brier": float(brier_score_loss(y, y_prob)),
        "label_distribution": df["label"].value_counts().to_dict(),
    }
    logger.info(f"RAGTruth zero-shot: {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in results.items()}, indent=2)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_DIR / "ragtruth_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {RESULTS_DIR / 'ragtruth_results.json'}")


if __name__ == "__main__":
    main()
