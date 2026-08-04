"""
HaluRISC error analysis (blueprint A10: inspect 20 wrong predictions, 10 FP + 10 FN).

Steps:
  1. Predict the test split with the calibrated model.
  2. Sample 10 false positives + 10 false negatives (seeded).
  3. Auto-tag each case with the blueprint error taxonomy (heuristic rules).
  4. Save a reviewable case dump + a category-count table.

The taxonomy tags are a starting point for manual review — verify and adjust
the categories when writing the paper's error analysis section.

Run (repo root, .venv):
  python src/models/error_analysis.py
"""

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("error_analysis")

from src.models.config import FEATURES_FALLBACK, FEATURES_FULL, MODELS_DIR, QA_CLEAN, RESULTS_DIR, SAMPLE_SEED

N_FP = 10
N_FN = 10


def load_test_set():
    path = FEATURES_FULL if FEATURES_FULL.exists() else FEATURES_FALLBACK
    df = pd.read_parquet(path)
    clean = pd.read_parquet(QA_CLEAN)
    text_cols = [c for c in ["question", "context", "answer"] if c in clean.columns]
    if text_cols:
        df = pd.concat([df, clean[text_cols]], axis=1)
    feature_cols = json.loads((MODELS_DIR / "feature_names.json").read_text())
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values
    return test_df, X_test, y_test, feature_cols


def load_predictions(X_test, feature_cols):
    bundle = joblib.load(MODELS_DIR / "model_xgboost_calibrated.joblib")
    if isinstance(bundle, dict) and bundle.get("kind") == "xgb+platt":
        raw, platt = bundle["model"], bundle["calibrator"]

        def predict_proba(X):
            p = raw.predict_proba(X)[:, 1]
            return platt.predict_proba(p.reshape(-1, 1))[:, 1]

    else:
        predict_proba = bundle.predict_proba
    y_prob = predict_proba(X_test)
    return y_prob


def tag_case(row: pd.Series) -> str:
    """Heuristic taxonomy tagging (blueprint A10 categories) - review manually."""
    n_words = float(row.get("n_words", 0))
    overlap = float(row.get("overlap_answer_context", 0.0))
    nli_contra = float(row.get("nli_ctx_contradicts_ans", 0.0))
    nli_entail = float(row.get("nli_ctx_entails_ans", 0.0))
    n_ents = float(row.get("n_entities_answer", 0))
    ctx_len = len(str(row.get("context", "")))
    ans_len = len(str(row.get("answer", "")))

    if n_words <= 4:
        return "short_answer_ambiguity"
    if nli_entail > 0.8 and overlap > 0.6:
        return "label_ambiguity"
    if nli_contra > 0.5 and overlap < 0.3:
        return "label_ambiguity"
    if n_ents == 0:
        return "entity_extraction_failure"
    if ctx_len < 80:
        return "weak_context"
    if ans_len < 40 and n_words <= 8:
        return "short_answer_ambiguity"
    if overlap > 0.6:
        return "unsupported_but_semantically_similar"
    return "other"


def main():
    test_df, X_test, y_test, feature_cols = load_test_set()
    y_prob = load_predictions(X_test, feature_cols)
    y_pred = (y_prob >= 0.5).astype(int)

    fp_idx = np.where((y_pred == 1) & (y_test == 0))[0]
    fn_idx = np.where((y_pred == 0) & (y_test == 1))[0]
    logger.info(f"Misclassified: FP={len(fp_idx)}, FN={len(fn_idx)}")

    rng = np.random.default_rng(SAMPLE_SEED)
    fp_sample = rng.choice(fp_idx, size=min(N_FP, len(fp_idx)), replace=False)
    fn_sample = rng.choice(fn_idx, size=min(N_FN, len(fn_idx)), replace=False)

    rows = []
    for idx in np.concatenate([fp_sample, fn_sample]):
        row = test_df.iloc[idx]
        case = {
            "sample_id": str(row["sample_id"]),
            "error_type": "false_positive" if y_test[idx] == 0 else "false_negative",
            "true_label": int(y_test[idx]),
            "predicted_label": int(y_pred[idx]),
            "probability": round(float(y_prob[idx]), 4),
            "category": tag_case(row),
            "question": str(row.get("question", ""))[:300],
            "context": str(row.get("context", ""))[:400],
            "answer": str(row.get("answer", ""))[:300],
            "features": {
                "n_words": float(row.get("n_words", 0)),
                "overlap_answer_context": round(float(row.get("overlap_answer_context", 0.0)), 4),
                "nli_ctx_contradicts_ans": round(float(row.get("nli_ctx_contradicts_ans", 0.0)), 4),
                "nli_ctx_entails_ans": round(float(row.get("nli_ctx_entails_ans", 0.0)), 4),
                "n_entities_answer": float(row.get("n_entities_answer", 0)),
                "entity_overlap_ratio": round(float(row.get("entity_overlap_ratio", 0.0)), 4),
                "cosine_ctx_ans": round(float(row.get("cosine_ctx_ans", 0.0)), 4),
            },
        }
        rows.append(case)

    cases_df = pd.DataFrame(rows)
    counts = cases_df.groupby(["error_type", "category"]).size().unstack(fill_value=0)

    summary = {
        "n_test": int(len(y_test)),
        "n_false_positives": int(len(fp_idx)),
        "n_false_negatives": int(len(fn_idx)),
        "sampled": {"fp": int(len(fp_sample)), "fn": int(len(fn_sample))},
        "category_counts": {
            "false_positive": cases_df[cases_df["error_type"] == "false_positive"]["category"].value_counts().to_dict(),
            "false_negative": cases_df[cases_df["error_type"] == "false_negative"]["category"].value_counts().to_dict(),
        },
        "note": "Auto-tagged with heuristic rules - verify categories manually before paper use.",
    }

    with open(RESULTS_DIR / "error_analysis_cases.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(RESULTS_DIR / "error_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(" HaluRISC Error Analysis (10 FP + 10 FN on test set)")
    print("=" * 80)
    print(counts.fillna(0).astype(int).to_string())
    print("=" * 80)
    logger.info(f"Saved error_analysis.json + error_analysis_cases.json to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
