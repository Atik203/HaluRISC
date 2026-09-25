"""
Export per-model risk scores for the manuscript case table (Table tab:cases).

Reads the frozen B2 predictions and the deployed B4 display calibrator, then
writes artifacts/results/b2/b2_case_scores.csv with the seed-42 score of every
served model for three illustrative HaluEval test rows:

  q_6775_correct      grounded span, label 0
  q_6775_hallucinated hallucinated sentence, label 1 (same question/context)
  q_3061_correct      grounded one-word answer, label 0

Run (repo root, .venv):
  python src/models/export_case_scores.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd

from src.models.config import DATA_PROCESSED, MODELS_DIR, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("export_case_scores")

B2_DIR = RESULTS_DIR / "b2"
B4_DIR = MODELS_DIR / "b4"

SAMPLE_IDS = ["q_6775_correct", "q_6775_hallucinated", "q_3061_correct"]
EXPECTED_LABELS = {"q_6775_correct": 0, "q_6775_hallucinated": 1, "q_3061_correct": 0}
MODEL_ORDER = [
    "heuristic_overlap",
    "lr_full",
    "rf_full",
    "nli_only",
    "tfidf_answer",
    "xgboost",
]


def display_calibrated(raw: float, bundle: dict) -> float:
    """Mirror the deployed display map in src/api/main.py."""
    method, cal = bundle["method"], bundle["calibrator"]
    if method == "isotonic":
        out = cal.predict(np.array([raw]))
    else:
        p = min(max(raw, 1e-12), 1 - 1e-12)
        z = np.log(p / (1 - p))
        if method == "temperature":
            out = 1.0 / (1.0 + np.exp(-z / float(cal)))
        else:
            out = cal.predict_proba(z.reshape(-1, 1))[:, 1]
    return float(np.asarray(out).reshape(-1)[0])


def main() -> None:
    qa = pd.read_parquet(DATA_PROCESSED / "qa_clean.parquet")
    features = pd.read_parquet(DATA_PROCESSED / "features_full.parquet")
    preds = pd.read_parquet(B2_DIR / "b2_predictions.parquet")
    bundle = joblib.load(B4_DIR / "calibrator_display.joblib")

    rows = []
    for sample_id in SAMPLE_IDS:
        qa_row = qa[qa["sample_id"] == sample_id]
        assert len(qa_row) == 1, f"{sample_id} not unique in qa_clean"
        qa_row = qa_row.iloc[0]
        assert qa_row["split"] == "test", f"{sample_id} is not a test row"
        assert int(qa_row["label"]) == EXPECTED_LABELS[sample_id], f"{sample_id} label mismatch"

        sub = preds[preds["sample_id"] == sample_id].drop_duplicates("model").set_index("model")
        assert set(MODEL_ORDER).issubset(sub.index), f"{sample_id} missing model scores"

        overlap = float(features.loc[features["sample_id"] == sample_id, "overlap_answer_context"].iloc[0])
        heuristic = 1.0 - overlap
        assert abs(heuristic - float(sub.loc["heuristic_overlap", "score"])) < 1e-9, "heuristic mismatch"

        for model in MODEL_ORDER:
            rows.append({
                "sample_id": sample_id,
                "label": int(qa_row["label"]),
                "model": model,
                "score": round(float(sub.loc[model, "score"]), 4),
            })
        raw = float(sub.loc["xgboost", "score"])
        rows.append({
            "sample_id": sample_id,
            "label": int(qa_row["label"]),
            "model": "deployed_calibrated",
            "score": round(display_calibrated(raw, bundle), 4),
        })

    out = pd.DataFrame(rows)
    out_path = B2_DIR / "b2_case_scores.csv"
    out.to_csv(out_path, index=False)

    meta = {
        "source": "b2_predictions.parquet (seed 42) + b4 calibrator_display.joblib",
        "display_calibrator": bundle["method"],
        "samples": {
            sid: {
                "label": EXPECTED_LABELS[sid],
                "question": qa.loc[qa["sample_id"] == sid, "question"].iloc[0],
                "answer": qa.loc[qa["sample_id"] == sid, "answer"].iloc[0],
            }
            for sid in SAMPLE_IDS
        },
    }
    (B2_DIR / "b2_case_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    logger.info(f"Saved {len(out)} rows to {out_path}")
    print(out.pivot(index="model", columns="sample_id", values="score").loc[
        MODEL_ORDER + ["deployed_calibrated"]
    ].to_string())


if __name__ == "__main__":
    main()
