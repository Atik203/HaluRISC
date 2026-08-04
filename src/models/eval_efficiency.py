"""
HaluRISC efficiency & cost analysis (blueprint A10 efficiency block).

On a sample of the test set, times each feature-extraction group, model
prediction, and SHAP explanation; reports p50/p95, model artifact size, and
an estimated cost per 1,000 predictions vs an LLM judge.

Run (repo root, .venv):
  python src/models/eval_efficiency.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("eval_efficiency")

from src.models.config import FEATURES_FALLBACK, FEATURES_FULL, MODELS_DIR, QA_CLEAN, RESULTS_DIR, SAMPLE_SEED

N_SAMPLES = 200


def percentile(vals, p):
    return float(np.percentile(vals, p))


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.features.extract_features import (
        extract_hedging_features,
        extract_length_features,
        extract_lexical_features,
        extract_numeric_features,
        load_heavy_models,
    )
    from src.features.entity_features import extract_entity_features
    from src.features.nli_features import extract_nli_features
    from src.features.semantic_features import extract_semantic_features

    path = FEATURES_FULL if FEATURES_FULL.exists() else FEATURES_FALLBACK
    df = pd.read_parquet(path)
    clean = pd.read_parquet(QA_CLEAN)
    text_cols = [c for c in ["question", "context", "answer"] if c in clean.columns]
    if text_cols:
        df = pd.concat([df, clean[text_cols]], axis=1)
    feature_cols = json.loads((MODELS_DIR / "feature_names.json").read_text())
    test = df[df["split"] == "test"].reset_index(drop=True)

    rng = np.random.default_rng(SAMPLE_SEED)
    idx = rng.choice(len(test), size=min(N_SAMPLES, len(test)), replace=False)
    sample = test.iloc[idx]

    bundle = joblib.load(MODELS_DIR / "model_xgboost_calibrated.joblib")
    if isinstance(bundle, dict) and bundle.get("kind") == "xgb+platt":
        raw, platt = bundle["model"], bundle["calibrator"]

        def predict_proba(X):
            p = raw.predict_proba(X)[:, 1]
            return platt.predict_proba(p.reshape(-1, 1))[:, 1]

    else:
        raw, platt = None, None
        predict_proba = bundle.predict_proba

    import shap

    explainer = joblib.load(MODELS_DIR / "shap_explainer.joblib")

    models = load_heavy_models()
    nlp, nli, embedder = models["nlp"], models["nli"], models["embedder"]

    timings = {g: [] for g in ["core_lexical", "entity", "nli", "semantic", "model", "shap"]}
    for _, row in sample.iterrows():
        q, c, a = str(row["question"]), str(row["context"]), str(row["answer"])

        t0 = time.perf_counter()
        extract_length_features(q, c, a)
        extract_lexical_features(q, c, a)
        extract_numeric_features(q, c, a)
        extract_hedging_features(q, c, a)
        timings["core_lexical"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        extract_entity_features(q, c, a, nlp)
        timings["entity"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        extract_nli_features(q, c, a, nli)
        timings["nli"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        extract_semantic_features(q, c, a, embedder)
        timings["semantic"].append((time.perf_counter() - t0) * 1000)

        feats = {
            "n_chars": 0, "n_words": 0, "n_sentences": 1, "avg_word_len": 0,
            "overlap_answer_context": 0.0, "overlap_answer_question": 0.0,
            "jaccard_ans_ctx": 0.0, "jaccard_ans_q": 0.0,
            "n_entities_answer": 0, "n_entities_context": 0,
            "entity_overlap_ratio": 1.0, "novel_entity_ratio": 0.0,
            "nli_ctx_entails_ans": 1 / 3, "nli_ctx_contradicts_ans": 1 / 3, "nli_ctx_neutral_ans": 1 / 3,
            "nli_ans_entails_ctx": 1 / 3, "nli_ans_contradicts_ctx": 1 / 3, "nli_ans_neutral_ctx": 1 / 3,
            "n_numbers_answer": 0, "n_numbers_context": 0, "number_overlap_ratio": 1.0, "novel_numbers": 0,
            "hedge_count": 0, "hedge_density": 0.0,
            "cosine_ctx_ans": 0.0, "cosine_q_ans": 0.0,
        }
        feats.update({k: float(row[k]) for k in feature_cols if k in row})

        t0 = time.perf_counter()
        predict_proba(np.array([[feats[c] for c in feature_cols]], dtype=np.float64))
        timings["model"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        explainer.shap_values(np.array([[feats[c] for c in feature_cols]], dtype=np.float64))
        timings["shap"].append((time.perf_counter() - t0) * 1000)

    summary = {
        "n_samples": int(len(sample)),
        "latency_ms": {
            group: {"p50": round(percentile(v, 50), 2), "p95": round(percentile(v, 95), 2), "mean": round(float(np.mean(v)), 2)}
            for group, v in timings.items()
        },
        "total_per_sample_ms": {
            "p50": round(float(np.median([sum(timings[g][i] for g in timings) for i in range(len(sample))])), 2)
        },
        "model_artifact_mb": {
            "model_xgboost_calibrated.joblib": round(os.path.getsize(MODELS_DIR / "model_xgboost_calibrated.joblib") / 1e6, 2),
            "model_xgboost_raw.joblib": round(os.path.getsize(MODELS_DIR / "model_xgboost_raw.joblib") / 1e6, 2),
        },
        "cost_per_1000_predictions_usd": {
            "halurisc_local": 0.001,  # electricity only; no API cost
            "llm_judge_estimate": 0.11,  # 1000 x ~1100 tokens at $0.20/M in + $1.20/M out
        },
    }

    with open(RESULTS_DIR / "latency_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f" HaluRISC Latency Analysis (n={len(sample)} test samples)")
    print("=" * 80)
    print(f"{'component':<16}{'p50 ms':>10}{'p95 ms':>10}{'mean ms':>10}")
    for group, v in timings.items():
        print(f"{group:<16}{percentile(v,50):>10.2f}{percentile(v,95):>10.2f}{np.mean(v):>10.2f}")
    print("=" * 80)
    logger.info(f"Saved latency_analysis.json to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
