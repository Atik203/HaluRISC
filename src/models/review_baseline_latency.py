"""Per-baseline inference latency and model size on the reference machine (P2, H1).

Measures model-only predict() for every persisted seed-42 baseline on the same
200 test rows, the overlap-heuristic computation, and TF-IDF vectorizer
transforms. NLI-only and TF-IDF linear heads were not persisted as joblibs, so
their model-only cost is reported through the shared pipeline numbers plus the
measured vectorizer cost; no number is fabricated. Writes
artifacts/results/review/baseline_latency.json.
"""

import json
import time

import joblib
import numpy as np
import pandas as pd

from src.models.config import (
    ARTIFACTS_DIR,
    DATA_PROCESSED,
    MODELS_DIR,
    RESULTS_DIR,
)

OUT = RESULTS_DIR / "review"
OUT.mkdir(parents=True, exist_ok=True)
N = 200
REPS = 5
SEED = 42


def timed(fn, n=N, reps=REPS):
    fn()  # warmup
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) / n * 1000.0)
    return {
        "p50_ms": round(float(np.median(samples)), 3),
        "p95_ms": round(float(np.percentile(samples, 95)), 3),
        "mean_ms": round(float(np.mean(samples)), 3),
    }


def main():
    feat = pd.read_parquet(DATA_PROCESSED / "features_full.parquet")
    test = feat[feat["split"] == "test"].reset_index(drop=True).iloc[:N]
    feature_names = json.loads((MODELS_DIR / "feature_names.json").read_text())
    X_full = test[feature_names].to_numpy(dtype=np.float64)
    X_26 = X_full[:, :26]

    rows = {}

    # Standardized-feature models (seed 42).
    scaler = joblib.load(MODELS_DIR / "b2" / "scaler_full.joblib")
    Xs = scaler.transform(X_26)
    lr = joblib.load(MODELS_DIR / "b2" / "logistic_regression_full_seed_42.joblib")
    rows["logistic_regression"] = {
        "inputs": "26 base features (standardized)",
        "size_bytes": int((MODELS_DIR / "b2" / "logistic_regression_full_seed_42.joblib").stat().st_size),
        **timed(lambda: lr.predict(Xs)),
    }
    rf = joblib.load(MODELS_DIR / "b2" / "random_forest_seed_42.joblib")
    rows["random_forest"] = {
        "inputs": "26 base features",
        "size_bytes": int((MODELS_DIR / "b2" / "random_forest_seed_42.joblib").stat().st_size),
        **timed(lambda: rf.predict(X_26)),
    }
    xgb = joblib.load(MODELS_DIR / "b2" / "xgboost_seed_42.joblib")
    rows["xgboost_standard"] = {
        "inputs": "26 base features",
        "size_bytes": int((MODELS_DIR / "b2" / "xgboost_seed_42.joblib").stat().st_size),
        **timed(lambda: xgb.predict(X_26)),
    }
    ec = joblib.load(MODELS_DIR / "b6" / "xgboost_m3_seed_42.joblib")
    from src.models.run_b6_modified import feature_matrix, variant_features

    from src.features.claim_features import FEATURE_COLUMNS as CLAIM_COLUMNS  # noqa: F401

    claims = pd.read_parquet(DATA_PROCESSED / "claim_features.parquet")
    merged = feat.merge(claims, on="sample_id", how="left", validate="one_to_one")
    merged["source_ragtruth"] = 0.0
    m3_names = variant_features("m3")
    X_35 = feature_matrix(
        merged[merged["split"] == "test"].reset_index(drop=True).iloc[:N], m3_names, True
    )
    rows["ec_xgb"] = {
        "inputs": "35 features (26 base + 8 claim aggregates + source)",
        "size_bytes": int((MODELS_DIR / "b6" / "xgboost_m3_seed_42.joblib").stat().st_size),
        **timed(lambda: ec.predict(X_35)),
    }

    # Overlap heuristic on raw texts (no model).
    qa = pd.read_parquet(DATA_PROCESSED / "qa_clean.parquet", columns=["sample_id", "question", "context", "answer"])
    qmap = {r.sample_id: (r.question, r.context, r.answer) for r in qa.itertuples()}
    texts = [qmap[s] for s in test["sample_id"].tolist()]

    def heuristic():
        out = np.empty(len(texts))
        for i, (q, c, a) in enumerate(texts):
            at = set(str(a).lower().split())
            ct = set(str(c).lower().split())
            out[i] = 1.0 - (len(at & ct) / max(1, len(at)))
        return out

    rows["heuristic_overlap"] = {"inputs": "answer + context tokens", "size_bytes": 0, **timed(heuristic)}

    # TF-IDF vectorizer transforms (saved artifacts; linear heads not persisted).
    ans_by_id = dict(zip(qa["sample_id"].tolist(), qa["answer"].fillna("").tolist()))
    q_by_id = dict(zip(qa["sample_id"].tolist(), qa["question"].fillna("").tolist()))
    ctx_by_id = dict(zip(qa["sample_id"].tolist(), qa["context"].fillna("").tolist()))
    sids = test["sample_id"].tolist()
    for name, short in (("tfidf_tfidf_answer", "tfidf_answer"), ("tfidf_tfidf_all", "tfidf_all")):
        vec = joblib.load(MODELS_DIR / "b2" / f"{name}.joblib")
        if short == "tfidf_answer":
            docs = [ans_by_id[s] for s in sids]
        else:
            docs = [q_by_id[s] + " " + ctx_by_id[s] + " " + ans_by_id[s] for s in sids]
        rows[short] = {
            "inputs": "answer text" if short == "tfidf_answer" else "question + context + answer",
            "size_bytes": int((MODELS_DIR / "b2" / f"{name}.joblib").stat().st_size),
            **timed(lambda v=vec, d=docs: v.transform(d)),
            "note": "vectorizer transform only; linear head not persisted",
        }

    rec = {
        "meta": {
            "source": "review_baseline_latency.py",
            "machine": "reference laptop, RTX 3060 6GB; sklearn/XGBoost on CPU",
            "n_rows": N,
            "reps": REPS,
            "seed": SEED,
            "unit": "ms per row, model-only unless noted",
        },
        "baselines": rows,
    }
    (OUT / "baseline_latency.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: {kk: v[kk] for kk in ("p50_ms", "p95_ms", "size_bytes")} for k, v in rows.items()}, indent=1))


if __name__ == "__main__":
    main()
