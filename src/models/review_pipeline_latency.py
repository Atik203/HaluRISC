"""Measure the deployed pipeline latency over HTTP (P2, WS-H2).

Times /predict (classifier), /analyze (classifier + SHAP) and /verify
(per-claim NLI verification against the provided context), plus a small
judge-fallback batch, against the running FastAPI service.

Run (repo root) with the API already serving on port 8000:
  python src/models/review_pipeline_latency.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import requests

from src.models.config import DATA_PROCESSED, RESULTS_DIR

BASE = os.environ.get("HALU_API_URL", "http://127.0.0.1:8000")
N = int(os.environ.get("HALU_LAT_N", "30"))
N_JUDGE = int(os.environ.get("HALU_LAT_N_JUDGE", "8"))
OUT = RESULTS_DIR / "review"


def payload(row):
    return {
        "question": str(row["question"])[:2000],
        "context": str(row["context"])[:4000],
        "answer": str(row["answer"])[:2000],
    }


def time_calls(path, rows, extra=None, limit=None):
    lat = []
    for _, row in rows.iloc[:limit].iterrows():
        body = payload(row)
        if extra:
            body.update(extra)
        t0 = time.perf_counter()
        r = requests.post(f"{BASE}{path}", json=body, timeout=180)
        lat.append((time.perf_counter() - t0) * 1000)
        # stay under the 30-per-minute /verify rate limit
        time.sleep(2.0 if path == "/verify" else 0.2)
        if r.status_code != 200:
            raise RuntimeError(f"{path} -> {r.status_code}: {r.text[:200]}")
    return lat


def stats(lat):
    return {
        "n": len(lat),
        "p50_ms": round(float(np.median(lat)), 1),
        "p95_ms": round(float(np.percentile(lat, 95)), 1),
        "mean_ms": round(float(np.mean(lat)), 1),
        "min_ms": round(float(np.min(lat)), 1),
        "max_ms": round(float(np.max(lat)), 1),
    }


def main() -> None:
    health = requests.get(f"{BASE}/health", timeout=30).json()
    qa = pd.read_parquet(DATA_PROCESSED / "qa_clean.parquet")
    rows = qa[qa["split"] == "test"].sample(N + N_JUDGE, random_state=42).reset_index(drop=True)

    predict = time_calls("/predict", rows)
    analyze = time_calls("/analyze", rows)
    verify = time_calls("/verify", rows, extra={"evidence_mode": "context", "judge_uncertain": False})
    judge = time_calls("/verify", rows, extra={"evidence_mode": "context", "judge_uncertain": True}, limit=N_JUDGE)

    bundle = {
        "meta": {
            "source": "review_pipeline_latency.py",
            "base_url": BASE,
            "n_requests": {"predict": len(predict), "analyze": len(analyze), "verify": len(verify), "verify_judge": len(judge)},
            "note": "End-to-end HTTP latency including JSON parsing and the feature model; judge batch uses the OpenAI judge fallback.",
        },
        "health": health,
        "predict": stats(predict),
        "analyze": stats(analyze),
        "verify_context": stats(verify),
        "verify_with_judge": stats(judge),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pipeline_latency.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(json.dumps({k: bundle[k] for k in ("predict", "analyze", "verify_context", "verify_with_judge")}, indent=2))


if __name__ == "__main__":
    main()
