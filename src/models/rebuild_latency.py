"""Time the m3 feature path, including claim-level NLI features.

The legacy 61.8 ms latency artifact covers the 26 base features only. EC-XGB m3
adds `compute_claim_features` on every request, so this script measures base,
claim, and total extraction on the same rows.

Run:
  python src/models/rebuild_latency.py --limit 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.api.main import load_feature_models
from src.features.claim_features import compute_claim_features
from src.features.extract_features import extract_all_features_single

OUT = Path("artifacts/results/rebuild/latency")
logger = logging.getLogger("rebuild_latency")


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean() * 1000),
        "p50_ms": float(np.percentile(arr, 50) * 1000),
        "p95_ms": float(np.percentile(arr, 95) * 1000),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--input", default="data/processed/qa_clean.parquet")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(args.input).head(args.limit)
    logger.info("rows: %d", len(frame))

    t0 = time.perf_counter()
    models = load_feature_models()
    load_s = time.perf_counter() - t0
    logger.info("feature models loaded in %.2fs on %s", load_s, models.get("device", "?"))

    # Warm up the NLI path so the first call does not dominate the tail.
    for _ in range(2):
        row = frame.iloc[0]
        extract_all_features_single(row["question"], row["context"], row["answer"], models)
        compute_claim_features(row["answer"], row["context"], models["nli"])

    base, claims, total = [], [], []
    for _, row in frame.iterrows():
        q, c, a = row["question"], row["context"], row["answer"]
        t = time.perf_counter()
        base_feats = extract_all_features_single(q, c, a, models)
        mid = time.perf_counter()
        claim_feats = compute_claim_features(a, c, models["nli"])
        end = time.perf_counter()
        base.append(mid - t)
        claims.append(end - mid)
        total.append(end - t)
        assert len(claim_feats) == 8

    result = {
        "schema": "rebuild-m3-latency-v1",
        "rows": len(frame),
        "device": models.get("device", "?"),
        "model_load_s": load_s,
        "base": summarize(base),
        "claim": summarize(claims),
        "total": summarize(total),
        "note": "Per-row Python timing on unique rows, no HTTP or response serialization.",
    }
    (OUT / "m3_latency.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
