"""H200 appendix: encoder throughput and VRAM on the MIG slice (E4).

Times the two GPU stages of feature extraction on a fixed sample and records
peak allocated VRAM, for the deployed encoders and the larger alternatives.

Measured per configuration:
  * NLI stage: CrossEncoder.predict over (context, answer) + (answer, context)
  * embedding stage: SentenceTransformer.encode over answer/question/context
  * peak VRAM allocated during the run

Outputs
  artifacts/results/h200/efficiency.csv
  artifacts/results/h200/efficiency.json

Run (repo root, GPU node):
  python src/models/h200_efficiency.py --n-rows 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("h200_efficiency")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "results" / "h200"
DATA = ROOT / "data" / "processed"

CONFIGS = [
    ("deployed", "cross-encoder/nli-deberta-v3-base", "sentence-transformers/all-MiniLM-L6-v2", 256),
    ("nli_large", "cross-encoder/nli-deberta-v3-large", "sentence-transformers/all-MiniLM-L6-v2", 256),
    ("bge_base", "cross-encoder/nli-deberta-v3-base", "BAAI/bge-base-en-v1.5", 256),
    ("deployed_bs128", "cross-encoder/nli-deberta-v3-base", "sentence-transformers/all-MiniLM-L6-v2", 128),
    ("deployed_bs512", "cross-encoder/nli-deberta-v3-base", "sentence-transformers/all-MiniLM-L6-v2", 512),
]


def load_rows(n: int) -> pd.DataFrame:
    path = DATA / "qa_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    text_cols = [c for c in ("question", "context", "answer") if c in df.columns]
    if len(text_cols) < 3:
        raise ValueError(f"expected question/context/answer, found {df.columns.tolist()}")
    return df.head(n).reset_index(drop=True)


def run_config(tag: str, nli_name: str, emb_name: str, batch: int, rows: pd.DataFrame) -> dict:
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    q = rows["question"].astype(str).tolist()
    c = rows["context"].astype(str).tolist()
    a = rows["answer"].astype(str).tolist()
    pairs = [[ci, ai] for ci, ai in zip(c, a)] + [[ai, ci] for ci, ai in zip(c, a)]

    t0 = time.perf_counter()
    nli = CrossEncoder(nli_name, device="cuda")
    load_nli = time.perf_counter() - t0

    t0 = time.perf_counter()
    nli.predict(pairs, batch_size=batch, apply_softmax=True)
    nli_s = time.perf_counter() - t0
    del nli
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    emb = SentenceTransformer(emb_name, device="cuda")
    load_emb = time.perf_counter() - t0

    texts = a + q + c
    t0 = time.perf_counter()
    emb.encode(texts, batch_size=batch)
    emb_s = time.perf_counter() - t0
    del emb

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    row = {
        "config": tag,
        "nli_model": nli_name,
        "embed_model": emb_name,
        "batch_size": batch,
        "n_rows": int(len(rows)),
        "nli_pairs": int(len(pairs)),
        "nli_seconds": nli_s,
        "nli_pairs_per_s": len(pairs) / nli_s,
        "nli_rows_per_s": len(rows) / nli_s,
        "embed_texts": int(len(texts)),
        "embed_seconds": emb_s,
        "embed_texts_per_s": len(texts) / emb_s,
        "load_seconds_nli": load_nli,
        "load_seconds_embed": load_emb,
        "peak_vram_mb": peak_mb,
    }
    logger.info("done %s batch=%d nli=%.1f rows/s peak=%.0f MB", tag, batch, row["nli_rows_per_s"], peak_mb)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rows", type=int, default=2000)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.n_rows)
    logger.info("timing on %d rows", len(rows))

    results = [run_config(tag, nli, emb, bs, rows) for tag, nli, emb, bs in CONFIGS]
    df = pd.DataFrame(results)
    df.to_csv(OUT / "efficiency.csv", index=False)
    (OUT / "efficiency.json").write_text(json.dumps({"n_rows": int(len(rows)), "runs": results}, indent=2))

    base = df[df["config"] == "deployed"].iloc[0]
    for _, r in df.iterrows():
        logger.info(
            "%s batch=%d: %.1fx NLI throughput, %.2fx VRAM",
            r["config"], r["batch_size"], r["nli_rows_per_s"] / base["nli_rows_per_s"], r["peak_vram_mb"] / base["peak_vram_mb"],
        )
    print(df[["config", "batch_size", "nli_rows_per_s", "embed_texts_per_s", "peak_vram_mb"]].to_string(index=False))


if __name__ == "__main__":
    main()
