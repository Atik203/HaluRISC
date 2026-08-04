"""
RAGTruth (ACL 2024) QA subset acquisition for HaluRISC external validation.

Downloads `wandb/RAGTruth-processed` from HuggingFace, keeps the QA task,
derives a binary label (has_hallucination = evident_conflict OR baseless_info),
and caches to data/raw/ragtruth/ragtruth_qa.parquet.

Run (repo root, .venv):
  python src/data/download_ragtruth.py
"""

import logging
import os
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("download_ragtruth")

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data" / "raw" / "ragtruth" / "ragtruth_qa.parquet"

DATASET = "wandb/RAGTruth-processed"


def download_ragtruth_qa(limit: int = 2000, split: str = "train") -> pd.DataFrame:
    from datasets import load_dataset

    if OUTPUT.exists():
        logger.info(f"RAGTruth QA cache already exists: {OUTPUT}")
        return pd.read_parquet(OUTPUT)

    logger.info(f"Loading {DATASET} [{split}] ...")
    ds = load_dataset(DATASET, split=split)
    rows = []
    for r in ds:
        if r["task_type"] != "QA":
            continue
        proc = r.get("hallucination_labels_processed") or {}
        label = int(bool(proc.get("evident_conflict", 0) or proc.get("baseless_info", 0)))
        rows.append({
            "sample_id": f"ragtruth_{r['id']}",
            "question": str(r.get("query", "")),
            "context": str(r.get("context", "")),
            "answer": str(r.get("output", "")),
            "label": label,
            "model": r.get("model", ""),
            "quality": r.get("quality", ""),
        })
        if limit and len(rows) >= limit:
            break

    df = pd.DataFrame(rows)
    os.makedirs(OUTPUT.parent, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)
    logger.info(f"Saved {len(df)} RAGTruth QA samples (label balance: {df['label'].value_counts().to_dict()}) to {OUTPUT}")
    return df


if __name__ == "__main__":
    download_ragtruth_qa()
