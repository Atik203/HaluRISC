"""
External-corpus feature extraction for the b6 multi-source variant.

Extracts the same 26 base features (7 groups) and 8 claim-level features that
the HaluEval pipeline uses, for every RAGTruth task and FaithBench row in
data/processed/unified_records.parquet. The b6 script trains on RAGTruth
non-QA train rows and evaluates on RAGTruth QA test and FaithBench.

Output columns: sample_id, source_dataset, task, official_split,
source_group_id, item_idx, label, split, <26 base features>, <8 claim features>.

Run (repo root, .venv):
  python src/features/extract_external_features.py
  python src/features/extract_external_features.py --limit 40    # smoke test
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.features.claim_features import FEATURE_COLUMNS as CLAIM_COLUMNS
from src.features.claim_features import extract_claim_features_df
from src.features.extract_features import extract_full_feature_set, load_heavy_models
from src.models.config import DATA_PROCESSED

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("external_features")

DEFAULT_INPUT = DATA_PROCESSED / "unified_records.parquet"
DEFAULT_OUTPUT = DATA_PROCESSED / "external_features.parquet"

META_COLUMNS = ["sample_id", "source_dataset", "task", "official_split", "source_group_id", "item_idx", "label", "split"]


def build_frame(input_path: Path, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    df = df[df["source_dataset"] != "halueval"].reset_index(drop=True)
    if limit:
        df = df.head(limit)
    df["item_idx"] = pd.factorize(df["source_group_id"])[0]
    df["split"] = df["official_split"].fillna("")
    df.loc[df["split"] == "", "split"] = "test"  # FaithBench ships as evaluation only
    for col in ("question", "context", "answer"):
        df[col] = df[col].fillna("").astype(str)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="External corpus features for b6")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=os.environ.get("HALU_EXTERNAL_DEVICE", "cuda"))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} not found")
    df = build_frame(args.input, args.limit)
    logger.info(f"External rows: {len(df)} | tasks: {df.groupby(['source_dataset', 'task', 'split']).size().to_dict()}")

    t0 = time.time()
    models = load_heavy_models(device=args.device)
    logger.info(f"Heavy models loaded in {time.time() - t0:.1f}s ({models.get('nli_name')})")

    t0 = time.time()
    base = extract_full_feature_set(df, models, batch_size=args.batch_size)
    logger.info(f"Base features done in {time.time() - t0:.1f}s")

    t0 = time.time()
    claims = extract_claim_features_df(
        df, models["nli"], partial_path=None, checkpoint_every=2000,
        batch_size=args.batch_size, chunk_samples=32,
    )
    logger.info(f"Claim features done in {time.time() - t0:.1f}s")

    out = base.merge(claims, on="sample_id", how="left")
    out = out.merge(df[["sample_id", "source_dataset", "task", "official_split", "source_group_id"]],
                    on="sample_id", how="left")
    missing = [c for c in CLAIM_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"missing claim columns: {missing}")
    if out[CLAIM_COLUMNS].isna().any().any():
        raise ValueError("claim features contain NaN after merge")
    out.to_parquet(args.output, index=False)
    logger.info(f"Saved {out.shape[0]} rows x {out.shape[1]} cols -> {args.output}")
    print(out.groupby(["source_dataset", "task", "split"]).size().to_string())


if __name__ == "__main__":
    main()
