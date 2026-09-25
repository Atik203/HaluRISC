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
    df["split"] = df["official_split"].fillna("")
    df.loc[df["split"] == "", "split"] = "test"  # FaithBench ships as evaluation only

    # M3 needs RAGTruth non-QA train rows plus every evaluation row. RAGTruth
    # QA train rows are deliberately held out (QA is the external test task),
    # so skipping them saves ~27% of the slow claim-feature extraction.
    rag = df["source_dataset"] == "ragtruth"
    needed = (
        (rag & (df["split"] == "test"))
        | (rag & (df["task"].isin(["summarization", "data_to_text"])) & (df["split"] == "train"))
        | (df["source_dataset"] == "faithbench")
    )
    df = df[needed].reset_index(drop=True)
    if limit:
        df = df.head(limit)
    df["item_idx"] = pd.factorize(df["source_group_id"])[0]
    for col in ("question", "context", "answer"):
        df[col] = df[col].fillna("").astype(str)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="External corpus features for b6")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=os.environ.get("HALU_EXTERNAL_DEVICE", "cuda"))
    # External contexts are full articles, so the model runs at the 512-token
    # limit. Small batches avoid VRAM thrashing on 6 GB cards (batch 256 was
    # ~26x slower than batch 64 because the GPU spilled into shared memory).
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--claim-batch-size", type=int, default=16)
    parser.add_argument("--fresh", action="store_true", help="ignore cached base features and claim checkpoints")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} not found")
    df = build_frame(args.input, args.limit)
    logger.info(f"External rows: {len(df)} | tasks: {df.groupby(['source_dataset', 'task', 'split']).size().to_dict()}")

    t0 = time.time()
    models = load_heavy_models(device=args.device)
    logger.info(f"Heavy models loaded in {time.time() - t0:.1f}s ({models.get('nli_name')})")

    # Base features are cached so a failed or interrupted claim pass never
    # repeats the NER/NLI/SBERT work (the NER stage alone is ~10 minutes).
    base_path = args.output.with_suffix(".base.parquet")
    if base_path.exists() and not args.fresh:
        cached = pd.read_parquet(base_path).drop(columns=["item_idx", "label", "split"], errors="ignore")
        base = cached.merge(df[["sample_id", "item_idx", "label", "split"]], on="sample_id", how="inner")
        logger.info(f"Reusing cached base features: {base.shape[0]} rows from {base_path}")
    else:
        t0 = time.time()
        base = extract_full_feature_set(df, models, batch_size=args.batch_size)
        base.to_parquet(base_path, index=False)
        logger.info(f"Base features done in {time.time() - t0:.1f}s (cached to {base_path})")

    claim_partial = args.output.with_suffix(".claims.partial.parquet")
    if claim_partial.exists():
        claim_partial.unlink()
        logger.info(f"Removed stale claim checkpoint: {claim_partial}")
    t0 = time.time()
    claims = extract_claim_features_df(
        df, models["nli"], partial_path=claim_partial, checkpoint_every=500,
        batch_size=args.claim_batch_size, chunk_samples=24,
    )
    logger.info(f"Claim features done in {time.time() - t0:.1f}s")
    if claim_partial.exists():
        claim_partial.unlink()

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
