"""
Data preparation script for HaluRISC.
Parses qa_data.json (JSONL) into a binary classification dataset (two rows per entry: correct & hallucinated),
performs a GROUP-AWARE train/val/test split (70/15/15) so that both answer variants of one original question
stay in the same partition, saves split indices, a leakage report, and an auto-sampled audit file.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RAW_DATA_PATH = os.path.join("data", "raw", "halueval", "qa_data.json")
PROCESSED_DIR = os.path.join("data", "processed")
ARTIFACTS_DIR = os.path.join("artifacts")
PROCESSED_PARQUET = os.path.join(PROCESSED_DIR, "qa_clean.parquet")
AUDIT_JSON = os.path.join(PROCESSED_DIR, "audit_50_samples.json")
SPLIT_INDICES_NPY = os.path.join(ARTIFACTS_DIR, "split_indices.npy")
SPLIT_INDICES_JSON = os.path.join(ARTIFACTS_DIR, "split_indices.json")
SPLIT_REPORT_JSON = os.path.join(ARTIFACTS_DIR, "split_integrity_report.json")

SPLIT_SEED = 42


def load_and_parse_raw_data(raw_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Loads HaluEval QA JSONL and expands into 2 rows per question (correct=0, hallucinated=1)."""
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data not found at {raw_path}. Run src/data/download.py first.")

    raw_items = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_items.append(json.loads(line))

    logging.info(f"Loaded {len(raw_items)} raw QA items.")
    rows = []

    for idx, item in enumerate(raw_items):
        question = item.get("question", "").strip()
        context = item.get("knowledge", "").strip()
        right_ans = item.get("right_answer", item.get("answer", "")).strip()
        hallucinated_ans = item.get("hallucinated_answer", "").strip()

        if not question or not right_ans:
            continue

        # Correct sample (label = 0)
        rows.append({
            "sample_id": f"q_{idx}_correct",
            "item_idx": idx,
            "question": question,
            "context": context,
            "answer": right_ans,
            "label": 0
        })

        # Hallucinated sample (label = 1)
        if hallucinated_ans:
            rows.append({
                "sample_id": f"q_{idx}_hallucinated",
                "item_idx": idx,
                "question": question,
                "context": context,
                "answer": hallucinated_ans,
                "label": 1
            })

    df = pd.DataFrame(rows)
    logging.info(f"Created dataset with {len(df)} rows ({df['label'].value_counts().to_dict()}).")
    return df


def build_integrity_report(df: pd.DataFrame) -> dict:
    """Leakage report: every item_idx (source question) must map to exactly one split."""
    per = df.groupby("item_idx")["split"].nunique()
    cross = int((per > 1).sum())
    report = {
        "split": "group_by_item_idx",
        "seed": SPLIT_SEED,
        "n_groups_total": int(per.size),
        "n_groups_per_split": {str(k): int(v) for k, v in df.groupby("split")["item_idx"].nunique().to_dict().items()},
        "n_rows_per_split": {str(k): int(v) for k, v in df.groupby("split").size().to_dict().items()},
        "label_mean_per_split": {str(k): round(float(v), 4) for k, v in df.groupby("split")["label"].mean().to_dict().items()},
        "groups_spanning_multiple_splits": cross,
        "leakage_free": cross == 0,
    }
    if cross > 0:
        raise AssertionError(f"Group leakage detected: {cross} item_idx values span multiple splits")
    return report


def group_split_by_item(df: pd.DataFrame, test_size: float = 0.30, val_share: float = 0.5, seed: int = SPLIT_SEED):
    """Group-aware 70/15/15 split: each original question (item_idx) goes to exactly one partition.

    Returns (df_with_split_column, integrity_report).
    """
    grp = (
        df.groupby("item_idx", sort=False)
        .agg(label=("label", "max"), n_rows=("label", "size"))
        .reset_index()
    )
    strat = grp["label"] if grp["label"].nunique() > 1 else None
    train_g, temp_g = train_test_split(grp, test_size=test_size, random_state=seed, stratify=strat)
    strat_t = temp_g["label"] if temp_g["label"].nunique() > 1 else None
    val_g, test_g = train_test_split(temp_g, test_size=val_share, random_state=seed, stratify=strat_t)

    split_of: dict = {}
    for g, name in ((train_g, "train"), (val_g, "val"), (test_g, "test")):
        for i in g["item_idx"]:
            split_of[int(i)] = name

    out = df.copy()
    out["split"] = out["item_idx"].map(split_of)
    report = build_integrity_report(out)
    logging.info(
        f"Group split: train {len(train_g)} / val {len(val_g)} / test {len(test_g)} groups; "
        f"leakage_free={report['leakage_free']}"
    )
    return out, report


def prepare_splits_and_save(df: pd.DataFrame):
    """Performs the group-aware train/val/test split and saves all artifacts."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    df, report = group_split_by_item(df, seed=SPLIT_SEED)

    # Save processed dataset to Parquet
    df.to_parquet(PROCESSED_PARQUET, index=False)
    logging.info(f"Saved processed dataset to {PROCESSED_PARQUET}")

    # Save split indices + group ids (exact reproducibility)
    split_indices = {
        "split": "group_by_item_idx",
        "train": df.index[df["split"] == "train"].tolist(),
        "val": df.index[df["split"] == "val"].tolist(),
        "test": df.index[df["split"] == "test"].tolist(),
        "group_ids": {
            "train": sorted(df.loc[df["split"] == "train", "item_idx"].unique().tolist()),
            "val": sorted(df.loc[df["split"] == "val", "item_idx"].unique().tolist()),
            "test": sorted(df.loc[df["split"] == "test", "item_idx"].unique().tolist()),
        },
        "seed": SPLIT_SEED,
        "n_train": int((df["split"] == "train").sum()),
        "n_val": int((df["split"] == "val").sum()),
        "n_test": int((df["split"] == "test").sum()),
    }

    with open(SPLIT_INDICES_JSON, "w") as f:
        json.dump(split_indices, f, indent=2)

    np.save(SPLIT_INDICES_NPY, split_indices, allow_pickle=True)
    logging.info(f"Saved split indices to {SPLIT_INDICES_JSON} and {SPLIT_INDICES_NPY}")

    with open(SPLIT_REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(f"Saved split integrity report to {SPLIT_REPORT_JSON}")

    # Sample 50 rows for the MANUAL audit (labels must be reviewed by a human before the paper)
    audit_samples = df.sample(n=min(50, len(df)), random_state=SPLIT_SEED).to_dict(orient="records")
    with open(AUDIT_JSON, "w") as f:
        json.dump(audit_samples, f, indent=2)
    logging.info(f"Saved 50 auto-sampled audit rows to {AUDIT_JSON} (manual review required)")


if __name__ == "__main__":
    df = load_and_parse_raw_data()
    prepare_splits_and_save(df)
