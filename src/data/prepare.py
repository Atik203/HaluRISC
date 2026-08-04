"""
Data preparation script for HaluRISC.
Parses qa_data.json (JSONL) into a binary classification dataset (two rows per entry: correct & hallucinated),
performs stratified train/val/test split (70/15/15), saves split indices and clean dataset.
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

def prepare_splits_and_save(df: pd.DataFrame):
    """Performs stratified train (70%), val (15%), test (15%) split and saves artifacts."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label"]
    )
    # 15% val, 15% test from temp
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
    )

    df.loc[train_df.index, "split"] = "train"
    df.loc[val_df.index, "split"] = "val"
    df.loc[test_df.index, "split"] = "test"

    # Save processed dataset to Parquet
    df.to_parquet(PROCESSED_PARQUET, index=False)
    logging.info(f"Saved processed dataset to {PROCESSED_PARQUET}")

    # Save split indices
    split_indices = {
        "train": train_df.index.tolist(),
        "val": val_df.index.tolist(),
        "test": test_df.index.tolist(),
        "seed": 42,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df)
    }

    with open(SPLIT_INDICES_JSON, "w") as f:
        json.dump(split_indices, f, indent=2)

    np.save(SPLIT_INDICES_NPY, split_indices, allow_pickle=True)
    logging.info(f"Saved split indices to {SPLIT_INDICES_JSON} and {SPLIT_INDICES_NPY}")

    # Sample 50 random rows for manual audit
    audit_samples = df.sample(n=min(50, len(df)), random_state=42).to_dict(orient="records")
    with open(AUDIT_JSON, "w") as f:
        json.dump(audit_samples, f, indent=2)
    logging.info(f"Saved 50 audit samples to {AUDIT_JSON}")

if __name__ == "__main__":
    df = load_and_parse_raw_data()
    prepare_splits_and_save(df)
