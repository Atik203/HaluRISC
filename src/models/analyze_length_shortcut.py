"""
Length-binned error analysis of the deployed XGBoost model (B5 follow-up).

Tests whether the brevity/overlap shortcut observed in the B5.5 expert audit
generalizes: are false positives concentrated in the shortest answer bins?

Reads : artifacts/results/b2/b2_predictions.parquet (model=xgboost, split=test)
        data/processed/features_full.parquet (n_words)
Writes: artifacts/results/b5/b5_length_error_analysis.csv
        artifacts/results/b5/b5_length_error_analysis.json

Run (repo root, .venv):
  python src/models/analyze_length_shortcut.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "artifacts" / "results" / "b2" / "b2_predictions.parquet"
FEATURES = ROOT / "data" / "processed" / "features_full.parquet"
OUT_DIR = ROOT / "artifacts" / "results" / "b5"
OUT_CSV = OUT_DIR / "b5_length_error_analysis.csv"
OUT_JSON = OUT_DIR / "b5_length_error_analysis.json"

MODEL = "xgboost"
SPLIT = "test"
THRESHOLD = 0.5
BINS = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, None)]
BIN_LABELS = ["1", "2", "3-4", "5-8", "9-16", "17+"]


def main() -> None:
    pred = pd.read_parquet(PREDICTIONS)
    feat = pd.read_parquet(FEATURES)[["sample_id", "n_words"]]

    df = pred[(pred["model"] == MODEL) & (pred["split"] == SPLIT)].copy()
    df = df.merge(feat, on="sample_id", how="left")
    assert df["n_words"].notna().all(), "missing n_words after merge"

    seeds = sorted(int(s) for s in df["seed"].unique())
    per_seed_rows = []
    for seed in seeds:
        d = df[df["seed"] == seed]
        for label, (lo, hi) in zip(BIN_LABELS, BINS):
            b = d[d["n_words"].between(lo, hi if hi is not None else 10**9)]
            n_neg = int((b["label"] == 0).sum())
            n_pos = int((b["label"] == 1).sum())
            fp = int(((b["pred"] == 1) & (b["label"] == 0)).sum())
            fn = int(((b["pred"] == 0) & (b["label"] == 1)).sum())
            tp = int(((b["pred"] == 1) & (b["label"] == 1)).sum())
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
            per_seed_rows.append(
                {
                    "seed": seed,
                    "bin": label,
                    "rows": len(b),
                    "negatives": n_neg,
                    "positives": n_pos,
                    "fp": fp,
                    "fn": fn,
                    "fp_rate": fp / n_neg if n_neg else float("nan"),
                    "fn_rate": fn / n_pos if n_pos else float("nan"),
                    "f1": f1,
                    "mean_score": float(b["score"].mean()),
                }
            )

    per_seed = pd.DataFrame(per_seed_rows)
    agg = (
        per_seed.groupby("bin", sort=False)
        .agg(
            rows=("rows", "first"),
            negatives=("negatives", "first"),
            positives=("positives", "first"),
            fp=("fp", "mean"),
            fn=("fn", "mean"),
            fp_rate=("fp_rate", "mean"),
            fn_rate=("fn_rate", "mean"),
            f1=("f1", "mean"),
            mean_score=("mean_score", "mean"),
        )
        .reindex(BIN_LABELS)
        .reset_index()
    )
    agg["pos_rate"] = agg["positives"] / agg["rows"]
    total_fp = agg["fp"].sum()
    agg["fp_share"] = agg["fp"] / total_fp if total_fp else np.nan

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUT_CSV, index=False, float_format="%.4f")
    OUT_JSON.write_text(
        json.dumps(
            {
                "model": MODEL,
                "split": SPLIT,
                "threshold": THRESHOLD,
                "seeds": seeds,
                "n_test_rows": int(agg["rows"].sum()),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source_files": [
                    str(PREDICTIONS.relative_to(ROOT)).replace("\\", "/"),
                    str(FEATURES.relative_to(ROOT)).replace("\\", "/"),
                ],
                "bins": agg.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    show = agg.copy()
    for col in ("fp", "fn"):
        show[col] = show[col].round(1)
    for col in ("fp_rate", "fn_rate", "f1", "mean_score", "fp_share", "pos_rate"):
        show[col] = show[col].round(3)
    print(show.to_string(index=False))
    print(f"\nWrote {OUT_CSV.relative_to(ROOT)} and {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
