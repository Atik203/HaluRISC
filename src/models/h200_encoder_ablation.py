"""H200 appendix: encoder-sensitivity ablation (E1).

Compares EC-XGB variants trained on features from three encoder configurations:

  base       cross-encoder/nli-deberta-v3-base   + all-MiniLM-L6-v2   (deployed stack)
  nli-large  cross-encoder/nli-deberta-v3-large  + all-MiniLM-L6-v2
  bge-base   cross-encoder/nli-deberta-v3-base   + BAAI/bge-base-en-v1.5

Inputs are the per-tag B6 runs written by ``run_b6_modified.py`` under
``artifacts/results/h200/b6_<tag>/``. The script only reads those files;
it never retrains and never touches the frozen ``b6`` results.

Outputs
  artifacts/results/h200/encoder_ablation.csv
  artifacts/results/h200/encoder_ablation.json

Run (repo root):
  python src/models/h200_encoder_ablation.py
"""

from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mcnemar

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("h200_encoder_ablation")

ROOT = Path(__file__).resolve().parents[2]
H200 = ROOT / "artifacts" / "results" / "h200"
FROZEN = ROOT / "artifacts" / "results" / "b6" / "b6_model_comparison.csv"
THRESHOLD = 0.5
DATASET = "halueval_test"

TAGS = ("base", "nli-large", "bge-base")
LABELS = {
    "base": "DeBERTa-v3-base + MiniLM-L6 (deployed)",
    "nli-large": "DeBERTa-v3-large + MiniLM-L6",
    "bge-base": "DeBERTa-v3-base + BGE-base",
}
# m2 isolates the 34-feature model (26 base + 8 claim) so the external
# RAGTruth features are not a confound in the encoder comparison.
VARIANT = "m2"


def load_tag(tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = H200 / f"b6_{tag}"
    comp = d / "b6_model_comparison.csv"
    pred = d / "b6_predictions.parquet"
    if not comp.exists() or not pred.exists():
        raise FileNotFoundError(f"missing results for tag={tag}: {d}")
    return pd.read_csv(comp, index_col=0), pd.read_parquet(pred)


def per_seed_frame(tag: str, variant: str = VARIANT) -> pd.DataFrame:
    _, pred = load_tag(tag)
    sub = pred[(pred["variant"] == variant) & (pred["dataset"] == DATASET)]
    if sub.empty:
        raise ValueError(f"no {variant} rows for tag={tag}")
    return sub.sort_values(["seed", "sample_id"]).reset_index(drop=True)


def mcnemar_pair(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Exact-free McNemar with continuity correction on paired rows per seed."""
    rows = []
    for seed in sorted(set(a["seed"]) & set(b["seed"])):
        sa = a[a["seed"] == seed].sort_values("sample_id")
        sb = b[b["seed"] == seed].sort_values("sample_id")
        if not sa["sample_id"].equals(sb["sample_id"]):
            raise ValueError("sample_id mismatch between runs")
        y = sa["label"].to_numpy()
        ok_a = (sa["score"].to_numpy() >= THRESHOLD) == y
        ok_b = (sb["score"].to_numpy() >= THRESHOLD) == y
        only_a = int(np.sum(ok_a & ~ok_b))
        only_b = int(np.sum(~ok_a & ok_b))
        p = float(mcnemar([[0, only_a], [only_b, 0]], exact=False, correction=True).pvalue)
        rows.append({"seed": int(seed), "only_a_correct": only_a, "only_b_correct": only_b, "pvalue": p})
    return {
        "per_seed": rows,
        "median_pvalue": float(np.median([r["pvalue"] for r in rows])),
        "total_only_a_correct": int(sum(r["only_a_correct"] for r in rows)),
        "total_only_b_correct": int(sum(r["only_b_correct"] for r in rows)),
    }


def main() -> None:
    H200.mkdir(parents=True, exist_ok=True)

    table = []
    for tag in TAGS:
        comp, _ = load_tag(tag)
        if VARIANT not in comp.index:
            raise ValueError(f"{VARIANT} missing from {tag} comparison")
        row = comp.loc[VARIANT]
        table.append(
            {
                "tag": tag,
                "encoder": LABELS[tag],
                "variant": VARIANT,
                "n_features": float(row["n_features"]),
                "precision_mean": float(row["precision_mean"]),
                "recall_mean": float(row["recall_mean"]),
                "f1_mean": float(row["f1_mean"]),
                "f1_std": float(row["f1_std"]),
                "auroc_mean": float(row["auroc_mean"]),
                "auroc_std": float(row["auroc_std"]),
                "mcc_mean": float(row["mcc_mean"]),
                "ece_mean": float(row["ece_mean"]),
                "brier_mean": float(row["brier_mean"]),
            }
        )
    df = pd.DataFrame(table)

    base = df[df["tag"] == "base"].iloc[0]
    df["delta_f1_vs_base"] = df["f1_mean"] - base["f1_mean"]
    df["delta_auroc_vs_base"] = df["auroc_mean"] - base["auroc_mean"]

    frames = {tag: per_seed_frame(tag) for tag in TAGS}
    stats = {}
    for a, b in combinations(TAGS, 2):
        stats[f"{a}_vs_{b}"] = mcnemar_pair(frames[a], frames[b])

    frozen_row = None
    if FROZEN.exists():
        comp = pd.read_csv(FROZEN, index_col=0)
        if "m2" in comp.index:
            frozen_row = {
                "variant": "m2",
                "f1_mean": float(comp.loc["m2", "f1_mean"]),
                "auroc_mean": float(comp.loc["m2", "auroc_mean"]),
                "note": "frozen artifacts/results/b6/b6_model_comparison.csv (tuned)",
            }

    payload = {
        "dataset": DATASET,
        "variant": VARIANT,
        "threshold": THRESHOLD,
        "seeds": sorted(int(s) for s in frames["base"]["seed"].unique()),
        "runs": table,
        "frozen_reference": frozen_row,
        "mcnemar": stats,
        "note": (
            "Encoder ablation only. Features and hyperparameter search are identical across "
            "tags; the sole change is the NLI checkpoint or the embedding checkpoint. "
            "The deployed model (frozen b6-ec-xgb-v1.0) is unchanged."
        ),
    }

    df.to_csv(H200 / "encoder_ablation.csv", index=False)
    (H200 / "encoder_ablation.json").write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", H200 / "encoder_ablation.csv")
    print(df[["tag", "f1_mean", "f1_std", "auroc_mean", "delta_f1_vs_base"]].to_string(index=False))
    for k, v in stats.items():
        print(k, "median p =", f"{v['median_pvalue']:.4g}")


if __name__ == "__main__":
    main()
