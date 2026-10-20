"""Length-shortcut controls with repeated matched sampling (2026-10-21 review, item 2).

The single exact-length-matched subset in `review_length_control.py` is a valid
control but rests on one draw of 79 pairs. This script adds two stronger tests
that use the same predictions and the same frozen split:

1. A length-conditional AUROC over the full test set. Exact word counts that
   contain both classes form strata, AUROC is computed inside each stratum, and
   the strata are combined by the number of positive-negative pairs. Length is
   constant inside a stratum, so a length-only model has AUROC 0.5 there by
   construction and any value above 0.5 is ranking skill beyond length.
2. Repeated exact-length matching. The matching is repeated 200 times with
   different sampling seeds, and the AUROC gap against the length-only control
   is reported as mean, standard deviation, and a 2.5-97.5 percentile band.

Run from the repository root:

  python src/models/review_length_matched_repeat.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
RES = ROOT / "artifacts" / "results"
OUT = RES / "review"

SEEDS = (42, 123, 456)
N_RESAMPLE = 200
LENGTH_COLS = ("n_words", "n_chars")


def load_test() -> pd.DataFrame:
    """HaluEval test rows with EC-XGB m3 scores for every seed."""
    feats = pd.read_parquet(DATA / "features_full.parquet", columns=["sample_id", "label", "n_words", "n_chars", "split"])
    preds = pd.read_parquet(RES / "b6" / "b6_predictions.parquet")
    preds = preds[(preds["variant"] == "m3") & (preds["dataset"] == "halueval_test")]
    wide = preds.pivot_table(index="sample_id", columns="seed", values="score")
    wide.columns = [f"score_{int(c)}" for c in wide.columns]
    test = feats[feats["split"] == "test"].merge(wide, left_on="sample_id", right_index=True, how="inner")
    return test.reset_index(drop=True)


def length_baseline() -> tuple[np.ndarray, np.ndarray]:
    """Logistic regression on the two log-scaled length features, trained on train."""
    feats = pd.read_parquet(DATA / "features_full.parquet", columns=["sample_id", "label", "n_words", "n_chars", "split"])
    train = feats[feats["split"] == "train"].dropna(subset=list(LENGTH_COLS))
    test = feats[feats["split"] == "test"].dropna(subset=list(LENGTH_COLS))

    def matrix(frame):
        return np.column_stack([np.log1p(frame[c].to_numpy(dtype=float)) for c in LENGTH_COLS])

    scaler = StandardScaler().fit(matrix(train))
    model = LogisticRegression(max_iter=2000, random_state=42).fit(scaler.transform(matrix(train)), train["label"])
    order = test.set_index("sample_id")
    return model.predict_proba(scaler.transform(matrix(test)))[:, 1], order.index.to_numpy()


def conditional_auroc(frame: pd.DataFrame, score_col: str) -> dict:
    """Pair-weighted AUROC inside exact-length strata that contain both classes."""
    total_pairs = 0.0
    weighted = 0.0
    strata = []
    for words, group in frame.groupby("n_words"):
        n_pos = int((group["label"] == 1).sum())
        n_neg = int((group["label"] == 0).sum())
        if n_pos < 2 or n_neg < 2:
            continue
        auc = float(roc_auc_score(group["label"], group[score_col]))
        pairs = n_pos * n_neg
        weighted += pairs * auc
        total_pairs += pairs
        strata.append({"words": int(words), "positives": n_pos, "negatives": n_neg, "auroc": round(auc, 4)})
    return {"auroc": weighted / total_pairs if total_pairs else float("nan"),
            "n_strata": len(strata),
            "n_pairs": int(total_pairs),
            "strata": strata}


def matched_subset(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Equal positives and negatives at every exact word count."""
    keep = []
    for _, group in frame.groupby("n_words"):
        pos = group.loc[group["label"] == 1, "sample_id"].to_numpy()
        neg = group.loc[group["label"] == 0, "sample_id"].to_numpy()
        k = min(len(pos), len(neg))
        if k == 0:
            continue
        keep.extend(rng.permutation(pos)[:k].tolist())
        keep.extend(rng.permutation(neg)[:k].tolist())
    return frame[frame["sample_id"].isin(keep)]


def main() -> None:
    test = load_test()
    length_p, length_ids = length_baseline()
    length_map = dict(zip(length_ids, length_p))
    test["score_length"] = test["sample_id"].map(length_map)

    # Full-pool matching, no resampling, for the pool-size statement.
    pool = matched_subset(test, np.random.default_rng(42))
    pool_pairs = len(pool) // 2

    record = {
        "meta": {
            "source": "review_length_matched_repeat.py",
            "test_rows": int(len(test)),
            "seeds": list(SEEDS),
            "n_resample": N_RESAMPLE,
            "score": "EC-XGB m3 raw probability, threshold 0.5",
            "length_baseline": "logistic regression on log1p(n_words) and log1p(n_chars), trained on the training split",
            "pool_matched_rows": int(len(pool)),
            "pool_matched_pairs": int(pool_pairs),
        },
        "conditional_auroc": {},
        "repeated_matching": {},
    }

    for seed in SEEDS:
        record["conditional_auroc"][str(seed)] = conditional_auroc(test, f"score_{seed}")

    cond_values = [record["conditional_auroc"][str(s)]["auroc"] for s in SEEDS]
    record["conditional_auroc"]["mean"] = float(np.mean(cond_values))
    record["conditional_auroc"]["std"] = float(np.std(cond_values))
    record["conditional_auroc"]["length_only_expectation"] = 0.5

    gaps, model_aucs, length_aucs, sizes = [], [], [], []
    for draw in range(N_RESAMPLE):
        matched = matched_subset(test, np.random.default_rng(draw))
        sizes.append(len(matched))
        aucs = [roc_auc_score(matched["label"], matched[f"score_{s}"]) for s in SEEDS]
        model_aucs.append(float(np.mean(aucs)))
        length_aucs.append(float(roc_auc_score(matched["label"], matched["score_length"])))
        gaps.append(model_aucs[-1] - length_aucs[-1])

    gaps = np.asarray(gaps)
    record["repeated_matching"] = {
        "n_draws": int(len(gaps)),
        "rows_per_draw": int(np.mean(sizes)),
        "ec_xgb_auroc_mean": float(np.mean(model_aucs)),
        "ec_xgb_auroc_std": float(np.std(model_aucs)),
        "length_only_auroc_mean": float(np.mean(length_aucs)),
        "length_only_auroc_std": float(np.std(length_aucs)),
        "gap_mean": float(gaps.mean()),
        "gap_std": float(gaps.std()),
        "gap_ci95": [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))],
    }

    # Single frozen draw, for continuity with the 158-row number in the paper.
    single = matched_subset(test, np.random.default_rng(42))
    record["single_draw"] = {
        "rows": int(len(single)),
        "ec_xgb_auroc": float(np.mean([roc_auc_score(single["label"], single[f"score_{s}"]) for s in SEEDS])),
        "ec_xgb_f1": float(np.mean([f1_score(single["label"], (single[f"score_{s}"] >= 0.5).astype(int)) for s in SEEDS])),
        "length_only_auroc": float(roc_auc_score(single["label"], single["score_length"])),
        "length_only_f1": float(f1_score(single["label"], (single["score_length"] >= 0.5).astype(int))),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "length_matched_repeat.json").write_text(json.dumps(record, indent=1))

    print(json.dumps({
        "conditional_auroc_mean": round(record["conditional_auroc"]["mean"], 4),
        "conditional_auroc_std": round(record["conditional_auroc"]["std"], 4),
        "strata": record["conditional_auroc"]["42"]["n_strata"],
        "pool_rows": record["meta"]["pool_matched_rows"],
        "repeat_rows": record["repeated_matching"]["rows_per_draw"],
        "ec_xgb": round(record["repeated_matching"]["ec_xgb_auroc_mean"], 4),
        "ec_xgb_sd": round(record["repeated_matching"]["ec_xgb_auroc_std"], 4),
        "length_only": round(record["repeated_matching"]["length_only_auroc_mean"], 4),
        "length_only_sd": round(record["repeated_matching"]["length_only_auroc_std"], 4),
        "gap": round(record["repeated_matching"]["gap_mean"], 4),
        "gap_sd": round(record["repeated_matching"]["gap_std"], 4),
        "gap_ci95": [round(x, 4) for x in record["repeated_matching"]["gap_ci95"]],
    }, indent=1))


if __name__ == "__main__":
    main()
