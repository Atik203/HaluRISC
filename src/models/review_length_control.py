"""Length-shortcut control for the 2026-10-20 review (P1, WS-B).

Builds the same 247-rows-per-bin subset used in the paper, then adds what the
review asked for:

  1. per-bin positives, negatives, prevalence, and class-length SMD;
  2. the length-only baseline on exactly the same rows;
  3. a matched subset with equal positives and negatives at each exact length;
  4. paired group-bootstrap confidence intervals for the F1/AUROC differences.

No retraining of EC-XGB. The length-only model is fit on the training split
with the same recipe as review_extras.py (standardized log n_chars/n_words).

Run (repo root):
  python src/models/review_length_control.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.models.config import DATA_PROCESSED, RESULTS_DIR
from src.models.train_pipeline import ece

ROOT = Path(__file__).resolve().parents[2]
B6 = RESULTS_DIR / "b6" / "b6_predictions.parquet"
B2 = RESULTS_DIR / "b2" / "b2_predictions.parquet"
FEATURES = DATA_PROCESSED / "features_full.parquet"
OUT = RESULTS_DIR / "review"
BINS = [0, 1, 2, 4, 8, 16, 10**9]
LABELS = ["1", "2", "3-4", "5-8", "9-16", "17+"]
SEED = 42
ROWS_PER_BIN = 247
N_BOOT = 1000


def nll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "negatives": int((1 - y).sum()),
        "prevalence": float(y.mean()),
        "ece": float(ece(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "nll": nll(y, p),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "predicted_positive_rate": float(pred.mean()),
    }


def smd(group: pd.DataFrame) -> float:
    pos = group.loc[group["label"] == 1, "n_words"].to_numpy(dtype=float)
    neg = group.loc[group["label"] == 0, "n_words"].to_numpy(dtype=float)
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    pooled = np.sqrt((pos.var(ddof=1) + neg.var(ddof=1)) / 2.0)
    return float((pos.mean() - neg.mean()) / pooled) if pooled > 0 else 0.0


def bin_stats(frame: pd.DataFrame) -> dict:
    out = {}
    for label in LABELS:
        g = frame[frame["bin"] == label]
        if g.empty:
            out[label] = {"n": 0}
            continue
        out[label] = {
            "n": int(len(g)),
            "positives": int(g["label"].sum()),
            "negatives": int((1 - g["label"]).sum()),
            "prevalence": float(g["label"].mean()),
            "mean_words": float(g["n_words"].mean()),
            "median_words": float(g["n_words"].median()),
            "min_words": int(g["n_words"].min()),
            "max_words": int(g["n_words"].max()),
            "smd": smd(g),
        }
    return out


def group_bootstrap_delta(y, p_model, p_base, groups, n_boot=N_BOOT, seed=SEED):
    y = np.asarray(y, dtype=int)
    p_model = np.asarray(p_model, dtype=float)
    p_base = np.asarray(p_base, dtype=float)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    index = {g: np.where(groups == g)[0] for g in unique}
    rng = np.random.default_rng(seed)
    f1_deltas, auc_deltas = [], []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index[g] for g in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        f1_deltas.append(
            f1_score(y[idx], (p_model[idx] >= 0.5).astype(int), zero_division=0)
            - f1_score(y[idx], (p_base[idx] >= 0.5).astype(int), zero_division=0)
        )
        auc_deltas.append(
            roc_auc_score(y[idx], p_model[idx]) - roc_auc_score(y[idx], p_base[idx])
        )
    return {
        "f1_delta": float(np.mean(f1_deltas)),
        "f1_ci95": [float(np.percentile(f1_deltas, 2.5)), float(np.percentile(f1_deltas, 97.5))],
        "auroc_delta": float(np.mean(auc_deltas)),
        "auroc_ci95": [float(np.percentile(auc_deltas, 2.5)), float(np.percentile(auc_deltas, 97.5))],
        "n_resamples": len(f1_deltas),
    }


def exact_match(frame: pd.DataFrame) -> pd.DataFrame:
    """Equal positives/negatives at each exact word count inside each bin."""
    rng = np.random.default_rng(SEED)
    keep = []
    for label in LABELS:
        gbin = frame[frame["bin"] == label]
        for _, glen in gbin.groupby("n_words", observed=True):
            pos = glen.loc[glen["label"] == 1, "sample_id"].to_numpy()
            neg = glen.loc[glen["label"] == 0, "sample_id"].to_numpy()
            k = min(len(pos), len(neg))
            if k == 0:
                continue
            pos = rng.permutation(pos)[:k]
            neg = rng.permutation(neg)[:k]
            keep.extend(pos.tolist())
            keep.extend(neg.tolist())
    return frame[frame["sample_id"].isin(keep)].copy()


def main() -> None:
    features = pd.read_parquet(
        FEATURES, columns=["sample_id", "item_idx", "label", "split", "n_words", "n_chars"]
    )
    b6 = pd.read_parquet(B6)
    b6 = b6[(b6["variant"] == "m3") & (b6["dataset"] == "halueval_test")]
    frame = b6.merge(
        features.drop(columns=["label"]), on="sample_id", how="left", validate="many_to_one"
    )
    if frame["n_words"].isna().any():
        raise ValueError("missing n_words after merge")
    frame["bin"] = pd.cut(frame["n_words"], bins=BINS, labels=LABELS, right=True)

    one = frame[frame["seed"] == SEED].copy()
    full_bins = one.groupby("bin", observed=True).size().to_dict()
    n_min = int(min(full_bins.values()))
    if n_min != ROWS_PER_BIN:
        raise ValueError(f"expected {ROWS_PER_BIN} rows in the smallest bin, got {n_min}")
    picks = []
    for label in LABELS:
        g = one[one["bin"] == label]
        picks.extend(g["sample_id"].sample(n=ROWS_PER_BIN, random_state=SEED).tolist())
    subset = frame[frame["sample_id"].isin(picks)].copy()
    subset_one = subset[subset["seed"] == SEED].copy()

    # Length-only baseline, trained once on the training split.
    train = features[features["split"] == "train"]
    cols = ["n_chars", "n_words"]
    Xtr = np.column_stack([np.log1p(train[c].to_numpy(dtype=float)) for c in cols])
    scaler = StandardScaler().fit(Xtr)
    length_model = LogisticRegression(max_iter=2000).fit(
        scaler.transform(Xtr), train["label"].to_numpy()
    )
    X_sub = np.column_stack([np.log1p(subset_one[c].to_numpy(dtype=float)) for c in cols])
    length_p = length_model.predict_proba(scaler.transform(X_sub))[:, 1]

    y = subset_one["label"].to_numpy()
    length_metrics = metrics(y, length_p)

    # EC-XGB per seed on exactly the same rows.
    per_seed = {}
    for seed, g in subset.groupby("seed"):
        per_seed[str(int(seed))] = metrics(g["label"].to_numpy(), g["score"].to_numpy())
    ec_mean = {
        k: float(np.mean([per_seed[s][k] for s in per_seed]))
        for k in ("ece", "brier", "nll", "auroc", "f1", "precision", "recall",
                  "predicted_positive_rate")
    }

    # Standard XGBoost on the same rows, for reference.
    b2 = pd.read_parquet(B2)
    b2 = b2[(b2["model"] == "xgboost") & (b2["split"] == "test")]
    b2 = b2.merge(subset[["sample_id", "seed"]].drop_duplicates(), on=["sample_id", "seed"])
    std_per_seed = {
        str(int(seed)): metrics(g["label"].to_numpy(), g["score"].to_numpy())
        for seed, g in b2.groupby("seed")
    }
    std_mean = {
        k: float(np.mean([std_per_seed[s][k] for s in std_per_seed]))
        for k in ("ece", "brier", "nll", "auroc", "f1", "precision", "recall",
                  "predicted_positive_rate")
    }

    # Paired bootstrap on seed 42: EC-XGB minus length-only.
    delta = group_bootstrap_delta(
        y, subset_one["score"].to_numpy(), length_p, subset_one["item_idx"].to_numpy()
    )
    delta_auc_point = ec_mean["auroc"] - length_metrics["auroc"]
    delta_f1_point = ec_mean["f1"] - length_metrics["f1"]

    # Matched subset: exact word-count matching, equal positives/negatives.
    matched_one = exact_match(subset_one)
    matched_bins = bin_stats(matched_one)
    X_mat = np.column_stack([np.log1p(matched_one[c].to_numpy(dtype=float)) for c in cols])
    matched_length_p = length_model.predict_proba(scaler.transform(X_mat))[:, 1]
    matched_y = matched_one["label"].to_numpy()
    matched_length = metrics(matched_y, matched_length_p)

    matched_all = subset[subset["sample_id"].isin(matched_one["sample_id"])]
    matched_ec = {}
    for seed, g in matched_all.groupby("seed"):
        matched_ec[str(int(seed))] = metrics(g["label"].to_numpy(), g["score"].to_numpy())
    matched_ec_mean = {
        k: float(np.mean([matched_ec[s][k] for s in matched_ec]))
        for k in ("ece", "brier", "nll", "auroc", "f1", "precision", "recall",
                  "predicted_positive_rate")
    }
    matched_seed = (
        matched_all[matched_all["seed"] == SEED]
        .set_index("sample_id")
        .loc[matched_one["sample_id"]]
    )
    matched_delta = group_bootstrap_delta(
        matched_y,
        matched_seed["score"].to_numpy(),
        matched_length_p,
        matched_one["item_idx"].to_numpy(),
    )

    bundle = {
        "meta": {
            "source": "review_length_control.py",
            "full_test_rows": int(len(one)),
            "rows_per_bin": ROWS_PER_BIN,
            "subset_rows": int(len(subset_one)),
            "seed_for_selection": SEED,
            "bins": LABELS,
            "length_only_model": "logistic regression on standardized log n_chars/n_words, trained on the train split",
            "ec_xgb": "b6 m3, seeds 42/123/456",
            "threshold": 0.5,
        },
        "full_test_bins": full_bins,
        "subset_bins": bin_stats(subset_one),
        "length_only_subset": length_metrics,
        "ec_xgb_subset": {"per_seed": per_seed, "mean": ec_mean},
        "standard_xgb_subset": {"per_seed": std_per_seed, "mean": std_mean},
        "delta_ec_xgb_minus_length_only_subset": {
            **delta,
            "f1_point": float(delta_f1_point),
            "auroc_point": float(delta_auc_point),
        },
        "matched": {
            "rows": int(len(matched_one)),
            "positives": int(matched_y.sum()),
            "negatives": int((1 - matched_y).sum()),
            "prevalence": float(matched_y.mean()),
            "bins": matched_bins,
            "length_only": matched_length,
            "ec_xgb": {"per_seed": matched_ec, "mean": matched_ec_mean},
            "delta_ec_xgb_minus_length_only": {
                **matched_delta,
                "f1_point": float(matched_ec_mean["f1"] - matched_length["f1"]),
                "auroc_point": float(matched_ec_mean["auroc"] - matched_length["auroc"]),
            },
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "length_control.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    rows = []
    for split_name, stats in (("full", bin_stats(one)), ("subset", bundle["subset_bins"]),
                              ("matched", matched_bins)):
        for label, s in stats.items():
            if s.get("n", 0):
                rows.append({"split": split_name, "bin": label, **s})
    pd.DataFrame(rows).to_csv(OUT / "length_control_bins.csv", index=False)

    summary = []
    for name, block in (
        ("length_only_subset", length_metrics),
        ("ec_xgb_subset", ec_mean),
        ("standard_xgb_subset", std_mean),
        ("length_only_matched", matched_length),
        ("ec_xgb_matched", matched_ec_mean),
    ):
        summary.append({"model": name, **{k: v for k, v in block.items() if isinstance(v, (int, float))}})
    pd.DataFrame(summary).to_csv(OUT / "length_control_summary.csv", index=False)

    pd.DataFrame(
        {
            "sample_id": subset_one["sample_id"].to_numpy(),
            "bin": subset_one["bin"].astype(str).to_numpy(),
            "n_words": subset_one["n_words"].to_numpy(),
            "label": subset_one["label"].to_numpy(),
        }
    ).to_csv(OUT / "length_control_subset_ids.csv", index=False)

    pd.DataFrame(
        {
            "sample_id": matched_one["sample_id"].to_numpy(),
            "bin": matched_one["bin"].astype(str).to_numpy(),
            "n_words": matched_one["n_words"].to_numpy(),
            "label": matched_one["label"].to_numpy(),
        }
    ).to_csv(OUT / "length_control_matched_ids.csv", index=False)

    print("Subset bins:")
    print(pd.DataFrame(bundle["subset_bins"]).T.to_string())
    print("\nModels on the subset:")
    print(pd.DataFrame(summary).to_string(index=False))
    print("\nMatched rows:", len(matched_one))
    print("\nEC-XGB minus length-only (seed 42, group bootstrap):")
    print(json.dumps(bundle["delta_ec_xgb_minus_length_only_subset"], indent=2))
    print("\nMatched delta:")
    print(json.dumps(bundle["matched"]["delta_ec_xgb_minus_length_only"], indent=2))
    print("\nSaved:", OUT / "length_control.json")


if __name__ == "__main__":
    main()
