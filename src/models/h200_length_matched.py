"""H200 appendix: length-matched external control (E2).

The frozen external results carry a known confound: hallucinated answers are
longer, and a length-only baseline already reaches AUROC 0.514 on HaluBench and
0.260 on MedHallu pairs. This script re-evaluates the frozen EC-XGB (m3)
predictions on length-matched slices so the external numbers can be read
without the length shortcut.

Two controls:

  HaluBench  equal-length subsampling. Answers are binned by log1p(n_words)
             within each source; each (source, bin) keeps an equal number of
             PASS and FAIL rows.
  MedHallu   tight-length pairs. Pair length gap is |log1p(n_hall) -
             log1p(n_ground)|; the matched slice keeps pairs at or below the
             median gap, so the two answers in a pair are close in length.

Nothing is retrained. The frozen m3 boosters and frozen per-row MedHallu
predictions are read as-is; the length-only baseline is a logistic regression
on log1p(n_words) fitted on the same 5,034 RAGTruth QA calibration rows used by
``src/models/rebuild_judge.py``.

Outputs
  artifacts/results/h200/length_matched.csv
  artifacts/results/h200/length_matched.json

Run (repo root):
  python src/models/h200_length_matched.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("h200_length_matched")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "artifacts" / "results"
MODELS = ROOT / "artifacts" / "models"
OUT = RESULTS / "h200"
THRESHOLD = 0.5
N_BOOT = 1000
SEEDS = (42, 123, 456)
N_BINS = 8


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(values), len(values))
        vals.append(float(np.mean(values[idx])))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def metrics(y: np.ndarray, s: np.ndarray, threshold: float | None = THRESHOLD) -> dict:
    """F1 at `threshold` (None -> not reported) plus AUROC."""
    auroc = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")
    out = {"n": int(len(y)), "auroc": auroc, "positive_rate": float(np.mean(y))}
    if threshold is None:
        out["f1"] = float("nan")
    else:
        pred = (s >= threshold).astype(int)
        out["f1"] = float(f1_score(y, pred, zero_division=0))
    return out


def fpr_threshold(y: np.ndarray, s: np.ndarray, target: float = 0.05) -> float:
    """Score threshold at a target false-positive rate."""
    neg = np.sort(s[np.asarray(y) == 0])[::-1]
    if len(neg) == 0:
        return THRESHOLD
    k = max(0, int(np.floor(target * len(neg))) - 1)
    return float(neg[min(k, len(neg) - 1)])


def length_baseline() -> tuple[LogisticRegression, float]:
    """Fit log1p(n_words) on the calibration rows and return a 5% FPR threshold."""
    cal = pd.read_parquet(DATA / "calibration_features.parquet", columns=["n_words", "label"])
    lr = LogisticRegression(max_iter=1000)
    x = np.log1p(cal["n_words"].to_numpy(dtype=float)).reshape(-1, 1)
    y = cal["label"].to_numpy(dtype=int)
    lr.fit(x, y)
    thr = fpr_threshold(y, lr.predict_proba(x)[:, 1], 0.05)
    return lr, thr


def length_scores(lr: LogisticRegression, n_words) -> np.ndarray:
    return lr.predict_proba(np.log1p(np.asarray(n_words, dtype=float)).reshape(-1, 1))[:, 1]


# --------------------------------------------------------------------------
# HaluBench
# --------------------------------------------------------------------------
def halubench_frame() -> pd.DataFrame:
    base = pd.read_parquet(DATA / "halubench_base.parquet")
    claims = pd.read_parquet(DATA / "halubench_claim_features.parquet")
    subset = pd.read_parquet(DATA / "halubench_subset.parquet", columns=["sample_id", "label", "source_ds"])
    df = base.merge(claims, on="sample_id", how="left", validate="one_to_one")
    df = df.merge(subset, on="sample_id", how="inner", validate="one_to_one")
    if df[["claim_mean_contradicts", "label"]].isna().any().any():
        raise ValueError("claim features or labels missing after merge")
    df["source_ragtruth"] = 1.0
    return df.reset_index(drop=True)


def score_m3(df: pd.DataFrame) -> dict[int, np.ndarray]:
    names = json.loads((RESULTS / "b6" / "b6_feature_names.json").read_text())["m3"]
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ValueError(f"missing m3 features: {missing}")
    X = df[names].to_numpy(dtype=float)
    out = {}
    for seed in SEEDS:
        path = MODELS / "b6" / f"xgboost_m3_seed_{seed}.joblib"
        if not path.exists():
            raise FileNotFoundError(path)
        out[seed] = joblib.load(path).predict_proba(X)[:, 1]
    return out


def equal_length_subsample(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Keep equal label counts per (source, length bin)."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    d["len_bin"] = d.groupby("source_ds")["n_words"].transform(
        lambda s: pd.qcut(np.log1p(s), q=N_BINS, duplicates="drop", labels=False)
    )
    keep = []
    for (_, _), g in d.groupby(["source_ds", "len_bin"], dropna=True):
        pos = g[g["label"] == 1]
        neg = g[g["label"] == 0]
        k = min(len(pos), len(neg))
        if k == 0:
            continue
        keep.append(pos.sample(k, random_state=int(rng.integers(0, 10**6))))
        keep.append(neg.sample(k, random_state=int(rng.integers(0, 10**6))))
    return pd.concat(keep).sort_index() if keep else d.iloc[0:0]


# --------------------------------------------------------------------------
# MedHallu
# --------------------------------------------------------------------------
def medhallu_frame() -> pd.DataFrame:
    pred = pd.read_parquet(RESULTS / "rebuild" / "p1" / "medhallu_predictions.parquet")
    pred = pred[pred["model"] == "m3"]
    feats = pd.read_parquet(DATA / "medhallu_base_features.parquet", columns=["sample_id", "n_words"])
    return pred.merge(feats, on="sample_id", how="inner", validate="many_to_one")


def pair_accuracy(df: pd.DataFrame) -> dict:
    rows = []
    for seed, g in df.groupby("seed"):
        wide = g.pivot_table(index="pair_id", columns="role", values="score")
        if not {"hallucinated", "grounded"}.issubset(wide.columns):
            continue
        ok = (wide["hallucinated"] > wide["grounded"]).to_numpy(dtype=float)
        rows.append({"seed": int(seed), "n_pairs": int(len(ok)), "pair_accuracy": float(ok.mean()), "correct": ok})
    if not rows:
        return {"pair_accuracy_mean": float("nan"), "per_seed": []}
    return {
        "pair_accuracy_mean": float(np.mean([r["pair_accuracy"] for r in rows])),
        "per_seed": [{"seed": r["seed"], "n_pairs": r["n_pairs"], "pair_accuracy": r["pair_accuracy"]} for r in rows],
        "correct": rows[0]["correct"],
    }


def restrict_pairs(df: pd.DataFrame, max_gap: float) -> pd.DataFrame:
    gap = (
        df.assign(lg=np.log1p(df["n_words"].astype(float)))
        .pivot_table(index="pair_id", columns="role", values="lg")
        .dropna()
    )
    gap["gap"] = (gap["hallucinated"] - gap["grounded"]).abs()
    keep = set(gap[gap["gap"] <= max_gap].index)
    return df[df["pair_id"].isin(keep)]


# --------------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lr, len_thr = length_baseline()
    rows = []

    # ---- HaluBench ----
    hb = halubench_frame()
    m3_scores = score_m3(hb)
    hb = hb.copy()
    hb["len_only"] = length_scores(lr, hb["n_words"])
    hb_matched = equal_length_subsample(hb)

    for tag, frame in (("full", hb), ("length_matched", hb_matched)):
        y = frame["label"].to_numpy(dtype=int)
        rows.append({
            "dataset": "halubench", "slice": tag, "model": "length_only",
            "threshold": float("nan"), "threshold_rule": "not reported (see note)",
            **metrics(y, frame["len_only"].to_numpy(), threshold=None),
        })
        for seed, scores in m3_scores.items():
            s = pd.Series(scores, index=hb.index).loc[frame.index].to_numpy()
            rows.append({
                "dataset": "halubench", "slice": tag, "model": f"ec_xgb_m3_seed{seed}",
                "threshold": THRESHOLD, "threshold_rule": "0.5",
                **metrics(y, s),
            })

    # ---- MedHallu ----
    mh = medhallu_frame()
    mh["len_only"] = length_scores(lr, mh["n_words"])
    gap_median = float(
        (mh.assign(lg=np.log1p(mh["n_words"].astype(float)))
         .pivot_table(index="pair_id", columns="role", values="lg")
         .dropna()
         .pipe(lambda d: (d["hallucinated"] - d["grounded"]).abs())
         .median())
    )
    mh_matched = restrict_pairs(mh, gap_median)

    mh_out = {}
    for tag, frame in (("full", mh), ("length_matched", mh_matched)):
        acc = pair_accuracy(frame)
        mh_out[tag] = {"pair_accuracy_mean": acc["pair_accuracy_mean"], "per_seed": acc["per_seed"], "n_rows": int(len(frame))}
        if "correct" in acc:
            mh_out[tag]["ci95"] = bootstrap_ci(acc["correct"])
        # per-row AUROC on the same slice
        y = frame["label"].to_numpy(dtype=int)
        for seed in SEEDS:
            g = frame[frame["seed"] == seed]
            rows.append({
                "dataset": "medhallu", "slice": tag, "model": f"ec_xgb_m3_seed{seed}",
                "threshold": THRESHOLD, "threshold_rule": "0.5",
                **metrics(g["label"].to_numpy(dtype=int), g["score"].to_numpy()),
            })
        rows.append({
            "dataset": "medhallu", "slice": tag, "model": "length_only",
            "threshold": float("nan"), "threshold_rule": "not reported (see note)",
            **metrics(g["label"].to_numpy(dtype=int), g["len_only"].to_numpy(), threshold=None),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "length_matched.csv", index=False)
    payload = {
        "schema": "h200-length-matched-v1",
        "threshold": THRESHOLD,
        "n_bins_halubench": N_BINS,
        "medhallu_pair_gap_median_log": gap_median,
        "medhallu": mh_out,
        "rows": rows,
        "note": (
            "Frozen m3 boosters and frozen MedHallu predictions are read as-is; nothing is "
            "retrained. The length-only baseline is log1p(n_words) logistic regression fitted "
            "on the 5,034 RAGTruth QA calibration rows. F1 is not reported for that "
            "baseline because its probability scale does not transfer across corpora with "
            "very different answer-length distributions; AUROC is the scale-free comparison."
        ),
    }
    (OUT / "length_matched.json").write_text(json.dumps(payload, indent=2, default=float))

    logger.info("wrote %s", OUT / "length_matched.csv")
    view = df.groupby(["dataset", "slice", "model"])[["f1", "auroc"]].mean().round(4)
    print(view.to_string())
    print("medhallu pair accuracy:", {k: round(v["pair_accuracy_mean"], 4) for k, v in mh_out.items()})


if __name__ == "__main__":
    main()
