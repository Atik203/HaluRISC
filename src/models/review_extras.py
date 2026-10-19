"""Review-response analyses computed from saved artifacts (no retraining).

X1  length-only baseline (logistic regression on log n_chars / log n_words)
X2  length-balanced EC-XGB evaluation (equal rows per answer-length bin)
X3  target-domain prevalence + threshold re-selected on the calibration split
X4  calibration label-budget curve (Platt on 1-100 percent of the 5,034 rows)
X5  group-aware bootstrap for the b2 XGBoost F1/AUROC confidence interval
X6  cross-domain confidence intervals (b3 standard model + b6 EC-XGB)

Writes artifacts/results/extras/review_extras.json. Never modifies the frozen
artifacts.

Run from the repo root:

  .venv/Scripts/python.exe src/models/review_extras.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.run_b4_calibration_shift import apply_calibrator, fit_calibrator  # noqa: E402

OUT = ROOT / "artifacts" / "results" / "extras"
BINS = [0, 1, 2, 4, 8, 16, 10**9]
BIN_LABELS = ["1", "2", "3-4", "5-8", "9-16", "17+"]


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def x1_length_only() -> dict:
    feat = pd.read_parquet(ROOT / "data" / "processed" / "features_full.parquet")
    tr = feat[feat["split"] == "train"]
    te = feat[feat["split"] == "test"]
    cols = ["n_chars", "n_words"]
    Xtr = np.column_stack([np.log1p(tr[c].to_numpy(dtype=float)) for c in cols])
    Xte = np.column_stack([np.log1p(te[c].to_numpy(dtype=float)) for c in cols])
    scaler = StandardScaler().fit(Xtr)
    model = LogisticRegression(max_iter=2000).fit(scaler.transform(Xtr), tr["label"].to_numpy())
    p = model.predict_proba(scaler.transform(Xte))[:, 1]
    y = te["label"].to_numpy()
    pred = (p >= 0.5).astype(int)
    return {
        "features": cols,
        "model": "logistic regression on standardized log lengths",
        "precision": float(y[pred == 1].mean()) if pred.any() else 0.0,
        "recall": float(pred[y == 1].mean()),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "note": "single deterministic fit; the 0.5 threshold is used for F1",
    }


def x2_length_balanced() -> dict:
    b6 = pd.read_parquet(ROOT / "artifacts" / "results" / "b6" / "b6_predictions.parquet")
    feat = pd.read_parquet(ROOT / "data" / "processed" / "features_full.parquet")[["sample_id", "n_words"]]
    h = b6[(b6["variant"] == "m3") & (b6["dataset"] == "halueval_test")].merge(feat, on="sample_id", how="left")
    h["bin"] = pd.cut(h["n_words"], bins=BINS, labels=BIN_LABELS, right=True)
    one = h[h["seed"] == 42]
    counts = one.groupby("bin", observed=True)["sample_id"].nunique()
    n_min = int(counts.min())
    picks = []
    for _, group in one.groupby("bin", observed=True):
        picks.extend(group["sample_id"].sample(n=n_min, random_state=42).tolist())
    sub = h[h["sample_id"].isin(picks)]
    rows = {}
    for seed, d in sub.groupby("seed"):
        y = d["label"].to_numpy()
        p = d["score"].to_numpy()
        pred = (p >= 0.5).astype(int)
        rows[int(seed)] = {
            "f1": float(f1_score(y, pred, zero_division=0)),
            "auroc": float(roc_auc_score(y, p)),
            "precision": float(y[pred == 1].mean()) if pred.any() else 0.0,
            "recall": float(pred[y == 1].mean()),
        }
    f1s = [v["f1"] for v in rows.values()]
    aucs = [v["auroc"] for v in rows.values()]
    return {
        "bin_counts_full": {str(k): int(v) for k, v in counts.items()},
        "rows_per_bin_sampled": n_min,
        "n_rows": int(n_min * len(counts)),
        "per_seed": {str(k): v for k, v in rows.items()},
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "auroc_mean": float(np.mean(aucs)),
        "auroc_std": float(np.std(aucs)),
    }


def x3_threshold_retune() -> dict:
    cal = pd.read_parquet(ROOT / "artifacts" / "results" / "b4" / "_stages" / "qa_cal_clean.parquet")
    test = pd.read_parquet(ROOT / "artifacts" / "results" / "b4" / "b4_predictions.parquet")
    test = test[(test["subset"] == "ragtruth_qa_test") & (test["method"] == "raw")]
    cal_s, cal_y = cal["score_42"].to_numpy(), cal["label"].to_numpy()
    te_s, te_y = test["score"].to_numpy(), test["label"].to_numpy()
    platt = fit_calibrator("platt", cal_s, cal_y)
    cal_p = apply_calibrator("platt", platt, cal_s)
    te_p = apply_calibrator("platt", platt, te_s)
    best = {"threshold": None, "val_recall": -1.0, "val_fpr": 1.0}
    for t in np.unique(np.round(cal_p, 6)):
        pred = cal_p >= t
        fp = int((pred & (cal_y == 0)).sum())
        tn = int((~pred & (cal_y == 0)).sum())
        tp = int((pred & (cal_y == 1)).sum())
        fn = int((~pred & (cal_y == 1)).sum())
        fpr = fp / max(fp + tn, 1)
        rec = tp / max(tp + fn, 1)
        if fpr <= 0.05 and (rec > best["val_recall"] or (rec == best["val_recall"] and best["threshold"] is None)):
            best = {"threshold": float(t), "val_recall": float(rec), "val_fpr": float(fpr)}
    pred = te_p >= best["threshold"]
    tp = int((pred & (te_y == 1)).sum())
    fp = int((pred & (te_y == 0)).sum())
    fn = int((~pred & (te_y == 1)).sum())
    tn = int((~pred & (te_y == 0)).sum())
    return {
        "positive_prevalence_test": float(te_y.mean()),
        "positive_prevalence_calibration": float(cal_y.mean()),
        "ece_raw_test": ece(te_y, te_s),
        "ece_target_platt_test": ece(te_y, te_p),
        "brier_target_platt_test": brier(te_y, te_p),
        "max_calibrated_score_test": float(te_p.max()),
        "share_above_0.5_test": float((te_p >= 0.5).mean()),
        "selected_threshold": best["threshold"],
        "calibration_fpr_at_threshold": best["val_fpr"],
        "calibration_recall_at_threshold": best["val_recall"],
        "test": {
            "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1)),
            "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "fpr": float(fp / max(fp + tn, 1)),
            "predicted_positive_rate": float(pred.mean()),
        },
    }


def x4_label_budget() -> dict:
    cal = pd.read_parquet(ROOT / "artifacts" / "results" / "b4" / "_stages" / "qa_cal_clean.parquet")
    test = pd.read_parquet(ROOT / "artifacts" / "results" / "b4" / "b4_predictions.parquet")
    test = test[(test["subset"] == "ragtruth_qa_test") & (test["method"] == "raw")]
    cal_s, cal_y = cal["score_42"].to_numpy(), cal["label"].to_numpy()
    te_s, te_y = test["score"].to_numpy(), test["label"].to_numpy()
    rng = np.random.default_rng(7)
    out = {}
    for frac in (0.01, 0.05, 0.10, 0.25, 0.50, 1.00):
        n = int(round(frac * len(cal_s)))
        idx = rng.choice(len(cal_s), size=n, replace=False)
        calib = fit_calibrator("platt", cal_s[idx], cal_y[idx])
        p = apply_calibrator("platt", calib, te_s)
        out[f"{frac:.2f}"] = {
            "n_labeled_rows": n,
            "ece": ece(te_y, p),
            "brier": brier(te_y, p),
        }
    return out


def x5_group_bootstrap_b2() -> dict:
    b2 = pd.read_parquet(ROOT / "artifacts" / "results" / "b2" / "b2_predictions.parquet")
    d = b2[(b2["model"] == "xgboost") & (b2["split"] == "test") & (b2["seed"] == 42)]
    y = d["label"].to_numpy()
    pred = d["pred"].to_numpy()
    score = d["score"].to_numpy()
    groups = d["item_idx"].to_numpy()
    index = {}
    for g, i in zip(groups, range(len(groups))):
        index.setdefault(g, []).append(i)
    keys = list(index.keys())
    rng = np.random.default_rng(42)
    f1s, aucs = [], []
    for _ in range(1000):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        rows = np.concatenate([index[keys[k]] for k in pick])
        f1s.append(f1_score(y[rows], pred[rows], zero_division=0))
        if len(np.unique(y[rows])) > 1:
            aucs.append(roc_auc_score(y[rows], score[rows]))
    return {
        "n_groups": len(keys),
        "n_resamples": 1000,
        "f1_point": float(f1_score(y, pred, zero_division=0)),
        "f1_ci95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
        "auroc_point": float(roc_auc_score(y, score)),
        "auroc_ci95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
        "method": "source-group bootstrap over item_idx",
    }


def boot_external(y: np.ndarray, p: np.ndarray, groups: np.ndarray, n: int = 1000, seed: int = 42) -> dict:
    index = {}
    for g, i in zip(groups, range(len(y))):
        index.setdefault(g, []).append(i)
    keys = list(index.keys())
    rng = np.random.default_rng(seed)
    f1s, aucs = [], []
    for _ in range(n):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        rows = np.concatenate([index[keys[k]] for k in pick])
        pred = (p[rows] >= 0.5).astype(int)
        f1s.append(f1_score(y[rows], pred, zero_division=0))
        if len(np.unique(y[rows])) > 1:
            aucs.append(roc_auc_score(y[rows], p[rows]))
    return {
        "n_resamples": n,
        "groups": len(keys),
        "f1_ci95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
        "auroc_ci95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
    }


def x6_cross_domain_cis() -> dict:
    out = {}
    # b3 standard-model intervals already frozen in the artifacts
    b3 = json.loads((ROOT / "artifacts" / "results" / "b3" / "b3_bootstrap_cis.json").read_text(encoding="utf-8"))
    for key in ("ragtruth_qa_test", "ragtruth_all", "faithbench", "ragtruth_task_summarization",
                "ragtruth_task_data_to_text"):
        if key in b3:
            out[f"b3_{key}"] = {"f1_ci95": b3[key]["f1_ci"], "auroc_ci95": b3[key]["auroc_ci"]}
    # EC-XGB (b6 m3, seed 42) group-aware intervals
    b6 = pd.read_parquet(ROOT / "artifacts" / "results" / "b6" / "b6_predictions.parquet")
    ext = pd.read_parquet(ROOT / "data" / "processed" / "external_features.parquet")
    rag = ext[ext["source_dataset"] == "ragtruth"]
    faith = ext[ext["source_dataset"] == "faithbench"]
    frames = {
        "ragtruth_qa_test": rag[(rag["task"] == "qa") & (rag["split"] == "test")],
        "ragtruth_all_test": rag[rag["split"] == "test"],
        "faithbench": faith,
    }
    for name, frame in frames.items():
        d = b6[(b6["variant"] == "m3") & (b6["seed"] == 42) & (b6["dataset"] == name)]
        m = d.merge(frame[["sample_id", "source_group_id"]], on="sample_id", how="left")
        out[f"b6_{name}"] = boot_external(
            m["label"].to_numpy(), m["score"].to_numpy(), m["source_group_id"].to_numpy()
        )
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "x1_length_only_baseline": x1_length_only(),
        "x2_length_balanced": x2_length_balanced(),
        "x3_threshold_retune": x3_threshold_retune(),
        "x4_label_budget": x4_label_budget(),
        "x5_bootstrap_b2_grouped": x5_group_bootstrap_b2(),
        "x6_cross_domain_cis": x6_cross_domain_cis(),
    }
    path = OUT / "review_extras.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=1))
    print()
    print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
