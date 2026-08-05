"""
B4 — Calibration under distribution shift (roadmap §14 B4).

Compares raw XGBoost, Platt (sigmoid), and isotonic calibration when the
calibrated source model is applied out of domain:

  - SOURCE calibration: calibrators fit on HaluEval validation only, applied
    unchanged to HaluEval test, RAGTruth (QA test, other tasks), and FaithBench.
  - TARGET calibration: calibrators fit on RAGTruth QA official train,
    evaluated on RAGTruth QA official test (source_id groups disjoint).
  - FaithBench has no official calibration split -> source-calibrated only.

Rules enforced here:
  - Calibrators are fit ONLY on the designated calibration data (B4.2).
  - Selection rule predeclared: Platt is the primary deployable calibrator;
    isotonic reported for comparison (blueprint B8/B4.2).
  - Metrics: ECE, adaptive ECE, Brier, NLL (log loss), calibration
    slope/intercept, reliability curves, F1/AUROC at fixed threshold 0.5.
  - Subgroup calibration only for >= 100 rows and >= 20 source groups;
    smaller groups are pooled into the aggregate (no tiny calibrators).
  - All produced artifacts are pure sklearn (portable across platforms);
    no CUDA-trained boosters are saved here.

Inputs (must exist):
  artifacts/results/b3/b3_predictions.parquet  (external per-seed raw scores)
  artifacts/models/b2/xgboost_seed_*.joblib    (B2 models, CPU-portable in Colab)
  data/processed/features_full.parquet         (HaluEval val/test features)
  data/processed/unified_records.parquet       (context/answer text for bins)

Run (repo root, .venv or Colab after B3):
  python src/models/run_b4_calibration_shift.py
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("b4_calibration_shift")

from src.models.config import (  # noqa: E402
    DATA_PROCESSED,
    FIGURES_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    ROOT,
)
from src.models.train_pipeline import ece  # noqa: E402

B2_MODELS_DIR = MODELS_DIR / "b2"
B2_RESULTS_DIR = RESULTS_DIR / "b2"
B3_RESULTS = RESULTS_DIR / "b3"
B4_RESULTS = RESULTS_DIR / "b4"
B4_MODELS = MODELS_DIR / "b4"
B4_FIGURES = FIGURES_DIR / "b4"
B3_PREDICTIONS = B3_RESULTS / "b3_predictions.parquet"
FEATURES_FULL = DATA_PROCESSED / "features_full.parquet"
UNIFIED = DATA_PROCESSED / "unified_records.parquet"

MODEL_THRESHOLD = 0.5
SEEDS = [42, 123, 456]

MIN_SUBGROUP_ROWS = 100
MIN_SUBGROUP_GROUPS = 20

CONTEXT_WORD_BINS = [("lt_128", 0, 128), ("128_511", 128, 512), ("512_1023", 512, 1024), ("ge_1024", 1024, None)]
ANSWER_WORD_BINS = [("lt_32", 0, 32), ("32_127", 32, 128), ("ge_128", 128, None)]

# Predeclared selection rule (B4.2): Platt is the deployable default.
DEPLOYABLE_CALIBRATOR = "platt"


def git_commit() -> str | None:
    try:
        import subprocess

        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def word_bin(text: str, bins) -> str:
    n = len(str(text).split())
    for name, lo, hi in bins:
        if hi is None:
            if n >= lo:
                return name
        elif lo <= n < hi:
            return name
    return bins[-1][0]


# --------------------------------------------------------------------------
# Calibration primitives (pure sklearn -> portable artifacts)
# --------------------------------------------------------------------------

def fit_calibrator(method: str, scores: np.ndarray, y: np.ndarray):
    """Fit Platt (sigmoid) or isotonic calibrator on raw scores + labels."""
    if method == "platt":
        lr = LogisticRegression(max_iter=5000)
        lr.fit(scores.reshape(-1, 1), y)
        return lr
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(scores, y)
        return iso
    raise ValueError(f"unknown calibration method: {method}")


def apply_calibrator(method: str, calibrator, scores: np.ndarray) -> np.ndarray:
    if method == "platt":
        return calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
    if method == "isotonic":
        return calibrator.predict(scores)
    raise ValueError(f"unknown calibration method: {method}")


def adaptive_ece(y, p, n_bins: int = 10) -> float:
    """Equal-frequency (adaptive) ECE."""
    order = np.argsort(p)
    y_s, p_s = np.asarray(y)[order], np.asarray(p)[order]
    n = len(y_s)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        conf = p_s[lo:hi].mean()
        acc = y_s[lo:hi].mean()
        total += (hi - lo) / n * abs(acc - conf)
    return float(total)


def calibration_slope_intercept(y, p) -> tuple:
    """Logistic regression of y on logit(p): slope and intercept."""
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    logit = np.log(p) - np.log1p(-p)
    lr = LogisticRegression(max_iter=5000)
    lr.fit(logit.reshape(-1, 1), np.asarray(y))
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def calibration_metrics(y_true, proba, n_bins: int = 10) -> dict:
    y = np.asarray(y_true)
    p = np.asarray(proba)
    slope, intercept = calibration_slope_intercept(y, p)
    preds = (p >= MODEL_THRESHOLD).astype(int)
    return {
        "ece": float(ece(y, p, n_bins)),
        "ace": float(adaptive_ece(y, p, n_bins)),
        "brier": float(brier_score_loss(y, p)),
        "nll": float(log_loss(y, p, labels=[0, 1])),
        "slope": slope,
        "intercept": intercept,
        "f1": float(f1_score(y, preds, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(p)) > 1 else None,
        "predicted_positive_rate": float(preds.mean()),
    }


def reliability_curve(y_true, proba, n_bins: int = 10) -> list:
    y = np.asarray(y_true)
    p = np.asarray(proba)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idxs = np.clip(np.searchsorted(bins, p, side="right") - 1, 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = idxs == b
        out.append({
            "bin_center": float((bins[b] + bins[b + 1]) / 2),
            "n": int(mask.sum()),
            "confidence": float(p[mask].mean()) if mask.any() else None,
            "accuracy": float(y[mask].mean()) if mask.any() else None,
        })
    return out


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_halueval_features() -> pd.DataFrame:
    if not FEATURES_FULL.exists():
        raise FileNotFoundError(f"{FEATURES_FULL} not found. Run src/features/extract_features.py first.")
    return pd.read_parquet(FEATURES_FULL)


def halueval_probs_per_seed(features: pd.DataFrame, feature_cols: list) -> dict:
    """Raw B2 XGBoost probabilities on HaluEval val/test for each seed."""
    out = {"val": {}, "test": {}}
    for seed in SEEDS:
        model = joblib.load(B2_MODELS_DIR / f"xgboost_seed_{seed}.joblib")
        for split in ("val", "test"):
            sub = features[features["split"] == split]
            out[split][seed] = model.predict_proba(sub[feature_cols].values)[:, 1]
        logger.info(f"HaluEval raw probs (seed {seed}) computed")
    return out


def load_external_predictions() -> pd.DataFrame:
    if not B3_PREDICTIONS.exists():
        raise FileNotFoundError(
            f"{B3_PREDICTIONS} not found. Run src/models/run_b3_cross_domain.py first."
        )
    preds = pd.read_parquet(B3_PREDICTIONS)
    for seed in SEEDS:
        name = f"xgboost_seed_{seed}"
        if name not in set(preds["model"]):
            raise ValueError(f"b3 predictions missing model rows for {name}")
    wide = preds[preds["model"] == f"xgboost_seed_{SEEDS[0]}"][
        ["sample_id", "source_dataset", "source_group_id", "task", "domain",
         "official_split", "quality", "generator_model", "label"]
    ].copy()
    for seed in SEEDS:
        sub = preds[preds["model"] == f"xgboost_seed_{seed}"][["sample_id", "score"]]
        wide = wide.merge(sub.rename(columns={"score": f"score_{seed}"}), on="sample_id", how="left")
    return wide


def merge_text_for_bins(df: pd.DataFrame) -> pd.DataFrame:
    if not UNIFIED.exists():
        raise FileNotFoundError(f"{UNIFIED} not found. Run src/data/prepare_unified.py first.")
    uni = pd.read_parquet(UNIFIED)[["sample_id", "context", "answer", "question"]]
    out = df.merge(uni, on="sample_id", how="left")
    assert out["context"].notna().all(), "text merge failed"
    return out


# --------------------------------------------------------------------------
# Calibration experiments
# --------------------------------------------------------------------------

def evaluate_subsets(subsets: dict, external: pd.DataFrame, halueval_test: dict, feature_cols: list) -> dict:
    """Aggregate calibration metrics per (subset, method): mean +/- std over seeds."""
    return {}


def target_calibration_experiment(cal_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Fit target calibrators on cal_df (RAGTruth QA train), evaluate on test_df.

    Removes source groups that span calibration/test; asserts disjointness.
    """
    overlap = set(cal_df["source_group_id"]) & set(test_df["source_group_id"])
    if overlap:
        logger.warning(f"Removing {len(overlap)} source groups spanning train/test from calibration")
        cal_df = cal_df[~cal_df["source_group_id"].isin(overlap)]
    out = {
        "n_calibration_rows": int(len(cal_df)),
        "n_calibration_groups": int(cal_df["source_group_id"].nunique()),
        "n_test_rows": int(len(test_df)),
        "n_test_groups": int(test_df["source_group_id"].nunique()),
        "overlapping_groups_removed": len(overlap),
        "methods": {},
    }
    for method in ("platt", "isotonic"):
        rows = []
        for seed in SEEDS:
            cal = fit_calibrator(method, cal_df[f"score_{seed}"].values, cal_df["label"].values)
            p = apply_calibrator(method, cal, test_df[f"score_{seed}"].values)
            rows.append(calibration_metrics(test_df["label"].values, p))
        out["methods"][method] = mean_std_rows(rows)
        logger.info(f"target [{method}]: ece={out['methods'][method]['ece_mean']:.4f} "
                    f"brier={out['methods'][method]['brier_mean']:.4f}")
    return out


def main():
    parser = argparse.ArgumentParser(description="B4 calibration under distribution shift")
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(B4_RESULTS, exist_ok=True)
    os.makedirs(B4_MODELS, exist_ok=True)
    os.makedirs(B4_FIGURES, exist_ok=True)

    # ---- inputs ----
    features = load_halueval_features()
    feature_cols, b2_cfg = (lambda cfg: (list(cfg["feature_cols"]), cfg))(
        json.loads((B2_RESULTS_DIR / "b2_run_config.json").read_text()))
    hal = halueval_probs_per_seed(features, feature_cols)
    hal_val_labels = features[features["split"] == "val"]["label"].values
    hal_test_labels = features[features["split"] == "test"]["label"].values
    hal_test_df = features[features["split"] == "test"].reset_index(drop=True)

    external = load_external_predictions()
    external = merge_text_for_bins(external)

    # ---- subsets ----
    rag = external[external["source_dataset"] == "ragtruth"]
    subsets = {
        "halueval_test": None,  # handled separately
        "ragtruth_qa_test": rag[(rag["task"] == "qa") & (rag["official_split"] == "test")],
        "ragtruth_summarization": rag[rag["task"] == "summarization"],
        "ragtruth_data_to_text": rag[rag["task"] == "data_to_text"],
        "ragtruth_all": rag,
        "faithbench": external[external["source_dataset"] == "faithbench"],
    }

    # ---- source calibration (fit on HaluEval val only, per seed) ----
    calibrators = {"platt": {}, "isotonic": {}}
    for seed in SEEDS:
        for method in ("platt", "isotonic"):
            calibrators[method][seed] = fit_calibrator(method, hal["val"][seed], hal_val_labels)

    def calibrated(scores: np.ndarray, method: str, seed: int) -> np.ndarray:
        if method == "raw":
            return scores
        return apply_calibrator(method, calibrators[method][seed], scores)

    # ---- 1. HaluEval test (source calibration) ----
    cal_metrics = {"halueval_test": {}}
    for method in ("raw", "platt", "isotonic"):
        rows = []
        for seed in SEEDS:
            p = calibrated(hal["test"][seed], method, seed)
            rows.append(calibration_metrics(hal_test_labels, p))
        cal_metrics["halueval_test"][method] = mean_std_rows(rows)
        logger.info(f"halueval_test [{method}]: ece={cal_metrics['halueval_test'][method]['ece_mean']:.4f} "
                    f"brier={cal_metrics['halueval_test'][method]['brier_mean']:.4f}")

    # ---- 2. External subsets (source calibration applied) ----
    for name, sub in subsets.items():
        if sub is None or len(sub) == 0:
            continue
        idx = external["sample_id"].isin(set(sub["sample_id"])).values
        sub_df = external[idx]
        y = sub_df["label"].values
        cal_metrics[name] = {}
        for method in ("raw", "platt", "isotonic"):
            rows = []
            for seed in SEEDS:
                p = calibrated(sub_df[f"score_{seed}"].values, method, seed)
                rows.append(calibration_metrics(y, p))
            cal_metrics[name][method] = mean_std_rows(rows)
            logger.info(f"{name} [{method}]: ece={cal_metrics[name][method]['ece_mean']:.4f} "
                        f"brier={cal_metrics[name][method]['brier_mean']:.4f}")

    # ---- 3. Target calibration (RAGTruth QA train -> QA test) ----
    qa_cal = rag[(rag["task"] == "qa") & (rag["official_split"] == "train")]
    qa_test = rag[(rag["task"] == "qa") & (rag["official_split"] == "test")]
    target = target_calibration_experiment(qa_cal, qa_test)
    # target vs source on the same QA test set
    target["methods"]["source_platt_reference"] = cal_metrics["ragtruth_qa_test"]["platt"]
    target["methods"]["source_isotonic_reference"] = cal_metrics["ragtruth_qa_test"]["isotonic"]
    target["methods"]["raw_reference"] = cal_metrics["ragtruth_qa_test"]["raw"]

    # ---- 4. Subgroup calibration (seed 42, source-calibrated) ----
    subgroup_rows = []
    rag_idx = external["source_dataset"] == "ragtruth"
    rag_df = external[rag_idx].reset_index(drop=True)
    for dim, col in (("task", "task"), ("official_split", "official_split"), ("domain", "domain"),
                     ("generator_model", "generator_model"), ("quality", "quality")):
        subgroup_rows += subgroup_calibration(rag_df, dim, calibrators, args.n_bins)
    subgroup_rows += subgroup_calibration(rag_df, "context_length", calibrators, args.n_bins,
                                          bin_fn=lambda t: word_bin(t, CONTEXT_WORD_BINS))
    subgroup_rows += subgroup_calibration(rag_df, "answer_length", calibrators, args.n_bins,
                                          bin_fn=lambda t: word_bin(t, ANSWER_WORD_BINS))
    fb_df = external[external["source_dataset"] == "faithbench"].reset_index(drop=True)
    subgroup_rows += subgroup_calibration(fb_df, "generator_model", calibrators, args.n_bins)

    # ---- 5. Reliability curves (seed 42) ----
    reliability = {}
    for name, sub in subsets.items():
        if sub is None or len(sub) == 0:
            continue
        idx = external["sample_id"].isin(set(sub["sample_id"])).values
        sub_df = external[idx]
        y = sub_df["label"].values
        s42 = sub_df["score_42"].values
        reliability[name] = {
            method: reliability_curve(y, calibrated(s42, method, 42), args.n_bins)
            for method in ("raw", "platt", "isotonic")
        }
    y_test = hal_test_labels
    s42_test = hal["test"][42]
    reliability["halueval_test"] = {
        method: reliability_curve(y_test, calibrated(s42_test, method, 42), args.n_bins)
        for method in ("raw", "platt", "isotonic")
    }

    # ---- 6. Per-sample calibrated predictions (seed 42) ----
    pred_rows = []
    for name, sub in subsets.items():
        if sub is None or len(sub) == 0:
            continue
        idx = external["sample_id"].isin(set(sub["sample_id"])).values
        sub_df = external[idx].reset_index(drop=True)
        for method in ("raw", "platt", "isotonic"):
            p = calibrated(sub_df["score_42"].values, method, 42)
            for pos, r in sub_df.iterrows():
                pred_rows.append({"sample_id": r["sample_id"], "source_dataset": r["source_dataset"],
                                  "subset": name, "method": method, "label": int(r["label"]),
                                  "score": round(float(p[pos]), 6),
                                  "pred": int(p[pos] >= MODEL_THRESHOLD)})
    hal_test_rows = []
    for method in ("raw", "platt", "isotonic"):
        p = calibrated(s42_test, method, 42)
        for i in range(len(hal_test_df)):
            hal_test_rows.append({"sample_id": hal_test_df.loc[i, "sample_id"],
                                  "source_dataset": "halueval", "subset": "halueval_test",
                                  "method": method, "label": int(y_test[i]),
                                  "score": round(float(p[i]), 6),
                                  "pred": int(p[i] >= MODEL_THRESHOLD)})
    pred_df = pd.DataFrame(pred_rows + hal_test_rows)

    # ---- 7. Save artifacts ----
    os.makedirs(B4_RESULTS, exist_ok=True)
    (B4_RESULTS / "b4_calibration_metrics.json").write_text(json.dumps(cal_metrics, indent=2))
    flat = []
    for subset, methods in cal_metrics.items():
        for method, m in methods.items():
            flat.append({"subset": subset, "method": method, **{k: v for k, v in m.items() if not isinstance(v, dict)}})
    pd.DataFrame(flat).to_csv(B4_RESULTS / "b4_calibration_metrics.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(B4_RESULTS / "b4_subgroup_calibration.csv", index=False)
    (B4_RESULTS / "b4_target_calibration.json").write_text(json.dumps(target, indent=2))
    (B4_RESULTS / "b4_reliability_data.json").write_text(json.dumps(reliability, indent=2))
    pred_df.to_parquet(B4_RESULTS / "b4_predictions.parquet", index=False)

    for method, seed in (("platt", 42), ("isotonic", 42)):
        joblib.dump(calibrators[method][seed], B4_MODELS / f"calibrator_{method}_source_seed_{seed}.joblib")
    qa_cal_s42 = qa_cal["score_42"].values
    for method in ("platt", "isotonic"):
        cal = fit_calibrator(method, qa_cal_s42, qa_cal["label"].values)
        joblib.dump(cal, B4_MODELS / f"calibrator_{method}_target_ragtruth_qa_seed_42.joblib")
    logger.info(f"Saved calibrators to {B4_MODELS}")

    config = {
        "schema": "b4-config-v1",
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "git_commit": git_commit(),
        "n_bins": args.n_bins,
        "threshold": MODEL_THRESHOLD,
        "selection_rule": f"Platt is the predeclared deployable calibrator ({DEPLOYABLE_CALIBRATOR}); isotonic reported for comparison",
        "source_calibration_data": "HaluEval validation (fit), seeds 42/123/456",
        "target_calibration_data": "RAGTruth QA official train -> QA official test (disjoint source groups)",
        "faithbench_calibration": "source-calibrated only (no official split)",
        "subgroup_minimums": {"rows": MIN_SUBGROUP_ROWS, "groups": MIN_SUBGROUP_GROUPS},
        "inputs": {
            "b3_predictions.parquet": sha256(B3_PREDICTIONS),
            "features_full.parquet": sha256(FEATURES_FULL),
            "b2_run_config.json": sha256(B2_RESULTS_DIR / "b2_run_config.json"),
        },
        "note": "All B4 calibrators are pure sklearn objects and load on any platform (no CUDA booster serialization).",
    }
    (B4_RESULTS / "b4_run_config.json").write_text(json.dumps(config, indent=2))

    make_figures(cal_metrics, subgroup_rows, reliability)

    print("\n" + "=" * 100)
    print(" B4 — Calibration under distribution shift (ECE / Brier / NLL, seeds 42/123/456)")
    print("=" * 100)
    print(pd.DataFrame(flat)[["subset", "method", "ece_mean", "ace_mean", "brier_mean", "nll_mean",
                              "slope_mean", "intercept_mean", "f1_mean", "auroc_mean"]].round(4).to_string(index=False))
    print("TARGET (RAGTruth QA train -> test):")
    print(json.dumps({k: v for k, v in target["methods"].items() if "mean" in str(v)}, indent=2)[:800])
    print("=" * 100)


def mean_std_rows(rows: list) -> dict:
    keys = ["ece", "ace", "brier", "nll", "slope", "intercept", "f1", "auroc", "predicted_positive_rate"]
    out = {"n_seeds": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[f"{k}_mean"] = float(np.mean(vals)) if vals else None
        out[f"{k}_std"] = float(np.std(vals)) if vals else None
    return out


def subgroup_calibration(df: pd.DataFrame, dimension: str, calibrators: dict, n_bins: int, bin_fn=None) -> list:
    """ECE/Brier/NLL per subgroup (source-calibrated, seed 42) with minimum-size rules."""
    rows = []
    y = df["label"].values
    work = df.copy()
    if bin_fn is not None:
        work["_bin"] = work["answer"].map(bin_fn)
        key_col = "_bin"
    else:
        key_col = dimension
    for key, sub in work.groupby(key_col):
        n_groups = sub["source_group_id"].nunique()
        if len(sub) < MIN_SUBGROUP_ROWS or n_groups < MIN_SUBGROUP_GROUPS:
            rows.append({"dimension": dimension, "subgroup": str(key), "n_rows": int(len(sub)),
                         "n_groups": int(n_groups), "reported": False,
                         "reason": "below minimum (rows<100 or groups<20)"})
            continue
        y_sub = y[work.index.isin(sub.index)]
        entry = {"dimension": dimension, "subgroup": str(key), "n_rows": int(len(sub)),
                 "n_groups": int(n_groups), "reported": True}
        for method in ("platt", "isotonic"):
            p = apply_calibrator(method, calibrators[method][42], sub["score_42"].values)
            m = calibration_metrics(y_sub, p, n_bins)
            for k in ("ece", "brier", "nll"):
                entry[f"{method}_{k}"] = round(m[k], 6)
        m_raw = calibration_metrics(y_sub, sub["score_42"].values, n_bins)
        for k in ("ece", "brier", "nll"):
            entry[f"raw_{k}"] = round(m_raw[k], 6)
        rows.append(entry)
    return rows


def make_figures(cal_metrics: dict, subgroup_rows: list, reliability: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(B4_FIGURES, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), squeeze=False)
    panel_sets = [("halueval_test", "HaluEval test"), ("ragtruth_qa_test", "RAGTruth QA test"), ("faithbench", "FaithBench")]
    for ax, (name, title) in zip(axes[0], panel_sets):
        if name not in cal_metrics:
            continue
        methods = list(cal_metrics[name].keys())
        ece = [cal_metrics[name][m]["ece_mean"] for m in methods]
        brier = [cal_metrics[name][m]["brier_mean"] for m in methods]
        x = np.arange(len(methods))
        ax.bar(x - 0.15, ece, 0.3, label="ECE")
        ax.bar(x + 0.15, brier, 0.3, label="Brier")
        ax.set_xticks(x, methods)
        ax.set_title(title)
        ax.set_ylim(0, max(0.7, max(ece + brier) * 1.2))
        ax.legend()
    fig.tight_layout()
    fig.savefig(B4_FIGURES / "calibration_shift.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), squeeze=False)
    for ax, (name, title) in zip(axes[0], panel_sets):
        if name not in reliability:
            continue
        for method, color in (("raw", "gray"), ("platt", "tab:blue"), ("isotonic", "tab:orange")):
            curve = reliability[name][method]
            conf = [c["confidence"] for c in curve if c["confidence"] is not None]
            acc = [c["accuracy"] for c in curve if c["accuracy"] is not None]
            ax.plot(conf, acc, marker="o", ms=3, label=method, color=color)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{title} reliability")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(B4_FIGURES / "reliability_diagrams.png", dpi=150)
    plt.close(fig)

    reported = [r for r in subgroup_rows if r.get("reported")]
    if reported:
        dims = sorted(set(r["dimension"] for r in reported))
        fig, axes = plt.subplots(1, len(dims), figsize=(6 * len(dims), 4.5), squeeze=False)
        for ax, dim in zip(axes[0], dims):
            sub = [r for r in reported if r["dimension"] == dim]
            labels = [r["subgroup"] for r in sub]
            x = np.arange(len(sub))
            ax.bar(x - 0.2, [r.get("raw_ece") or 0 for r in sub], 0.25, label="raw")
            ax.bar(x + 0.0, [r.get("platt_ece") or 0 for r in sub], 0.25, label="platt")
            ax.bar(x + 0.2, [r.get("isotonic_ece") or 0 for r in sub], 0.25, label="isotonic")
            ax.set_xticks(x, labels, rotation=20, ha="right")
            ax.set_title(f"ECE by {dim}")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(B4_FIGURES / "subgroup_calibration.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
