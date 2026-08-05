"""
B2 — Corrected baseline and artifact-control experiments (roadmap §14 B2).

Runs the 26-feature pipeline on the corrected grouped HaluEval split with
artifact controls: majority, overlap heuristic (validation-tuned), TF-IDF
(all / answer-only / context-only), NLI-only, Logistic Regression, Random
Forest, and tuned XGBoost — repeated with seeds 42/123/456.

Leakage controls (B2 exit criteria):
  - inputs validated against qa_clean.parquet / split_indices.json
  - XGBoost tuning uses StratifiedGroupKFold keyed by item_idx
  - TF-IDF vocabulary and IDF fit on train text only
  - thresholds: 0.5 for all models; overlap threshold tuned on validation only

All B2 artifacts are namespaced under artifacts/{results,models}/b2/ and
never overwrite Version A artifacts.

Run (repo root, .venv or Colab):
  python src/models/run_b2_baselines.py
  python src/models/run_b2_baselines.py --smoke-test   # synthetic, fast
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import joblib
import pandas as pd
import scipy.stats as scipy_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.contingency_tables import mcnemar
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("b2_baselines")

from src.models.config import (  # noqa: E402
    BOOTSTRAP_SEED,
    DATA_PROCESSED,
    MODELS_DIR,
    N_BOOTSTRAP,
    RESULTS_DIR,
    ROOT,
    SEEDS,
)
from src.models.train_pipeline import (  # noqa: E402
    FEATURE_GROUPS,
    TUNING_GRID,
    bootstrap_ci,
    ece,
    make_xgb,
    xgb_device,
)

QA_CLEAN = DATA_PROCESSED / "qa_clean.parquet"
FEATURES_FULL = DATA_PROCESSED / "features_full.parquet"
SPLIT_INDICES = ROOT / "artifacts" / "split_indices.json"
SPLIT_REPORT = ROOT / "artifacts" / "split_integrity_report.json"

DEFAULT_RESULTS_DIR = RESULTS_DIR / "b2"
DEFAULT_MODELS_DIR = MODELS_DIR / "b2"

MODEL_THRESHOLD = 0.5

# Documented historical reference: the README benchmark table from before the
# leakage repair (row-level split, 2026-08-04 era). Kept for the B2.6
# leakage-removal impact report; the corrected Version A numbers are read from
# artifacts/results/final_results.json.
HISTORICAL_LEAKED_XGB = {
    "f1": 0.9886,
    "auroc": 0.9980,
    "source": "README.md pre-repair benchmark table (row-level split, leaky)",
}

SMOKE_GRID = {
    "max_depth": [3, 4],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [50, 100],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


@dataclass
class B2Config:
    results_dir: Path = DEFAULT_RESULTS_DIR
    models_dir: Path = DEFAULT_MODELS_DIR
    seeds: list = field(default_factory=lambda: list(SEEDS))
    n_iter: int = 30
    tuning_grid: dict = field(default_factory=lambda: dict(TUNING_GRID))
    tfidf_max_features: int = 100_000
    tfidf_min_df: int = 2
    smoke: bool = False


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        import subprocess

        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def load_and_validate() -> dict:
    """Validate the corrected HaluEval inputs; returns feature frame + metadata."""
    for p in (QA_CLEAN, FEATURES_FULL, SPLIT_INDICES, SPLIT_REPORT):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run src/data/prepare.py and src/features/extract_features.py first.")

    qa = pd.read_parquet(QA_CLEAN)
    features = pd.read_parquet(FEATURES_FULL)
    split_indices = json.loads(SPLIT_INDICES.read_text())
    split_report = json.loads(SPLIT_REPORT.read_text())

    assert len(qa) == 20000, f"qa_clean rows {len(qa)} != 20000"
    assert qa["label"].value_counts().to_dict() == {0: 10000, 1: 10000}, "label balance broken"
    assert len(features) == len(qa), "feature matrix row count mismatch"
    assert set(features["sample_id"]) == set(qa["sample_id"]), "sample_id sets mismatch"

    counts = features.groupby("split").size()
    assert counts.to_dict() == {"train": 14000, "val": 3000, "test": 3000}, f"split sizes {counts.to_dict()}"
    assert split_report.get("leakage_free") is True, "split_integrity_report not leakage-free"
    per = features.groupby("item_idx")["split"].nunique()
    assert per.max() == 1, f"{int((per > 1).sum())} item_idx groups span multiple splits"

    # TF-IDF controls need the raw text; features_full.parquet carries features only
    features = features.merge(qa[["sample_id", "question", "context", "answer"]], on="sample_id", how="left")
    assert features[["question", "context", "answer"]].notna().all().all(), "text merge produced NaN"

    feature_cols = []
    for group, cols in FEATURE_GROUPS.items():
        missing = [c for c in cols if c not in features.columns]
        if missing:
            raise ValueError(f"group '{group}' missing columns {missing}")
        feature_cols.extend(cols)

    logger.info(
        f"Validated inputs: {len(features)} rows, {features['item_idx'].nunique()} groups, "
        f"{len(feature_cols)} features, leakage-free split confirmed."
    )
    return {
        "features": features,
        "feature_cols": feature_cols,
        "split_indices": split_indices,
        "split_report": split_report,
        "input_hashes": {
            "qa_clean.parquet": sha256(QA_CLEAN),
            "features_full.parquet": sha256(FEATURES_FULL),
            "split_indices.json": sha256(SPLIT_INDICES),
            "split_integrity_report.json": sha256(SPLIT_REPORT),
        },
    }


def build_synthetic(n_groups: int = 60, seed: int = 7) -> dict:
    """Synthetic corrected-like data for smoke tests (no downloads)."""
    rng = np.random.default_rng(seed)
    qa_rows, feat_rows = [], []
    for i in range(n_groups):
        q = f"question {i} about topic"
        c = f"context passage for question {i} with facts"
        a0 = f"the answer derived from the context for question {i}"
        a1 = f"a fabricated answer that contradicts everything said before {i}"
        for sid, label, ans in ((0, 0, a0), (1, 1, a1)):
            qa_rows.append({"sample_id": f"q_{i}_{'c' if label == 0 else 'h'}", "item_idx": i,
                            "question": q, "context": c, "answer": ans, "label": label})
            feat = {cname: float(rng.random()) for cname in FEATURE_GROUPS["length"] + FEATURE_GROUPS["lexical"]}
            feat.update({cname: float(rng.random()) for cname in FEATURE_GROUPS["nli"] + FEATURE_GROUPS["semantic"]})
            feat.update({"n_numbers_answer": 0, "n_numbers_context": 2, "number_overlap_ratio": 1.0, "novel_numbers": 0})
            feat.update({"hedge_count": 0, "hedge_density": 0.0})
            feat.update({cname: 0.0 for cname in FEATURE_GROUPS["entity"]})
            feat_rows.append({"sample_id": f"q_{i}_{'c' if label == 0 else 'h'}", "item_idx": i,
                              "question": q, "context": c, "answer": ans, "label": label, **feat})
    qa = pd.DataFrame(qa_rows)
    from src.data.prepare import group_split_by_item

    split_df, report = group_split_by_item(qa)
    features = pd.DataFrame(feat_rows).merge(split_df[["sample_id", "split"]], on="sample_id")
    feature_cols = []
    for group, cols in FEATURE_GROUPS.items():
        feature_cols.extend(cols)
    return {"features": features, "feature_cols": feature_cols,
            "split_indices": {}, "split_report": report, "input_hashes": {}}


def split_views(df: pd.DataFrame, feature_cols: list):
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    return train_df, val_df, test_df


def check_group_cv_disjoint(cv, X, y, groups) -> dict:
    """Assert every CV fold keeps item_idx groups disjoint; returns fold report."""
    report = {"n_splits": 0, "groups_per_fold": [], "overlapping_groups_across_folds": 0}
    for tr_idx, va_idx in cv.split(X, y, groups=groups):
        tr_g = set(groups[tr_idx])
        va_g = set(groups[va_idx])
        overlap = tr_g & va_g
        assert not overlap, f"group leakage across CV folds: {len(overlap)} shared groups"
        report["n_splits"] += 1
        report["groups_per_fold"].append({"train_groups": len(tr_g), "val_groups": len(va_g)})
    logger.info(f"Group CV check: {report['n_splits']} folds, 0 overlapping groups.")
    return report


def run_tuning(X_train, y_train, groups_train, seed: int, cfg: B2Config) -> tuple:
    """Tune XGBoost with grouped 5-fold CV; returns (best_params, best_score, cv_report)."""
    t0 = time.time()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_report = check_group_cv_disjoint(cv, X_train, y_train, groups_train)
    xgb = make_xgb({}, seed, scale_pos_weight=1.0)
    rs = RandomizedSearchCV(
        xgb, cfg.tuning_grid, n_iter=cfg.n_iter, cv=cv, scoring="roc_auc",
        n_jobs=1, random_state=seed, verbose=0,
    )
    rs.fit(X_train, y_train, groups=groups_train)
    logger.info(f"Seed {seed}: best params {rs.best_params_} cv_auc={rs.best_score_:.4f} ({time.time() - t0:.0f}s)")
    return rs.best_params_, float(rs.best_score_), cv_report


def evaluate(y_true, y_pred, y_prob) -> dict:
    """Classification metrics + calibration diagnostics + confusion matrix."""
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "ece": float(ece(y_true, y_prob)) if y_prob is not None else None,
        "brier": float(brier_score_loss(y_true, y_prob)) if y_prob is not None else None,
    }
    if y_prob is not None and len(np.unique(y_prob)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["auroc"] = None
        metrics["pr_auc"] = None
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return metrics


def heuristic_overlap(train_df, val_df, test_df, col: str = "overlap_answer_context"):
    """Risk = 1 - overlap; threshold tuned on validation F1 only."""
    thresholds = np.linspace(0, 1, 101)
    best_thresh, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(val_df["label"], (val_df[col] < t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    test_preds = (test_df[col] < best_thresh).astype(int)
    test_probs = (1.0 - test_df[col]).to_numpy()
    info = {"threshold": float(best_thresh), "val_f1": float(best_f1)}
    return test_preds, test_probs, info


def run_experiment(cfg: B2Config, data: dict) -> dict:
    """Execute the full B2 experiment and save artifacts under cfg dirs."""
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.models_dir, exist_ok=True)

    features = data["features"].reset_index(drop=True)
    feature_cols = data["feature_cols"]
    train_df, val_df, test_df = split_views(features, feature_cols)
    y_train, y_val, y_test = (train_df["label"].values, val_df["label"].values, test_df["label"].values)
    groups_train = train_df["item_idx"].values

    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values

    scaler_full = StandardScaler().fit(X_train)
    X_train_s = scaler_full.transform(X_train)
    X_val_s = scaler_full.transform(X_val)
    X_test_s = scaler_full.transform(X_test)

    nli_cols = FEATURE_GROUPS["nli"]
    scaler_nli = StandardScaler().fit(train_df[nli_cols].values)
    X_train_nli = scaler_nli.transform(train_df[nli_cols].values)
    X_val_nli = scaler_nli.transform(val_df[nli_cols].values)
    X_test_nli = scaler_nli.transform(test_df[nli_cols].values)

    text_views = {
        "tfidf_all": train_df["question"] + " " + train_df["context"] + " " + train_df["answer"],
        "tfidf_answer": train_df["answer"],
        "tfidf_context": train_df["context"],
    }
    tfidf_fit = {
        name: TfidfVectorizer(
            ngram_range=(1, 2), min_df=cfg.tfidf_min_df,
            max_features=cfg.tfidf_max_features, sublinear_tf=True,
        ).fit(texts)
        for name, texts in text_views.items()
    }
    tfidf_train = {n: v.transform(train_df["question"] + " " + train_df["context"] + " " + train_df["answer"] if n == "tfidf_all" else (train_df["answer"] if n == "tfidf_answer" else train_df["context"])) for n, v in tfidf_fit.items()}
    tfidf_val = {n: v.transform(val_df["question"] + " " + val_df["context"] + " " + val_df["answer"] if n == "tfidf_all" else (val_df["answer"] if n == "tfidf_answer" else val_df["context"])) for n, v in tfidf_fit.items()}
    tfidf_test = {n: v.transform(test_df["question"] + " " + test_df["context"] + " " + test_df["answer"] if n == "tfidf_all" else (test_df["answer"] if n == "tfidf_answer" else test_df["context"])) for n, v in tfidf_fit.items()}

    results_rows = []
    prediction_rows = []
    tuning_report = {}
    item_idx_of = dict(zip(test_df["sample_id"], test_df["item_idx"]))

    def record(model_name, seed, deterministic, threshold, preds, probs, val_f1=None):
        metrics = evaluate(y_test, preds, probs)
        row = {"model": model_name, "seed": seed, "deterministic": deterministic,
               "threshold": threshold, "val_f1": val_f1, **metrics}
        results_rows.append(row)
        for sid, label, score, pred in zip(test_df["sample_id"], y_test, probs if probs is not None else [None] * len(y_test), preds):
            prediction_rows.append({"sample_id": sid, "item_idx": int(item_idx_of[sid]),
                                    "label": int(label), "split": "test", "model": model_name,
                                    "seed": seed, "threshold": threshold,
                                    "score": float(score) if score is not None else None,
                                    "pred": int(pred)})
        return metrics

    # ---- 1. Majority (deterministic; balanced data, ties resolve to 0) ----
    majority_preds = np.zeros(len(y_test), dtype=int)
    record("majority", None, True, MODEL_THRESHOLD, majority_preds, None, val_f1=0.5)

    # ---- 2. Overlap heuristic (threshold tuned on validation) ----
    h_pred, h_prob, h_info = heuristic_overlap(train_df, val_df, test_df)
    record("heuristic_overlap", None, True, h_info["threshold"], h_pred, h_prob, val_f1=h_info["val_f1"])

    # ---- 3. TF-IDF controls + NLI-only + LR/RF (3 seeds) ----
    def fit_lr(Xtr, Xte, seed):
        lr = LogisticRegression(max_iter=2000, random_state=seed)
        lr.fit(Xtr, y_train)
        return lr.predict_proba(Xte)[:, 1]

    model_probs = {"lr_full": {}, "rf_full": {}, "nli_only": {}, "tfidf_all": {}, "tfidf_answer": {}, "tfidf_context": {}}

    for seed in cfg.seeds:
        model_probs["lr_full"][seed] = fit_lr(X_train_s, X_test_s, seed)
        model_probs["nli_only"][seed] = fit_lr(X_train_nli, X_test_nli, seed)
        for name in ("tfidf_all", "tfidf_answer", "tfidf_context"):
            model_probs[name][seed] = fit_lr(tfidf_train[name], tfidf_test[name], seed)
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed)
        rf.fit(X_train, y_train)
        model_probs["rf_full"][seed] = rf.predict_proba(X_test)[:, 1]
        joblib.dump(rf, cfg.models_dir / f"random_forest_seed_{seed}.joblib")

    for name, prob_by_seed in model_probs.items():
        for seed in cfg.seeds:
            p = prob_by_seed[seed]
            preds = (p >= MODEL_THRESHOLD).astype(int)
            record(name, seed, False, MODEL_THRESHOLD, preds, p)

    # ---- 4. Tuned XGBoost per seed (grouped CV) ----
    xgb_probs = {}
    xgb_models = {}
    best_params_by_seed = {}
    for seed in cfg.seeds:
        best_params, best_cv_auc, cv_report = run_tuning(X_train, y_train, groups_train, seed, cfg)
        best_params_by_seed[seed] = {"params": best_params, "cv_auc": best_cv_auc}
        tuning_report[str(seed)] = {"best_params": best_params, "best_cv_auc": best_cv_auc, "group_cv": cv_report}
        xgb = make_xgb(best_params, seed, scale_pos_weight=1.0, early_stopping=True)
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        xgb_models[seed] = xgb
        xgb_probs[seed] = xgb.predict_proba(X_test)[:, 1]
        p = xgb_probs[seed]
        record("xgboost", seed, False, MODEL_THRESHOLD, (p >= MODEL_THRESHOLD).astype(int), p)
        joblib.dump(xgb, cfg.models_dir / f"xgboost_seed_{seed}.joblib")

    for name, vec in tfidf_fit.items():
        joblib.dump(vec, cfg.models_dir / f"tfidf_{name}.joblib")
    joblib.dump(scaler_full, cfg.models_dir / "scaler_full.joblib")
    joblib.dump(scaler_nli, cfg.models_dir / "scaler_nli.joblib")
    for seed in cfg.seeds:
        lr = LogisticRegression(max_iter=2000, random_state=seed)
        lr.fit(X_train_s, y_train)
        joblib.dump(lr, cfg.models_dir / f"logistic_regression_full_seed_{seed}.joblib")

    results_df = pd.DataFrame(results_rows)
    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_parquet(cfg.results_dir / "b2_predictions.parquet", index=False)

    # ---- 5. Best predeclared non-XGB baseline (rule: best mean val F1) ----
    val_f1s = {}
    for name, prob_by_seed in model_probs.items():
        p_val_seed0 = None
        # compute val probs for seed 42 only (rule applied on validation)
        seed0 = cfg.seeds[0]
        if name == "lr_full":
            lr = LogisticRegression(max_iter=2000, random_state=seed0).fit(X_train_s, y_train)
            pv = lr.predict_proba(X_val_s)[:, 1]
        elif name == "rf_full":
            rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed0).fit(X_train, y_train)
            pv = rf.predict_proba(X_val)[:, 1]
        elif name == "nli_only":
            lr = LogisticRegression(max_iter=2000, random_state=seed0).fit(X_train_nli, y_train)
            pv = lr.predict_proba(X_val_nli)[:, 1]
        else:
            lr = LogisticRegression(max_iter=2000, random_state=seed0).fit(tfidf_train[name], y_train)
            pv = lr.predict_proba(tfidf_val[name])[:, 1]
        val_f1s[name] = float(f1_score(y_val, (pv >= MODEL_THRESHOLD).astype(int), zero_division=0))
    best_baseline = max(val_f1s, key=val_f1s.get)
    logger.info(f"Best predeclared non-XGB baseline (val F1 rule): {best_baseline} (val_f1={val_f1s[best_baseline]:.4f})")

    # ---- 6. Statistics ----
    def mcnemar_p(pred_a, pred_b):
        b = int(((pred_a == 0) & (pred_b == 1)).sum())
        c = int(((pred_a == 1) & (pred_b == 0)).sum())
        return float(mcnemar([[0, b], [c, 0]], exact=False, correction=True).pvalue)

    xgb_pred_42 = (xgb_probs[cfg.seeds[0]] >= MODEL_THRESHOLD).astype(int)
    stats = {"mcnemar_vs_best_baseline": mcnemar_p(xgb_pred_42, (model_probs[best_baseline][cfg.seeds[0]] >= MODEL_THRESHOLD).astype(int)),
             "best_baseline": best_baseline,
             "best_baseline_rule": "max mean validation F1 over non-XGB model baselines (seed 42)",
             "mcnemar_pairs": {}}
    for name in ("lr_full", "rf_full", "nli_only", "tfidf_all", "tfidf_answer", "tfidf_context", "heuristic_overlap", "majority"):
        preds = (model_probs[name][cfg.seeds[0]] >= MODEL_THRESHOLD).astype(int) if name in model_probs else (h_pred if name == "heuristic_overlap" else majority_preds)
        stats["mcnemar_pairs"][name] = mcnemar_p(xgb_pred_42, preds)

    xgb_row_42 = next(r for r in results_rows if r["model"] == "xgboost" and r["seed"] == cfg.seeds[0])
    boot = bootstrap_ci(y_test, xgb_pred_42, xgb_probs[cfg.seeds[0]])
    stats["bootstrap_xgb_f1_ci"] = boot["f1_ci"]
    stats["bootstrap_xgb_auroc_ci"] = boot["auroc_ci"]

    def wilcoxon(name):
        xgb_f1 = [next(r for r in results_rows if r["model"] == "xgboost" and r["seed"] == s)["f1"] for s in cfg.seeds]
        base_f1 = [next(r for r in results_rows if r["model"] == name and r["seed"] == s)["f1"] for s in cfg.seeds]
        if len(set(xgb_f1)) == 1 and xgb_f1 == base_f1:
            return None
        try:
            return float(scipy_stats.wilcoxon(xgb_f1, base_f1).pvalue)
        except ValueError:
            return None

    stats["wilcoxon_xgb_vs_rf_p"] = wilcoxon("rf_full")
    stats["wilcoxon_xgb_vs_lr_p"] = wilcoxon("lr_full")
    stats["wilcoxon_xgb_vs_nli_only_p"] = wilcoxon("nli_only")
    stats["wilcoxon_xgb_vs_tfidf_all_p"] = wilcoxon("tfidf_all")

    # ---- 7. Leakage comparison ----
    a_corrected = None
    final_path = RESULTS_DIR / "final_results.json"
    if final_path.exists():
        fr = json.loads(final_path.read_text())
        a_corrected = {"f1": fr["xgboost"]["f1"], "auroc": fr["xgboost"]["auroc"]}
    b2_xgb_f1 = float(np.mean([r["f1"] for r in results_rows if r["model"] == "xgboost"]))
    b2_xgb_auc = float(np.mean([r["auroc"] for r in results_rows if r["model"] == "xgboost" and r["auroc"] is not None]))
    leakage = {
        "historical_leaked_row_level": HISTORICAL_LEAKED_XGB,
        "version_a_corrected_grouped": a_corrected,
        "b2_xgboost_grouped_cv": {"f1_mean": b2_xgb_f1, "auroc_mean": b2_xgb_auc},
        "delta_b2_vs_leaked_f1": round(b2_xgb_f1 - HISTORICAL_LEAKED_XGB["f1"], 4),
        "delta_b2_vs_leaked_auroc": round(b2_xgb_auc - HISTORICAL_LEAKED_XGB["auroc"], 4),
        "note": "Historical leaked numbers come from the pre-repair README table (row-level split). "
                "B2 uses the corrected grouped split AND grouped 5-fold CV for tuning; Version A used the "
                "corrected split with row-level stratified CV.",
    }

    # ---- 8. Save reports ----
    comparison = {}
    for model in results_df["model"].unique():
        sub = results_df[results_df["model"] == model]
        deterministic = bool(sub["deterministic"].iloc[0])
        entry = {"model": model, "deterministic": deterministic,
                 "threshold": float(sub["threshold"].iloc[0]),
                 "n_seeds": 1 if deterministic else len(sub)}
        if not deterministic:
            for key in ("precision", "recall", "f1", "auroc", "pr_auc", "mcc", "ece", "brier"):
                vals = [v for v in sub[key] if v is not None]
                entry[f"{key}_mean"] = float(np.mean(vals)) if vals else None
                entry[f"{key}_std"] = float(np.std(vals)) if vals else None
        else:
            row = sub.iloc[0]
            for key in ("precision", "recall", "f1", "auroc", "pr_auc", "mcc", "ece", "brier"):
                entry[f"{key}_mean"] = row[key]
                entry[f"{key}_std"] = 0.0
        if model == "heuristic_overlap":
            entry["val_f1"] = h_info["val_f1"]
        comparison[model] = entry

    comparison_df = pd.DataFrame(comparison).T
    comparison_df = comparison_df[[c for c in ["model", "deterministic", "threshold", "n_seeds",
                                               "precision_mean", "precision_std", "recall_mean", "recall_std",
                                               "f1_mean", "f1_std", "auroc_mean", "auroc_std",
                                               "pr_auc_mean", "pr_auc_std", "mcc_mean", "mcc_std",
                                               "ece_mean", "ece_std", "brier_mean", "brier_std", "val_f1"] if c in comparison_df.columns]]
    comparison_df.to_csv(cfg.results_dir / "b2_model_comparison.csv")
    (cfg.results_dir / "b2_model_comparison.json").write_text(json.dumps(comparison, indent=2))

    per_seed = results_df.drop(columns=["confusion"]).copy()
    per_seed.to_csv(cfg.results_dir / "b2_per_seed_metrics.csv", index=False)

    confusions = {}
    for _, r in results_df.iterrows():
        key = f"{r['model']}" + (f"_seed_{r['seed']}" if r["seed"] is not None else "")
        confusions[key] = r["confusion"]
    (cfg.results_dir / "b2_confusion_matrices.json").write_text(json.dumps(confusions, indent=2))

    (cfg.results_dir / "b2_tuning.json").write_text(json.dumps(tuning_report, indent=2))
    (cfg.results_dir / "b2_statistical_tests.json").write_text(json.dumps(stats, indent=2))
    (cfg.results_dir / "b2_bootstrap_cis.json").write_text(json.dumps({"xgb_seed42": boot}, indent=2))
    (cfg.results_dir / "b2_leakage_comparison.json").write_text(json.dumps(leakage, indent=2))

    config_out = {
        "schema": "b2-config-v1",
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "git_commit": git_commit(),
        "device": xgb_device(),
        "seeds": cfg.seeds,
        "n_iter_tuning": cfg.n_iter,
        "threshold_rule": f"models use threshold {MODEL_THRESHOLD}; overlap heuristic threshold tuned on validation only",
        "tuning_cv": "StratifiedGroupKFold(5) keyed by item_idx",
        "tfidf": {"ngram_range": (1, 2), "min_df": cfg.tfidf_min_df, "max_features": cfg.tfidf_max_features, "sublinear_tf": True},
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "best_baseline_selection": stats["best_baseline_rule"],
        "inputs": data["input_hashes"],
    }
    (cfg.results_dir / "b2_run_config.json").write_text(json.dumps(config_out, indent=2))

    logger.info(f"Saved B2 artifacts to {cfg.results_dir}")
    print("\n" + "=" * 96)
    print(" B2 — Corrected baselines + artifact controls (test set; seeds 42/123/456)")
    print("=" * 96)
    display_cols = [c for c in ["model", "deterministic", "precision_mean", "recall_mean", "f1_mean",
                                "auroc_mean", "pr_auc_mean", "mcc_mean", "ece_mean", "threshold"] if c in comparison_df.columns]
    print(comparison_df[display_cols].round(4).to_string())
    print("=" * 96)
    return {"comparison": comparison, "stats": stats, "leakage": leakage, "tuning": tuning_report}


def main():
    parser = argparse.ArgumentParser(description="B2 corrected baselines + artifact controls")
    parser.add_argument("--smoke-test", action="store_true", help="tiny synthetic run")
    parser.add_argument("--seeds", default=None, help="comma-separated seeds (default 42,123,456)")
    parser.add_argument("--n-iter", type=int, default=None, help="random search iterations (default 30)")
    args = parser.parse_args()

    if args.smoke_test:
        cfg = B2Config(
            results_dir=ROOT / "artifacts" / "results" / "b2_smoke",
            models_dir=ROOT / "artifacts" / "models" / "b2_smoke",
            seeds=[42],
            n_iter=2,
            tuning_grid=SMOKE_GRID,
            tfidf_max_features=500,
            tfidf_min_df=1,
            smoke=True,
        )
        data = build_synthetic()
    else:
        cfg = B2Config()
        if args.seeds:
            cfg.seeds = [int(s) for s in args.seeds.split(",")]
        if args.n_iter:
            cfg.n_iter = args.n_iter
        data = load_and_validate()
    run_experiment(cfg, data)


if __name__ == "__main__":
    main()

