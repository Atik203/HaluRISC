"""
B3 — Cross-domain robustness (roadmap §14 B3).

Zero-shot evaluation of the B2 HaluEval-trained XGBoost models on external
datasets from the B1 unified layer:
  - RAGTruth QA official test  (primary external benchmark, group by source_id)
  - RAGTruth all tasks/splits  (secondary descriptive transfer analysis)
  - FaithBench summarization   (locked external stress test, CC BY-NC-SA)

Rules enforced here:
  - NO training, threshold tuning, calibration, vectorizer, or normalization
    fitting on external data (fixed threshold 0.5, raw XGBoost probabilities).
  - Subgroups are predeclared; metrics only reported for groups with >= 100
    rows and >= 20 source groups (else counts only).
  - Confidence intervals use source-group bootstrap (1000 resamples).
  - B2/Version A artifacts are never overwritten (outputs under
    artifacts/{results,figures}/b3/).

Run (repo root, .venv):
  python src/models/run_b3_cross_domain.py
  python src/models/run_b3_cross_domain.py --skip-features   # reuse cached external features
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("b3_cross_domain")

from src.models.config import (  # noqa: E402
    BOOTSTRAP_SEED,
    DATA_PROCESSED,
    FIGURES_DIR,
    MODELS_DIR,
    N_BOOTSTRAP,
    RESULTS_DIR,
    ROOT,
)
from src.models.train_pipeline import ece  # noqa: E402
from src.models.run_b2_baselines import evaluate  # noqa: E402

UNIFIED = DATA_PROCESSED / "unified_records.parquet"
B2_MODELS_DIR = MODELS_DIR / "b2"
B2_RESULTS_DIR = RESULTS_DIR / "b2"
B3_RESULTS = RESULTS_DIR / "b3"
B3_FIGURES = FIGURES_DIR / "b3"
FEATURES_CACHE = DATA_PROCESSED / "b3_external_features.parquet"
FEATURES_CACHE_META = DATA_PROCESSED / "b3_external_features.meta.json"

MODEL_THRESHOLD = 0.5
B2_SEEDS = [42, 123, 456]

MIN_SUBGROUP_ROWS = 100
MIN_SUBGROUP_GROUPS = 20

CONTEXT_WORD_BINS = [("lt_128", 0, 128), ("128_511", 128, 512), ("512_1023", 512, 1024), ("ge_1024", 1024, None)]
ANSWER_WORD_BINS = [("lt_32", 0, 32), ("32_127", 32, 128), ("ge_128", 128, None)]


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


def word_bin(text: str, bins) -> str:
    n = len(str(text).split())
    for name, lo, hi in bins:
        if hi is None:
            if n >= lo:
                return name
        elif lo <= n < hi:
            return name
    return bins[-1][0]


def load_unified() -> pd.DataFrame:
    if not UNIFIED.exists():
        raise FileNotFoundError(f"{UNIFIED} not found. Run src/data/prepare_unified.py first.")
    df = pd.read_parquet(UNIFIED)
    df["span_annotations"] = df["span_annotations"].fillna("[]")
    return df


def select_datasets(df: pd.DataFrame) -> dict:
    """Predeclared B3 evaluation subsets."""
    rag = df[df["source_dataset"] == "ragtruth"]
    subsets = {
        "ragtruth_qa_test": rag[(rag["task"] == "qa") & (rag["official_split"] == "test")],
        "ragtruth_all": rag,
        "ragtruth_train": rag[rag["official_split"] == "train"],
        "faithbench": df[df["source_dataset"] == "faithbench"],
    }
    for task in ("qa", "summarization", "data_to_text"):
        subsets[f"ragtruth_task_{task}"] = rag[rag["task"] == task]
    for name, sub in subsets.items():
        logger.info(f"{name}: {len(sub)} rows / {sub['source_group_id'].nunique()} groups")
    return subsets


def load_b2_model_config() -> tuple:
    """Returns (feature_cols, b2_config)."""
    cfg_path = B2_RESULTS_DIR / "b2_run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} not found. Run src/models/run_b2_baselines.py first.")
    cfg = json.loads(cfg_path.read_text())
    return list(cfg["feature_cols"]), cfg


def load_b2_models() -> dict:
    models = {}
    for seed in B2_SEEDS:
        path = B2_MODELS_DIR / f"xgboost_seed_{seed}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run src/models/run_b2_baselines.py first.")
        models[seed] = joblib.load(path)
        logger.info(f"Loaded B2 XGBoost seed {seed}")
    return models


def extract_or_load_external_features(df: pd.DataFrame, feature_cols: list, device: str, batch_size: int, skip_features: bool = False) -> pd.DataFrame:
    """Extract the 26 B2 features on external rows (cached; cache keyed by input hash)."""
    input_sha = sha256(UNIFIED)

    if skip_features:
        if not FEATURES_CACHE.exists():
            raise FileNotFoundError("--skip-features but no cache found")
        cache_meta = json.loads(FEATURES_CACHE_META.read_text())
        if cache_meta.get("input_sha256") != input_sha:
            raise ValueError("cached external features do not match the current unified parquet")
        cached = pd.read_parquet(FEATURES_CACHE)
        logger.info(f"Using cached external features ({len(cached)} rows)")
        return df.merge(cached[["sample_id"] + feature_cols], on="sample_id", how="left")

    from src.features.extract_features import extract_full_feature_set, load_heavy_models

    t0 = time.time()
    models = load_heavy_models(device=device)
    work = df.copy()
    work["item_idx"] = work["source_group_id"]
    work["label"] = work["label"].astype(int)
    work["split"] = work["official_split"].fillna("")
    feats = extract_full_feature_set(work, models, batch_size=batch_size)
    logger.info(f"External feature extraction done in {time.time() - t0:.0f}s")

    cols = ["sample_id"] + feature_cols
    merged = df.merge(feats[cols], on="sample_id", how="left")
    missing = merged[feature_cols].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} rows missing extracted features")
    merged.to_parquet(FEATURES_CACHE, index=False)
    FEATURES_CACHE_META.write_text(json.dumps({
        "input_sha256": input_sha,
        "n_rows": int(len(merged)),
        "feature_cols": feature_cols,
        "extracted_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "device": device,
        "batch_size": batch_size,
    }, indent=2))
    logger.info(f"Cached external features to {FEATURES_CACHE}")
    return merged


def predict_zero_shot(model, df: pd.DataFrame, feature_cols: list) -> tuple:
    X = df[feature_cols].values
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= MODEL_THRESHOLD).astype(int)
    return proba, preds


def aggregate_metrics(y_true, proba, preds) -> dict:
    """Classification + calibration diagnostics for one subset (threshold fixed at 0.5)."""
    return evaluate(np.asarray(y_true), preds, proba)


def mean_std_over_seeds(metric_rows: list) -> dict:
    keys = ["precision", "recall", "f1", "auroc", "pr_auc", "mcc", "ece"]
    out = {"n_seeds": len(metric_rows)}
    for k in keys:
        vals = [r[k] for r in metric_rows if r.get(k) is not None]
        out[f"{k}_mean"] = float(np.mean(vals)) if vals else None
        out[f"{k}_std"] = float(np.std(vals)) if vals else None
    return out


def group_bootstrap_cis(y_true, proba, groups, n: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> dict:
    """Group-aware bootstrap 95% CIs for F1 and AUROC.

    Groups are sampled WITH replacement and every row of a sampled group is
    kept, including duplicate group draws (np.isin would deduplicate and
    silently turn this into a smaller resample).
    """
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    y = np.asarray(y_true)
    p = np.asarray(proba)
    g = np.asarray(groups)
    row_ids = {grp: np.where(g == grp)[0] for grp in unique_groups}
    f1s, aucs = [], []
    for _ in range(n):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([row_ids[grp] for grp in sampled])
        if len(np.unique(y[idx])) < 2 or len(np.unique(p[idx])) < 2:
            continue
        f1s.append(f1_score(y[idx], (p[idx] >= MODEL_THRESHOLD).astype(int), zero_division=0))
        aucs.append(roc_auc_score(y[idx], p[idx]))
    out = {"n_resamples": len(f1s), "groups": len(unique_groups)}
    if f1s:
        out["f1_ci"] = [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))]
        out["auroc_ci"] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
    else:
        out["f1_ci"] = None
        out["auroc_ci"] = None
    return out


def subgroup_metrics(df: pd.DataFrame, proba, preds, dimension: str, bin_fn=None) -> list:
    """Metrics per subgroup with minimum-size rules; small subgroups returned as counts."""
    y_true = df["label"].values
    rows = []
    if bin_fn is not None:
        df = df.copy()
        df["_bin"] = df["answer"].map(lambda t: bin_fn(t))
        key_col = "_bin"
    else:
        key_col = dimension
    for key, sub in df.groupby(key_col):
        idx = df.index.isin(sub.index)
        y_sub = y_true[idx]
        p_sub = proba[idx]
        n_groups = sub["source_group_id"].nunique()
        if len(sub) < MIN_SUBGROUP_ROWS or n_groups < MIN_SUBGROUP_GROUPS:
            rows.append({"dimension": dimension, "subgroup": str(key), "n_rows": int(len(sub)),
                         "n_groups": int(n_groups), "reported": False,
                         "reason": f"below minimum (rows<{MIN_SUBGROUP_ROWS} or groups<{MIN_SUBGROUP_GROUPS})"})
            continue
        if len(np.unique(y_sub)) < 2 or len(np.unique(p_sub)) < 2:
            rows.append({"dimension": dimension, "subgroup": str(key), "n_rows": int(len(sub)),
                         "n_groups": int(n_groups), "reported": False, "reason": "degenerate subset"})
            continue
        m = aggregate_metrics(y_sub, p_sub, (p_sub >= MODEL_THRESHOLD).astype(int))
        rows.append({"dimension": dimension, "subgroup": str(key), "n_rows": int(len(sub)),
                     "n_groups": int(n_groups), "reported": True,
                     **{k: m[k] for k in ("precision", "recall", "f1", "auroc", "pr_auc", "mcc", "ece")}})
    return rows


def span_type_subgroups(df: pd.DataFrame, proba, preds) -> list:
    """RAGTruth rows may belong to several span-type groups; membership is overlapping."""
    import json as _json

    rows = []
    y_true = df["label"].values
    types = ["Evident Conflict", "Evident Baseless Info", "Subtle Conflict", "Subtle Baseless Info"]
    masks = {}
    for t in types:
        masks[t] = df["span_annotations"].map(lambda s: t in s).values
    masks["no_span"] = df["span_annotations"].map(lambda s: s.strip() in ("[]", "")).values
    for t, mask in masks.items():
        n = int(mask.sum())
        n_groups = int(df[mask]["source_group_id"].nunique()) if n else 0
        if n < MIN_SUBGROUP_ROWS or n_groups < MIN_SUBGROUP_GROUPS:
            rows.append({"dimension": "label_type", "subgroup": t, "n_rows": n, "n_groups": n_groups,
                         "reported": False, "reason": "below minimum"})
            continue
        y_sub, p_sub = y_true[mask], proba[mask]
        if len(np.unique(y_sub)) < 2 or len(np.unique(p_sub)) < 2:
            rows.append({"dimension": "label_type", "subgroup": t, "n_rows": n, "n_groups": n_groups,
                         "reported": False, "reason": "degenerate subset"})
            continue
        m = aggregate_metrics(y_sub, p_sub, (p_sub >= MODEL_THRESHOLD).astype(int))
        rows.append({"dimension": "label_type", "subgroup": t, "n_rows": n, "n_groups": n_groups,
                     "reported": True, "note": "overlapping membership",
                     **{k: m[k] for k in ("precision", "recall", "f1", "auroc", "pr_auc", "mcc", "ece")}})
    return rows


def faithbench_sensitivity(df: pd.DataFrame, proba) -> dict:
    """FaithBench label-mapping sensitivity: predictions fixed, only labels change."""
    from src.data.mappings import FAITHBENCH_SENSITIVITY_CONFIGS, faithbench_label

    import json as _json

    out = {}
    for cfg_name, cfg in FAITHBENCH_SENSITIVITY_CONFIGS.items():
        labels = df["span_annotations"].map(lambda s: faithbench_label(_json.loads(s), **cfg)).astype(int).values
        preds = (proba >= MODEL_THRESHOLD).astype(int)
        m = aggregate_metrics(labels, proba, preds)
        out[cfg_name] = {"n_positive": int(labels.sum()), "n_negative": int((labels == 0).sum()),
                         "f1": m["f1"], "auroc": m["auroc"], "mcc": m["mcc"]}
    return out


def sample_error_cases(df: pd.DataFrame, proba, preds, cap: int = 10) -> list:
    rng = np.random.default_rng(42)
    y = df["label"].values
    cases = []
    groups = {
        "false_positive": (y == 0) & (preds == 1),
        "false_negative": (y == 1) & (preds == 0),
        "high_conf_correct": ((y == preds) & (np.maximum(proba, 1 - proba) >= 0.8)),
        "high_conf_incorrect": ((y != preds) & (np.maximum(proba, 1 - proba) >= 0.8)),
    }
    for name, mask in groups.items():
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        chosen = rng.choice(idx, size=min(cap, len(idx)), replace=False)
        for i in chosen:
            r = df.iloc[i]
            cases.append({
                "group": name, "sample_id": r["sample_id"], "source_dataset": r["source_dataset"],
                "task": r["task"], "domain": r["domain"], "generator_model": r["generator_model"],
                "question": str(r["question"])[:500], "context": str(r["context"])[:2000],
                "answer": str(r["answer"])[:2000], "label": int(y[i]),
                "prediction": int(preds[i]), "raw_score": round(float(proba[i]), 4),
                "span_annotations": r["span_annotations"][:2000], "source_group_id": r["source_group_id"],
            })
    return cases


def transfer_comparison(subset_metrics: dict, b2_comp: dict) -> list:
    in_f1 = b2_comp["xgboost"]["f1_mean"]
    in_auc = b2_comp["xgboost"]["auroc_mean"]
    rows = []
    for name, m in subset_metrics.items():
        if not m or m.get("f1_mean") is None:
            continue
        rows.append({
            "subset": name, "n_rows": m["n_rows"], "n_groups": m["n_groups"],
            "f1": round(m["f1_mean"], 4), "auroc": round(m["auroc_mean"], 4),
            "delta_f1_vs_in_domain": round(m["f1_mean"] - in_f1, 4),
            "delta_auroc_vs_in_domain": round(m["auroc_mean"] - in_auc, 4),
            "predicted_positive_rate": round(m["predicted_positive_rate"], 4),
            "label_positive_rate": round(m["label_positive_rate"], 4),
        })
    return rows


def make_figures(subset_metrics: dict, subgroup_rows: list, subset_scores: dict, transfer_rows: list):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(B3_FIGURES, exist_ok=True)

    names = list(subset_metrics.keys())
    f1s = [subset_metrics[n]["f1_mean"] for n in names if subset_metrics[n].get("f1_mean") is not None]
    aucs = [subset_metrics[n]["auroc_mean"] for n in names if subset_metrics[n].get("auroc_mean") is not None]
    label_names = [n for n in names if subset_metrics[n].get("f1_mean") is not None]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(label_names))
    ax.bar(x - 0.2, f1s, 0.4, label="F1")
    ax.bar(x + 0.2, aucs, 0.4, label="AUROC")
    ax.set_xticks(x, label_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("In-domain vs out-of-domain (zero-shot, B2 XGBoost)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(B3_FIGURES / "in_domain_vs_ood.png", dpi=150)
    plt.close(fig)

    reported = [r for r in subgroup_rows if r["reported"] and r["f1"] is not None]
    if reported:
        dims = sorted(set(r["dimension"] for r in reported))
        n_panels = len(dims)
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5), squeeze=False)
        for ax, dim in zip(axes[0], dims):
            sub = [r for r in reported if r["dimension"] == dim]
            ax.bar([r["subgroup"] for r in sub], [r["f1"] for r in sub])
            ax.set_xticks(range(len(sub)), [r["subgroup"] for r in sub], rotation=20, ha="right")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"F1 by {dim}")
            ax.axhline(subset_metrics["ragtruth_qa_test"]["f1_mean"] if "ragtruth_qa_test" in subset_metrics else 0.5,
                       color="red", ls="--", lw=1)
        fig.tight_layout()
        fig.savefig(B3_FIGURES / "subgroup_performance.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, len(subset_scores), figsize=(5.5 * len(subset_scores), 4), squeeze=False)
    for ax, (name, (scores, labels)) in zip(axes[0], subset_scores.items()):
        ax.hist(scores[labels == 0], bins=40, alpha=0.6, label="label 0")
        ax.hist(scores[labels == 1], bins=40, alpha=0.6, label="label 1")
        ax.axvline(MODEL_THRESHOLD, color="red", ls="--")
        ax.set_title(f"{name} score distribution")
        ax.set_xlim(0, 1)
        ax.legend()
    fig.tight_layout()
    fig.savefig(B3_FIGURES / "transfer_score_distributions.png", dpi=150)
    plt.close(fig)

    ctx = [r for r in subgroup_rows if r["dimension"] == "context_length" and r["reported"]]
    if ctx:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar([r["subgroup"] for r in ctx], [r["f1"] for r in ctx])
        ax.set_ylim(0, 1.05)
        ax.set_title("F1 by context length (words)")
        fig.tight_layout()
        fig.savefig(B3_FIGURES / "context_length_robustness.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="B3 cross-domain zero-shot evaluation")
    parser.add_argument("--device", default="cuda", help="feature extraction device (cuda|cpu)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--skip-features", action="store_true", help="reuse cached external features")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()

    os.makedirs(B3_RESULTS, exist_ok=True)
    os.makedirs(B3_FIGURES, exist_ok=True)

    df = load_unified()
    subsets = select_datasets(df)
    feature_cols, b2_cfg = load_b2_model_config()

    external = pd.concat([subsets[name] for name in ("ragtruth_all", "faithbench")], ignore_index=True)
    external = external.sort_values(["source_dataset", "sample_id"]).reset_index(drop=True)

    t0 = time.time()
    external = extract_or_load_external_features(external, feature_cols, args.device, args.batch_size, args.skip_features)
    logger.info(f"Features ready in {time.time() - t0:.0f}s")

    models = load_b2_models()
    seed_probs = {}
    for seed, model in models.items():
        proba, preds = predict_zero_shot(model, external, feature_cols)
        seed_probs[seed] = {"proba": proba, "preds": preds}
        logger.info(f"Seed {seed}: zero-shot predictions done")

    subset_metrics = {}
    subset_scores = {}
    bootstrap = {}
    predictions_rows = []
    error_cases = []
    for name, sub in subsets.items():
        idx = external["sample_id"].isin(set(sub["sample_id"])).values
        sub_df = external[idx].reset_index(drop=True)
        if len(sub_df) == 0:
            subset_metrics[name] = None
            continue
        per_seed = []
        for seed in B2_SEEDS:
            p = seed_probs[seed]["proba"][idx]
            preds = seed_probs[seed]["preds"][idx]
            m = aggregate_metrics(sub_df["label"].values, p, preds)
            m["seed"] = seed
            per_seed.append(m)
        agg = mean_std_over_seeds(per_seed)
        agg["n_rows"] = int(len(sub_df))
        agg["n_groups"] = int(sub_df["source_group_id"].nunique())
        agg["predicted_positive_rate"] = float(np.mean(seed_probs[B2_SEEDS[0]]["preds"][idx]))
        agg["label_positive_rate"] = float(sub_df["label"].mean())
        subset_metrics[name] = agg
        subset_scores[name] = (seed_probs[B2_SEEDS[0]]["proba"][idx], sub_df["label"].values)
        for seed in B2_SEEDS:
            for i, r in sub_df.iterrows():
                predictions_rows.append({
                    "sample_id": r["sample_id"], "source_dataset": r["source_dataset"],
                    "source_group_id": r["source_group_id"], "task": r["task"], "domain": r["domain"],
                    "official_split": r["official_split"], "quality": r["quality"],
                    "generator_model": r["generator_model"], "label": int(r["label"]),
                    "model": f"xgboost_seed_{seed}",
                    "score": float(seed_probs[seed]["proba"][idx][i]),
                    "pred": int(seed_probs[seed]["preds"][idx][i]),
                })
        bootstrap[name] = group_bootstrap_cis(
            sub_df["label"].values, seed_probs[B2_SEEDS[0]]["proba"][idx],
            sub_df["source_group_id"].values, n=args.n_bootstrap,
        )
        logger.info(f"{name}: f1={agg['f1_mean']:.4f} auroc={agg['auroc_mean']:.4f} "
                    f"ece={agg['ece_mean']:.4f} pred_pos={agg['predicted_positive_rate']:.3f}")

    rag_idx = external["source_dataset"] == "ragtruth"
    rag_df = external[rag_idx].reset_index(drop=True)
    rag_proba = seed_probs[B2_SEEDS[0]]["proba"][rag_idx]
    rag_preds = seed_probs[B2_SEEDS[0]]["preds"][rag_idx]
    subgroup_rows = []
    for dim, col in (("task", "task"), ("official_split", "official_split"), ("domain", "domain"),
                     ("generator_model", "generator_model"), ("quality", "quality")):
        subgroup_rows += subgroup_metrics(rag_df, rag_proba, rag_preds, dim)
    subgroup_rows += subgroup_metrics(rag_df, rag_proba, rag_preds, "context_length",
                                      bin_fn=lambda t: word_bin(t, CONTEXT_WORD_BINS))
    subgroup_rows += subgroup_metrics(rag_df, rag_proba, rag_preds, "answer_length",
                                      bin_fn=lambda t: word_bin(t, ANSWER_WORD_BINS))
    subgroup_rows += span_type_subgroups(rag_df, rag_proba, rag_preds)

    fb_idx = external["source_dataset"] == "faithbench"
    fb_df = external[fb_idx].reset_index(drop=True)
    fb_proba = seed_probs[B2_SEEDS[0]]["proba"][fb_idx]
    subgroup_rows += subgroup_metrics(fb_df, fb_proba, (fb_proba >= MODEL_THRESHOLD).astype(int), "generator_model")

    for name, sub in subsets.items():
        if sub is None or len(sub) == 0:
            continue
        idx = external["sample_id"].isin(set(sub["sample_id"])).values
        sub_df = external[idx]
        error_cases += sample_error_cases(sub_df, seed_probs[B2_SEEDS[0]]["proba"][idx],
                                          seed_probs[B2_SEEDS[0]]["preds"][idx])

    sensitivity = faithbench_sensitivity(fb_df, fb_proba)

    b2_comp = json.loads((B2_RESULTS_DIR / "b2_model_comparison.json").read_text())
    transfer = transfer_comparison(subset_metrics, b2_comp)

    make_figures(subset_metrics, subgroup_rows, subset_scores, transfer)

    pred_df = pd.DataFrame(predictions_rows)
    pred_df.to_parquet(B3_RESULTS / "b3_predictions.parquet", index=False)
    (B3_RESULTS / "b3_dataset_metrics.json").write_text(json.dumps(subset_metrics, indent=2))
    pd.DataFrame({k: v for k, v in subset_metrics.items() if v is not None}).T.to_csv(B3_RESULTS / "b3_dataset_metrics.csv")
    pd.DataFrame(subgroup_rows).to_csv(B3_RESULTS / "b3_subgroup_metrics.csv", index=False)
    (B3_RESULTS / "b3_bootstrap_cis.json").write_text(json.dumps(bootstrap, indent=2))
    pd.DataFrame(transfer).to_csv(B3_RESULTS / "b3_transfer_comparison.csv", index=False)
    (B3_RESULTS / "b3_error_cases.json").write_text(json.dumps(error_cases, indent=2))
    (B3_RESULTS / "b3_label_sensitivity.json").write_text(json.dumps(sensitivity, indent=2))

    provenance = {
        "schema": "b3-config-v1",
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "git_commit": git_commit(),
        "unified_parquet_sha256": sha256(UNIFIED),
        "b2_model_hashes": {f"seed_{s}": sha256(B2_MODELS_DIR / f"xgboost_seed_{s}.joblib") for s in B2_SEEDS},
        "feature_cols": feature_cols,
        "threshold_rule": f"fixed {MODEL_THRESHOLD}; no external tuning",
        "bootstrap": {"n": args.n_bootstrap, "seed": BOOTSTRAP_SEED, "method": "source-group resampling"},
        "subgroup_minimums": {"rows": MIN_SUBGROUP_ROWS, "groups": MIN_SUBGROUP_GROUPS},
        "device": args.device,
        "batch_size": args.batch_size,
        "note": "NLI/embedding truncate long external texts at model max length (512 tokens).",
    }
    (B3_RESULTS / "b3_run_config.json").write_text(json.dumps(provenance, indent=2))

    logger.info(f"Saved B3 artifacts to {B3_RESULTS}")
    print("\n" + "=" * 96)
    print(" B3 — Cross-domain zero-shot (B2 XGBoost, threshold 0.5, seeds 42/123/456)")
    print("=" * 96)
    table = {k: v for k, v in subset_metrics.items() if v is not None}
    df_out = pd.DataFrame({
        k: {"f1": v["f1_mean"], "auroc": v["auroc_mean"], "ece": v["ece_mean"],
            "pred_pos": v["predicted_positive_rate"], "label_pos": v["label_positive_rate"],
            "n": v["n_rows"]}
        for k, v in table.items()
    }).T.round(4)
    print(df_out.to_string())
    print("=" * 96)


if __name__ == "__main__":
    main()
