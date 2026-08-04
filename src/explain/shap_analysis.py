"""
SHAP explainability analysis for HaluRISC (blueprint A9, roadmap Phase 6).

Produces (saved to artifacts/figures + artifacts/results):
  - Global: SHAP beeswarm summary + mean|SHAP| bar chart
  - Local: waterfall plots for 3 hand-picked test cases
  - ROC / PR curves + reliability diagram (with ECE/Brier annotation)
  - shap_summary.json (global top features + per-case local SHAP for the dashboard)

Run (repo root, .venv):
  python src/explain/shap_analysis.py
"""

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("shap_analysis")

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "processed" / "features_full.parquet"
FEATURES_FALLBACK = ROOT / "data" / "processed" / "features_core.parquet"
MODELS_DIR = ROOT / "artifacts" / "models"
RESULTS_DIR = ROOT / "artifacts" / "results"
FIGURES_DIR = ROOT / "artifacts" / "figures"

SHAP_SUBSAMPLE = 1000  # kept for reference; full test set is used (tree explainer is fast)
N_TOP_FEATURES = 10


def load_test_set():
    path = FEATURES_PATH if FEATURES_PATH.exists() else FEATURES_FALLBACK
    df = pd.read_parquet(path)
    clean = pd.read_parquet(ROOT / "data" / "processed" / "qa_clean.parquet")
    text_cols = [c for c in ["question", "answer", "context"] if c in clean.columns]
    if text_cols:
        df = pd.concat([df, clean[text_cols]], axis=1)
    feature_cols = json.loads((MODELS_DIR / "feature_names.json").read_text())
    test_df = df[df["split"] == "test"].copy()
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values
    return test_df, X_test, y_test, feature_cols


def case_indexes(y_prob: np.ndarray) -> dict:
    """3 hand-picked cases: clear hallucination, clearly correct, borderline."""
    idx_high = int(np.argmax(y_prob))
    idx_low = int(np.argmin(y_prob))
    idx_border = int(np.argmin(np.abs(y_prob - 0.5)))
    return {"high_risk": idx_high, "low_risk": idx_low, "borderline": idx_border}


def plot_roc_pr(y_test, y_prob, out: Path):
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(fpr, tpr, lw=2, color="#8b5cf6")
    axes[0].plot([0, 1], [0, 1], ls="--", color="gray", alpha=0.6)
    axes[0].set_title(f"ROC (AUC={roc_auc_score(y_test, y_prob):.4f})")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[1].plot(rec, prec, lw=2, color="#6366f1")
    axes[1].set_title(f"PR curve (AUC={average_precision_score(y_test, y_prob):.4f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out.name}")


def plot_reliability(y_test, y_prob, out: Path, n_bins: int = 10):
    bins = np.linspace(0, 1, n_bins + 1)
    idxs = np.clip(np.searchsorted(bins, y_prob, side="right") - 1, 0, n_bins - 1)
    confs, accs, counts = [], [], []
    for b in range(n_bins):
        mask = idxs == b
        if mask.sum() == 0:
            continue
        confs.append(y_prob[mask].mean())
        accs.append(y_test[mask].mean())
        counts.append(mask.sum())

    from src.models.train_pipeline import ece
    from sklearn.metrics import brier_score_loss

    ece_val = ece(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", alpha=0.7, label="Perfect calibration")
    ax.plot(confs, accs, marker="o", lw=2, color="#8b5cf6", label="Model")
    for c, a, n in zip(confs, accs, counts):
        ax.annotate(str(int(n)), (c, a), textcoords="offset points", xytext=(4, 4), fontsize=8, alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence (predicted probability)")
    ax.set_ylabel("Accuracy (empirical frequency)")
    ax.set_title(f"Reliability diagram\nECE={ece_val:.4f} | Brier={brier:.4f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out.name}")
    return ece_val, brier


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    import shap

    test_df, X_test, y_test, feature_cols = load_test_set()
    raw = joblib.load(MODELS_DIR / "model_xgboost_raw.joblib")
    y_prob = raw.predict_proba(X_test)[:, 1]
    logger.info(f"Test set: {len(X_test)} samples, {len(feature_cols)} features")

    # ---- Global SHAP (full test set; tree explainer is cheap) ----
    explainer = shap.TreeExplainer(raw)
    shap_values = explainer.shap_values(X_test)

    # Beeswarm summary
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False, max_display=15)
    plt.savefig(FIGURES_DIR / "fig_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Mean |SHAP| bar
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [feature_cols[i] for i in order[:N_TOP_FEATURES]][::-1],
        mean_abs[order[:N_TOP_FEATURES]][::-1],
        color="#8b5cf6",
    )
    ax.set_title("Mean |SHAP| feature importance")
    ax.set_xlabel("mean |SHAP value|")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Local: 3 waterfall cases ----
    cases = case_indexes(y_prob)
    case_shap = {}
    for label, idx in cases.items():
        shap.waterfall_plot(
            shap.Explanation(
                shap_values[idx],
                base_values=explainer.expected_value,
                data=X_test[idx],
                feature_names=feature_cols,
            ),
            max_display=10,
            show=False,
        )
        plt.savefig(FIGURES_DIR / f"fig_shap_waterfall_{label}.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Case '{label}': index={idx}, prob={y_prob[idx]:.4f}, true_label={y_test[idx]}")
        case_shap[label] = {
            "sample_id": str(test_df.iloc[idx]["sample_id"]),
            "question": str(test_df.iloc[idx]["question"])[:200],
            "answer": str(test_df.iloc[idx]["answer"])[:200],
            "probability": float(y_prob[idx]),
            "true_label": int(y_test[idx]),
        }

    # ---- Calibration & ranking figures ----
    ece_val, brier_val = plot_reliability(y_test, y_prob, FIGURES_DIR / "fig_reliability.png")
    plot_roc_pr(y_test, y_prob, FIGURES_DIR / "fig_roc_pr.png")

    # ---- Machine-readable summary for the dashboard ----
    summary = {
        "model_version": "xgboost-v1.0",
        "n_test": int(len(X_test)),
        "ece": ece_val,
        "brier": brier_val,
        "top_features": [
            {"feature": feature_cols[i], "mean_abs_shap": float(mean_abs[i])}
            for i in order[:N_TOP_FEATURES]
        ],
        "cases": case_shap,
    }
    with open(RESULTS_DIR / "shap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved shap_summary.json and figures to {FIGURES_DIR}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
