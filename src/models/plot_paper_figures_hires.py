"""Regenerate the paper figures at print size with readable fonts and 600 DPI.

Design rule: the figure size in inches equals the width it occupies in the
manuscript, so matplotlib font sizes print at their true point sizes, and
dpi=600 gives exactly 600 DPI at that width. This fixes the earlier problem
where 16-18-inch-wide figures were drawn with 8-10 pt fonts and shrunk onto
the page at about 3 pt.

Writes ONLY to Journal_Paper/figures/. The artifact figures under
artifacts/figures/ are hashed in docs/manifest.frozen.json and are never
overwritten.

  figures/b4/reliability_diagrams.png        6.5 in wide, 600 DPI, full text width
  figures/b4/calibration_shift.png           6.5 in wide, 600 DPI, full text width
  figures/b3/transfer_score_distributions.png 6.5 in wide, 600 DPI, full text width
  figures/fig_shap_importance_b5.png         5.2 in wide, 600 DPI, 0.8 text width

Run from the repo root:

  .venv/Scripts/python.exe src/models/plot_paper_figures_hires.py
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIG_PAPER = ROOT / "Journal_Paper" / "figures"
THRESHOLD = 0.5
SEED = 42
DPI = 600

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "figure.dpi": 120,
    }
)


def _save(fig, out: Path) -> None:
    fig.savefig(out, dpi=DPI)
    px = int(round(fig.get_size_inches()[0] * DPI))
    print(f"wrote {out.relative_to(ROOT)}  {px} px wide  {DPI} DPI at print width")
    plt.close(fig)


def _legend_hist(ax) -> None:
    leg = ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    leg.get_frame().set_linewidth(0.5)


def calibration_shift() -> None:
    b4 = pd.read_parquet(ROOT / "artifacts" / "results" / "b4" / "b4_predictions.parquet")
    qa = b4[(b4["subset"] == "ragtruth_qa_test") & (b4["method"] == "raw")]
    y = qa["label"].to_numpy()
    raw = qa["score"].to_numpy()
    cal = joblib.load(
        ROOT / "artifacts" / "models" / "b4" / "calibrator_isotonic_target_ragtruth_qa_seed_42.joblib"
    )
    target = np.clip(cal.predict(raw), 0.0, 1.0)
    metrics = json.loads(
        (ROOT / "artifacts" / "results" / "b4" / "b4_target_calibration.json").read_text(encoding="utf-8")
    )["methods"]
    rows = [
        ("Raw", metrics["raw_reference"]),
        ("Src Platt", metrics["source_platt_reference"]),
        ("Tgt Platt", metrics["platt"]),
        ("Tgt isotonic", metrics["isotonic"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.3))
    for ax, scores, title in (
        (axes[0], raw, "Raw scores"),
        (axes[1], target, "Target-isotonic scores"),
    ):
        ax.hist(scores[y == 0], bins=np.linspace(0, 1, 41), alpha=0.65, label="label 0", color="#4C72B0")
        ax.hist(scores[y == 1], bins=np.linspace(0, 1, 41), alpha=0.65, label="label 1", color="#DD8452")
        ax.axvline(THRESHOLD, color="crimson", linestyle="--", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("score")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("count")
    _legend_hist(axes[0])
    _legend_hist(axes[1])
    x = np.arange(len(rows))
    width = 0.38
    ece = [r[1]["ece_mean"] for r in rows]
    brier = [r[1]["brier_mean"] for r in rows]
    axes[2].bar(x - width / 2, ece, width, label="ECE", color="#4C72B0")
    axes[2].bar(x + width / 2, brier, width, label="Brier", color="#DD8452")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([r[0] for r in rows], fontsize=8, rotation=18, ha="right")
    axes[2].set_ylim(0, 1.0)
    axes[2].set_title("Error before and after")
    axes[2].legend(loc="upper right", fontsize=7, framealpha=0.9).get_frame().set_linewidth(0.5)
    fig.tight_layout()
    _save(fig, FIG_PAPER / "b4" / "calibration_shift.png")


def transfer_scores() -> None:
    b2 = pd.read_parquet(ROOT / "artifacts" / "results" / "b2" / "b2_predictions.parquet")
    b3 = pd.read_parquet(ROOT / "artifacts" / "results" / "b3" / "b3_predictions.parquet")
    hal = b2[(b2["model"] == "xgboost") & (b2["seed"] == SEED) & (b2["split"] == "test")]
    b3s = b3[b3["model"] == f"xgboost_seed_{SEED}"]
    rt = b3s[b3s["source_dataset"] == "ragtruth"]
    qa = rt[(rt["task"] == "qa") & (rt["official_split"] == "test")]
    fb = b3s[b3s["source_dataset"] == "faithbench"]
    panels = [
        ("HaluEval test (source)", hal),
        ("RAGTruth QA test", qa),
        ("RAGTruth all", rt),
        ("FaithBench", fb),
    ]
    for name, d in panels:
        print(f"{name}: n={len(d)}")
    assert len(hal) == 3000 and len(qa) == 900 and len(rt) == 17790 and len(fb) == 750
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.15))
    for i, (ax, (name, d)) in enumerate(zip(axes, panels)):
        y = d["label"].to_numpy()
        s = d["score"].to_numpy()
        ax.hist(s[y == 0], bins=np.linspace(0, 1, 41), alpha=0.65, label="label 0", color="#4C72B0")
        ax.hist(s[y == 1], bins=np.linspace(0, 1, 41), alpha=0.65, label="label 1", color="#DD8452")
        ax.axvline(THRESHOLD, color="crimson", linestyle="--", linewidth=1.2)
        ax.set_title(f"{name}\n(n = {len(d):,})")
        ax.set_xlabel("raw model score")
        ax.set_xlim(0, 1)
        if i == 0:
            ax.set_ylabel("count")
            _legend_hist(ax)
    fig.tight_layout()
    _save(fig, FIG_PAPER / "b3" / "transfer_score_distributions.png")


def reliability_diagrams() -> None:
    reliability = json.loads(
        (ROOT / "artifacts" / "results" / "b4" / "b4_reliability_data.json").read_text(encoding="utf-8")
    )
    panels = [
        ("halueval_test", "HaluEval test"),
        ("ragtruth_qa_test", "RAGTruth QA test"),
        ("faithbench", "FaithBench"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.3), squeeze=False)
    for i, (ax, (name, title)) in enumerate(zip(axes[0], panels)):
        if name not in reliability:
            continue
        for method, color in (("raw", "gray"), ("platt", "tab:blue"), ("isotonic", "tab:orange")):
            curve = reliability[name][method]
            conf = [c["confidence"] for c in curve if c["confidence"] is not None]
            acc = [c["accuracy"] for c in curve if c["accuracy"] is not None]
            ax.plot(conf, acc, marker="o", ms=2.5, label=method, color=color)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_xlabel("confidence")
        if i == 0:
            ax.set_ylabel("accuracy")
            ax.legend(loc="upper left", fontsize=7, framealpha=0.9).get_frame().set_linewidth(0.5)
    fig.tight_layout()
    _save(fig, FIG_PAPER / "b4" / "reliability_diagrams.png")


def shap_importance() -> None:
    data = json.loads(
        (ROOT / "artifacts" / "results" / "b5" / "b5_feature_importance.json").read_text(encoding="utf-8")
    )
    items = sorted(data["mean_abs_shap"].items(), key=lambda kv: kv[1], reverse=True)[:12]
    names = [k for k, _ in items]
    values = [v for k, v in items]
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.barh(range(len(values)), values, color="#4f46e5")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("mean |SHAP|")
    ax.set_title("Top-12 features by mean |SHAP| (B-run, seed 42)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, FIG_PAPER / "fig_shap_importance_b5.png")


if __name__ == "__main__":
    calibration_shift()
    transfer_scores()
    reliability_diagrams()
    shap_importance()
    print("done: paper figures redrawn at print size, readable fonts, 600 DPI; artifacts/figures untouched")
