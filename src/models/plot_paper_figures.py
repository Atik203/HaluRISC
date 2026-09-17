"""
Regenerate the two figures whose plotted content did not match their captions.

  1. b4/calibration_shift.png            target recalibration on RAGTruth QA
     Panel 1: raw score histogram (900-row test set, threshold marked)
     Panel 2: target-isotonic calibrated score histogram
     Panel 3: ECE and Brier bars for raw, source Platt, target Platt, target isotonic

  2. b3/transfer_score_distributions.png raw model scores, source + zero-shot corpora
     Panels: HaluEval test | RAGTruth QA test | RAGTruth all | FaithBench

Both files are written to artifacts/figures/ and copied into report/figures/
(both manuscripts resolve figures from report/figures first).

Run (repo root, .venv):
  python src/models/plot_paper_figures.py
"""

import json
import shutil
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIG_ARTIFACTS = ROOT / "artifacts" / "figures"
FIG_REPORT = ROOT / "report" / "figures"

THRESHOLD = 0.5
SEED = 42
DPI = 150


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def calibration_shift() -> None:
    b4 = pd.read_parquet(ROOT / "artifacts" / "results" / "b4" / "b4_predictions.parquet")
    qa = b4[(b4["subset"] == "ragtruth_qa_test") & (b4["method"] == "raw")]
    y = qa["label"].to_numpy()
    raw = qa["score"].to_numpy()

    cal = joblib.load(ROOT / "artifacts" / "models" / "b4" / "calibrator_isotonic_target_ragtruth_qa_seed_42.joblib")
    target = np.clip(cal.predict(raw), 0.0, 1.0)

    metrics = json.loads((ROOT / "artifacts" / "results" / "b4" / "b4_target_calibration.json").read_text(encoding="utf-8"))["methods"]
    rows = [
        ("Raw", metrics["raw_reference"]),
        ("Source Platt", metrics["source_platt_reference"]),
        ("Target Platt", metrics["platt"]),
        ("Target isotonic", metrics["isotonic"]),
    ]

    print(f"sanity: raw ECE {_ece(y, raw):.4f} (json {metrics['raw_reference']['ece_mean']:.4f})")
    print(f"sanity: target isotonic ECE {_ece(y, target):.4f} (json {metrics['isotonic']['ece_mean']:.4f})")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    for ax, scores, title in (
        (axes[0], raw, "Raw scores (RAGTruth QA test)"),
        (axes[1], target, "Target-isotonic scores (RAGTruth QA test)"),
    ):
        ax.hist(scores[y == 0], bins=np.linspace(0, 1, 41), alpha=0.65, label="label 0", color="#4C72B0")
        ax.hist(scores[y == 1], bins=np.linspace(0, 1, 41), alpha=0.65, label="label 1", color="#DD8452")
        ax.axvline(THRESHOLD, color="crimson", linestyle="--", linewidth=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("score")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)

    x = np.arange(len(rows))
    width = 0.38
    ece = [r[1]["ece_mean"] for r in rows]
    brier = [r[1]["brier_mean"] for r in rows]
    axes[2].bar(x - width / 2, ece, width, label="ECE", color="#4C72B0")
    axes[2].bar(x + width / 2, brier, width, label="Brier", color="#DD8452")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([r[0] for r in rows], fontsize=8)
    axes[2].set_ylim(0, 1.0)
    axes[2].set_title("Calibration error before and after", fontsize=10)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    out = FIG_ARTIFACTS / "b4" / "calibration_shift.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    shutil.copyfile(out, FIG_REPORT / "b4" / "calibration_shift.png")
    print(f"wrote {out.relative_to(ROOT)} and report/figures/b4/calibration_shift.png")


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

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.0))
    for ax, (name, d) in zip(axes, panels):
        y = d["label"].to_numpy()
        s = d["score"].to_numpy()
        ax.hist(s[y == 0], bins=np.linspace(0, 1, 41), alpha=0.65, label="label 0", color="#4C72B0")
        ax.hist(s[y == 1], bins=np.linspace(0, 1, 41), alpha=0.65, label="label 1", color="#DD8452")
        ax.axvline(THRESHOLD, color="crimson", linestyle="--", linewidth=1.2)
        ax.set_title(f"{name}\n(n = {len(d):,})", fontsize=10)
        ax.set_xlabel("raw model score")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)

    fig.tight_layout()
    out = FIG_ARTIFACTS / "b3" / "transfer_score_distributions.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    shutil.copyfile(out, FIG_REPORT / "b3" / "transfer_score_distributions.png")
    print(f"wrote {out.relative_to(ROOT)} and report/figures/b3/transfer_score_distributions.png")


if __name__ == "__main__":
    calibration_shift()
    transfer_scores()
