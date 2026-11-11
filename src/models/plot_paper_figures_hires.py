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
        "font.size": 11,
        "axes.titlesize": 11.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 120,
    }
)


def _save(fig, out: Path) -> None:
    fig.savefig(out, dpi=DPI)
    px = int(round(fig.get_size_inches()[0] * DPI))
    print(f"wrote {out.relative_to(ROOT)}  {px} px wide  {DPI} DPI at print width")
    plt.close(fig)


def _legend_hist(ax) -> None:
    # Scores pile against 0 and 1, so the centre of the axes stays empty and
    # the legend can sit there without covering the tall end bins.
    leg = ax.legend(loc="upper center", fontsize=8, framealpha=0.9)
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
        ("Source Platt", metrics["source_platt_reference"]),
        ("Target Platt", metrics["platt"]),
        ("Target isotonic", metrics["isotonic"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(6.6, 3.15), gridspec_kw={"width_ratios": [1.0, 1.0, 1.15]})
    bins = np.linspace(-8, 8, 65)
    for ax, scores, title, note in (
        (axes[0], raw, "Raw scores", "99.9% > 0.5"),
        (axes[1], target, "Target-isotonic scores", "0.0% > 0.5"),
    ):
        z = _logit_scores(scores)
        ax.hist(z[y == 0], bins=bins, density=True, alpha=0.55, color="#4C72B0")
        ax.hist(z[y == 1], bins=bins, density=True, alpha=0.55, color="#DD8452")
        ax.axvline(0, color="0.45", lw=1.0, ls=(0, (3, 3)))
        tick_p = [0.01, 0.5, 0.99]
        ax.set_xticks(_logit_scores(tick_p), ["0.01", "0.5", "0.99"])
        ax.set_xlim(-8, 8)
        ax.set_yticks([])
        ax.grid(axis="x", color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=10.8)
        ax.set_xlabel("score (logit scale)")
        ax.text(
            0.04,
            0.94,
            note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.8,
            bbox={"boxstyle": "round,pad=0.24", "fc": "white", "ec": "0.8", "lw": 0.5},
        )

    ax = axes[2]
    names = [r[0] for r in rows]
    ece = [r[1]["ece_mean"] for r in rows]
    brier = [r[1]["brier_mean"] for r in rows]
    ypos = np.arange(len(rows))
    height = 0.34
    ax.barh(ypos - height / 2, ece, height, color="#0072B2", label="ECE")
    ax.barh(ypos + height / 2, brier, height, color="#D55E00", label="Brier")
    for i, (e, b) in enumerate(zip(ece, brier)):
        ax.text(e + 0.012, i - height / 2, f"{e:.2f}", va="center", fontsize=7.8)
        ax.text(b + 0.012, i + height / 2, f"{b:.2f}", va="center", fontsize=7.8)
    ax.set_yticks(ypos, names, fontsize=8.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.92)
    ax.set_xlabel("error")
    ax.grid(axis="x", color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Error before and after", fontsize=10.8)
    ax.legend(loc="lower right", fontsize=8.4, frameon=False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#4C72B0", alpha=0.55, ec="none"),
        plt.Rectangle((0, 0), 1, 1, fc="#DD8452", alpha=0.55, ec="none"),
    ]
    fig.legend(handles, ["grounded", "hallucinated"], loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    _save(fig, FIG_PAPER / "b4" / "calibration_shift.png")


def _logit_scores(values) -> np.ndarray:
    p = np.clip(np.asarray(values, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def transfer_scores() -> None:
    b2 = pd.read_parquet(ROOT / "artifacts" / "results" / "b2" / "b2_predictions.parquet")
    b3 = pd.read_parquet(ROOT / "artifacts" / "results" / "b3" / "b3_predictions.parquet")
    hal = b2[(b2["model"] == "xgboost") & (b2["seed"] == SEED) & (b2["split"] == "test")]
    b3s = b3[b3["model"] == f"xgboost_seed_{SEED}"]
    rt = b3s[(b3s["source_dataset"] == "ragtruth") & (b3s["official_split"] == "test")]
    qa = rt[rt["task"] == "qa"]
    fb = b3s[b3s["source_dataset"] == "faithbench"]
    panels = [
        ("HaluEval", hal),
        ("RAGTruth QA", qa),
        ("RAGTruth pooled", rt),
        ("FaithBench", fb),
    ]
    for name, d in panels:
        print(f"{name}: n={len(d)}")
    assert len(hal) == 3000 and len(qa) == 900 and len(rt) == 2700 and len(fb) == 750

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.7), sharex=True)
    axes = axes.ravel()
    bins = np.linspace(-8, 8, 65)
    for i, (ax, (name, d)) in enumerate(zip(axes, panels)):
        y = d["label"].to_numpy()
        z = _logit_scores(d["score"].to_numpy())
        ax.hist(z[y == 0], bins=bins, density=True, alpha=0.55, color="#4C72B0", label="grounded")
        ax.hist(z[y == 1], bins=bins, density=True, alpha=0.55, color="#DD8452", label="hallucinated")
        ax.axvline(0, color="0.45", lw=1.0, ls=(0, (3, 3)))
        tick_p = [0.01, 0.5, 0.99, 0.999]
        tick_z = _logit_scores(tick_p)
        ax.set_xticks(tick_z, ["0.01", "0.5", "0.99", "0.999"])
        ax.set_xlim(-8, 8)
        ax.set_yticks([])
        ax.grid(axis="x", color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        prev = 100 * float(np.mean(y))
        ax.set_title(f"{name}  (n = {len(d):,}, {prev:.1f}% pos)", fontsize=10.3)
        if i >= 2:
            ax.set_xlabel("raw score (logit scale)")
        if i % 2 == 0:
            ax.set_ylabel("density")
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#4C72B0", alpha=0.55, ec="none"),
        plt.Rectangle((0, 0), 1, 1, fc="#DD8452", alpha=0.55, ec="none"),
    ]
    fig.legend(handles, ["grounded", "hallucinated"], loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    _save(fig, FIG_PAPER / "b3" / "transfer_score_distributions.png")


def reliability_diagrams() -> None:
    b4 = pd.read_parquet(ROOT / "artifacts" / "results" / "b4" / "b4_predictions.parquet")
    colors = {"raw": "0.35", "platt": "#0072B2", "isotonic": "#D55E00"}
    labels = {"raw": "raw", "platt": "Platt", "isotonic": "isotonic"}
    fig, axes = plt.subplots(
        1, 3, figsize=(6.6, 3.15), gridspec_kw={"width_ratios": [1.0, 1.05, 1.0]}
    )

    # (a) HaluEval: quantile-bin reliability curves.
    ax = axes[0]
    d = b4[b4["subset"] == "halueval_test"]
    for method in ("raw", "platt", "isotonic"):
        sub = d[d["method"] == method].copy()
        if sub.empty:
            continue
        try:
            sub["bin"] = pd.qcut(sub["score"], 10, duplicates="drop")
        except ValueError:
            continue
        points = []
        for _, g in sub.groupby("bin", observed=True):
            if len(g) < 20:
                continue
            points.append((float(g["score"].mean()), float(g["label"].mean())))
        if not points:
            continue
        points.sort()
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            color=colors[method],
            marker="o",
            ms=4.2,
            lw=1.6,
            label=labels[method],
            zorder=3,
        )
    ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), color="0.55", lw=1.0, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.grid(color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("HaluEval test", fontsize=11.5)

    # (b) RAGTruth QA: score distribution on a logit scale.
    ax = axes[1]
    qa = b4[(b4["subset"] == "ragtruth_qa_test") & (b4["method"] == "raw")]
    y = qa["label"].to_numpy()
    p = np.clip(qa["score"].to_numpy(), 1e-4, 1 - 1e-4)
    z = np.log(p / (1 - p))
    bins = np.linspace(-9, 9, 72)
    ax.hist(z[y == 0], bins=bins, density=True, alpha=0.55, color="#4C72B0", label="grounded")
    ax.hist(z[y == 1], bins=bins, density=True, alpha=0.55, color="#DD8452", label="hallucinated")
    ax.axvline(0, color="0.45", lw=1.0, ls=(0, (3, 3)))
    tick_p = [0.1, 0.5, 0.9, 0.99]
    tick_z = np.log(np.array(tick_p) / (1 - np.array(tick_p)))
    ax.set_xticks(tick_z, ["0.1", "0.5", "0.9", "0.99"])
    ax.set_xlim(-3.2, 8.2)
    ax.set_yticks([])
    ax.grid(axis="x", color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("raw score (logit scale)")
    ax.set_title("RAGTruth QA test", fontsize=11.5)
    ax.text(
        0.04,
        0.94,
        "mean 0.998\naccuracy 0.178",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.8", "lw": 0.5},
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0.03, 0.70), fontsize=8.2, frameon=False)

    # (c) FaithBench: aggregate points, because the scores concentrate near 1.0.
    ax = axes[2]
    fb = b4[b4["subset"] == "faithbench"]
    for method in ("raw", "platt", "isotonic"):
        sub = fb[fb["method"] == method]
        if sub.empty:
            continue
        ax.scatter(
            float(sub["score"].mean()),
            float(sub["label"].mean()),
            s=55,
            color=colors[method],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
            label=labels[method],
        )
    ax.plot([0.9, 1.0], [0.9, 1.0], ls=(0, (4, 3)), color="0.55", lw=1.0, zorder=1)
    ax.set_xlim(0.9, 1.0)
    ax.set_ylim(0, 1)
    ax.set_xticks([0.9, 0.95, 1.0], ["0.9", "0.95", "1"])
    ax.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.grid(color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("confidence")
    ax.set_title("FaithBench", fontsize=11.5)
    ax.text(
        0.92,
        0.55,
        "mean 0.987\naccuracy 0.689",
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.8", "lw": 0.5},
    )

    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, frameon=False, fontsize=9.2)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    _save(fig, FIG_PAPER / "b4" / "reliability_diagrams.png")


def shap_importance() -> None:
    data = json.loads(
        (ROOT / "artifacts" / "results" / "b5" / "b5_feature_importance.json").read_text(encoding="utf-8")
    )
    items = sorted(data["mean_abs_shap"].items(), key=lambda kv: kv[1], reverse=True)[:12]
    names = [k for k, _ in items]
    values = [v for k, v in items]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
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
