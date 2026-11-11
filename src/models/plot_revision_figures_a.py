"""Revision figure set A: ROC/PR curves, confusion matrices, operating point and label budget.

Writes ONLY to ``Journal_Paper/figures/revision/`` (no artifacts, no manuscript edits).

Figures
-------
roc_pr_curves.png              2x2  (a) ROC HaluEval, (b) ROC RAGTruth task slices,
                                    (c) PR HaluEval, (d) PR RAGTruth task slices.
confusion_matrices.png         2x2  (a) HaluEval EC-XGB, (b) HaluEval conventional,
                                    (c) RAGTruth QA EC-XGB, (d) RAGTruth QA conventional.
operating_point_label_budget.png  1x2  (a) threshold sweep, (b) ECE/Brier vs labeled rows.

Data provenance
---------------
HaluEval test (3,000 rows; b2 ``split == "test"``, identical sample_ids to
``b6_predictions.parquet`` dataset ``halueval_test``):
  * EC-XGB            b6_predictions.parquet, variant m3
  * conventional XGB  b2_predictions.parquet, model ``xgboost``
  * random forest     b2_predictions.parquet, model ``rf_full``
  * logistic regr.    b2_predictions.parquet, model ``lr_full``
  * NLI only          b2_predictions.parquet, model ``nli_only``
  * TF-IDF answer     b2_predictions.parquet, model ``tfidf_answer``

RAGTruth (2,700 rows = 900 per task; ``ragtruth_all_test`` joined to
``data/processed/external_features.parquet`` on ``sample_id`` for ``task``):
  * EC-XGB            b6_predictions.parquet, variant m3   (the deployed model)
  * conventional      b6_predictions.parquet, variant m0   (standard tuned XGBoost;
                      the checkpoint behind the "Conventional" column of the
                      cross-domain table; b2 has no RAGTruth rows and the b4
                      per-row file holds the B3 checkpoint, AUROC 0.540, which is
                      a different model)

Seeds
-----
Curves are drawn for seed 42 and the legend reports the SEED-42 AUROC / PR-AUC of
the curves actually drawn (the manuscript tables report three-seed means; the
caption says so). A separate check still recomputes the three-seed means and
compares them with the published values (<= 0.001, hard failure otherwise), and
the seed-42 numbers are printed alongside.

Confusion matrices
------------------
(a) artifacts/results/b6/b6_confusion_matrices.json  key m3_seed_42
(b) artifacts/results/b2/b2_confusion_matrices.json  key xgboost_seed_42.0
(c) derived from b6_predictions.parquet (dataset ragtruth_qa_test, variant m3,
    seed 42), threshold 0.5, raw score
(d) derived from b6_predictions.parquet (dataset ragtruth_qa_test, variant m0,
    seed 42), threshold 0.5, raw score
Row-normalised percentages use the actual-class row totals. Every panel is
therefore thresholded on the raw score at 0.5, which the x-axis label states.

Operating point (fig. 3a)
-------------------------
Only the conventional model is swept, on its target-Platt display score
(b4_predictions.parquet subset ragtruth_qa_test, method raw, mapped through
calibrator_platt_target_ragtruth_qa_seed_42.joblib). That is the series that
reproduces the published P2 operating point: at the selected threshold
0.312738 (labelled 0.313) recall = 5.0% and FPR = 5.8%. The EC-XGB display
series is deliberately not drawn: its display scores live in [0.044, 0.354], so
0.313 is not a 5%-FPR operating point on that scale.

Run from the repo root:

  .venv/Scripts/python.exe src/models/plot_revision_figures_a.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Journal_Paper" / "figures" / "revision"
RESULTS = ROOT / "artifacts" / "results"
MODELS = ROOT / "artifacts" / "models"
PROCESSED = ROOT / "data" / "processed"

DPI = 650
SEEDS = (42, 123, 456)
SEED = 42
THRESHOLD = 0.5
P2_THRESHOLD = 0.313  # label shown in the figure (the paper rounds 0.312738)
P2_THRESHOLD_EXACT = 0.312738  # threshold selected on the calibration split at a 5% FPR budget
SHOW_PANEL_NOTE = True

# Okabe-Ito colourblind-safe palette.
OKABE = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}
GREY = "#5A5A5A"

HAL_MODELS = [
    ("EC-XGB (m3)", "EC-XGB", OKABE["blue"]),
    ("XGBoost (conventional)", "Conv. XGB", OKABE["vermillion"]),
    ("Random forest", "RF", OKABE["green"]),
    ("Logistic regression", "LR", OKABE["purple"]),
    ("NLI only", "NLI", OKABE["orange"]),
    ("TF-IDF (answer only)", "TF-IDF", OKABE["sky"]),
]
TASKS = [
    ("qa", "QA", "QA", OKABE["blue"]),
    ("summarization", "Summarization", "Summ.", OKABE["vermillion"]),
    ("data_to_text", "Data-to-text", "D2T", OKABE["green"]),
]

PUBLISHED_HAL = {
    "EC-XGB (m3)": (0.9984, 0.9987),
    "XGBoost (conventional)": (0.9979, 0.9983),
    "Random forest": (0.9980, 0.9983),
    "Logistic regression": (0.9932, 0.9901),
    "NLI only": (0.7177, 0.7531),
    "TF-IDF (answer only)": (0.9696, 0.9665),
}
PUBLISHED_RAG = {
    ("qa", "m3"): 0.5491,
    ("qa", "m0"): 0.5465,
    ("summarization", "m3"): 0.7358,
    ("summarization", "m0"): 0.6343,
    ("data_to_text", "m3"): 0.7705,
    ("data_to_text", "m0"): 0.5665,
}
TOL = 0.001

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 9,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    b2 = pd.read_parquet(RESULTS / "b2" / "b2_predictions.parquet")
    b6 = pd.read_parquet(RESULTS / "b6" / "b6_predictions.parquet")
    ext = pd.read_parquet(
        PROCESSED / "external_features.parquet", columns=["sample_id", "task"]
    )
    if "split" in b2.columns:
        b2 = b2[b2["split"] == "test"].copy()
    return b2, b6, ext


def halueval_scores(b2: pd.DataFrame, b6: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Model -> frame(seed, label, score) on the 3,000-row HaluEval test set."""
    out: dict[str, pd.DataFrame] = {}
    model_map = {
        "EC-XGB (m3)": None,
        "XGBoost (conventional)": "xgboost",
        "Random forest": "rf_full",
        "Logistic regression": "lr_full",
        "NLI only": "nli_only",
        "TF-IDF (answer only)": "tfidf_answer",
    }
    for name, _, _ in HAL_MODELS:
        model = model_map[name]
        if model is None:
            d = b6[(b6["dataset"] == "halueval_test") & (b6["variant"] == "m3")]
            d = d[d["seed"].isin(SEEDS)][["seed", "label", "score"]]
        else:
            d = b2[b2["model"] == model]
            d = d[d["seed"].isin(SEEDS)][["seed", "label", "score"]]
        d = d.assign(seed=d["seed"].astype(int))
        assert len(d) == 3000 * len(SEEDS), (name, len(d))
        assert d.groupby("seed").size().eq(3000).all(), name
        out[name] = d.reset_index(drop=True)
    return out


def ragtruth_all(b6: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    d = b6[(b6["dataset"] == "ragtruth_all_test") & (b6["seed"].isin(SEEDS))]
    d = d.merge(ext, on="sample_id", how="left", validate="many_to_one")
    assert d["task"].notna().all()
    assert d["score"].notna().all()
    per_seed = d[(d["seed"] == SEED) & (d["variant"] == "m3")]
    assert len(per_seed) == 2700, len(per_seed)
    assert per_seed.groupby("task").size().eq(900).all()
    return d


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def verify(
    hal: dict[str, pd.DataFrame], rag: pd.DataFrame
) -> tuple[
    dict[str, tuple[float, float]],
    dict[tuple[str, str], float],
    dict[str, tuple[float, float]],
    dict[tuple[str, str], float],
]:
    print("\n=== validation: recomputed vs published (mean over seeds 42/123/456) ===")
    hal_stats: dict[str, tuple[float, float]] = {}
    hal_seed: dict[str, tuple[float, float]] = {}
    problems: list[str] = []
    for name, d in hal.items():
        aurocs, aps = [], []
        for s in SEEDS:
            g = d[d["seed"] == s]
            aurocs.append(roc_auc_score(g["label"], g["score"]))
            aps.append(average_precision_score(g["label"], g["score"]))
        auroc, ap = float(np.mean(aurocs)), float(np.mean(aps))
        hal_stats[name] = (auroc, ap)
        g42 = d[d["seed"] == SEED]
        hal_seed[name] = (
            float(roc_auc_score(g42["label"], g42["score"])),
            float(average_precision_score(g42["label"], g42["score"])),
        )
        pub_a, pub_p = PUBLISHED_HAL[name]
        da, dp = auroc - pub_a, ap - pub_p
        flag = "" if max(abs(da), abs(dp)) <= TOL else "  <-- MISMATCH"
        print(
            f"  {name:26s} AUROC {auroc:.6f} (pub {pub_a:.4f}, d {da:+.4f})"
            f"  PR-AUC {ap:.6f} (pub {pub_p:.4f}, d {dp:+.4f}){flag}"
        )
        if flag:
            problems.append(f"HaluEval {name}: dAUROC {da:+.4f}, dPR {dp:+.4f}")

    rag_stats: dict[tuple[str, str], float] = {}
    rag_seed: dict[tuple[str, str], float] = {}
    for task, label, _, _ in TASKS:
        for variant, short in (("m3", "EC-XGB"), ("m0", "conventional")):
            aurocs = []
            for s in SEEDS:
                g = rag[(rag["variant"] == variant) & (rag["seed"] == s) & (rag["task"] == task)]
                assert len(g) == 900, (task, variant, s, len(g))
                aurocs.append(roc_auc_score(g["label"], g["score"]))
            auroc = float(np.mean(aurocs))
            rag_stats[(task, variant)] = auroc
            g42 = rag[(rag["variant"] == variant) & (rag["seed"] == SEED) & (rag["task"] == task)]
            rag_seed[(task, variant)] = float(roc_auc_score(g42["label"], g42["score"]))
            pub = PUBLISHED_RAG[(task, variant)]
            delta = auroc - pub
            flag = "" if abs(delta) <= TOL else "  <-- MISMATCH"
            print(f"  {label:15s} {short:13s} AUROC {auroc:.6f} (pub {pub:.4f}, d {delta:+.4f}){flag}")
            if flag:
                problems.append(f"RAGTruth {task} {short}: dAUROC {delta:+.4f}")

    if problems:
        raise SystemExit("STOP: recomputed metrics disagree with published values: " + "; ".join(problems))

    print("\n=== seed-42 values drawn in the figure (legends) vs the 3-seed mean ===")
    for name, d in hal.items():
        auroc, ap = hal_seed[name]
        mean_a, mean_p = hal_stats[name]
        print(f"  {name:26s} AUROC {auroc:.4f} (mean {mean_a:.4f}, d {auroc - mean_a:+.4f})"
              f"  PR-AUC {ap:.4f} (mean {mean_p:.4f}, d {ap - mean_p:+.4f})")
    for task, label, _, _ in TASKS:
        for variant, short in (("m3", "EC-XGB"), ("m0", "conventional")):
            auroc = rag_seed[(task, variant)]
            mean_a = rag_stats[(task, variant)]
            print(f"  {label:15s} {short:13s} AUROC {auroc:.4f} (mean {mean_a:.4f}, d {auroc - mean_a:+.4f})")

    print("\n=== row counts ===")
    print(f"  HaluEval test rows            {len(hal['EC-XGB (m3)']) // len(SEEDS):,}")
    print(f"  RAGTruth QA rows              {len(rag[(rag['variant'] == 'm3') & (rag['seed'] == SEED) & (rag['task'] == 'qa')]):,}")
    print(f"  RAGTruth task total rows      {len(rag[(rag['variant'] == 'm3') & (rag['seed'] == SEED)]):,}")
    return hal_stats, rag_stats, hal_seed, rag_seed


# --------------------------------------------------------------------------- #
# drawing helpers
# --------------------------------------------------------------------------- #
def panel_label(ax, letter: str, note: str | None = None) -> None:
    ax.text(
        0.0,
        1.03,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    if note and SHOW_PANEL_NOTE:
        ax.text(
            0.175,
            1.038,
            note,
            transform=ax.transAxes,
            fontsize=11,
            ha="left",
            va="bottom",
            color=GREY,
        )


def tidy(ax) -> None:
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)
    ax.set_xticks([0.0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.set_yticks([0.0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.tick_params(direction="out")


def curve_legend(ax, loc: str, ncol: int = 1, title: str | None = None) -> None:
    leg = ax.legend(
        loc=loc,
        ncol=ncol,
        fontsize=9,
        title=title,
        frameon=True,
        framealpha=0.94,
        edgecolor="0.85",
        borderpad=0.35,
        labelspacing=0.28,
        handlelength=1.9,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    leg.get_frame().set_linewidth(0.6)
    if title:
        leg.get_title().set_fontsize(9)


def task_legend(ax, rag_stats, loc: str, metric: str = "AUROC") -> None:
    handles = [
        Line2D([], [], color=colour, lw=1.9, label=f"{short}  {rag_stats[(task, 'm3')]:.3f} / {rag_stats[(task, 'm0')]:.3f}")
        for task, _, short, colour in TASKS
    ]
    leg = ax.legend(
        handles=handles,
        loc=loc,
        fontsize=8.4,
        title=f"{metric}, EC-XGB / conv.\n(solid / dashed)",
        title_fontsize=8.4,
        frameon=True,
        framealpha=0.94,
        edgecolor="0.85",
        borderpad=0.28,
        labelspacing=0.22,
        handlelength=1.7,
        handletextpad=0.45,
    )
    leg.get_frame().set_linewidth(0.6)
    leg.get_title().set_fontsize(9)


def metric_legend(ax, loc: str) -> None:
    """Three metric rows for the single (conventional, target-Platt) sweep."""
    handles = [
        Line2D([], [], color=OKABE["blue"], lw=1.9, label="Recall"),
        Line2D([], [], color=OKABE["vermillion"], lw=1.9, label="FPR"),
        Line2D([], [], color=OKABE["green"], lw=1.9, label="Precision"),
    ]
    leg = ax.legend(
        handles=handles,
        loc=loc,
        fontsize=9,
        frameon=True,
        framealpha=0.94,
        edgecolor="0.85",
        borderpad=0.35,
        labelspacing=0.28,
        handlelength=1.9,
        handletextpad=0.5,
    )
    leg.get_frame().set_linewidth(0.6)


def save(fig, name: str, note: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=DPI, facecolor="white", metadata={"Description": note, "Software": "matplotlib"})
    w, h = fig.get_size_inches() * DPI
    print(f"wrote {path.relative_to(ROOT)}  {int(round(w))} x {int(round(h))} px  ({DPI} DPI)")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# figure 1: ROC / PR
# --------------------------------------------------------------------------- #
def figure_roc_pr(hal, rag, hal_seed, rag_seed) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 5.5))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # (a) ROC HaluEval -------------------------------------------------------
    for name, short, colour in HAL_MODELS:
        g = hal[name][hal[name]["seed"] == SEED]
        fpr, tpr, _ = roc_curve(g["label"], g["score"])
        ax_a.plot(fpr, tpr, color=colour, lw=1.8, label=f"{short}  {hal_seed[name][0]:.3f}")
    ax_a.plot([0, 1], [0, 1], color=GREY, lw=1.0, ls=(0, (3, 2)), zorder=1)
    ax_a.set_xlabel("False positive rate")
    ax_a.set_ylabel("True positive rate")
    tidy(ax_a)
    panel_label(ax_a, "a", "HaluEval")
    curve_legend(ax_a, "lower right", title="AUROC")

    # (b) ROC RAGTruth per task ---------------------------------------------
    for task, _label, short, colour in TASKS:
        for variant, ls, lw in (("m3", "-", 1.9), ("m0", (0, (4, 2)), 1.7)):
            g = rag[(rag["variant"] == variant) & (rag["seed"] == SEED) & (rag["task"] == task)]
            fpr, tpr, _ = roc_curve(g["label"], g["score"])
            ax_b.plot(fpr, tpr, color=colour, ls=ls, lw=lw)
    ax_b.plot([0, 1], [0, 1], color=GREY, lw=1.0, ls=(0, (3, 2)), zorder=1)
    ax_b.set_xlabel("False positive rate")
    ax_b.set_ylabel("True positive rate")
    tidy(ax_b)
    panel_label(ax_b, "b", "RAGTruth")
    task_legend(ax_b, rag_seed, "lower right")

    # (c) PR HaluEval --------------------------------------------------------
    ax_c.axhline(0.5, color=GREY, lw=1.0, ls=(0, (2, 2)), zorder=1)
    for name, short, colour in HAL_MODELS:
        g = hal[name][hal[name]["seed"] == SEED]
        prec, rec, _ = precision_recall_curve(g["label"], g["score"])
        ax_c.plot(rec, prec, color=colour, lw=1.8, label=f"{short}  {hal_seed[name][1]:.3f}")
    ax_c.set_xlabel("Recall")
    ax_c.set_ylabel("Precision")
    tidy(ax_c)
    panel_label(ax_c, "c", "HaluEval")
    curve_legend(ax_c, "lower left", title="AP")

    # (d) PR RAGTruth per task ----------------------------------------------
    for task, _label, short, colour in TASKS:
        g0 = rag[(rag["variant"] == "m3") & (rag["seed"] == SEED) & (rag["task"] == task)]
        ax_d.axhline(g0["label"].mean(), color=colour, lw=1.0, ls=(0, (2, 2)), alpha=0.5, zorder=1)
        for variant, ls, lw in (("m3", "-", 1.9), ("m0", (0, (4, 2)), 1.7)):
            g = rag[(rag["variant"] == variant) & (rag["seed"] == SEED) & (rag["task"] == task)]
            prec, rec, _ = precision_recall_curve(g["label"], g["score"])
            ax_d.plot(rec, prec, color=colour, ls=ls, lw=lw)
    ax_d.set_xlabel("Recall")
    ax_d.set_ylabel("Precision")
    tidy(ax_d)
    panel_label(ax_d, "d", "RAGTruth")
    task_legend(ax_d, rag_seed, "lower right", metric="AP")

    fig.tight_layout(h_pad=1.5, w_pad=1.1)
    save(
        fig,
        "roc_pr_curves.png",
        "ROC/PR curves, seed 42 (the seed drawn). Legend values are the seed-42 AUROC "
        "(panels a, b) and PR-AUC (panels c, d) of the plotted curves. HaluEval n=3000; "
        "RAGTruth n=900 per task (2700 total). EC-XGB = b6 m3; conventional = b2 xgboost "
        "on HaluEval and b6 m0 on RAGTruth. Dotted horizontals: positive prevalence.",
    )


# --------------------------------------------------------------------------- #
# figure 2: confusion matrices
# --------------------------------------------------------------------------- #
CMAP = LinearSegmentedColormap.from_list("okabe_blue", ["#FFFFFF", "#0072B2"])


def _confusion_from_scores(d: pd.DataFrame) -> dict[str, int]:
    pred = (d["score"] >= THRESHOLD).astype(int)
    return {
        "tn": int(((pred == 0) & (d["label"] == 0)).sum()),
        "fp": int(((pred == 1) & (d["label"] == 0)).sum()),
        "fn": int(((pred == 0) & (d["label"] == 1)).sum()),
        "tp": int(((pred == 1) & (d["label"] == 1)).sum()),
    }


def figure_confusion(b6: pd.DataFrame) -> None:
    b2_json = json.loads((RESULTS / "b2" / "b2_confusion_matrices.json").read_text(encoding="utf-8"))
    b6_json = json.loads((RESULTS / "b6" / "b6_confusion_matrices.json").read_text(encoding="utf-8"))
    hal_ec = b6_json["m3_seed_42"]
    hal_conv = b2_json["xgboost_seed_42.0"]

    qa6 = b6[(b6["dataset"] == "ragtruth_qa_test") & (b6["seed"] == SEED)]
    rag_ec = _confusion_from_scores(qa6[qa6["variant"] == "m3"])
    rag_conv = _confusion_from_scores(qa6[qa6["variant"] == "m0"])

    panels = [
        ("a", "HaluEval EC-XGB", hal_ec),
        ("b", "HaluEval conventional", hal_conv),
        ("c", "RAGTruth QA EC-XGB", rag_ec),
        ("d", "RAGTruth QA conventional", rag_conv),
    ]
    for tag, title, cm in panels:
        n = sum(cm.values())
        print(f"  confusion ({tag}) {title:22s} n = {n}: {cm}")

    fig, axes = plt.subplots(2, 2, figsize=(6.2, 5.75))
    for ax, (tag, title, cm) in zip(axes.ravel(), panels):
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=float)
        row_tot = mat.sum(axis=1, keepdims=True)
        share = mat / row_tot
        ax.imshow(share, cmap=CMAP, vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_aspect("equal")
        for i in range(2):
            for j in range(2):
                colour = "white" if share[i, j] > 0.55 else "black"
                ax.text(
                    j,
                    i,
                    f"{int(mat[i, j]):,}\n{100 * share[i, j]:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=colour,
                    linespacing=1.25,
                )
        ax.set_xticks([0, 1], ["0", "1"])
        ax.set_yticks([0, 1], ["0", "1"])
        ax.set_xlabel("Predicted label (raw score ≥ 0.5)")
        ax.set_ylabel("Actual label")
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-0.5, 2.0, 1.0), minor=True)
        ax.set_yticks(np.arange(-0.5, 2.0, 1.0), minor=True)
        ax.grid(which="minor", color="#B0B0B0", linewidth=0.8)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        panel_label(ax, tag, title)

    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    save(
        fig,
        "confusion_matrices.png",
        "Confusion matrices, raw score threshold 0.5 in every panel. (a) HaluEval EC-XGB: "
        "b6_confusion_matrices.json m3_seed_42; (b) HaluEval conventional: "
        "b2_confusion_matrices.json xgboost_seed_42.0; (c) RAGTruth QA EC-XGB: derived from "
        "b6_predictions.parquet dataset ragtruth_qa_test variant m3 seed 42; (d) RAGTruth QA "
        "conventional: derived from b6_predictions.parquet dataset ragtruth_qa_test variant "
        "m0 seed 42. Cell text: count and row-normalised percentage.",
    )


# --------------------------------------------------------------------------- #
# figure 3: operating point and label budget
# --------------------------------------------------------------------------- #
def _sweep(y: np.ndarray, s: np.ndarray, thresholds: np.ndarray):
    rec = np.full(thresholds.size, np.nan)
    fpr = np.full(thresholds.size, np.nan)
    prec = np.full(thresholds.size, np.nan)
    pos = y == 1
    neg = ~pos
    for k, t in enumerate(thresholds):
        flag = s >= t
        tp = float(np.sum(flag & pos))
        fp = float(np.sum(flag & neg))
        fn = float(np.sum(~flag & pos))
        tn = float(np.sum(~flag & neg))
        rec[k] = tp / (tp + fn) if (tp + fn) else np.nan
        fpr[k] = fp / (fp + tn) if (fp + tn) else np.nan
        prec[k] = tp / (tp + fp) if (tp + fp) else np.nan
    return rec, fpr, prec


def _point_metrics(y: np.ndarray, s: np.ndarray, t: float) -> tuple[float, float, float]:
    flag = s >= t
    pos = y == 1
    neg = ~pos
    tp = float(np.sum(flag & pos))
    fp = float(np.sum(flag & neg))
    fn = float(np.sum(~flag & pos))
    tn = float(np.sum(~flag & neg))
    return (
        tp / (tp + fn) if (tp + fn) else np.nan,
        fp / (fp + tn) if (fp + tn) else np.nan,
        tp / (tp + fp) if (tp + fp) else np.nan,
    )


def figure_operating_point() -> None:
    # Conventional target-Platt scores: the series behind the published P2 point.
    b4 = pd.read_parquet(RESULTS / "b4" / "b4_predictions.parquet")
    conv_raw = b4[(b4["subset"] == "ragtruth_qa_test") & (b4["method"] == "raw")]
    assert len(conv_raw) == 900
    y = conv_raw["label"].to_numpy()
    conv_cal = joblib.load(MODELS / "b4" / "calibrator_platt_target_ragtruth_qa_seed_42.joblib")
    conv_display = conv_cal.predict_proba(conv_raw["score"].to_numpy().reshape(-1, 1))[:, 1]
    print(
        "\nconventional target Platt: "
        f"p = sigmoid({float(conv_cal.coef_[0, 0]):.7f} * raw {float(conv_cal.intercept_[0]):+.7f}); "
        f"test range [{conv_display.min():.4f}, {conv_display.max():.4f}]"
    )

    rec_p2, fpr_p2, prec_p2 = _point_metrics(y, conv_display, P2_THRESHOLD_EXACT)
    print(
        f"  P2 at {P2_THRESHOLD_EXACT}: recall {100 * rec_p2:.1f}%, FPR {100 * fpr_p2:.1f}%, "
        f"precision {100 * prec_p2:.1f}%"
    )

    thresholds = np.linspace(0.0, 1.0, 1001)
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(6.6, 3.0), gridspec_kw={"width_ratios": [1.12, 0.88]}
    )

    rec, fpr, prec = _sweep(y, conv_display, thresholds)
    ax.plot(thresholds, rec, color=OKABE["blue"], lw=1.9)
    ax.plot(thresholds, fpr, color=OKABE["vermillion"], lw=1.9)
    ax.plot(thresholds, prec, color=OKABE["green"], lw=1.9)

    ax.axvline(P2_THRESHOLD_EXACT, color="black", lw=1.1, ls=(0, (5, 2, 1, 2)), zorder=1)
    ax.axvline(THRESHOLD, color="black", lw=1.1, ls=(0, (1, 1.6)), zorder=1)
    ax.text(P2_THRESHOLD_EXACT - 0.018, 1.02, "P2 0.313", ha="right", va="bottom", fontsize=9.2)
    ax.text(THRESHOLD + 0.018, 1.02, "0.5", ha="left", va="bottom", fontsize=9.2)

    ax.plot([P2_THRESHOLD_EXACT], [rec_p2], marker="o", ms=4.5, color=OKABE["blue"], zorder=6)
    ax.plot([P2_THRESHOLD_EXACT], [fpr_p2], marker="o", ms=4.5, color=OKABE["vermillion"], zorder=6)
    ax.text(
        0.56,
        0.62,
        f"recall {100 * rec_p2:.1f}%\nFPR {100 * fpr_p2:.1f}%",
        fontsize=9.5,
        ha="left",
        va="center",
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "0.85", "lw": 0.6},
        zorder=6,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_yticks([0.0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.tick_params(direction="out")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Rate")
    panel_label(ax, "a", "RAGTruth QA, conventional")
    metric_legend(ax, "lower right")

    # (b) label budget ------------------------------------------------------
    fast_budget = RESULTS / "rebuild" / "calibration_fast" / "label_budget.csv"
    full_budget = RESULTS / "rebuild" / "calibration" / "label_budget.csv"
    budget_path = full_budget if full_budget.exists() else fast_budget
    if budget_path.exists():
        budget_df = pd.read_csv(budget_path)
        if "seed" in budget_df.columns:
            budget_df = budget_df[budget_df["seed"].astype(str) == "pooled"]
        budget_df = budget_df.sort_values("size")
        n_rows = budget_df["size"].tolist()
        ece = budget_df["ece_mean"].tolist()
        ece_lo = budget_df["ece_q025"].tolist()
        ece_hi = budget_df["ece_q975"].tolist()
        brier = budget_df["brier_mean"].tolist()
        print(f"  label budget: {budget_path.name} 100-draw means")
    else:
        budget = json.loads((RESULTS / "extras" / "review_extras.json").read_text(encoding="utf-8"))["x4_label_budget"]
        rows = sorted(budget.values(), key=lambda r: r["n_labeled_rows"])
        n_rows = [r["n_labeled_rows"] for r in rows]
        ece = [r["ece"] for r in rows]
        ece_lo = ece_hi = None
        brier = [r["brier"] for r in rows]
        print("  label budget: legacy single draw")
    print("  label budget:", list(zip(n_rows, np.round(ece, 4), np.round(brier, 4))))

    ax2.plot(n_rows, ece, color=OKABE["blue"], lw=1.9, marker="o", ms=4.5, label="ECE")
    if ece_lo is not None:
        ax2.fill_between(n_rows, ece_lo, ece_hi, color=OKABE["blue"], alpha=0.16, lw=0)
    ax2.plot(n_rows, brier, color=OKABE["vermillion"], lw=1.9, marker="s", ms=4.2, label="Brier")
    ax2.set_xscale("log")
    ax2.set_xticks(n_rows)
    ax2.set_xticklabels(
        [f"{n}" for n in n_rows], rotation=30, ha="right", rotation_mode="anchor", fontsize=8.4
    )
    ax2.minorticks_off()
    ax2.set_xlim(42, 6000)
    ax2.set_ylim(0.0, 0.28)
    ax2.set_xlabel("Labeled target rows")
    ax2.set_ylabel("Error")
    ax2.set_yticks([0.0, 0.1, 0.2], ["0", "0.10", "0.20"])
    ax2.tick_params(direction="out")
    panel_label(ax2, "b", "target label budget")
    curve_legend(ax2, "upper right", ncol=2)

    fig.tight_layout(w_pad=1.2)
    save(
        fig,
        "operating_point_label_budget.png",
        "Threshold sweep and label budget. (a) 900-row RAGTruth QA test set, conventional "
        "model on its target-Platt display score (b4 raw scores mapped through "
        "calibrator_platt_target_ragtruth_qa_seed_42.joblib); vertical lines at the selected "
        "P2 threshold 0.312738 (labelled 0.313) and the fixed 0.5 threshold; the marked P2 "
        "point is recall 5.0% / FPR 5.8%. (b) 100-draw mean ECE and Brier for the fixed-hp m3 "
        "target-Platt calibrator; the shaded band is the 2.5-97.5% range across draws.",
    )


# --------------------------------------------------------------------------- #
def main() -> None:
    b2, b6, ext = load_frames()
    hal = halueval_scores(b2, b6)
    rag = ragtruth_all(b6, ext)
    _hal_means, _rag_means, hal_seed, rag_seed = verify(hal, rag)

    figure_roc_pr(hal, rag, hal_seed, rag_seed)
    figure_confusion(b6)
    figure_operating_point()
    print("\ndone")


if __name__ == "__main__":
    main()

