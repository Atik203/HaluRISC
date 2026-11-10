"""Revision figures for the HaluRISC journal manuscript.

Style: 600 DPI PNG, white background, Okabe-Ito colour-blind-safe palette,
panel labels in bold at the top-left, no in-panel titles, tight_layout.
Every plotted number is read from an artifact or matches the manuscript table;
the assertions in each function fail loudly if an artifact drifts.

Writes ONLY to Journal_Paper/figures/revision/ (created on demand) and never
touches artifacts/ or Journal_Paper/*.tex.

  length_analysis.png   1x2  hallucination rate and mean score / FP / FN by length bin
  per_seed_stability.png 1x3 F1, AUROC and ECE for m0-m3 across seeds 42/123/456
  latency_breakdown.png 1x1  per-component p50 and p95 latency
  shap_summary.png      1x2  mean |SHAP| (top 12) and beeswarm on the test split
  ablation_delta.png    1x1  feature-group dF1

Run from the repo root:

  .venv/Scripts/python.exe src/models/plot_revision_figures_b.py
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FixedFormatter, FixedLocator, FormatStrFormatter
from matplotlib.transforms import blended_transform_factory

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "artifacts" / "results"
OUT = ROOT / "Journal_Paper" / "figures" / "revision"
DPI = 600
SEEDS = (42, 123, 456)

# Okabe-Ito colour-blind-safe palette.
BLACK = "#000000"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
YELLOW = "#F0E442"
BLUE = "#0072B2"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,  # unused: no in-panel titles
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 9.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "0.9",
        "grid.linewidth": 0.6,
    }
)


def _save(fig, name: str, rect=None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.tight_layout(rect=rect) if rect is not None else fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    w, h = fig.get_size_inches()
    print(f"[fig] {path.relative_to(ROOT)}  {int(round(w * DPI))}x{int(round(h * DPI))} px @ {DPI} DPI")
    plt.close(fig)


def _panel(ax, letter: str, dx: float = -0.02, suffix: str | None = None) -> None:
    ax.text(
        dx,
        1.03,
        f"({letter})" if suffix is None else f"({letter}) {suffix}",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=BLACK,
    )


# --------------------------------------------------------------------------
# 1. length analysis
# --------------------------------------------------------------------------
def length_analysis() -> None:
    src = RESULTS / "b5" / "b5_length_error_analysis.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    bins = data["bins"]
    labels = [b["bin"] for b in bins]
    rows = np.array([b["rows"] for b in bins], dtype=float)
    pos = np.array([b["pos_rate"] for b in bins])
    fp = np.array([b["fp_rate"] for b in bins])
    fn = np.array([b["fn_rate"] for b in bins])
    score = np.array([b["mean_score"] for b in bins])

    # Cross-check against the manuscript table (halurisc.tex, length analysis).
    assert labels == ["1", "2", "3-4", "5-8", "9-16", "17+"]
    assert rows.tolist() == [524, 514, 474, 641, 600, 247]
    np.testing.assert_allclose(np.round(pos, 3), [0.021, 0.033, 0.127, 0.902, 0.982, 0.992], atol=5e-4)
    np.testing.assert_allclose(np.round(score, 3), [0.024, 0.035, 0.126, 0.888, 0.980, 0.993], atol=5e-4)
    np.testing.assert_allclose(np.round(fp, 3), [0.004, 0.000, 0.002, 0.106, 0.000, 0.333], atol=5e-4)
    np.testing.assert_allclose(np.round(fn, 3), [0.182, 0.176, 0.178, 0.031, 0.003, 0.000], atol=5e-4)
    print("length source:", src.relative_to(ROOT), "(bin-level values complete; manuscript fallback unused)")

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.35, 2.85))

    bars = ax1.bar(x, pos * 100.0, width=0.64, color=SKY, edgecolor=BLUE, linewidth=0.6)
    for rect, v in zip(bars, pos * 100.0):
        if v >= 50.0:
            ax1.text(
                rect.get_x() + rect.get_width() / 2,
                v - 2.5,
                f"{v:.1f}%",
                ha="center",
                va="top",
                fontsize=9,
                color="white",
                fontweight="bold",
                rotation=90,
            )
        else:
            ax1.text(
                rect.get_x() + rect.get_width() / 2,
                v + 1.6,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color=BLACK,
            )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("answer length (words)")
    ax1.set_ylabel("hallucination rate (%)")
    ax1.set_ylim(0, 112)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.yaxis.grid(True)
    ax1.set_axisbelow(True)
    _panel(ax1, "a", dx=-0.14)

    ax2.bar(x, score, width=0.64, color="#D6E6F2", edgecolor=BLUE, linewidth=0.6, label="mean score")
    ax2.plot(x, fp, marker="o", ms=4.5, lw=1.5, color=VERMILION, label="FP rate")
    ax2.plot(x, fn, marker="s", ms=4.2, lw=1.5, color=GREEN, label="FN rate")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("answer length (words)")
    ax2.set_ylabel("mean score / error rate")
    ax2.set_ylim(-0.02, 1.30)
    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.yaxis.grid(True)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper left", frameon=False, handlelength=1.4, borderaxespad=0.2)
    _panel(ax2, "b", dx=-0.12)

    _save(fig, "length_analysis.png")
    print(
        "  rows/bin:",
        dict(zip(labels, rows.astype(int).tolist())),
        "hallucinated:",
        [round(v, 3) for v in pos],
    )


# --------------------------------------------------------------------------
# 2. per-seed stability
# --------------------------------------------------------------------------
def per_seed_stability() -> None:
    rebuild_src = RESULTS / "rebuild" / "fixed_hp" / "fixed_hp_ablation.csv"
    src = rebuild_src if rebuild_src.exists() else RESULTS / "b6" / "b6_per_seed_metrics.csv"
    df = pd.read_csv(src)
    variants = ["m0", "m1", "m2", "m3"]
    assert sorted(df["variant"].unique()) == variants
    assert sorted(df["seed"].unique()) == list(SEEDS)
    assert len(df) == 12

    stats = {}
    for v in variants:
        sub = df[df["variant"] == v]
        stats[v] = {
            m: (float(sub[m].mean()), float(sub[m].std(ddof=1))) for m in ("f1", "auroc", "ece")
        }
    for v in variants:
        f1m, f1s = stats[v]["f1"]
        aum, aus = stats[v]["auroc"]
        em, es = stats[v]["ece"]
        print(f"  {v}: F1 {f1m:.4f} +/- {f1s:.4f} | AUROC {aum:.4f} +/- {aus:.4f} | ECE {em:.4f} +/- {es:.4f}")

    specs = [
        ("f1", "F1", (0.9835, 0.9865), "%.4f"),
        ("auroc", "AUROC", (0.9972, 0.9989), "%.4f"),
        ("ece", "ECE", (0.0025, 0.0105), "%.4f"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(6.35, 3.35))
    offsets = {42: -0.075, 123: 0.0, 456: 0.075}
    seed_colors = {42: BLUE, 123: ORANGE, 456: GREEN}
    rng = np.random.default_rng(42)

    for ax, (key, label, ylim, fmt), letter in zip(axes, specs, "abc"):
        for i, v in enumerate(variants):
            sub = df[df["variant"] == v].set_index("seed")
            for seed in SEEDS:
                ax.scatter(
                    i + offsets[seed] + rng.uniform(-0.012, 0.012),
                    sub.loc[seed, key],
                    s=26,
                    color=seed_colors[seed],
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=3,
                    label=f"seed {seed}" if i == 0 else None,
                )
            m, s = stats[v][key]
            ax.hlines(m, i - 0.26, i + 0.26, color=BLACK, lw=2.0, zorder=4)
            tr = blended_transform_factory(ax.transData, ax.transAxes)
            # Alternate label heights so six-character numbers never collide.
            top = 0.90 if i % 2 == 0 else 0.78
            ax.text(i, top, fmt % m, transform=tr, ha="center", va="top", fontsize=8.5)
            ax.text(i, top - 0.060, "\u00b1" + fmt % s, transform=tr, ha="center", va="top", fontsize=8.5)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, fontsize=9.5)
        ax.set_xlim(-0.70, len(variants) - 0.60)
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        _panel(ax, letter, dx=-0.10, suffix=label)

    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=5, color=seed_colors[s], label=f"seed {s}") for s in SEEDS
    ]
    handles.append(plt.Line2D([], [], color=BLACK, lw=2.0, label="mean"))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        handlelength=1.2,
        columnspacing=1.8,
        bbox_to_anchor=(0.5, 0.004),
    )
    _save(fig, "per_seed_stability.png", rect=(0.0, 0.10, 1.0, 1.0))
    print("  source:", src.relative_to(ROOT))


# --------------------------------------------------------------------------
# 3. latency breakdown
# --------------------------------------------------------------------------
def latency_breakdown() -> None:
    src = RESULTS / "latency_analysis.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    order = [
        ("core_lexical", "Core lexical"),
        ("entity", "Entity (spaCy)"),
        ("nli", "NLI"),
        ("semantic", "Semantic (MiniLM)"),
        ("model", "XGBoost"),
        ("shap", "SHAP"),
    ]
    comp = data["latency_ms"]
    p50 = [comp[k]["p50"] for k, _ in order]
    p95 = [comp[k]["p95"] for k, _ in order]
    total = data["total_per_sample_ms"]["p50"]
    assert abs(total - 61.77) < 1e-9
    for k, _ in order:
        assert "p50" in comp[k] and "p95" in comp[k], f"missing p95 for {k}"
    print("latency source:", src.relative_to(ROOT), "| n =", data["n_samples"], "| end-to-end p50 =", total)

    fig, ax = plt.subplots(figsize=(6.2, 2.95))
    y = np.arange(len(order))[::-1]
    h = 0.36
    ax.barh(y + h / 2, p50, height=h, color=BLUE, label="p50")
    ax.barh(y - h / 2, p95, height=h, color=ORANGE, label="p95")
    for yy, v in zip(y + h / 2, p50):
        ax.text(v * 1.13, yy, f"{v:.2f}", va="center", ha="left", fontsize=9)
    for yy, v in zip(y - h / 2, p95):
        ax.text(v * 1.13, yy, f"{v:.2f}", va="center", ha="left", fontsize=9)

    ax.set_xscale("log")
    ax.set_xlim(0.07, 260)
    ax.xaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3, 10, 30, 100]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.1", "0.3", "1", "3", "10", "30", "100"]))
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in order])
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel("latency per sample (ms, log scale)")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=2,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.4,
        borderaxespad=0.1,
    )
    ax.text(
        0.99,
        0.97,
        f"end-to-end median {total:.2f} ms",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        color=BLACK,
    )
    _save(fig, "latency_breakdown.png")
    print("  components:", dict(zip([o[1] for o in order], zip(p50, p95))))


# --------------------------------------------------------------------------
# 4. SHAP summary
# --------------------------------------------------------------------------
def shap_summary() -> None:
    imp_src = RESULTS / "b5" / "b5_feature_importance.json"
    imp = json.loads(imp_src.read_text(encoding="utf-8"))["mean_abs_shap"]
    top = sorted(imp, key=lambda k: imp[k], reverse=True)[:12]
    top_vals = [imp[k] for k in top]

    feats = json.loads((ROOT / "artifacts" / "models" / "feature_names.json").read_text(encoding="utf-8"))
    parquet = ROOT / "data" / "processed" / "features_full.parquet"
    df = pd.read_parquet(parquet)
    X = df.loc[df["split"] == "test", feats]
    assert len(X) == 3000

    # The published b5 importances come from the B2 seed-42 booster. The
    # generic API explainer (artifacts/models/shap_explainer.joblib) wraps a
    # different booster (model_xgboost_raw.joblib) and does NOT reproduce the
    # b5 numbers, so the beeswarm is computed from the matching B2 model
    # through shap.TreeExplainer, the same path used to build the artifact.
    import shap

    model = joblib.load(ROOT / "artifacts" / "models" / "b2" / "xgboost_seed_42.joblib")
    sv_all = np.asarray(shap.TreeExplainer(model).shap_values(X.values))
    if sv_all.ndim == 3:
        sv_all = sv_all[:, :, 1]
    recomputed = dict(zip(feats, np.abs(sv_all).mean(axis=0).tolist()))
    max_diff = max(abs(recomputed[k] - imp[k]) for k in feats)
    assert max_diff < 1e-6, "beeswarm model must reproduce the b5 mean |SHAP| values"
    print(f"  b5 mean|SHAP| reproduction max abs diff = {max_diff:.3e}")

    api_explainer = joblib.load(ROOT / "artifacts" / "models" / "shap_explainer.joblib")
    sv_api = np.asarray(api_explainer.shap_values(X.values))
    if sv_api.ndim == 3:
        sv_api = sv_api[:, :, 1]
    api_rec = dict(zip(feats, np.abs(sv_api).mean(axis=0).tolist()))
    api_diff = max(abs(api_rec[k] - imp[k]) for k in feats)
    print(f"  shap_explainer.joblib vs b5 max abs diff = {api_diff:.4f} (different booster; not used)")

    idx = [feats.index(k) for k in top]
    sv = sv_all[:, idx]
    vals = X[top].to_numpy(dtype=float)

    print("  top-12 order and mean |SHAP| (b5 json / recomputed):")
    for k in top:
        print(f"    {k:26s} {imp[k]:.4f} / {recomputed[k]:.4f}")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.35, 3.55), gridspec_kw={"width_ratios": [1.0, 1.22]}
    )

    y = np.arange(len(top))
    ax1.barh(y, top_vals, color=SKY, edgecolor=BLUE, linewidth=0.6, height=0.7)
    for yy, v in zip(y, top_vals):
        ax1.text(v + 0.03, yy, f"{v:.3f}", va="center", ha="left", fontsize=9)
    ax1.set_yticks(y)
    ax1.set_yticklabels(top, fontsize=9.5)
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(top_vals) * 1.22)
    ax1.set_xlabel("mean |SHAP|")
    ax1.xaxis.grid(True)
    ax1.set_axisbelow(True)
    _panel(ax1, "a", dx=-0.42)

    cmap = LinearSegmentedColormap.from_list("okabe_bo", [BLUE, "#B9D9EA", "#F6D9C4", VERMILION])
    norm = Normalize(vmin=0.0, vmax=1.0)
    rng = np.random.default_rng(42)
    for j in range(len(top)):
        vv = vals[:, j]
        lo, hi = np.percentile(vv, [1, 99])
        if hi <= lo:
            rank = np.zeros_like(vv)
        else:
            rank = np.clip((vv - lo) / (hi - lo), 0.0, 1.0)
        jitter = rng.uniform(-0.26, 0.26, size=len(vv))
        ax2.scatter(
            sv[:, j],
            j + jitter,
            c=rank,
            cmap=cmap,
            norm=norm,
            s=2.0,
            alpha=0.55,
            linewidths=0,
            rasterized=True,
        )
    ax2.axvline(0.0, color="0.35", lw=0.8, ls="--", zorder=1)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_ylim(len(top) - 0.4, -0.6)
    ax2.set_xlabel("SHAP value (log-odds)")
    ax2.grid(True, axis="y")
    ax2.set_axisbelow(True)
    _panel(ax2, "b", dx=-0.07)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, fraction=0.035, pad=0.015, ticks=[0.0, 1.0])
    cbar.ax.set_yticklabels(["low", "high"], fontsize=9)
    cbar.set_label("feature value", fontsize=9.5)
    cbar.outline.set_linewidth(0.6)

    _save(fig, "shap_summary.png")
    print("  importance source:", imp_src.relative_to(ROOT), "| beeswarm:", parquet.relative_to(ROOT))
    print("  beeswarm model: artifacts/models/b2/xgboost_seed_42.joblib")


# --------------------------------------------------------------------------
# 5. feature-group ablation
# --------------------------------------------------------------------------
def ablation_delta() -> None:
    # Values as printed in the manuscript ablation table (rounded to 4 dp).
    rows = [
        ("length", -0.0045, 0.9801),
        ("lexical", -0.0247, 0.9600),
        ("entity", 0.0001, 0.9847),
        ("nli", -0.0023, 0.9823),
        ("numeric", -0.0002, 0.9844),
        ("hedging", 0.0001, 0.9848),
        ("semantic", 0.0014, 0.9860),
    ]
    art_src = RESULTS / "ablation_results.csv"
    art = pd.read_csv(art_src).set_index("removed_group")
    b2 = pd.read_csv(RESULTS / "b2" / "b2_per_seed_metrics.csv")
    ref = float(b2.loc[b2["model"] == "xgboost", "f1"].mean())
    print(f"ablation source: manuscript Table (tab:ablation); check against {art_src.relative_to(ROOT)}")
    print(f"  reference full-model F1 from b2 xgboost seeds = {ref:.6f}")
    for name, delta, f1 in rows:
        computed = float(art.loc[name, "f1_mean"]) - ref
        assert abs(round(float(art.loc[name, "f1_mean"]), 4) - f1) < 6e-5, name
        flag = "ok" if abs(computed - delta) < 5e-4 else "ROUNDING DIFF"
        print(f"  {name:9s} F1={f1:.4f} dF1(table)={delta:+.4f} dF1(recomputed)={computed:+.4f} [{flag}]")

    names = [r[0] for r in rows]
    deltas = np.array([r[1] for r in rows])
    y = np.arange(len(names))[::-1]
    colors = [VERMILION if d < 0 else BLUE for d in deltas]

    fig, ax = plt.subplots(figsize=(6.0, 2.75))
    ax.barh(y, deltas, height=0.62, color=colors)
    ax.axvline(0.0, color=BLACK, lw=0.9)
    for yy, d in zip(y, deltas):
        ha = "right" if d < 0 else "left"
        off = -0.0006 if d < 0 else 0.0006
        ax.text(d + off, yy, f"{d:+.4f}".replace("-", "\u2212"), va="center", ha=ha, fontsize=9.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(-0.0325, 0.0075)
    ax.set_xlabel("\u0394F1 after removing the feature group")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.65, len(names) - 0.35)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=VERMILION, label="F1 decrease"),
            plt.Rectangle((0, 0), 1, 1, color=BLUE, label="F1 increase"),
        ],
        loc="lower left",
        frameon=False,
        fontsize=9,
        handlelength=1.2,
        borderaxespad=0.2,
    )
    _save(fig, "ablation_delta.png")


if __name__ == "__main__":
    print("== length_analysis ==")
    length_analysis()
    print("== per_seed_stability ==")
    per_seed_stability()
    print("== latency_breakdown ==")
    latency_breakdown()
    print("== shap_summary ==")
    shap_summary()
    print("== ablation_delta ==")
    ablation_delta()
    print("done: revision figures written to", OUT.relative_to(ROOT))
