"""Redraw the m3 SHAP summary from the exported importance table.

The SHAP values are not recomputed; this only restyles the figure so the
manuscript gets a clean, full-width chart.

Run:
  python src/models/plot_m3_shap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts" / "results" / "rebuild" / "explanations" / "m3_global_shap_importance.csv"
OUT = ROOT / "Journal_Paper" / "figures" / "revision" / "m3_shap_summary.png"
DPI = 650

GROUP_COLORS = {
    "lexical": "#6C5CE7",
    "length": "#0984E3",
    "semantic": "#E84393",
    "nli": "#E17055",
    "claim_nli": "#D63031",
    "entity": "#00B894",
    "numeric": "#FDCB6E",
    "hedging": "#636E72",
    "source": "#95A5A6",
}

plt.rcParams.update(
    {
        "font.size": 10.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
    }
)


def main() -> None:
    data = pd.read_csv(SRC).head(15).iloc[::-1]
    colors = [GROUP_COLORS.get(g, "#636E72") for g in data["feature_group"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.55), constrained_layout=True)
    bars = ax.barh(
        data["feature"],
        data["mean_abs_shap"],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        height=0.74,
        zorder=3,
    )
    for bar, value, const in zip(bars, data["mean_abs_shap"], data["constant_on_sample"]):
        if bool(const):
            bar.set_hatch("///")
            bar.set_edgecolor("0.45")
        ax.text(
            value + data["mean_abs_shap"].max() * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=8.2,
        )
    ax.set_xlabel("mean |SHAP| (raw margin)")
    ax.set_xlim(0, data["mean_abs_shap"].max() * 1.16)
    ax.grid(axis="x", color="0.92", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    present = [g for g in GROUP_COLORS if g in set(data["feature_group"])]
    handles = [Patch(facecolor=GROUP_COLORS[g], label=g.replace("_", " ")) for g in present]
    ax.legend(
        handles=handles,
        loc="lower right",
        ncol=2,
        frameon=False,
        handlelength=0.9,
        columnspacing=1.0,
        labelspacing=0.3,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=DPI, facecolor="white")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

