"""Paper-grade figure for the H200 appendix (throughput and VRAM on the slice).

Reads artifacts/results/h200/efficiency.csv and writes
Journal_Paper/figures/revision/h200_efficiency.png at 600 dpi.

Left panel : NLI throughput vs batch size for the deployed stack, with the two
             alternative encoders marked at batch 256.
Right panel: peak allocated VRAM on the same runs.

Run (repo root):
  python src/models/plot_h200_efficiency.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts" / "results" / "h200" / "efficiency.csv"
OUT = ROOT / "Journal_Paper" / "figures" / "revision" / "h200_efficiency.png"
DPI = 600

C_BASE = "#2C3E7B"
C_LARGE = "#C0392B"
C_BGE = "#1E8449"
C_REF = "#7F8C8D"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "legend.fontsize": 8.5,
    }
)


def main() -> None:
    df = pd.read_csv(SRC)
    df = df[df["status"] == "ok"].copy()

    sweep = df[df["config"].isin(["deployed", "deployed_bs128", "deployed_bs512"])].sort_values("batch_size")
    if len(sweep) != 3:
        raise ValueError(f"expected three batch points, found {len(sweep)}")
    large = df[df["config"] == "nli_large"].iloc[0]
    bge = df[df["config"] == "bge_base"].iloc[0]
    base256 = df[df["config"] == "deployed"].iloc[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.75), constrained_layout=True)

    bs = sweep["batch_size"].to_numpy()
    ax1.plot(bs, sweep["nli_rows_per_s"], "-o", color=C_BASE, lw=1.6, ms=5, label="DeBERTa-base + MiniLM")
    ax1.axhline(base256["nli_rows_per_s"], color=C_REF, lw=0.8, ls=":", zorder=0)
    ax1.plot([256], [large["nli_rows_per_s"]], "s", color=C_LARGE, ms=6, label="DeBERTa-large + MiniLM")
    ax1.plot([256], [bge["nli_rows_per_s"]], "^", color=C_BGE, ms=6, label="DeBERTa-base + BGE-base")
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("NLI throughput (rows/s)")
    ax1.set_xticks([128, 256, 512])
    ax1.set_ylim(0, max(sweep["nli_rows_per_s"].max(), large["nli_rows_per_s"]) * 1.35)
    ax1.legend(loc="upper right", frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(bs, sweep["peak_vram_mb"] / 1024, "-o", color=C_BASE, lw=1.6, ms=5, label="DeBERTa-base + MiniLM")
    ax2.plot([256], [large["peak_vram_mb"] / 1024], "s", color=C_LARGE, ms=6, label="DeBERTa-large + MiniLM")
    ax2.plot([256], [bge["peak_vram_mb"] / 1024], "^", color=C_BGE, ms=6, label="DeBERTa-base + BGE-base")
    ax2.axhline(15.9, color=C_REF, lw=0.8, ls="--")
    ax2.text(130, 16.3, "slice limit 15.9 GB", fontsize=7.5, color=C_REF)
    ax2.set_xlabel("Batch size")
    ax2.set_ylabel("Peak VRAM (GB)")
    ax2.set_xticks([128, 256, 512])
    ax2.set_ylim(0, 18.6)
    ax2.spines[["top", "right"]].set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=DPI, facecolor="white")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
