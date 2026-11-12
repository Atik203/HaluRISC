"""Plot the seed-42 HaluEval confusion matrix for the fixed-hp m3 model.

Reads the saved predictions so the figure is derived from the artifact rather
than hard-coded counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "artifacts/results/rebuild/fixed_hp/fixed_hp_predictions.parquet"
DEFAULT_OUT = ROOT / "Journal_Paper/figures/revision/confusion_matrix.png"


def build_matrix(path: Path = PREDICTIONS) -> np.ndarray:
    frame = pd.read_parquet(path)
    rows = frame[
        (frame["variant"] == "m3")
        & (frame["seed"] == 42)
        & (frame["dataset"] == "halueval_test")
    ]
    if rows.empty:
        raise SystemExit(f"No m3 seed-42 HaluEval rows in {path}")
    y = rows["label"].to_numpy()
    pred = (rows["score"].to_numpy() >= 0.5).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    matrix = np.array([[tp, fn], [fp, tn]])
    assert matrix.sum() == 3000, matrix.sum()
    print(f"TP={tp} FN={fn} FP={fp} TN={tn}")
    return matrix


def plot(matrix: np.ndarray, out: Path, dpi: int) -> None:
    row_pct = matrix / matrix.sum(axis=1, keepdims=True) * 100.0

    cmap = LinearSegmentedColormap.from_list(
        "halurisc", ["#f7f8ff", "#dbe3ff", "#8b8ff0", "#4338ca", "#1e1b4b"]
    )

    fig, ax = plt.subplots(figsize=(4.2, 3.5), dpi=dpi)
    image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=float(matrix.max()))

    for i in range(2):
        for j in range(2):
            dark = matrix[i, j] > matrix.max() * 0.55
            ax.text(
                j,
                i,
                f"{matrix[i, j]:,}\n({row_pct[i, j]:.1f}%)",
                ha="center",
                va="center",
                fontsize=11,
                color="white" if dark else "#111827",
                linespacing=1.35,
            )

    labels = ["Hallucinated", "Grounded"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("Actual label", fontsize=10)
    ax.tick_params(axis="both", labelsize=9.5, length=0)

    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out} at {dpi} DPI")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=650)
    args = parser.parse_args()
    plot(build_matrix(), args.out, args.dpi)


if __name__ == "__main__":
    main()
