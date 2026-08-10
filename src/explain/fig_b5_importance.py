"""Regenerate the SHAP importance figure for the paper from verified B5 data.

Reads artifacts/results/b5/b5_feature_importance.json (frozen B-run) and
renders the top-12 mean |SHAP| bar chart. Run from the repo root:

  .venv/Scripts/python.exe src/explain/fig_b5_importance.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts" / "results" / "b5" / "b5_feature_importance.json"
OUT = ROOT / "artifacts" / "figures" / "fig_shap_importance_b5.png"


def main() -> None:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    mean_abs_shap = data["mean_abs_shap"]
    items = sorted(mean_abs_shap.items(), key=lambda kv: kv[1], reverse=True)[:12]
    names = [k for k, _ in items]
    values = [v for k, v in items]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(values)), values, color="#4f46e5")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("mean |SHAP|")
    ax.set_title("Top-12 features by mean |SHAP| (B-run, seed 42)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
