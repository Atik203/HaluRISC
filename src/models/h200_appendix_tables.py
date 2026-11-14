"""H200 appendix: assemble the LaTeX rows for the appendix tables (numbers only).

Reads the E1 encoder ablation, E2 length-matched control, E3 local-judge metrics
(when present), and E4 efficiency sweep, and writes ready-to-paste LaTeX rows to
artifacts/results/h200/appendix_tables.tex.

This script does NOT edit the manuscript. It only formats verified numbers so the
author can paste them and write the surrounding prose.

Run (repo root):
  python src/models/h200_appendix_tables.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("h200_appendix_tables")

ROOT = Path(__file__).resolve().parents[2]
H200 = ROOT / "artifacts" / "results" / "h200"
JUDGE = ROOT / "artifacts" / "results" / "rebuild" / "judge"
OUT_TEX = H200 / "appendix_tables.tex"
OUT_CSV = H200 / "appendix_tables.csv"


def fmt(x: float, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "--"
    return f"{x:.{nd}f}"


def delta(x: float, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "--"
    sign = "+" if x >= 0 else "-"
    return f"${sign}${abs(x):.{nd}f}"


def enc_rows() -> list[str]:
    p = H200 / "encoder_ablation.json"
    if not p.exists():
        logger.warning("missing %s", p)
        return []
    d = json.loads(p.read_text())
    runs = {r["tag"]: r for r in d["runs"]}
    base = runs["base"]
    stats = d["mcnemar"]
    p_nli = stats["base_vs_nli-large"]["median_pvalue"]
    p_bge = stats["base_vs_bge-base"]["median_pvalue"]

    names = {
        "base": "DeBERTa-v3-base + MiniLM",
        "nli-large": "DeBERTa-v3-large + MiniLM",
        "bge-base": "DeBERTa-v3-base + BGE-base",
    }
    pvals = {"base": "--", "nli-large": fmt(p_nli, 2), "bge-base": fmt(p_bge, 2)}

    rows = []
    for tag in ("base", "nli-large", "bge-base"):
        r = runs[tag]
        is_base = tag == "base"
        cells = [
            names[tag],
            f"{fmt(r['f1_mean'])} $\\pm$ {fmt(r['f1_std'])}",
            f"{fmt(r['auroc_mean'])} $\\pm$ {fmt(r['auroc_std'])}",
            "--" if is_base else delta(r["f1_mean"] - base["f1_mean"]),
            "--" if is_base else delta(r["auroc_mean"] - base["auroc_mean"]),
            f"${pvals[tag]}$" if pvals[tag] != "--" else "--",
        ]
        rows.append(" & ".join(cells) + r" \\")
    return rows


def length_rows() -> list[str]:
    p = H200 / "length_matched.json"
    if not p.exists():
        logger.warning("missing %s", p)
        return []
    d = json.loads(p.read_text())
    df = pd.DataFrame(d["rows"])
    rows = []

    for dataset, label in (("halubench", "HaluBench"), ("medhallu", "MedHallu")):
        sub = df[df["dataset"] == dataset]
        full_ec = sub[(sub["slice"] == "full") & (sub["model"].str.startswith("ec_xgb"))]["auroc"].mean()
        for slice_name, slice_label in (("full", "full"), ("length_matched", "length-matched")):
            s = sub[sub["slice"] == slice_name]
            ec = s[s["model"].str.startswith("ec_xgb")]
            lo = s[s["model"] == "length_only"]
            rows.append(
                " & ".join(
                    [
                        label,
                        slice_label,
                        f"{fmt(ec['auroc'].mean())} $\\pm$ {fmt(ec['auroc'].std())}",
                        fmt(lo["auroc"].mean()),
                        "--" if slice_name == "full" else delta(ec["auroc"].mean() - full_ec),
                    ]
                )
                + r" \\"
            )

    pair = d.get("medhallu", {})
    full = pair.get("full", {})
    if full:
        rows.append(
            "\\multicolumn{5}{l}{MedHallu pair accuracy: full "
            f"{fmt(full.get('pair_accuracy_mean'))}, length-matched "
            f"{fmt(pair.get('length_matched', {}).get('pair_accuracy_mean'))}}}"
            + r" \\"
        )
    return rows


def eff_rows() -> list[str]:
    p = H200 / "efficiency.csv"
    if not p.exists():
        logger.warning("missing %s", p)
        return []
    df = pd.read_csv(p)
    names = {
        "deployed": "DeBERTa-base + MiniLM",
        "nli_large": "DeBERTa-large + MiniLM",
        "bge_base": "DeBERTa-base + BGE-base",
        "deployed_bs128": "DeBERTa-base + MiniLM",
        "deployed_bs512": "DeBERTa-base + MiniLM",
    }
    rows = []
    for _, r in df.iterrows():
        if r.get("status") != "ok":
            continue
        label = names.get(r["config"], str(r["config"]).replace("_", " "))
        rows.append(
            " & ".join(
                [
                    label,
                    str(int(r["batch_size"])),
                    f"{r['nli_rows_per_s']:.0f}",
                    f"{r['peak_vram_mb'] / 1024:.1f}",
                ]
            )
            + r" \\"
        )
    return rows


def judge_rows() -> list[str]:
    rows = []
    for tag, label in (("qwen9b", "Qwen3.5-9B"), ("qwen4b", "Qwen3.5-4B")):
        p = JUDGE / f"metrics_qwen_{tag}.json"
        if not p.exists():
            p = JUDGE / f"metrics_{tag}.json"
        if not p.exists():
            logger.info("judge metrics not ready for %s", tag)
            continue
        d = json.loads(p.read_text())
        m = d.get("mean_across_seeds", {})
        lat = d.get("latency", {})
        cost = d.get("cost", {})
        rows.append(
            " & ".join(
                [
                    label,
                    f"{m.get('precision', {}).get('mean', float('nan')):.3f}",
                    f"{m.get('recall', {}).get('mean', float('nan')):.3f}",
                    f"{m.get('f1', {}).get('mean', float('nan')):.3f}",
                    f"{m.get('auroc', {}).get('mean', float('nan')):.3f}",
                    f"{m.get('predicted_positive_rate', {}).get('mean', float('nan')) * 100:.1f}",
                    f"{lat.get('p50_ms', float('nan')):.0f}",
                    f"{cost.get('usd_per_1000_calls', float('nan')):.0f}",
                ]
            )
            + r" \\"
        )
    return rows


def main() -> None:
    sections = {
        "encoder_ablation": enc_rows(),
        "length_matched": length_rows(),
        "efficiency": eff_rows(),
        "local_judge": judge_rows(),
    }
    lines = [
        "% Generated by src/models/h200_appendix_tables.py -- numbers only, no prose.",
        "% Source of truth: artifacts/results/h200/*.json and artifacts/results/rebuild/judge/.",
        "",
    ]
    for name, rows in sections.items():
        lines.append(f"% ---- {name} ----")
        if not rows:
            lines.append("% (not available yet)")
        lines.extend(rows)
        lines.append("")
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", OUT_TEX)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
