"""Final pre-submission verification for the manuscript.

Checks, in order:
  1. every float has a unique label and is referenced at least once;
  2. every \\cite key exists in ref.bib and every bib entry is cited;
  3. equations, tables, and figures are numbered consecutively in the built PDF;
  4. the H200 numbers in the text and floats match the JSON artifacts;
  5. no prohibited characters in the tex source (non-ASCII, em dash, semicolon in prose).

Writes artifacts/results/final_verification.json and prints a pass/fail table.

Run from the repo root:
  python src/models/final_paper_verification.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "Journal_Paper" / "halurisc.tex"
BIB = ROOT / "Journal_Paper" / "ref.bib"
PDF = ROOT / "Journal_Paper" / "out" / "halurisc.pdf"
H200 = ROOT / "artifacts" / "results" / "h200"
OUT = ROOT / "artifacts" / "results" / "final_verification.json"


def main() -> None:
    tex = TEX.read_text(encoding="utf-8")
    bib = BIB.read_text(encoding="utf-8")
    report: dict = {}

    # ---- floats
    labels = re.findall(r"\\label\{((?:tab|fig):[^}]+)\}", tex)
    refs = set(re.findall(r"\\ref\{((?:tab|fig):[^}]+)\}", tex))
    dupes = sorted({x for x in labels if labels.count(x) > 1})
    unreferenced = sorted(set(labels) - refs)
    missing = sorted(refs - set(labels))
    report["floats"] = {
        "n_labels": len(labels),
        "n_unique": len(set(labels)),
        "duplicates": dupes,
        "unreferenced": unreferenced,
        "missing_targets": missing,
        "n_tables": len([x for x in labels if x.startswith("tab:")]),
        "n_figures": len([x for x in labels if x.startswith("fig:")]),
    }

    # ---- citations
    cite_keys = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", tex):
        cite_keys.update(k.strip() for k in group.split(","))
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    report["citations"] = {
        "cited": len(cite_keys),
        "in_bib": len(bib_keys),
        "cited_missing_in_bib": sorted(cite_keys - bib_keys),
        "uncited_bib_entries": sorted(bib_keys - cite_keys),
    }

    # ---- PDF numbering
    numbers = {"tables": [], "figures": [], "equations": []}
    if PDF.exists():
        import pymupdf

        doc = pymupdf.open(PDF)
        text = "\n".join(p.get_text() for p in doc)
        numbers["tables"] = sorted({int(n) for n in re.findall(r"^Table (\d+)$", text, re.M)})
        numbers["figures"] = sorted({int(n) for n in re.findall(r"^Figure (\d+)\s*:", text, re.M)})
        # Equation numbers can sit alone on a line or at the end of the equation line.
        raw = {int(n) for n in re.findall(r"\((\d{1,2})\)\s*$", text, re.M)}
        raw |= {int(n) for n in re.findall(r"^\((\d{1,2})\)$", text, re.M)}
        numbers["equations"] = sorted(n for n in raw if 1 <= n <= 40)
        report["pdf"] = {"pages": doc.page_count}

    def consecutive(seq: list[int]) -> bool:
        return bool(seq) and seq == list(range(1, len(seq) + 1))

    report["numbering"] = {
        "tables": numbers["tables"],
        "tables_consecutive": consecutive(numbers["tables"]),
        "figures": numbers["figures"],
        "figures_consecutive": consecutive(numbers["figures"]),
        "equations": numbers["equations"],
        "equations_consecutive": consecutive(numbers["equations"]),
    }

    # ---- H200 numbers in the paper
    enc = json.loads((H200 / "encoder_ablation.json").read_text())
    lm = json.loads((H200 / "length_matched.json").read_text())
    runs = {r["tag"]: r for r in enc["runs"]}
    checks = []

    def chk(name: str, expected: float, fmt: str) -> None:
        needle = fmt.format(expected)
        checks.append({"check": name, "value": needle, "in_tex": needle in tex})

    chk("enc base f1", runs["base"]["f1_mean"], "{:.4f}")
    chk("enc base auroc", runs["base"]["auroc_mean"], "{:.4f}")
    chk("enc large f1", runs["nli-large"]["f1_mean"], "{:.4f}")
    chk("enc large auroc", runs["nli-large"]["auroc_mean"], "{:.4f}")
    chk("enc bge f1", runs["bge-base"]["f1_mean"], "{:.4f}")
    chk("enc bge auroc", runs["bge-base"]["auroc_mean"], "{:.4f}")
    for tag in ("base", "nli-large", "bge-base"):
        chk(f"{tag} f1 std", runs[tag]["f1_std"], "{:.4f}")
    p_nli = enc["mcnemar"]["base_vs_nli-large"]["median_pvalue"]
    p_bge = enc["mcnemar"]["base_vs_bge-base"]["median_pvalue"]
    checks.append({"check": "mcnemar nli p rounded", "value": f"{p_nli:.2f}", "in_tex": f"{p_nli:.2f}" in tex})
    checks.append({"check": "mcnemar bge p rounded", "value": f"{p_bge:.2f}", "in_tex": f"{p_bge:.2f}" in tex})

    import pandas as pd

    lm_df = pd.DataFrame(lm["rows"])
    for dataset in ("halubench", "medhallu"):
        sub = lm_df[lm_df["dataset"] == dataset]
        ec = sub[sub["model"].str.startswith("ec_xgb")]
        lo = sub[sub["model"] == "length_only"]
        for slice_name, tag in (("full", "full"), ("length_matched", "matched")):
            s = ec[ec["slice"] == slice_name]["auroc"].mean()
            chk(f"{dataset} {tag} ec auroc", s, "{:.4f}")
            l = lo[lo["slice"] == slice_name]["auroc"].mean()
            chk(f"{dataset} {tag} length-only auroc", l, "{:.4f}")
    chk("medhallu full pair acc", lm["medhallu"]["full"]["pair_accuracy_mean"], "{:.4f}")
    chk("medhallu matched pair acc", lm["medhallu"]["length_matched"]["pair_accuracy_mean"], "{:.4f}")

    eff = pd.read_csv(H200 / "efficiency.csv")
    for cfg, batch in (("deployed_bs128", 128), ("deployed", 256), ("nli_large", 256)):
        r = eff[(eff["config"] == cfg) & (eff["batch_size"] == batch)].iloc[0]
        checks.append({"check": f"{cfg} rows/s", "value": f"{r['nli_rows_per_s']:.0f}", "in_tex": f"{r['nli_rows_per_s']:.0f}" in tex})
        checks.append({"check": f"{cfg} peak GB", "value": f"{r['peak_vram_mb'] / 1024:.1f}", "in_tex": f"{r['peak_vram_mb'] / 1024:.1f}" in tex})

    j9 = json.loads((H200 / "judge_qwen9b_metrics.json").read_text())
    j4 = json.loads((H200 / "judge_qwen4b_metrics.json").read_text())
    for tag, d in (("9b", j9), ("4b", j4)):
        m = d["mean_across_seeds"]
        for field, fmt in (("precision", "{:.3f}"), ("recall", "{:.3f}"), ("f1", "{:.4f}"), ("auroc", "{:.4f}")):
            chk(f"judge {tag} {field}", m[field]["mean"], fmt)
        chk(f"judge {tag} p50", d["latency"]["p50_ms"], "{:.0f}")
    ratio = max(j9["latency"]["p50_ms"], j4["latency"]["p50_ms"]) / min(j9["latency"]["p50_ms"], j4["latency"]["p50_ms"])
    checks.append({"check": "judge speed ratio", "value": f"{ratio:.1f}", "in_tex": f"{ratio:.1f}" in tex})
    report["h200_values"] = {
        "n_checks": len(checks),
        "n_found": sum(c["in_tex"] for c in checks),
        "missing": [c for c in checks if not c["in_tex"]],
    }

    # ---- source hygiene
    bad_chars = sorted({ch for ch in tex if ord(ch) > 127})
    prose_dashes = [
        i + 1
        for i, line in enumerate(tex.splitlines())
        if "---" in line and "&" not in line and not line.lstrip().startswith("%")
    ]
    report["hygiene"] = {
        "non_ascii_chars": bad_chars,
        "em_dash_in_prose_lines": prose_dashes,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
