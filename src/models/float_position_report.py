"""Report where each float sits relative to its first in-text reference.

Reads the built PDF, finds caption pages (a line that is just "Table N" or starts
with "Figure N:") and the first page where the float is referenced in prose, then
prints the distance in pages.

Run from the repo root:
  python src/models/float_position_report.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "Journal_Paper" / "out" / "halurisc.pdf"
OUT = ROOT / "artifacts" / "results" / "float_position_report.json"


def main() -> None:
    import pymupdf

    doc = pymupdf.open(PDF)
    pages = [p.get_text() for p in doc]

    cap: dict[str, int] = {}
    refs: dict[str, list[int]] = {}

    for i, text in enumerate(pages, 1):
        lines = [ln.strip() for ln in text.splitlines()]
        for ln in lines:
            m = re.fullmatch(r"Table (\d+)", ln)
            if m:
                cap.setdefault(f"Table {m.group(1)}", i)
            m = re.match(r"Figure (\d+):", ln)
            if m:
                cap.setdefault(f"Figure {m.group(1)}", i)
        for kind, pat in (("Table", r"Table~?\s*(\d+)"), ("Figure", r"Figure~?\s*(\d+)")):
            for m in re.finditer(pat, text):
                n = m.group(1)
                key = f"{kind} {n}"
                # skip the caption line itself (nothing else on that line)
                line_start = text.rfind("\n", 0, m.start()) + 1
                line_end = text.find("\n", m.end())
                line = text[line_start:line_end].strip()
                if re.fullmatch(rf"{kind} {n}:?", line) or line.startswith(f"{kind} {n}:"):
                    continue
                refs.setdefault(key, []).append(i)

    rows = []
    for key, cp in sorted(cap.items(), key=lambda kv: (kv[0].split()[0], int(kv[0].split()[1]))):
        rp = min(refs.get(key, [10**6]))
        rows.append({"float": key, "caption_page": cp, "first_ref_page": rp if rp < 10**6 else None,
                     "distance": (cp - rp) if rp < 10**6 else None})

    bad = [r for r in rows if r["first_ref_page"] is None or abs(r["distance"]) > 2]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"n_floats": len(rows), "rows": rows, "bad": bad}, indent=2))

    print(f"{'float':<12}{'caption':>8}{'1st ref':>9}{'dist':>6}")
    for r in rows:
        print(f"{r['float']:<12}{r['caption_page']:>8}{str(r['first_ref_page']):>9}{str(r['distance']):>6}")
    print()
    print("outside the two-page rule:", [r["float"] for r in bad] or "none")


if __name__ == "__main__":
    main()
