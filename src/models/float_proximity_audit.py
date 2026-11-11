"""Check that every table and figure is mentioned within two pages of its float.

Reads Journal_Paper/out/halurisc.aux for the label -> number/page map and
extracts the PDF text with pypdf. Run after a clean build:

  python src/models/float_proximity_audit.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "Journal_Paper"
AUX = PAPER / "out" / "halurisc.aux"
PDF = PAPER / "out" / "halurisc.pdf"


def main() -> int:
    if not AUX.exists() or not PDF.exists():
        print("missing aux or pdf; build the paper first", file=sys.stderr)
        return 2

    labels = {}
    pattern = re.compile(r"\\newlabel\{((?:tab|fig):[^}]+)\}\{\{([^}]+)\}\{([^}]+)\}")
    for line in AUX.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            labels[match.group(1)] = (match.group(2), int(match.group(3)))

    reader = PdfReader(str(PDF))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(re.sub(r"\s+", " ", text))

    rows = []
    for label, (number, float_page) in sorted(labels.items()):
        kind = "Table" if label.startswith("tab:") else "Figure"
        regex = re.compile(rf"\b{kind}\s+{re.escape(number)}\b")
        first_page = None
        for index, text in enumerate(pages, start=1):
            if regex.search(text):
                first_page = index
                break
        if first_page is None:
            status = "not found"
            gap = None
        else:
            gap = float_page - first_page
            if gap < 0:
                status = "float before mention"
            elif gap <= 2:
                status = "ok"
            else:
                status = "too far"
        rows.append({
            "label": label,
            "kind": kind,
            "number": number,
            "float_page": float_page,
            "first_mention_page": first_page,
            "gap": gap,
            "status": status,
        })

    bad = [r for r in rows if r["status"] != "ok"]
    print(json.dumps({
        "n_floats": len(rows),
        "n_ok": len(rows) - len(bad),
        "n_bad": len(bad),
        "bad": bad,
    }, indent=2))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
