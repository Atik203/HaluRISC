"""Verify that rewriten.md's four paragraphs and captions match halurisc.tex verbatim.

Reads the built manuscript and the rewrite ledger, resolves \ref to float numbers,
normalises LaTeX, and checks each block body and caption appears in the tex.
Also proves no original prose line was lost relative to main.

Run from the repo root:
  python src/models/ledger_audit.py
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "Journal_Paper" / "halurisc.tex"
LEDGER = ROOT / "rewriten.md"
OUT = ROOT / "artifacts" / "results" / "ledger_audit.json"


def main() -> None:
    tex = TEX.read_text(encoding="utf-8")
    ledger = LEDGER.read_text(encoding="utf-8")
    main_tex = subprocess.check_output(
        ["git", "show", "main:Journal_Paper/halurisc.tex"], cwd=ROOT, text=True, encoding="utf-8"
    )

    counters = {"table": 0, "figure": 0}
    label2num: dict[str, str] = {}
    for m in re.finditer(r"\\begin\{(table|figure)\}([^\n]*)(.*?)\\end\{\1\}", tex, re.S):
        kind = m.group(1)
        counters[kind] += 1
        lab = re.search(r"\\label\{([^}]+)\}", m.group(3))
        if lab:
            label2num[lab.group(1)] = str(counters[kind])

    def norm(s: str) -> str:
        s = re.sub(r"~\s*\\ref\{([^}]+)\}", lambda m: " " + label2num.get(m.group(1), "??"), s)
        s = re.sub(r"\\ref\{([^}]+)\}", lambda m: label2num.get(m.group(1), "??"), s)
        s = s.replace(r"\%", "%").replace(r"\_", "_").replace("~", " ")
        s = s.replace("$", "").replace("{", "").replace("}", "")
        s = re.sub(r"\\(?:mathrm|emph|textbf|textit)", "", s)
        return re.sub(r"\s+", " ", s).strip().lower()

    norm_tex = norm(tex)
    checks = []
    for letter in "ABCD":
        m = re.search(
            rf"## Block {letter}\..*?\n\nDrop in[^\n]*\n\n(.*?)\n\nCaption\.\s*(.*?)(?=\n\n---|\Z)",
            ledger, re.S,
        )
        assert m, f"Block {letter} not parsed from ledger"
        body, caption = m.group(1).strip(), m.group(2).strip()
        checks.append({
            "block": letter,
            "body_words": len(body.split()),
            "body_in_tex": norm(body) in norm_tex,
            "caption_in_tex": norm(caption) in norm_tex,
        })

    def prose_lines(t: str) -> Counter:
        return Counter(
            ln.strip() for ln in t.splitlines()
            if ln.strip() and not ln.strip().startswith("%")
        )

    cm, ch = prose_lines(main_tex), prose_lines(tex)
    removed = list((cm - ch).elements())
    report = {
        "block_checks": checks,
        "original_prose_lines_missing": removed,
        "n_original_prose_lines_missing": len(removed),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for c in checks:
        print(f"Block {c['block']}: {c['body_words']:>3} words | body={c['body_in_tex']} caption={c['caption_in_tex']}")
    print(f"\nOriginal prose lines missing vs main: {len(removed)}")
    for ln in removed:
        print("  -", ln[:130])


if __name__ == "__main__":
    main()
