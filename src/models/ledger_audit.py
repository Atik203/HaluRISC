"""Verify that rewriten.md's four paragraphs and captions match halurisc.tex verbatim.

Parses the current ledger format:
  ## Block A. ...
  Drop in ...
  <body>
  Caption. <caption>
  > Status: ...

Normalises LaTeX (resolves \\ref to float numbers, strips markup) and checks each body
and caption appears in the manuscript. Also reports first-person pronouns in prose.

Run from the repo root:
  python src/models/ledger_audit.py
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "Journal_Paper" / "halurisc.tex"
LEDGER = ROOT / "rewriten.md"
OUT = ROOT / "artifacts" / "results" / "ledger_audit.json"


def main() -> None:
    tex = TEX.read_text(encoding="utf-8")
    ledger = LEDGER.read_text(encoding="utf-8")

    counters = {"table": 0, "figure": 0}
    label2num: dict[str, str] = {}
    for m in re.finditer(r"\\begin\{(table|figure)\}(.*?)\\end\{\1\}", tex, re.S):
        kind = m.group(1)
        counters[kind] += 1
        lab = re.search(r"\\label\{([^}]+)\}", m.group(2))
        if lab:
            label2num[lab.group(1)] = str(counters[kind])

    def norm(s: str) -> str:
        s = re.sub(r"~\s*\\ref\{([^}]+)\}", lambda m: " " + label2num.get(m.group(1), "??"), s)
        s = re.sub(r"\\ref\{([^}]+)\}", lambda m: label2num.get(m.group(1), "??"), s)
        s = s.replace(r"\%", "%").replace(r"\_", "_").replace(r"\$", "$").replace("~", " ")
        s = s.replace("$", "").replace("{", "").replace("}", "")
        s = re.sub(r"\\(?:mathrm|emph|textbf|textit)", "", s)
        return re.sub(r"\s+", " ", s).strip().lower()

    norm_tex = norm(tex)
    checks = []
    for letter in "ABCD":
        sec = re.search(rf"## Block {letter}\..*?(?=\n## Block |\Z)", ledger, re.S)
        assert sec, f"Block {letter} header not found"
        chunk = sec.group(0)
        bm = re.search(r"Drop in[^\n]*\n\n(.*?)(?=\n\n> Status|\n\nCaption\.)", chunk, re.S)
        cm = re.search(r"Caption\.\s*(.*?)(?=\n\n---|\Z)", chunk, re.S)
        assert bm and cm, f"Block {letter} body/caption not parsed"
        body, caption = bm.group(1).strip(), cm.group(1).strip()
        checks.append({
            "block": letter,
            "body_words": len(body.split()),
            "body_in_tex": norm(body) in norm_tex,
            "caption_in_tex": norm(caption) in norm_tex,
        })

    # first-person pronouns in non-comment prose
    pronouns = []
    for i, line in enumerate(tex.splitlines(), 1):
        if line.strip().startswith("%"):
            continue
        for pm in re.finditer(r"\b(we|our|ours|us)\b", line, re.I):
            pronouns.append({"line": i, "word": pm.group(0)})

    report = {"block_checks": checks, "first_person_pronouns": pronouns}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for c in checks:
        print(f"Block {c['block']}: {c['body_words']:>3} words | body={c['body_in_tex']} caption={c['caption_in_tex']}")
    print(f"\nfirst-person pronouns in prose: {len(pronouns)}")
    for p in pronouns:
        print(f"  L{p['line']}: {p['word']}")


if __name__ == "__main__":
    main()
