"""
B5.5 review tally — reads the manually filled reviewer sheet and prints
descriptive agreement statistics. No ML, no thresholds (B5.7: distributions
only). Reviews are a human step; this script only tallies what humans wrote.

Input (default): artifacts/results/b5/b5_review_cases_reviewed.csv
Columns expected: ... reviewer_1, reviewer_2, agreement (as produced by the
B5 export; reviewers fill reviewer_1/reviewer_2/agreement by hand).

Run (repo root, .venv):
  python src/models/review_tally.py [path/to/reviewed.csv]
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHEET = ROOT / "artifacts" / "results" / "b5" / "b5_review_cases_reviewed.csv"

VALID_MARKS = {"plausible", "implausible", "unsure"}


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tally B5.5 reviewer agreement (descriptive only)")
    parser.add_argument("sheet", nargs="?", default=str(DEFAULT_SHEET), help="path to the reviewed CSV")
    args = parser.parse_args(argv)

    sheet = Path(args.sheet)
    if not sheet.exists():
        print(f"Sheet not found: {sheet}")
        print("Run the manual review first (docs/b5-explanation-reliability.md) and save as")
        print("artifacts/results/b5/b5_review_cases_reviewed.csv")
        return 1

    import csv

    with open(sheet, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("Sheet is empty.")
        return 1

    def mark(row, col: str):
        v = (row.get(col) or "").strip().lower()
        return v if v in VALID_MARKS else None

    r1 = Counter(mark(r, "reviewer_1") for r in rows)
    r2 = Counter(mark(r, "reviewer_2") for r in rows)
    reviewed = [r for r in rows if mark(r, "reviewer_1") or mark(r, "reviewer_2")]
    agreed_yes = sum(1 for r in rows if (r.get("agreement") or "").strip().lower() == "yes")
    agreed_no = sum(1 for r in rows if (r.get("agreement") or "").strip().lower() == "no")

    print(f"Cases in sheet:        {len(rows)}")
    print(f"Cases with ≥1 mark:    {len(reviewed)}")
    print(f"Agreement=yes:         {agreed_yes}  ({agreed_yes / len(rows):.0%} of all cases)" if rows else "")
    print(f"Agreement=no:          {agreed_no}")
    print(f"Reviewer 1 marks:      {dict(r1)}")
    print(f"Reviewer 2 marks:      {dict(r2)}")

    # Cross-check: agreement column vs reviewer marks (discrepancies are worth a look)
    flagged = []
    for r in rows:
        a = (r.get("agreement") or "").strip().lower()
        m1, m2 = mark(r, "reviewer_1"), mark(r, "reviewer_2")
        if m1 and m2:
            computed = "yes" if m1 == m2 else "no"
            if a in ("yes", "no") and a != computed:
                flagged.append(r.get("sample_id"))
    if flagged:
        print(f"WARN: agreement column contradicts reviewer marks for {len(flagged)} cases: {flagged[:10]}{' ...' if len(flagged) > 10 else ''}")
        print("(Disagreements are a deliverable — report them, don't hide them.)")

    disagreements = [r.get("sample_id") for r in rows if (r.get("agreement") or "").strip().lower() == "no"]
    if disagreements:
        print(f"Recorded disagreements ({len(disagreements)}): {disagreements[:15]}{' ...' if len(disagreements) > 15 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
