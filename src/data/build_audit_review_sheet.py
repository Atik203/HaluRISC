"""
Build the human review sheet for the 50-sample HaluEval label audit.

Reads  : data/processed/audit_50_samples.json (auto-sampled by src/data/prepare.py)
Writes : data/processed/audit_50_samples_review.csv (review columns empty)

Review by hand:
  - human_label_correct: does the dataset label match what you read?
      yes   = label is correct
      no    = label looks wrong (mislabeled sample)
      unsure = genuinely ambiguous
  - human_confidence: high | medium | low
  - reviewer: your name
  - notes: one short line when the verdict is no or unsure

Save the filled sheet as data/processed/audit_50_samples_reviewed.csv and report
the mismatch rate in the paper's dataset-limitations subsection.

Run (repo root, .venv):
  python src/data/build_audit_review_sheet.py
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "processed" / "audit_50_samples.json"
OUT = ROOT / "data" / "processed" / "audit_50_samples_review.csv"

COLUMNS = [
    "sample_id",
    "item_idx",
    "question",
    "context",
    "answer",
    "label",
    "split",
    "human_label_correct",
    "human_confidence",
    "reviewer",
    "notes",
]

REVIEW_COLUMNS = ("human_label_correct", "human_confidence", "reviewer", "notes")


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Audit file not found: {SOURCE}. Run src/data/prepare.py first.")

    rows = json.loads(SOURCE.read_text(encoding="utf-8"))
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            record = {k: r.get(k, "") for k in COLUMNS if k not in REVIEW_COLUMNS}
            record.update({k: "" for k in REVIEW_COLUMNS})
            writer.writerow(record)

    print(f"Wrote {len(rows)} audit rows to {OUT}")
    print("Fill human_label_correct (yes|no|unsure); save as audit_50_samples_reviewed.csv.")


if __name__ == "__main__":
    main()
