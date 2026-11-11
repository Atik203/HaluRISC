"""Merge the three AI reviewer copies into the P0 ingest layout.

This is the bridge for the three-reviewer draft. It does not decide labels and
does not edit the original sheets. It writes:

  docs/rebuild/p0/author_review/p0_blind_overlap_merged.csv
  docs/rebuild/p0/author_review/ingest/p0_error_taxonomy.csv
  docs/rebuild/p0/author_review/ingest/p0_explanation_audit.csv
  docs/rebuild/p0/author_review/ingest/p0_claim_verdicts.csv

Run after the three reviewers have filled their copies:

  python src/models/merge_p0_author_review.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "docs" / "rebuild" / "p0" / "author_review"
INGEST = REVIEW / "ingest"

BLIND = {
    "a1": REVIEW / "blind_reviewer_a_scored.csv",
    "a2": REVIEW / "blind_reviewer_b_scored.csv",
    "a3": REVIEW / "blind_reviewer_c_scored.csv",
}
MAIN = {
    "p0_error_taxonomy.csv": REVIEW / "taxonomy_reviewer_a.csv",
    "p0_explanation_audit.csv": REVIEW / "explanation_reviewer_b.csv",
    "p0_claim_verdicts.csv": REVIEW / "claims_reviewer_c.csv",
}


def main() -> int:
    for label, path in BLIND.items():
        if not path.exists():
            print(f"missing blind copy: {path}", file=sys.stderr)
            return 2
        frame = pd.read_csv(path)
        if len(frame) != 60:
            print(f"{path.name}: expected 60 rows, found {len(frame)}", file=sys.stderr)
            return 2
        if frame["author_label"].isna().any():
            print(f"{path.name}: blank author_label rows", file=sys.stderr)
            return 2

    frames = {}
    for label, path in BLIND.items():
        frame = pd.read_csv(path).set_index("item_id")
        frames[label] = frame["author_label"].rename(label)

    merged = pd.concat(frames.values(), axis=1)
    sheet = pd.read_csv(BLIND["a1"]).set_index("item_id")["sheet"]
    merged = merged.join(sheet)
    merged = merged.reset_index()[["item_id", "sheet", "a1", "a2", "a3"]]
    merged.to_csv(REVIEW / "p0_blind_overlap_merged.csv", index=False)
    print(f"wrote {REVIEW / 'p0_blind_overlap_merged.csv'}  rows={len(merged)}")

    INGEST.mkdir(parents=True, exist_ok=True)
    for target, source in MAIN.items():
        if not source.exists():
            print(f"missing main review: {source}", file=sys.stderr)
            return 2
        frame = pd.read_csv(source)
        if "author_label" not in frame or frame["author_label"].isna().any():
            print(f"{source.name}: blank author_label rows", file=sys.stderr)
            return 2
        shutil.copyfile(source, INGEST / target)
        print(f"wrote {INGEST / target}  rows={len(frame)}")

    shutil.copyfile(REVIEW / "p0_blind_overlap_merged.csv", INGEST / "p0_blind_overlap_merged.csv")
    print(f"wrote {INGEST / 'p0_blind_overlap_merged.csv'}")
    print("next: python src/models/p0_ingest_author_review.py --docs-dir docs/rebuild/p0/author_review/ingest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
