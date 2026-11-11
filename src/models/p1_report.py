"""Combine the two P1 calibration reports into one uniquely named file.

Both analyses write a detailed report next to their CSV/JSON outputs.  The
shared ``p1`` directory also receives reports from other P1 workstreams
(retrieval stress, task shift), so this helper deliberately does **not** write
the generic ``REPORT.md`` name.  It concatenates only the two calibration
reports into ``p1_external_baseline_and_conformal_REPORT.md``.  It never invents
numbers; it only stitches the generated files together.
"""

from __future__ import annotations

from pathlib import Path


REPORTS = (
    ("external_baseline_REPORT.md", "P1.3 External Baseline Reimplementation"),
    ("conformal_REPORT.md", "P1.4 Split Conformal and Multicalibration"),
)
DEFAULT_TARGET = "p1_external_baseline_and_conformal_REPORT.md"


def write_combined_report(p1_dir: Path, target_name: str = DEFAULT_TARGET) -> Path:
    """Write the combined calibration report from the individual reports."""
    p1_dir = Path(p1_dir)
    sections = []
    present = []
    for filename, title in REPORTS:
        path = p1_dir / filename
        if path.exists():
            sections.append(path.read_text(encoding="utf-8"))
            present.append(title)
    header = (
        "# P1 - External Validity and Calibration Analyses\n\n"
        "This file combines the two P1 analysis reports. Each section is also\n"
        "available as a standalone file with the same content. All numbers come\n"
        "from the CSV and JSON artifacts in this directory.\n\n"
        f"Included: {', '.join(present) if present else 'none yet'}.\n"
    )
    body = "\n\n---\n\n".join(sections)
    target = p1_dir / target_name
    target.write_text(header + "\n\n" + body + "\n", encoding="utf-8")
    return target
