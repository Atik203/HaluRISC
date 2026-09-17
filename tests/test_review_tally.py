"""B5.5 review-tally tests: descriptive agreement stats over a synthetic sheet."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.review_tally import main  # noqa: E402


def _sheet(tmp_path, rows: list[dict]) -> Path:
    import csv

    cols = ["sample_id", "question", "context", "answer", "label", "raw_score",
            "calibrated_score", "top5_shap_features", "reviewer_1", "reviewer_2", "agreement"]
    p = tmp_path / "reviewed.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return p


def test_tally_prints_agreement_stats(tmp_path, capsys):
    sheet = _sheet(tmp_path, [
        {"sample_id": "a", "reviewer_1": "plausible", "reviewer_2": "plausible", "agreement": "yes"},
        {"sample_id": "b", "reviewer_1": "plausible", "reviewer_2": "implausible", "agreement": "no"},
        {"sample_id": "c", "reviewer_1": "unsure", "reviewer_2": "unsure", "agreement": "yes"},
    ])
    assert main([str(sheet)]) == 0
    out = capsys.readouterr().out
    assert "Cases in sheet:        3" in out
    assert "Agreement=yes:         2" in out
    assert "Agreement=no:          1" in out
    assert "a" in out  # disagreement list mentions sample b


def test_tally_missing_sheet_returns_1(tmp_path, capsys):
    assert main([str(tmp_path / "nope.csv")]) == 1


def test_tally_flags_contradictory_agreement(tmp_path, capsys):
    sheet = _sheet(tmp_path, [
        {"sample_id": "x", "reviewer_1": "plausible", "reviewer_2": "implausible", "agreement": "yes"},
    ])
    assert main([str(sheet)]) == 0
    assert "contradicts reviewer marks" in capsys.readouterr().out
