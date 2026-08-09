"""T2 eval tool tests: pure metrics + CSV writing (no NLI models needed)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.eval_claims import evaluate_sheet, flagged, write_eval_csv  # noqa: E402


def _sheet(tmp_path, rows: list[dict]) -> Path:
    import csv

    cols = ["sample_id", "claim_id", "claim_text", "context", "model_verdict",
            "model_confidence", "human_verdict", "notes"]
    p = tmp_path / "reviewed.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return p


def test_flagged_mapping():
    assert flagged("contradicted") is True
    assert flagged("unsupported") is True
    assert flagged("supported") is False


def test_evaluate_metrics(tmp_path, capsys):
    p = _sheet(tmp_path, [
        {"model_verdict": "contradicted", "human_verdict": "contradicted"},  # TP
        {"model_verdict": "contradicted", "human_verdict": "supported"},     # FP
        {"model_verdict": "supported", "human_verdict": "supported"},        # TN
        {"model_verdict": "supported", "human_verdict": "contradicted"},     # FN
        {"model_verdict": "unsupported", "human_verdict": "unsupported"},    # TP
    ])
    out = evaluate_sheet(p)
    assert out["contingency"] == {"tp": 2, "fp": 1, "tn": 1, "fn": 1}
    assert out["precision"] == pytest.approx(2 / 3)
    assert out["recall"] == pytest.approx(2 / 3)
    assert out["overall_agreement"] == pytest.approx(3 / 5)
    capsys.readouterr()


def test_evaluate_ignores_unreviewed(tmp_path, capsys):
    p = _sheet(tmp_path, [
        {"model_verdict": "contradicted", "human_verdict": ""},
        {"model_verdict": "supported", "human_verdict": "supported"},
    ])
    out = evaluate_sheet(p)
    assert out["contingency"] == {"tp": 0, "fp": 0, "tn": 1, "fn": 0}
    capsys.readouterr()


def test_evaluate_empty_sheet(tmp_path, capsys):
    p = _sheet(tmp_path, [{"model_verdict": "supported", "human_verdict": ""}])
    assert evaluate_sheet(p) == {}
    capsys.readouterr()


def test_write_eval_csv(tmp_path):
    rows = [{"sample_id": "s1", "claim_id": 0, "claim_text": "c", "context": "x",
             "model_verdict": "supported", "model_confidence": 0.9, "human_verdict": "", "notes": ""}]
    out = tmp_path / "eval.csv"
    write_eval_csv(rows, out)
    assert out.exists()
    assert "human_verdict" in out.read_text(encoding="utf-8")
