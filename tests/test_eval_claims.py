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


def test_export_feedback_csv(tmp_path):
    from src.models.eval_claims import export_feedback_csv

    rows = [
        {"claim_text": "Paris is the capital.", "verdict": "supported", "feedback": "agree", "context": "c"},
        {"claim_text": "", "verdict": "supported", "feedback": "agree"},  # skipped (no claim)
    ]
    out = tmp_path / "fb.csv"
    n = export_feedback_csv(rows, out)
    assert n == 1
    import csv

    with open(out, encoding="utf-8", newline="") as f:
        parsed = list(csv.DictReader(f))
    assert parsed[0]["claim_text"] == "Paris is the capital."
    assert "feedback=agree" in parsed[0]["notes"]


class _StubIndex:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, top_k=5):
        return self._hits[:top_k]


def test_citation_recall_at_k():
    from src.models.eval_claims import citation_recall

    passages = [
        {"id": "gold-1", "source": "doc:a", "url": "", "text": "t", "score": 1.0},
        {"id": "x", "source": "doc:b", "url": "", "text": "t", "score": 0.5},
        {"id": "y", "source": "doc:c", "url": "", "text": "t", "score": 0.2},
    ]
    queries = [
        {"query": "q1", "gold_source": "gold-1"},   # top-1 -> recall@1
        {"query": "q2", "gold_source": "doc:c"},    # 3rd -> recall@3 only
        {"query": "q3", "gold_source": "missing"},  # never
    ]
    report = citation_recall(queries, _StubIndex(passages))
    assert report["n"] == 3
    assert report["recall_at_k"][1] == 1 / 3
    assert report["recall_at_k"][3] == 2 / 3
    assert report["recall_at_k"][5] == 2 / 3
