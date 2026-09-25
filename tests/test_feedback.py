"""T4: feedback + threshold-tuning tests (no NLI models; stub where needed)."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HALU_API_PRELOAD", "0")

from src.api import main as api  # noqa: E402


@pytest.fixture
def client():
    with TestClient(api.app) as c:
        yield c


def test_feedback_endpoint_appends_jsonl(client, monkeypatch, tmp_path):
    from src.api import main as api

    log = tmp_path / "feedback_log.jsonl"
    monkeypatch.setattr(api, "FEEDBACK_LOG", log)
    r = client.post("/feedback", json={
        "question": "q", "context": "c", "answer": "a",
        "claim_text": "Paris is the capital of France.",
        "verdict": "supported", "evidence_sentence": "Paris is the capital of France.",
        "feedback": "disagree", "note": "actually contradicted",
    })
    assert r.status_code == 200
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["feedback"] == "disagree"
    assert rows[0]["model_version"] == api._model_version()
    assert "ts" in rows[0]


def test_feedback_endpoint_validates_feedback(client):
    r = client.post("/feedback", json={"feedback": "maybe"})
    assert r.status_code == 422


# ---------------------------------------------------------------- tuning
def _fake_nli(probs_by_pair):
    """Deterministic NLI stub keyed by evidence text -> [contra, entail, neutral]."""

    class N:
        def predict(self, pairs, batch_size=1, apply_softmax=True):
            return np.array([probs_by_pair[p[0]] for p in pairs])

    return N()


def test_tune_finds_better_thresholds(tmp_path, monkeypatch):
    import src.models.tune_thresholds as tt

    rows = [{"claim_text": f"claim {i}", "evidence_sentence": f"ev {i}",
             "correct_verdict": "contradicted"} for i in range(10)]

    # contra 0.9 -> 'contradicted' at BOTH grid thresholds (0.5 and 0.7):
    # the tuned best must agree perfectly
    probs = {f"ev {i}": [0.9, 0.05, 0.05] for i in range(10)}
    monkeypatch.setattr(tt, "GRID", np.array([0.5, 0.7]))
    result = tt.tune(rows, _fake_nli(probs))
    assert result["n_labeled"] == 10
    assert result["best"]["agreement"] == 1.0
    assert result["best"]["entail"] in (0.5, 0.7)


def test_tune_reports_unresolved_rows(tmp_path):
    import src.models.tune_thresholds as tt

    rows = [{"claim_text": "c1", "correct_verdict": "supported"},
            {"claim_text": "c2"}]  # no correct_verdict
    assert tt.labeled_rows(rows) == [rows[0]]


def test_tune_apply_writes_thresholds(tmp_path, monkeypatch):
    import src.models.tune_thresholds as tt

    out = tmp_path / "verdict_thresholds.json"
    monkeypatch.setattr(tt, "THRESHOLDS_FILE", out)
    monkeypatch.setattr(tt, "GRID", np.array([0.5, 0.75]))
    rows = [{"claim_text": "c", "evidence_sentence": "e", "correct_verdict": "supported"}]
    probs = {"e": [0.1, 0.9, 0.0]}  # entail 0.9 -> supported at any threshold

    import src.models.tune_thresholds as m
    nli = _fake_nli(probs)
    result = m.tune(rows, nli)
    out.write_text(json.dumps({"entail": result["best"]["entail"], "contra": result["best"]["contra"]}))
    d = json.loads(out.read_text())
    assert d["entail"] in (0.5, 0.75)


def test_verify_reads_tuned_thresholds(tmp_path, monkeypatch):
    """_effective_thresholds reads verdict_thresholds.json when present."""
    from src.claims import verify as v

    f = tmp_path / "verdict_thresholds.json"
    f.write_text(json.dumps({"entail": 0.8, "contra": 0.7}))
    monkeypatch.setattr(v, "THRESHOLDS_FILE", f)
    assert v._effective_thresholds() == (0.8, 0.7)

    monkeypatch.setattr(v, "THRESHOLDS_FILE", tmp_path / "missing.json")
    assert v._effective_thresholds() == (v.ENTAIL_THRESHOLD, v.CONTRA_THRESHOLD)


def test_rate_limit_returns_429():
    """T4: slowapi wiring returns 429 beyond the per-IP window."""
    from fastapi import FastAPI, Request
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address, default_limits=[])
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.get("/t")
    @limiter.limit("1/minute")
    def t(request: Request):
        return {"ok": True}

    with TestClient(app) as c:
        assert c.get("/t").status_code == 200
        assert c.get("/t").status_code == 429
