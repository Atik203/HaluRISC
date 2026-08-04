"""API contract tests (health, validation, artifact gating). No model loading."""

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Keep tests fast: skip eager preload of spaCy/NLI/SBERT models.
os.environ.setdefault("HALU_API_PRELOAD", "0")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import main as api  # noqa: E402


@pytest.fixture
def client():
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert body["model"] == "xgboost-v1.0"
    assert "artifacts_loaded" in body


def test_predict_empty_answer_400(client):
    r = client.post("/predict", json={"question": "q", "context": "c", "answer": "   "})
    assert r.status_code == 400


def test_predict_requires_answer(client):
    r = client.post("/predict", json={"question": "q", "context": "c"})
    assert r.status_code == 422


def test_predict_503_without_artifacts(client, monkeypatch):
    monkeypatch.setattr(api, "STATE", {"model": None, "explainer": None, "feature_models": None, "feature_cols": None, "params": None})
    r = client.post("/predict", json={"question": "q", "context": "c", "answer": "a"})
    assert r.status_code == 503


def test_judge_503_without_key(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = client.post("/judge", json={"question": "q", "context": "c", "answer": "a"})
    assert r.status_code == 503


def test_explain_503_without_artifacts(client, monkeypatch):
    monkeypatch.setattr(api, "STATE", {"model": None, "explainer": None, "feature_models": None, "feature_cols": None, "params": None})
    r = client.post("/explain", json={"question": "q", "context": "c", "answer": "a"})
    assert r.status_code == 503
