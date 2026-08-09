"""API contract tests (health, validation, artifact gating). No model loading."""

import os
import sys
from pathlib import Path

import numpy as np
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
    assert body["model"] == "b2-xgboost-v1.0"
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


def test_meta_contract(client, monkeypatch):
    """B7 additive /meta endpoint: thresholds, warning, groups, versions."""
    monkeypatch.setattr(api, "STATE", {"model": "m", "explainer": None, "feature_models": None,
                                       "feature_cols": ["a", "b"], "params": None})
    r = client.get("/meta")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "b2-xgboost-v1.0"
    assert body["feature_version"] == "course-v1.0"
    assert body["n_features"] == 2
    assert body["thresholds"] == {"low": 0.30, "medium": 0.70, "high": 1.0}
    assert "warning" in body and "device" in body
    assert body["features_available"] is True
    assert body["feature_groups"] is not None and "length" in body["feature_groups"]


def test_analyze_combined_contract(client, monkeypatch):
    """B7.5 Tier 1: /analyze returns prediction + explanation in one call."""
    class FakeExplainer:
        expected_value = 0.5

        def shap_values(self, X):
            return np.array([[0.3, -0.1]])

    monkeypatch.setattr(api, "STATE", {
        "model": {"raw": object(), "predict_proba": lambda X: np.array([[0.38, 0.62]])},
        "explainer": FakeExplainer(),
        "feature_models": object(),
        "feature_cols": ["a", "b"],
        "params": {},
    })
    monkeypatch.setattr(api, "_feature_vector", lambda req: {"a": 1.0, "b": 0.0})
    r = client.post("/analyze", json={"question": "q", "context": "c", "answer": "a"})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"]["calibrated_score"] == 0.62
    assert body["prediction"]["features"]["a"] == 1.0
    assert body["explanation"]["top_features"][0]["feature"] == "a"
    assert body["explanation"]["base_value"] == 0.5


def test_verify_claims_endpoint(client, monkeypatch):
    """B7.5 Tier 2: /verify returns per-claim verdicts + prediction."""
    class FakeExplainer:
        expected_value = 0.5

        def shap_values(self, X):
            return np.array([[0.3, -0.1]])

    class FakeNli:
        def predict(self, pairs, batch_size=64, apply_softmax=True):
            return np.tile([0.05, 0.9, 0.05], (len(pairs), 1))  # strong entailment

    monkeypatch.setattr(api, "STATE", {
        "model": {"raw": object(), "predict_proba": lambda X: np.array([[0.38, 0.62]])},
        "explainer": FakeExplainer(),
        "feature_models": {"nli": FakeNli()},
        "feature_cols": ["a", "b"],
        "params": {},
    })
    monkeypatch.setattr(api, "_feature_vector", lambda req: {"a": 1.0, "b": 0.0})
    r = client.post("/verify", json={
        "question": "q",
        "context": "Paris is the capital of France.",
        "answer": "Paris is the capital of France. It has two million people.",
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["claims"]) >= 2
    assert body["claims"][0]["verdict"] == "supported"
    assert body["aggregate"]["overall"] in ("supported", "contradicted", "unsupported")
    assert body["prediction"]["calibrated_score"] == 0.62
    assert body["explanation"]["top_features"][0]["feature"] == "a"


def test_verify_400_on_empty_claims(client, monkeypatch):
    monkeypatch.setattr(api, "STATE", {
        "model": {"raw": object(), "predict_proba": lambda X: np.array([[0.5, 0.5]])},
        "explainer": None, "feature_models": {"nli": object()},
        "feature_cols": ["a"], "params": {},
    })
    r = client.post("/verify", json={"question": "q", "context": "c", "answer": "   "})
    assert r.status_code == 400


class FakeIndex:
    """Stub RetrievalIndex for T3 endpoint tests."""

    def __init__(self, passages=None):
        self._passages = passages or []

    def status(self):
        return {"n_passages": len(self._passages), "n_documents": 1, "dim": 4,
                "index_dir": "/tmp/fake"}

    def search(self, query, top_k=5, rerank_fn=None):
        return self._passages[:top_k]

    def add_documents(self, documents):
        return [{"source": d["source"], "chunks": 1, "added": 1} for d in documents]

    def clear(self):
        self._passages = []


class _FakeNli:
    def predict(self, pairs, batch_size=64, apply_softmax=True):
        return np.tile([0.05, 0.9, 0.05], (len(pairs), 1))


def _state_with_nli():
    return {
        "model": {"raw": object(), "predict_proba": lambda X: np.array([[0.38, 0.62]])},
        "explainer": None, "feature_models": {"nli": _FakeNli()},
        "feature_cols": ["a", "b"], "params": {},
    }


def test_index_endpoints(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "get_retrieval_index", lambda: FakeIndex())
    monkeypatch.setattr(api, "_embed_texts", lambda texts: np.zeros((len(texts), 4), dtype="float32"))

    r = client.get("/index")
    assert r.status_code == 200 and r.json()["n_passages"] == 0

    r = client.post("/index", files=[("files", ("note.txt", b"Paris is the capital of France.", "text/plain"))])
    assert r.status_code == 200
    assert r.json()["indexed"][0]["added"] == 1

    r = client.delete("/index")
    assert r.status_code == 200 and r.json()["cleared"] is True


def test_retrieve_endpoint(client, monkeypatch):
    monkeypatch.setattr(api, "get_retrieval_index", lambda: FakeIndex([{
        "id": "p1", "source": "doc:note.txt", "url": "", "text": "Paris is the capital of France.", "score": 0.9,
    }]))
    monkeypatch.setattr(api, "_maybe_rerank", lambda q, c, k: c[:k])
    r = client.post("/retrieve", json={"query": "capital of France", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["evidence_mode"] == "index"
    assert body["passages"][0]["source"] == "doc:note.txt"


def test_verify_evidence_mode_index_with_citations(client, monkeypatch):
    monkeypatch.setattr(api, "STATE", _state_with_nli())
    monkeypatch.setattr(api, "_feature_vector", lambda req: {"a": 1.0, "b": 0.0})
    monkeypatch.setattr(api, "get_retrieval_index", lambda: FakeIndex([{
        "id": "p1", "source": "web:https://example.com/france", "url": "https://example.com/france",
        "text": "Paris is the capital of France.", "score": 0.9,
    }]))
    monkeypatch.setattr(api, "_maybe_rerank", lambda q, c, k: c[:k])

    r = client.post("/verify", json={
        "question": "q", "context": "c",
        "answer": "Paris is the capital of France.",
        "evidence_mode": "index",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["aggregate"]["evidence_mode"] == "index"
    assert body["claims"][0]["evidence_source"] == "web:https://example.com/france"
    assert body["claims"][0]["evidence_url"] == "https://example.com/france"
    assert body["claims"][0]["abstained"] is False


def test_verify_abstains_in_index_mode_when_empty(client, monkeypatch):
    monkeypatch.setattr(api, "STATE", _state_with_nli())
    monkeypatch.setattr(api, "_feature_vector", lambda req: {"a": 1.0, "b": 0.0})
    monkeypatch.setattr(api, "get_retrieval_index", lambda: FakeIndex([]))
    r = client.post("/verify", json={
        "question": "q", "context": "c",
        "answer": "Paris is the capital of France.",
        "evidence_mode": "index",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["claims"][0]["abstained"] is True
    assert body["aggregate"]["abstained"] == 1


def test_verify_auto_falls_back_to_context_without_index_or_web(client, monkeypatch):
    monkeypatch.setattr(api, "STATE", _state_with_nli())
    monkeypatch.setattr(api, "_feature_vector", lambda req: {"a": 1.0, "b": 0.0})
    monkeypatch.setattr(api, "get_retrieval_index", lambda: FakeIndex([]))
    monkeypatch.setattr(api, "get_web_search", lambda: type("W", (), {"enabled": False, "search": lambda q: []})())
    r = client.post("/verify", json={
        "question": "q", "context": "Paris is the capital of France.",
        "answer": "Paris is the capital of France.",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["aggregate"]["evidence_mode"] == "context"
    assert body["claims"][0]["evidence_source"] == "context"


def test_analyze_degrades_when_explainer_missing(client, monkeypatch):
    """/analyze still returns the prediction when the explainer is unavailable."""
    monkeypatch.setattr(api, "STATE", {
        "model": {"raw": object(), "predict_proba": lambda X: np.array([[0.6, 0.4]])},
        "explainer": None,
        "feature_models": object(),
        "feature_cols": ["a", "b"],
        "params": {},
    })
    monkeypatch.setattr(api, "_feature_vector", lambda req: {"a": 1.0, "b": 0.0})
    r = client.post("/analyze", json={"question": "q", "context": "c", "answer": "a"})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"]["calibrated_score"] == 0.4
    assert body["explanation"] is None
