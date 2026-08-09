"""T2: claim-verification tests (deterministic; no heavy models)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.claims.decompose import split_claims, split_sentences  # noqa: E402
from src.claims.verify import verify_claims  # noqa: E402


class FakeNLI:
    """CrossEncoder-shaped stub: returns fixed probs [contra, entail, neutral]."""

    def __init__(self, probs):
        self._probs = np.asarray(probs, dtype=float)

    def predict(self, pairs, batch_size=64, apply_softmax=True):
        assert apply_softmax is True
        assert len(pairs) == len(self._probs)
        return self._probs


# ---- decompose ----

def test_sentences_split():
    assert split_sentences("First. Second! Third?") == ["First.", "Second!", "Third?"]


def test_claims_basic_sentences():
    claims = split_claims("Paris is the capital of France. It has 2 million people.")
    assert claims == ["Paris is the capital of France.", "It has 2 million people."]


def test_claims_clause_refinement_for_long_sentences():
    long = (
        "The Arctic melt season has lengthened by five days per decade since 1979, "
        "dominated by a later autumn freeze-up across the entire region, "
        "and this trend is expected to continue over the coming decades, "
        "with consequences for coastal communities and shipping routes alike."
    )
    assert len(long) >= 200
    claims = split_claims(long)
    assert len(claims) >= 2
    assert all(len(c) >= 20 for c in claims)


def test_claims_dedup_and_cap():
    text = "Same claim. Same claim. " * 20
    claims = split_claims(text, max_claims=5)
    assert len(claims) <= 5


def test_claims_empty_input():
    assert split_claims("   ") == []


# ---- verify ----

def _ctx():
    return "Paris is the capital of France. The city has 2 million people."


def test_verify_supported_and_contradicted():
    claims = ["Paris is the capital of France.", "The city has 3 million people."]
    # per claim x 2 evidence sentences: [contra, entail, neutral]
    probs = np.array([
        [0.05, 0.92, 0.03],  # claim1 vs "Paris is the capital of France."
        [0.10, 0.40, 0.50],  # claim1 vs "The city has 2 million people."
        [0.85, 0.08, 0.07],  # claim2 vs "Paris is the capital of France."
        [0.70, 0.15, 0.15],  # claim2 vs "The city has 2 million people."
    ])
    out = verify_claims(claims, _ctx(), FakeNLI(probs))
    vs = {c["text"]: c["verdict"] for c in out["claims"]}
    assert vs[claims[0]] == "supported"
    assert vs[claims[1]] == "contradicted"
    assert out["aggregate"]["supported"] == 1
    assert out["aggregate"]["contradicted"] == 1
    assert out["aggregate"]["overall"] == "contradicted"


def test_verify_unsupported_when_neutral():
    claims = ["Elephants live in Antarctica."]
    # two evidence sentences, both neutral
    probs = np.array([
        [0.10, 0.15, 0.75],
        [0.12, 0.13, 0.75],
    ])
    out = verify_claims(claims, _ctx(), FakeNLI(probs))
    assert out["claims"][0]["verdict"] == "unsupported"
    assert out["aggregate"]["overall"] == "unsupported"


def test_verify_evidence_sentence_attribution():
    claims = ["The city has 2 million people."]
    probs = np.array([
        [0.10, 0.30, 0.60],  # vs "Paris is the capital of France."
        [0.05, 0.90, 0.05],  # vs "The city has 2 million people."
    ])
    out = verify_claims(claims, _ctx(), FakeNLI(probs))
    c = out["claims"][0]
    assert c["verdict"] == "supported"
    assert "2 million people" in c["evidence_sentence"]


def test_verify_empty_claims():
    out = verify_claims([], _ctx(), FakeNLI(np.empty((0, 3))))
    assert out["claims"] == []
    assert out["aggregate"]["n"] == 0


def test_verify_empty_evidence():
    claims = ["Anything."]
    probs = np.array([[0.05, 0.10, 0.85]])
    out = verify_claims(claims, "", FakeNLI(probs))
    assert out["claims"][0]["verdict"] == "unsupported"
