"""T4: LLM-judge routing tests (mocked judge call, no network)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.claims.judge import should_route, judge_claims  # noqa: E402


def _claim(conf=0.9, abstained=False, verdict="supported"):
    return {"id": 0, "text": "claim", "verdict": verdict, "confidence": conf,
            "evidence_sentence": "e", "abstained": abstained}


def test_route_uncertain_band_only():
    assert should_route(_claim(conf=0.60), evidence_present=True) is True   # in band
    assert should_route(_claim(conf=0.90), evidence_present=True) is False  # confident
    assert should_route(_claim(conf=0.30), evidence_present=True) is False  # below band
    assert should_route(_claim(conf=0.74), evidence_present=True) is True   # band upper
    assert should_route(_claim(conf=0.75), evidence_present=True) is False  # band exclusive top


def test_route_abstained_only_with_evidence():
    c = _claim(conf=0.0, abstained=True)
    assert should_route(c, evidence_present=True) is True
    assert should_route(c, evidence_present=False) is False


def test_judge_claims_updates_uncertain_claims():
    claims = [_claim(conf=0.60), _claim(conf=0.95), _claim(conf=0.65)]
    passages = [[{"text": "evidence A"}], [{"text": "evidence B"}], [{"text": "evidence C"}]]

    def judge_call(text, ev):
        return {"verdict": "contradicted", "confidence": 0.9, "reasoning": "contradicts evidence"}

    out = judge_claims(claims, passages, judge_call)
    assert out[0]["verdict"] == "contradicted" and out[0]["judged_by"] == "llm"
    assert out[1]["verdict"] == "supported" and out[1]["judged_by"] == "nli"
    assert out[2]["verdict"] == "contradicted" and out[2]["judged_by"] == "llm"
    assert out[0]["judge_reasoning"] == "contradicts evidence"


def test_judge_claims_caps_per_answer():
    claims = [_claim(conf=0.60) for _ in range(5)]
    passages = [[{"text": "e"}]] * 5
    calls = {"n": 0}

    def judge_call(text, ev):
        calls["n"] += 1
        return {"verdict": "unsupported", "confidence": 0.8, "reasoning": "r"}

    out = judge_claims(claims, passages, judge_call, max_claims=3)
    assert calls["n"] == 3
    assert sum(1 for c in out if c["judged_by"] == "llm") == 3


def test_judge_failure_keeps_nli_verdict():
    claim = _claim(conf=0.60)

    def judge_call(text, ev):
        raise RuntimeError("offline")

    out = judge_claims([claim], [[{"text": "e"}]], judge_call)
    assert out[0]["verdict"] == "supported"
    assert out[0]["judged_by"] == "nli"


def test_judge_invalid_verdict_ignored():
    claim = _claim(conf=0.60)

    def judge_call(text, ev):
        return {"verdict": "maybe", "confidence": 0.9, "reasoning": "x"}

    out = judge_claims([claim], [[{"text": "e"}]], judge_call)
    assert out[0]["verdict"] == "supported"
    assert out[0]["judged_by"] == "nli"
