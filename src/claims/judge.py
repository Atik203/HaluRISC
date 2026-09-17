"""T4: LLM-judge routing for uncertain claims (hybrid verification).

Cost-controlled: only claims whose NLI signal is UNCERTAIN are sent to the
LLM judge together with their retrieved evidence passages:
  - NLI confidence band: max(entail, contra) in [CONF_LOW, CONF_HIGH)
  - abstained claims that had evidence retrieved (evidence exists but NLI
    gave no verdict)
Cap: at most MAX_CLAIMS_PER_ANSWER claims per answer.

Graceful: with no OPENAI_API_KEY (or any failure) claims keep their NLI
verdict and are tagged judged_by="nli".
"""

import json
import os

CONF_LOW = float(os.environ.get("HALU_JUDGE_CONF_LOW", "0.50"))
CONF_HIGH = float(os.environ.get("HALU_JUDGE_CONF_HIGH", "0.75"))
MAX_CLAIMS_PER_ANSWER = int(os.environ.get("HALU_JUDGE_MAX_CLAIMS", "3"))

JUDGE_SYSTEM = (
    "You are a strict factual-verification judge. Given a CLAIM and EVIDENCE "
    "passages, decide whether the claim is supported by the evidence, "
    "contradicted by it, or unsupported (the evidence neither supports nor "
    "contradicts it). Respond with JSON only: "
    '{"verdict": "supported"|"contradicted"|"unsupported", "confidence": 0.0-1.0, '
    '"reasoning": "<one short sentence>"}.'
)


def _confidence(claim: dict) -> float:
    return float(claim.get("confidence") or 0.0)


def should_route(claim: dict, evidence_present: bool) -> bool:
    """Routing rule: uncertain NLI band, or abstained-with-evidence."""
    if claim.get("judged_by") == "llm":
        return False
    if claim.get("abstained"):
        return evidence_present
    conf = _confidence(claim)
    return CONF_LOW <= conf < CONF_HIGH


def judge_claims(claims: list, passages_by_claim: list, judge_call, max_claims: int = MAX_CLAIMS_PER_ANSWER) -> list:
    """Route uncertain claims to the LLM judge; returns the updated list.

    judge_call(claim_text, evidence_texts) -> dict {verdict, confidence, reasoning}
    or None on failure (claim keeps its NLI verdict).
    """
    routed = 0
    for i, claim in enumerate(claims):
        claim.setdefault("judged_by", "nli")
        ps = passages_by_claim[i] if i < len(passages_by_claim) else []
        evidence_present = any((p.get("text") or "").strip() for p in ps)
        if not should_route(claim, evidence_present) or routed >= max_claims:
            continue
        try:
            result = judge_call(claim["text"], [p.get("text", "") for p in ps if p.get("text")])
        except Exception:
            result = None
        if result and result.get("verdict") in ("supported", "contradicted", "unsupported"):
            claim["verdict"] = result["verdict"]
            claim["confidence"] = round(float(result.get("confidence") or _confidence(claim)), 4)
            claim["judged_by"] = "llm"
            claim["judge_reasoning"] = str(result.get("reasoning") or "")[:300]
            routed += 1
    return claims


def openai_judge_call(api_key: str, model: str | None = None) -> callable:
    """Returns a judge_call bound to the OpenAI client (lazy import)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    model_name = model or os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")

    def _call(claim_text: str, evidence_texts: list) -> dict | None:
        evidence = "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(evidence_texts[:5])) or "(no evidence)"
        user = f"CLAIM: {claim_text}\n\nEVIDENCE:\n{evidence}"
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=160,
        )
        content = resp.choices[0].message.content.strip()
        data = json.loads(content[content.find("{") : content.rfind("}") + 1])
        return {
            "verdict": str(data.get("verdict", "")).lower(),
            "confidence": float(data.get("confidence", 0.0)),
            "reasoning": str(data.get("reasoning", "")),
        }

    return _call
