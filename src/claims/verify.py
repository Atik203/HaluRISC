"""T2: per-claim NLI verification against evidence.

For every claim the NLI cross-encoder scores (claim, evidence_sentence) pairs
for every evidence sentence; the verdict uses the strongest sentence signal:

  supported      entailment  >= ENTAIL_THRESHOLD and beats contradiction
  contradicted   contradiction >= CONTRA_THRESHOLD
  unsupported    neither (neutral / no supporting evidence)

Evidence sentence = the one that drove the verdict (surfaced in the UI).
Verdict thresholds are documented constants (roadmap B7.5 T2) and are
evaluated against human labels by src/claims/eval_claims.py — they are
decision rules, not pass/fail quality gates.

The NLI model is the sentence-transformers CrossEncoder already loaded by the
API (predict(pairs, batch_size=..., apply_softmax=True) -> probs in order
[contradiction, entailment, neutral]).
"""

import json
from pathlib import Path

import numpy as np

from src.claims.decompose import split_sentences

LABELS = ["contradiction", "entailment", "neutral"]

ENTAIL_THRESHOLD = 0.5
CONTRA_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 64

THRESHOLDS_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "verdict_thresholds.json"


def _effective_thresholds() -> tuple:
    """Thresholds may be tuned from feedback (T4.2 tune_thresholds.py --apply)."""
    try:
        d = json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))
        return float(d["entail"]), float(d["contra"])
    except Exception:
        return ENTAIL_THRESHOLD, CONTRA_THRESHOLD


def _score_claim_blocks(probs: np.ndarray, n_sents: int):
    """Per-claim verdict decision from an (n_claims * n_sents, 3) prob block."""
    ent_thr, con_thr = _effective_thresholds()
    out = []
    for ci in range(len(probs) // n_sents if n_sents else 0):
        block = probs[ci * n_sents : (ci + 1) * n_sents]
        contra = block[:, 0]
        entail = block[:, 1]
        best_ent = int(np.argmax(entail))
        best_con = int(np.argmax(contra))
        ent_prob = float(entail[best_ent])
        con_prob = float(contra[best_con])
        if ent_prob >= ent_thr and ent_prob >= con_prob:
            out.append(("supported", ent_prob, best_ent))
        elif con_prob >= con_thr:
            out.append(("contradicted", con_prob, best_con))
        else:
            out.append(("unsupported", max(ent_prob, con_prob), best_ent))
    return out


def _aggregate(out_claims: list) -> dict:
    counts = {"supported": 0, "contradicted": 0, "unsupported": 0}
    for c in out_claims:
        counts[c["verdict"]] += 1
    overall = "supported"
    if counts["contradicted"] > 0:
        overall = "contradicted"
    elif counts["unsupported"] > 0:
        overall = "unsupported"
    ent_thr, con_thr = _effective_thresholds()
    return {
        "n": len(out_claims),
        "supported": counts["supported"],
        "contradicted": counts["contradicted"],
        "unsupported": counts["unsupported"],
        "overall": overall,
        "rule": f"entailment>={ent_thr} beats contradiction | contradiction>={con_thr} | else unsupported",
    }


def verify_claims(claims: list, evidence: str, nli_model, batch_size: int = DEFAULT_BATCH_SIZE) -> dict:
    """Returns {claims: [{id, text, verdict, confidence, evidence_sentence}],
    aggregate: {n, supported, contradicted, unsupported, overall}}."""
    ev_sents = split_sentences(evidence) or ([evidence] if evidence.strip() else [])
    claims = [c for c in claims if c.strip()]

    out_claims = []
    if not ev_sents:
        # No evidence to check against: every claim is unsupported (no basis).
        for ci, claim in enumerate(claims):
            out_claims.append({
                "id": ci, "text": claim, "verdict": "unsupported", "confidence": 0.0,
                "evidence_sentence": "",
            })
    else:
        # CrossEncoder NLI expects (premise, hypothesis): premise = evidence,
        # hypothesis = claim.
        pairs = [(s, c) for c in claims for s in ev_sents]
        probs = np.asarray(nli_model.predict(pairs, batch_size=batch_size, apply_softmax=True))
        # probs[i, :] = [contradiction, entailment, neutral] for pairs[i]
        n_sents = len(ev_sents)
        for ci, (verdict, confidence, sent_idx) in enumerate(_score_claim_blocks(probs, n_sents)):
            out_claims.append({
                "id": ci,
                "text": claims[ci],
                "verdict": verdict,
                "confidence": round(confidence, 4),
                "evidence_sentence": ev_sents[sent_idx],
            })

    return {"claims": out_claims, "aggregate": _aggregate(out_claims)}


def verify_claims_against_passages(claims: list, passages_by_claim: list, nli_model,
                                   batch_size: int = DEFAULT_BATCH_SIZE) -> dict:
    """Per-claim verification against retrieved passages (Tier 3).

    passages_by_claim[ci] = list of {id, source, url, text} for claims[ci].
    Each claim is scored only against ITS OWN retrieved passages; the verdict
    records the driving passage as evidence (source + url = citation). Claims
    without passages abstain (unsupported, abstained=True).
    """
    claims = [c for c in claims if c.strip()]
    out_claims = []
    if claims:
        pairs = []
        block_sizes = []
        for ci, claim in enumerate(claims):
            ps = passages_by_claim[ci] if ci < len(passages_by_claim) else []
            block_sizes.append(len(ps))
            for p in ps:
                # (premise, hypothesis) = (evidence passage, claim)
                pairs.append((p.get("text", ""), claim))
        if pairs:
            probs = np.asarray(nli_model.predict(pairs, batch_size=batch_size, apply_softmax=True))
        else:
            probs = np.empty((0, 3))

        cursor = 0
        for ci, claim in enumerate(claims):
            n = block_sizes[ci]
            ps = passages_by_claim[ci] if ci < len(passages_by_claim) else []
            if n == 0:
                out_claims.append({
                    "id": ci, "text": claim, "verdict": "unsupported", "confidence": 0.0,
                    "evidence_sentence": "", "evidence_source": "", "evidence_url": "",
                    "abstained": True,
                })
                continue
            verdict, confidence, idx = _score_claim_blocks(probs[cursor : cursor + n], n)[0]
            cursor += n
            passage = ps[idx]
            out_claims.append({
                "id": ci, "text": claim, "verdict": verdict,
                "confidence": round(confidence, 4),
                "evidence_sentence": passage.get("text", ""),
                "evidence_source": passage.get("source", ""),
                "evidence_url": passage.get("url", ""),
                "abstained": False,
            })

    agg = _aggregate(out_claims)
    agg["abstained"] = sum(1 for c in out_claims if c.get("abstained"))
    return {"claims": out_claims, "aggregate": agg}
