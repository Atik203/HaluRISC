"""T2: per-claim NLI verification against evidence.

For every claim the NLI cross-encoder scores (claim, evidence_sentence) pairs
for every evidence sentence. Verdicts are decided per pair and then aggregated
with support priority:

  per pair     entailment >= ENTAIL_THRESHOLD and entailment >= contradiction
               -> support candidate
               contradiction >= CONTRA_THRESHOLD and contradiction > entailment
               -> contradiction candidate
  per claim    any support candidate       -> supported
               else any contradiction      -> contradicted
               else                        -> unsupported

The support-first aggregation matters because a compound or mismatched claim
can draw a high contradiction score from an unrelated sentence while another
sentence entails it with high confidence. Comparing the maxima across
different sentences used to turn such claims into false contradictions.

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


def score_block(entail: np.ndarray, contra: np.ndarray) -> tuple:
    """One claim's verdict from its per-sentence entailment/contradiction arrays.

    Support priority: a claim is grounded when any evidence sentence entails
    it (entailment >= threshold and entailment >= that pair's contradiction).
    A contradiction only decides when no sentence supports the claim.
    Returns (verdict, confidence, best_sentence_index).
    """
    ent_thr, con_thr = _effective_thresholds()
    support = (entail >= ent_thr) & (entail >= contra)
    contradict = (contra >= con_thr) & (contra > entail)
    if support.any():
        best = int(np.argmax(np.where(support, entail, -1.0)))
        return "supported", float(entail[best]), best
    if contradict.any():
        best = int(np.argmax(np.where(contradict, contra, -1.0)))
        return "contradicted", float(contra[best]), best
    best = int(np.argmax(np.maximum(entail, contra)))
    return "unsupported", float(max(entail[best], contra[best])), best


def _score_claim_blocks(probs: np.ndarray, n_sents: int):
    """Per-claim verdict decision from an (n_claims * n_sents, 3) prob block."""
    out = []
    for ci in range(len(probs) // n_sents if n_sents else 0):
        block = probs[ci * n_sents : (ci + 1) * n_sents]
        out.append(score_block(block[:, 1], block[:, 0]))
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
        "rule": (f"supported if any sentence entails (>={ent_thr} and >= its contradiction) | "
                 f"contradicted if none supports and a sentence contradicts (>{con_thr} and > its entailment) | else unsupported"),
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

    SENTENCE-LEVEL: each passage is split into sentences and every
    (evidence_sentence, claim) pair is scored with the same support-first rule
    as verify_claims (any entailing sentence wins, contradiction only decides
    when no sentence supports). The driving sentence becomes the evidence or
    quote, so long passages cannot drown a supporting sentence.
    Claims without passages abstain (unsupported, abstained=True).
    """
    claims = [c for c in claims if c.strip()]
    out_claims = []
    if claims:
        pairs = []           # (sentence, claim)
        block_sizes = []     # sentences per claim
        sent_passage = []    # per (claim, sentence): passage index
        for ci, claim in enumerate(claims):
            ps = passages_by_claim[ci] if ci < len(passages_by_claim) else []
            n = 0
            for pi, p in enumerate(ps):
                text = p.get("text", "")
                sents = split_sentences(text) or ([text] if text.strip() else [])
                for s in sents:
                    pairs.append((s, claim))
                    sent_passage.append((ci, pi))
                    n += 1
            block_sizes.append(n)

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
            block = probs[cursor : cursor + n]
            cursor += n
            verdict, confidence, sent_idx = score_block(block[:, 1], block[:, 0])

            _, pi = sent_passage[cursor - n + sent_idx]
            passage = ps[pi]
            entry = {
                "id": ci, "text": claim, "verdict": verdict,
                "confidence": round(confidence, 4),
                "evidence_sentence": pairs[cursor - n + sent_idx][0],
                "evidence_source": passage.get("source", ""),
                "evidence_url": passage.get("url", ""),
                "abstained": False,
            }
            if verdict == "contradicted":
                entry["evidence_quote"] = entry["evidence_sentence"][:300]
            out_claims.append(entry)

    agg = _aggregate(out_claims)
    agg["abstained"] = sum(1 for c in out_claims if c.get("abstained"))
    return {"claims": out_claims, "aggregate": agg}
