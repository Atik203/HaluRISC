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

import numpy as np

from src.claims.decompose import split_sentences

LABELS = ["contradiction", "entailment", "neutral"]

ENTAIL_THRESHOLD = 0.5
CONTRA_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 64


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
        pairs = [(c, s) for c in claims for s in ev_sents]
        probs = np.asarray(nli_model.predict(pairs, batch_size=batch_size, apply_softmax=True))
        # probs[i, :] = [contradiction, entailment, neutral] for pairs[i]
        n_sents = len(ev_sents)
        for ci, claim in enumerate(claims):
            block = probs[ci * n_sents : (ci + 1) * n_sents]  # (n_sents, 3)
            contra = block[:, 0]
            entail = block[:, 1]
            best_ent = int(np.argmax(entail))
            best_con = int(np.argmax(contra))
            ent_prob = float(entail[best_ent])
            con_prob = float(contra[best_con])
            if ent_prob >= ENTAIL_THRESHOLD and ent_prob >= con_prob:
                verdict, confidence, sent_idx = "supported", ent_prob, best_ent
            elif con_prob >= CONTRA_THRESHOLD:
                verdict, confidence, sent_idx = "contradicted", con_prob, best_con
            else:
                verdict, confidence, sent_idx = "unsupported", max(ent_prob, con_prob), best_ent
            out_claims.append({
                "id": ci,
                "text": claim,
                "verdict": verdict,
                "confidence": round(confidence, 4),
                "evidence_sentence": ev_sents[sent_idx],
            })

    n = len(out_claims)
    counts = {"supported": 0, "contradicted": 0, "unsupported": 0}
    for c in out_claims:
        counts[c["verdict"]] += 1
    overall = "supported"
    if counts["contradicted"] > 0:
        overall = "contradicted"
    elif counts["unsupported"] > 0:
        overall = "unsupported"
    aggregate = {
        "n": n,
        "supported": counts["supported"],
        "contradicted": counts["contradicted"],
        "unsupported": counts["unsupported"],
        "overall": overall,
        "rule": f"entailment>={ENTAIL_THRESHOLD} beats contradiction | contradiction>={CONTRA_THRESHOLD} | else unsupported",
    }
    return {"claims": out_claims, "aggregate": aggregate}
