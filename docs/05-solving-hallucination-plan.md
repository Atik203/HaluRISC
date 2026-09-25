# 05 - "How would you solve hallucination?" - the plan

A faculty member may ask: "You are doing risk prediction. If you could actually
fix the problem, how would you do it, and what would you use?"

This page is the prepared answer. The short version: prediction alone does not
fix hallucination, so the plan turns the risk model into the routing and
verification layer of an evidence-first pipeline: retrieve evidence, generate
with citations, check every claim, repair or abstain, and learn from feedback.

Everything below is grounded in what this project already measured, so the plan
does not promise more than the evidence supports.

## Contents

1. [What "solving" means](#1-what-solving-means)
2. [What our own results already tell us](#2-what-our-own-results-already-tell-us)
3. [The plan in four layers](#3-the-plan-in-four-layers)
4. [The routing policy](#4-the-routing-policy)
5. [What can be used (toolbox)](#5-what-can-be-used-toolbox)
6. [How the result would be evaluated](#6-how-the-result-would-be-evaluated)
7. [What is already built vs what is new](#7-what-is-already-built-vs-what-is-new)
8. [What I would not claim](#8-what-i-would-not-claim)
9. [The 60-second spoken answer](#9-the-60-second-spoken-answer)
10. [Likely follow-up questions](#10-likely-follow-up-questions)

---

## 1. What "solving" means

Hallucination has three different problems, and they need different tools.

| Level | Question | Where it lives | Our position |
|---|---|---|---|
| Prevent | Stop the model from making things up | training, decoding, prompting, retrieval | partly available to us through evidence-first generation |
| Detect | Find unsupported claims after generation | verification models, evidence checks | this is HaluRISC |
| Repair | Replace or remove bad content | retrieval, regeneration, editing, routing to a human | the next step built on top of detection |

If a faculty member asks "can you solve hallucination", the honest answer is
that no single component does. A credible solution is a closed loop over all
three levels, and detection is the part this project contributes.

## 2. What our own results already tell us

The plan is shaped by four measured facts, not by intuition.

1. A lightweight verifier works well in-domain: F1 = 0.9846 and AUROC = 0.9979
   on the HaluEval QA test split, at 61.8 ms per analysis (B2, efficiency
   artifacts).
2. Calibration does not transfer for free: the source-calibrated model shows
   ECE = 0.8185 on natural RAGTruth QA, and refitting on target data with
   disjoint groups lowers it to 0.1335 (B4).
3. Zero-shot ranking of natural responses fails: F1 = 0.3022 on RAGTruth QA,
   with an answer-length confound explaining part of the pattern (B3, B5).
4. Claim-level checks with retrieved evidence are the strongest signal the
   deployed system has: per-claim NLI verdicts plus citations lead the chat
   risk card, and the calibrated score becomes a second opinion (B7.5 tiers
   1 to 4).

Facts 2 and 3 are the reason the plan is not "train a better classifier". A
classifier cannot add the missing evidence, so the solution has to bring
evidence to the answer and verify against it.

## 3. The plan in four layers

```
user question
   |
   v
[1] EVIDENCE LAYER            retrieve before generating
    hybrid search, rerank, authoritative sources, freshness
   |
   v
[2] GENERATION LAYER          answer under evidence constraints
    cite-as-you-write prompt, tool calls, temperature control,
    optional self-consistency sampling
   |
   v
[3] VERIFICATION LAYER        this is HaluRISC
    claim decomposition -> per-claim NLI + retrieval check
    -> calibrated risk (low / medium / high) -> SHAP reasons
   |
   v
[4] ACTION LAYER              route, repair, or abstain
    low: deliver with citations
    medium: verify harder or regenerate with the failing claim fixed
    high: abstain and hand to human review
    every verdict logs feedback for retraining and recalibration
```

Layer 3 is the part that exists today (FastAPI `/verify`, `/judge`, `/index`,
`/feedback`, plus the chat risk card and the analyze dashboard). Layer 1 exists
partially: pasted context, uploaded documents, and Brave web search. Layers 2
and 4 are the extension the plan proposes.

## 4. The routing policy

The value of a calibrated score is that it can drive decisions, not just flags.
The cutoffs are the ones the API already reports.

| Risk band | Typical cause | Action | Why |
|---|---|---|---|
| Low (< low cutoff) | claims supported by retrieved evidence | deliver the answer with citations | no extra cost, keep latency at 61.8 ms |
| Medium (between cutoffs) | unsupported claims, thin evidence, style-sensitive score | run the claim-level check, retrieve more, ask the model to rewrite only the failing claims | targeted repair is cheaper than regenerating everything |
| High (>= high cutoff) | contradicted claims or no evidence found | abstain, show what the evidence says, offer a human review path | a wrong confident answer is worse than no answer |

The measured cost asymmetry justifies the routing: the local model costs about
$0.001 per 1,000 predictions and the measured LLM judge about $0.105 per 1,000,
roughly 100 times more (efficiency artifacts). Spending judge calls only on the
medium and high bands keeps quality while staying cheap.

## 5. What can be used (toolbox)

| Layer | Option | What it does | Notes |
|---|---|---|---|
| Evidence | Okapi BM25 (Elasticsearch / OpenSearch) | exact term retrieval for names, numbers, dates | cheap and strong on rare tokens |
| Evidence | Dense retrieval: FAISS, Qdrant, pgvector with E5 or BGE embeddings | semantic match when wording differs | combine with BM25 in a hybrid score |
| Evidence | Cross-encoder reranker (for example BGE-reranker) | reorder top candidates by true relevance | small latency cost, large precision gain |
| Evidence | Knowledge sources: Wikipedia / Wikidata APIs, domain databases, Brave search | authoritative grounding material | our Tier 3 already calls Brave (LLM Context) |
| Generation | Cite-as-you-write prompting and structured outputs | force the model to attach evidence spans | easy to add to the existing `/api/chat` prompt |
| Generation | Tool calling for arithmetic, dates, and lookups | removes a whole class of numeric hallucinations | our chat already defines a tool schema |
| Generation | Self-consistency or self-check sampling (SelfCheckGPT style) | catch unstable answers by disagreement | costs multiple samples, use only on risky bands |
| Verification | DeBERTa-class NLI model (entailment / contradiction) | per-claim evidence check | already used in the feature pipeline and Tier 2 |
| Verification | Small fact-checking models (MiniCheck, AlignScore, HHEM) | cheaper alternative verifiers | trainable on our own labels |
| Verification | FActScore-style atomic claim decomposition | turn a paragraph into checkable units | needed before per-claim checks |
| Verification | HaluRISC XGBoost + calibration (this project) | calibrated risk with SHAP reasons | 26 features, 7 groups, CPU inference |
| Verification | LLM-as-judge with a rubric | handles hard semantic cases | use selectively, it is the expensive tier |
| Action | Guardrail frameworks: Guardrails AI, NeMo Guardrails | enforce policies and fallbacks | wraps the action layer |
| Action | Orchestration: LangGraph or a small state machine in FastAPI | implement verify-retrieve-repair loops | keeps the loop auditable |
| Learning | Feedback store (our `feedback_log.jsonl`) | human agree / disagree labels | already collected from the chat thumbs |
| Learning | Fine-tuning: SFT on grounded answers, DPO with supported vs unsupported pairs | teach the generator to prefer evidence-backed text | needs preference data, which our loop produces |
| Learning | Recalibration on target data (our B4 procedure) | keep probabilities honest after drift | already implemented, rerun when data shifts |
| Evaluation | HaluEval, RAGTruth, FaithBench (used here) plus TruthfulQA, FEVER, ExpertQA | measure detection and factuality | keep an external set for honesty |
| Evaluation | RAGAS or custom scripts for citation recall@k | measure whether citations actually support claims | pairs with our claim verdicts |

## 6. How the result would be evaluated

A repair loop can look good while hiding failures, so the evaluation keeps the
same discipline as this project.

- Detection quality: F1, AUROC, PR-AUC, MCC on grouped splits, plus ECE and
  Brier for the probabilities.
- Verification quality: per-claim precision and recall against human labels,
  and citation recall@k for retrieved evidence.
- Repair quality: does the rewritten answer remove contradicted claims without
  losing correct content, measured on paired before/after examples.
- Safety behaviour: abstention rate on unanswerable questions, and the share of
  high-risk answers that were correctly blocked.
- Cost and latency: per-1,000-request cost and p50 latency, reported per band so
  the routing policy stays justified.
- Human audit: the B5.5 procedure, error-enriched sampling with an explicit
  caveat, repeated on the repaired outputs.

## 7. What is already built vs what is new

Already in the repository:

- calibrated risk model with SHAP explanations (`/predict`, `/explain`)
- claim-level verification with citations (`/verify`, tier 2)
- document index and web retrieval (`/index`, tier 3)
- hybrid LLM judge for hard cases (`/judge`, tier 4)
- feedback capture (`/feedback`, `feedback_log.jsonl`)
- chat UI that shows verdicts, evidence quotes, and the calibrated score
- evaluation tooling: statistical tests, calibration procedure, audit sheets

New work the plan adds:

1. A repair endpoint that takes an answer, its failing claims, and the evidence,
   then asks the model for a minimal rewrite with citations.
2. A routing policy implemented as configuration (band cutoffs already exist in
   the API response).
3. Hybrid retrieval upgrade (BM25 + dense + rerank) behind the existing index
   API, so no UI changes are needed.
4. Preference data collection from the feedback loop, then a DPO or SFT run on
   the generator model.
5. Recalibration and drift monitoring scheduled runs, reusing B4 code.

## 8. What I would not claim

- A detector cannot guarantee truth. It estimates evidence consistency.
- The zero-shot transfer gap (F1 = 0.3022 on RAGTruth QA) shows that a model
  trained on synthetic detection data does not automatically work on natural
  long-form answers. Any solution must bring its own evidence and re-measure.
- Probability quality depends on the target domain. Calibration must be refit
  when the data shifts.
- A repair loop can introduce new errors. That is why the evaluation includes
  before-after human checks.

## 9. The 60-second spoken answer

"I would not try to fix hallucination inside the model, because we cannot see
the model. I would build a loop around it. First, retrieve evidence before
generating, using hybrid search over authoritative sources. Second, make the
generator cite that evidence. Third, verify each claim with the same calibrated
risk model this project already provides, which is fast and explainable. Fourth,
route by the calibrated score: low risk goes straight out with citations,
medium risk gets a targeted repair, high risk abstains and goes to human review.
Every verdict feeds a feedback file that supports recalibration and fine-tuning.
The detection and verification layers already exist in my system. The solution
is the closed loop around them."

## 10. Likely follow-up questions

**Why not just use a bigger model?**
Bigger models hallucinate less but not zero, and they cost 100 times more per
check. Verification stays necessary, and it is cheaper to verify than to
regenerate.

**Why not fine-tune the generator with RLHF?**
That helps and belongs in the plan, but it needs preference data, which the
feedback loop produces. It also does not remove the need for verification,
because fine-tuning changes tendencies, not guarantees.

**What is the single most important addition?**
Evidence retrieval with citations. Our B3 and B4 results show that the model's
weakness outside its training domain is an evidence problem, not a classifier
problem.

**How does this stay cheap?**
The local verifier costs about $0.001 per 1,000 predictions. Expensive judge
calls are reserved for the medium and high bands, which are a minority of
traffic.

**How do you know the repair worked?**
Paired before-after scoring with the same claim-level checks, plus a human
audit using the B5.5 procedure.

**Could this be a student-scale project?**
Yes, and most of it exists already. The new parts are one repair endpoint, a
retrieval upgrade, and a fine-tuning experiment on collected feedback.

## Related documents

- [`01-project-and-methods.md`](01-project-and-methods.md) - the system and its features
- [`02-experiments-and-results.md`](02-experiments-and-results.md) - the numbers cited above
- [`03-reproduction-and-defense.md`](03-reproduction-and-defense.md) - defense Q&A and the demo script
- [`04-interface-guide.md`](04-interface-guide.md) - what the demo screens show
- [`b7-research-ui.md`](b7-research-ui.md) - the tiered verification API this plan builds on
