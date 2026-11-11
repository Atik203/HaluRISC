# Strengthening Plan P0 / P1 / P2

Status: **proposed**. Nothing in this plan has been executed. Recommended path: **P0 + P1**. P2 is deferred as a separate benchmark/resource paper; it is not part of the current submission. Start only after the author confirms.

Target: take the `rebuild` paper from ~88/100 to ~90 (P0), ~92–94 (P1), or a separate benchmark paper (~94+) for P2.

Working rule: `main` stays frozen. All work lands on `rebuild` or a child branch, never merged without approval.

---

## 1. Where the paper stands

| Item | Current |
|---|---|
| Branch | `rebuild`, commit `167bd32` |
| Paper | 25 pages, 11 figures, 16 tables, 40 references |
| Build | 0 overfull/underfull, 0 undefined refs, 27/27 floats within two pages |
| Audits | 298/298 original + 116/116 rebuild |
| Main weakness | evidence is concentrated in HaluEval; audit is still AI-assisted; no retrieval/end-to-end; judges beat EC-XGB on held-out QA |
| Estimated score | **88/100** |
| Mid Q1 transaction odds | **50–60%** |

P0 closes the remaining reviewer-visible protocol weaknesses. P1 adds external validity. P2 is a new benchmark/resource paper.

---

## 2. Integrity rules for all packages

- AI may pre-label and draft, but the final label is the author's.
- The paper says **"three-author blinded review with AI pre-annotation"**, never "independent human annotation."
- Every sheet records `ai_suggested`, `author_label`, `final_label`, `annotator_id`, `changed`, `review_status`, `notes`.
- The 50–60 row overlap set is labeled blind: no AI suggestion, no other author's label, no gold label.
- Do not tune any model or threshold on test rows.
- Every new number must be written to `artifacts/results/rebuild/p0|p1|p2/` and added to the audit.
- Record AI model, prompt version, codebook version, and date in `docs/rebuild/ai_use_ledger.md`.
- No merge to `main` until the final reviewer pass and author sign-off.

---

## 3. P0 — close the audit and protocol gaps

Goal: make the existing paper defensible at a mid Q1 transaction. No new dataset.

### P0.1 Annotation package (3 authors + agents)

**Samples**

| Sheet | Rows | Contents |
|---|---:|---|
| `p0_error_taxonomy.csv` | 50 | 25 m3 false positives, 25 m3 false negatives; 20-category codebook |
| `p0_explanation_audit.csv` | 100 | 50 error-enriched + 50 length-stratified HaluEval cases; plausible / implausible / unsure |
| `p0_claim_verdicts.csv` | 100 claims | from 50 answers; supported / contradicted / unsupported / unsure |
| `p0_blind_overlap.csv` | 60 | 20 from each sheet; all three authors label blind |

**Procedure**

1. Agents freeze `codebook_v2.md` and pre-label every row with a rationale and evidence span.
2. Author A owns taxonomy, Author B owns explanations, Author C owns claims.
3. Each author labels 60 overlap rows blind, then reviews their own slice with AI suggestions visible.
4. Adjudication meeting: resolve disagreements, update codebook if a rule is ambiguous, freeze final labels.
5. Compute:
   - Fleiss' kappa on the 60-row overlap (3 raters).
   - Binary kappa (content error vs artifact) for taxonomy; 4-class kappa for claims.
   - AI–human agreement and correction rate.
6. Acceptance: 100% rows reviewed, no `needs_author_verification`, kappa reported, correction rate reported. If kappa < 0.4, revise the codebook and repeat the overlap once.

**Outputs**

- `docs/rebuild/p0/*.csv`
- `artifacts/results/rebuild/p0/annotation_agreement.json`
- `docs/rebuild/p0/codebook_v2.md`
- `docs/rebuild/ai_use_ledger.md`

**Effort:** 3 authors × 8–10 h; agents 1 day.

---

### P0.2 Fixed-hp m3 feature-group ablation

**Why:** the current feature-group ablation is B2/tuned. The m3 claim is inferred, not directly tested.

**Design**

- Fixed configuration: depth 5, lr 0.05, 300 trees, subsample 0.7, colsample 0.7.
- Remove one group at a time: length, lexical, entity, nli, numeric, hedging, semantic.
- Keep claim features and source indicator for m3; also run two controls:
  - m3 without claim features (m1 equivalent).
  - m3 without source indicator.
- Seeds 42/123/456.

**Output:** `artifacts/results/rebuild/p0/feature_group_ablation.csv`, `feature_group_ablation.json`.

**Metrics:** F1, AUROC, MCC, ECE, Brier; mean ± SD; two-level CI for each delta if time permits.

**Acceptance:** 27 fits, same config logged, no test tuning, table matches artifact.

**Effort:** ~30 min compute, 2 h code.

---

### P0.3 Judge on the length-matched subset

**Why:** reviewer will ask whether GPT/Qwen also exploit the HaluEval length artifact.

**Design**

- 158-row length-matched set (79 pairs) from `length_control_fixed_hp.json`.
- GPT-5.6 Luna, 3 seeds, same prompt; Qwen 3.5 4B, 1 seed.
- Metrics: F1, AUROC, precision, recall, flag rate; compare with EC-XGB and length-only on the same rows.

**Output:** `artifacts/results/rebuild/p0/judge_length_matched.csv`.

**Acceptance:** if judges drop, report that the benchmark artifact affects them too; do not spin it.

**Effort:** ~$0.15 API, 15 min.

---

### P0.4 Judge prompt sensitivity

**Why:** one prompt is a standard judge critique.

**Design**

- 100 rows × 3 prompts × 3 seeds = 900 GPT calls.
- Prompts: current; strict evidence-only; concise JSON.
- Qwen on the same 100 rows, 1 seed per prompt.
- Report F1/AUROC range and pairwise agreement.

**Output:** `artifacts/results/rebuild/p0/prompt_sensitivity.csv`.

**Acceptance:** prompt text published; range reported; no prompt selected by test performance.

**Effort:** ~$0.25 API, 2 h code.

---

### P0.5 Cross-seed SHAP stability

**Why:** explanations are seed-42 only.

**Design**

- TreeSHAP on 1,000 rows for seeds 42/123/456.
- Report mean |SHAP| rank correlation (Spearman/Kendall) and top-10 overlap.
- Add a small figure or a table row.

**Output:** `artifacts/results/rebuild/p0/shap_seed_stability.csv`.

**Effort:** ~5 min compute, 1 h code.

---

### P0.6 CPU latency

**Why:** deployment claim is currently GPU-only.

**Design**

- 200 rows on CPU, same instrumented path as `rebuild_latency.py`.
- Report base, claim, total p50/p95.
- Update Table 13/14 with a CPU row.

**Output:** `artifacts/results/rebuild/p0/m3_latency_cpu.json`.

**Effort:** ~10 min.

---

### P0.7 Claim-verdict audit

**Why:** the UI headline uses claim verdicts, but verdict accuracy is not evaluated.

**Design**

- 100 claims from 50 answers.
- Labels: supported, contradicted, unsupported, unsure.
- 3 authors, 20-claim overlap for kappa; evidence span required.
- Report accuracy against the author consensus and kappa.

**Output:** `docs/rebuild/p0/p0_claim_verdicts.csv`, `artifacts/results/rebuild/p0/claim_verdict_accuracy.json`.

**Effort:** 2–3 h per author.

---

### P0.8 Paper integration

- Abstract: add three-author blinded review + kappa; length-matched judges; fixed-hp feature ablation. Keep 250–300 words.
- §5.2: replace the B2 feature-group table with the fixed-hp m3 table, or add it and move B2 to the supplement.
- §5.5: add length-matched judge row, prompt-sensitivity range, claim-verdict accuracy.
- §5.6: replace the AI-draft audit paragraph with the three-author protocol, kappa, correction rate; update limitations.
- Tables: add feature-group ablation and claim-verdict table; combine prompt sensitivity into one table or supplement.
- Figures: add SHAP seed-stability panel if it fits; otherwise a table.
- Build: 0 overfull/underfull, floats within two pages, audits updated.

**Acceptance:** every number maps to a P0 artifact; two reviewer passes; no stale AI-only audit language.

---

### P0 schedule (4 days)

| Day | Agents | Authors |
|---|---|---|
| 0 | Freeze codebook; generate sheets; pre-label; build blind overlap | 90-min calibration on 20 rows |
| 1 | Run P0.2, P0.3, P0.5, P0.6 | Blind overlap + primary review (5–6 h) |
| 2 | Run P0.4; compute agreement; draft tables | Finish review; adjudication (2 h) |
| 3 | Integrate P0 results; regenerate figures/tables; build | Check tables and claims |
| 4 | Audit, float check, reviewer pass | Final read and sign-off |

**P0 cost:** API **$0.40–0.60**; compute <$0.10; human **3 × 8–10 h**.

**P0 exit gate:** 90/100, mid Q1 odds **60–70%**.

---

## 4. P1 — external validity and deployment relevance

Goal: answer “one benchmark, no retrieval, no external baseline.”

### P1.1 Second in-domain corpus

**Option A - existing labeled corpus.** Selected: **MedHallu** `pqa_labeled` (MIT, 1,000 human-annotated medical QA pairs with gold knowledge and paired hallucinated answers). Secondary breadth check: **HaluBench** non-overlap subset (DROP, FinanceBench, covidQA; exclude HaluEval, RAGTruth, and pubmedQA to avoid double-counting; CC BY-NC 2.0).

**Option B — 300–500 new labels.** Sample natural QA answers, 3 authors label grounded/hallucinated with evidence; 50-row overlap for kappa.

**Evaluation:** EC-XGB zero-shot, target-calibrated, GPT/Qwen, length-only; F1/AUROC/ECE with CIs.

**Output:** `artifacts/results/rebuild/p1/second_corpus_metrics.csv`.

**Acceptance:** dataset documented, no test tuning, label protocol stated, kappa if new labels.

**Effort:** 3–5 days if existing; 10–15 h per author if new.

---

### P1.2 Retrieval stress test

**Why:** deployment uses gold context; retrieval quality is unevaluated.

**Design**

- 300 questions with qrels (relevant passages), or 150 if time is tight.
- BM25 and FAISS retrieval; recall@k, context coverage.
- Run EC-XGB and the GPT judge with gold vs retrieved context.
- Report verdict accuracy and calibration drop.

**Output:** `artifacts/results/rebuild/p1/retrieval_stress.csv`.

**Acceptance:** qrels documented; retrieval recall reported; no claim of end-to-end quality beyond the tested set.

**Effort:** 15–20 h human; 1 h compute; ~$0.25 API.

---

### P1.3 External baseline reimplementation

**Why:** the external bar is context, not a head-to-head.

**Design**

- Reimplement a published feature-based detector on the same 3,000 HaluEval rows: ROUGE-L, SBERT similarity, BERTScore, DeBERTa NLI, score ensemble.
- Tune the threshold on validation only; bootstrap CIs.
- Cite the original; label as a reimplementation, not the authors’ number.

**Output:** `artifacts/results/rebuild/p1/external_baseline.csv`.

**Effort:** 2–3 days; 1–2 GPU-h.

---

### P1.4 Conformal prediction and multicalibration

**Why:** turn calibration from a negative audit into a constructive contribution.

**Design**

- Split conformal on the 5,034 QA calibration rows; 90% and 95% coverage.
- Report coverage and interval width on QA, FaithBench, and HaluEval.
- Multicalibration by length bin and claim count; ECE per subgroup.
- Compare with Platt and prior-only.

**Output:** `artifacts/results/rebuild/p1/conformal.csv`, `multicalibration.csv`.

**Acceptance:** coverage close to nominal on calibration; shift coverage reported honestly; no test fitting.

**Effort:** 1 day code; 1 CPU-h.

---

### P1.5 Paper integration

- Add second-corpus table and retrieval table; move detailed per-task results to supplement if needed.
- Add external baseline row/table.
- Add conformal coverage figure and multicalibration paragraph.
- Reframe contribution: “audit + practical protocol” with a constructive calibration section.
- Keep 40 references; add only verified citations.

**P1 schedule:** days 5–9 after P0.

**P1 cost:** API **$0.60–1.10**; compute 3–5 GPU-h; human **3 × 12–15 h**.

**P1 exit gate:** 92–94/100, mid Q1 odds **70–80%**.

---

## 5. P2 - deferred benchmark/resource paper

**Decision: deferred.** P2 is a different contribution, not a revision. It should not delay the mid Q1 submission. Start it only after P0+P1 is submitted or accepted, and only if the authors want a benchmark/resource paper.

Goal: a separate paper, not a revision.

### P2.1 HaluEval-LC

- Length-controlled split: 3,000 rows, 1,500 pairs, six length bins.
- Human-verify 500 rows; publish the split and construction script.

### P2.2 RAGTruth-Claim

- 1,000 responses with claim-level labels: supported, contradicted, unsupported.
- Two annotators + adjudication; kappa; datasheet.

### P2.3 Detector zoo

- 12 detectors: heuristic, LR, RF, XGB, NLI-only, TF-IDF, SBERT, RoBERTa, GPT, Qwen, EC-XGB, ensemble.
- Same splits, same metrics, same calibration protocol.

### P2.4 Label-free prevalence calibration

- BBSE, EM, prior-shift correction; evaluate under target shift without target labels.
- Report coverage, ECE, Brier, and failure modes.

### P2.5 Release

- Dataset card, license, leaderboard, code, reproducibility script.

**P2 effort:** 200–400 human hours; API **$3–6**; compute 20–50 GPU-h.

**P2 go/no-go:** deferred. Revisit only after P0+P1 is submitted or accepted, and only if two authors can commit 3-6 weeks. Do not start P2 for a mid Q1 transaction.

---

## 6. Cost summary

| Package | API | Compute/electricity | Human time | Paid annotation at $15–25/h |
|---|---:|---:|---:|---:|
| P0 | $0.40–0.60 | <$0.10 | 3 × 8–10 h | $360–750 |
| P0 + P1 | $1.00–1.70 | $0.30–0.50 | 3 × 20–25 h | $900–1,875 |
| P0 + P1 + P2 | $4–8 | $3–8 laptop / $20–60 cloud | 200–400 h | $3,000–10,000 |

API is not the main cost. Human review is.

---

## 7. Risk register

| Risk | Mitigation |
|---|---|
| Authors anchor on AI labels | 60-row blind overlap; AI suggestions hidden until blind labels are frozen |
| Low kappa | Revise codebook, re-run overlap once, report the final kappa honestly |
| Annotation fatigue | 60–90 min blocks; two passes; adjudication on a separate day |
| Second corpus unavailable | Use the external baseline reimplementation; scope the claim to one benchmark |
| Retrieval qrels too costly | 150 questions instead of 300; report the reduced scope |
| API rate limits | Batch, cache, Qwen local, retry with backoff |
| Time slip | Ship P0 only; P1 becomes the next revision |
| Reviewer still says “single benchmark” | P1.1 + P1.3 are the answer; P2 is the long-term answer |
| Positive QA result still absent | Do not force it; the paper’s contribution is the audit, not a leaderboard win |

---

## 8. Decision gates

1. **After P0 annotation:** if kappa < 0.4, stop and fix the codebook before any paper edits.
2. **After P0 experiments:** if length-matched judges collapse, make that a headline finding, not a footnote.
3. **After P1.1:** if MedHallu and HaluBench disagree, report per-corpus results; do not pool them into a single headline.
4. **Before P2:** require a written go/no-go from all authors and a 3-week minimum commitment.
5. **Before submission:** two reviewer passes, audit 100%, build clean, floats in range.

---

## 9. Todo checklist

### P0

- [ ] Freeze codebook v2 and AI annotation prompt
- [ ] Generate 50 taxonomy, 100 explanation, 100 claim sheets
- [ ] Agents pre-label all sheets
- [ ] 60-row blind overlap labeled by all 3 authors
- [ ] Primary slices reviewed by each author
- [ ] Adjudication and final labels frozen
- [ ] Fleiss' kappa, AI–human agreement, correction rate
- [ ] Fixed-hp m3 feature-group ablation (27 fits)
- [ ] GPT/Qwen on 158-row length-matched subset
- [ ] Prompt sensitivity (3 prompts × 100 rows × 3 seeds)
- [ ] Cross-seed SHAP stability
- [ ] CPU latency (200 rows)
- [ ] Claim-verdict audit (100 claims)
- [ ] Paper integration and build
- [ ] P0 reviewer pass

### P1

- [ ] Select second corpus or define 300–500 new labels
- [ ] Label second corpus; kappa on overlap
- [ ] 300-question retrieval qrels
- [ ] BM25/FAISS retrieval stress test
- [ ] External baseline reimplementation
- [ ] Conformal prediction (90/95%)
- [ ] Multicalibration by length/claim count
- [ ] Paper integration and build
- [ ] P1 reviewer pass

### P2 (deferred)

- [ ] HaluEval-LC split and 500-row human verification
- [ ] RAGTruth-Claim 1,000 responses, 2 annotators, kappa
- [ ] 12-detector zoo
- [ ] Label-free prevalence calibration
- [ ] Dataset card, license, leaderboard, code
- [ ] P2 go/no-go review

Do not start these before P0+P1 is submitted or accepted.

---

## 10. Start condition

Do not start until the author confirms:

- the package, recommended **P0 + P1**;
- the three author roles;
- the annotation budget (author time or paid annotators);
- whether a second corpus is available.

P2 is deferred. It is not part of the current submission.

On confirmation, begin with **P0 day 0**: freeze the codebook and generate the sheets.
