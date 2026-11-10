# WP5 Audit Summary (AI-assisted draft)

**Status:** draft for author verification. All 46 rows in the two label sheets carry
`review_status = needs_author_verification`. Nothing in this package is a human
annotation, an independent review, or a two-annotator study. Do not report a Cohen's
kappa from these labels. The kappa worksheets in §5 are blank and require a second
human rater.

**Date:** 2026-11-11  
**Branch:** `rebuild`  
**Checklist:** `docs/rebuild/human_verification_checklist.md`

---

## 1. What was labelled

| File | Rows | Source | Model provenance |
|---|---:|---|---|
| `wp5_error_labels.csv` | 20 | `artifacts/results/rebuild/fixed_hp/fixed_hp_predictions.parquet` (m3, seed 42, `halueval_test`), joined to `data/processed/qa_clean.parquet` for full text | EC-XGB m3 fixed-hp seed 42 (`m3-fixed-hp-seed-42`), 35 features |
| `wp5_explanation_labels.csv` | 26 | `artifacts/results/rebuild/explanations/m3_audit_cases.csv` | EC-XGB m3 fixed-hp seed 42; exact TreeSHAP top-5 features, input values, and signed contributions |

Both sheets are now m3, not B2. The explanation sheet keeps the 26 cases selected in
the old B5 audit, but every score and SHAP value is re-computed under m3. The error
sheet is a fresh m3 draw: **all 8 m3 false positives** and **12 of the 35 m3 false
negatives** from the 3,000-row HaluEval test set. The 12 false negatives are a
purposive codebook-coverage sample, not a random sample.

In `wp5_error_labels.csv`, `label` is the benchmark gold label (0 = grounded,
1 = hallucinated). The false-positive/false-negative direction is stated at the start
of each rationale.

---

## 2. Error-label counts

### 2.1 Error direction

| Direction | Rows | Gold label |
|---|---:|---|
| Detector false positive (model flagged a grounded answer) | 8 | 0 |
| Detector false negative (model missed a hallucinated answer) | 12 | 1 |
| **Total** | **20** | |

The full m3 HaluEval test confusion table is 8 FP and 35 FN out of 3,000 rows, so the
test error rate is 43/3000 = 1.43%. The 20-case sheet includes every FP and 12 of the
35 FNs.

### 2.2 Counts by category

Primary is the category that most directly explains the error. Secondary is an
independent co-occurring defect; 11 rows carry a secondary label.

| # | Category | Family | Primary | Secondary | Total |
|---:|---|---|---:|---:|---:|
| 1 | `entity_misattribution` | content | 1 | 1 | 2 |
| 2 | `entity_conflation` | content | 1 | 0 | 1 |
| 3 | `entity_substitution` | content | 0 | 3 | 3 |
| 4 | `entity_name_distortion` | content | 1 | 0 | 1 |
| 5 | `fabricated_entity` | content | 1 | 1 | 2 |
| 6 | `generic_or_descriptive_answer` | content | 2 | 0 | 2 |
| 7 | `relation_inversion` | content | 1 | 0 | 1 |
| 8 | `attribute_error` | content | 1 | 0 | 1 |
| 9 | `numeric_or_temporal_error` | content | 1 | 0 | 1 |
| 10 | `wrong_granularity` | content | 1 | 0 | 1 |
| 11 | `coreference_error` | content | 1 | 0 | 1 |
| 12 | `non_responsive_or_partial_answer` | content | 1 | 0 | 1 |
| 13 | `unsupported_specificity` | content | 0 | 1 | 1 |
| 14 | `context_contradiction` | content | 0 | 1 | 1 |
| 15 | `answer_span_copy` | content | 1 | 0 | 1 |
| 16 | `artifact_grounded_overflag` | artifact | 4 | 2 | 6 |
| 17 | `artifact_short_answer_fp` | artifact | 2 | 0 | 2 |
| 18 | `artifact_label_noise` | artifact | 1 | 0 | 1 |
| 19 | `artifact_truncated_context` | artifact | 0 | 1 | 1 |
| 20 | `artifact_question_ambiguity` | artifact | 0 | 1 | 1 |
| | **Total primary labels** | | **20** | | |

**Family totals:** 13 content primaries and 7 artifact primaries. All 20 categories
appear at least once as a primary or secondary label, so this sheet exercises the
whole codebook. That is a property of the purposive selection, not evidence that every
category is equally common.

Among the 8 false positives, 6 have an artifact primary (4
`artifact_grounded_overflag`, 2 `artifact_short_answer_fp`) and 2 have a content-form
primary (`answer_span_copy`, `generic_or_descriptive_answer`). Among the 12 false
negatives, 11 have a content primary and 1 is `artifact_label_noise`.

### 2.3 Confidence distribution

| Confidence | Rows |
|---:|---:|
| 5 | 10 |
| 4 | 6 |
| 3 | 2 |
| 2 | 2 |
| 1 | 0 |

The two confidence-2 rows are E01 (`q_3027`, long copied span / operator-versus-acquirer
confusion) and E15 (`q_5861`, coreference and credit ambiguity). They are the first
rows the author should resolve.

### 2.4 What the sample suggests, with the caveat

The m3 model has far fewer errors on HaluEval than the old B2 model: only 8 false
positives remain, and several B2 false positives are now correctly below threshold.
The remaining false positives are still dominated by answer-form and length/overlap
shortcuts: correct one-word answers, grounded names, and one descriptive answer. The
false negatives are dominated by entity, relation, and answer-form errors that reuse
context words and therefore keep overlap high.

This paragraph describes a **purposive, error-enriched audit sample**. It is not a
test-set prevalence estimate and must not be written as one. See §4.

---

## 3. Explanation-label counts

### 3.1 Label distribution

| Label | Rows |
|---|---:|
| `plausible` | 10 |
| `implausible` | 13 |
| `unsure` | 3 |
| **Total** | **26** |

These are fresh AI-drafted labels on the m3 SHAP output. They are not a reconciliation
of the old B5 `reviewer_1` / `reviewer_2` passes, and they are not human labels.

### 3.2 m3 audit roles

The 26 cases were selected under B2 and re-scored under m3. Their m3 roles are:

| m3 role | Cases |
|---|---:|
| False positive | 7 |
| False negative | 12 |
| Borderline (0.35-0.65) | 3 |
| Other (correctly scored, non-FP/FN) | 3 |
| True positive | 1 |
| **Total** | **26** |

The m3 labels are 13 label-0 and 13 label-1 rows. The set is deliberately error-heavy
and borderline-heavy; it is not a random sample. The 26-case audit set contains 7 of
the 8 test-set false positives; the eighth (`q_3027_correct`) is only in the 20-case
error sheet. `X01`-`X26` follow the row order of `m3_audit_cases.csv`; join on
`sample_id` rather than assuming the old B5 case IDs.

### 3.3 Confidence distribution

| Confidence | Rows |
|---:|---:|
| 5 | 4 |
| 4 | 12 |
| 3 | 9 |
| 2 | 1 |
| 1 | 0 |

### 3.4 What the explanation sample suggests, with the caveat

Thirteen of 26 explanations are implausible on the m3 reading. The dominant failure is
a grounded answer with a high score driven by question similarity, length, or
zero-overlap terms, and the mirror failure is a hallucinated answer with a low score
because it reuses context entities. Ten are plausible, including several cases that
were B2 false positives but are now correctly below threshold under m3. Three are
genuinely unresolved: `q_7742` (0.515, just over), `q_1729` (0.490, just under), and
`q_6989` (0.461, borderline description).

Two of the ten plausible cases are benchmark label-noise cases (`q_2251`, `q_2436`):
the answer is supported and the low score is sensible, even though the dataset label is
1. The explanation audit judges the feature story against the text, not against the
label.

Again, this is an audit sample. Do not report 13/26 as a test-set rate.

---

## 4. Sampling caveat (required wherever these numbers appear)

1. **Error sheet.** 20 cases = all 8 m3 false positives + 12 of 35 m3 false negatives
   from a 3,000-row test set. The 12 FNs were chosen for codebook coverage, not at
   random. The test-set error rate is 43/3000 = 1.43%, not 60%. Category counts
   describe the audit sheet only.
2. **Explanation sheet.** 26 cases, selected under B2 and re-scored under m3. The set
   has 7 FP, 12 FN, 3 borderline, 3 other, and 1 TP under m3. It is deliberately
   error-heavy and borderline-heavy, not a random or prevalence sample.
3. **No prevalence claims.** Do not write "most errors are X%" or "X% of hallucinations
   are..." from these sheets. Report counts and the sampling frame, or report the
   separate full-test metrics from `artifacts/results/rebuild/`.
4. **Selection is not blinding.** The 12 error FNs were chosen to cover categories.
   That makes the sheet useful for codebook validation but unsuitable for estimating
   category shares. If the paper needs category shares, draw a random sample from all
   35 FNs and label it separately.

---

## 5. Kappa worksheet (blank; second rater required)

The AI draft is **not** a rater. A second human must label independently, without
seeing the AI labels or rationales, before any agreement statistic is computed. Give
the second rater the question, context, answer, and top-2 feature names, but not the
`explanation_plausible`, `rationale`, or `review_status` columns.

### 5.1 Explanation plausibility: 3x3 worksheet

Fill the cells with counts from the second rater. Row totals are the AI draft counts.

| AI draft \ Second rater | plausible | implausible | unsure | Row total |
|---|---:|---:|---:|---:|
| plausible | | | | 10 |
| implausible | | | | 13 |
| unsure | | | | 3 |
| **Column total** | | | | **26** |

After the table is complete:

- `p_o = (diagonal sum) / 26`
- `p_e = (row_plausible x col_plausible + row_implausible x col_implausible + row_unsure x col_unsure) / 26^2`
- `kappa = (p_o - p_e) / (1 - p_e)`
- Report `n = 26`, observed agreement, kappa, and a 95% CI (1,000-resample bootstrap
  or the analytical Fleiss-Cohen formula). State which method was used.

Do not compute this from the AI labels alone. A single-rater "kappa" is not a kappa.

### 5.2 Error sheet: collapsed 2x2 worksheet

Twenty categories across 20 cases are too sparse for a stable 20x20 kappa. Use the
content-versus-artifact collapse for a defensible 2x2, and report raw agreement on the
full 20-category sheet as a descriptive supplement.

| AI draft \ Second rater | content error | artifact / no content error | Row total |
|---|---:|---:|---:|
| content error | | | 13 |
| artifact / no content error | | | 7 |
| **Column total** | | | **20** |

If the second rater prefers to keep the full 20-category labels, report percent
agreement and a confusion table, not a single kappa. With 20 cases and 20 categories,
kappa is unstable and can be misleading.

### 5.3 Per-case agreement sheet

Use the blank columns in `docs/rebuild/human_verification_checklist.md` §5. For each
case, record the second rater's primary and secondary categories, whether they match
the AI draft, and a short note. Do not discuss cases before both label sets are
complete.

---

## 6. The three most ambiguous rows for the author

### 1. E01 / `q_3027_correct` — is a copied span an answer?

m3 score 0.526, just above the threshold. The answer is an 80-word verbatim context
span that begins "The JKWI was the designated operator..." and later says the JKWI
"had acquired from Conrail" the line segments. The question asks which company
acquired the railroad. The gold label is 0, so the dataset treats the span as a
grounded answer; the score is only 0.026 above the threshold.

**Decision:** is this (a) a grounded but verbose answer, in which case the FP count
stays at 8 and `answer_span_copy` is only a form note, or (b) a role-confused answer
that never names the acquirer, in which case the gold label is questionable and the
row should be flagged as a benchmark/answer-form ambiguity? This is the most
contestable row in the error sheet.

### 2. E15 / `q_5861_hallucinated` — Goldcrest versus Puttnam

m3 score 0.226. The context says *You and I* was "produced by David Puttnam for his
company Goldcrest Films." The answer is `"Goldcrest Films also produced it."` The
pronoun "it" can refer to *You and I* (supported by the context) or the 1984 film (not
stated in the context, although Goldcrest was a production company for *The Killing
Fields* in the real world). The question asks "who produced" and the answer names a
company, not the person.

**Decision:** is this (a) a coreference error, (b) a question-ambiguity / answer-form
case, (c) benchmark label noise because Goldcrest is a valid co-producer, or (d) a
genuine false negative because the expected answer is David Puttnam? The current draft
uses `coreference_error` with `artifact_question_ambiguity` as secondary and
confidence 2.

### 3. E05 / `q_6775_correct` — does a description count as an answer?

m3 score 0.799. The question asks "Which president was present for the Mexican Civil
War who originated from Oaxaca?" The answer is `"liberal politician of Zapotec origin
from Oaxaca."` It describes Benito Juárez but never names him. The gold label is 0, so
the benchmark accepts the description. E12 / `q_1702_hallucinated` is the mirror case:
the answer `"Northern France Battle"` is rejected as a generic label for the Dieppe
Raid.

**Decision:** adopt a name-only rule, a unique-description rule, or a documented
case-by-case rule. This changes E05, E12, and the content/artifact split. The
current draft treats E05 as `generic_or_descriptive_answer` with
`artifact_grounded_overflag` secondary and E12 as `generic_or_descriptive_answer` with
`fabricated_entity` secondary.

### Also check

E13 / `q_3297_hallucinated` (m3 0.312) is a clear label-noise case: the context gives
Fred Phelps Sr.'s death as March 19, 2014. If the author excludes label-noise rows from
the FN count, the 12-FN sheet becomes 11 FNs plus one benchmark-quality case.

---

## 7. Suggested wording (only after author verification)

Use:

- "AI-assisted author review" or "author-verified audit" (only after the author has
  actually checked the rows).
- "purposive, error-enriched audit sample" whenever category counts appear.
- "the audit sample, not the test set" when reporting proportions.
- "a second rater is required before reporting agreement" if no second rater has
  labelled the sheet.
- "the 26 cases were selected under B2 and re-scored under m3" whenever the
  explanation sheet is described.

Do not use:

- "two human annotators", "independent human review", "inter-annotator agreement".
- "human-verified" before the author has changed the `review_status` values.
- "kappa = ..." from the AI labels alone.
- "most hallucinations are..." from the 20-case error-enriched sheet.
- "the m3 audit does not exist" or any B2 provenance for the current sheets.

---

## 8. Files and next step

| File | Purpose |
|---|---|
| `docs/rebuild/wp5_codebook.md` | 20-category definitions, examples, decision rules, primary/secondary rule |
| `docs/rebuild/wp5_error_labels.csv` | 20 m3 error cases with AI-drafted categories and rationales |
| `docs/rebuild/wp5_explanation_labels.csv` | 26 m3 explanation cases with AI-drafted plausibility labels and rationales |
| `docs/rebuild/wp5_summary.md` | This summary, counts, caveats, and kappa worksheets |
| `docs/rebuild/human_verification_checklist.md` | Step-by-step author verification checklist with per-case checkboxes |

Next step: open `human_verification_checklist.md` and work through §2 (error sheet)
and §3 (explanation sheet). Resolve the three ambiguous rows in §6 before changing any
label that depends on them.
