# WP4 — explanations for the rebuilt EC-XGB (m3)

Generated: 2026-11-10T20:02:31.830895+00:00

- Model: `artifacts\models\rebuild\fixed_hp\xgboost_m3_seed_42.joblib`
- Model version: `m3-fixed-hp-seed-42`
- Feature set: 35 m3 features (26 base + 8 claim-level NLI + `source_ragtruth`)
- SHAP: exact TreeSHAP via `pred_contribs=True` (`approx_contribs=False`), raw margin (log-odds)
- Global sample: 1,000 label-stratified HaluEval test rows, seed 42
- Human judgment columns in `m3_audit_cases.csv`: left empty

## 1. Commands

```powershell
& .venv\Scripts\python.exe src\models\rebuild_explanations.py
& .venv\Scripts\python.exe src\models\rebuild_explanations.py --perturbation --n-perturb 20
```

## 2. Global TreeSHAP

Exactness check: `max |sum(SHAP) + bias - margin| = 9.241e-06`. Mean bias (expected margin) = 0.015585; mean raw score = 0.498492.
Feature-construction check: the m3 scores reproduce `artifacts\results\rebuild\fixed_hp\b6_predictions.parquet` on all 3,000 test rows (max absolute difference = 1.192e-07).

Top 15 by mean |SHAP|:

| rank | feature | feature_group | mean_abs_shap | share_pct | constant_on_sample |
| --- | --- | --- | --- | --- | --- |
| 1 | overlap_answer_context | lexical | 1.661161 | 18.618 | False |
| 2 | n_words | length | 1.108346 | 12.422 | False |
| 3 | cosine_q_ans | semantic | 0.957156 | 10.728 | False |
| 4 | n_chars | length | 0.822751 | 9.221 | False |
| 5 | jaccard_ans_q | lexical | 0.779911 | 8.741 | False |
| 6 | overlap_answer_question | lexical | 0.604596 | 6.776 | False |
| 7 | nli_ctx_contradicts_ans | nli | 0.418748 | 4.693 | False |
| 8 | claim_mean_contradicts | claim_nli | 0.221509 | 2.483 | False |
| 9 | nli_ctx_neutral_ans | nli | 0.206988 | 2.320 | False |
| 10 | jaccard_ans_ctx | lexical | 0.167657 | 1.879 | False |
| 11 | source_ragtruth | source | 0.165838 | 1.859 | True |
| 12 | nli_ctx_entails_ans | nli | 0.156164 | 1.750 | False |
| 13 | nli_ans_contradicts_ctx | nli | 0.153920 | 1.725 | False |
| 14 | entity_overlap_ratio | entity | 0.147240 | 1.650 | False |
| 15 | avg_word_len | length | 0.130318 | 1.461 | False |

Figure: `artifacts\results\rebuild\explanations\m3_shap_top15.png` (600 DPI, top 15, horizontal bars).

The top 10 features account for 77.88% of the total mean |SHAP| mass; the top 15 account for 86.33%.

## 3. Per-case audit

The 26 cases come from `artifacts/results/b5/b5_review_cases_reviewed.csv`. Under m3 the roles redistribute to 7 FP, 12 FN, 1 TP, 3 borderline, 3 other (B2 roles were 9 FP, 10 FN, 7 borderline).

Top-3 m3 features per case (values and signed contributions are in the CSV):

| case_id | sample_id | label | raw_score | top3 |
| --- | --- | --- | --- | --- |
| m3_fp_01 | q_8518_correct | 0 | 0.803403 | overlap_answer_context, n_words, nli_ctx_contradicts_ans |
| m3_fp_02 | q_3061_correct | 0 | 0.863881 | overlap_answer_context, n_words, cosine_q_ans |
| m3_fp_03 | q_7742_correct | 0 | 0.514651 | overlap_answer_context, overlap_answer_question, n_words |
| m3_borderline_01 | q_1402_correct | 0 | 0.400871 | overlap_answer_context, nli_ctx_contradicts_ans, cosine_q_ans |
| m3_fp_04 | q_8650_correct | 0 | 0.957739 | overlap_answer_context, cosine_q_ans, jaccard_ans_q |
| m3_other_01 | q_4156_correct | 0 | 0.319524 | overlap_answer_context, overlap_answer_question, jaccard_ans_q |
| m3_fp_05 | q_5922_correct | 0 | 0.602081 | overlap_answer_context, jaccard_ans_q, overlap_answer_question |
| m3_fp_06 | q_4329_correct | 0 | 0.777704 | overlap_answer_context, jaccard_ans_q, overlap_answer_question |
| m3_borderline_02 | q_7876_correct | 0 | 0.402577 | overlap_answer_context, jaccard_ans_q, overlap_answer_question |
| m3_fn_01 | q_4539_hallucinated | 1 | 0.378627 | overlap_answer_context, overlap_answer_question, nli_ctx_contradicts_ans |
| m3_fn_02 | q_2910_hallucinated | 1 | 0.153801 | overlap_answer_context, n_words, n_chars |
| m3_fn_03 | q_1921_hallucinated | 1 | 0.118741 | overlap_answer_context, cosine_q_ans, n_words |
| m3_fn_04 | q_5384_hallucinated | 1 | 0.004173 | overlap_answer_context, n_words, n_chars |
| m3_fn_05 | q_1729_hallucinated | 1 | 0.489567 | overlap_answer_context, jaccard_ans_q, overlap_answer_question |
| m3_fn_06 | q_1739_hallucinated | 1 | 0.143418 | overlap_answer_context, n_words, jaccard_ans_q |
| m3_fn_07 | q_2184_hallucinated | 1 | 0.075858 | overlap_answer_context, nli_ctx_contradicts_ans, claim_mean_contradicts |
| m3_fn_08 | q_2251_hallucinated | 1 | 0.269877 | overlap_answer_context, jaccard_ans_q, n_words |
| m3_fn_09 | q_2436_hallucinated | 1 | 0.296046 | overlap_answer_context, jaccard_ans_q, overlap_answer_question |
| m3_fn_10 | q_5173_hallucinated | 1 | 0.010656 | overlap_answer_context, n_words, n_chars |
| m3_fn_11 | q_6989_hallucinated | 1 | 0.460817 | overlap_answer_context, overlap_answer_question, n_words |
| m3_other_02 | q_6297_correct | 0 | 0.282159 | overlap_answer_context, nli_ctx_neutral_ans, n_words |
| m3_tp_01 | q_4092_hallucinated | 1 | 0.668624 | overlap_answer_context, nli_ctx_neutral_ans, n_words |
| m3_borderline_03 | q_9100_correct | 0 | 0.403215 | overlap_answer_context, n_words, nli_ctx_contradicts_ans |
| m3_fn_12 | q_8342_hallucinated | 1 | 0.403363 | overlap_answer_context, nli_ctx_neutral_ans, overlap_answer_question |
| m3_fp_07 | q_6775_correct | 0 | 0.799388 | overlap_answer_context, jaccard_ans_q, n_words |
| m3_other_03 | q_6955_correct | 0 | 0.318879 | overlap_answer_context, nli_ctx_neutral_ans, nli_ctx_contradicts_ans |

`m3_audit_cases.csv` has exactly the requested columns: `case_id, sample_id, label, raw_score, top_features, top_values, top_shap, snippet, answer_plausible, review_status`. `answer_plausible` and `review_status` are empty for the author. `top_values` are model-input values; `n_chars` and `n_words` are therefore log1p-transformed.

## 4. Comparison with the frozen B2 ranking

Reference: `artifacts\results\b5\b5_feature_importance.json` (`frozen B5 SHAP ranking (B2 seed 42)`, n=3000). No B2 SHAP CSV exists, so the frozen B5 JSON is the reference.

Top 10 side by side (m3 rank / B2 rank):

| m3_rank | m3_feature | b2_rank | b2_feature |
| --- | --- | --- | --- |
| 1 | overlap_answer_context | 1 | overlap_answer_context |
| 2 | n_words | 2 | n_words |
| 3 | cosine_q_ans | 3 | n_chars |
| 4 | n_chars | 4 | nli_ctx_contradicts_ans |
| 5 | jaccard_ans_q | 5 | cosine_q_ans |
| 6 | overlap_answer_question | 6 | nli_ctx_neutral_ans |
| 7 | nli_ctx_contradicts_ans | 7 | cosine_ctx_ans |
| 8 | claim_mean_contradicts | 8 | jaccard_ans_q |
| 9 | nli_ctx_neutral_ans | 9 | entity_overlap_ratio |
| 10 | jaccard_ans_ctx | 10 | overlap_answer_question |

Rank agreement over the 26 shared features: Spearman rho = 0.878208, Kendall tau = 0.729529. Top-10 Jaccard = 0.666667; top-3 Jaccard = 0.500000.

The core explanation story persists: context overlap and answer length still dominate, and two of the three top features are shared with B2. The change is mainly a relative reweighting plus an added claim-level signal, not a replacement of the original drivers. The m3 top-10 adds claim_mean_contradicts, jaccard_ans_ctx and drops cosine_ctx_ans, entity_overlap_ratio relative to B2.

The main relative change is `cosine_q_ans`: B2 rank 5 (0.339883) to m3 rank 3 (0.957156). `claim_mean_contradicts` enters at rank 8 (0.221509); its margin over `nli_ctx_neutral_ans` at rank 9 is only 0.014521, so the exact ordering at that cut-off is soft.

Same-1,000-row B2 control: Spearman rho = 0.878208, Kendall tau = 0.723346, top-10 Jaccard = 0.538462. This control rules out the 1,000-row subsample as the main reason for the m3/B2 differences.

Full per-feature comparison: `m3_vs_b2_importance.csv`.

## 5. Perturbation pilot

Status: **completed**. 20/20 samples, 48 changed perturbations (72 unchanged), 36.2s wall time including 27.1s model load, budget 15.0 min, device `cuda:0`. Mean time per changed perturbation after load: 0.188s.

| perturbation | n | mean_abs_score_delta | max_abs_score_delta | large_delta_rate | top1_flip_rate | mean_top5_overlap | mean_spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clause_shuffle | 2 | 0.000085 | 0.000170 | 0.000000 | 0.000000 | 1.000000 | 0.999160 |
| date | 2 | 0.482044 | 0.964040 | 0.500000 | 0.000000 | 0.700000 | 0.785714 |
| entity | 11 | 0.339129 | 0.982230 | 0.363636 | 0.545455 | 0.781818 | 0.847823 |
| irrelevant_insert | 20 | 0.002739 | 0.012740 | 0.000000 | 0.000000 | 0.920000 | 0.922451 |
| numeric | 3 | 0.331666 | 0.994556 | 0.333333 | 0.000000 | 0.733333 | 0.731372 |
| support_removal | 10 | 0.464341 | 0.983737 | 0.500000 | 0.000000 | 0.780000 | 0.808431 |

Full rows: `m3_perturbations.csv`; aggregates: `m3_perturbation_aggregates.csv`; timing: `m3_perturbation_summary.json`. These are multi-feature interventions: one perturbation can move length, overlap, NLI and claim features at once, so the deltas are not single-feature causal effects. Per-type n is small (2-20); treat this as a timing and plumbing pilot, not an estimate.

## 6. Caveats

- SHAP values are additive on the **raw margin (log-odds)**, not on the probability or the Platt display score. A positive value raises the hallucination margin; it is not a percentage-point change in risk.
- The audit cases are **error-enriched**: they were selected from B2 false positives, false negatives and borderline cases, not sampled at random. Their m3 scores and roles differ from the B2 roles, and the set says nothing about population-level rates.
- `source_ragtruth` is constant 0 on HaluEval. Its nonzero mean |SHAP| (0.165838) comes from interaction structure in the trees, not from within-domain source variation. Do not read it as a causal 'trained on RAGTruth' effect for individual HaluEval rows.
- The global ranking uses 1,000 rows. It is a deterministic stratified subsample, not a bootstrap; small rank differences near the cut-off should not be over-read.
- Only the seed-42 m3 model is explained. The seed-123 and seed-456 models exist, but their SHAP rankings were not computed, so cross-seed explanation stability is untested.
- `n_chars` and `n_words` enter the m3 model as log1p values. The audit CSV reports model-input values, so those two features appear on the log scale.
- The B5.5 audit was AI-assisted (`review_mode = ai-expert`). This export leaves the human columns empty and does not describe the old judgments as human review.
