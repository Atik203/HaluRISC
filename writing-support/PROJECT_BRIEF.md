# PROJECT_BRIEF.md

- **Last scanned:** 2026-10-24
- **Files read:** `Journal_Paper/halurisc.tex`, `Journal_Paper/ref.bib`, `Journal_Paper/out/halurisc.pdf`,
  `artifacts/results/b2/b2_model_comparison.csv`, `b2/b2_statistical_tests.json`,
  `artifacts/results/b6/b6_model_comparison.csv`, `b6/b6_external_metrics.csv`,
  `artifacts/results/review/calibration_protocol.json`, `review/length_matched_repeat.json`,
  `review/neural_baseline/neural_baseline_summary.json`,
  `review/neural_adapted/neural_adapted_summary.json`,
  `review/llm_judge_gpt_ec_xgb.json`, `review/llm_judge_qwen.json`,
  `artifacts/results/latency_analysis.json`, `artifacts/results/ablation_results.csv`,
  `src/models/train_neural_baseline.py`, `src/models/train_neural_adapted.py`,
  `src/models/review_number_audit.py`, `src/models/run_b6_modified.py`,
  `docs/writing-guideline.md`, `docs/prompt.md`, `Review_Checklist.md`, `README.md` (partial).
- **Skimmed or skipped:** `web/` (frontend, not evidence for claims), `colab/` notebook (reproduction path),
  `data/raw/` and `data/processed/` (schemas and counts read through result files, not row-level data),
  secrets listed by the scanner (never opened). The 231-test suite was executed, not read line by line.

## 1. Problem, gap, objective

- **Problem:** LLM answers can be fluent and unsupported by the evidence. Detection must be cheap, fast,
  and explainable enough to deploy.
- **Gap (author's framing):** judge models are slow and costly, encoder classifiers need model weights and a
  GPU, and feature-based detectors rarely report calibration or behavior outside the training domain
  [tex:abstract; §1]. Four issues remain open together: calibration, explanation quality, cross-domain
  behavior, and case-level error analysis [tex:§1 P3].
- **Objective:** estimate $P(\text{hallucinated}\mid q,c,a)$ with a small black-box model, return a calibrated
  probability, explain it per feature, and test it under domain shift [tex:§1 P3, §3.1].
- **Article type:** method-based (datasets inside Experiments), CAS single column, 17 pages
  [tex:documentclass; docs/writing-guideline.md G5].

## 2. Data

| Corpus | Rows | Groups | Use | Source |
|---|---|---|---|---|
| HaluEval QA | 20,000 (10,000 questions × 2 answers) | 10,000 | train 14,000 / val 3,000 / test 3,000, grouped 70/15/15 | [tex:§4.2; Tab. 3] |
| RAGTruth QA | 5,934 | 989 | calib 5,034 / held-out test 900 | [tex:§4.2; Tab. 3] |
| RAGTruth full | 17,790 | 2,965 | transfer, 2,700-row test | [tex:§4.2, §5.4; Tab. 3] |
| FaithBench | 750 | 750 | zero-shot | [tex:§4.2; Tab. 3] |

- Licenses: HaluEval MIT, RAGTruth MIT, FaithBench CC BY-NC-SA 4.0. Raw files not redistributed.
  [tex:§4.2]
- Preprocessing: whitespace normalization; lexical features lowercased; NER/NLI keep casing; missing context
  gives overlap 0 and NLI 1/3 [tex:§4.2]. Status: code + manuscript; not re-executed in this pass.

## 3. Methods

- **Baseline:** tuned XGBoost on 26 hand-crafted evidence-consistency features in seven groups.
  Chosen for nonlinear interactions, fast tabular training, exact SHAP values [tex:§2.2].
- **Proposed (EC-XGB):** 35 features = 26 base + 8 claim-level NLI aggregates + binary source indicator;
  log-scaled length with monotone constraints; multi-source training on HaluEval plus RAGTruth non-QA rows
  [tex:§3.3]. Contributions are the combination and the evaluation protocol, not a new learner [tex:§3.3].
- **Calibration:** Platt (default) and isotonic, fitted on validation only; target recalibration uses a
  labeled target split [tex:§3.4, §4.6; Tab. 4].
- **Explanations:** SHAP TreeExplainer; stability via permutation, bootstrap, perturbation, neutralization
  [tex:§3.4, §4.6].
- **Key notation:** $z_i=\sum_k f_k(x_i)$; logistic loss + tree regularizer; ECE over 10 bins;
  $\psi(x)$ claim aggregates [tex:Eq. 1–4].

## 4. Workflow

1. Download HaluEval, RAGTruth, FaithBench from original sources; save revisions and SHA-256 hashes
   [tex:§4.2, §3.5].
2. Clean text, build grouped 70/15/15 split, save split indices [tex:§4.2; `artifacts/split_indices.json`].
3. Extract 26 base features + 8 claim aggregates + source indicator [tex:§3.2–3.3].
4. Tune XGBoost with a 30-configuration randomized search and grouped 5-fold CV [tex:§4.4].
5. Train baseline and EC-XGB for seeds 42/123/456; fit calibrators on validation [tex:§4.3–4.4].
6. Evaluate in-domain, then RAGTruth/FaithBench; run calibration-shift, ablation, SHAP, length controls,
   latency/judge comparisons [tex:§4.6, §5].
7. Audit every reported number against frozen artifacts; build the PDF [tex:§3.5; `review_number_audit.py`].

## 5. Experiments

- **Setup:** Windows 11, RTX 3060 Laptop 6 GB, CUDA 12.8, torch 2.11.0+cu128, 32 GB RAM, Python 3.12;
  scikit-learn 1.9.0, XGBoost 3.3.0, SHAP 0.52.0, spaCy 3.8.14, sentence-transformers 5.6.1 [tex:§4.1].
- **Seeds:** 42, 123, 456; every headline number is a three-seed mean [tex:§4.3–4.4].
- **Hyperparameters:** depth 4, lr 0.01, 500 trees, subsample 0.9, colsample 0.7; CV AUROC 0.9970
  [tex:§4.4]. RoBERTa: 256 tokens, 3 epochs, binary cross-entropy, batch 8, grad accumulation 2, lr 2e-5
  [`train_neural_baseline.py`].
- **Metrics:** precision, recall, F1, AUROC, PR-AUC, MCC; ECE (10 bins), Brier, NLL; McNemar, bootstrap
  95% CIs (1,000 resamples), Wilcoxon across seeds [tex:§4.5].
- **Protocols:** P0 source-only, P1 target-calibrated, P2 target-thresholded [tex:Tab. 4].

## 6. Results index

| Result | Value | Source file | Notes |
|---|---|---|---|
| Conventional XGBoost, HaluEval test | F1 0.9846, AUROC 0.9979 | `b2/b2_model_comparison.csv` | 3-seed mean |
| EC-XGB, HaluEval test | F1 0.9855, AUROC 0.9984 | `b6/b6_model_comparison.csv` | m3, 3-seed mean |
| Source-only RoBERTa | F1 0.9912, AUROC 0.9990 | `review/neural_baseline/neural_baseline_summary.json` | 3 seeds |
| Adapted RoBERTa (multi-source) | F1 0.9919, AUROC 0.9981 | `review/neural_adapted/neural_adapted_summary.json` | 3 seeds |
| RAGTruth all, conventional | F1 0.5181, AUROC 0.4753, flag 99.9%, ECE 0.6346 | `b6/b6_external_metrics.csv` | seen + unseen tasks |
| RAGTruth all, EC-XGB | F1 0.5608, AUROC 0.5815, flag 61.3%, ECE 0.2782 | `b6/b6_external_metrics.csv` | adaptation on seen tasks |
| RAGTruth QA, conventional | F1 0.3022, AUROC 0.5465, flag 99.9% | `b6/b6_external_metrics.csv` | held-out |
| RAGTruth QA, EC-XGB | F1 0.3081, AUROC 0.5491, flag 97.4% | `b6/b6_external_metrics.csv` | held-out, near chance |
| FaithBench, conventional | F1 0.8130, AUROC 0.5089, flag 99.5% | `b6/b6_external_metrics.csv` | zero-shot |
| FaithBench, EC-XGB | F1 0.1201, AUROC 0.5941, flag 7.3% | `b6/b6_external_metrics.csv` | conservative regime |
| Adapted encoder, RAGTruth all | AUROC 0.6941 | `review/neural_adapted/...` | passes EC-XGB on seen tasks |
| Adapted encoder, held-out QA | AUROC 0.4898 | `review/neural_adapted/...` | near chance |
| Adapted encoder, FaithBench | AUROC 0.4338 | `review/neural_adapted/...` | worse than source-only 0.4883 |
| Target calibration, seed 42 | ECE 0.726 → 0.126 | `review/calibration_protocol.json` | P1 Platt |
| Target calibration, 3 seeds | ECE 0.7433 → 0.1263 | `review/calibration_protocol.json` | Tab. 9 means |
| 252-row calibration | ECE 0.133 | `extras/review_extras.json` | label-budget result |
| Latency | 61.8 ms p50 | `latency_analysis.json` | component pipeline |
| Cost | $0.001 / 1,000 | `latency_analysis.json`, paper §5.7 | local estimate |
| GPT judge | F1 0.846, $0.106/1k, 1443 ms | `review/llm_judge_gpt_ec_xgb.json` | 200 balanced rows |
| Length-matched control | AUROC 0.9608 vs 0.5257, gap 0.4351 [0.429, 0.442] | `review/length_matched_repeat.json` | 163 pairs, 200 draws |
| Length-conditional AUROC | 0.9773 ± 0.0009 | `review/length_matched_repeat.json` | 12 strata |
| McNemar, XGB vs RF | p = 0.044 | `b2/b2_statistical_tests.json` | seed-42 pairs |
| McNemar, m0 vs m3 | p = 0.45; Wilcoxon p = 0.25 | `b6/b6_statistical_tests.json` | no in-domain gain |

## 7. Claim-evidence map

| Claim (abstract/conclusion) | Evidence | Support |
|---|---|---|
| EC-XGB matches the conventional model in-domain | F1 0.9855 vs 0.9846; p = 0.45 | direct |
| AUROC rises 0.475 → 0.582 on RAGTruth tasks used in training | `b6_external_metrics.csv` | direct |
| Flag rate falls 99.9% → 61.3% | same file | direct |
| Held-out QA stays near chance | AUROC 0.549; CI [0.502, 0.603] | direct |
| Calibration cuts target ECE 0.726 → 0.126 | `calibration_protocol.json` | direct |
| 252 rows give ECE 0.133 | `review_extras.json` | direct |
| 61.8 ms and $0.001/1k | `latency_analysis.json` | direct |
| Adapted encoder passes EC-XGB on seen tasks | 0.694 vs 0.582 | direct |
| Adapted encoder fails on unseen QA/FaithBench | 0.490 vs 0.549; 0.434 vs 0.594 | direct |
| Length shortcut is controlled | matched AUROC 0.9608 vs 0.5257 | direct |
| Fine-tuned encoder wins in-domain | 0.9912 vs 0.9846 | direct |

## 8. Terminology lock

| Concept | Canonical name | Acronym/symbol | Do not use |
|---|---|---|---|
| Proposed model | evidence-consistent XGBoost | EC-XGB | "the framework", "the approach" as a name |
| Baseline | conventional XGBoost | — | "standard model" |
| Training benchmark | HaluEval QA | — | "HaluEval" when the QA subset is meant |
| External corpora | RAGTruth QA, RAGTruth all, FaithBench | — | "RAGTruth" alone for both scopes |
| Display probability | calibrated score / display score | — | "confidence" |
| Raw output | raw score / risk score | — | "probability" for the uncalibrated score |
| Metric | expected calibration error | ECE | "calibration loss" |
| Feature block | claim-level NLI aggregates | — | "claim features" alone |

## 9. Conflicts between sources

| What | Where | Source trusted | Author decision needed |
|---|---|---|---|
| Conclusion says 252 rows give "the same test ECE of 0.133" | tex:conclusion | `calibration_protocol.json` (0.126) and `review_extras.json` (0.133) | Yes: 0.133 is close, not identical. Fix to "bring the test ECE to 0.133" |
| Audit script labels say "Table 4/5/..." but current PDF numbers shifted after the 4.4 insertion | `review_number_audit.py` | Values, not labels, are authoritative | No; labels are internal |
| Adapted encoder FaithBench 0.434 is worse than source-only 0.488 | tex:§5.1 | `neural_adapted_summary.json` | No; the paper says "falls behind EC-XGB", which is true. The source-only comparison is an optional nuance |
| "roughly 100 times less than the judge" | tex:conclusion | 0.106 / 0.001 = 106x | No |
| 99.9% flag rate | tex:abstract | 0.998889 | No |

## 10. Venue and supervisor constraints

- CAS single-column template, 17-page target, Elsevier numbered references [tex; `docs/writing-guideline.md`].
- Supervisor: no standalone limitations subsection; "conventional" for the baseline (2026-10-21).
- Faculty guide G1–G15: four-paragraph introduction, six-question abstract, method-based order, CRediT,
  five-item declaration [docs/writing-guideline.md G0–G15].
- AI-use policy: **unknown**. The manuscript has been agent-edited at rung B; the author should confirm the
  institution's disclosure rule once.

## 11. Section status

| Section | Status | Main gaps |
|---|---|---|
| Abstract | complete, 281 words | dense; one long training sentence |
| Introduction | complete, 4 paragraphs | clear |
| Conventional Method | complete | baseline rationale added 2026-10-23 |
| Proposed Method | complete | long paragraphs in 3.4/3.5 but technical |
| Experiments | complete | 4.1–4.6 aligned with G9 |
| Results | complete | §5.4 and §5.8 paragraphs too long; §5.1 ambiguity after the encoder run |
| Conclusion | complete | "same ECE" mismatch; encoder named generically |
| Declarations | complete | 5 items |
| References | complete | 34/34 cited |

## 12. Open questions (max 7)

1. **Target journal?** Affects reference style, abstract limit, and figure sizing. Files only say CAS 17 pages.
2. **AI-use disclosure policy?** The manuscript is agent-edited; the institution rule is unknown.
3. **Which text did the author write personally?** Needed for a real voice profile; no reliable sample found.
4. **Is the adapted-encoder result approved for the manuscript?** It was added on 2026-10-24 from the new run.
5. **Is 17 pages hard, or is 18 acceptable?** Splitting the two long result paragraphs needs ~2 lines.
6. **Should the 252-row result move into the abstract?** It is currently in the conclusion only.
7. **Is the FaithBench F1 drop acceptable as-is?** It is reported honestly but dominates the transfer table.
