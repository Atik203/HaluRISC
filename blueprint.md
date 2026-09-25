# HaluRISC — Project Blueprint (Unified)

**Status: complete.** All experiments, the deployed system, and the journal
manuscript are finished and verified against frozen artifacts. This document is
the **single source of truth** for the research design, scientific claims, and
scope.

The earlier two-version structure is historical, not operational. Version A was
the 8-week course deliverable, frozen on branch `version-A`; Version B was the
publication extension, completed on branch `version-B`. This blueprint merges
both into one design: what is described below is what was built, evaluated, and
written.

## Contents

- [1. Project at a glance](#1-project-at-a-glance)
- [2. Research question](#2-research-question)
- [3. Contributions](#3-contributions)
- [4. Positioning and research gap](#4-positioning-and-research-gap)
  - [4.1 The gap](#41-the-gap)
  - [4.2 Must-cite overlapping work and differentiation](#42-must-cite-overlapping-work-and-differentiation)
  - [4.3 Venue realism](#43-venue-realism)
- [5. Datasets](#5-datasets)
  - [5.1 Corpora](#51-corpora)
  - [5.2 Splits and leakage control](#52-splits-and-leakage-control)
  - [5.3 HaluEval limitations (must be disclosed in the paper)](#53-halueval-limitations-must-be-disclosed-in-the-paper)
- [6. Features (26 features, 7 groups)](#6-features-26-features-7-groups)
- [7. ML pipeline](#7-ml-pipeline)
  - [7.1 Models](#71-models)
  - [7.2 Protocol discipline](#72-protocol-discipline)
  - [7.3 Explainability](#73-explainability)
- [8. Evaluation strategy](#8-evaluation-strategy)
  - [8.1 Metrics](#81-metrics)
  - [8.2 Statistical testing](#82-statistical-testing)
  - [8.3 Ablations and controls](#83-ablations-and-controls)
  - [8.4 External bar](#84-external-bar)
  - [8.5 Error analysis](#85-error-analysis)
- [9. System artifact](#9-system-artifact)
  - [9.1 Architecture](#91-architecture)
  - [9.2 Claim-level verification tiers](#92-claim-level-verification-tiers)
- [10. Verified results (frozen B-run)](#10-verified-results-frozen-b-run)
- [11. Reproducibility](#11-reproducibility)
- [12. Paper and submission plan](#12-paper-and-submission-plan)
  - [12.1 Manuscripts (both complete)](#121-manuscripts-both-complete)
  - [12.2 Submission checklist](#122-submission-checklist)
- [13. Remaining manual work](#13-remaining-manual-work)
- [14. Risks and mitigations](#14-risks-and-mitigations)
- [15. Verified reference backbone](#15-verified-reference-backbone)
- [16. Final verdict](#16-final-verdict)

**Where to read more:** the operational documentation starts at
[`docs/README.md`](docs/README.md), then
[`docs/01-project-and-methods.md`](docs/01-project-and-methods.md) (datasets,
features, models, system),
[`docs/02-experiments-and-results.md`](docs/02-experiments-and-results.md)
(every experiment and number, artifact traceability), and
[`docs/03-reproduction-and-defense.md`](docs/03-reproduction-and-defense.md)
(rebuild and verify, demo script, defense Q&A, glossary). Phase deep-dives are in
`docs/b5-*`, `docs/b6-*`, and `docs/b7-*`; the frozen artifact identity is
`docs/manifest.frozen.json`.

---

## 1. Project at a glance

**Theme:** calibrated, explainable hallucination-risk prediction for black-box
LLM outputs using lightweight machine learning, evidence-consistency features,
and a presentation-quality web dashboard.

**Name:** HaluRISC (Hallucination Risk Scoring and Calibration). The name is kept
for the repository, the API, and the deployed application. The manuscript title
and text do not lead with it: the paper uses "this study" and "the proposed
model".

**Title in use:**

- Journal manuscript (`Journal_Paper/halurisc.tex`, Elsevier CAS single-column):
  *Estimating Hallucination Risk in Black-Box Large Language Model Answers with
  Calibrated and Explainable Machine Learning*

Title rules from supervisor feedback: no colon construction, no product name in
the title, problem stated first, method family stated second, readable as one
sentence. Earlier titles (*HaluRISC: Cross-Domain Calibrated and Explainable
Hallucination Risk Estimation for Black-Box LLM Responses* for the journal
manuscript, and the retired course-format title *HaluRISC: A Calibrated and
Explainable Machine Learning Framework for Hallucination Risk Prediction in
Black-Box LLM Outputs*) are historical; the course-format sources remain in git
history.

**Framing rule:** predict **risk**, not truth. HaluRISC flags answers that are
unsupported by, or contradictory to, the available evidence. It never inspects
the generating model and never claims universal truth detection. Risk
prediction is defensible; single-pass truth detection from surface text is not.

**Naming caution:** HalluLens (double "l", Bang et al., ACL 2025) is a distinct
external benchmark. It is never an abbreviation of this project.

---

## 2. Research question

> How robustly can a calibrated, explainable, black-box hallucination-risk model
> generalize across datasets and domains using evidence-consistency and
> linguistic features, and how reliable are its explanations under distribution
> shift?

This single question covers the course scope (supervised ML, feature
engineering, calibration, SHAP, deployment) and the publication scope
(cross-domain robustness, calibration under shift, explanation reliability).

---

## 3. Contributions

The contribution is compound, not any single component:

1. A reproducible black-box risk-prediction pipeline with 26 engineered
   evidence-consistency features in seven groups, including bidirectional NLI
   entailment and contradiction signals.
2. A rigorous comparison of heuristic, NLI-only, TF-IDF control, logistic
   regression, random forest, and XGBoost baselines under grouped 5-fold
   cross-validation with a three-seed protocol.
3. Calibration analysis (raw vs Platt vs isotonic, source domain and
   target-domain recalibration, subgroup diagnostics) with calibrators fitted
   on validation or designated calibration data only.
4. Explanation reliability measured, not visualized: SHAP-vs-permutation
   agreement, bootstrap stability, neutralization, controlled perturbations,
   and a structured expert audit of an error-enriched review sheet.
5. A deployed, reproducible artifact: FastAPI inference service, claim-level
   evidence verification, and a Next.js/assistant-ui dashboard, backed by a
   frozen manifest and a CPU-compatible Dockerfile.
6. An evidence-consistent XGBoost variant (EC-XGB): log-scaled length features
   under monotone evidence constraints, eight claim-level NLI aggregates over
   atomic clauses (conjunction-aware splitting), multi-source training on
   RAGTruth non-QA tasks with a source indicator, and a dedicated display
   calibrator. It preserves in-domain F1 (0.9858 vs 0.9844 standard) and fixes
   shift over-flagging (99.9% to 61.6% flagged on RAGTruth, AUROC 0.475 to
   0.568, ECE 0.635 to 0.286; QA ECE 0.743 to 0.126 with the display
   calibrator). Deployed as the API default with version b6-ec-xgb-v1.0.

**Honesty rules:**

- Do not claim to be the first to use SHAP for hallucination detection. The
  Multimedia Systems 2026 paper already did that.
- Do not claim state-of-the-art status or a guaranteed journal quartile.
- Do not describe HaluEval results as general hallucination-detection
  performance. Frame them as performance on the HaluEval benchmark.
- Report negative results with the same care as positive ones.

---

## 4. Positioning and research gap

### 4.1 The gap

Many detection methods are computationally expensive, opaque, dependent on
model internals, or presented only as offline benchmark results. There is
limited applied work testing whether lightweight black-box detectors stay
**calibrated, explainable, and useful across domains**. Individual components
(feature engineering, SHAP visualization, calibration, cross-domain evaluation)
have been explored in isolation. No published work combines all of them into
one lightweight framework with a reproducible deployable artifact.

### 4.2 Must-cite overlapping work and differentiation

| Work | Overlap | Differentiation |
| --- | --- | --- |
| Haq et al., *Multimedia Systems* 32(2), 2026 — SHAP-LIME hallucination score (DOI 10.1007/s00530-025-02150-4) | Token-level SHAP+LIME attribution with a custom hallucination score | HaluRISC explains **feature-level** evidence signals, adds calibrated risk, cross-domain evaluation, and measured explanation reliability |
| Sundaragiri et al., *IJERT* 15(04), 2026 — multi-signal framework (DOI 10.5281/zenodo.20025987) | Lexical overlap, entity coverage, semantic similarity, NLI, numeric consistency + XGBoost on HaluEval | HaluRISC adds hedging features, probability calibration, cross-domain transfer, explanation reliability, and a deployable artifact |
| Yadav & Verma, *IEEE TAI*, 2026 — hybrid encoder framework (DOI 10.1109/TAI.2026.3653354) | Encoder-based hallucination classifier | HaluRISC uses **engineered interpretable features**, not frozen neural embeddings, so explanations are semantically meaningful |
| Deng et al., *SpikeScore*, ICLR 2026 — cross-domain detection | Cross-domain hallucination detection | HaluRISC quantifies the transfer gap on RAGTruth and FaithBench and reports the length confound that drives it |

### 4.3 Venue realism

Do not target ACL/EMNLP Findings as the primary venue. This is an applied,
reproducible, calibration-aware study. Realistic candidates after the evidence
gate: IEEE Access, Multimedia Systems (differentiate from the 2026 SHAP-LIME
paper), Expert Systems with Applications, Applied Soft Computing. Venue choice,
scope, quartile, APC, and author guidelines are checked at submission time.
The journal-format manuscript is typeset in the Elsevier CAS single-column
template (`Journal_Paper/`).

---

## 5. Datasets

### 5.1 Corpora

| Corpus | Role | Rows / groups | License | Redistribution |
| --- | --- | --- | --- | --- |
| HaluEval QA (Li et al., EMNLP 2023) | In-domain training and test | 20,000 rows / 10,000 `item_idx` groups (70/15/15) | MIT | Allowed with attribution |
| RAGTruth, official (Niu et al., ACL 2024) | Primary external test + target recalibration | 17,790 rows / 2,965 `source_id` groups; QA split 5,934 rows / 989 groups | MIT | Allowed |
| FaithBench (Bao et al., NAACL 2025) | Stress test (summarization) | 750 rows / 750 groups | CC BY-NC-SA 4.0 | **Not allowed**; ship downloader + hashes only |

Pinned provenance (recorded in `data/raw/*/revision.json` and
`artifacts/results/dataset_license_manifest.json`):

- HaluEval: commit `b7253db3cdaa0ab2c382f92b26b390109174f77e`,
  `qa_data.json` SHA-256 `89ed139ec5e3…`, 6,164,420 bytes.
- RAGTruth: commit `c103204b9ce2…` (`response.jsonl`, `source_info.jsonl`).
- FaithBench: commit `cf89797d8281…` (batches 1–16, no batch 13).

Dataset acquisition is scripted (`src/data/download*.py`) and never uses Kaggle
or unverified mirrors. No permission request is required for any corpus, since
each license already grants research use. Obligations are attribution (all
three), a license notice, no redistribution of FaithBench, and share-alike for
derivatives.

### 5.2 Splits and leakage control

- Split 70/15/15 **stratified by label at the question-group level** so both
  answer variants of one question stay in one partition. Row-level splits let
  the model memorize answer pairs and inflate F1 by 0.004.
- RAGTruth QA: 5,034 rows (839 groups) for target-calibrator fitting, a
  disjoint 900 rows (150 groups) for testing. No overlapping groups.
- FaithBench label mapping: worst-severity aggregation, pre-registered
  (`Benign/empty → 0`, `Questionable/Unwanted* → 1`), with a sensitivity check.
- Split indices are saved to `artifacts/split_indices.json` + `.npy`; an
  automated assertion verifies zero cross-split group overlap.

### 5.3 HaluEval limitations (must be disclosed in the paper)

1. ~85% of the corpus is ChatGPT-generated through sampling-then-filtering, so
   the hallucination patterns are synthetic by construction.
2. ChatGPT generated and filtered the negatives, a circularity risk.
3. Binary labels ignore severity and partially supported answers.
4. The data reflects an early-2023 ChatGPT generation process.
5. HalluLens (ACL 2025) notes that HaluEval conflates factuality with
   hallucination, since it tests consistency with Wikipedia rather than with
   model training data.
6. A structured audit found benchmark label noise (items 2251 and 2436 carry a
   hallucinated label while the context supports the answer) and a severe
   answer-length confound (see §10.6).
7. The optional 50-sample manual audit file exists; the human review pass and
   claim/feedback labeling remain open (§13).

---

## 6. Features (26 features, 7 groups)

NLI features are mandatory, not optional. NLI entailment and contradiction is
the most informative published signal across feature-importance analyses.

| Group | Features | Source |
| --- | --- | --- |
| Length/style | `n_chars`, `n_words`, `n_sentences`, `avg_word_len` | regex |
| Lexical | overlap vs. context and question, Jaccard ×2 | lowercased token sets |
| Entity | entity counts, overlap ratio, novel-entity ratio | spaCy `en_core_web_sm` |
| NLI | entail / contradict / neutral × 2 directions (6 features) | `cross-encoder/nli-deberta-v3-base` (SNLI + MultiNLI) |
| Numeric | number counts, overlap ratio, novel numbers | regex |
| Hedging | hedge count, hedge density | lexicon |
| Semantic | cosine(context, answer), cosine(question, answer) | `all-MiniLM-L6-v2` sentence embeddings |

- NLI fallback if the primary checkpoint is unavailable:
  `cross-encoder/nli-MiniLM2-L6-H768`, documented as a limitation.
- Missing context: lexical overlap 0, NLI probabilities 1/3 (neutral).
- Preprocessing: whitespace normalization; lowercase only for lexical
  features; NER and NLI keep original casing.
- Scaling: `StandardScaler` for logistic regression; tree models use raw
  features.
- Feature extraction runs once to `data/processed/features_full.parquet` and is
  hash-verified against the frozen run.
- Each feature group must be ablated. New features are only justified if they
  improve out-of-domain behaviour without harming in-domain results, and
  negative results (e.g., neutral entity/hedging groups, mildly harmful
  semantic group) are reported.

---

## 7. ML pipeline

### 7.1 Models

1. Heuristic baseline: `1 − overlap_answer_context`, threshold tuned on
   validation (0.97).
2. Logistic regression on standardized features.
3. Random forest, 300 trees.
4. XGBoost: randomized search over 30 iterations (5-fold grouped CV) across
   `max_depth`, `learning_rate`, `n_estimators`, `subsample`,
   `colsample_bytree`; best configuration depth 4, learning rate 0.01, 500
   trees, subsample 0.9, column subsample 0.7 (CV AUROC 0.9970).
5. EC-XGB (deployed, B9): the same learner with log-scaled length features
   under monotone evidence constraints, eight claim-level NLI aggregates over
   atomic clauses, RAGTruth non-QA multi-source training with a source
   indicator (35 features), a Platt display calibrator fitted on the RAGTruth
   QA calibration split, and a recall-first operating point at a 5%
   false-positive budget.
6. Controls: NLI-only, TF-IDF answer-only, TF-IDF full-input, majority, and
   overlap-heuristic models quantify benchmark shortcuts.

### 7.2 Protocol discipline

- **Seeds:** 42, 123, 456; every experiment repeated and reported as mean ± std.
- **Tuning:** grouped 5-fold stratified CV on the training split only.
- **Calibration:** Platt (sigmoid) and isotonic fitted on the validation split
  only, compared on test. The test split is used once.
- **Grouping:** the two rows of one question never cross partitions, in
  training or in CV folds.
- **Artifacts:** every run saves model `.joblib`, scaler, params JSON, and
  result tables under `artifacts/`.

### 7.3 Explainability

- SHAP `TreeExplainer` over the raw XGBoost model; global mean-|SHAP| and
  beeswarm summaries; local waterfalls for case studies.
- Explanation-reliability measurements are part of the protocol, not optional:
  - **Importance triangulation:** Spearman/Kendall correlation between
    mean-|SHAP| ranking and permutation importance.
  - **Bootstrap stability:** top-5 feature set Jaccard across 1,000 resamples.
  - **Neutralization:** set top-*k* SHAP features to the median and track
    prediction change for *k* = 1, 3, 5, 10.
  - **Perturbation stability:** entity/number/date replacement, support
    removal, irrelevant insertion, clause shuffle, with Spearman correlation,
    large-change rate, and top-1 flip rate per operator.
  - **Expert audit:** structured review of an error-enriched export sheet.
- No arbitrary pass thresholds (no fixed FAC or PSI cut-offs). Report measured
  correlations, distributions, confidence intervals, and failure cases.

---

## 8. Evaluation strategy

### 8.1 Metrics

- Classification: precision, recall, F1, AUROC, PR-AUC, MCC.
- Calibration: ECE (10 bins), Brier score, NLL, reliability diagrams,
  calibration slope/intercept, subgroup ECE with sample counts.
- Efficiency: per-component latency (lexical, NER, NLI, semantic, model, SHAP),
  p50/p95 totals, artifact size, memory, and cost per 1,000 predictions versus
  an LLM judge.

### 8.2 Statistical testing

- McNemar's test on paired test predictions for the primary comparison.
- Bootstrap 95% CIs (1,000 resamples) for F1 and AUROC.
- Wilcoxon signed-rank across the three seeds.
- Report the row-level vs grouped-split delta as a leakage-control result.

### 8.3 Ablations and controls

- Remove each feature group one at a time and retrain (7 groups × 3 seeds).
- Artifact controls: answer-only, context-only, lexical/TF-IDF, and overlap.
- Subgroup analysis by task, context length, and generator where available.

### 8.4 External bar

- Zero-shot evaluation on RAGTruth (all tasks + QA) and FaithBench, with task
  and context-length breakdowns.
- A measured LLM-as-judge comparison on 200 test samples (GPT 5.6 Luna, fixed
  prompt, token-count cost accounting).
- Cite published HaluEval QA detection numbers for context; never claim SOTA.

### 8.5 Error analysis

- Inspect sampled false positives and false negatives, classify them (label
  ambiguity, short-answer ambiguity), and report the taxonomy with counts.
- Bin the full test set by answer length and report error rates against the
  label base rate, to expose benchmark shortcuts rather than hide them.

---

## 9. System artifact

### 9.1 Architecture

- **FastAPI backend** (port 8000): serves `/predict`, `/explain`, `/verify`,
  `/index`, `/retrieve`, `/judge`, `/feedback`, `/meta`, `/health`. Loads the
  model, calibrator, SHAP explainer, and feature models once at startup
  (skippable with `HALU_API_PRELOAD=0`). One worker, no `--reload`, inference
  lock, bounded inputs, LRU feature cache, CUDA fp16 with CPU fallback.
- **Next.js frontend** (port 3000): chat (`assistant-ui` + Vercel AI SDK
  streaming), analyze, dashboard, demo, about pages. The Next.js layer is a
  backend-for-frontend that proxies `/api/ml/*` to FastAPI and keeps the OpenAI
  key server-side.
- **Display score:** the headline percentage comes from an evidence-domain
  isotonic calibrator fitted on 5,034 natural RAGTruth QA rows (ECE 0.133 on
  the disjoint 900-row test, versus 0.819 raw), adjusted by per-claim verdicts
  in `/verify`. `legacy_score` (HaluEval-Platt) is shown as a labelled
  secondary signal.

### 9.2 Claim-level verification tiers

1. **Tier 1 — auto risk card** per answer, rendered inside the chat thread.
2. **Tier 2 — per-claim NLI verdicts** (`supported` / `contradicted` /
   `unsupported`) with the driving evidence sentence quoted.
3. **Tier 3 — retrieval**: document index (BM25 + FAISS + reciprocal rank
   fusion + cross-encoder reranker) and Brave web search (LLM Context + Web
   Search, with Tavily as the legacy fallback).
4. **Tier 4 — LLM judge routing** for borderline claims, feedback capture, and
   threshold tuning from labelled feedback.

The verdict headline leads the card because the calibrated score is
style-sensitive on full sentences; the model score is secondary.

---

## 10. Verified results (frozen B-run)

All numbers below come from `artifacts/results/b2..b5` and
`docs/manifest.frozen.json`. Do not restate or edit them without regenerating
and re-verifying the manifest.

### 10.1 In-domain (HaluEval QA test, n = 3,000, mean over seeds)

| Model | F1 | AUROC | MCC | ECE |
| --- | --- | --- | --- | --- |
| Heuristic (1 − overlap) | 0.935 | 0.913 | 0.872 | 0.354 |
| Logistic regression | 0.966 | 0.993 | 0.933 | 0.0099 |
| Random forest | 0.9842 | 0.9980 | 0.969 | 0.0143 |
| NLI-only | 0.671 | 0.718 | 0.319 | 0.068 |
| TF-IDF answer / full | 0.922 / 0.600 | 0.970 / 0.675 | 0.852 / 0.248 | — |
| **XGBoost (final)** | **0.9846** | **0.9979** | **0.9697** | **0.0045** |

McNemar XGBoost vs random forest `p = 0.044`; vs logistic regression
`p = 0.43`; vs heuristic `p = 0.35`. Bootstrap 95% CI: F1
[0.9812, 0.9895], AUROC [0.9964, 0.9989]. Row-level split inflates F1 to
0.9886; the grouped split corrects it to 0.9846.

### 10.2 Calibration

- Source domain: raw XGBoost is already well calibrated (ECE 0.0045); Platt
  (0.0091) and isotonic (0.0070) do **not** improve it. Honest negative result.
- Target domain (RAGTruth QA, test n = 900): raw ECE 0.8185, source Platt
  0.8089; target recalibration cuts ECE to 0.1335 (Platt) / 0.1316 (isotonic)
  and Brier from 0.801 to 0.164. The decision threshold must be retuned as
  well, since the predicted positive rate collapses at 0.5.

### 10.3 Cross-domain zero-shot transfer (recall = 1.0 everywhere)

| Corpus | n | F1 | AUROC | ECE |
| --- | --- | --- | --- | --- |
| RAGTruth QA | 900 | 0.302 | 0.540 | 0.819 |
| RAGTruth all | 17,790 | 0.603 | 0.497 | 0.560 |
| FaithBench | 750 | 0.813 | 0.531 | 0.301 |

This is the central negative result: synthetic HaluEval patterns do not
transfer to natural responses. The deployed system covers the gap with
claim-level verification.

### 10.4 Feature-group ablation (ΔF1)

Lexical −0.0247, length −0.0045, NLI −0.0023, numeric −0.0002, entity +0.0001,
hedging +0.0001, semantic +0.0014. Lexical features dominate; entity, numeric,
and hedging groups are neutral; the semantic group is mildly harmful. All
negative deltas are reported.

### 10.5 Explanation reliability

- SHAP vs permutation importance: Kendall τ = 0.66. Top-5 set identical across
  1,000 bootstrap resamples (Jaccard 1.0).
- Neutralization: mean score change −0.079 (k=1) → −0.412 (k=10), monotone.
- Perturbations: entity 0.360, number 0.345, date 0.279, support removal
  0.272, irrelevant insert 0.006, clause shuffle 0.001; top-1 SHAP feature
  never flips; Spearman 0.84–0.95. Small operator samples (dates n=13, clause
  shuffle n=14) are reported openly.
- Structured expert audit of 26 error-enriched cases (9 FP / 10 FN / 7
  borderline), two AI-assisted review passes: 23/26 agreement (88%), 15 cases
  implausible on both passes, plausible 6 and 7, unsure 5 and 4. Three
  disagreements (`q_4156_correct`, `q_1729_hallucinated`,
  `q_5173_hallucinated`) all sit on borderline scores. The sheet is
  deliberately error-heavy, so the proportions describe the audit, not the
  test set, and it must never be described as a two-human review.

### 10.6 Error analysis and the length confound

- Test set: 48 errors (12 FP, 36 FN), error rate 1.6%; sampled taxonomy: label
  ambiguity 7 FP / 4 FN, short-answer ambiguity 3 FP / 6 FN.
- Length-binned analysis (`artifacts/results/b5/b5_length_error_analysis.csv`):
  the hallucinated share rises from 2.1% (one word) to 99.2% (17+ words), and
  the model reproduces it (mean score 0.024 → 0.993). Two-thirds of false
  positives fall in the 5–8 word bin (10.6% FP rate); false negatives
  concentrate in the 1–4 word bins (~18% FN rate). This is the mechanism
  behind the zero-shot transfer failure: natural responses are long sentences,
  so their length reads as hallucination evidence.

### 10.7 Efficiency and judge comparison

| Component | p50 (ms) |
| --- | --- |
| NLI (2 directions) | 27.1 |
| spaCy NER | 21.1 |
| SBERT (MiniLM) | 9.25 |
| XGBoost inference | 1.26 |
| SHAP explanation | 1.91 |
| Core lexical | 0.10 |
| **Total per sample** | **61.8** |

Model artifact 0.82 MB; marginal cost ≈ $0.001 per 1,000 predictions. On the
same 200 samples, GPT 5.6 Luna as judge reaches F1 0.841 at $0.105 per 1,000
and 1,310 ms median, versus 0.985 at 61.8 ms for HaluRISC (McNemar
`p = 1.6 × 10⁻⁵`, agreement 84.5%).

---

## 11. Reproducibility

- `requirements.txt` uses exact pins (`==`). Exception: `xgboost==3.3.0` is
  deliberate, because the Colab-produced booster serialization does not load
  under 3.4.0. The spaCy model is pinned to the official GitHub release wheel,
  since it is not published on PyPI.
- Python 3.12 in `.venv`; heavy training runs in Colab Pro
  (`colab/HaluRISC_Training_Version_B.ipynb`, self-contained, hash-verified);
  local inference runs on the RTX 3060 6 GB reference laptop.
- `run_all_experiments.py` reproduces the run from `configs/version_b.yaml`;
  `make_manifest.py` freezes artifact hashes; `verify_artifacts.py` checks every
  reported number against the frozen artifacts.
- 227 tests pass; `pnpm run build` and `pnpm run lint` pass; API `/health`
  reports artifacts loaded on CUDA.
- Release ships: split indices, model/calibrator/scaler/SHAP artifacts, the
  dataset license manifest with revisions and SHA-256 hashes, the B5.5
  reviewed sheet, and the length-binned error table. Restricted raw data is
  never bundled.

---

## 12. Paper and submission plan

### 12.1 Manuscript (complete)

| Format | Entry point | Build |
| --- | --- | --- |
| Journal (Elsevier CAS single column, 15 pages) | `Journal_Paper/halurisc.tex` | `latexmk -pdf -outdir=out halurisc.tex` from `Journal_Paper/` |

- Figures resolve from `Journal_Paper/figures/` and `Journal_Paper/screenshots/`;
  `src/models/plot_paper_figures.py` refreshes the shift and transfer figures.
- The retired course-format manuscript, its proposal, and its reference database
  remain in git history only.
- Writing register follows `docs/prompt.md` (student register).
- The journal manuscript uses the CAS sample's six-section skeleton
  (Introduction, Conventional Method, Proposed Method, Experiment, Results,
  Conclusion) plus Declarations, and cites 30 verified references.

### 12.2 Submission checklist

1. Click through every DOI/URL in `Journal_Paper/ref.bib` (30 entries) at
   submission time; the `ieeeTai` DOI in particular must be re-checked on IEEE
   Xplore, which blocks scraping.
2. Confirm every number in the manuscript against the frozen manifest
   (`docs/manifest.frozen.json`).
3. Decide the venue, then check scope, quartile, APC, and author guidelines.
4. Optional: refresh the System chapter screenshots and the 5-minute demo
   script.

---

## 13. Remaining manual work

Only human/administrative steps remain. No code changes are required.

1. **50-sample manual audit** — `data/processed/audit_50_samples.json` is
   auto-sampled; the human label-quality review fields are still pending.
2. **Claim-eval labeling** — build the sheet with `eval_claims.py --build`,
   label claims by hand, measure verdict agreement and flagging precision and
   recall with `--evaluate`; annotate the Tier-3 retrieval queries and report
   citation recall@k.
3. **Feedback labeling** — export collected chat feedback with
   `--from-feedback`, label `correct_verdict`, then run `tune_thresholds.py`
   (`--apply` is an explicit human step).
4. **Submission checks** — DOI click-through, venue selection, and the
   optional screenshot refresh from §12.2.

---

## 14. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| HaluEval synthetic artifacts inflate performance | Grouped splits, artifact-control baselines, manual audit, external transfer evaluation, disclosed limitations |
| Paired answers crossing partitions | Group by `item_idx` (HaluEval) and `source_id` (RAGTruth); automated overlap assertions |
| Surface shortcuts (length, overlap) | Length-binned error analysis, ablations, and the claim-verification layer in the deployed system |
| Calibration degrades out of domain | Separate source/target calibration experiments; recalibrate and retune the threshold for the target domain |
| Explanation claims overreach | Measured reliability metrics, expert audit with disclosed provenance, no arbitrary pass thresholds |
| NLI model unavailable or slow | MiniLM fallback checkpoint, cached NLI outputs |
| Class imbalance | `scale_pos_weight`; PR-AUC reported alongside AUROC |
| Laptop VRAM pressure | Colab for heavy runs; one API worker, fp16, caching, no reload |
| External dataset licensing | Ship downloaders, hashes, citations, and license notes; never bundle restricted files |
| Calibrated score misleading on style | Evidence-domain display calibrator and per-claim verdicts lead; legacy score secondary |

---

## 15. Verified reference backbone

Foundations: Ribeiro et al. (LIME, KDD 2016), Lundberg & Lee (SHAP, NeurIPS
2017), Guo et al. (calibration, ICML 2017), Reimers & Gurevych (Sentence-BERT,
EMNLP-IJCNLP 2019), Chen & Guestrin (XGBoost, KDD 2016), Platt (1999),
Zadrozny & Elkan (isotonic, KDD 2002), DeBERTaV3 (ICLR 2023), SNLI (EMNLP
2015), MultiNLI (NAACL 2018), spaCy (Zenodo DOI 10.5281/zenodo.1212303).

Benchmarks and methods: HaluEval (EMNLP 2023), RAGTruth (ACL 2024), FaithBench
(NAACL 2025), HalluLens (ACL 2025), TruthfulQA (ACL 2022), FActScore (EMNLP
2023), SelfCheckGPT (EMNLP 2023), HaluAgent (EMNLP 2024), Luna (COLING 2025
Industry), Valentin et al. (cost-effective detection, arXiv:2407.21424).

Must-cite and differentiate: Haq et al. (Multimedia Systems 2026), Sundaragiri
et al. (IJERT 2026), Yadav & Verma (IEEE TAI 2026), Deng et al. (SpikeScore,
ICLR 2026).

Rules: verify every entry against ACL Anthology / arXiv / DOI before adding it;
these are the only references currently cited in the manuscripts.

---

## 16. Final verdict

Both phases succeeded on their own terms. The course phase produced a
paper-first applied ML project with a working demo, clean experiments,
statistical tests, SHAP explanations, and a polished dashboard. The publication
extension adds leakage-controlled cross-domain evaluation, calibration under
shift, measured explanation reliability, and a frozen reproducibility package.
The honest transfer failure and the length-confound analysis are features of
the work, not defects to hide: they define the limits of lightweight
benchmark-trained detectors and motivate the claim-level verification layer.

| Dimension | Score / 10 |
| --- | --- |
| Research quality | 8.5 |
| Compound novelty | 8 |
| Practical value | 9 |
| Reproducibility | 9 |
| Demo quality | 8 |
| Publication potential | 8 |

Pursue the journal submission after the submission-time checks in §12.2 and the
manual labeling steps in §13. Do not claim SOTA, guaranteed acceptance, or
human-reviewed explanation reliability.
