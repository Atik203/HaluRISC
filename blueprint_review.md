# Blueprint Review: HaluLens / EvidenceLens

**Reviewer Role:** Senior AI Researcher, Journal Reviewer (ACL/EMNLP/NeurIPS/ICML/ICLR/IEEE/ESWA)  
**Date:** July 2026  
**Document Reviewed:** `blueprint.md` (1672 lines, two-version architecture)  
**Review Type:** Pre-implementation verification, novelty assessment, feasibility analysis

---

## Table of Contents

1. [Verification Report](#1-verification-report)
2. [Novelty Assessment](#2-novelty-assessment)
3. [Version A Feasibility (8-Week Course)](#3-version-a-feasibility-8-week-course)
4. [Version B Feasibility (Q2 Publication)](#4-version-b-feasibility-q2-publication)
5. [Dataset Evaluation](#5-dataset-evaluation)
6. [Feature Engineering Assessment](#6-feature-engineering-assessment)
7. [ML Pipeline Assessment](#7-ml-pipeline-assessment)
8. [Evaluation Strategy Review](#8-evaluation-strategy-review)
9. [Paper Quality Review](#9-paper-quality-review)
10. [Software Architecture Review](#10-software-architecture-review)
11. [Literature Gap Analysis](#11-literature-gap-analysis)
12. [Critical Overlaps and Threats](#12-critical-overlaps-and-threats)
13. [Name Review](#13-name-review)
14. [Suggested Modifications to blueprint.md](#14-suggested-modifications-to-blueprintmd)
15. [Updated References (2024-2026)](#15-updated-references-2024-2026)
16. [Final Verdict](#16-final-verdict)

---

## 1. Verification Report

### Technical Claim Verification

| # | Claim | Status | Evidence |
|---|-------|--------|----------|
| 1 | HaluEval is the best primary dataset for the course | ⚠ Partial | HaluEval is well-known and large, but 85% of its 35K samples are ChatGPT-generated synthetic hallucinations. The remaining 5K are human-annotated. The binary classification task oversimplifies real-world detection. RAGTruth (ACL 2024, 18K natural responses with word-level spans) is methodologically stronger. The blueprint acknowledges artifacts but understates their severity. |
| 2 | HaluEval "large enough" and "directly related to hallucination evaluation" | ✔ Correct | 35,000 samples across QA/Dialogue/Summarization/General. Widely cited at EMNLP 2023. The synthetic generation pipeline is a known limitation but doesn't invalidate its use for a course project. |
| 3 | Feature groups (length, lexical, entity, numeric, hedging, semantic) are appropriate | ⚠ Partial | All six groups are individually validated in the literature. However, the IJERT framework (April 2026) already combines 5/6 of these (lexical overlap, entity coverage, semantic similarity, NLI contradiction, numeric consistency) into a single XGBoost pipeline. The hedging/uncertainty group is the only truly distinctive addition. Missing: NLI entailment/contradiction features (DeBERTa-based) are the **single most important signal** across all published feature importance analyses, yet the blueprint only mentions NLI as a Version B addition (B8). |
| 4 | XGBoost + Platt calibration is appropriate final model | ✔ Correct | XGBoost is standard for tabular features. Platt scaling is a well-understood calibration method. The AWS "Cost-Effective Hallucination Detection" paper (Valentin et al., 2024) validates that Platt/isotonisotonic calibration significantly improves risk-aware hallucination decisions. |
| 5 | SHAP TreeExplainer is appropriate for XGBoost | ✔ Correct | SHAP TreeExplainer is the standard method for tree-based models. |
| 6 | Black-box hallucination detection using classical ML is a valid approach | ✔ Correct | Multiple papers validate this: the IEEE Access comparative study (2024-25) using TF-IDF + standard classifiers; the IJERT framework (2026) using XGBoost on engineered features; "Blending Human and LLM Expertise" (2026) using XGBoost/LightGBM for mental health hallucination detection achieving 91.3% AUC. |
| 7 | No one has done SHAP for hallucination detection | ✘ Incorrect | A **direct competitor paper** titled "Quantifying Factual Divergence in Generative Models: SHAP-LIME Based Hallucination Score for LLMs" was published in **Multimedia Systems (Springer, Vol. 32, 2026)**. This paper uses token-level SHAP+LIME attribution with a quantitative Hallucination Score, tested on TruthfulQA and QAGS with GPT-3.5, LLaMA-2-13B, and Falcon-40B (F1=0.84, AUC=0.89, R²=0.84). This significantly overlaps with the proposed approach and was not addressed in the blueprint. |
| 8 | The project is "first" to combine calibration, SHAP, and classical ML for hallucination detection | ⚠ Partial | The IEEE Access "Hybrid Framework" (2026, DOI: 11346950) uses decoupled encoder-classifier with lightweight neural classifiers on HaluEval. The Multimedia Systems paper (2026) already combines SHAP+LIME+hallucination scoring. However, **no paper combines all five** (classical ML + SHAP + calibration + cross-domain + deployable artifact) in a single framework. This compound novelty gap is real but narrower than the blueprint implies. |
| 9 | Course success probability 8-8.5/10 | ⚠ Partial | The 8-week timeline is **tight but achievable**. The blueprint correctly identifies risks. However, feature extraction complexity (spaCy NER setup, sentence-transformer embeddings, NLI if added) is underestimated. The biggest hidden risk is dataset quality — HaluEval's synthetic nature may produce artificially clean results that collapse on real data. |
| 10 | Q2 publication potential 7/10 | ⚠ Partial | Realistic for IEEE Access (bar: moderate). Possible for Multimedia Systems (validated by similar paper). Borderline for Expert Systems with Applications (bar: higher, needs novel architecture or strong applied framing). The blueprint's self-assessment is roughly accurate. |
| 11 | The extended pipeline can remain calibrated under distribution shift | ⚠ Partial | The AWS Cost-Effective paper shows that multicalibration (per-cluster calibration via embedding-based grouping) is critical — raw Platt scaling degrades under distribution shift. The blueprint mentions "calibration under distribution shift" (B1) but doesn't specify multicalibration methodology. |
| 12 | "Lightweight" classical ML is 2-4 orders of magnitude cheaper than LLM-based methods | ✔ Correct | Classical ML detectors require ~0 LLM calls at inference (<1ms per sample) vs. 1-10+ LLM calls for self-consistency/judge methods (seconds per sample). CO2 and cost analysis is a legitimate differentiator. |

### Aggregate Assessment

- **Correct claims:** 5/12
- **Partially correct claims:** 6/12
- **Incorrect claims:** 1/12

The blueprint is **generally well-researched** but contains one significant factual error (SHAP novelty claim) and several areas where the literature has advanced beyond what the document accounts for.

---

## 2. Novelty Assessment

### Rating: 5/10 (Version A), 6.5/10 (Version B)

**Rationale:**

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Core Idea | 4/10 | Black-box hallucination detection using engineered features is already explored in IJERT (2026), IEEE Access (2024-25), and the Multimedia Systems SHAP+LIME paper (2026). Not fundamentally new. |
| Method Combination | 7/10 | The specific combination of (classical ML + SHAP + calibration + cross-domain + deployable artifact) has no complete published instance, though individual components have been published separately. |
| Feature Set | 5/10 | The proposed feature groups are standard in the literature. IJERT covers 5/6 of them. The hedging/uncertainty group is the distinguishing addition but is individually weak as a novelty signal. |
| Calibration Contribution | 6/10 | Calibrating classical ML for hallucination detection adds genuine practical value. The AWS framework (2024) does this for multi-scoring but not for classical ML classifiers. |
| Explanation Reliability | 8/10 | **Genuinely novel.** No published paper tests SHAP explanation fidelity or stability for hallucination detection. This is the strongest novelty contribution. |
| Cross-Domain Generalization | 7/10 | Cross-domain evaluation of classical ML hallucination detectors is rare. SpikeScore (ICLR 2026) does cross-domain but uses uncertainty fluctuation, not engineered features. |
| Deployable Artifact | 5/10 | Web dashboards for ML models are common. The integration with hallucination detection is a minor differentiator. |

### Verdict on Novelty

The project has **moderate overall novelty**, primarily driven by explanation reliability testing and cross-domain calibration — both underexplored. The individual components (XGBoost, SHAP, engineered features) are well-established. The blueprint correctly frames novelty as "balanced system design" rather than inventing hallucination detection. This is honest but the Multimedia Systems SHAP+LIME paper (2026) means the core pitch ("SHAP for hallucination detection") is no longer unique.

**Recommendation:** Shift the novelty framing from "SHAP-based explanations" to "explanation reliability and cross-domain calibration of lightweight detectors." This differentiates from the Multimedia Systems paper, which only visualizes SHAP without testing reliability.

---

## 3. Version A Feasibility (8-Week Course)

### Overall: 7.5/10 — Achievable with discipline

### Week-by-Week Feasibility Assessment

| Week | Blueprint Task | Feasibility | Risk Level | Notes |
|------|---------------|-------------|------------|-------|
| 1 | Literature + HaluEval setup | ✔ Feasible | Low | Dataset loading is straightforward. The risk is dataset format confusion — the blueprint correctly identifies this. |
| 2 | Feature extraction MVP | ⚠ Tight | Medium | Implementing 5+ feature groups (lexical, entity/NER, numeric, hedging, semantic embeddings) in one week is aggressive. Requires spaCy installation, sentence-transformer download (90MB+ model), and NER pipeline. The blueprint correctly offers fallback to regex-only. |
| 3 | Baselines and first results | ✔ Feasible | Low | Training LR/RF on tabular features is fast. |
| 4 | XGBoost, calibration, ablation | ✔ Feasible | Low-Medium | Standard sklearn/XGBoost workflow. |
| 5 | SHAP + error analysis | ✔ Feasible | Medium | SHAP TreeExplainer is fast for XGBoost. Manual error inspection (20 samples) is time-consuming but valuable. |
| 6 | Backend API | ✔ Feasible | Low | FastAPI is lightweight. Model serving via pickle/joblib is straightforward. |
| 7 | Frontend dashboard | ⚠ Tight | High | Building 4 pages (Analyze, Result, Explanation, Experiment Summary) with charts in 1 week is the biggest time risk. The blueprint correctly identifies this. |
| 8 | Paper + presentation | ⚠ Tight | High | Finalizing a journal-style paper AND preparing a presentation AND debugging the demo in 1 week is unrealistic. The paper should be written incrementally, not left to Week 8. |

### Critical Timeline Issues

1. **Paper writing is compressed.** The paper should be a parallel track starting Week 1, not a Week 8 sprint. The blueprint's "paper priority rule" (A13) acknowledges this implicitly but the timeline doesn't reflect it. Paper sections should be drafted weekly alongside implementation.

2. **Feature extraction Week 2 is underestimated.** The blueprint lists 5-6 feature groups for the course version. Implementing lexical overlap (easy), entity extraction (spaCy NER — moderate), semantic similarity (sentence-transformer — moderate), hedging regex (easy), and numeric consistency (easy) in one week while also documenting features is aggressive. A realistic split: core features in Week 2 (lexical, hedging, numeric, length), advanced features in Week 3 (entity NER, semantic embeddings).

3. **Frontend in 1 week is risky.** The React/TypeScript + charts + API integration for 4 pages assumes significant frontend experience. For an undergraduate, this is a 2-week minimum. **Recommendation:** Build a simplified 2-page frontend (Analyze + Results) and use static images for experiment results, then expand if time permits.

4. **No buffer week.** The 8-week schedule has zero slack. Any delay in weeks 1-3 cascades. Add an implicit buffer by treating Week 8 as overflow.

### Missing Components in Version A

1. **No logging/metrics infrastructure.** The architecture diagram shows `[(SQLite / JSON Demo Logs)]` but the timeline never allocates time for it. Either remove from the diagram or add a 2-hour task in Week 6.

2. **No testing strategy.** The "unit test on small examples" mitigation (A15) mentions testing but never allocates time. A basic test suite (`test_extract_features.py`, `test_api.py`) should be a Week 6 deliverable.

3. **No CI/CD or reproducibility scripts.** Even for a course project, a `requirements.txt` or `pyproject.toml` with pinned versions should be specified. The blueprint mentions reproducibility (A4 score 8/10) but doesn't prescribe version pinning.

### Simplification Recommendations

The blueprint's simplification rule (A6.1) is good. Additional simplifications:

- **Week 2:** If semantic similarity setup is slow, defer to Week 3-4 and use only lexical/hedging/numeric features initially.
- **Week 7:** Implement Analyze Page + Result Panel only. Make the Experiment Summary page a static image gallery, not interactive charts.
- **Week 8:** If presentation slides are incomplete, prioritize the paper. A good paper + recorded demo video is better than a mediocre paper + live demo.

### Version A Verdict

**Feasible at 7.5/10** with these adjustments:
1. Write paper sections weekly (not just Week 8).
2. Defer semantic embedding features to Week 3-4 if needed.
3. Reduce frontend scope to 2 pages + static experiment images.
4. Pin all Python dependencies by Week 2.

---

## 4. Version B Feasibility (Q2 Publication)

### Overall: 6.5/10 — Possible with significant effort and the right venue selection

### Publication Venue Analysis

| Venue | Tier | Feasibility | Required Contribution | Key Differentiator to Emphasize |
|-------|------|-------------|----------------------|--------------------------------|
| IEEE Access | Q2 (IF ~3.5) | **High (8/10)** | Comparative study, modular framework, 2-3 benchmarks | Cross-domain calibration + explanation reliability |
| Multimedia Systems (Springer) | Q2 (IF ~3.0) | **Medium-High (7/10)** | XAI-integrated framework | Builds on existing SHAP+LIME paper (2026) — must differentiate clearly |
| Expert Systems with Applications | Q1/Q2 (IF ~8.5) | **Borderline (5/10)** | Novel architecture or strong applied framing | Applied decision-support + cross-domain robustness |
| Applied Soft Computing | Q1/Q2 | **Borderline (5/10)** | Soft computing angle needed | Ensemble-based uncertainty quantification |
| Findings of ACL/EMNLP | Top NLP | **Low (3/10)** | Strong NLP contribution required | Not viable for this methodology |
| EACL/NAACL | Top NLP | **Low (3/10)** | Same as above | Same as above |

### Recommended Venue Strategy

1. **Primary target: IEEE Access.** The bar for the "Hybrid Framework for Hallucination Detection" (DOI: 11346950, 2026) is achievable for EvidenceLens. Cost: ~$1,950 APC (check current fees).

2. **Fallback target: Multimedia Systems.** The SHAP+LIME paper validates the approach for this venue. Risk: must clearly differentiate from that existing paper.

3. **Stretch target: Expert Systems with Applications.** Only if cross-domain results are exceptionally strong AND explanation reliability analysis is thorough AND the applied decision-support framing is compelling.

### What Must Be Added for Q2 Publication (Beyond Blueprint B)

1. **Multicalibration** (not just Platt/isotonisotonic). The AWS "Cost-Effective" paper shows that embedding-based cluster calibration dramatically improves reliability. This is a necessary upgrade from basic Platt scaling.

2. **At least one neural baseline** (not just classical ML). The blueprint mentions "affordable LLM-as-judge baseline on a subset" and "sentence-transformer + classifier" as optional. For Q2 publication, at least one of these is mandatory. Luna (COLING 2025) or a DeBERTa-based NLI classifier are good options.

3. **Statistical significance testing** is mentioned (B10: "bootstrap confidence intervals, paired significance tests where applicable") but should be mandatory, not optional. Use McNemar's test or paired bootstrap for comparing classifiers.

4. **Explanation faithfulness metric** must be operationalized. The blueprint says "feature removal consistency" and "perturb answer text slightly and observe explanation stability" but doesn't specify actual metrics. Use:
   - Feature Ablation Correlation (correlation between SHAP importance rank and actual feature removal impact)
   - Perturbation Stability Index (Jaccard similarity of top-K features under synonym/paraphrase perturbation)
   - These metrics have precedent in XAI evaluation literature and would satisfy reviewers.

5. **Ablation should include feature group interactions.** Not just "remove one group," but test 2-way and 3-way feature group combinations to show complementarity. This is standard practice in applied ML papers.

6. **Error taxonomy must be systematic.** The blueprint proposes manual inspection of 20 errors. For publication, this should be: (a) 100+ error inspection, (b) systematic taxonomy with inter-annotator agreement if multiple annotators, (c) quantitative error type distribution.

### Publication Risks (Beyond Blueprint B14)

| Risk | Severity | Mitigation |
|------|----------|------------|
| Multimedia Systems SHAP+LIME paper (2026) undermines novelty | **High** | Reframe around explanation *reliability* (not just *visualization*). Cite the paper as related work and differentiate clearly. |
| IJERT (2026) already combines 5/6 feature groups with XGBoost | **Medium** | Add hedging features + calibration analysis + cross-domain evaluation as differentiators. |
| OOD performance drops severely | **High** | This is expected and not disqualifying. Frame honestly as motivation for calibration and domain adaptation. |
| Reviewers dismiss classical ML as "not novel enough" | **Medium** | Emphasize efficiency (2-4 orders of magnitude cheaper than LLM methods), deployability, and interpretability as practical contributions. |
| 4-month post-course insufficient for publication extension | **Medium** | The 4-month post-course roadmap is optimistic for full publication preparation. Budget 6 months realistically. |

### Version B Verdict

**The Q2 publication plan is plausible but fragile.** The biggest threats are:
1. The Multimedia Systems SHAP+LIME paper (2026) partially preempts the novelty.
2. The IJERT framework (2026) pre-covers the feature engineering combination.
3. The 4-month post-course timeline is optimistic for adding datasets, calibration analysis, explanation reliability testing, and manuscript preparation.

With strong execution, clear differentiation from existing work, and realistic venue targeting (IEEE Access), the publication probability is approximately **60-65%**.

---

## 5. Dataset Evaluation

### HaluEval Assessment

**Strengths:**
- 35,000 samples, 4 task types (QA, Dialogue, Summarization, General)
- EMNLP 2023 publication, highly cited
- Direct hallucination labels (hallucinated / non-hallucinated)
- Widely used as a standard benchmark

**Weaknesses (more severe than blueprint acknowledges):**

1. **85% synthetic data.** 30K of 35K samples are ChatGPT-generated hallucinations via "sampling-then-filtering." These are engineered, not naturally occurring. Real-world hallucination patterns may differ systematically.

2. **Self-reinforcement risk.** ChatGPT generates and then filters the hallucinated samples. If ChatGPT's hallucination patterns are systematic (e.g., fabricates more for certain entity types), the benchmark may overfit to recognizing ChatGPT-specific artifacts rather than generalizable hallucination patterns.

3. **Binary task oversimplification.** "Is this answer hallucinated given the knowledge? Yes/No" ignores graded hallucination severity, partial hallucination, and ambiguity. Real-world deployment needs risk scores, not binary flags.

4. **Staleness.** Generated with early-2023 ChatGPT. RLHF, model architectures, and hallucination patterns have evolved significantly.

5. **No license.** The GitHub repository lacks a LICENSE file. This is a reproducibility concern.

6. **HalluLens (ACL 2025) critique.** Bang et al. explicitly note HaluEval conflates factuality and hallucination. The benchmark tests consistency with Wikipedia, not consistency with model training data, blurring the line between "factually wrong" and "hallucinated."

### Better Alternatives

| Dataset | Year | Venue | Size | Key Advantage | Recommendation |
|---------|------|-------|------|---------------|----------------|
| RAGTruth | 2024 | ACL | 18K | Natural responses, word-level spans, diverse LLMs | **Primary alternative for Version B** |
| HalluLens | 2025 | ACL | Dynamic | Clear intrinsic/extrinsic taxonomy, prevents leakage | **Best for publication evaluation** |
| FaithBench | 2025 | NAACL | Challenging | Human-annotated from 10 modern LLMs, detector-disagreement focus | **Strong supplementary benchmark** |
| HalluDial | 2024 | arXiv | 14.9K | Dialogue-level, spontaneous+induced hallucinations | **Alternative for dialogue domain** |
| FELM | 2023 | NeurIPS (D&B) | 847Q | Fine-grained segment labels, 5 domains | **Small but high-quality** |

### Recommended Dataset Strategy (Updated)

**Version A (Course):**
- Keep HaluEval as primary (it's large enough and easy to work with).
- **Mandatory addition:** Add a "Limitations" subsection in the paper explicitly discussing the synthetic generation, binary format, and staleness of HaluEval. Do not claim the model generalizes based on HaluEval results alone.
- **Optional:** Run a quick sanity check on 50 manually inspected HaluEval samples to confirm labels are reasonable.

**Version B (Publication):**
- **Train on HaluEval** (as in-domain).
- **Test on RAGTruth** (as primary out-of-domain evaluation). RAGTruth's word-level spans and diverse LLM sources make it a much stronger test of generalization.
- **Optional stress test:** FaithBench or HalluLens for challenging cases.
- **Remove TruthfulQA** from the primary plan. It has only 817 questions, is partially saturated, has incorrect gold answers (per HalluLens), and measures factuality, not hallucination. Keep it only as an optional discussion point.

---

## 6. Feature Engineering Assessment

### Feature Set Adequacy: 7/10

**Strengths:**
- All six groups (length, lexical overlap, entity, numeric, hedging, semantic similarity) are individually validated in published work.
- The blueprint correctly identifies high-importance groups (lexical overlap, semantic similarity, entity features).
- The cost/expected importance table (A8) is well-structured and honest.

**Critical Gap: Missing NLI Features**

NLI entailment/contradiction is the **single most important feature** across all published feature importance analyses. The IJERT framework (2026) shows NLI probability contributes *most strongly* to final predictions. The EGC paper (2026) recommends combining graph features with NLI. The AWS Cost-Effective paper uses NLI as a key scoring method.

The blueprint mentions NLI only as a **Version B addition** (B8: "NLI-based consistency"). This is a mistake. NLI should be included in **Version A** if at all feasible.

**Recommendation:** Add a lightweight NLI feature using `cross-encoder/nli-deberta-v3-base` (DeBERTa-based NLI model, HuggingFace, ~500MB, <50ms per pair). Extract:
- Entailment probability: p(answer entails context) and p(context entails answer)
- Contradiction probability: p(answer contradicts context)
- Neutral probability

If DeBERTa is too heavyweight for the course, use a distilled version (`cross-encoder/nli-MiniLM2-L6`) at <100MB.

**Existing Overlap to Acknowledge:**

The IJERT framework (April 2026) already combines: lexical overlap, entity coverage, semantic similarity, NLI contradiction score, and numeric consistency — 5 of the 6 proposed groups — with XGBoost. The blueprint should explicitly cite IJERT and differentiate by adding hedging features + calibration analysis + cross-domain evaluation.

### Updated Feature Priority (Version A)

| Priority | Feature Group | Rationale |
|----------|--------------|-----------|
| **Essential** | Lexical overlap (answer-context, answer-question) | Strongest baseline signal, zero cost |
| **Essential** | NLI entailment/contradiction (DeBERTa) | Most important individual feature in literature |
| **High** | Entity overlap (NER-based) | Catches fabricated entities |
| **High** | Semantic similarity (Sentence-BERT) | Captures semantic drift beyond exact words |
| **Medium** | Hedging/uncertainty markers | Distinguishing addition from IJERT |
| **Medium** | Numeric consistency | Fabricated numbers are common in biomedical/finance |
| **Low-Medium** | Length/style features | EMNLP 2025 found length heuristics can match sophisticated detectors — useful but double-edged |

---

## 7. ML Pipeline Assessment

### Overall: 8/10 — Sound and well-specified

**Strengths:**
- Clear train/val/test split strategy (70/15/15 with stratification).
- Correct model progression (heuristic → LR → RF → XGBoost → calibrated XGBoost).
- Appropriate calibration method (Platt scaling for XGBoost).
- Preprocessing steps are sensible (whitespace normalization, tokenization, label mapping).

**Issues to Address:**

1. **Class imbalance handling.** The blueprint doesn't mention whether HaluEval has balanced classes or how imbalance will be handled. If the QA subset has 70% non-hallucinated / 30% hallucinated, metrics like accuracy become misleading. The blueprint should specify: (a) class balance check before splitting, (b) use of `scale_pos_weight` in XGBoost if imbalanced.

2. **Feature scaling.** The blueprint doesn't specify whether features will be scaled/normalized. Logistic Regression requires feature scaling; XGBoost/RF do not. Since multiple models are compared, the pipeline should explicitly handle this (StandardScaler for LR, raw features for tree-based).

3. **Cross-validation is under-specified.** The blueprint says "5-fold cross-validation for model selection" as "better" but doesn't make it mandatory. For the course paper, 5-fold CV on the training set for hyperparameter selection is standard and should be required, not optional.

4. **Hyperparameter tuning scope.** The blueprint doesn't specify hyperparameter tuning at all. Even basic grid search or random search with 3-5 parameter combinations (max_depth, learning_rate, n_estimators, scale_pos_weight, subsample) would significantly improve experimental rigor. This is a 30-minute addition that dramatically improves the paper.

5. **Random seed discipline.** The blueprint mentions "one random seed" as minimum. Minimum should be **three seeds** with reported mean ± std. Single-seed results are increasingly criticized.

### Updated Pipeline Specification

```
1. Load HaluEval → Check class balance → Document
2. Feature extraction → Save raw features → Version pin
3. Stratified split (70/15/15, seed=42) → Save split indices
4. For each model:
   a. 5-fold CV on training set for hyperparameter selection (if applicable)
   b. Train on full training set with best parameters
   c. Evaluate on validation set for calibration
   d. Final evaluation on locked test set
5. Calibration: Platt scaling on validation set → Evaluate on test set
6. SHAP on test set predictions
7. Error analysis on test set errors
8. Repeat with seed=123, seed=456 for variance estimation
```

---

## 8. Evaluation Strategy Review

### Overall: 8/10 — Comprehensive and well-planned

**Strengths:**
- Good metric selection (Precision, Recall, F1, AUROC, PR-AUC, MCC).
- Calibration metrics included (Brier score, calibration curve, optional ECE).
- Efficiency metrics planned (latency, memory, cost).
- Ablation plan is structured (remove one group at a time).
- Error analysis is planned (manual inspection of false positives/negatives).
- Operational evaluation table (A10) is excellent — rarely seen in student papers.

**Missing or Under-specified:**

1. **No baseline comparison with existing systems.** The blueprint compares against heuristic, LR, RF, XGBoost — but these are all internal. For the course paper, at least one external comparison is needed:
   - SelfCheckGPT performance on the same HaluEval subset (the original paper or re-implemented)
   - Or a published baseline number from HaluEval benchmark leaderboard
   - Without this, readers cannot calibrate whether the achieved F1/AUROC is good or bad relative to the field.

2. **No inter-rater reliability for error analysis.** The manual inspection of 20 errors is good, but without a second annotator or at least a clear annotation protocol, it remains subjective. For the course: document the error taxonomy protocol with clear criteria.

3. **Confusion matrix should be normalized.** Both raw and row-normalized confusion matrices should be reported to show per-class accuracy in imbalanced settings.

4. **Cost analysis needs refinement.** The "approximate inference cost per 1,000 predictions" (A10) should include:
   - Feature extraction cost (spaCy NER time, sentence-transformer embedding time)
   - Total end-to-end cost (feature extraction + inference + SHAP)
   - Compare against estimated LLM-as-judge cost (e.g., GPT-3.5-turbo API cost for 1,000 evaluations)

### Recommended Additions

| Evaluation Component | Currently | Recommended |
|---------------------|-----------|--------------|
| External baseline | ✘ Missing | Add SelfCheckGPT performance on same subset (from literature or lightweight re-implementation) |
| Cross-validation | Optional | Mandatory 5-fold CV with 3 seeds |
| Hyperparameter tuning | Not specified | Grid/random search on 3-5 parameters, reported in appendix |
| Statistical tests | Not in Version A | Add McNemar's test comparing XGBoost vs best baseline |
| Error analysis scope | 20 samples | 20 samples is fine for course; document protocol clearly |
| Confusion matrix | Single | Add row-normalized version |

---

## 9. Paper Quality Review

### Section-by-Section Assessment

| Section | Present? | Quality | Issues |
|---------|----------|---------|--------|
| Title | ✔ | 8/10 | "HaluLens: A Calibrated and Explainable Machine Learning Framework for Hallucination Risk Prediction in Black-Box LLM Outputs" — good, descriptive, includes key terms. |
| Abstract outline | ✔ | 7/10 | Reasonable structure. Should include a concrete number ("achieved F1 of X.X on HaluEval QA") in the final version. |
| Introduction | ✔ | 7/10 | Motivation and problem statement are clear. Should include explicit contributions as a bulleted list. |
| Related Work | ✔ | 6/10 | Coverage is adequate for a course paper but doesn't engage with the most relevant recent work (IJERT 2026, Multimedia Systems SHAP paper 2026). See Section 11 below. |
| Methodology | ✔ | 8/10 | Well-structured: task definition → dataset → features → models → calibration → explanation. |
| Experiments | ✔ | 7/10 | Good structure. Missing: hyperparameter details, baseline comparison justification. |
| Web Application | ✔ | 6/10 | Present but risks being a "system description" section. Should be framed as "proof of deployability" rather than a separate contribution. |
| Results and Discussion | ✔ | 7/10 | Good to separate from Experiments. Error analysis section is strong. |
| Conclusion | ✔ | 7/10 | Standard structure. |

### Missing Sections

1. **Limitations (subsection within Discussion).** The blueprint discusses limitations in the context of risks but doesn't mandate a formal Limitations subsection. This is **required** for publication journals and highly recommended for a strong course paper. Must include: dataset artifacts, synthetic nature of HaluEval, binary label simplification, language limitation (English only), lack of multi-turn dialogue evaluation.

2. **Ethical Considerations.** No mention of ethical implications. Should discuss: (1) risk of over-reliance on automated hallucination detectors, (2) potential for false negatives creating false trust, (3) bias in hallucination detection across languages/dialects.

3. **Reproducibility Statement.** The blueprint mentions reproducibility (A4 score 8/10) but no formal reproducibility section. Required for most venues now.

4. **Broader Impact (optional for course).** Required for NeurIPS/ICML, recommended for others.

### Writing Quality Concerns

1. **Risk of overclaiming.** The blueprint correctly warns against this, but the actual paper must execute it. The most common student mistake is subtle overclaiming in the Introduction ("we propose a novel framework...") when the method is incremental.

2. **The "demo" trap.** The Web Application section (paper section 5) should demonstrate deployment feasibility, not be a feature walkthrough. Frame as: "To demonstrate practical deployability, we built a lightweight API and dashboard. The entire pipeline processes requests in Xms end-to-end."

3. **Screenshots in paper.** The blueprint recommends 5 screenshots (A12). At most 2-3. Screenshots should be small, clearly captioned, and directly referenced in the text.

---

## 10. Software Architecture Review

### Overall: 7/10 — Reasonable but can be simplified

### Stack Assessment

| Component | Choice | Assessment |
|-----------|--------|------------|
| Backend framework | FastAPI | ✔ Excellent choice. Fast, modern, built-in OpenAPI docs. |
| Data validation | Pydantic | ✔ Correct pairing with FastAPI. |
| Frontend framework | Next.js or React + TypeScript | ⚠ Next.js adds unnecessary complexity. React (Vite) + TypeScript is sufficient for a single-page dashboard. Next.js server-side features are irrelevant here. |
| CSS framework | Tailwind CSS | ✔ Good if known. |
| Charting library | Recharts or Plotly | ⚠ Recharts is React-native and simpler. Plotly.js is heavy. Recommend Recharts for the course. |
| Database | SQLite / JSON logs | ✔ Appropriate for course. |
| Model serving | pickle/joblib at startup | ✔ Correct for single-instance deployment. |

### Architecture Simplifications

1. **React (Vite) instead of Next.js.** Next.js adds SSR, file-based routing, and a build pipeline that is overkill for a dashboard with 2-4 static-ish pages. Vite + React + TypeScript is simpler and faster to develop. The blueprint should default to this.

2. **Model registry is over-specified for course.** The Version B architecture (B11) adds `Model Registry / Artifacts`. For the course, `model.pkl` in the repository is sufficient. Don't over-engineer.

3. **The architecture diagram (A11) is cleaner than (B11).** Version B's diagram adds Gateway, separate services, and model registry — this is microservice-style for a single-developer project. Keep the monolith (A11) even for Version B unless there is a specific scaling need.

### Missing Architecture Components

1. **Error handling.** No mention of what happens when the API receives malformed input, missing fields, or excessively long text. Add Pydantic validation with meaningful error messages.

2. **CORS configuration.** Mentioned (A11: "CORS enabled for frontend") but should specify: development (allow all origins), production (restrict to known origin).

3. **API rate limiting.** Not needed for course. Include a note that it would be needed for production.

4. **Model version in API response.** The `/predict` response (A11) should include `model_version` and `feature_version` fields for reproducibility.

### Updated API Contract

```json
{
  "risk_score": 0.18,
  "calibrated_probability": 0.21,
  "label": "low_risk",
  "thresholds": {"low": 0.3, "medium": 0.7, "high": 1.0},
  "top_features": [...],
  "latency_ms": 34,
  "model_version": "xgboost-v1.0",
  "feature_version": "course-v1.0",
  "warning": "This model was trained on HaluEval synthetic data. Results may not generalize to real-world LLM outputs."
}
```

---

## 11. Literature Gap Analysis

### What the Blueprint References (Verified Starter Reference List)

The blueprint's reference list (Section 6) is a good foundation but is **missing critical 2024-2026 papers** that directly compete with or contextualize the proposed work.

### Missing Critical References (Must Add)

| # | Paper | Venue | Year | Why It Must Be Cited |
|---|-------|-------|------|----------------------|
| 1 | **IJERT: "Combining Lexical, Entity, Semantic, and NLI Features with XGBoost for Hallucination Detection"** | arXiv / Preprint | Apr 2026 | Most similar feature combination. Pre-covers 5/6 proposed feature groups with XGBoost. Must differentiate. |
| 2 | **"Quantifying Factual Divergence in Generative Models: SHAP-LIME Based Hallucination Score for LLMs"** | Multimedia Systems (Springer), Vol. 32 | 2026 | Direct competitor: SHAP + LIME for hallucination detection. Must cite and differentiate. |
| 3 | **Valentin et al. "Cost-Effective Hallucination Detection for LLMs"** | arXiv (AWS AI Labs) | Aug 2024 | Calibration framework (Platt, isotonic, multicalibration, ECE) directly relevant to the proposed calibration pipeline. |
| 4 | **Belyi et al. "Luna: A Lightweight Evaluation Model to Catch LLM Hallucinations"** | COLING 2025 Industry Track | 2025 | Lightweight deployable detector. DeBERTa-large fine-tuned. 97%/91% cost/latency reduction. Baseline for efficiency comparison. |
| 5 | **Deng et al. "SpikeScore: Beyond In-Domain Detection for Cross-Domain Hallucination Detection"** | ICLR 2026 | 2026 | First major cross-domain hallucination detection paper at a top venue. Directly relevant to B's cross-domain framing. |
| 6 | **Cheng et al. "Small Agent Can Also Rock! Empowering Small Language Models as Hallucination Detector"** | EMNLP 2024 | 2024 | Small LLM (7B) as hallucination detector. Comparable to GPT-4. Baseline for cost/performance trade-off analysis. |
| 7 | **Kang et al. "Uncertainty Quantification for Hallucination Detection in LLMs"** | arXiv (USC) | Oct 2025 | Comprehensive UQ survey. Covers calibration, epistemic vs aleatoric uncertainty. Critical for calibrating the framing. |
| 8 | **HalluLens (Bang et al.)** | ACL 2025 | 2025 | New benchmark with intrinsic/extrinsic taxonomy. Explicitly critiques HaluEval and TruthfulQA. Must cite for dataset discussion. |
| 9 | **FaithBench (Sanchez et al.)** | NAACL 2025 | 2025 | Human-annotated from 10 modern LLMs. Shows SOTA detectors near 50% accuracy on challenging cases. |
| 10 | **RAGTruth (Niu et al.)** | ACL 2024 | 2024 | Stronger evidence-grounded dataset for Version B. Cited in blueprint but not in the reference list — must add. |

### What the Blueprint's References Get Right

The foundation references (LIME 2016, SHAP 2017, Guo/Calibration 2017, Sentence-BERT 2019, XGBoost 2016, FEVER 2018, TruthfulQA 2022, SelfCheckGPT 2023, FActScore 2023, HaluEval 2023) are all correct and well-chosen. These are the citations that reviewers expect to see.

---

## 12. Critical Overlaps and Threats

### Threat #1: Multimedia Systems SHAP+LIME Paper (2026)

**"Quantifying Factual Divergence in Generative Models: SHAP-LIME Based Hallucination Score for LLMs"**  
*Multimedia Systems, Springer, Vol. 32, 2026*

**Overlap:** Token-level SHAP+LIME attribution + custom Hallucination Score on TruthfulQA + QAGS with GPT-3.5, LLaMA-2-13B, Falcon-40B. F1=0.84, AUC=0.89, R²=0.84.

**Severity: HIGH.** This paper does essentially what HaluLens proposes — SHAP-based hallucination explanation for LLM outputs. The key differences:
- They use token-level SHAP (on LLM output tokens), HaluLens uses feature-level SHAP (on engineered features).
- They test on TruthfulQA/QAGS, HaluLens tests primarily on HaluEval.
- They don't do calibration or cross-domain analysis.
- Their SHAP is mostly visualized, not reliability-tested.

**Required Response:**
1. **Cite this paper prominently** in Related Work.
2. **Differentiate clearly:** "Unlike [citation], which applies token-level SHAP directly to LLM outputs, our work uses engineered evidence-consistency features, enabling explanation at semantically meaningful feature levels and calibration under domain shift."
3. **Emphasize what they don't do:** calibration analysis, cross-domain evaluation, explanation reliability testing, deployable API.

### Threat #2: IJERT Framework (2026)

**Feature combination:** Lexical overlap + entity coverage + semantic similarity + NLI contradiction + numeric consistency → XGBoost.

**Severity: MEDIUM.** This paper covers 5/6 of the proposed feature groups. The differentiation comes from: (1) hedging features (not covered), (2) calibration analysis (probably not covered), (3) cross-domain evaluation (probably not covered), (4) SHAP explanation (not covered), (5) deployable artifact (not covered).

### Threat #3: IEEE Access "Hybrid Framework" (2026)

**Decoupled encoder-classifier:** Frozen BERT/RoBERTa/DeBERTa as feature extractors + lightweight neural classifiers. Evaluated on PolyFEVER, FactCHD, HaluEval.

**Severity: LOW-MEDIUM.** Different approach (neural features vs engineered features). But both claim "lightweight" and "deployable." The differentiator is that HaluLens uses *interpretable engineered features* vs *frozen neural embeddings* — which is actually an advantage for explainability.

### How to Strengthen Against These Threats

The blueprint's strongest defense is the **compound contribution**:
1. Engineered features (different from token-level SHAP in Multimedia Systems)
2. Calibration analysis (different from raw XGBoost in IJERT)
3. Cross-domain evaluation (different from single-dataset IJERT/IEEE Access)
4. Explanation reliability testing (unique — no paper does this)
5. Deployable web artifact (unique — no paper provides this)

The paper should make this compound contribution explicit in the introduction: "While individual components have been explored in isolation, no work has combined engineered evidence-consistency features with calibrated risk estimation, explanation reliability analysis, cross-domain evaluation, and a deployable artifact in a single lightweight framework."

---

## 13. Name Review

### HaluLens

**Score: 7/10**

**Strengths:**
- Memorable and distinctive
- Short (3 syllables)
- Clearly evokes "hallucination lens"
- Good for a demo/product brand

**Weaknesses:**
- Slightly informal for a journal paper title
- "Lens" metaphor is not unique (HalluLens benchmark already exists at ACL 2025 by Bang et al. — **name collision!**)
- Doesn't communicate evidence-grounding or calibration
- May sound like a startup product, not a research contribution

### EvidenceLens

**Score: 8/10**

**Strengths:**
- More professional/academic tone
- Communicates evidence-grounding (core contribution)
- Doesn't collide with existing named systems
- Suitable for journal submission

**Weaknesses:**
- Less memorable than HaluLens
- Generic suffix "Lens" is slightly overused in ML naming

### Critical Issue: Name Collision

The **HalluLens** benchmark by Bang et al. (ACL 2025) is a different project — it's a hallucination evaluation benchmark, not a detection system. However, the name overlap could cause:
- SEO/Google Scholar confusion
- Reviewer confusion if not clearly disambiguated
- Citation confusion (someone searching "HalluLens" in Semantic Scholar may find the wrong paper)

**Recommendation:** Use **HaluLens** only for the course project (where name uniqueness is less critical) and **EvidenceLens** for the publication version. Add a footnote in the paper: "Not to be confused with the HalluLens hallucination evaluation benchmark [citation]."

### Alternative Names (Ranked)

1. **EvidenceLens** (8/10) — Professional, descriptive, no collisions
2. **HaluRISC** (8/10) — Hallucination Risk Scoring and Calibration. Acronym is memorable, communicates risk+calibration. More publication-appropriate.
3. **FactGauge** (7/10) — Concise, professional, suggests measurement
4. **TrustLens** (6/10) — Generic; "trust" is overused in ML naming
5. **VeriLens** (6/10) — Sounds like a verification tool, not hallucination-specific
6. **GroundScore** (5/10) — Too informal; "score" is weak
7. **HaluLens** (7/10) — Good for course, risk of confusion for publication

**Final recommendation:**
- Course: **HaluLens** (keep as-is, it's already established in the proposal)
- Publication: **HaluRISC** or **EvidenceLens**, with preference for **HaluRISC** (more distinctive, communicates risk+calibration in the acronym)

---

## 14. Suggested Modifications to blueprint.md

### Changelog

The following changes should be made to `blueprint.md`:

#### [MODIFY] Section: Core Decision → Recommended project name

**Current:** HaluLens for course, EvidenceLens for publication  
**Change:** Add HaluRISC as alternative publication name. Add note about HalluLens (ACL 2025) benchmark name collision.

#### [MODIFY] Section: A6 — Course Research Gap

**Current:** General gap statement  
**Add:** Explicitly acknowledge IJERT (2026) as most similar feature combination work, and differentiate by hedging features + calibration + cross-domain evaluation.

#### [MODIFY] Section: A7 — Dataset Plan

**Current:** HaluEval is the "best primary choice"  
**Change:** Add mandatory limitations section about HaluEval. Add note that 85% of data is synthetic ChatGPT-generated. Add recommendation to manually inspect 50 samples for label quality.

**Current:** TruthfulQA listed as optional  
**Change:** Downgrade TruthfulQA. Note: only 817 questions, partially saturated, measures factuality not hallucination. Use only for optional discussion.

#### [MODIFY] Section: A8 — Feature Engineering Plan

**Current:** Feature groups without NLI  
**Add:** NLI entailment/contradiction features using DeBERTa (`cross-encoder/nli-deberta-v3-base`) as a high-priority MVP feature. Add fallback smaller model recommendation (`cross-encoder/nli-MiniLM2-L6`).

**Add note:** The IJERT framework (April 2026) already combines 5/6 of these feature groups with XGBoost — cite and differentiate.

#### [MODIFY] Section: A9 — ML Pipeline

**Current:** "Better: 5-fold cross-validation for model selection" as optional  
**Change:** Make 5-fold CV mandatory for the paper.

**Add:** Hyperparameter tuning specification (grid/random search on 3-5 params, reported in appendix).

**Add:** Feature scaling note (StandardScaler for LR, raw features for tree-based models).

**Add:** Class imbalance handling (check balance, `scale_pos_weight` if needed).

**Current:** "one random seed" as minimum  
**Change:** Minimum 3 seeds with mean ± std reporting.

#### [MODIFY] Section: A10 — Evaluation Strategy

**Current:** Only internal baselines (heuristic, LR, RF, XGBoost)  
**Add:** Requirement for at least one external baseline comparison: SelfCheckGPT performance on the same HaluEval subset (from paper or lightweight re-implementation).

**Add:** McNemar's test for comparing XGBoost vs best baseline.

**Add:** Row-normalized confusion matrix in addition to raw.

**Add:** Cost comparison: estimated GPT-3.5-turbo API cost for same number of evaluations.

#### [MODIFY] Section: A11 — System Architecture

**Current:** "Next.js or React + TypeScript"  
**Change:** Default to React (Vite) + TypeScript. Next.js adds unnecessary complexity for a single-page dashboard.

**Current:** API response schema without model version  
**Add:** `model_version`, `feature_version`, and `warning` fields to the API response.

**Add:** Error handling and Pydantic validation specification.

#### [MODIFY] Section: A13 — Course Paper Blueprint

**Add:** Missing sections: Limitations (subsection in Discussion), Reproducibility Statement, optional Ethical Considerations.

**Add warning:** Do not use "novel" or "first" unless demonstrably true. Use "we propose," "we present," "we evaluate."

#### [MODIFY] Section: A14 — 8-Week Roadmap

**Change:** Week 2 split into Week 2 (core features: lexical, hedging, numeric, length) and Week 3 (advanced features: entity NER, semantic embeddings, NLI if included).

**Change:** Week 7: Reduce frontend scope to 2 pages (Analyze + Results) + static experiment image gallery.

**Change:** Week 8: Add explicit paper writing time allocation. Paper should be drafted incrementally starting Week 1.

**Add:** Paper writing as parallel track throughout all weeks:
- Week 1: Introduction draft
- Week 2-3: Related Work draft
- Week 4: Methodology draft
- Week 5-6: Experiments + Results draft
- Week 7: Discussion + Conclusion draft
- Week 8: Polish, proofread, finalize

#### [ADD] Section: A18 — Mandatory Reproducibility Checklist

```
1. Pinned requirements.txt or pyproject.toml with exact versions
2. Saved model artifact (model.pkl or model.joblib)
3. Saved train/val/test split indices or split script with fixed seed
4. Feature extraction script with version pin
5. README with exact reproduction steps (from pip install to running evaluation)
6. Saved SHAP explanation objects for paper figures
```

#### [MODIFY] Section: B7 — Extended Dataset Plan

**Current:** TruthfulQA as secondary dataset  
**Change:** Replace TruthfulQA with RAGTruth as primary secondary dataset. Mention HalluLens (ACL 2025) or FaithBench (NAACL 2025) as optional stress test datasets.

#### [MODIFY] Section: B8 — Extended Feature Engineering

**Add:** Evidence graph structural features (from EGC, 2026) as optional — note model-dependence caveat.

**Add:** Multicalibration via embedding-based clustering (from Valentin et al., 2024).

#### [MODIFY] Section: B10 — Publication Evaluation Strategy

**Add:** Explanation faithfulness metrics:
- Feature Ablation Correlation (correlation between SHAP rank and actual removal impact)
- Perturbation Stability Index (Jaccard of top-K features under paraphrase perturbation)

**Add:** Error taxonomy with quantitative breakdown (100+ errors, systematic categories).

**Make mandatory:** Statistical significance testing (McNemar or paired bootstrap).

#### [MODIFY] Section: Verified Starter Reference List (Section 6)

**Add:** 10 new references (see Section 15 below).

#### [MODIFY] Section: Final Recommendation / Overall Scores

**Current:** Version B novelty 6.5/10  
**Change:** Keep at 6.5/10 but add caveat about Multimedia Systems SHAP+LIME paper (2026) reducing novelty of SHAP-based hallucination detection.

#### [ADD] Section: Known Threats and Mitigations

Add a section documenting the three critical overlaps (Multimedia Systems SHAP+LIME paper, IJERT framework, IEEE Access Hybrid Framework) and how this project differentiates from each.

---

## 15. Updated References (2024-2026)

### New References to Add to blueprint.md

These must be added to the Verified Starter Reference List (Section 6).

### Explainability and Calibration

1. **Valentin, G., et al. (2024).** "Cost-Effective Hallucination Detection for LLMs." arXiv:2407.21424. AWS AI Labs. Calibration framework for hallucination scoring with Platt/isotonisotonic scaling, multicalibration, and ECE.

2. **Kang, M., et al. (2025).** "Uncertainty Quantification for Hallucination Detection in LLMs: Foundations, Methodology, and Future Directions." arXiv:2510.12040. USC. Comprehensive UQ survey covering calibration, epistemic/aleatoric uncertainty.

### Hallucination Detection Methods

3. **IJERT: "A Multi-Indicator Ensemble Framework for Hallucination Detection in LLMs."** (April 2026). Combines lexical overlap, entity coverage, semantic similarity, NLI contradiction, and numeric consistency with XGBoost. Most similar feature combination to HaluLens.

4. **Belyi, M., et al. (2025).** "Luna: A Lightweight Evaluation Model to Catch LLM Hallucinations." COLING 2025 Industry Track. DeBERTa-large fine-tuned for RAG hallucination detection. 97%/91% cost/latency reduction vs GPT-3.5. Lightweight deployable baseline.

5. **Cheng, X., et al. (2024).** "Small Agent Can Also Rock! Empowering Small Language Models as Hallucination Detector." EMNLP 2024. Small LLM (7B) as hallucination detector, comparable to GPT-4 with 2K tuning samples.

6. **Sriramanan, G., et al. (2024).** "LLM-Check: Investigating Detection of Hallucinations in LLMs." NeurIPS 2024. Uses internal hidden states and attention maps for detection. Single-response, 45-450x speedup.

### SHAP + Hallucination (Critical Overlap)

7. **[Author(s)]. (2026).** "Quantifying Factual Divergence in Generative Models: SHAP-LIME Based Hallucination Score for LLMs." Multimedia Systems, Springer, Vol. 32. Token-level SHAP+LIME for hallucination detection on TruthfulQA/QAGS. **Must cite and differentiate.**

### Cross-Domain Hallucination Detection

8. **Deng, Y., et al. (2026).** "SpikeScore: Beyond In-Domain Detection for Cross-Domain Hallucination Detection." ICLR 2026. Uncertainty fluctuation-based cross-domain detection across 4 LLMs and 6 benchmarks.

### Benchmarks and Datasets

9. **Bang, Y., et al. (2025).** "HalluLens: LLM Hallucination Benchmark." ACL 2025. New benchmark with intrinsic/extrinsic taxonomy and dynamic test generation. **Note: name collision with HaluLens project name.**

10. **Sanchez, P., et al. (2025).** "FaithBench: A Challenging Hallucination Benchmark." NAACL 2025. Human-annotated from 10 modern LLMs. SOTA detectors near 50% accuracy.

### IEEE Access Comparable Work (for Venue Targeting)

11. **"A Hybrid Framework for Hallucination Detection in Large Language Models."** IEEE Access, 2026. DOI: 11346950. Decoupled encoder-classifier with frozen BERT/RoBERTa/DeBERTa + lightweight neural classifiers. Evaluated on PolyFEVER, FactCHD, HaluEval.

12. **"Detecting Hallucinations in Large Language Models Using Machine Learning Classifiers: A Comparative Study."** IEEE Access, 2024-2025. DOI: 11013655. TF-IDF + standard ML classifiers. Demonstrates viability of classical ML for hallucination detection in IEEE Access.

---

## 16. Final Verdict

### Strengths

1. **Honest self-assessment.** The blueprint consistently avoids overclaiming and correctly identifies its own weaknesses. This is rare and commendable.

2. **Well-structured methodology.** The two-version approach, clear separation of concerns, and explicit "what not to do" sections (A16) are excellent project management.

3. **Comprehensive evaluation plan.** The operational evaluation table (A10), ablation design, error analysis plan, and calibration metrics show strong experimental thinking.

4. **Practical focus.** The emphasis on deployability, efficiency, and explainability — rather than chasing SOTA — makes the project defensible and realistically scoped.

5. **Good documentation ambition.** The plan for screenshots, demo scenarios, and reproducibility is thorough for a student project.

### Weaknesses

1. **Dataset naivety.** HaluEval is treated as strong-enough without sufficient critical discussion of its synthetic generation, binary format, and staleness. The field has moved toward RAGTruth, HalluLens, and FaithBench.

2. **Missing NLI features in Version A.** NLI is the single most important feature signal in the literature. Deferring it to Version B is a mistake.

3. **Undiscovered critical overlap.** The blueprint was written without awareness of the Multimedia Systems SHAP+LIME paper (2026), which directly competes on precisely the novelty claim. This is the most significant weakness.

4. **Timeline compression.** The paper + frontend + demo rehearsal all compressed into Week 7-8 is unrealistic. Paper writing should be parallel throughout.

5. **No statistical testing specification.** Version A needs at least McNemar's test for comparing classifiers. Single-seed reporting is insufficient.

6. **No hyperparameter tuning plan.** Even basic grid search would significantly improve experimental rigor.

### Major Risks

| Risk | Severity | Probability | Impact |
|------|----------|-------------|--------|
| Multimedia Systems paper undermines SHAP novelty | Critical | Already realized | Forces reframing of contribution |
| HaluEval artifacts produce artificially good results | High | Medium-High | Collapses on external evaluation |
| Feature extraction complexity delays Week 2-4 | High | Medium | Cascading timeline failure |
| Frontend over-engineering delays paper | Medium | High | Paper quality suffers |
| Q2 reviewers reject "just XGBoost + SHAP" | Medium | Medium | Publication rejected |
| Name collision with HalluLens benchmark | Low-Medium | Already realized | Confusion in literature search |

### Missing Work

1. **HaluEval label quality audit.** Before building the pipeline, manually inspect 50-100 samples to verify label correctness. This is 1-2 hours that prevents weeks of debugging.

2. **External baseline comparison.** Without at least one external reference point (SelfCheckGPT or published HaluEval baseline), the results are uncalibrated — readers can't tell if F1=0.75 is good or bad.

3. **NLI feature integration.** Should be in Version A, not deferred to B. Even a lightweight distilled NLI model adds significant signal.

4. **Hyperparameter sweep.** 30 minutes of work that dramatically improves experimental credibility.

5. **Reproducibility package.** The blueprint mentions reproducibility but doesn't prescribe a checklist. Add one.

### Publication Potential

| Venue | Probability | Conditions |
|-------|-------------|------------|
| IEEE Access | **65-75%** | Cross-domain evaluation, calibration analysis, external baseline, 3-seed reporting |
| Multimedia Systems | **50-60%** | Must clearly differentiate from existing SHAP+LIME paper (2026) |
| Expert Systems with Applications | **40-50%** | Needs novel architecture or exceptionally strong applied framing |
| ACL/EMNLP Findings | **15-20%** | Unlikely without stronger NLP contribution |
| Other Q2 (Applied Soft Computing, Neural Computing & Applications) | **50-60%** | Venue-dependent framing |

### Course Success Probability

**7.5/10** (adjusted from blueprint's 8/10)

The 0.5-point downgrade reflects:
- Underestimated feature extraction complexity
- Compressed paper writing timeline
- No buffer for dataset/Docker/integration issues

**With blueprint adjustments applied: 8.5/10**

### Overall Score

| Dimension | Blueprint Self-Score | Reviewer Score | Delta | Notes |
|-----------|---------------------|----------------|-------|-------|
| Novelty (Version A) | 7/10 | 5/10 | -2 | SHAP claim weakened by Multimedia Systems paper |
| Novelty (Version B) | 6.5/10 | 6.5/10 | 0 | Accurate self-assessment |
| Research Quality | 7-7.5/10 | 7/10 | 0 | Good foundation, needs NLI+stats |
| Feasibility (Version A) | 8/10 | 7.5/10 | -0.5 | Minor timeline compression issues |
| Publication Potential (Version B) | 7/10 | 6.5/10 | -0.5 | Competitive field, critical overlaps |
| Demo Quality | 8/10 | 7/10 | -1 | Frontend timeline is optimistic |
| Reproducibility | 8/10 | 7/10 | -1 | Needs explicit checklist and version pinning |
| **Overall** | **8/10 (A), 7/10 (B)** | **7.5/10 (A), 7/10 (B)** | **-0.5 (A), 0 (B)** | Blueprint is generally honest |

### Bottom Line

The blueprint is **well-researched and honestly scoped** for an undergraduate ML project. Version A is achievable with minor timeline adjustments and the addition of NLI features. Version B is a plausible Q2 publication target if differentiated clearly from the Multimedia Systems SHAP+LIME paper and the IJERT framework.

The single biggest action item is **addressing the critical overlap with existing SHAP hallucination detection work** by reframing the novelty around explanation reliability, cross-domain calibration, and the deployable artifact — not SHAP visualization alone.

**The project should proceed.** With the modifications documented in this review, both versions become stronger and more defensible.

---

*End of Review*

*Reviewer signature: Senior AI Researcher, July 2026*
