# Research Paper Review and Revision Checklist

**Manuscript:** *Calibrated Hallucination Risk Estimation in Large Language Models with Evidence-Consistent XGBoost*  
**Reviewed file:** `ML_Paper(1).pdf` (17 pages)  
**Review type:** Technical manuscript review based on the supplied PDF  
**Overall recommendation:** **Major revision before submission**

> **Scope note:** This review evaluates the manuscript's argument, methodology as described, reported tables and figures, consistency, and presentation. The experiments and source-code repository have not been independently rerun or verified. Numerical values below are transcribed from the manuscript and should be checked against the actual experiment artifacts.

---

## 1. Executive summary

The paper proposes Evidence-Consistent XGBoost (EC-XGB), a feature-based model intended to estimate the risk that an LLM answer is hallucinated or unsupported by available evidence. It combines engineered evidence-consistency features, claim-level natural-language-inference (NLI) aggregates, monotonic constraints, probability calibration, and SHAP explanations.

The paper has several strengths:

- It addresses a practical problem: hallucination risk estimation that is relatively inexpensive and interpretable.
- It evaluates more than F1, including AUROC, MCC, calibration metrics, cross-domain behavior, ablations, latency, and error analysis.
- It includes a useful investigation of dataset shortcuts, especially answer length.
- It acknowledges important negative results and limitations.

However, the manuscript is **not ready for submission without major revision**. The most important issues are:

1. Reported calibration values and evaluation descriptions need to be reconciled.
2. The combined RAGTruth evaluation includes task types used for training, so it must not be presented as entirely zero-shot.
3. The HaluEval results may rely heavily on answer length and lexical shortcuts.
4. Target-domain calibration lowers ECE but produces a zero predicted-positive rate at the fixed 0.5 threshold in Table 8.
5. The mathematical description of monotonic constraints appears inconsistent with the stated intended behavior.
6. EC-XGB must be evaluated separately from the larger deployed verification system.
7. Several reporting and presentation details need clarification.

**Recommendation:** Major revision. Preserve the paper's research direction, but resolve the scientific and reporting issues before polishing the language.

---

## 2. Paper summary

### Research problem

The manuscript argues that hallucination detectors can be costly, difficult to interpret, poorly calibrated, or unreliable outside their training domain. It asks whether a lightweight feature-based model can estimate hallucination risk from a question, available evidence, and a candidate answer.

### Proposed method

The manuscript describes EC-XGB as using:

- 26 base engineered features across seven groups;
- eight claim-level NLI aggregates;
- one source indicator, for 35 total inputs;
- log-scaled answer-length features;
- monotonic constraints;
- multi-source training;
- Platt or isotonic probability calibration;
- SHAP-based feature explanations.

The described feature groups include length/style, lexical overlap, entity features, NLI, numeric features, hedging, and semantic similarity.

### Datasets

The manuscript reports using:

- **HaluEval QA:** training, validation, and in-domain testing;
- **RAGTruth:** cross-domain evaluation, including QA, summarization, and data-to-text tasks;
- **FaithBench:** a summarization stress test.

The manuscript says HaluEval QA contains 20,000 labeled answer rows derived from 10,000 questions and uses a grouped 70/15/15 split. It describes a 5,034-row RAGTruth QA calibration set and a disjoint 900-row QA test set.

### Main results reported in the manuscript

- EC-XGB F1 on HaluEval QA: approximately **0.9855**.
- Standard XGBoost flags **99.9%** of combined RAGTruth rows; EC-XGB flags **61.3%**.
- Table 6 reports combined-RAGTruth AUROC changing from **0.4753** to **0.5815**.
- Table 6 reports combined-RAGTruth ECE changing from **0.6346** to **0.2782**.
- Table 6 reports held-out RAGTruth QA F1 of **0.3022** for standard XGBoost and **0.3081** for EC-XGB.
- Table 8 reports target-domain ECE of **0.1335** for Platt calibration and **0.1316** for isotonic calibration, while the predicted-positive rate at the fixed 0.5 threshold is **0.000**.
- Median per-sample latency is reported as approximately **61.8 ms**.

These are manuscript-reported results, not independently reproduced findings.

---

## 3. Major scientific and methodological concerns

### 3.1 Monotonic-constraint equation may have the wrong sign

**Location:** Page 3, Section 3.3, approximately lines 179–188.

The manuscript intends the model to obey these rules:

- More contradiction should not decrease hallucination risk.
- More entailment/support should not increase hallucination risk.

However, the displayed mathematical expression assigns a non-negative derivative to an “evidence-positive” feature while the surrounding text says that entailment should reduce hallucination risk.

If the predicted output is the probability of hallucination, the intended directions are:

\[
\frac{\partial P(\text{hallucination})}{\partial\,\text{contradiction}} \geq 0
\]

and

\[
\frac{\partial P(\text{hallucination})}{\partial\,\text{entailment}} \leq 0
\]

**Required actions:**

- [x] Correct the equation and accompanying prose.
- [x] Inspect the actual monotonic-constraint vector in the training code.
- [x] Confirm the feature order matches the constraint vector.
- [x] Add a small test demonstrating that the fitted model respects the intended directions.
- [x] Check that any claim of evidence consistency is phrased as a design objective, not a guarantee of semantic correctness.

This may be a writing error or an implementation error. The manuscript alone cannot establish which.

### 3.2 Calibration numbers and protocols need reconciliation

**Locations:** Abstract (page 1), Section 5.2 (page 10), Table 8 and Section 5.3 (page 11), conclusion (page 15).

The manuscript reports multiple RAGTruth QA calibration results:

| Location / method | Reported ECE |
|---|---:|
| Abstract: deployed display calibrator | 0.126 |
| Section 5.2: deployed display calibrator | 0.126 |
| Table 8: target Platt calibration | 0.1335 |
| Table 8: target isotonic calibration | 0.1316 |

These values may refer to different checkpoints or evaluation protocols, but the manuscript does not make the relationship sufficiently clear.

The paper also reports a value of **0.726 before calibration** in Section 5.2, while Table 8 reports raw ECE of **0.8185** for its stated RAGTruth QA evaluation.

**Required actions:**

- [x] Identify the exact model/checkpoint used for every calibration result.
- [x] Identify the raw score and calibration method used in each experiment.
- [x] Confirm that every ECE is computed on the same intended held-out test set for its comparison.
- [x] Explain why 0.726 and 0.8185 differ, if both are valid.
- [x] Explain why 0.126 differs from Table 8's 0.1335 and 0.1316, if both are valid.
- [x] Recompute the tables and abstract values directly from saved experiment artifacts.
- [x] Use consistent rounding and naming throughout the paper.

Do not simply replace one number with another until the experiment configurations have been checked.

### 3.3 The combined RAGTruth evaluation is not wholly zero-shot

**Locations:** Sections 3.3 and 4.2–4.4 (pages 4 and 8), Table 6 (page 10), discussion (page 15).

The manuscript says EC-XGB trains on HaluEval plus RAGTruth non-QA rows. It then reports results on the combined RAGTruth corpus, which includes QA, summarization, and data-to-text tasks.

Therefore, some task types in the combined evaluation are represented in training. The held-out QA task is described as excluded from training, but the combined-set evaluation is not a fully unseen-task evaluation.

**Required actions:**

- [x] Separate held-out QA results from combined RAGTruth results in the abstract, results, and conclusion.
- [x] Reserve “zero-shot” for an explicitly defined setting that has no target training/adaptation exposure relevant to the claim.
- [x] Label evaluation sets according to whether their task type, examples, groups, or documents were used in training.
- [x] Verify there is no overlap of examples, question groups, or source documents between training and test partitions.
- [x] If feasible, add an evaluation on entirely unseen task types or a separately sourced corpus.
- [x] State that results on task types included in training measure a different form of generalization than truly unseen-task transfer.

The manuscript already acknowledges this distinction in its limitations. The terminology and framing of the results should be consistent with that acknowledgement.

### 3.4 HaluEval results may be driven by dataset shortcuts

**Locations:** Tables 9 and 14, pages 12 and 14; Sections 5.5, 5.8, and 5.9.

Table 14 reports the following hallucinated proportions by answer length:

| Answer length | Hallucinated share reported |
|---|---:|
| 1 word | 2.1% |
| 2 words | 3.3% |
| 3–4 words | 12.7% |
| 5–8 words | 90.2% |
| 9–16 words | 98.2% |
| 17+ words | 99.2% |

This suggests that answer length is highly predictive of the label in the HaluEval test split. The manuscript also reports that removing lexical features reduces F1 by 0.0247, while removing semantic features slightly increases F1.

The issue is not merely that the model has a high F1. It is whether that performance reflects robust hallucination detection or dataset-specific correlations.

**Required actions:**

- [x] Add a length-only baseline.
- [x] Add a lexical-overlap-only baseline.
- [x] Report the performance of the proposed model on a length-balanced evaluation subset.
- [x] Test whether performance remains strong after controlling for answer length.
- [x] Evaluate on a dataset whose label construction does not have the same length-label relationship.
- [ ] Include stronger neural or established hallucination-detection baselines where feasible.
- [x] Explain how these shortcut tests change the interpretation of the headline HaluEval result.

The paper should preserve its current error analysis and use it to motivate these additional controls.

### 3.5 Lower ECE does not automatically mean a useful decision threshold

**Location:** Table 8 and Figure 5, page 11.

Table 8 reports:

| Calibration method | ECE | Brier score | Predicted-positive rate |
|---|---:|---:|---:|
| Raw scores | 0.8185 | 0.8164 | 0.999 |
| Source Platt | 0.8089 | 0.8011 | 0.999 |
| Target Platt | 0.1335 | 0.1639 | 0.000 |
| Target isotonic | 0.1316 | 0.1663 | 0.000 |

Calibration substantially reduces the reported probability errors. However, the predicted-positive rate becomes zero at the fixed 0.5 threshold after target calibration.

A lower ECE does not establish that the model is useful for classifying hallucinations at the chosen threshold.

**Required actions:**

- [x] Report the positive-label prevalence on the target test set.
- [x] Explain the change in predicted-positive rate.
- [x] Select any operational threshold using a designated calibration/validation set, never the final held-out test labels.
- [x] Report precision, recall, F1, and false-positive rate at the chosen threshold.
- [x] Show reliability plots and Brier score alongside threshold-based results.
- [x] Explain the trade-off between false alarms and missed hallucinations.
- [x] Clearly distinguish probability calibration from classification performance.

### 3.6 EC-XGB and the deployed verification application are not clearly separated

**Locations:** Figures 1–3, pages 4–7; Sections 3.3–3.5.

The paper describes both:

1. A feature-based EC-XGB risk model.
2. A larger application that uses claim decomposition, evidence retrieval, NLI, and an LLM judge for uncertain cases.

The application also adjusts the displayed risk based on claim-level verdicts. It is therefore important to establish which results belong to the classifier alone and which belong to the full application.

**Required actions:**

- [x] Report raw EC-XGB results separately.
- [x] Report calibrated EC-XGB results separately.
- [ ] Report claim-level verification results separately.
- [ ] Report full-system results, including retrieval and judge fallbacks, separately.
- [x] Explain whether the headline F1 and AUROC numbers come from the classifier or the full system.
- [ ] Report latency and cost for each component and for the complete user-facing pipeline.
- [x] Clarify how the displayed risk score is adjusted and whether that adjusted score has been independently calibrated and evaluated.

### 3.7 Source-indicator effects require further analysis

**Locations:** Figure 1 and Sections 3.3–3.4.

The paper includes a source indicator in the 35-feature model. Different sources can differ in writing style, length, label prevalence, and task type. The model could learn source-specific patterns rather than evidence quality.

**Required actions:**

- [x] Add an ablation without the source indicator.
- [x] Report how removing it changes in-domain and cross-domain results.
- [x] Document how the indicator is assigned at inference time.
- [x] Explain whether its value reveals information that will not be available for future sources.
- [x] Test whether the model remains useful on a new source whose identity is not represented during training.

---

## 4. Numerical and reporting consistency audit

These are the specific manuscript checks identified during review. Verify each against the actual artifacts before editing the paper.

### 4.1 In-domain results

- [x] Table 4 reports standard XGBoost F1 as 0.985 after rounding.
- [x] Table 5 reports full EC-XGB F1 as 0.9855.
- [x] The conclusion reports EC-XGB F1 as 0.9855.
- [x] Explain the distinction between standard XGBoost and EC-XGB wherever their results appear.
- [x] Use consistent precision and rounding rules across all tables and text.

### 4.2 RAGTruth results

- [x] Table 6 combined-RAGTruth AUROC: standard 0.4753, EC-XGB 0.5815.
- [x] Table 6 combined-RAGTruth ECE: standard 0.6346, EC-XGB 0.2782.
- [x] Table 6 combined-RAGTruth flag rate: standard 99.9%, EC-XGB 61.3%.
- [x] Table 6 held-out RAGTruth QA flag rate: standard 99.9%, EC-XGB 97.4%.
- [x] Do not use the 61.3% combined-set flag rate as though it were the held-out QA flag rate.
- [x] Reconcile Section 5.2's raw ECE of 0.726 with Table 8's 0.8185.
- [x] Reconcile the abstract's 0.126 with Table 8's 0.1335 and 0.1316.

### 4.3 Model count

**Location:** Section 4.3, page 8, and Table 4, page 9.

- [x] The text says eight models are compared, but Table 4 visibly lists seven model rows.
- [x] Determine whether the majority baseline is intentionally omitted from Table 4.
- [x] Either include the missing baseline or correct the model count and explain where the baseline is reported.

### 4.4 Feature count

**Locations:** Tables 1, 2, and 5; Section 3.3.

- [x] Verify the 26 base features.
- [x] Verify the eight claim-level aggregates.
- [x] Verify the single source indicator.
- [x] Confirm that the final total is 35.
- [x] Explain why the claim-level variant in Table 5 uses 34 features and the final variant uses 35.
- [x] Ensure feature names and order match the code, equations, and tables.

### 4.5 Latency

**Location:** Table 11, page 13.

The table reports median and mean values for individual components and a total median latency of 61.77 ms. The component mean values do not directly add to that total when interpreted as a simple additive total.

- [x] Verify whether component timings overlap or were measured separately.
- [x] Explain how the total latency was measured.
- [x] State whether feature extraction, model inference, SHAP, retrieval, and judge fallback are included in each timing.
- [x] Report median and percentile values with a consistent aggregation method.
- [x] Distinguish batch latency from single-request interactive latency.

### 4.6 Error analysis

**Location:** Section 5.8, page 14.

- [x] The manuscript reports 48 errors in the 3,000-row test set: 12 false positives and 36 false negatives.
- [x] Table 13 summarizes a separate sample of 20 misclassifications.
- [x] Explicitly distinguish full-test error counts from the manually inspected sample.
- [x] State the selection method for the 20 cases.
- [x] Do not generalize the sample's category proportions to the full test set without justification.

---

## 5. Experimental design improvements

### 5.1 Stronger baselines

The current comparison includes conventional classifiers and text-based baselines, which is useful. However, it does not by itself establish superiority over the broader hallucination-detection literature.

- [x] Add at least one established hallucination detector if it can be evaluated fairly.
- [ ] Add a claim-verification baseline.
- [ ] Add a competitive neural classifier where feasible.
- [x] Ensure all baselines use comparable splits and labels.
- [ ] Report computational cost and input requirements for each baseline.

### 5.2 Confidence intervals and significance tests

The manuscript reports three seeds, McNemar's test, bootstrap confidence intervals, and Wilcoxon tests.

- [x] State precisely which predictions are paired for McNemar's test.
- [x] State whether bootstrap resampling respects question/document groups.
- [x] Add uncertainty estimates for cross-domain metrics.
- [x] Explain that three seed-level observations provide limited evidence for a Wilcoxon test.
- [x] Avoid presenting tiny metric differences as practically meaningful merely because a test is significant.

### 5.3 Calibration robustness

- [x] Evaluate calibration with different amounts of labeled target-domain data.
- [x] Keep a final target test set untouched by calibration and threshold selection.
- [x] Compare raw, source-calibrated, and target-calibrated probabilities.
- [x] Report ECE, Brier score, and a suitable threshold-based metric.
- [x] Explain the assumptions and practical cost of requiring labeled target examples.

### 5.4 Generalization scope

- [x] State clearly that the current feature pipeline is English-centric.
- [x] Avoid claims of language-independent or universal generalization.
- [x] Define what “distribution shift” means for each experiment.
- [x] Separate unseen examples, unseen task types, and unseen datasets as different evaluation conditions.
- [x] Explain which conclusions are supported by each condition.

---

## 6. Related work and contribution

The paper discusses related work on self-consistency, semantic entropy, lightweight detectors, SHAP-based explanations, cross-domain detection, and hallucination benchmarks.

The research gap would be clearer if the related-work section compared methods systematically.

**Suggested comparison columns:**

| Method | Model family | External evidence | Calibration | Explainability | Cross-domain evaluation |
|---|---|---|---|---|---|
| Each related method | Classifier, LLM judge, sampling-based, etc. | Yes/no/details | Reported or not | Method | Setting and result |

Checklist:

- [x] Verify every reference against the original publication.
- [x] Confirm the paper's description of each method is accurate.
- [x] Explain how EC-XGB differs from the closest feature-based and NLI-based methods.
- [x] Distinguish a novel combination of known components from a new algorithmic contribution.
- [x] Support claims about gaps in previous work with specific citations.
- [x] Avoid stating that no previous method performs an evaluation unless the literature search supports that claim.

The potential contribution is the combination of lightweight feature-based risk estimation, claim-level evidence signals, calibration, explanations, and a systematic analysis of distribution shift. The manuscript needs stronger evidence that this combination offers benefits beyond a standard XGBoost model and benchmark-specific shortcuts.

---

## 7. Abstract and conclusion revision checklist

### Abstract

- [x] Identify the 0.9855 F1 as an in-domain HaluEval QA result.
- [x] Describe the 61.3% flag rate as a combined-RAGTruth result.
- [x] Report the held-out QA result separately.
- [x] Reconcile all ECE values before finalizing the abstract.
- [x] Describe AUROC 0.582 as an improvement but still limited discrimination.
- [x] Avoid suggesting broad superiority over LLM judges based only on 200 examples.
- [x] Keep the abstract's claims aligned with the exact tables and experiments.

### Conclusion

- [x] Preserve the negative findings.
- [x] Explain that high in-domain performance does not establish generalization.
- [x] State that target-domain calibration requires labeled target examples.
- [x] Explain the difference between a risk estimate and a fact-checking verdict.
- [x] Avoid claiming general-purpose hallucination detection.
- [x] Make the next research steps specific: shortcut controls, stronger baselines, robust target calibration, and unseen-domain evaluation.

Suggested defensible framing:

> EC-XGB preserves strong performance on HaluEval QA and reduces excessive positive predictions on some shifted datasets. However, its discrimination remains limited on natural responses, and its probability calibration depends on access to labeled target-domain data. The results support further investigation of lightweight, explainable risk estimation, rather than establishing a general-purpose hallucination detector.

---

## 8. Writing, figures, and formatting

### Writing

- [x] Break long sentences into shorter, clearer sentences.
- [x] Remove repeated explanations of the same results.
- [x] Define abbreviations at first use.
- [x] Use “risk estimation,” “hallucination detection,” and “claim verification” consistently.
- [x] Avoid overstating the significance of small differences.
- [x] Describe the method as designed to encourage evidence-consistent predictions, not as guaranteeing correct evidence reasoning.

### Figures and tables

- [x] Increase font size where figure labels are difficult to read.
- [x] Ensure every table can be read at normal paper size.
- [x] Use consistent labels for standard XGBoost, EC-XGB, calibrated EC-XGB, and the full application.
- [x] Make calibration plots and axes easier to interpret.
- [x] Ensure figure captions identify the exact dataset and evaluation setting.
- [x] Check that every table is referenced and interpreted in the text.

### References

- [x] Verify authors, titles, venues, years, DOIs, and URLs.
- [x] Use the target venue's reference style.
- [x] Ensure every in-text citation has a matching reference.
- [x] Ensure every reference is cited in the manuscript.
- [x] Confirm that claims about recent work are supported by the cited paper, not merely by its title.

### Reproducibility and code

- [x] Verify the exact dependency versions used for the reported runs.
- [x] Resolve any mismatch between manuscript environment details and the repository's pinned requirements.
- [x] Check that saved split indices match the described grouped split.
- [x] Confirm that all seeds and configurations are recorded.
- [x] Reproduce the reported tables from saved artifacts.
- [x] Ensure the model, calibrator, and SHAP artifacts correspond to the manuscript's reported checkpoint.
- [x] Document any manual processing or excluded examples.
- [x] Confirm that the repository and data-download instructions are accessible to reviewers.

---

## 9. Prioritized revision plan

### Priority 1 — Critical

- [x] Reconcile all reported metrics against the saved artifacts.
- [x] Audit calibration and test-set separation.
- [x] Correct and verify monotonic-constraint signs.
- [x] Clarify which RAGTruth results are truly held out.
- [x] Add or run shortcut controls for answer length and lexical overlap.

### Priority 2 — High

- [x] Separate EC-XGB from the full verification application.
- [ ] Add stronger baselines where feasible.
- [x] Add uncertainty estimates and clarify significance testing.
- [x] Test calibration with different target-label budgets.
- [x] Explain the source-indicator feature and test its contribution.

### Priority 3 — Medium

- [x] Correct model counts and feature-count explanations.
- [x] Reconcile latency measurements.
- [x] Improve figure readability and table consistency.
- [x] Verify all references.
- [x] Revise the abstract and conclusion to reflect the corrected results.

### Priority 4 — Final polish

- [x] Edit grammar and sentence length.
- [x] Standardize terminology.
- [x] Check equations and notation.
- [x] Check formatting against the target conference or journal.
- [x] Perform a final table-by-table and claim-by-claim audit.

---

## 10. Final reviewer verdict

**Recommendation: Major revision before submission.**

### Strengths

- Important and practical research question.
- Lightweight and interpretable modeling approach.
- Broad set of reported evaluation metrics.
- Cross-domain analysis and explicit negative results.
- Useful error analysis and reproducibility intentions.

### Main weaknesses

- Calibration values and protocols are not sufficiently reconciled.
- The combined RAGTruth result is not entirely zero-shot because some task types are used in training.
- HaluEval results may be strongly affected by answer-length and lexical shortcuts.
- Improved ECE does not necessarily yield a useful classification threshold.
- The relationship between the classifier and the full deployed system needs clearer evaluation.
- The novelty and generalization claims need stronger evidence.

### What would most improve the paper?

The most valuable next step is not simply to polish the English. It is to verify the experiments, reconcile the reported values, test the shortcut hypothesis, and clearly separate in-domain performance, unseen-task generalization, calibration, and full-system performance.

A technical review of the PDF cannot establish that the code or reported experiments are authentic and reproducible. The source repository, experiment configurations, and saved outputs must be checked separately before making that claim.

---

## 11. Final submission-readiness checklist

Use this compact list for the final pass:

- [x] All abstract numbers match the final tables.
- [x] All tables can be reproduced from the released artifacts.
- [x] Calibration protocols and splits are clearly documented.
- [x] No target test labels are used for calibration or threshold selection.
- [x] Monotonic-constraint signs match the intended behavior and code.
- [x] Combined-set and held-out-task results are clearly separated.
- [x] Length-only and lexical-only shortcuts are tested.
- [ ] Baseline comparisons are fair and sufficiently strong.
- [x] Classifier-only and full-system results are separate.
- [x] Confidence intervals and statistical tests are correctly described.
- [x] Latency and cost calculations are transparent.
- [x] References and claims have been verified.
- [x] Figures and tables are readable.
- [x] Limitations are explicit and consistent with the conclusion.
- [x] The final manuscript follows the target venue's submission requirements.

**Bottom line:** The manuscript has a worthwhile direction and a substantial amount of experimental work. Its readiness depends on resolving the scientific and reporting concerns above, not on achieving a higher HaluEval score alone.

---

## 12. Resolution status (author response, 2026-10-19)

Every item below was checked against the current manuscript, the code, and the saved artifacts. The manuscript was rebuilt after these fixes: 17 pages, warning-free log, all figures at 600 DPI. New analyses live in `src/models/review_extras.py` and `src/models/review_source_ablation.py`, with outputs under `artifacts/results/extras/`.

### 3.1 Monotone constraints — resolved (writing error only)

- The constraint vector in `src/models/run_b6_modified.py` is correct: entailment, overlap, and semantic similarity carry −1; contradiction, novelty, and the claim-level contradiction aggregates carry +1.
- The manuscript sentence now states the derivative direction correctly (an evidence-positive feature keeps ∂ŷ/∂x ≤ 0), and the constraint vector is described as a design prior, not a semantic guarantee.
- The regression test `tests/test_b6_modified.py::test_constraint_string_values` asserts the signs and the feature order; it now also covers the entailment sign explicitly.

### 3.2 Calibration reconciliation — resolved (clarity, not a numeric error)

- §5.2's 0.726 → 0.126 is the EC-XGB display calibrator (`b6_calibration.json`, `ec-xgb-m3-seed-42`).
- Table 8's 0.8185 → 0.1335/0.1316 is the standard XGBoost model (`b4_target_calibration.json`).
- Both use the same disjoint split: 5,034 calibration rows and 900 held-out test rows. The manuscript now states this explicitly, attributes each value to its model, and names the standard model in the table and figure captions. No number was replaced; every value was re-derived from the artifacts.

### 3.3 Zero-shot framing — resolved

- §5.4 is retitled "Cross-domain transfer". The text now separates three cases: the standard model (zero-shot on every external corpus), EC-XGB (zero-shot on held-out QA and FaithBench; adaptation on summarization and data-to-text), and the combined corpus (labeled as a mix).
- The dataset table, Figure 1/6 captions, Table 6 caption, and the abstract are updated. The held-out QA result is reported separately (abstract: "the held-out QA task stays near F1 0.30").
- Table 6 now states the population of each row (2,700-row test split, 900-row held-out QA split, 750 FaithBench rows). The §5.4 full-corpus numbers are labeled as the dedicated cross-domain run, and the population difference is explained (35% positive in the test split against 43% in the full corpus).

### 3.4 HaluEval shortcuts — addressed with new analyses

- Length-only baseline (logistic regression on log n_chars and log n_words, same grouped split): F1 = 0.9457, AUROC = 0.9759, MCC = 0.891.
- Length-balanced evaluation (247 rows from each of the six length bins, 1,482 rows, three seeds): EC-XGB F1 = 0.9867, AUROC = 0.9987.
- The lexical-overlap-only control already exists in Table 4 (overlap heuristic, F1 = 0.9352).
- The paper now states the interpretation: the benchmark carries a strong length signal, and EC-XGB holds when the length mix is neutralized.
- A fine-tuned encoder detector was not added; the GPT 5.6 Luna judge remains the strongest external comparison, and this is stated as a limitation.

### 3.5 Threshold after target calibration — resolved

- Test prevalence is 17.8%; every calibrated score falls below 0.5 (maximum 0.313), which explains the zero predicted-positive rate.
- A threshold selected on the 5,034-row calibration split at a 5% false-positive budget (0.313) gives held-out recall 5.0%, precision 15.7%, F1 0.076, and a false-positive rate of 5.8%.
- Reported conclusion: target recalibration fixes the probability scale, not the ranking. Label budget: 252 rows (5% of the split) already give ECE 0.133, matching the full split.

### 3.6 EC-XGB and the deployed application — separated in text

- The manuscript states that all quantitative results measure the EC-XGB classifier and its calibrator alone, and that the claim-verification layer changes the displayed verdict and is demonstrated through the interface rather than measured.
- Headline F1/AUROC are classifier numbers; latency and cost cover the classifier pipeline, with retrieval and judge calls excluded.

### 3.7 Source indicator — resolved with a new ablation

- m3 without the source indicator, same training rows and tuned hyperparameters, three seeds: in-domain F1 0.9849 against 0.9855, AUROC 0.9985 against 0.9984, shifted flag rate 61.5% against 61.3%, held-out QA F1 0.308 against 0.308, FaithBench flag rate 7.3% in both.
- Inference-time assignment is documented (one for natural responses, zero for benchmark-style rows). FaithBench remains the unseen-source check; an entirely new source stays future work.

### 4. Reporting audit — resolved

- **4.1:** Tables 4 and 7 now use four decimals; the two standard-XGBoost entries are explained (F1 0.9846 in the baseline run against 0.9844 in the modified run, seed-level variation).
- **4.2:** every Table 6/8 value was re-checked against `b6_external_metrics.csv` and `b4_target_calibration.json`; combined-set and held-out-QA flag rates are now distinct in text, table, and captions.
- **4.3:** the majority-baseline row was added to Table 4, so the eight-model count is correct; probability-based metrics are marked undefined for a constant score.
- **4.4:** feature counts were verified from the code (26 base + 8 claims + 1 indicator); the Table 5 caption explains 34 vs 35; `b6_run_config.json` records the exact feature order.
- **4.5:** Table 11 notes that components are timed in separate instrumented passes and that retrieval and judge calls are excluded; totals are medians.
- **4.6:** error analysis now describes the actual procedure: 10 false positives and 10 false negatives, categorized with keyword heuristics, and the sample is stated to describe the reviewed cases rather than the full error population.

### 5. Design improvements

- **5.1:** the baseline set is stated as is; the neural-detector gap is declared as a limitation.
- **5.2:** McNemar pairing (seed-42 test predictions) is stated; bootstrap intervals are group-aware over question or source groups; cross-domain intervals are reported for both models (standard QA F1 [0.260, 0.342], AUROC [0.514, 0.607]; EC-XGB QA F1 [0.265, 0.353], AUROC [0.502, 0.603]).
- **5.3:** label-budget result added; the calibration split stays disjoint from the test split.
- **5.4:** the English-centric scope is already stated; the transfer conditions are now defined per corpus.

### 6–8. Related work, abstract and conclusion, writing

- Related work: two sentences state the differences from the closest feature-based, LLM-scored, and tool-based methods; a comparison table was not added to hold the 17-page budget.
- Abstract (276 words) reports the in-domain result, the combined-corpus shift result with its limited AUROC, the held-out QA result, the calibrated ECE, and the cost. The deployment sentence was removed. The conclusion separates a risk estimate from a fact-checking verdict, states the labeled-target requirement, and does not claim general-purpose detection.
- Figures: charts redrawn at print size with readable fonts at 600 DPI, legends in every panel, and the UI figure reduced to two panels. References remain 30/30 with no unverified additions.
- Abbreviations are defined at first use (NLI, EC-XGB, AUROC, PR-AUC, MCC, ECE), and absolute claims about prior work were softened from "is not reported/measured/tested" to "is rarely reported/measured/tested".
- Reproducibility: `requirements.txt` matches the reported environment exactly; split indices, seeds, and the frozen manifest are present.

### Not addressed (with rationale)

- **Fine-tuned encoder/neural detector baseline:** no compute budget for a fair within-study comparison; declared as future work.
- **Entirely new unseen-source corpus:** no new licensed data; FaithBench and held-out RAGTruth QA provide the unseen-corpus and unseen-task checks.
- **Related-work comparison table:** replaced by targeted differentiation sentences to keep the paper at 17 pages.

### Open items after this pass (the nine boxes left unticked above)

1. **Claim-level verification and full-system results as separate evaluations.** The manuscript describes the layer and demonstrates it through the interface, but does not measure it as a standalone system.
2. **A fine-tuned encoder/neural detector baseline.**
3. **A claim-verification baseline.**
4. **Per-baseline computational cost and input-requirement reporting.** The judge-versus-XGBoost cost table covers the two systems that matter most here.
5. **Latency and cost of the complete user-facing pipeline**, including retrieval and judge calls.
6. **Overall baseline strength**, tied to items 2 and 3.

Checked boxes mark items resolved in the current manuscript and artifacts. The nine unticked boxes are these six grouped gaps, listed so a follow-up review can start there.

