# HaluRISC - fix checklist

> Temporary working file. Delete before submission.

Source: `Journal_Paper/halurisc.tex` (892 lines)
Build checked: 2026-10-29 11:15
Build facts: 18 pages; page 18 has 17 lines (references 28–34); 0 LaTeX warnings; 34/34 citations; table audit 203/203.

How to use:

1. Apply P0 first, then P1. Rebuild after each group.
2. Line numbers refer to the current file and will shift after edits. Match on the quoted text, not only the line number.
3. Do not change tables, figures, or algorithms unless an item explicitly says so.
4. After the final edit, run the verification section at the bottom.
5. This checklist is intentionally limited to correctness, clarity, and page count. It does not promise a detector score.

Priority key:

- **P0 — Critical:** breaks the PDF, states a wrong number, or makes a false claim.
- **P1 — Major:** must fix before submission; affects a caption, claim, citation, or page budget.
- **P2 — Minor:** clarity or consistency; fix if the edit is safe.
- **P3 — Optional:** style only; do not apply unless asked.

---

## P0 — Build blockers

### [ ] BLD-01 — §1 Introduction, line 294 — P0

**Current**

```latex
This paper addresses all four issues by developing a black-box model that takes a query $, a reference context $, and a candidate answer $, and outputs (H \mid Q, C, A).
```

**Fix**

```latex
This paper addresses all four issues by developing a black-box model that takes a query $q$, a reference context $c$, and a candidate answer $a$, and outputs $P(H \mid q,c,a)$.
```

**Why:** the current string renders on PDF page 2 as “query ,areferencecontext…”. The variables and the probability expression are lost.

---

### [ ] BLD-02 — §1 Introduction, line 296 — P0

**Current**

```latex
one analysis takes 61.8~ms at about \.001 per 1{,}000 predictions.
```

**Fix**

```latex
one analysis takes 61.8~ms at about \$0.001 per 1{,}000 predictions.
```

**Why:** `\.` is a dot accent. The PDF renders “˙001” instead of “$0.001”.

---

### [ ] BLD-03 — §3.5 System and Reproducibility, line 438 — P0

**Current:** `This study quantify the reliability by calculating four different measures`

**Fix:** `This study quantifies the reliability by calculating four different measures`

**Why:** subject–verb agreement.

---

### [ ] BLD-04 — Abstract, line 273 — P0

**Current**

```text
For example, they require a large judge model; a computer with graphics capability to run an encoder classifier; feature-based detectors provide little information regarding how well their estimates are calibrated or how well these detectors perform outside the domain of their training data.
```

**Fix**

```text
For example, they require a large judge model or a graphics card to run an encoder classifier, and feature-based detectors provide little information about how well their estimates are calibrated or how well they perform outside their training domain.
```

**Why:** the semicolon list mixes two requirements with a separate clause about feature-based detectors. The fix keeps the meaning and removes the grammar error.

---

## P1 — Numbers, captions, and claims

### [ ] NUM-01 — §5.1 Model comparison, line 555 — P0

**Current:** `A neural baseline trained solely on the source material outperforms EC-XGB in-domain (F1 0.9912 vs 0.9846)`

**Fix:** `A neural baseline trained solely on the source material outperforms EC-XGB in-domain (F1 0.9912 vs 0.9855)`

**Why:** 0.9846 is the conventional XGBoost F1 from Table 5. EC-XGB (m3) is 0.9855. The sentence compares the encoder with EC-XGB, so the comparator is wrong.

---

### [ ] NUM-02 — §5.3 Calibration, Table 8 caption — P0

**Current**

```text
Calibration on the same 900-row RAGTruth QA test set (17.8\% positive; means over seeds 42/123/456). The deployed EC-XGB display calibrator is the seed-42 target Platt row (ECE 0.126). AUROC is unchanged by calibration: 0.540 for the conventional model and 0.549 for EC-XGB.
```

**Fix**

```text
Calibration on the same 900-row RAGTruth QA test set (17.8\% positive). EC-XGB cells are means over seeds 42/123/456; conventional rows use the Table 5 checkpoint. The deployed EC-XGB display calibrator is the seed-42 target Platt row (ECE 0.126). AUROC is unchanged by calibration: 0.540 for the conventional model and 0.549 for EC-XGB.
```

**Why:** the conventional cells are the single Table 5 checkpoint, not seed means. The 0.540 value belongs to that checkpoint. Table 9 (`tab:shift`) reports 0.5465 for m0, which is a separate tuning run. Without this note, the two tables look inconsistent.

---

### [ ] NUM-03 — Table 5 caption or §5.1, after the table — P1

**Current:** no note explains why the conventional XGBoost F1 is 0.9846 in Table 5 and 0.9844 for m0 in Table 6.

**Fix (append to the Table 5 caption):**

```text
The conventional row in this table and the m0 row in Table 6 come from separate tuning runs.
```

**Why:** the difference is small, but a reviewer will notice it. One sentence prevents a consistency query.

---

### [ ] NUM-04 — §4.1 Experimental Setup, line 481 — P0

**Current**

```text
The configuration used for XGBoost has a maximum tree depth of 4, a learning rate of 0.01, 500 trees, a subsampling ratio of 0.9, and a column sampling ratio of 0.7.
```

**Fix**

```text
For the seed-42 conventional model, the configuration has a maximum tree depth of 4, a learning rate of 0.01, 500 trees, a subsampling ratio of 0.9, and a column sampling ratio of 0.7. Each seed and each EC-XGB variant was tuned separately.
```

**Why:** `b6_tuning.json` shows different configurations for seeds 123 and 456 and for m1–m3. The current wording reads as one configuration for every run, which is not reproducible.

---

### [ ] NUM-05 — §3.3 Evidence-Consistent XGBoost, line 404 — P0

**Current**

```latex
Let $\phi'(x)$ be the complete model input and $\tilde{\phi}(x)$ the remaining 25 base features excluding $\ell'$; there are 35 input features in total.
```

**Fix**

```latex
Let $\phi'(x)$ be the complete model input and $\tilde{\phi}(x)$ the remaining 24 base features excluding the two log-scaled length columns; there are 35 input features in total (24 base + 2 log-scaled length + 8 claim + 1 source).
```

**Why:** `b6_run_config.json` lists `n_chars` and `n_words` in `log_length_columns`. Two features are log-scaled, not one, so 24 base features remain. The current arithmetic gives 25 + 1 + 8 + 1 = 35 only if ℓ′ is one feature.

---

### [ ] NUM-06 — §4.2 Dataset, line 487 — P0

**Current**

```text
The RAGTruth QA task contributed 5{,}934 rows of QA tasks in 989 groups. The remaining 900 rows in 150 groups formed the test set, and none of these groups overlap with those used in the recalibration set.
```

**Fix**

```text
The RAGTruth QA task contributed 5{,}934 rows in 989 groups. Of these, 900 rows in 150 groups form the test set and 5{,}034 rows form the recalibration set; the groups do not overlap.
```

**Why:** 900 is part of 5,934, not what remains after it. The calibration split is 5,034 rows.

---

### [ ] NUM-07 — §5.1 Model comparison, line 551 — P0

**Current**

```text
... and the majority baseline (wherein the classifier always predicts the most common label) produced an F1 value of 0.0000, providing evidence that the data is well-balanced.
```

**Fix**

```text
... and the majority baseline (wherein the classifier always predicts the most common label) produced an F1 value of 0.0000; the test set contains 1,500 rows per class.
```

**Why:** a majority-negative classifier scores 0 at any prevalence. The 1,500/1,500 test composition is the evidence for balance, not the baseline result.

---

### [ ] NUM-08 — §3.5 System and Reproducibility, line 442 — P0

**Current:** `All quantitative results in Section~\ref{sec:results} pertain to the EC-XGB classifier and its calibrator.`

**Fix:** `Unless a baseline or ablation variant is named, the results in Section~\ref{sec:results} use the EC-XGB classifier and its calibrator.`

**Why:** §5.1 reports baselines and §5.5 ablates conventional XGBoost. The current sentence is false.

---

### [ ] NUM-09 — §5.6 Explanation reliability, line 757 — P0

**Current:** `A structured expert audit assessed every explanation.`

**Fix:** `A structured AI-assisted audit assessed the 26 exported explanations.`

**Why:** the audit covers 26 error-enriched cases, not every explanation in the test set. The review mode is `ai-expert`, so “expert” without qualification may imply human reviewers.

---

### [ ] NUM-10 — Abstract, line 273 — P0

**Current:** `Performance on a QA evaluation task not seen during training is consistent with random guessing (i.e., an F1 near 0.30).`

**Fix:** `Performance on a QA task not seen during training remains near chance (AUROC 0.549; F1 0.30).`

**Why:** AUROC 0.549 is near chance. An F1 of 0.30 is not a random-guess F1 at 17.8% prevalence. The current sentence overstates what “random guessing” means.

---

### [ ] NUM-11 — §5.4 Cross-domain transfer, line 670 — P0

**Current**

```text
However, the ECE on the RAGTruth QA task decreased from 0.815 to 0.743. Therefore, multi-source training does not improve accuracy on previously unseen tasks for free.
```

**Fix**

```text
The ECE on the RAGTruth QA task decreased from 0.815 to 0.743, but the flat F1 shows that multi-source training does not improve accuracy on unseen tasks for free.
```

**Why:** the ECE drop does not support “therefore”. The flat F1 does.

---

### [ ] NUM-12 — §5.8 Error analysis, line 829 — P0

**Current:** `Table~\ref{tab:length} shows how many were false positives and false negatives.`

**Fix:** `Table~\ref{tab:length} shows the false-positive and false-negative rates by answer length.`

**Why:** Table 15 has rates, not counts.

---

### [ ] NUM-13 — §5.9 Discussion, line 853 — P1

**Current:** `The gains come from evidence-consistent features and multi-source training, not from a different type of learning system.`

**Fix:** `The gain coincides with multi-source training rather than a different type of learning system.`

**Why:** the claim-level features do not improve RAGTruth-all AUROC. m2 is 0.502 and m1 is 0.543. The gain appears when the RAGTruth non-QA rows are added in m3.

---

### [ ] NUM-14 — §5.2 Evidence-consistent XGBoost, line 586 — P1

**Current:** `The Wilcoxon signed-rank test for the differences across seeds between models was also not significant ($p = 0.25$).`

**Fix:** `The Wilcoxon signed-rank test across the three seeds was underpowered ($p = 0.25$, the smallest attainable two-sided value with three pairs).`

**Why:** with three pairs, p = 0.25 is the minimum possible two-sided value. Reporting it as “not significant” is technically true but uninformative.

---

### [ ] NUM-15 — §5.1 Model comparison, line 553 — P1

**Current:** `it was statistically significant in a McNemar test comparing the paired seed-42 test predictions ($p=0.044$).`

**Fix:** `it was nominally significant in an uncorrected McNemar test comparing the paired seed-42 test predictions ($p=0.044$).`

**Why:** the paragraph tests three comparisons. p = 0.044 would not survive a Bonferroni correction (0.017). “Nominally significant” is accurate.

---

### [ ] NUM-16 — §3.5, lines 448–452 — P1 — needs decision

**Current:** the text says a contradicted claim sets the risk to high, but the hallucinated-number example is a contradicted number and the card shows 35.4% moderate risk.

**Fix option A:** clarify that 35.4% is the calibrated score before claim-level adjustment and that the claim card separately marks the contradiction.

**Fix option B:** change the example to a supported answer whose calibrated score is moderate.

**Why:** a reader who follows the verdict rule will see an inconsistency. Do not guess. Record the author’s decision.

---

### [ ] NUM-17 — §5.3 Calibration, line 664 — P2

**Current:** `Recall stays between 99.0\% and 99.5\%`

**Fix:** `Recall stays between 99.1\% and 99.5\%`

**Why:** `b6_strict_mode.json` gives 99.4–99.5% for m3 and 99.1–99.2% for m0. 99.0 is not attained.

---

### [ ] NUM-18 — §5.3 Calibration, line 664 — P2

**Current:** `Source calibration has little impact on the ECE`

**Fix:** `Source calibration does not reduce the ECE`

**Why:** conventional ECE moves from 0.8185 to 0.8089, and EC-XGB moves from 0.7433 to 0.7735. “Little impact” hides the wrong direction for EC-XGB.

---

### [ ] NUM-19 — §5.6 Explanation reliability, line 757 — P2

**Current:** `False negatives occur when a wrong answer reuses entities from its context, which keeps overlap high.`

**Fix:** `Many false negatives occur when a wrong answer reuses entities from its context, which keeps overlap high.`

**Why:** several audited false negatives are fabricated titles or off-target answers, not entity reuse.

---

### [ ] NUM-20 — Abstract, line 273 — P2

**Current:** `Eight of the 35 features represent the extent to which the claims are supported.`

**Fix:** `Eight of the 35 features summarize claim-level evidence.`

**Why:** one of the eight is the number of claims, which is not a support extent.

---

### [ ] NUM-21 — §5.1 Model comparison, line 551 — P1

**Current**

```text
The TF-IDF models were positioned between these two extremes; the answer-only linear model scored 0.9224, and the full-input model scored 0.5999.
```

**Fix**

```text
The answer-only TF-IDF model scored 0.9224, between the NLI-only and XGBoost results, while the full-input model scored 0.5999, below both.
```

**Why:** 0.5999 is below the NLI-only 0.6714, so “between these two extremes” is false if the extremes are NLI-only and XGBoost.

---

## P1 — Language and grammar

### [ ] LANG-01 — Abstract, line 273 — P1

**Current:** `When evaluating results on task type categories (RAGTruth) included in the training set`

**Fix:** `On the RAGTruth task types seen during training`

**Why:** removes the unclear “task type categories” and the banned “When” opener.

---

### [ ] LANG-02 — §1 Introduction, line 290 — P2

**Current:** `Judges carry high computational costs and encoder models are hard to interpret.`

**Fix:** `Judges carry high computational costs, and encoder models are hard to interpret.`

**Why:** comma before a compound sentence.

---

### [ ] LANG-03 — §2.1 Preliminaries, line 302 — P2

**Current:** `Only if a risk score has a numeric value that is consistent with the probability of generating a hallucination can that risk score be used effectively.`

**Fix:** `Only if a risk score matches the observed hallucination frequency can it be used effectively.`

**Why:** every score has a numeric value. The intended condition is calibration, not the existence of a number.

---

### [ ] LANG-04 — §2.2 Baseline Method, line 306 — P2

**Current:** `because it handles nonlinear interactions, it trains quickly on a tabular design matrix, and its tree structure supports exact SHAP values.`

**Fix:** `because it handles nonlinear interactions, trains quickly on a tabular design matrix, and supports exact SHAP values through its tree structure.`

**Why:** parallel list structure.

---

### [ ] LANG-05 — Table 2 caption, line 380 — P2

**Current**

```text
The answer is split into at most six atomic clauses, with conjunctions such as ``and'' or ``but'' broken into separate clauses, each clause is scored against its four best-matching context sentences, and the pair probabilities are aggregated.
```

**Fix**

```text
The answer is split into at most six atomic clauses, with conjunctions such as ``and'' or ``but'' broken into separate clauses. Each clause is scored against its four best-matching context sentences, and the pair probabilities are aggregated.
```

**Why:** the current caption is a comma splice.

---

### [ ] LANG-06 — §4.2 Dataset, line 485 — P1

**Current**

```text
Since both versions of the answer to each question come from the same knowledge passage, the model can easily learn the relationship between the context and the answers, so the split is at the group level.
```

**Fix**

```text
Both answers to a question come from the same knowledge passage, so the split is at the group level to prevent leakage.
```

**Why:** the reason for the group split is leakage prevention, not that the model learns the relationship easily.

---

### [ ] LANG-07 — §4.4 Additional Protocols, line 517 — P1

**Current:** `As well as being influenced by distribution shift \cite{ovadia2019}, the same comparison was performed during the target-domain evaluation.`

**Fix:** `The same comparison was performed during the target-domain evaluation, where distribution shift affects calibration \cite{ovadia2019}.`

**Why:** the current sentence has a dangling modifier. The comparison is not “influenced by distribution shift”.

---

### [ ] LANG-08 — §4.4 Additional Protocols, line 539 — P2

**Current**

```text
In content edits, entities, numbers, or dates are changed, supporting context is removed or an unrelated sentence is added to the content; each edit should affect the calibrated score if it alters the evidence.
```

**Fix**

```text
In content edits, entities, numbers, or dates are changed, or supporting context is removed or an unrelated sentence is added. Each edit should affect the calibrated score if it alters the evidence.
```

**Why:** run-on sentence with a semicolon.

---

### [ ] LANG-09 — §5.1 Model comparison, line 551 — P1

**Current:** `The performance of all the baselines has been averaged across the three runs to report their overall performance on the test set.`

**Fix:** `The baselines are averaged over the three runs to report their performance on the test set.`

**Why:** “performance … to report their overall performance” is redundant.

---

### [ ] LANG-10 — §5.3 Calibration, line 664 — P1

**Current:** `This is an example of selecting based on a recall-first approach and results in approximately a one-point loss in F1.`

**Fix:** `This recall-first choice costs about one point of F1.`

**Why:** the current sentence has two predicates and an unclear subject.

---

### [ ] LANG-11 — §5.4 Cross-domain transfer, line 670 — P1

**Current:** `the conservative regime decreased the F1 from 0.813 to 0.120.`

**Fix:** `the 0.5-threshold EC-XGB regime decreased the F1 from 0.813 to 0.120.`

**Why:** “conservative regime” is undefined and appears only here.

---

### [ ] LANG-12 — §5.7 Efficiency and the LLM judge, line 761 — P2

**Current:** `Table~\ref{tab:latency} displays the total average and distribution of the latency over the 200 test samples.`

**Fix:** `Table~\ref{tab:latency} reports the latency distribution over the 200 test samples.`

**Why:** the table shows p50, p95, and mean, not a “total average”.

---

### [ ] LANG-13 — §5.7 Efficiency and the LLM judge, line 761 — P2

**Current:** `The latency for XGBoost was much lower than that for either method, at 1.3\,ms.`

**Fix:** `The latency for XGBoost was much lower than that for either of those two components, at 1.3\,ms.`

**Why:** “either method” has no clear referent after three components.

---

### [ ] LANG-14 — §5.8 Error analysis, line 829 — P2

**Current:** `The key aspect of the data is how the answer length of incorrect answers is distributed.`

**Fix:** `The distribution of answer length for incorrect answers is the key pattern.`

**Why:** clearer subject and verb.

---

### [ ] LANG-15 — Conclusion, line 857 — P1

**Current:** `To be effective, hallucinations need to be detected quickly, inexpensively, and explainably.`

**Fix:** `A useful hallucination detector must be fast, inexpensive, and explainable.`

**Why:** hallucinations are detected; they do not need to be detected. The current subject is wrong.

---

### [ ] LANG-16 — Conclusion, line 859 — P2

**Current:** `An encoder trained solely on source material outperforms all other models on the benchmark (F1 = 0.9912); however, the encoder performs poorly on unseen tasks`

**Fix:** `An encoder trained solely on source material outperforms all other models on the benchmark (F1 = 0.9912). However, the encoder performs poorly on unseen tasks`

**Why:** the supervisor’s guideline discourages semicolons in prose. Split into two sentences.

---

### [ ] LANG-17 — §5.2 Evidence-consistent XGBoost, line 586 — P2

**Current:** `The increases in the measures of performance are small.`

**Fix:** `The gains are small.`

**Why:** wordy and vague.

---

### [ ] LANG-18 — §5.2 Evidence-consistent XGBoost, line 586 — P2

**Current:** `the McNemar test (test of m0 vs.\ m3)`

**Fix:** `the McNemar test of m0 against m3`

**Why:** removes the repeated “test”.

---

### [ ] LANG-19 — §5.5 Ablation, line 704 — P1

**Current:** `The ablation study in Table~\ref{tab:ablation} demonstrates how much each feature group contributes to this model`

**Fix:** `The ablation study in Table~\ref{tab:ablation} demonstrates how much each feature group contributes to the conventional XGBoost model`

**Why:** “this model” is ambiguous; the table uses the conventional model as the full model.

---

### [ ] LANG-20 — §5.6 Explanation reliability, line 727 — P2

**Current:** `The overlap in contexts has the greatest effect`

**Fix:** `Context overlap has the greatest effect`

**Why:** clearer noun phrase.

---

### [ ] LANG-21 — §5.6 Explanation reliability, line 727 — P2

**Current:** `then the contradictions between contexts and answers (in terms of probability)`

**Fix:** `then the NLI contradiction probability`

**Why:** the current phrase is vague about which quantity is ranked.

---

### [ ] LANG-22 — §5.6 Explanation reliability, line 729 — P3

**Current:** `the size of each group was reported openly in Table~\ref{tab:perturb}.`

**Fix:** `the size of each group is reported in Table~\ref{tab:perturb}.`

**Why:** “openly” adds nothing.

---

## P1 — Terminology sweep: conventional XGBoost

No “standard XGBoost” or “standard model” was found. These are the remaining ambiguous uses.

### [ ] TERM-01 — §5.1, line 551 — P1

**Current:** `when using the XGBoost method`

**Fix:** `when using conventional XGBoost`

---

### [ ] TERM-02 — §5.1, line 553 — P1

**Current:** `Although the XGBoost margin over random forest`

**Fix:** `Although the conventional XGBoost margin over random forest`

---

### [ ] TERM-03 — §5.1, line 553 — P1

**Current:** `when comparing XGBoost to logistic regression`

**Fix:** `when comparing conventional XGBoost to logistic regression`

---

### [ ] TERM-04 — §5.1, line 553 — P1

**Current:** `for XGBoost are F1`

**Fix:** `for conventional XGBoost are F1`

---

### [ ] TERM-05 — §4.1, line 481 — P1

**Current:** `The configuration used for XGBoost`

**Fix:** `The configuration used for conventional XGBoost`

---

### [ ] TERM-06 — Abstract, line 273 — P2

**Current:** `Evidence Consistent eXtreme Gradient Boosting`

**Fix:** `Evidence-Consistent eXtreme Gradient Boosting`

**Why:** hyphenate the compound adjective, matching the title and body.

---

### [ ] TERM-07 — §4.4, line 545 — P2

**Current:** `a biased model that serves as a standard judge comparison`

**Fix:** `a commercial judge baseline`

**Why:** “biased” is unexplained, and “standard” can be confused with the conventional model. If the bias is intended, state which bias and cite it.

---

### [ ] TERM-08 — §5.2, line 588 — P3

**Current:** `RAGTruth-all` (two occurrences)

**Fix:** `RAGTruth all` (match the table and the rest of the prose)

**Why:** consistency only.

---

### [ ] TERM-09 — §3.4, §5.3 — P3

**Current:** `fit on` and `fitted on` are both used.

**Fix:** standardize to `fitted on`.

**Why:** consistency only.

---

## P1 — Citations and provenance

### [ ] CIT-01 — §2.3 Limitations and Research Gap, line 338 — P0

**Current:** `close to 85\% of HaluEval answers were produced by sampling and filtering~\cite{bang-etal-2025-hallulens}`

**Fix:** `close to 85\% of the full HaluEval benchmark was produced by ChatGPT through sampling and filtering~\cite{haluEval}`

**Why:** the 85% figure comes from HaluEval’s 30K/35K construction, not from HalluLens. HalluLens is cited for the factuality-versus-hallucination point. The QA subset used here is entirely from the 30K task-specific examples, so “85% of HaluEval answers” is also imprecise.

---

### [ ] CIT-02 — §4.4 Additional Protocols, line 545 — P1

**Current:** `using GPT 5.6 Luna \cite{zheng2023judge}`

**Fix:** `using GPT 5.6 Luna with the LLM-as-a-judge protocol of Zheng et al. \cite{zheng2023judge}`

**Why:** `zheng2023judge` is the LLM-as-a-judge paper, not the GPT 5.6 Luna model. The current citation implies the paper introduces the model.

---

### [ ] CIT-03 — §5.8 Error analysis, line 827 — P1

**Current:** `A heuristic analysis of 20 randomly selected errors (10 false positives and 10 false negatives)`

**Fix:** `A heuristic analysis of 20 errors sampled evenly across the two directions (10 false positives and 10 false negatives)`

**Why:** the sample was stratified 10/10, not random from all 48 errors.

---

### [ ] CIT-04 — Acknowledgements, line 882 — P1

**Current**

```text
The authors thank their supervisor, Ohidujjaman Tuhin, for his supervision and input on this work.
```

**Fix:** delete the sentence. Keep the dataset-maintainer sentence.

**Why:** Ohidujjaman Tuhin is an author. G14 says acknowledgements name non-author support. His supervision is already in CRediT.

---

### [ ] CIT-05 — §5.6, line 757 — P1

**Current:** `A structured expert audit`

**Fix:** `A structured AI-assisted audit`

**Why:** same as NUM-09. Apply once.

---

## P1 — Page trim (18 → 17)

Page 18 has 17 lines (references 28–34). Cut about 255 words net. The cuts below total 275 words; the caption clarifications above add about 20 words.

### [ ] TRIM-01 — §5.1 final paragraph — 59 words

**Delete**

```text
Single-answer behavior shows how individual answers can disagree in ways that aggregate measures hide. For example, if two answers were generated from one question and one context, both would have an exact match under the overlap heuristic. They could also produce an identical result for an NLI-only model. However, logistic regression, random forest, and XGBoost all produced different results.
```

**Why:** unclear, has no numbers, and repeats the aggregate point. Deleting it also removes a confusing passage.

---

### [ ] TRIM-02 — §5.3, after Figure 4 — 25 words

**Delete**

```text
The table shows average values over three seeds, so each cell reflects an average, and the EC-XGB raw and calibrated cells read 0.7433 and 0.1263.
```

**Why:** the caption already states the means.

---

### [ ] TRIM-03 — §5.7, HTTP timing block — 60 words

**Delete**

```text
For end-to-end HTTP timing, the median values are 42.8\,ms for prediction, 57.1\,ms for analysis, and 143.6\,ms for verification with retrieval over 26 requests, and 502.5\,ms with the judge over 6 requests. The component table does not include retrieval, judge, or model-loading time. The HTTP verification number includes retrieval but not the judge.
```

**Why:** deployment detail. Table 12 already reports component latency. If the author wants the HTTP numbers, keep the first sentence and delete the two caveats, but then the page count may not reach 17.

---

### [ ] TRIM-04 — §5.2, line 586 — 20 words

**Delete**

```text
All the ablated versions of the model have almost identical F1 scores compared to the original version of the model.
```

**Why:** repeats the saturation point.

---

### [ ] TRIM-05 — §5.4, line 670 — 10 words

**Delete**

```text
Recall rates did remain quite high for the conventional model.
```

**Why:** redundant with the 99.9% flag rate.

---

### [ ] TRIM-06 — §5.4, line 670 — 15 words

**Delete**

```text
As such, at this operating point, the AUROC and flag rate are the appropriate comparisons.
```

**Why:** the next sentence already says the F1 drop is a limitation.

---

### [ ] TRIM-07 — §5.5, line 704 — 6 words

**Delete:** `, and the paper reports that result`

**Why:** meta prose.

---

### [ ] TRIM-08 — §3.5, lines 442–444 — 40 words saved

**Current**

```text
The Next.js frontend provides a streaming LLM conversation on the chat page using the assistant-ui library, while the FastAPI backend hosts the model, the calibrator, the SHAP explainer, and the feature models. All of these are loaded into memory at startup. A Next.js proxy forwards each request to a FastAPI endpoint to avoid cross-origin issues and keep the API key on the server.
```

**Fix**

```text
The system is a Next.js frontend and a FastAPI backend that loads the model, calibrator, explainer, and feature models at startup.
```

**Why:** the figure and the surrounding text already describe the deployment. The proxy detail is implementation, not a result.

---

### [ ] TRIM-09 — §5.9, line 853 — 20 words

**Delete**

```text
The gains come from evidence-consistent features and multi-source training, not from a different type of learning system.
```

**Why:** unsupported by the ablation. This is also NUM-13. If NUM-13 is applied as a rewrite, delete the original sentence and use the replacement; do not keep both.

---

### [ ] TRIM-10 — Acknowledgements — 20 words

**Delete**

```text
The authors thank their supervisor, Ohidujjaman Tuhin, for his supervision and input on this work.
```

**Why:** he is an author. This is also CIT-04.

---

**Fallback if the build is still 18 pages:** delete LANG-10 (“This is an example of selecting…”, 21 words), or reduce Figure 1 from 0.76 to 0.72 line width.

---

## P3 — Optional style sweep (do not apply unless asked)

These do not block understanding. Apply only if the author wants a mechanical style pass.

- [ ] Replace the remaining prose semicolons with periods or commas. The supervisor’s guideline says no semicolons in prose.
- [ ] Rewrite the four sentences that start with `When` or `Although`.
- [ ] Reduce the five `However` and five `Therefore` transitions.
- [ ] Tighten the first paragraph of §5.1.
- [ ] Replace `utilizes` with `uses` in §4.1.

---

## Decisions needed before applying

- [ ] **DEC-01 — §3.5 risk card.** Is 35.4% the pre-verdict calibrated score, or should the contradicted number produce a high-risk card? Pick option A or B in NUM-16.
- [ ] **DEC-02 — Table 8 conventional model.** Keep the B2 checkpoint (0.540) and add the Table 5 checkpoint note, or recompute the table with m0 (0.5465). Recommended: keep and clarify, no recomputation.
- [ ] **DEC-03 — Abstract AUROC.** Expand `AUROC` at first use: `area under the receiver operating characteristic curve (AUROC)`. Adds 8 words, abstract goes from 277 to about 285, still inside 250–300.
- [ ] **DEC-04 — 85% claim.** Use `\cite{haluEval}` and say “the full benchmark is about 85% ChatGPT-generated”, or keep the current wording and cite both HaluEval and HalluLens. Recommended: HaluEval only.
- [ ] **DEC-05 — HTTP timings.** Delete the block (TRIM-03) or keep it and cut 60 words elsewhere. Recommended: delete.
- [ ] **DEC-06 — Table 5/Table 6 note.** Add the one-sentence clarification (NUM-03) or leave the 0.9846/0.9844 difference unexplained. Recommended: add it.

---

## Verification after edits

- [ ] Build: `cd Journal_Paper; latexmk -pdf -interaction=nonstopmode -outdir=out halurisc.tex`
- [ ] Pages = 17, and no reference block on page 18.
- [ ] `Underfull 0`, `Overfull 0`, `LaTeX Error 0`, undefined citation 0, undefined reference 0.
- [ ] Abstract 250–300 words, one paragraph.
- [ ] Keywords 4–8.
- [ ] Every `\cite` key exists in `ref.bib`, and every entry is cited (expect 34/34).
- [ ] Run `python src/models/review_number_audit.py` from the repo root. Expect 203/203. Then manually spot-check the new prose numbers (0.9855, 0.540/0.5465, 99.1–99.5, 0.7433/0.7735).
- [ ] PDF check: the Introduction reads `a query q, a reference context c, and a candidate answer a` and `$0.001`.
- [ ] PDF check: Table 8 caption and Table 9 agree on the conventional model identity.
- [ ] Search the `.tex` for `standard XGBoost`, `standard model`, `\.001`, and `query $,`.
- [ ] Confirm the trimmed sections still read as coherent paragraphs, not fragments.
- [ ] Confirm no table, figure, or algorithm content changed.

---

## Expected result

- 17 pages, 0 warnings, 34/34 citations, 203/203 table audit.
- No wrong comparator, no broken math, no overclaimed audit coverage.
- The remaining prose stays in the author’s wording. No paragraph is rewritten for detector reasons.
