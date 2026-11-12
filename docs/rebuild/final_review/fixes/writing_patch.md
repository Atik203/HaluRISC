# Writing and readability patch — exact LaTeX replacements

**Target:** `Journal_Paper/halurisc.tex` (current 25-page build, 1{,}029 lines, 12 November 2026)
**Review:** `docs/rebuild/final_review/05_writing.md`
**Guides:** `docs/prompt.md`, `docs/writing-guideline.md`
**Status:** prepared, not applied. No manuscript, PDF, artifact, bibliography, or existing file was modified.

Line numbers are for orientation only. Match the exact strings, and apply from the bottom of the file upward if your editor renumbers lines. Every replacement below is a single paragraph unless it explicitly contains two paragraphs.

## Coordination with other ready patches

The current file already contains the corrections from `results_patch.md` 1A, 1B, 3C2, and 4 and from `provenance_patch.md` 2. The replacements below preserve that corrected wording, so do not apply those sub-edits again after this patch.

| This patch | Overlaps | Action |
| :--- | :--- | :--- |
| L672 | `results_patch.md` 1A (P0 flag rate) | The current L672 already has `all 900 rows` and `every score exceeds 0.095`. The new L672 keeps them. |
| L672 | `results_patch.md` 1B (maximum target-Platt score) | The current L672 already has max 0.355, median 0.331, 16\% at 0.35, and `35.4\% before rounding`. The new L672 keeps them and keeps 0.313 only for P2. |
| L932 | `results_patch.md` 3C2 (adapted encoder label) | The current L932 already says `pooled RAGTruth test set`. The new L932 keeps it. |
| L960 | `results_patch.md` 4 (WP5 count) and `provenance_patch.md` 2 (50-case reviewer) | The current L960 already says `seven of the eight` and `reviewed by one author`. The new L960 keeps them. |

**Number rule.** No value is invented, altered, or moved to a different claim. The current file already contains the corrected values from the earlier patches, and the replacements below preserve them. Where a prose number is removed, the same value remains in the table or figure named in the same paragraph, or it follows by subtraction from a table row. Values that exist only in prose are kept. The conclusion deliberately drops repeated numbers because the abstract and the results tables already carry them.

**Citation rule.** No citation key is added, removed, or moved. `\cite{wang2024fair}` stays in L932.

**Length rule.** New body paragraphs are at least 91 words. The 58-word equation note after Eq. 6 is exempt under the review's own exclusion for equation continuations. The L640 merge is split at a natural seam into 141 and 91 words, and the L672 block into 116 and 207 words.

---

## Item 1 — Merge the short blocks

### 1.1a — L416: leave only the equation note

**Location:** L416, §3.3, immediately after Eq. 6

**Current text (exact):**

```latex
Here $x=(q,c,a)$ is the input triplet, and $\overline{C}$ and $\overline{E}$ are the means of $C(k)$ and $E(k)$ over the claims. For $K>0$, the three ratios sum to one. The ablation in Section~\ref{sec:results} shows that the block adds no aggregate F1 gain. The model input has 35 dimensions: the 26 base features (two of them log-scaled), the eight claim features, and one source indicator. The claim count $K$ is kept because an answer with many claims is harder to verify.
```

**Replacement (exact):**

```latex
Here $x=(q,c,a)$ is the input triplet, and $\overline{C}$ and $\overline{E}$ are the means of $C(k)$ and $E(k)$ over the claims. For $K>0$, the three ratios sum to one. The model input has 35 dimensions: the 26 base features (two of them log-scaled), the eight claim features, and one source indicator.
```

**Why:** The paragraph currently mixes the notation for Eq. 6 with two body claims. The ablation result and the reason for keeping $K$ move to L418, where the training change is described. What remains defines the triplet, the two means, the ratio constraint, and the 35-dimensional input, so it reads as a note on the equation rather than a third short argument.

**Length:** 58 words, about 4 printed lines. This is an equation continuation, which the review excludes from the 90-word floor.

---

### 1.1b — L418: one paragraph for the multi-source change, the ladder, and Algorithm 2

**Location:** L418, §3.3. Replaces the old `Third, ...` paragraph and absorbs the old L420 algorithm paragraph.

**Current text (exact):**

```latex
Third, the training set adds the RAGTruth summarization and data-to-text rows. A binary source indicator is one for those rows and zero for HaluEval, so the classifier can separate the two sources. The classifier never sees RAGTruth QA rows, so its QA ranking is zero-shot. The display calibrator does see labeled QA rows below, so the display score is not zero-shot. The combination is called EC-XGB. Seed 42 is the deployed checkpoint, and the other seeds are a stability check.
```

**Replacement (exact):**

```latex
The training set adds the RAGTruth summarization and data-to-text rows. The ablation ladder starts from m0, the 26-feature conventional reference, and adds log-scaled length counts with monotone constraints at m1, the eight claim features at m2 (34 features), and the RAGTruth non-QA rows with the source indicator at m3 (35 features). A binary source indicator is one for those rows and zero for HaluEval, so the classifier can separate the two sources. The classifier never sees RAGTruth QA rows, so its QA ranking is zero-shot. The display calibrator does see labeled QA rows below, so the display score is not zero-shot. The 35-feature combination is called EC-XGB, and seed 42 is the deployed checkpoint while the other seeds are a stability check. The ablation in Section~\ref{sec:results} shows that the claim block adds no aggregate F1 gain, and the claim count $K$ is kept because an answer with many claims is harder to verify. Algorithm~\ref{alg:ecxgb} lists the procedure: steps 1 and 2 build the feature matrix, and steps 3 to 9 repeat the fit for the three seeds. Each repetition builds the 35-feature input, trains on HaluEval plus the RAGTruth non-QA rows, fits the display calibrator on the 5{,}034 QA calibration rows, and selects the source and target thresholds.
```

**Why:** Three jobs are combined here because each is too short to stand alone. The paragraph names the multi-source change, defines the m0-to-m3 ladder before the results, keeps the zero-shot distinction between the classifier and the display calibrator, and folds in the Algorithm 2 walkthrough. This removes the `Third` frame, gives the third change the same space as the first two, and deletes the duplicated sentence about QA rows staying out of training.

**Length:** 215 words, about 13 printed lines. The paragraph is deliberately long because splitting it would create a sub-90-word block.

---

### 1.1c — L420: delete after folding its content into L418

**Location:** L420, §3.3, immediately before Algorithm 2

**Current text (exact):**

```latex
Algorithm~\ref{alg:ecxgb} lists the procedure. Steps 1 and 2 build the feature matrix. Steps 3 to 9 repeat the fit for the three seeds: each repetition builds the 35-feature input, trains on HaluEval plus the RAGTruth non-QA rows, fits the display calibrator on the 5{,}034 QA calibration rows, and selects the source and target thresholds. The QA rows stay out of classifier training, which keeps the QA ranking zero-shot.
```

**Replacement:** delete the paragraph. Its content is folded into the replacement above.

**Why:** The algorithm pointer and the step-by-step description now sit at the end of the new L418. The final sentence about QA rows staying out of classifier training duplicated the zero-shot sentence already in L418, so it is not carried over.

**Note:** The Algorithm 2 float itself is unchanged.

---

### 1.2a — L640: fold the Figure 5 paragraph into the Table 7 argument

**Location:** L640, §5.2, Table 7 paragraph. The Figure 5 float at L642-647 stays where it is.

**Current text (exact):**

```latex
Table~\ref{tab:ablation} repeats the ablation on the fixed-hp m3 model, whose full reference reaches F1 0.9861, AUROC 0.9985, and MCC 0.9726. Lexical overlap is the largest contributor, followed by length and NLI. The claim block costs about one seed standard deviation, so an aggregate claim gain is not resolved; entity, numeric, hedging, semantic, and source each change F1 by less than 0.0011. Lexical overlap and length therefore carry the in-domain score, and the remaining groups add ranking information without moving F1 far. Deltas are relative to the full m3 refit under the same configuration, trained on the same 24{,}056 rows and evaluated once on 3{,}000 test rows without tuning or early stopping; the refit reference differs from Table~\ref{tab:ec} by 0.0007 F1, within its seed standard deviation of 0.0010.
```

**Replacement (exact):**

```latex
Table~\ref{tab:ablation} repeats the ablation on the fixed-hp m3 model, whose full reference reaches F1 0.9861, AUROC 0.9985, and MCC 0.9726. Lexical overlap is the largest contributor, followed by length and NLI. The claim block costs about one seed standard deviation, so an aggregate claim gain is not resolved. Entity, numeric, hedging, semantic, and source each change F1 by less than 0.0011. Lexical overlap and length therefore carry the in-domain score, and the remaining groups add ranking information without moving F1 far. Deltas are relative to the full m3 refit under the same configuration, trained on the same 24{,}056 rows and evaluated once on 3{,}000 test rows without tuning or early stopping. The refit reference differs from Table~\ref{tab:ec} by 0.0007 F1, within its seed standard deviation of 0.0010.

Figure~\ref{fig:perseed} shows the seed-level spread behind Table~\ref{tab:ec}. The F1 and AUROC panels are tight, with within-variant differences below 0.0007 F1 and 0.0005 AUROC. The ECE panel is noisier: m0 is stable across seeds, while m2 and m3 vary more. The mean ordering is m3 $>$ m1 $>$ m2 $>$ m0, but the per-seed ranks flip and the differences are below seed noise. The ECE sign is stable: m0 is lower than m3 on every seed, so the ablation does not establish a calibration advantage for m3.
```

**Why:** The old Table 7 paragraph ended with the refit caveat, and the Figure 5 paragraph then stood alone between two long ablation blocks. The new L640 keeps the Table 7 interpretation, carries the refit caveat, and then carries the seed-spread reading as a second paragraph. Splitting at the refit sentence gives 141 and 91 words, both above the review's 90-word floor. The figure float itself is unchanged.

**Length:** 141 and 91 words, about 9 and 6 printed lines.

**Note:** The old L649 paragraph is deleted by 1.2b. The `Figure~\ref{fig:perseed}` pointer now sits in the second paragraph, so the reader still reaches Figure 5.

---

### 1.2b — L649: delete the standalone Figure 5 paragraph

**Location:** L649, §5.2, immediately after the Figure 5 float

**Current text (exact):**

```latex
Figure~\ref{fig:perseed} shows the seed-level spread behind Table~\ref{tab:ec}. The F1 and AUROC panels are tight, with within-variant differences below 0.0007 F1 and 0.0005 AUROC. The ECE panel is noisier: m0 is stable across seeds, while m2 and m3 vary more. The mean ordering is m3 $>$ m1 $>$ m2 $>$ m0, but the per-seed ranks flip and the differences are below seed noise. The ECE sign is stable: m0 is lower than m3 on every seed, so the ablation does not establish a calibration advantage for m3.
```

**Replacement:** delete the paragraph. Its content is folded into the replacement above.

**Why:** The seed-level spread is now the second paragraph of the merged L640 block. Keeping the old paragraph would repeat the same 0.0007 F1, 0.0005 AUROC, and ECE-sign statements.

**Note:** Do not delete the Figure 5 float at L642-647. Only the paragraph is removed.

---

## Item 2 — Define the model labels and standardize the feature name

### 2.1 — L298: gloss m3 at its first use

**Location:** L298, Introduction P4, first sentence that reports a model result

**Current text (exact):**

```latex
The fixed-hyperparameter EC-XGB (m3) reaches F1 0.9854 and AUROC 0.9986 on HaluEval QA.
```

**Replacement (exact):**

```latex
The fixed-hyperparameter EC-XGB stack (m3) reaches F1 0.9854 and AUROC 0.9986 on HaluEval QA.
```

**Why:** m3 first appears here, roughly 300 lines before the ladder was defined. `Stack` tells the reader that m3 is the full EC-XGB combination rather than a separate learner. The full ladder is defined in the new L418, before m0, m1, and m2 are used.

---

### 2.2 — L418: define the m0-to-m3 ladder before first use

Covered by edit 1.1b. The new L418 opens with the ladder:

```latex
The ablation ladder starts from m0, the 26-feature conventional reference, and adds log-scaled length counts with monotone constraints at m1, the eight claim features at m2, and the RAGTruth non-QA rows with the source indicator at m3.
```

No separate edit is needed. This sentence makes m0, m1, m2, and m3 readable before §5.2, and it keeps the feature counts (26, 34, and 35) that the old L601 definition carried.

---

### 2.3 — L457: define B2 and B6 m0 at first use

**Location:** L457, §3.5, the sentence that compares the deployed checkpoint with the references

**Current text (exact):**

```latex
Section~\ref{sec:results} compares it with the conventional B2 checkpoint and with B6 m0, a 26-feature reference from the same tuning protocol that lacks the EC-XGB changes.
```

**Replacement (exact):**

```latex
Section~\ref{sec:results} compares it with B2, the conventional 26-feature checkpoint tuned on HaluEval QA, and with B6 m0, the fixed-hyperparameter 26-feature reference from the same tuning protocol that lacks the EC-XGB changes.
```

**Why:** B2 and B6 m0 are used throughout the results tables but were never expanded. The new sentence says what B2 is (the tuned 26-feature conventional checkpoint), what B6 m0 is (the fixed-hyperparameter reference), and keeps the existing point that m0 lacks the EC-XGB changes.

---

### 2.4 — L601: replace the repeated ladder definition with a back-reference

**Location:** L601, §5.2, second sentence

**Current text (exact):**

```latex
m0 is the conventional reference, m1 log-scales the length counts and adds monotone constraints, m2 adds the eight claim aggregates (34 features), and m3 adds the RAGTruth non-QA rows and the source indicator (35 features); ECE is computed on the raw score.
```

**Replacement (exact):**

```latex
The ladder from m0 to m3 is defined in Section~\ref{sec:proposed}, and ECE is computed on the raw score.
```

**Why:** The ladder is now defined in §3.3, before the results. Repeating it here made the labels arrive twice and encouraged the two definitions to drift. The new sentence keeps the ECE provenance and points back to the definition.

---

### 2.5 — L451: rename the corpus indicator

**Location:** L451, §3.4, sentence about the target calibration rows

**Current text (exact):**

```latex
Those rows are scored with the corpus indicator set to one
```

**Replacement (exact):**

```latex
Those rows are scored with the source indicator set to one
```

**Why:** The paper uses `source indicator` in the abstract, the method, the algorithm, and the results. `Corpus indicator` is the only second name for the same binary feature, and it suggests that the feature encodes corpus identity rather than the HaluEval-versus-other split.

---

### 2.6a — L479: rename the corpus indicator in the contribution list

**Location:** L479, §3.5, first sentence of the contribution paragraph

**Current text (exact):**

```latex
multi-source training with a corpus indicator
```

**Replacement (exact):**

```latex
multi-source training with a source indicator
```

**Why:** Same feature, same reason as 2.5. Standardizing the name removes the last two uses of `corpus indicator` in the manuscript.

---

### 2.6b — L479: rename the second corpus indicator

**Location:** L479, §3.5, sentence about the seen-task gain

**Current text (exact):**

```latex
adds the RAGTruth rows and the corpus indicator
```

**Replacement (exact):**

```latex
adds the RAGTruth rows and the source indicator
```

**Why:** Same feature as 2.6a. This is the second and last occurrence in L479.

---

## Item 3 — Make the conclusion do new work

### 3.1 — L998: rewrite the conclusion around deployment meaning and next tests

**Location:** L998, Conclusion, the whole paragraph

**Current text (exact):**

```latex
This study tested whether a simple feature model can estimate hallucination risk and how its calibration and ranking change under domain and task shift. EC-XGB combines 26 base features, eight claim-level NLI aggregates, and a source indicator, with calibration and SHAP explanations. On HaluEval QA the fixed-hp model reaches F1 0.9854 and AUROC 0.9986, but lexical overlap and length carry that score. On RAGTruth the tuned model cuts the pooled flag rate from 99.9\% to 61.3\%, but the fixed-hp model still flags 96.8\% of held-out QA at AUROC 0.558. Target Platt calibration lowers ECE from 0.731 to 0.125, but flags no target rows at 0.5. Split conformal covers 94.4\% at 90\%, but positive coverage is only 0.685. On MedHallu it ranks the hallucinated answer above the grounded one in 86.7\% of pairs, but flags every row and is badly calibrated (ECE 0.493). The ranking survives the domain change, but the probability does not. Dialogue and summarization fail under task shift. EC-XGB is therefore a domain-specific risk score, not a general-purpose detector. Future work should test adaptation to unseen QA domains and select thresholds under prevalence shift.
```

**Replacement (exact):**

```latex
This study tested EC-XGB, a 35-feature evidence-consistent model, as a hallucination risk score and asked how its calibration and ranking change under domain and task shift. Its ranking survives a change of corpus, so the score can order answers for review, and the per-claim verdicts give a reviewer a reason for each flag. Its probability does not survive that change: the display score needs labeled target rows, and the target prevalence then sets its level. EC-XGB is therefore a domain-specific screening score, not a general-purpose detector. The next test is adaptation to an unseen QA domain, with thresholds selected under prevalence shift. A seen-task source ablation would then separate the RAGTruth rows from the source indicator.
```

**Why:** The old conclusion repeated five abstract results almost verbatim: F1 0.9854, the 99.9-to-61.3 flag drop, 96.8\% at AUROC 0.558, ECE 0.731-to-0.125, conformal 94.4/0.685, and MedHallu 86.7 with ECE 0.493. The new paragraph keeps the objective and the contribution, states the ranking-versus-probability finding once, turns it into a deployment reading (review ordering, per-claim reasons, labeled target rows, prevalence-dependent level), and ends with two concrete next tests. It contains no repeated number.

**Length:** 116 words, about 8 printed lines.

---

### 3.2 — Abstract: no change

The abstract is the summary and keeps its five headline results, so the redundancy is removed at the conclusion rather than by shortening the abstract. The abstract remains 281 words and still answers the six guideline questions in order.

---

## Item 4 — Let the tables carry the numbers

### 4.1 — L559: one ranking headline and one calibration interpretation

**Location:** L559, §5.1, the whole paragraph

**Current text (exact):**

```latex
Table~\ref{tab:models} reports the HaluEval test results for eleven baselines and EC-XGB, averaged over three seeds where a seed applies. The F1 ranking is tight: the random forest is 0.0004 behind the conventional model, and EC-XGB is 0.0009 ahead. That lead is much smaller than the width of the conventional seed-42 bootstrap interval [0.9812, 0.9895], so the ordering is not resolved by the data. The benchmark is saturated, and the differences are small. The ranking columns tell the same story. EC-XGB is first among the feature-based models on AUROC and PR-AUC, with the conventional model and random forest close behind; RoBERTa is slightly higher. Calibration is separate: the conventional model has the lowest ECE, but its Brier score is higher than EC-XGB and RoBERTa, so the two summaries disagree. The weak controls are NLI-only and full-input TF-IDF, while answer-only TF-IDF is much stronger, so adding the question and context hurts. The reduced external ensemble reaches F1 0.9517 and AUROC 0.9850; the fixed-hyperparameter (fixed-hp) m3 model is higher by 0.0337 F1 [0.0262, 0.0419] and 0.0135 AUROC [0.0098, 0.0177], with ECE 0.0078 against 0.0141. The gap is consistent with the nonlinear learner and wider feature set, but the baseline is reduced and trained on fewer rows, so the comparison is descriptive. The EC-XGB row is the tuned m3 checkpoint, the conventional row is B2, and the m3 row in Table~\ref{tab:ec} is the fixed-hp refit. ECE and Brier use the raw score, and the length-only control reports neither. That control is a single deterministic logistic regression on the two counts at 0.5. Figure~\ref{fig:rocpr} shows the curves behind Table~\ref{tab:models}: the feature-based models overlap near the top-left corner on HaluEval, while in RAGTruth EC-XGB is above the conventional model on summarization and data-to-text and both stay close to the diagonal on QA. The curves use seed 42, and the table values are three-seed means.
```

**Replacement (exact):**

```latex
Table~\ref{tab:models} reports the HaluEval test results for eleven baselines and EC-XGB, averaged over three seeds where a seed applies. EC-XGB leads the conventional model by 0.0009 F1, which is smaller than the width of the conventional seed-42 bootstrap interval [0.9812, 0.9895], so the benchmark does not resolve the ordering. The ranking columns repeat that pattern: EC-XGB is first among the feature-based models on AUROC and PR-AUC, the conventional model and random forest are close behind, and RoBERTa is slightly higher. Calibration gives a different summary, because the conventional model has the lowest ECE while EC-XGB and RoBERTa have lower Brier scores. The weak controls are NLI-only and full-input TF-IDF, while answer-only TF-IDF is much stronger, so adding the question and context hurts. The reduced external ensemble is below EC-XGB by 0.0337 F1 [0.0262, 0.0419] and 0.0135 AUROC [0.0098, 0.0177], but it is trained on fewer rows and with a narrower feature set, so that gap is descriptive. Figure~\ref{fig:rocpr} shows the curves behind the table, and the curves use seed 42 while the table values are three-seed means.
```

**Why:** The old paragraph transcribed the F1 gaps, the bootstrap interval, the external-ensemble gap, its AUROC interval, and two ECE values. The new paragraph keeps the 0.0009 F1 lead and the interval that shows it is unresolved, states the ranking and calibration interpretations, and keeps the external-ensemble comparison with its two intervals because those intervals are not in Table 5. The remaining table cells are left to Table 5.

**Length:** 192 words, about 12 printed lines. The paragraph stays above the floor because it carries three jobs: ranking, calibration, and the external baseline.

**Note:** The external-ensemble intervals are prose-only, so they are kept. The 0.0004 random-forest gap, the 0.9517/0.9850 point values, and the ECE/Brier point values are in Table 5 or follow from it. The fixed 0.5 threshold is stated in the table caption and in §4.

---

### 4.2 — L672: split the calibration shift from the operating point

**Location:** L672, §5.3, the whole paragraph

**Current text (exact):**

```latex
Source Platt cannot repair the base-rate shift. The fixed-hp m3 raw score is severely overconfident on RAGTruth QA: ECE 0.731 and Brier 0.689, with a Cox slope of 0.18 on the 5{,}034 calibration rows. Target Platt lowers ECE to 0.125 and Brier to 0.161, but the Brier skill score against the test prior is $-$0.10, and a constant predictor at the test prevalence has Brier 0.146, lower than every calibrated row. Calibration fixes the scale, not the ranking; AUROC stays at 0.558. The conventional target-Platt row behaves the same way at ECE 0.134 and Brier 0.164. Figure~\ref{fig:calibshift} shows the raw and target-recalibrated scores. The source-only operating point (P0) selects the threshold on the HaluEval validation split at a 5\% false-positive budget. On HaluEval test, recall is 99.0--99.5\% and FPR stays near 5\%, but on RAGTruth QA the same threshold flags all 900 rows of the deployed seed-42 checkpoint, because every score exceeds 0.095, so P0 is not usable. Table~\ref{tab:target} reports both models on the same test set. A prevalence-only predictor at 0.1778 reaches ECE 0.000 and Brier 0.146 and predicts no positives; at the calibration prevalence of 0.3107 it gives ECE 0.133 and Brier 0.164, so the calibrator sets the level to the calibration prevalence and the residual ECE is dominated by the prevalence gap. The conventional target-thresholded mode (P2) selects 0.313 at a 5\% false-positive budget and reaches 5.0\% recall at 5.8\% FPR (precision 15.7\%, F1 0.076). At 17.8\% prevalence, random flagging at 5.8\% FPR yields 5.8\% recall, so P2 is chance-level. At 0.5, P1 flags no RAGTruth QA rows; for the deployed seed-42 calibrator, the maximum target-Platt score is 0.355 (35.4\% before rounding, 35\% on the card), the median is 0.331, and 16\% of QA rows reach 0.35, so the card usually relies on the per-claim verdict adjustment. Figure~\ref{fig:operating} shows the operating point and label budget: mean ECE is 0.130 at 50 rows, 0.124 at 252 rows, and 0.125 at 5{,}034 rows, but a single 252-row draw can miss the full-split ECE by about 0.06. All calibration metrics refer to the pre-verdict score; the displayed percentage is a design aid, not a validated probability.
```

**Replacement (exact):**

```latex
Source Platt cannot repair the base-rate shift. The fixed-hp m3 raw score is severely overconfident on RAGTruth QA, and target Platt lowers ECE from 0.731 to 0.125. The Brier skill score against the test prior is $-$0.10, and a constant predictor at the test prevalence has a lower Brier score than every calibrated row. Calibration therefore sets the level to the calibration prevalence, not the test prevalence, so the residual ECE follows the prevalence gap, and it leaves AUROC at 0.558. The conventional target-Platt row behaves the same way. Figure~\ref{fig:calibshift} shows the raw and target-recalibrated scores, and Table~\ref{tab:target} reports both models on the same test set.

The source-only operating point (P0) selects its threshold on the HaluEval validation split at a 5\% false-positive budget. On HaluEval test, recall is 99.0--99.5\% and FPR stays near 5\%, but on RAGTruth QA the same threshold flags all 900 rows of the deployed seed-42 checkpoint, because every score exceeds 0.095, so P0 is not usable. The conventional target-thresholded mode (P2) selects 0.313 at a 5\% false-positive budget and reaches 5.0\% recall at 5.8\% FPR (precision 15.7\%, F1 0.076), which is chance-level at 17.8\% prevalence. At 0.5, P1 flags no RAGTruth QA rows. For the deployed seed-42 calibrator, the maximum target-Platt score is 0.355 (35.4\% before rounding, 35\% on the card), the median is 0.331, and 16\% of QA rows reach 0.35, so the card usually relies on the per-claim verdict adjustment. Figure~\ref{fig:operating} shows the label budget: mean ECE is 0.130 at 50 rows, 0.124 at 252 rows, and 0.125 at 5{,}034 rows, but a single 252-row draw can miss the full-split ECE by about 0.06. All calibration metrics refer to the pre-verdict score, and the displayed percentage is a design aid, not a validated probability.
```

**Why:** The old paragraph narrated the prevalence-only predictor, the P0/P2 operating points, the P1 empty flag set, the maximum score, and the label budget in one 336-word block. The new version separates the calibration result from the operating-point result. The first paragraph keeps the 0.731-to-0.125 headline, the negative BSS, and AUROC 0.558. The second keeps the P0/P2/P1 results and the label-budget caveat, and leaves the prevalence-only rows to Table 8; the label-budget values stay because they are prose-only.

**Length:** 116 and 207 words, about 7 and 13 printed lines. Both are above the floor.

**Note:** The current L672 already contains `results_patch.md` 1A and 1B: all 900 rows at 0.095, maximum 0.355, median 0.331, 16\% at 0.35, and 0.313 only in the P2 sentence. This replacement keeps those values and only splits and trims the surrounding prose.

---

### 4.3 — L887: one latency headline and one cost interpretation

**Location:** L887, §5.5, the whole paragraph

**Current text (exact):**

```latex
Table~\ref{tab:latency} gives the component latency. Components are timed in separate instrumented passes over the same samples, so they do not sum to the end-to-end median; retrieval and judge calls are excluded. NLI and NER dominate the base median, while XGBoost and SHAP are negligible. Adding the eight claim features raises the m3 path to 127.5\,ms at the median, with a p95 of 234.6\,ms; claim NLI alone is 43.8\,ms. The artifact is 0.68\,MB, and the marginal cost is about \$0.001 per 1{,}000 predictions. The same 200-row protocol on CPU takes 2.00\,s per answer in a single contended run (p50 0.98\,s, p95 7.56\,s), so the 127.5\,ms figure depends on GPU feature extraction; the classifier and SHAP steps remain negligible. Feature extraction, not the model, is the cost. Table~\ref{tab:cost} compares cost and latency: classical heads run in under a millisecond, the m3 pipeline is 127.5\,ms, and the two judges are 18 to 24 times slower. RoBERTa-base inference on the same 200 rows takes 6.97\,ms per row at batch 8 and fp16, so the m3 feature pipeline is about 18 times slower than the encoder alone. The encoder still needs a fine-tuned classification head, and its training cost is not included. Size is the seed-42 artifact on disk; model-only m3 inference is 0.008\,ms, and the NLI-only head was not persisted, so its cost is the NLI step plus a linear head. RoBERTa timing is encoder-only at 512 tokens, an upper bound for the 256-token baseline, and costs use the recorded token counts and an assumed \$1.20/M output price.
```

**Replacement (exact):**

```latex
Table~\ref{tab:latency} gives the component latency. Components are timed in separate instrumented passes over the same samples, so they do not sum to the end-to-end median, and retrieval and judge calls are excluded. NLI and NER dominate the base median, while XGBoost and SHAP are negligible. Adding the eight claim features raises the m3 path to 127.5~ms at the median. The same 200-row protocol on CPU takes 2.00~s per answer in a single contended run (p50 0.98~s, p95 7.56~s), so this figure depends on GPU feature extraction, and feature extraction, not the model, is the cost. Table~\ref{tab:cost} compares cost and latency: classical heads run in under a millisecond, and the two judges are 18 to 24 times slower than the m3 pipeline. The encoder alone takes 6.97~ms per row at batch 8 and fp16, so the feature pipeline is about 18 times slower, but the encoder still needs a fine-tuned classification head and its training cost is not included. Model-only m3 inference is 0.008~ms, and the NLI-only head is the NLI step plus a linear head. Size is the seed-42 artifact on disk. The cost uses recorded token counts and an assumed \$1.20/M output price, and the encoder timing is at 512 tokens, an upper bound for the 256-token baseline.
```

**Why:** The old paragraph listed the m3 median, p95, claim-NLI time, artifact size, marginal cost, CPU p50 and p95, the RoBERTa time, the 18-times ratio, the model-only time, the token lengths, and the assumed price. The new paragraph keeps the 127.5 ms headline, the CPU 2.00 s caveat with its two percentiles, and the judge comparison, then states the interpretation: feature extraction, not the model, is the cost. Size and the marginal cost stay in Table 16; the encoder timing, the model-only time, and the 512/256-token caveat stay in prose.

**Length:** 229 words, about 14 printed lines.

**Note:** The p95 234.6 ms and claim-NLI 43.8 ms are in Table 15, and the size, cost, and RoBERTa columns are in Table 16. The CPU percentiles, the model-only 0.008 ms, and the 512/256-token caveat are prose-only, so they are kept.

---

### 4.4 — L932: one judge headline and one length-matched interpretation

**Location:** L932, §5.5, the whole paragraph

**Current text (exact):**

```latex
LLM judges are not fair evaluators, and their preferences can shift with answer order \cite{wang2024fair}. Table~\ref{tab:judge} compares EC-XGB with two judges on the 900 RAGTruth QA rows at 17.8\% prevalence. GPT-5.6 Luna reaches F1 0.565 and AUROC 0.910; Qwen 3.5 4B reaches 0.602 and 0.778; EC-XGB reaches 0.310 and 0.558. Both judges outperform EC-XGB on F1 and AUROC, and the target-fitted length-only baseline at AUROC 0.712 also ranks the QA rows better than the full feature model. The feature model is cheaper, but it is not the better QA detector. Under matched multi-source training, the adapted encoder reaches 0.694 on the pooled RAGTruth test set, above EC-XGB's 0.582, so the encoder is stronger there; the feature model is less degraded on unseen tasks, but both are near chance. The length-matched panel changes the ranking: on the 158-row exact-length set, EC-XGB matches the stronger judge on AUROC and leads on F1, with a 0.042 higher seed-mean F1. The paired interval for GPT minus EC-XGB is $-$0.052 [$-0.127$, 0.024], which includes zero; the AUROC interval is $+$0.008 [$-0.034$, 0.048]. The natural-prevalence QA row is a domain-shift result, not a general judge ranking. Prompt choice moves the judge: GPT's F1 ranges from 0.814 to 0.854 across three prompts, about the size of the model gap. GPT uses default decoding and three seeds; Qwen is a single deterministic run. The lower panel controls length but not domain, so it is a stress test, not a second natural-prevalence estimate.
```

**Replacement (exact):**

```latex
LLM judges are not fair evaluators, and their preferences can shift with answer order \cite{wang2024fair}. Table~\ref{tab:judge} compares EC-XGB with two judges on the RAGTruth QA rows at natural prevalence. GPT-5.6 Luna reaches AUROC 0.910 against 0.558 for EC-XGB, and Qwen 3.5 4B also outperforms it, so both judges are better QA detectors and the feature model's advantage is cost. The target-fitted length-only baseline also ranks the QA rows better than the full feature model. Under matched multi-source training, the adapted encoder reaches 0.694 on the pooled RAGTruth test set, above EC-XGB's 0.582, so the encoder is stronger there. The feature model is less degraded on unseen tasks, but both are near chance. The length-matched panel changes the ranking: on the exact-length set, EC-XGB matches the stronger judge on AUROC and leads on F1, so the natural-prevalence gap is a domain-shift result rather than a length effect. The paired interval for GPT minus EC-XGB is $-$0.052 [$-0.127$, 0.024], and the AUROC interval is $+$0.008 [$-0.034$, 0.048]. Both include zero. Prompt choice moves GPT's F1 from 0.814 to 0.854 across three prompts, about the size of the model gap. GPT uses default decoding and three seeds, while Qwen is a single deterministic run, so the judge ranking is not stable.
```

**Why:** The old paragraph read the judge F1 and AUROC cells, the length-only AUROC, the adapted-encoder comparison, the length-matched F1 lead, two paired intervals, the prompt range, and the seed setting. The new paragraph keeps one judge comparison (0.910 against 0.558), the adapted-encoder result with its corrected pooled label, and the length-matched reversal, then interprets the reversal as a domain effect rather than a length effect. The remaining table cells stay in Table 17.

**Length:** 226 words, about 14 printed lines.

**Note:** The current L932 already contains `results_patch.md` 3C2: `pooled RAGTruth test set`, not `seen RAGTruth tasks`. The 900 rows, 17.8\%, the Qwen F1 and AUROC, the length-only AUROC, and the 158-row label are in Table 17 or its caption. The paired intervals, the prompt range, and the seed settings are prose-only, so they are kept.

---

### 4.5 — L960: replace the length-table transcript and the `Four controls` frame

**Location:** L960, §5.6, the whole paragraph

**Current text (exact):**

```latex
On the seed-42 fixed-hyperparameter run, m3 makes 43 errors on the HaluEval test set (Figure~\ref{fig:confusion}), an error rate of 1.4\%. The WP5 audit covers seven of the eight false positives and 12 of the 35 false negatives, chosen to cover the codebook. A separate 50-case sheet, reviewed by one author, adds 31 artifact cases and 19 content cases across 12 categories. These describe the reviewed cases, not the 3{,}000-row test population. Table~\ref{tab:length} shows why length matters: the hallucination rate rises from 2.1\% for one-word answers to above 90\% for five or more words, false positives concentrate in the five-to-eight bin, and false negatives concentrate in the short bins. Length is therefore a near-perfect label indicator on HaluEval, although the 17+ bin has only two negative rows. Four controls test this explanation. A logistic regression on the two length variables reaches F1 0.9457 and AUROC 0.9759. A reweighting control equalizes six length bins with 247 examples each; EC-XGB gains +0.039 F1 and +0.023 AUROC over a length-only baseline. A length-matching control pairs one positive and one negative at each length, giving 79 pairs (158 examples) over 1{,}000 bootstraps. EC-XGB holds AUROC 0.956 while the length-only baseline falls to 0.521, a point gap of 0.435 (bootstrap mean 0.436 [0.351, 0.522]). The F1 point gap is 0.431 (bootstrap mean 0.440 [0.334, 0.552]). The length cue explains the zero-shot failure but not the in-domain ranking.
```

**Replacement (exact):**

```latex
On the seed-42 fixed-hyperparameter run, m3 makes 43 errors on the HaluEval test set (Figure~\ref{fig:confusion}), an error rate of 1.4\%. The WP5 audit covers seven of the eight false positives and 12 of the 35 false negatives, chosen to cover the codebook. A separate 50-case sheet, reviewed by one author, adds 31 artifact cases and 19 content cases across 12 categories. These describe the reviewed cases, not the 3{,}000-row test population. Table~\ref{tab:length} shows that length is a near-perfect label indicator on HaluEval, with false positives concentrated in the five-to-eight bin and false negatives in the short bins, although the 17+ bin has only two negative rows. The controls change the length distribution in two ways. A reweighting control equalizes six length bins with 247 examples each, and a length-matching control pairs one positive and one negative at each length. Reweighting gives EC-XGB a gain of 0.039 F1 and 0.023 AUROC over the length-only regression, which is already competitive on its own (Table~\ref{tab:models}). Length matching gives 79 pairs (158 examples) over 1{,}000 bootstraps. EC-XGB holds AUROC 0.956 while the length-only baseline falls to 0.521. The AUROC point gap is 0.435 (bootstrap mean 0.436 [0.351, 0.522]), and the F1 point gap is 0.431 (bootstrap mean 0.440 [0.334, 0.552]).
```

**Why:** The old paragraph recited the length-bin rates (2.1\%, above 90\%, the five-to-eight and short-bin concentrations), then announced `Four controls` and listed three. The new version points to Table 18 for the bin pattern, names the two controls that actually change the length distribution, and reports their results. The audit counts stay because they are not in a table and because the error-enriched sample caveat depends on them.

**Length:** 230 words, about 14 printed lines.

**Note:** The current L960 already contains `results_patch.md` 4 (`seven of the eight`) and `provenance_patch.md` 2 (`reviewed by one author`). The 2.1\% and above-90\% bin rates are in Table 18; the point gaps and bootstrap means are prose-only and are kept.

---

### 4.6 — L994: drop the repeated flag-rate transcript and fix `they`

**Location:** L994, §5.6, the whole paragraph

**Current text (exact):**

```latex
Figure~\ref{fig:length} shows the same pattern. Complete sentences are long, so the conventional model reads them as hallucinations. The ablation leaves lexical overlap and length as the two groups that matter, so the in-domain model is mostly a length and overlap detector. Hence the conventional model flags 99.9\% of natural long answers and EC-XGB still flags 61.3\%. Calibration cannot repair the ranking, and held-out QA stays near 0.55. Published HaluEval QA results use different splits, supervision, and evidence access. For example, a supervised 13B agent reports F1 0.724 on 100 rows, so they are context rather than a head-to-head bar. The benchmark also asks for hallucinated answers about five words longer; the observed gap is 8.5 words, and a length-only model reaches AUROC 0.976 (Table~\ref{tab:models}). Any comparison that ignores this construction artifact overstates semantic detection. Several limits bound these claims. Retrieval quality and end-to-end verdict accuracy are not evaluated. The judge comparison controls length but not domain, and the audit is error-enriched. HaluEval is saturated, so its in-domain differences are not resolvable. The deployed display calibrator was selected in an earlier fit that compared Platt and isotonic on the test split. Table~\ref{tab:target} refits Platt without that selection.
```

**Replacement (exact):**

```latex
Figure~\ref{fig:length} shows the same pattern. Complete sentences are long, so the conventional model reads them as hallucinations, and the in-domain model is mostly a length and overlap detector. Published HaluEval QA results use different splits, supervision, and evidence access. A supervised 13B agent reports F1 0.724 on 100 rows, so these published numbers are context rather than a head-to-head bar. The benchmark also asks for hallucinated answers about five words longer, and the observed gap is 8.5 words, so any comparison that ignores this construction artifact overstates semantic detection. The claims in this paper are bounded by the tests that were not run. Retrieval quality and end-to-end verdict accuracy are not evaluated, the judge comparison controls length but not domain, and the audit is error-enriched. The deployed display calibrator was selected in an earlier fit that compared Platt and isotonic on the test split, and Table~\ref{tab:target} refits Platt without that selection.
```

**Why:** The old paragraph repeated the 99.9/61.3 flag rates and the 0.55 held-out AUROC, which are already in Table 6, and recited the length-only AUROC 0.976, which is in Table 5. It also used `they` for the published HaluEval results. The new paragraph keeps the length interpretation, the published 13B F1 0.724 on 100 rows, the 8.5-word construction gap, and the three limits, and names the published numbers as the referent.

**Length:** 159 words, about 11 printed lines.

**Note:** The 99.9/61.3 and 0.55 values remain in Table 6; 0.976 remains in Table 5. The 13B F1, the 100 rows, the five-word construction, and the 8.5-word gap are prose-only and are kept.

---

## Item 5 — Fix the ambiguous pronouns

### 5.1 — L362: `it` to `this probability`

**Location:** L362, §3.1, sentence that introduces the pipeline figure

**Current text (exact):**

```latex
Figure~\ref{fig:pipeline} shows the pipeline that computes it.
```

**Replacement (exact):**

```latex
Figure~\ref{fig:pipeline} shows the pipeline that computes this probability.
```

**Why:** `It` can refer to evidence consistency, the probability, or the flag. The sentence introduces the pipeline, so the ambiguity blocks the figure. `This probability` names the output that the pipeline computes.

---

### 5.2 — L715: `It` to `The marginal guarantee`

**Location:** L715, §5.3, sentence after the 0.944 coverage result

**Current text (exact):**

```latex
It hides the class split: positive coverage is 0.685, while negative coverage is 1.000.
```

**Replacement (exact):**

```latex
The marginal guarantee hides the class split: positive coverage is 0.685, while negative coverage is 1.000.
```

**Why:** `It` can refer to the 0.944 coverage, the marginal guarantee, or the split. The sentence is about the guarantee, and naming it keeps the positive/negative contrast attached to the right subject.

---

### 5.3 — L815: `That` to the named comparison

**Location:** L815, §5.4, after the two p-values

**Current text (exact):**

```latex
That is the primary comparison, and it is not part of the Holm family, which covers the four baseline comparisons against the conventional model. The fixed-hp row compares m0 and m3 under the single configuration of Table~\ref{tab:ec}.
```

**Replacement (exact):**

```latex
The fixed-hp m0-vs-m3 comparison is the primary test, and it is not part of the Holm family, which covers the four baseline comparisons against the conventional model. The test uses the single configuration of Table~\ref{tab:ec}.
```

**Why:** `That` follows $p=0.45$ and $p=0.789$, so it could name either test. The new sentence names the fixed-hp m0-vs-m3 comparison, and the following sentence no longer repeats the same row definition.

---

### 5.4 — L994: `they` to `these published numbers`

Covered by edit 4.6. The replacement sentence is:

```latex
A supervised 13B agent reports F1 0.724 on 100 rows, so these published numbers are context rather than a head-to-head bar.
```

The old `they` could refer to the 13B agent, the 100 rows, or the published HaluEval results. The new wording names the published numbers.

---

## Item 6 — Replace the formulaic frames

### 6.1 — L404: drop `First` and `Second`

**Location:** L404, §3.3, first two sentences and the claim-feature sentence

**Current text (exact):**

```latex
EC-XGB changes the conventional setup in three places. First, the character and word counts are log-scaled, and monotone constraints are placed on the evidence features. The constraints are negative on context overlap, context Jaccard, context entailment, context cosine, claim minimum and mean entailment, and the supported-claim ratio, and positive on context contradiction, novel-entity ratio, novel numbers, claim maximum and mean contradiction, and the contradicted-claim ratio. All other features are unconstrained. Second, the eight claim features in Table~\ref{tab:claim} are added: seven NLI aggregates and a claim count. They use the same NLI scorer and support-first rule as the verification layer. A claim is supported when any retained sentence has an entailment probability of at least 0.5 and at least as high as its contradiction probability. It is contradicted when no sentence supports it and a sentence has a contradiction probability of at least 0.5 and above its entailment probability. Otherwise it is unsupported.
```

**Replacement (exact):**

```latex
EC-XGB changes the conventional setup in its feature treatment and its training rows. The character and word counts are log-scaled, and monotone constraints are placed on the evidence features. The constraints are negative on context overlap, context Jaccard, context entailment, context cosine, claim minimum and mean entailment, and the supported-claim ratio, and positive on context contradiction, novel-entity ratio, novel numbers, claim maximum and mean contradiction, and the contradicted-claim ratio. All other features are unconstrained. The eight claim features in Table~\ref{tab:claim} are added: seven NLI aggregates and a claim count. They use the same NLI scorer and support-first rule as the verification layer. A claim is supported when any retained sentence has an entailment probability of at least 0.5 and at least as high as its contradiction probability. It is contradicted when no sentence supports it and a sentence has a contradiction probability of at least 0.5 and above its entailment probability. Otherwise it is unsupported.
```

**Why:** The old opening announced three places and then enumerated them with `First` and `Second`; the `Third` arrived 14 lines later after two equations. The new opening names the two areas that actually change (feature treatment and training rows), and the claim-feature sentence drops `Second`. The `Third` is removed by edit 1.1b.

**Length:** 160 words, about 10 printed lines.

**Note:** The paragraph keeps every constraint, every feature name, and every threshold. Only the ordinal scaffolding changes.

---

### 6.2 — L418: `Third` removed

Covered by edit 1.1b. The new L418 begins `The training set adds the RAGTruth summarization and data-to-text rows.` and contains no ordinal.

### 6.3 — L960: `Four controls` removed

Covered by edit 4.5. The new L960 says `The controls change the length distribution in two ways.` and names the reweighting and length-matching controls. The old text announced four controls but listed three.

### 6.4 — L994: `Several limits` removed

Covered by edit 4.6. The new L994 replaces `Several limits bound these claims.` with `The claims in this paper are bounded by the tests that were not run.` and then names the retrieval, judge-domain, and audit-sampling limits in separate clauses.

---

## Summary

| Edit | Location | What it does | New paragraph words |
| :--- | :--- | :--- | ---: |
| 1.1a | L416 | Leaves only the equation note | 58 (equation note, exempt) |
| 1.1b | L418 | Combines the multi-source change, ladder, and Algorithm 2 | 215 |
| 1.1c | L420 | Deletes the duplicate algorithm paragraph | — |
| 1.2a | L640 | Merges Table 7 and Figure 5, split at a natural seam | 141 + 91 |
| 1.2b | L649 | Deletes the standalone figure paragraph | — |
| 2.1 | L298 | Glosses m3 | — |
| 2.2 | L418 | Defines the ladder (inside 1.1b) | — |
| 2.3 | L457 | Defines B2 and B6 m0 | — |
| 2.4 | L601 | Replaces the duplicate ladder with a back-reference | — |
| 2.5 | L451 | Renames corpus indicator | — |
| 2.6a/b | L479 | Renames corpus indicator twice | — |
| 3.1 | L998 | Rewrites the conclusion around deployment and next tests | 116 |
| 4.1 | L559 | One ranking headline and one calibration interpretation | 192 |
| 4.2 | L672 | Splits calibration from the operating point | 116 + 207 |
| 4.3 | L887 | One latency headline and one cost interpretation | 229 |
| 4.4 | L932 | One judge headline and one length-matched interpretation | 226 |
| 4.5 | L960 | Replaces the length-table transcript and the control frame | 230 |
| 4.6 | L994 | Drops the repeated flag-rate transcript and fixes they | 159 |
| 5.1 | L362 | Names the probability | — |
| 5.2 | L715 | Names the marginal guarantee | — |
| 5.3 | L815 | Names the m0-vs-m3 comparison | — |
| 5.4 | L994 | Names the published numbers (inside 4.6) | — |
| 6.1 | L404 | Drops First and Second | 160 |
| 6.2 | L418 | Drops Third (inside 1.1b) | — |
| 6.3 | L960 | Drops Four controls (inside 4.5) | — |
| 6.4 | L994 | Drops Several limits (inside 4.6) | — |

Body paragraphs after the patch: 91, 108, 116, 116, 141, 159, 160, 191, 192, 207, 215, 226, 229, and 230 words. The only block below 90 words is the 58-word L416 equation note, which the review excludes from its paragraph count.

## Items not changed without author input

- **Abstract.** `We present EC-XGB` is first person, and the six-item evaluation list is hard to read in one pass (review W18). Changing either would alter the 281-word abstract and the six-question structure, so it needs an author decision. The conclusion rewrite already removes the redundancy, so the abstract can stay as the summary.
- **L296, four-issue frame.** The paragraph is 288 words and carries the gap list, the baseline limits, the method, and the results preview (review W3). Splitting it is a structural change to the Introduction, not a sentence-level fix, and the four issues would need to be mapped to the design choices. This patch does not touch it.
- **L517 and L815 splits.** Both paragraphs have multiple jobs (review W6). This patch fixes the L815 pronoun but does not move the test definitions to §4.4 or split the McNemar and bootstrap blocks, because that reorders the statistics section and needs author approval.
- **L561 unlabeled encoder values.** `(0.490 vs 0.549)` still has no model names (review W8). The fix is a one-sentence clarification, but it is outside the four pronouns assigned here and would overlap the encoder comparison in `results_patch.md` 3 and 3C1.
- **L859 audit provenance.** The sentence now uses the approved pre-annotation wording from `provenance_patch.md` 1. This writing patch does not touch it, so the provenance patch remains the source of truth there.
- **L672 corrected values.** The current L672 already contains the corrected P0 and P1 values from `results_patch.md` 1A and 1B. This patch preserves them while splitting the paragraph. Reopening those values is a separate number decision, not a writing choice.
- **L513, source-indicator values.** The rename to `source indicator` is applied, but the evaluation assignment there (`1 for FaithBench and MedHallu and 0 for HaluBench and the task-shift sets`) is not the same as the training rule in the new L418 (`1` for the added RAGTruth rows, `0` for HaluEval). This patch does not change values. The author should confirm the intended coding before the table captions or the API description are finalized.
- **L994 HaluEval saturation sentence.** The new paragraph drops `HaluEval is saturated, so its in-domain differences are not resolvable.` because the same point is made in §5.1 and in the conclusion's ranking-versus-probability reading. If the author wants it in the error analysis, it should be restored as a separate sentence, not as a fourth limit in the same list.
- **Table 7 split seam.** The merge is split after the refit sentence, giving 141 and 91 words. If the author prefers one block, keep the two paragraphs joined; both versions are above the 90-word floor.
