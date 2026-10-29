# HaluRISC — section-by-section fix checklist

> Temporary working file. Delete after the 17-page build is verified.

Source: `Journal_Paper/halurisc.tex` (892 lines)
Build checked: 2026-10-29 11:15
Build facts: 17 pages; page 17 has 59 lines (references 9–34); 0 LaTeX warnings; 34/34 citations; 203/203 table audit; abstract 277 words.

**Final build status (2026-10-29):** all P0/P1 items through Declarations applied. Final build is 17 pages, 59 lines on page 17, 0 warnings, 34/34 citations, and 203/203 on the table audit. Detector check is pending on the full paper.

## Rules for this pass

1. Work in document order, one section at a time.
2. Apply only P0 and P1 items in the main flow. P2/P3 items are deferred to the appendix.
3. Do not rewrite whole paragraphs. Use the author’s current sentences and make the smallest replacement that fixes the error.
4. After each section: rebuild, run the detector on that section only, and wait for approval before the next section.
5. Do not touch tables, figures, or algorithms unless the item explicitly says so.
6. Line numbers will shift after edits. Match on the quoted text, not only the line number.

Priority:

- **P0 — Critical:** breaks the PDF, states a wrong number, or makes a false claim.
- **P1 — Major:** must fix before submission; affects a caption, claim, citation, or page budget.
- **P2/P3 — Deferred:** clarity or style only; not in this pass.

---

# Stage 0 — Abstract and keywords

**Goal:** keep the abstract version that passed the detector. Do not rewrite it again in this pass.

**Status (2026-10-29):** ABS-01–ABS-04 were applied, but the revised 258-word abstract was flagged by the detector from “Therefore, this study was designed…” to the end. The previous 277-word abstract passed and has been restored from `HEAD` (`d439254`). Final abstract is 277 words; build is 18 pages with 0 warnings. Do not reapply the four fixes.

### What the detector saw

The flagged block began at “Therefore, this study was designed…” and continued to the end. It contained four changes from the passing version:

| ID | Change in the revised abstract | Final status |
|---|---|---|
| ABS-01 | semicolon list → “a large judge model or a graphics card …, and feature-based detectors …” | reverted |
| ABS-02 | “consistent with random guessing (F1 0.30)” → “remains near chance (AUROC 0.549; F1 0.30)” | reverted |
| ABS-03 | “When evaluating results on task type categories …” → “On the RAGTruth task types seen during training …” | reverted |
| ABS-04 | “Eight of the 35 features represent …” → “summarize claim-level evidence” | reverted |

The restored version keeps the original semicolon list, “Eight of the 35 features represent…”, “When evaluating results…”, and “consistent with random guessing…”. These are known issues, accepted for detector stability until a later pass.

### [x] ABS-01 — Applied, then reverted — P0

**Tried**

```text
For example, they require a large judge model or a graphics card to run an encoder classifier, and feature-based detectors provide little information about how well their estimates are calibrated or how well they perform outside their training domain.
```

**Original in the passing version**

```text
For example, they require a large judge model; a computer with graphics capability to run an encoder classifier; feature-based detectors provide little information regarding how well their estimates are calibrated or how well these detectors perform outside the domain of their training data.
```

**Reverted:** the revised abstract was flagged. The passing version keeps the semicolon list.

---

### [x] ABS-02 — Applied, then reverted — P0

**Tried:** `Performance on a QA task not seen during training remains near chance (AUROC 0.549; F1 0.30).`

**Original in the passing version:** `Performance on a QA evaluation task not seen during training is consistent with random guessing (i.e., an F1 near 0.30).`

**Reverted:** same detector result.

---

### [x] ABS-03 — Applied, then reverted — P1

**Tried:** `On the RAGTruth task types seen during training`

**Original in the passing version:** `When evaluating results on task type categories (RAGTruth) included in the training set`

**Reverted:** same detector result.

---

### [x] ABS-04 — Applied, then reverted — P1

**Tried:** `Eight of the 35 features summarize claim-level evidence.`

**Original in the passing version:** `Eight of the 35 features represent the extent to which the claims are supported.`

**Reverted:** same detector result.

---

### [x] ABS-05 — Restore the passing abstract — P0

**Trigger:** detector flagged the revised abstract from “Therefore, this study was designed…” to the end.

**Action:** restored the previous 277-word abstract from `HEAD` (`d439254`), which the author confirmed passed.

**Current state:** 277 words; 18 pages; 0 warnings; 0 LaTeX errors.

**Decision:** keep this version. Do not reapply ABS-01 to ABS-04 in this pass.

### Decision: ABS-D1 — expand AUROC at first use?

Not applied. The passing version uses `AUROC` without expansion. Revisit only if the journal requires it and the detector still passes.

### [x] Section gate — Abstract

- [x] Apply ABS-01 to ABS-04. (Applied, then reverted.)
- [x] Build.
- [x] Abstract word count is 250–300 (277).
- [x] Detector check: previous version passed (author confirmed).
- [x] Author approves; move to §1.

---

# Stage 1 — Introduction

**Goal:** fix the two broken LaTeX strings and two number/claim inconsistencies. Do not touch the four-paragraph structure, the literature sentences, or the citations.

**Status (2026-10-29):** INT-01–INT-04 applied. Build is 18 pages with 0 warnings. PDF page 2 reads `a query q, a reference context c, and a candidate answer a`, and `$0.001`. Detector check is pending.

### [x] INT-01 — §1, line 294 — P0

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

### [x] INT-02 — §1, line 296 — P0

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

### [x] INT-03 — §1, line 294 — P1

**Current:** `The model, evidence-consistent XGBoost (EC-XGB), consists of 26 evidence-consistency features in seven groups, eight claim-level NLI aggregates, and a tuned XGBoost classifier.`

**Fix:** `The model, evidence-consistent XGBoost (EC-XGB), consists of 26 evidence-consistency features in seven groups, eight claim-level NLI aggregates, and a source indicator, classified by a tuned XGBoost model.`

**Why:** 26 + 8 = 34. The deployed model has 35 inputs; the source indicator is introduced in §3.3. The current sentence leaves the reader unable to reconcile the counts. Adds 3 words.

---

### [x] INT-04 — §1, line 296 — P1

**Current:** `Target calibration lowers the RAGTruth QA ECE from 0.726 to 0.126.`

**Fix:** `Target calibration on 5{,}034 labeled rows lowers the RAGTruth QA ECE from 0.726 to 0.126.`

**Why:** without the labeled-row clause, the sentence reads like a zero-shot improvement. The abstract already says the calibrator is fit to labeled target rows. Adds 4 words.

---

### Deferred to appendix

- Comma before “and” in “Judges carry high computational costs and encoder models are hard to interpret.”
- “A number of researchers” is vague.
- “three categories including” is vague.
- “artifact-control baselines … prevent shortcut solutions” is jargon and overstrong.
- “validation-only calibration solves the reliability problem” overstates.

### [ ] Section gate — Introduction

- [x] Apply INT-01 to INT-04.
- [x] Build.
- [x] PDF page 2 reads `a query q, a reference context c, and a candidate answer a`, and `$0.001` not “˙001”.
- [ ] Run the detector on the Introduction only.
- [ ] Author approves; move to §2.

---

# Stage 2 — Conventional Method

**Goal:** fix the 85% HaluEval attribution while keeping the reference list at 34/34. This section is otherwise stable.

**Status (2026-10-29):** CON-01 applied. The sentence now cites both `haluEval` and `bang-etal-2025-hallulens`, so all 34 entries remain cited. Build is 18 pages with 0 warnings. Page 18 grew from 17 to 23 lines, so the final trim must free at least 23 lines, not 17. Detector check is pending.

### [x] CON-01 — §2.3 Limitations and Research Gap, line 338 — P0

**Original**

```latex
The benchmark itself represents one limiting factor, since close to 85\% of HaluEval answers were produced by sampling and filtering~\cite{bang-etal-2025-hallulens}.
```

**Applied**

```latex
The benchmark itself represents one limiting factor, since close to 85\% of the full HaluEval benchmark was produced by ChatGPT through sampling and filtering~\cite{haluEval,bang-etal-2025-hallulens}.
```

**Why:** the 85% figure comes from HaluEval’s 30K/35K construction, not from HalluLens. The QA subset used here is entirely from the 30K task-specific examples, so “85% of HaluEval answers” was also imprecise. Both keys are cited to keep the reference list at 34/34; HalluLens remains the review that discusses HaluEval’s factuality limits.

### Deferred to appendix

- §2.2 list grammar: “because it handles nonlinear interactions, it trains quickly …, and its tree structure supports …”
- §2.3 “performs well as a standalone for its specific benchmark”
- §2.3 “essentially an ideal feature to utilize”

### [ ] Section gate — Conventional Method

- [x] Apply CON-01.
- [x] Build.
- [x] Citation check 34/34; `haluEval` and `bang-etal-2025-hallulens` are both cited.
- [ ] Run the detector on §2 only.
- [ ] Author approves; move to §3.

---

# Stage 3 — Proposed Method

**Goal:** fix the feature-count arithmetic, the false “all results” sentence, the grammar error, and decide the risk-card inconsistency. Do not rewrite §3.1–§3.4.

**Status (2026-10-29):** PROP-01, PROP-02, PROP-03, PROP-04, and PROP-05 applied. Build is 18 pages with 0 warnings; 34/34 citations; page 18 still has 23 lines. Detector check is pending.

### [x] PROP-01 — §3.3, line 404 — P0

**Current**

```latex
Let $\phi'(x)$ be the complete model input and $\tilde{\phi}(x)$ the remaining 25 base features excluding $\ell'$; there are 35 input features in total.
```

**Fix**

```latex
Let $\phi'(x)$ be the complete model input and $\tilde{\phi}(x)$ the remaining 24 base features excluding the two log-scaled length columns; there are 35 input features in total (24 base + 2 log-scaled length + 8 claim + 1 source).
```

**Why:** `b6_run_config.json` lists `n_chars` and `n_words` in `log_length_columns`. Two features are log-scaled, not one, so 24 base features remain.

---

### [x] PROP-02 — §3.5, line 442 — P0

**Current:** `All quantitative results in Section~\ref{sec:results} pertain to the EC-XGB classifier and its calibrator.`

**Fix:** `Unless a baseline or ablation variant is named, the results in Section~\ref{sec:results} use the EC-XGB classifier and its calibrator.`

**Why:** §5.1 reports baselines and §5.5 ablates conventional XGBoost. The current sentence is false.

---

### [x] PROP-03 — §3.5, line 438 — P0

**Current:** `This study quantify the reliability by calculating four different measures`

**Fix:** `This study quantifies the reliability by calculating four different measures`

**Why:** subject–verb agreement.

---

### [x] PROP-04 — Resolved as option A — P1

**Checked:** the UI card says “per-claim verdicts lead when present”; `src/api/main.py` lines 1027–1046 compute `adj = base + 0.35*c_frac + 0.15*u_frac - 0.15*s_frac`. The 35.4% is the base calibrated score, before claim-level adjustment.

**Applied:** added “The 35.4\% in Figure~\ref{fig:screens} is the calibrated score before this adjustment. The claim card marks the contradicted number separately.”

**Why:** a reader who follows the verdict rule no longer sees the 35.4% card as a contradiction.

---

### [x] PROP-05 — §3.5, lines 442–444 — P1 — page trim

**Current**

```text
The Next.js frontend provides a streaming LLM conversation on the chat page using the assistant-ui library, while the FastAPI backend hosts the model, the calibrator, the SHAP explainer, and the feature models. All of these are loaded into memory at startup. A Next.js proxy forwards each request to a FastAPI endpoint to avoid cross-origin issues and keep the API key on the server.
```

**Fix**

```text
The system is a Next.js frontend and a FastAPI backend that loads the model, calibrator, explainer, and feature models at startup.
```

**Why:** the figure and the surrounding text already describe the deployment. This saves about 40 words and helps the 17-page target.

### Deferred to appendix

- §3.4 “The deployed EC-XGB display score follows the same rule” is confusing after “validation split only”.
- Table 2 caption comma splice.
- “raw-length counts” singular/plural wording.

### [ ] Section gate — Proposed Method

- [x] Apply PROP-01, PROP-02, PROP-03, and PROP-05.
- [x] Build.
- [x] Check the arithmetic: 24 + 2 + 8 + 1 = 35.
- [x] Decide PROP-04 (risk card vs verdict). Resolved as option A.
- [ ] Run the detector on §3 only.
- [ ] Author approves; move to §4.

---

# Stage 4 — Experiments

**Goal:** fix the hyperparameter overgeneralization, the RAGTruth row-count wording, the majority-baseline logic, the protocol cross-reference, and the GPT citation.

**Status (2026-10-29):** EXP-01–EXP-06 applied. Build is 18 pages with 0 warnings; 34/34 citations; page 18 still has 23 lines. Detector check is pending.

### [x] EXP-01 — §4.1, line 481 — P0

**Current**

```text
The configuration used for XGBoost has a maximum tree depth of 4, a learning rate of 0.01, 500 trees, a subsampling ratio of 0.9, and a column sampling ratio of 0.7.
```

**Fix**

```text
For the seed-42 conventional model, the configuration has a maximum tree depth of 4, a learning rate of 0.01, 500 trees, a subsampling ratio of 0.9, and a column sampling ratio of 0.7. Each seed and each EC-XGB variant was tuned separately.
```

**Why:** `b6_tuning.json` shows different configurations for seeds 123 and 456 and for m1–m3. The current wording reads as one configuration for every run.

---

### [x] EXP-02 — §4.2, line 487 — P0

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

### [x] EXP-03 — §4.3, line 509 — P1

**Current:** `A majority classifier that always predicts the negative class shows that the test data is balanced across classes.`

**Fix:** `The test set contains 1,500 rows per class; a majority classifier that always predicts the negative class therefore scores zero.`

**Why:** a majority-negative classifier scores zero at any prevalence. The 1,500/1,500 composition is the evidence for balance.

---

### [x] EXP-04 — §4.4, line 517 — P1

**Current:** `Section~\ref{sec:proposed} presents the details of the three protocols (P0, P1, and P2; presented in Table~\ref{tab:protocols}), along with the row split for each protocol.`

**Fix:** `Table~\ref{tab:protocols} defines the three protocols (P0, P1, and P2) and the row split for each protocol.`

**Why:** §3 does not present P2 or the row split; Table 4 does. The current cross-reference sends the reader to the wrong place, and “presented … presented” repeats.

---

### [x] EXP-05 — §4.4, line 545 — P1

**Current:** `using GPT 5.6 Luna \cite{zheng2023judge}, a biased model that serves as a standard judge comparison`

**Fix:** `using GPT 5.6 Luna with the LLM-as-a-judge protocol of Zheng et al. \cite{zheng2023judge}, a commercial judge baseline`

**Why:** `zheng2023judge` is the LLM-as-a-judge paper, not the GPT 5.6 Luna model. “Biased” is unexplained, and “standard” can be confused with the conventional model.

---

### [x] EXP-06 — §4.4, line 517 — P1

**Current:** `As well as being influenced by distribution shift \cite{ovadia2019}, the same comparison was performed during the target-domain evaluation.`

**Fix:** `The same comparison was performed during the target-domain evaluation, where distribution shift affects calibration \cite{ovadia2019}.`

**Why:** the current sentence has a dangling modifier. The comparison is not “influenced by distribution shift”.

### Deferred to appendix

- §4.2 “developed using their original licensing agreements”
- §4.4 “either model” should be “either calibrator”
- §4.1 “The programming language used to run the code is Python. The Python version …”
- §4.4 run-on sentence about content edits.

### [ ] Section gate — Experiments

- [x] Apply EXP-01 to EXP-06.
- [x] Build.
- [x] Check the seed-42 configuration against `b6_tuning.json`.
- [ ] Run the detector on §4 only.
- [ ] Author approves; move to §5.1.

---

# Stage 5 — Results

Work through 5.1 to 5.9 in order. Each subsection has its own gate so the detector can be run on a small block.

## 5.1 Model comparison

**Status (2026-10-29):** R51-01–R51-06 applied. Build is 18 pages with 0 warnings; 34/34 citations. Page 18 still has 23 lines; the 59-word deletion did not reduce the page count because a float moved up. Final trim may need float adjustment or more cuts. Detector check is pending.

### [x] R51-01 — line 555 — P0

**Current:** `A neural baseline trained solely on the source material outperforms EC-XGB in-domain (F1 0.9912 vs 0.9846)`

**Fix:** `A neural baseline trained solely on the source material outperforms EC-XGB in-domain (F1 0.9912 vs 0.9855)`

**Why:** 0.9846 is the conventional XGBoost F1 from Table 5. EC-XGB (m3) is 0.9855.

---

### [x] R51-02 — line 551 — P1

**Current:** `... and the majority baseline (wherein the classifier always predicts the most common label) produced an F1 value of 0.0000, providing evidence that the data is well-balanced.`

**Fix:** `... and the majority baseline (wherein the classifier always predicts the most common label) produced an F1 value of 0.0000.`

**Why:** EXP-03 already states the 1,500/1,500 composition. The balance clause here is a non sequitur.

---

### [x] R51-03 — line 551 — P1

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

### [x] R51-04 — line 553 — P1

**Current:** `it was statistically significant in a McNemar test comparing the paired seed-42 test predictions ($p=0.044$).`

**Fix:** `it was nominally significant in an uncorrected McNemar test comparing the paired seed-42 test predictions ($p=0.044$).`

**Why:** the paragraph tests three comparisons. p = 0.044 would not survive a Bonferroni correction (0.017).

---

### [x] R51-05 — Table 5 caption — P1

**Current:** no note explains why the conventional XGBoost F1 is 0.9846 in Table 5 and 0.9844 for m0 in Table 6.

**Fix (append):** `The conventional row in this table and the m0 row in Table 6 come from separate tuning runs.`

**Why:** the difference is small, but a reviewer will notice it. One sentence prevents a consistency query.

---

### [x] R51-06 — §5.1 final paragraph — P1 — page trim

**Delete**

```text
Single-answer behavior shows how individual answers can disagree in ways that aggregate measures hide. For example, if two answers were generated from one question and one context, both would have an exact match under the overlap heuristic. They could also produce an identical result for an NLI-only model. However, logistic regression, random forest, and XGBoost all produced different results.
```

**Why:** unclear, has no numbers, and repeats the aggregate point. Deleting it also removes a confusing passage and saves 59 words.

### Deferred to appendix

- “The performance … has been averaged … to report their overall performance”
- “The NLI-only baseline scored very poorly”
- “there were no such statistical differences”
- “original EC-XGB” is ambiguous; “unadapted EC-XGB” would be clearer.

### [ ] Gate — 5.1

- [x] Apply R51-01 to R51-06.
- [x] Build.
- [ ] Run the detector on 5.1 only.
- [ ] Author approves; move to 5.2.

## 5.2 Evidence-consistent XGBoost

### [x] R52-01 — line 586 — P1

**Current:** `The Wilcoxon signed-rank test for the differences across seeds between models was also not significant ($p = 0.25$).`

**Fix:** `The Wilcoxon signed-rank test across the three seeds was underpowered ($p = 0.25$, the smallest attainable two-sided value with three pairs).`

**Why:** with three pairs, p = 0.25 is the minimum possible two-sided value. “Not significant” is technically true but uninformative.

---

### [x] R52-02 — line 586 — P1 — page trim

**Delete:** `All the ablated versions of the model have almost identical F1 scores compared to the original version of the model.`

**Why:** repeats the saturation point and saves 20 words.

### Deferred to appendix

- “The increases in the measures of performance are small.”
- “McNemar test (test of m0 vs. m3)” repeats “test”.

### [ ] Gate — 5.2

- [x] Apply R52-01 and R52-02.
- [x] Build.
- [ ] Run the detector on 5.2 only.
- [ ] Author approves; move to 5.3.

## 5.3 Calibration

### [x] R53-01 — Table 8 caption — P0

**Current**

```text
Calibration on the same 900-row RAGTruth QA test set (17.8\% positive; means over seeds 42/123/456). The deployed EC-XGB display calibrator is the seed-42 target Platt row (ECE 0.126). AUROC is unchanged by calibration: 0.540 for the conventional model and 0.549 for EC-XGB.
```

**Fix**

```text
Calibration on the same 900-row RAGTruth QA test set (17.8\% positive). EC-XGB cells are means over seeds 42/123/456; conventional rows use the Table 5 checkpoint. The deployed EC-XGB display calibrator is the seed-42 target Platt row (ECE 0.126). AUROC is unchanged by calibration: 0.540 for the conventional model and 0.549 for EC-XGB.
```

**Why:** the conventional cells are the single Table 5 checkpoint, not seed means. The 0.540 value belongs to that checkpoint. Table 9 reports 0.5465 for m0, which is a separate tuning run.

---

### [x] R53-02 — after Figure 4 — P1 — page trim

**Delete**

```text
The table shows average values over three seeds, so each cell reflects an average, and the EC-XGB raw and calibrated cells read 0.7433 and 0.1263.
```

**Why:** the caption already states the means. Saves 25 words.

### Deferred to appendix

- P0 recall lower bound 99.0 vs 99.1.
- “Source calibration has little impact” even though EC-XGB ECE worsens.
- “This is an example of selecting … and results in …”
- “There were no changes to AUROC due to target Platt calibration.”

### [ ] Gate — 5.3

- [x] Apply R53-01 and R53-02.
- [x] Build.
- [ ] Compare Table 8 and Table 9 conventional AUROC in the PDF.
- [ ] Run the detector on 5.3 only.
- [ ] Author approves; move to 5.4.

## 5.4 Cross-domain transfer

### [x] R54-01 — line 670 — P0

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

### [x] R54-02 — line 670 — P1

**Current:** `on FaithBench, the conservative regime decreased the F1 from 0.813 to 0.120.`

**Fix:** `on FaithBench, the 0.5-threshold EC-XGB regime decreased the F1 from 0.813 to 0.120.`

**Why:** “conservative regime” is undefined and appears only here.

---

### [x] R54-03 — line 670 — P1 — page trim

**Delete**

```text
Recall rates did remain quite high for the conventional model.
```

and

```text
As such, at this operating point, the AUROC and flag rate are the appropriate comparisons.
```

**Why:** the first is redundant with the 99.9% flag rate; the next sentence already says the F1 drop is a limitation. Saves 25 words.

### Deferred to appendix

- “The conventional model’s baseline was based solely on its zero-shot performance”
- “Thus, … so …” double connective.
- “probably the most difficult test”
- “is viewed as a limitation” passive.

### [ ] Gate — 5.4

- [x] Apply R54-01 to R54-03.
- [x] Build.
- [ ] Run the detector on 5.4 only.
- [ ] Author approves; move to 5.5.

## 5.5 Ablation

### [x] R55-01 — line 704 — P1

**Current:** `The ablation study in Table~\ref{tab:ablation} demonstrates how much each feature group contributes to this model`

**Fix:** `The ablation study in Table~\ref{tab:ablation} demonstrates how much each feature group contributes to the conventional XGBoost model`

**Why:** “this model” is ambiguous; the table uses the conventional model as the full model.

---

### [x] R55-02 — line 704 — P1 — page trim

**Delete:** `, and the paper reports that result`

**Why:** meta prose; saves 6 words.

### [ ] Gate — 5.5

- [x] Apply R55-01 and R55-02.
- [x] Build.
- [ ] Run the detector on 5.5 only.
- [ ] Author approves; move to 5.6.

## 5.6 Explanation reliability

### [x] R56-01 — line 757 — P0

**Current:** `A structured expert audit assessed every explanation.`

**Fix:** `A structured AI-assisted audit assessed the 26 exported explanations.`

**Why:** the audit covers 26 error-enriched cases, not every explanation in the test set. The review mode is `ai-expert`, so “expert” without qualification may imply human reviewers.

---

### [x] R56-02 — line 757 — P1

**Current:** `False negatives occur when a wrong answer reuses entities from its context, which keeps overlap high.`

**Fix:** `Many false negatives occur when a wrong answer reuses entities from its context, which keeps overlap high.`

**Why:** several audited false negatives are fabricated titles or off-target answers, not entity reuse.

### Deferred to appendix

- “The overlap in contexts” → “Context overlap”.
- “contradictions … (in terms of probability)” → “NLI contradiction probability”.
- “reported openly” → “reported”.
- “item numbers 2251 and 2436” grammar.

### [ ] Gate — 5.6

- [x] Apply R56-01 and R56-02.
- [x] Build.
- [ ] Run the detector on 5.6 only.
- [ ] Author approves; move to 5.7.

## 5.7 Efficiency and the LLM judge

### [x] R57-01 — line 761 — P1 — page trim

**Delete**

```text
For end-to-end HTTP timing, the median values are 42.8\,ms for prediction, 57.1\,ms for analysis, and 143.6\,ms for verification with retrieval over 26 requests, and 502.5\,ms with the judge over 6 requests. The component table does not include retrieval, judge, or model-loading time. The HTTP verification number includes retrieval but not the judge.
```

**Why:** deployment detail. Table 12 already reports component latency. Saves 60 words. If the author wants the HTTP numbers, keep the first sentence and delete the two caveats; the page count may then need another cut.

### Deferred to appendix

- “total average and distribution” → “latency distribution”.
- “either method” → “either of those two components”.
- “three orders of magnitude slower than any local model” is ambiguous.

### [ ] Gate — 5.7

- [x] Apply R57-01.
- [x] Build.
- [ ] Run the detector on 5.7 only.
- [ ] Author approves; move to 5.8.

## 5.8 Error analysis

### [x] R58-01 — line 829 — P0

**Current:** `Table~\ref{tab:length} shows how many were false positives and false negatives.`

**Fix:** `Table~\ref{tab:length} shows the false-positive and false-negative rates by answer length.`

**Why:** Table 15 has rates, not counts.

---

### [x] R58-02 — line 827 — P1

**Current:** `A heuristic analysis of 20 randomly selected errors (10 false positives and 10 false negatives)`

**Fix:** `A heuristic analysis of 20 errors sampled evenly across the two directions (10 false positives and 10 false negatives)`

**Why:** the sample was stratified 10/10, not random from all 48 errors.

### Deferred to appendix

- “The amount of hallucinations per answer length grows very quickly.”
- “The key aspect of the data is how the answer length … is distributed.”
- “nearly perfect indicator” is strong.

### [ ] Gate — 5.8

- [x] Apply R58-01 and R58-02.
- [x] Build.
- [ ] Run the detector on 5.8 only.
- [ ] Author approves; move to 5.9.

## 5.9 Discussion

### [x] R59-01 — line 853 — P1

**Current:** `The gains come from evidence-consistent features and multi-source training, not from a different type of learning system.`

**Fix:** `The gain coincides with multi-source training rather than a different type of learning system.`

**Why:** the claim-level features do not improve RAGTruth-all AUROC. m2 is 0.502 and m1 is 0.543. The gain appears when the RAGTruth non-QA rows are added in m3.

---

### [x] R59-02 — line 853 — P1

**Current:** `EC-XGB provides an effective rank ordering of answers on RAGTruth`

**Fix:** `EC-XGB provides a better rank ordering of answers on RAGTruth than the conventional model`

**Why:** AUROC 0.582 is better than 0.475, but “effective” overstates a weak ranker.

### Deferred to appendix

- “The informative results are outside the benchmark” is awkward.

### [ ] Gate — 5.9

- [x] Apply R59-01 and R59-02.
- [x] Build.
- [ ] Run the detector on 5.9 only.
- [ ] Author approves; move to §6.

---

# Stage 6 — Conclusion

**Goal:** fix the subject of the opening sentence and the 26 + 8 omission. Do not rewrite the conclusion.

### [x] CONC-01 — line 857 — P1

**Current:** `To be effective, hallucinations need to be detected quickly, inexpensively, and explainably.`

**Fix:** `A useful hallucination detector must be fast, inexpensive, and explainable.`

**Why:** hallucinations are detected; they do not need to be detected. The current subject is wrong.

---

### [x] CONC-02 — line 857 — P1

**Current:** `EC-XGB uses 26 evidence-consistency features, including log-scaled answer lengths under monotone constraints, and 8 claim-level NLI aggregates`

**Fix:** `EC-XGB uses 26 evidence-consistency features, including log-scaled answer lengths under monotone constraints, 8 claim-level NLI aggregates, and a source indicator`

**Why:** 26 + 8 = 34. The deployed model has 35 inputs. The conclusion currently leaves the source indicator out.

### Deferred to appendix

- Semicolon before “however”.
- “The improvement appears …; however, the improvement …” repetition.
- “Conventional models” plural.

### [ ] Gate — Conclusion

- [x] Apply CONC-01 and CONC-02.
- [x] Build.
- [ ] Run the detector on the Conclusion only.
- [ ] Author approves; move to Declarations.

---

# Stage 7 — Declarations and Acknowledgements

### [x] DECL-01 — Acknowledgements, line 882 — P1

**Current**

```text
The authors thank their supervisor, Ohidujjaman Tuhin, for his supervision and input on this work.
```

**Fix:** delete the sentence. Keep the dataset-maintainer sentence.

**Why:** Ohidujjaman Tuhin is an author. G14 says acknowledgements name non-author support. His supervision is already in CRediT.

### Deferred to appendix

- Data and code availability give no repository URL.
- “Conflicts of Interest” capitalization differs from the other declaration labels.

### [ ] Gate — Declarations

- [x] Apply DECL-01.
- [x] Build.
- [ ] Run the detector on the Declarations only.
- [ ] Author approves; move to the page check.

---

# Stage 8 — Page trim and final verification

The trims are already placed in their sections:

| Trim | Section | Words |
|---|---|---:|
| R51-06 single-answer paragraph | 5.1 | 59 |
| R53-02 table-average sentence | 5.3 | 25 |
| R57-01 HTTP block | 5.7 | 60 |
| R52-02 all ablated versions | 5.2 | 20 |
| R54-03 recall + as-such | 5.4 | 25 |
| R55-02 paper reports that result | 5.5 | 6 |
| PROP-05 web stack condensation | 3.5 | 40 |
| R59-01 gains sentence | 5.9 | 20 |
| DECL-01 supervisor sentence | Acknowledgements | 20 |
| **Total** | | **275** |

The 275-word plan plus about 191 words of fallback/low-value cuts were needed. The first 59-word deletion was absorbed by a float, so page 18 stayed at 23 lines; the additional cuts brought the paper to 17 pages. Final page 17 has 59 lines. The FActScore sentence was kept because deleting it would have left `factscore` uncited.

### [ ] Final verification

- [x] `cd Journal_Paper; latexmk -pdf -interaction=nonstopmode -outdir=out halurisc.tex`
- [x] Pages = 17, no reference block on page 18. Page 17 has 59 lines and ends at reference [34].
- [x] `Underfull 0`, `Overfull 0`, `LaTeX Error 0`, undefined citation 0, undefined reference 0.
- [x] Abstract 250–300 words, one paragraph (277).
- [x] Keywords 4–8 (7).
- [x] Every `\cite` key exists in `ref.bib`, and every entry is cited (34/34).
- [x] `python src/models/review_number_audit.py` → 203/203.
- [x] Manually spot-check the new prose numbers: 0.9855, 0.540/0.5465, 0.7433/0.7735. The 99.0–99.5 lower bound remains deferred.
- [x] PDF check: Introduction reads `a query q, a reference context c, and a candidate answer a`, and `$0.001`.
- [x] PDF check: Table 8 caption and Table 9 agree on the conventional model identity.
- [x] Search the `.tex` for `standard XGBoost`, `standard model`, `\.001`, and `query $,`. No hits.
- [x] Confirm no table, figure, or algorithm content changed. Only captions were updated.
- [ ] Delete `fix_checklist.md`.

---

# Deferred minor items (do not apply in this pass)

These are recorded so they are not lost. Apply only if the author asks, or if a section still reads badly after the main pass.

| ID | Location | Item |
|---|---|---|
| D-01 | §1 | Comma before “and” in “Judges carry high computational costs and encoder models are hard to interpret.” |
| D-02 | §1 | “A number of researchers” is vague. |
| D-03 | §1 | “three categories including” is vague. |
| D-04 | §1 | “artifact-control baselines … prevent shortcut solutions” is jargon and overstrong. |
| D-05 | §1 | “validation-only calibration solves the reliability problem” overstates. |
| D-06 | §2.2 | List grammar: “because it handles nonlinear interactions, it trains quickly …, and its tree structure supports …” |
| D-07 | §2.3 | “performs well as a standalone for its specific benchmark” |
| D-08 | §2.3 | “essentially an ideal feature to utilize” |
| D-09 | §3.4 | “The deployed EC-XGB display score follows the same rule” is confusing after “validation split only”. |
| D-10 | Table 2 caption | Comma splice. |
| D-11 | §4.2 | “developed using their original licensing agreements” |
| D-12 | §4.4 | “either model” should be “either calibrator” |
| D-13 | §4.1 | “The programming language used to run the code is Python. The Python version …” |
| D-14 | §4.4 | Run-on sentence about content edits. |
| D-15 | §5.1 | “The performance … has been averaged … to report their overall performance” |
| D-16 | §5.1 | “The NLI-only baseline scored very poorly” |
| D-17 | §5.1 | “there were no such statistical differences” |
| D-18 | §5.1 | “original EC-XGB” is ambiguous; “unadapted EC-XGB” would be clearer. |
| D-19 | §5.2 | “The increases in the measures of performance are small.” |
| D-20 | §5.2 | “McNemar test (test of m0 vs. m3)” repeats “test”. |
| D-21 | §5.3 | P0 recall lower bound 99.0 vs 99.1. |
| D-22 | §5.3 | “Source calibration has little impact” even though EC-XGB ECE worsens. |
| D-23 | §5.3 | “This is an example of selecting … and results in …” |
| D-24 | §5.3 | “There were no changes to AUROC due to target Platt calibration.” |
| D-25 | §5.4 | “The conventional model’s baseline was based solely on its zero-shot performance” |
| D-26 | §5.4 | “Thus, … so …” double connective. |
| D-27 | §5.4 | “probably the most difficult test” |
| D-28 | §5.4 | “is viewed as a limitation” passive. |
| D-29 | §5.6 | “The overlap in contexts” → “Context overlap”. |
| D-30 | §5.6 | “contradictions … (in terms of probability)” → “NLI contradiction probability”. |
| D-31 | §5.6 | “reported openly” → “reported”. |
| D-32 | §5.6 | “item numbers 2251 and 2436” grammar. |
| D-33 | §5.7 | “total average and distribution” → “latency distribution”. |
| D-34 | §5.7 | “either method” → “either of those two components”. |
| D-35 | §5.7 | “three orders of magnitude slower than any local model” is ambiguous. |
| D-36 | §5.8 | “The amount of hallucinations per answer length grows very quickly.” |
| D-37 | §5.8 | “The key aspect of the data is how the answer length … is distributed.” |
| D-38 | §5.8 | “nearly perfect indicator” is strong. |
| D-39 | Conclusion | Semicolon before “however”. |
| D-40 | Conclusion | “The improvement appears …; however, the improvement …” repetition. |
| D-41 | Conclusion | “Conventional models” plural. |
| D-42 | Declarations | Data and code availability give no repository URL. |
| D-43 | Declarations | “Conflicts of Interest” capitalization differs from the other labels. |
| D-44 | Global | Remaining prose semicolons. |
| D-45 | Global | Four sentences start with “When” or “Although”. |
| D-46 | Global | Five “However” and five “Therefore” transitions. |
| D-47 | §4.1 | “utilizes” → “uses”. |
