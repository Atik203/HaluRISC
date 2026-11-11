# Section 2 Review: Conventional Method

- **Reviewer:** independent subagent (journal reviewer, Section 2)
- **Manuscript:** `Journal_Paper/halurisc.tex`, lines 296-349 (`\section{Conventional Method}` through `\subsection{Limitations and Research Gap}`, before `\section{Proposed Method}`)
- **Guides read:** `docs/prompt.md`, `docs/writing-guideline.md`
- **Code cross-checked:** `src/models/run_b2_baselines.py`, `src/models/train_pipeline.py`, `src/models/review_calibration_protocol.py`, `src/api/main.py`
- **Artifacts cross-checked:** `artifacts/results/b2/b2_model_comparison.csv`, `artifacts/results/b2/b2_tuning.json`, `artifacts/results/b4/b4_calibration_metrics.json`, `artifacts/models/b2/xgboost_seed_{42,123,456}.joblib`, `Journal_Paper/out/halurisc.pdf`
- **Constraints respected:** no edits to the manuscript, artifacts, or any existing file. Only this report was written.

**Revision note.** The manuscript changed while this review was in progress. The current text already contains the protocol, algorithm, objective-notation, terminology, and G7-heading fixes that an earlier draft lacked. This report reviews the **current** text and lists only what remains. The earlier Major findings (protocol conflating three runs, tuning outside the seed loop, missing early stopping, probability used as the loss argument, missing 2.3 heading, "26 evidence-consistency" mislabel, uncited boosted-trees claim) are resolved and are not repeated as open issues.

## Verdict

**Minor revision.** The section now matches the B2 run and the paper's G7 structure. The protocol is accurate, Algorithm 1 is per-seed and includes early stopping, the objective uses the raw margin correctly, the leaf weight and gain are correct, and the limitations subsection exists and maps the limits to the proposed method. What remains is copy-editing and readability: three acronyms are never defined in the body, the baseline paragraph never says what the 26 features are or which checkpoint it describes, the prose uses "raw score" for two different quantities, the split rule is slightly overstrong, and the limitations sentence bundles two limits in a three-item list. No number or equation is wrong, and no claim misstates the run.

## Strengths

- **The protocol now matches the code.** The section states per-seed 30-configuration searches, stratified group five-fold cross-validation, the 14,000/3,000/3,000 split, early stopping on validation (209 trees for seed 42, with 300 as the tuned upper bound), the raw 0.5 headline threshold, Platt and isotonic as separate calibration variants, the P0 threshold, and the XGBoost defaults for `γ`, `λ`, and the minimum child weight. This is traceable to `run_b2_baselines.py`, `b2_tuning.json`, and the loaded seed models.
- **The objective notation is correct.** `z_i` is the raw margin, Eq. (1) uses `l(y_i,z_i)`, `g_i` and `h_i` are derivatives with respect to `z` evaluated at `z_i^{(t-1)}`, `\hat{p}_i = \sigma(z_i)` is the probability, and `f_t` and `q(x_i)` are defined. The constant term is explicitly dropped. Eqs. (3) and (4) are the correct leaf weight and gain, with the leading one-half and the `-γ` term.
- **Algorithm 1 matches the text and the code.** Tuning sits inside the seed loop, early stopping uses the validation rows, the headline evaluation uses the raw score at 0.5, the calibration maps are fitted on validation, and the final step returns the mean metrics.
- **The G7 structure is complete.** `Preliminaries` defines the task and the two deployment demands, `Baseline Method` gives the selection rationale, formulation, and algorithm, and `Limitations and Research Gap` separates three method limits from two evaluation limits and maps them to the proposed method and the evaluation design.
- **The prose is compact and in the student register.** The uncited "boosted trees" claim, the unused TruthfulQA and FActScore citations, the two `--` dashes, and the "evidence-consistency" mislabel are gone. The section reads as a methods section, not a literature review.

## Issues

| # | Severity | Exact location (quote) | Issue | Why it matters | Suggested fix |
|---|---|---|---|---|---|
| 1 | Minor | Line 300: `A detector is usually scored with precision, recall, F1, or AUROC on labeled rows.` / `and ECE summarizes the gap across confidence bins` / `and SHAP attributions provide that for a tabular model` | `AUROC`, `ECE`, and `SHAP` are used without expansion. The abstract and §1 also use them, but no body section defines them. The previous draft expanded all three in §2; the revision removed the expansions. | Journal copy editors require acronyms to be expanded before or at first use. The body currently has no expansion for AUROC or ECE, and a reader who starts at §2 sees three unexplained abbreviations in the first paragraph. | **R1**: expand each acronym in place. |
| 2 | Minor | Line 304: `The conventional method is a tuned XGBoost classifier \cite{xgboost} over the 26 base features trained on HaluEval QA. ... A published feature-based detector uses the same learner ...` | The phrase `over the 26 base features trained on HaluEval QA` reads as if the features were trained. The 26 features are not described or forward-referenced, the prior detector is unnamed, and the paragraph does not label the B2 checkpoint. | The reader cannot tell what the baseline's input is (the seven groups and Table 1 appear only in §3.2), which checkpoint §2 describes (B2 vs B6 m0), or which prior detector the sentence means. | **R2**: replace the paragraph. |
| 3 | Minor | Lines 304, 306, 328: `summed into a raw score` / `sums $K$ trees into a raw margin` / `the raw score at a fixed 0.5 threshold` | The prose uses `raw score` for the summed tree output on line 304 and for the uncalibrated probability on line 328, while line 306 calls the same summed output `raw margin`. | The paper later fits Platt to the `raw score` as a probability (§3.4) and reports ECE on raw scores (§5.3). If §2 uses the same term for the margin, the reader cannot tell what the calibrator consumes. | **R2**: use `raw margin` for the summed tree output and keep `raw score` for the probability. |
| 4 | Minor | Line 326: `A split is kept only when the gain is positive. This is the criterion used to grow each tree.` | Positive gain is necessary but not sufficient. XGBoost also enforces `min_child_weight` and `max_depth`, and the tuned model uses a depth limit. | A reader who knows XGBoost will see an overstatement. The sentence makes the growth rule look simpler than it is. | **R3**: add the minimum child-weight and depth qualifier. |
| 5 | Minor | Line 328: `The final step returns the mean over the three seeds.` | Mean of what? The algorithm returns the mean metrics, and the seeded tables report metric means. | Small ambiguity in an otherwise precise protocol paragraph. | **R4**: `the mean metrics over the three seeds`. |
| 6 | Minor | Line 328: `The source-only P0 threshold is selected on those rows at a 5\% false-positive budget.` | `P0` is used before it is defined. The P0/P1/P2 table is in §4.3, and a reader who starts at §2 does not know the label. | A forward reference makes the paragraph self-contained. | **R5**: add `(P0, Table 2)`. |
| 7 | Minor | Line 349: `The baseline has three method limits. The score depends on answer length, it has no claim-level evidence, and it trains on HaluEval alone with a source-fitted calibrator.` | The third item bundles two different limits (single-corpus training and source-only calibration), and the sentence is an exact three-item list. | The style guide flags exact three-item lists, and the bundled item weakens the mapping to §3.3 (multi-source training) and §3.4 (calibration). | **R6**: split the third item. |
| 8 | Minor | §3.4, line 444: `Platt scaling fits a sigmoid $p(x) = \sigma(A f(x) + B)$ to the raw score $f(x)$` | `f(x)` is the raw score here, while §2 uses `f_k` for a tree. | Cross-section symbol collision. A reader who carries §2's notation into §3.4 may read `f(x)` as a tree. | **R7**: rename the raw score to `s(x)`. |
| 9 | Minor | Lines 304 and 310: `A logistic link maps the raw score to a probability.` / `The probability is the sigmoid of the raw margin.` | The logistic link is stated twice within two paragraphs. | Text weight. The section is compact, so the duplicate is visible. | **R2** removes the earlier sentence; line 310 keeps the formal definition. |
| 10 | Minor | Line 328: `Each seed trains on the 14,000 rows with early stopping on the 3,000 validation rows (30 rounds); 300 is the tuned upper bound, and seed 42 stops at 209 trees.` | The semicolon joins two independent clauses. The prompt bans semicolons outside complex lists. | Style guide compliance, and the sentence reads more cleanly as two sentences. | **R8**: split the sentence. |

**Bibliography side effect.** Removing the TruthfulQA and FActScore sentence from §2 leaves `truthqa` and `factscore` in `Journal_Paper/ref.bib` uncited. The paper's G15 check requires every entry to be cited. Either cite both once in §2 as related benchmarks or delete the two `ref.bib` entries. This is outside §2 but is a direct consequence of the revision.

## Suggested exact edits

### R1. Expand the three acronyms at first use (line 300)

Replace:

```latex
A detector is usually scored with precision, recall, F1, or AUROC on labeled rows.
```

with:

```latex
A detector is usually scored with precision, recall, F1, or the area under the receiver operating characteristic curve (AUROC) on labeled rows.
```

Replace `and ECE summarizes the gap across confidence bins` with `and the expected calibration error (ECE) summarizes the gap across confidence bins`.

Replace `and SHAP attributions provide that for a tabular model` with `and SHapley Additive exPlanations (SHAP) attributions provide that for a tabular model`.

### R2. Replace the baseline paragraph (line 304)

```latex
The conventional method is the B2 checkpoint: a tuned XGBoost classifier \cite{xgboost} over 26 base features, trained on HaluEval QA. The features cover length and style, lexical overlap, entities, NLI, numbers, hedging, and semantic similarity (Table~\ref{tab:features}). XGBoost models nonlinear interactions and gives exact TreeSHAP values, which the explanation analysis needs. The feature-based detector of Sundaragiri et al. uses the same learner with retrieval and NLI features \cite{ijert}, so this model is the natural baseline. Trees are added one at a time, and each tree fits the current gradient and curvature of the loss. A row enters at the root of each tree, follows the split conditions to one leaf per tree, and the leaf weights are summed into a raw margin.
```

This fixes the grammar (`features, trained` instead of `features trained`), adds the seven feature groups and the Table 1 pointer, names the prior detector, labels the B2 checkpoint, and changes `raw score` to `raw margin`. It also removes the duplicate logistic-link sentence, because line 310 already defines the probability.

### R3. Qualify the split rule (line 326)

Replace:

```latex
A split is kept only when the gain is positive. This is the criterion used to grow each tree.
```

with:

```latex
A candidate split is accepted when its gain is positive and both children meet the minimum child-weight and depth limits.
```

### R4. Say what is averaged (line 328)

Replace `The final step returns the mean over the three seeds.` with `The final step returns the mean metrics over the three seeds.`

### R5. Define P0 at first use (line 328)

Replace `The source-only P0 threshold is selected on those rows at a 5\% false-positive budget.` with `The source-only protocol (P0, Table~\ref{tab:protocols}) selects a threshold on those rows at a 5\% false-positive budget.`

### R6. Split the third method limit (line 349)

Replace:

```latex
The baseline has three method limits. The score depends on answer length, it has no claim-level evidence, and it trains on HaluEval alone with a source-fitted calibrator.
```

with:

```latex
The baseline has three method limits. The score depends on answer length, and the model has no claim-level evidence. It also trains on HaluEval alone, and its calibrator is fitted on the source split.
```

This removes the exact three-item list and separates the single-corpus limit from the calibration limit.

### R7. Rename the raw score in §3.4 (line 444)

Replace `Platt scaling fits a sigmoid $p(x) = \sigma(A f(x) + B)$ to the raw score $f(x)$` with `Platt scaling fits a sigmoid $p(x) = \sigma(A s(x) + B)$ to the raw score $s(x)$`, and change the next sentence from `$p(x) = g(f(x))$` to `$p(x) = g(s(x))$`.

Use `s(x)` rather than `z(x)` because §2's `z_i` is the raw margin, while §3.4's raw score is the uncalibrated probability. `s(x)` avoids both the `f_k` collision and the margin/probability confusion.

### R8. Split the semicolon sentence (line 328)

Replace:

```latex
Each seed trains on the 14,000 rows with early stopping on the 3,000 validation rows (30 rounds); 300 is the tuned upper bound, and seed 42 stops at 209 trees.
```

with:

```latex
Each seed trains on the 14,000 rows with early stopping on the 3,000 validation rows (30 rounds). The tuned upper bound is 300 trees, and seed 42 stops at 209.
```

## Text-weight and readability assessment

The section is about 750 words of prose plus four equations and one algorithm. The allocation is now balanced:

| Block | Approx. words | What it teaches |
|---|---:|---|
| Preliminaries (line 300) | 113 | Task definition, metrics, calibration and explanation |
| Baseline (line 304) | 104 | Checkpoint, features, selection rationale, working principle |
| Objective (lines 306-316) | 174 | Eq. (1), Eq. (2), symbols, logistic derivatives |
| Leaf weight and gain (lines 318-326) | 82 | Eqs. (3)-(4), split rule |
| Protocol (line 328) | 162 | Tuning, early stopping, headline threshold, calibration, P0, defaults |
| Limitations (line 349) | 117 | Three method limits, two evaluation limits, mapping to §3 |

The section is **not too long**. The old preliminaries spent 164 words on generic evaluation context and two unused resources (TruthfulQA and FActScore); the current paragraph is about 115 words and defines the task. The protocol paragraph is the longest single paragraph at about 160 words. It covers tuning, early stopping, the headline threshold, the calibration maps, the P0 threshold, the test-set guarantee, and the defaults. Splitting it into two paragraphs, one for tuning and fitting and one for evaluation and thresholds, would improve readability without changing the content.

The remaining repetition is small. The logistic link appears on line 304 and line 310; **R2** removes the earlier one. The additive-tree idea appears on line 304 (`Trees are added one at a time`) and line 312 (`a new tree ... is added`); that pair is acceptable because the first is the intuition and the second is the formal step. The shift limitation is now stated once. The test-set guarantee still appears in §2 and §4.3, but one is a protocol statement and the other is a split guarantee, so the repetition is defensible.

The built PDF shows no overfull boxes in the section. Algorithm 1's step 3 wraps onto two lines, but the continuation is indented, so it reads acceptably and is not a layout defect.

Readability is good. Sentences are short, the active voice dominates, and the section avoids the banned transitions and hype adjectives. The remaining weak spots are the three undefined acronyms, the ambiguous `features trained` phrase, the semicolon in the protocol paragraph, and the compressed limitations sentence.

## Equation and notation consistency check

| Item | Verdict | Detail |
|---|---|---|
| Eq. (1), objective structure | **Correct** | Sum of a differentiable logistic loss and a per-tree regularizer. |
| Eq. (1), loss argument | **Correct** | `l(y_i, z_i)` uses the raw margin, not the probability. |
| Eq. (1), regularizer | **Correct** | `\gamma T_k + \frac{1}{2}\lambda \lVert w_k \rVert^2`; `w_k \in \mathbb{R}^{T_k}` is a vector, so the norm is well defined. |
| Eq. (2), second-order expansion | **Correct** | `g_i` and `h_i` are derivatives with respect to `z`, evaluated at `z_i^{(t-1)}`. The constant term is explicitly dropped. |
| Eq. (2), tree output | **Correct** | `f_t(x_i) = w_{q(x_i)}` is the leaf weight for row `i`; `q(x_i)` is defined as the leaf index. |
| Eq. (3), leaf weight | **Correct** | `w_j^{*} = -G_j/(H_j+\lambda)`, with `G_j` and `H_j` the sums of first and second derivatives in leaf `j`. No sign error. |
| Eq. (4), split gain | **Correct** | `\frac{1}{2}[G_L^2/(H_L+\lambda) + G_R^2/(H_R+\lambda) - (G_L+G_R)^2/(H_L+H_R+\lambda)] - \gamma`. The leading `\frac{1}{2}` and the `-\gamma` are consistent with Eq. (2). No missing factor. |
| Split acceptance rule | **Slightly overstrong** | Positive gain is necessary but not sufficient; add `min_child_weight` and `max_depth`. See **R3**. |
| Symbol `n` | **Defined** | Number of training rows, first sentence of the objective. |
| Symbol `K` | **Defined** | Number of trees. |
| Symbols `T_k`, `T` | **Defined** | `T_k` is the leaves of tree `k`; `T` is the leaves of the new tree. |
| Symbol `I_j` | **Defined** | Introduced in the sentence that defines `G_j` and `H_j`, immediately after Eq. (3). Acceptable. |
| Symbols `G_L`, `G_R`, `H_L`, `H_R` | **Defined** | "for the left child" and "for the right child". |
| Symbol `\hat{p}_i` | **Consistent** | Same symbol as §4.5 for the probability. |
| `raw score` vs `raw margin` | **Inconsistent in prose** | Line 304 uses `raw score` for the summed tree output; line 306 uses `raw margin`; line 328 uses `raw score` for the uncalibrated probability. See **R2**. |
| `f(x)` vs `f_k` | **Cross-section collision** | §3.4 uses `f(x)` for the raw score, while §2 uses `f_k` for a tree. See **R7**. |
| `\gamma`, `\lambda`, `min_child_weight` | **Defined** | The protocol paragraph now states that all three stay at their XGBoost defaults. |

Bottom line: **the two equations that matter for XGBoost correctness, the leaf weight and the gain, are right.** The objective notation is now right as well. The only equation-adjacent issue is the split acceptance rule, which is a one-sentence qualifier, not an algebra error.

## Top 3 fixes

1. **Replace the baseline paragraph (R2).** This is the only remaining substantive readability gap. The section introduces a classifier over 26 features without saying what the features are or where to find them, reads `features trained` as if the features were fitted, and uses `raw score` where the rest of the section says `raw margin`. The replacement fixes all four in one paragraph and labels the B2 checkpoint, which the B2/B6 m0 distinction in §3.5 assumes.
2. **Expand the three acronyms (R1) and add the P0 pointer (R5).** These are journal copy-editing requirements. The first paragraph of §2 currently contains `AUROC`, `ECE`, and `SHAP` without expansions, and the protocol paragraph uses `P0` before the P0/P1/P2 table. Both are one-line fixes.
3. **Tighten the limitations sentence (R6) and the split rule (R3).** The limitations sentence bundles single-corpus training and source-only calibration into one item inside an exact three-item list. Split the item so the mapping to §3.3 (multi-source training) and §3.4 (calibration) is one-to-one. The split rule should say that positive gain is necessary but not sufficient, because XGBoost also enforces the minimum child weight and depth.

## Summary

The section now matches the code and the supervisor's G7 structure. The protocol is accurate down to the per-seed search and the 209-tree seed-42 model, Algorithm 1 is per-seed and includes early stopping, the objective uses the raw margin correctly, the leaf weight and gain are correct, and the limitations subsection maps the baseline's weaknesses to the proposed method and the evaluation design. What remains is copy-editing: three acronyms are never defined in the body, the baseline paragraph does not describe the 26 features or name the B2 checkpoint, the prose uses `raw score` for both the margin and the probability, the split rule is slightly overstrong, and the limitations sentence compresses two limits into one item. Fix R1, R2, R3, and R6 first; R4, R5, R7, and R8 are one-line copy edits. No number or equation is wrong, and no claim misstates the run.
