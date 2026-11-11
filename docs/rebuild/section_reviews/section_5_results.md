# Section 5 (Results) — Journal Review

Manuscript: `Journal_Paper/halurisc.tex`, lines 553–1060 (`\section{Results}` through the line before `\section{Conclusion}`)
Review date: 2026-11-11
Scope: six Results subsections, all tables and figures in that range, and the P0/P1/fixed-hp/B6 artifacts.

## Verdict

**Major revision.**

The core analysis is strong and most reported numbers trace cleanly to artifacts. The section is also unusually honest about negative results. It is not submission-ready because of one provenance contradiction, several local misdescriptions that a careful reviewer will catch, a failed float-placement gate, and a text-weight problem that makes the densest passages hard to follow. Most fixes are local. The one substantive question is the causal claim about the seen-task gain, which may need a small extra ablation if the sentence is kept.

## Strengths

- **Negative results are reported, not hidden.** Held-out RAGTruth QA stays near chance (AUROC 0.549), the task shift collapses, MedHallu flags every row and is badly calibrated (ECE 0.493), and both judges beat EC-XGB on natural QA. The paper says so directly.
- **The calibration and shift battery is unusually complete.** Prior-only baselines, Brier skill score, Cox slope, source and target Platt, split and class-conditional conformal, multicalibration by length and claim count, and a label-budget analysis all appear in one subsection. This is more than most student journal papers attempt.
- **The ablation is well controlled.** Table 6 holds one hyperparameter configuration across m0–m3, Table 7 repeats it on the fixed-hp m3 model, and Figure 5 shows the per-seed spread. The separation of feature effects from tuning effects is explicit.
- **Provenance is mostly disciplined.** Sample sizes, seeds, split sizes, and threshold rules are stated in the table notes. Most values match the artifacts to rounding, and the failure cases are identifiable.
- **The best prose is plain and active.** Sentences such as "Calibration fixes the scale, not the ranking" and "Feature extraction, not the model, is the cost" do real interpretive work without promotional language.

## Issues

| ID | Severity | Location (exact quote) | Issue | Why it matters | Suggested fix |
|---|---|---|---|---|---|
| C1 | Critical | "The labels are a draft for author verification, not independent human annotation, so the counts below are provisional." … "so the labels are author-verified." | The same paragraph says the labels are provisional and author-verified. | This is a direct contradiction in the paper's provenance claim. A reviewer who notices it will question every audit number. It also conflicts with the completed three-author review in `author_review_summary.json`. | Delete the first sentence. End the paragraph with "The labels are therefore author-verified." The AI pre-annotation is already disclosed in the last sentence. |
| M1 | Major | "the in-domain model is dominantly a length and overlap detector; that is why natural long answers are flagged at 99.9%" | 99.9% is the conventional m0 pooled flag rate. EC-XGB's pooled flag rate is 61.3% (Table 10). | The sentence attributes the conventional model's failure to the proposed model and overstates EC-XGB's flag rate by 38 points. It weakens trust in the discussion. | "that is why the conventional model flags 99.9% of natural long answers and EC-XGB still flags 61.3%" |
| M2 | Major | Table 5 note: "with the threshold selected on validation and evaluated once on test" | The table's external-ensemble P/R/F1 (0.9508 / 0.9527 / 0.9517) are the `external_lr_train` row at threshold 0.5. The validation-selected row is threshold 0.55 with P/R/F1 0.9575 / 0.9453 / 0.9514. | The note misdescribes the protocol. The values and the text's delta are internally consistent with the 0.5 row, so the note is the wrong part. | "with the threshold fixed at 0.5; a validation-selected threshold of 0.55 gives F1 0.9514, so the choice does not change the comparison." |
| M3 | Major | "its seed-mean F1 is 0.042 higher, but the paired item-bootstrap interval for GPT minus EC-XGB is −0.052 [−0.127, 0.024]" | The interval compares GPT's majority-vote predictions with the EC-XGB seed mean. The table means imply a difference of −0.042, not −0.052. The AUROC sentence has the same problem: table means imply +0.001, the bootstrap mean is +0.008. | The text mixes two estimands in one sentence. A reviewer can subtract the table and conclude the arithmetic is wrong. | "The paired item-bootstrap interval for GPT's majority-vote predictions against the EC-XGB seed mean is −0.052 [−0.127, 0.024] … the AUROC interval is +0.008 [−0.034, 0.048]." |
| M4 | Major | "a gap of 0.436 [0.351, 0.522]; the F1 gap is 0.440 [0.334, 0.552]" | Table 17 gives 0.956 - 0.521 = 0.435 AUROC and 0.907 - 0.476 = 0.431 F1. The reported 0.436 and 0.440 are bootstrap means. | Same estimand mismatch as M3. The conclusions do not change, but the numbers look inconsistent with the table. | "a point gap of 0.435 (bootstrap mean 0.436 [0.351, 0.522]); the F1 point gap is 0.431 (bootstrap mean 0.440 [0.334, 0.552])." |
| M5 | Major | `\begin{table}[H]` before `\label{tab:ablation}` | Table 7 is first mentioned on page 11 and placed on page 14, a 3-page gap. `float_proximity_audit.py` reports 28/29, not the 29/29 in `P0_P1_RESULTS_INDEX.md`. | The table is unreadable at the point of discussion, and the build gate fails. The index is stale. | Change `[H]` to `[pos=htbp]` or move the table block before the §5.2 paragraph; re-run the audit and update the index. |
| M6 | Major | Lines 710, 868, 997 (see text-weight section) | Three paragraphs carry 891 words and 46 sentences. §5.4 also re-derives McNemar, Holm, the bootstrap, and Wilcoxon in five equations before reporting a result. | This is the largest section in the paper, and the densest passages read as tables in prose. The interpretation is hard to find. | Split the three paragraphs as described below; move the five statistical equations to §4.5; cut the per-seed rank recitation in §5.2. |
| M7 | Major | Table 16 row: `RoBERTa-base & 256 tokens`; same table note: "RoBERTa timing is encoder-only (batch 8, fp16, 512 tokens, RTX 3060)" | The timing artifact uses `max_length = 512`, and 81/200 rows were truncated. The 6.97 ms figure is a 512-token timing, not a 256-token timing. | The table contradicts itself, and the 18× comparison is not against the 256-token baseline. The claim is conservative but the presentation is wrong. | Change the row to "512 tokens (timing)" and add "the baseline uses 256 tokens, so 6.97 ms is an upper bound on its inference time." |
| M8 | Major | "Both models fail to transfer; the dialogue failure is the all-positive baseline" | On dialogue, the conventional m0 AUROC is 0.741, not chance. The failure is F1 and flag rate, not ranking. | The paper elsewhere separates ranking from thresholded decisions. This sentence blurs them in the central negative result. | "On dialogue, both models collapse to the all-positive baseline in F1, though their AUROCs stay above chance; on summarization, both rank below the length-only model." |
| M9 | Major | Table 5 note: "EC-XGB is the m3 configuration." | The row is the tuned m3 checkpoint (F1 0.9855, AUROC 0.9984). Table 6's m3 is the fixed-hp refit (0.9854 / 0.9986). The note labels the conventional row but not the EC-XGB row. | The tuned-versus-fixed-hp distinction is the paper's central comparison. A reader cannot tell which m3 is in Table 5. | Add "The EC-XGB row is the tuned m3 checkpoint; the m3 row in Table 6 is the fixed-hp refit." |
| M10 | Major | "The seen-task gain comes from the multi-source rows, not the claim features" | The m3−m2 contrast bundles the RAGTruth rows with the source indicator. The source-zero sensitivity is on MedHallu, not on the seen RAGTruth tasks. | This is a causal claim about the paper's main contribution. The evidence supports "not the claim features" but not the split between rows and indicator. | Soften to "The gain is not from the claim features; it appears when the multi-source rows and source indicator are added. The MedHallu source-zero check suggests the indicator is not the main driver, but a seen-task source ablation would settle it." |
| m1 | Minor | "AUROC 0.458 and 0.491 vs 0.582 for EC-XGB" | Per-task encoder values are compared to a pooled EC-XGB value. EC-XGB's per-task values are 0.736 and 0.770. | The comparison is apples-to-oranges and understates the gap. | "0.458 on summarization and 0.491 on data-to-text, against 0.736 and 0.770 for EC-XGB on those tasks." |
| m2 | Minor | "The XGBoost margin over the random forest"; "That is the primary comparison" | "XGBoost" could mean conventional or EC-XGB. "That" follows the fixed-hp p=0.789 clause but refers to the deployed p=0.45 comparison. | Two ambiguous references in the statistical section. | "The conventional XGBoost margin…"; "The deployed EC-XGB-versus-conventional comparison is the primary test…" |
| m3 | Minor | "the range is about the size of the model gap on this benchmark" | The prompt F1 range is 0.040. The length-matched gap is 0.042; the natural-QA gap is 0.255. | The reader cannot tell which gap is meant. | "the 0.040 F1 range is close to the 0.042 length-matched gap between EC-XGB and GPT." |
| m4 | Minor | Table 14 note: "Large rate" is undefined | The artifact defines it as the share of changed cases with abs(Δscore) > 0.3. | A reader cannot interpret the column. | Add "Large rate is the share of changed cases with abs(Δscore) > 0.3." |
| m5 | Minor | Table 18 note: "18 of the 36 false negatives" | The table is the conventional B2 model (36 FN), while the paragraph just reported m3's 35 FN. | The reader may attribute the count to m3. | "18 of the conventional model's 36 false negatives." |
| m6 | Minor | "The conventional m0 model flags 99.2% of dialogue rows and 97.7% of summarization rows, with AUROC 0.741 and 0.605." | Table 11 has no m0 rows for dialogue or summarization. | The numbers are in `task_shift.csv` but not in the table the sentence cites. | Add the m0 rows or say "not tabulated" and cite the artifact. |
| m7 | Minor | "the display calibrator gives readable target scores" | The target-calibrated max score is 0.313, below the API's lowest band of 0.35, and P1 flags no rows at 0.5. | "Readable" overclaims usability and contradicts the earlier caveat. | "the target calibrator lowers ECE to 0.125, but the scores stay below the API's lowest risk band." |
| m8 | Minor | "no published HaluEval QA result uses the grouped split" | No citation and likely overbroad. The claim should be about this 70/15/15 grouped split. | A reviewer may challenge the literature claim. | "no published HaluEval QA result uses this 70/15/15 grouped split." |
| m9 | Minor | "The deployed checkpoint reaches 0.9855 under its own tuned configuration, 0.0001 above the fixed ablation." | Table 6's fixed-hp m3 is 0.9854, but Table 7's refit is 0.9861. The reader sees 0.9855 as 0.0006 below, not 0.0001 above. | Ambiguous reference in the tuned-versus-fixed-hp discussion. | "0.0001 above the fixed-hp m3 row in Table 6; the Table 7 refit is 0.0006 higher because the feature matrix is rebuilt." |
| m10 | Minor | P2: "recall of 5.0% … precision 15.7%, F1 0.076, Wilson 95% intervals" | The threshold 0.312738 is in `plot_revision_figures_a.py`, but the P2 metrics and Wilson intervals are not in an indexed P0/P1 artifact. | The audit cannot check the P2 row. | Save a small `p2_operating_point.json` and add it to `P0_P1_RESULTS_INDEX.md`. |
| m11 | Minor | Table 16 note: "Costs are USD per 1,000 predictions" | The EC-XGB cost is marginal, with GPU hardware and electricity excluded. Qwen's `$0` is local inference only. | The table can be read as total cost. | Add "marginal cost; hardware and electricity excluded." |

### Exact replacements for the Critical and Major items

**C1.** Delete: "The labels are a draft for author verification, not independent human annotation, so the counts below are provisional." Keep the final sentence, changed to: "The labels are therefore author-verified."

**M1.** Replace "that is why natural long answers are flagged at 99.9%, calibration cannot repair the ranking, and held-out QA stays near 0.55" with "that is why the conventional model flags 99.9% of natural long answers and EC-XGB still flags 61.3%, calibration cannot repair the ranking, and held-out QA stays near 0.55".

**M2.** Replace the Table 5 note phrase "with the threshold selected on validation and evaluated once on test" with "with the threshold fixed at 0.5; a validation-selected threshold of 0.55 gives F1 0.9514, so the choice does not change the comparison".

**M3.** Replace "its seed-mean F1 is 0.042 higher, but the paired item-bootstrap interval for GPT minus EC-XGB is −0.052 [−0.127, 0.024], which includes zero; the AUROC interval is +0.008 [−0.034, 0.048]" with "its seed-mean F1 is 0.042 higher. The paired item-bootstrap interval for GPT's majority-vote predictions against the EC-XGB seed mean is −0.052 [−0.127, 0.024], which includes zero; the AUROC interval is +0.008 [−0.034, 0.048]".

**M4.** Replace "a gap of 0.436 [0.351, 0.522]; the F1 gap is 0.440 [0.334, 0.552]" with "a point gap of 0.435 (bootstrap mean 0.436 [0.351, 0.522]); the F1 point gap is 0.431 (bootstrap mean 0.440 [0.334, 0.552])".

**M5.** Change `\begin{table}[H]` at line 621 to `\begin{table}[pos=htbp]`, or move the table block before the paragraph that first cites it. Re-run `float_proximity_audit.py` and update the 29/29 line in `P0_P1_RESULTS_INDEX.md`.

**M6.** Split the three paragraphs as follows. In §5.3, start a new paragraph at "The conventional target-thresholded mode (P2)…" and another at "At the 0.5 threshold, P1 flags no RAGTruth QA rows…". In §5.4, start a new paragraph at "On held-out RAGTruth QA…". In §5.5, start a new paragraph at "The length-matched panel changes the ranking." Move the McNemar, Holm, bootstrap, and Wilcoxon equations to §4.5.

**M7.** Change the Table 16 row to `RoBERTa-base & 512 tokens (timing) & 125M params & 6.97 & training only` and add "the baseline uses 256 tokens, so 6.97 ms is an upper bound on its inference time" to the note.

**M8.** Use the replacement in the table above.

**M9.** Add "The EC-XGB row is the tuned m3 checkpoint; the m3 row in Table 6 is the fixed-hp refit" to the Table 5 note.

**M10.** Use the softened sentence in the table above.

## Text weight and readability

The Results section is about 8,300 words including tables and equations. Excluding floats, the prose is still roughly 5,500 words, and several paragraphs exceed the 2–7 sentence target in `docs/writing-guideline.md`.

### Worst three paragraphs

1. **§5.3, line 710: 348 words, 18 sentences.** This is the worst paragraph in the section. It covers the P0 source operating point, the 17.8% test prevalence, the 31.1% calibration prevalence, the prior-only predictor, the 0.146 Brier comparison, P2, the Wilson intervals, the random-flagging argument, and the API card caveat. It also states the prior-only Brier twice and repeats "Calibration fixes the scale, not the ranking" from the previous paragraph. The reader has no place to pause. Split into three paragraphs: P0, target calibration and the prior, P2 and the deployment caveat. Cut the duplicate Brier sentence and the repeated calibration phrase.
2. **§5.5, line 997: 296 words, 14 sentences.** The paragraph mixes natural-prevalence QA, cost and latency, the adapted encoder, the length-matched panel, the paired bootstrap, and prompt sensitivity. The encoder result (0.694 vs 0.582) repeats §5.1 almost word for word. Split at "The length-matched panel changes the ranking" and cut the repeated encoder comparison with "as in §5.1".
3. **§5.4, line 868: 247 words, 14 sentences.** The paragraph recites every delta in Table 13, then adds the same intervals in prose. The five equations before it re-derive tests that §4.5 already defines. The reader reaches the interpretation ("the claim features do not add an aggregate gain") only after ten numbers. Move the formulas to §4.5 and report only the deltas that change a conclusion: m3−m0 F1 and AUROC on HaluEval, m3−m0 on QA, and the pooled adaptation row. The other intervals can stay in the table.

### Worst table recitation

**§5.2, line 653: 172 words, 9 sentences.** The paragraph lists the mean ordering and then the exact per-seed rank order for all three seeds: "seed 42 gives m1 > m3 > m2 > m0, seed 123 gives m1 > m3 > m0 > m2, and seed 456 gives m3 > m2 > m1 > m0." Figure 5 already shows this. Cut the three orderings and keep "The mean ordering is m3 > m1 > m2 > m0, but the per-seed ranks flip, and the differences are below seed noise." This saves about 40 words and makes the ECE point easier to see.

### Redundancy and cuts

- §5.1: "The benchmark is saturated, and the differences are small. The ranking columns tell the same story." Both sentences repeat the preceding "the ordering is not resolved by the data." Keep one.
- §5.1: the logistic and heuristic p-values appear again in §5.4 and Table 12. Cut the parenthetical p-values here.
- §5.2: the five hyperparameters (max depth 5, learning rate 0.05, 300 trees, subsample 0.7, colsample 0.7) appear in the prose and again in the Table 6 and Table 7 notes. Keep them in the notes and write "under the fixed configuration in Table 6."
- §5.3: the prior-only Brier is stated twice in line 710. State it once, then compare.
- §5.3: "Calibration fixes the scale, not the ranking" appears in line 671 and again in line 710. Keep the first.
- §5.5: the adapted-encoder 0.694 vs 0.582 comparison appears in §5.1 and again in §5.5. Keep §5.1 and refer back.
- §5.5: "The heuristic needs no model and costs almost nothing" is visible in Table 16. It can go.
- §5.6: the length-matched judge result appears in Table 17 and again in the F1 gap sentence. The F1 gap is the new information; the 0.907/0.865 restatement is not.

A 10–12% cut is feasible without losing any interpretation. The three paragraphs above account for most of it.

## Number-provenance sample

I checked 49 paper numbers against `artifacts/results/rebuild/p0/`, `p1/`, `fixed_hp/`, and `b6/`. Forty-eight match exactly. One (Table 5 external ensemble) matches numerically but carries a wrong protocol note. The eight issues below combine the remaining number and label problems with the structure and provenance problems found in the same pass.

| # | Check (paper value vs artifact) | Result |
|---|---|---|
| 1 | Table 5 EC-XGB F1 0.9855 vs B6 m3 0.985546 | match |
| 2 | Table 5 EC-XGB AUROC 0.9984 vs 0.998416 | match |
| 3 | Table 5 conventional F1 0.9846 vs B2 xgboost 0.984646 | match |
| 4 | Table 5 RF F1 0.9842 vs B2 rf_full 0.984248 | match |
| 5 | Table 5 external ensemble F1 0.9517 vs P1 `external_lr_train` 0.951715 | value match, note mismatch (M2) |
| 6 | Table 5 external ensemble AUROC 0.9850 vs P1 0.985025 | match |
| 7 | Table 5 delta F1 0.0337 [0.0262, 0.0419] vs P1 paired delta | match |
| 8 | Table 5 delta AUROC 0.0135 [0.0098, 0.0177] vs P1 paired delta | match |
| 9 | Table 6 m0 F1 0.9849 vs fixed-hp m0 0.984881 | match |
| 10 | Table 6 m3 F1 0.9854 vs fixed-hp m3 0.985433 | match |
| 11 | Table 6 m3 AUROC 0.9986 vs 0.998552 | match |
| 12 | Table 6 m3 ECE 0.0078 vs 0.007812 | match |
| 13 | Table 6 RAGTruth m0 AUROC 0.4937 vs B6 0.493676 | match |
| 14 | Table 6 RAGTruth m3 AUROC 0.5898 vs B6 0.589825 | match |
| 15 | Table 7 reference F1 0.9861 vs P0 0.986108 | match |
| 16 | Table 7 lexical delta −0.0171 vs P0 −0.017065 | match |
| 17 | Table 7 length delta −0.0066 vs P0 −0.006590 | match |
| 18 | Table 7 NLI delta −0.0040 vs P0 −0.004013 | match |
| 19 | Table 8 m3 raw ECE/Brier 0.731/0.689 vs 0.73097/0.68855 | match |
| 20 | Table 8 m3 target ECE/Brier 0.125/0.161 vs 0.12510/0.16082 | match |
| 21 | Table 8 prior test Brier 0.146 vs 0.146173 | match |
| 22 | Table 8 prior calibration Brier 0.164 vs 0.163838 | match |
| 23 | Table 9 QA 0.944/0.685/1.59 vs disjoint means | match |
| 24 | Table 9 class-conditional 0.886/0.887/1.69 vs disjoint means | match |
| 25 | Table 9 FaithBench 0.320/0.003 vs disjoint means | match |
| 26 | Multicalibration ≤60 words ECE 0.275 vs 0.275 | match |
| 27 | Multicalibration one-claim ECE 0.346 vs 0.346 | match |
| 28 | FaithBench target/raw ECE 0.629/0.453 vs 0.62912/0.45287 | match |
| 29 | Table 11 MedHallu pair 0.867 [0.845, 0.887] vs 0.867 [0.845, 0.887] | match |
| 30 | Table 11 HaluBench AUROC 0.600 vs 0.600172 | match |
| 31 | Table 11 dialogue AUROC 0.763, flag 99.3% vs 0.763394, 0.993 | match |
| 32 | Table 11 summarization AUROC 0.717, flag 37.5% vs 0.716722, 0.375 | match |
| 33 | Retrieval recall@1 0.94, @5 1.00, MRR 0.959 vs 0.94/1.0/0.959 | match |
| 34 | Retrieval delta AUROC −0.023 [−0.061, 0.011] vs −0.02337 [−0.06119, 0.01056] | match |
| 35 | Table 17 GPT F1 0.565, AUROC 0.910 vs `metrics_openai_full` | match |
| 36 | Table 17 Qwen F1 0.602, AUROC 0.778 vs 0.60177/0.77830 | match |
| 37 | Table 18 length rows 524/514/474/641/600/247 vs B5 | match |
| 38 | Table 15 p50/p95 127.51/234.59 vs 127.51/234.59 | match |
| 39 | CPU latency 2.00 s, p50 0.98 s, p95 7.56 s vs JSON | match |
| 40 | SHAP seed Spearman 0.972/0.950 vs 0.9717/0.9501 | match |
| 41 | SHAP top-10 share 77.9% vs 77.881% | match |
| 42 | TreeSHAP margin 9.2e-6 vs 9.2406e-6 | match |
| 43 | B2-vs-m3 Kendall τ 0.730 vs 0.72953 | match |
| 44 | Author Fleiss κ 0.87/0.94/0.76 vs 0.868/0.940/0.761 | match |
| 45 | Author agreement 94/98/88 vs 0.94/0.98/0.88 | match |
| 46 | Length-matched EC-XGB F1 0.907 vs 0.90710 | match |
| 47 | Length-matched GPT F1 0.865 vs 0.864998 | match |
| 48 | Prompt F1 range 0.814–0.854 vs prompt_sensitivity.csv | match |
| 49 | P2 threshold 0.313 vs 0.312738 | match |

### Mismatches and label problems

1. **Table 5 external-ensemble note vs values (M2).** The note says the threshold was selected on validation. The table values are the threshold-0.5 row. The validation-selected row is 0.9575/0.9453/0.9514.
2. **Table 16 token count (M7).** The row says 256 tokens; the note and `encoder_timing.json` say 512.
3. **§5.6 length-matched F1 gap (M4).** 0.440 is the bootstrap mean; the table difference is 0.431. AUROC 0.436 vs table difference 0.435.
4. **§5.5 GPT−EC-XGB delta (M3).** −0.052 is the bootstrap mean for the GPT majority vote; the table difference is −0.042. AUROC +0.008 vs table +0.001.
5. **§5.2 "0.0001 above the fixed ablation" (m9).** Table 6 m3 is 0.9854, so 0.9855 is 0.0001 above. Table 7's refit is 0.9861, so the reader sees 0.0006 below. The reference is ambiguous.
6. **Float gate (M5).** Current `float_proximity_audit.py`: `n_floats 29`, `n_ok 28`, `n_bad 1`, `tab:ablation`, pages 11→14, gap 3. `P0_P1_RESULTS_INDEX.md` says 29/29. `SCORE_ESTIMATE.md` says 29 pages, current PDF 28.
7. **Audit provenance (C1).** The paragraph says "draft/provisional" and "author-verified" in the same breath.
8. **Table 18 false-negative count (m5).** 36 is the conventional B2 count; m3 has 35. The table caption says B2, but the surrounding paragraph discusses m3.

Items 1-4 are concrete table-versus-text mismatches. Items 5-8 are labeling or documentation problems. None of the headline conclusions change.

## Top 3 fixes

1. **Fix the audit provenance contradiction (C1).** Delete the "draft/provisional" sentence and keep the author-verified statement. This is the only issue that can damage trust in the paper's central honesty claim, and it is a one-sentence fix.
2. **Correct the two misdescriptions that change what the evidence says (M1, M2).** The 99.9% flag rate belongs to the conventional model, not EC-XGB, and the external ensemble was not thresholded on validation. Both are local wording fixes, but both affect how a reviewer reads the negative results.
3. **Cut the three densest paragraphs and fix the float (M5, M6).** Split the §5.3 operating-point paragraph, the §5.5 judge paragraph, and the §5.4 delta paragraph; move the statistical formulas to §4.5; change Table 7 to a normal float. This makes the section readable and restores the 29/29 gate.

## Summary

The Results section is scientifically strong. The experimental design is careful, the negative results are reported honestly, and most numbers trace to artifacts. The problems are presentation and provenance, not the core analysis. The Critical issue is the contradictory audit wording. The Major issues are the misattributed 99.9% flag rate, the external-ensemble threshold note, two bootstrap estimands reported as point differences, the Table 7 float gap, the 256/512 token contradiction, the tuned/fixed-hp labeling gap, the task-shift conflation, and the text weight of three paragraphs. With those fixes, this would be a strong Results section. The one item that may need a new run is the seen-task causal claim (M10), and even there the paper can soften the sentence instead of adding an experiment.
