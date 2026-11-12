# Results and claims audit — `Journal_Paper/halurisc.tex`

**Scope.** Results and claims only. Prose style, structure, method reproducibility, and reference formatting were not reviewed. Sources read: `Journal_Paper/halurisc.tex` (all 1,024 lines), `Journal_Paper/out/halurisc.pdf` (25 pages), `docs/rebuild/P0_P1_RESULTS_INDEX.md`, `docs/rebuild/SCORE_ESTIMATE.md`, and the result files under `artifacts/results/rebuild/`, `artifacts/results/rebuild/fixed_hp/`, `artifacts/results/b6/`, and `artifacts/results/b2/`.

**Path note.** There is no top-level `artifacts/results/fixed_hp/`; the fixed-hp artifacts are at `artifacts/results/rebuild/fixed_hp/`. The manuscript's "fixed-hp" numbers were checked against that directory.

**PDF check.** The 25-page PDF renders the same values as the `.tex` for the sampled tables: Table 9 on p. 15, Table 10 on p. 16, the author-review paragraph on p. 19, Table 18 on p. 22, and the conclusion on p. 23. The PDF's Table 9 (0.944/0.685), Table 10 (61.3%/0.5815), and conclusion (0.9854/0.9986; 96.8%/0.558; 86.7%/0.493) all match the source and artifacts.

---

## Verdict

**CONDITIONAL PASS.** The headline claims, the abstract, the conclusion, and the values in Tables 5–11, 17, and 18 are numerically consistent with the artifacts. No headline result is wrong, and the negative results are disclosed. However, five medium-severity provenance/scoping errors should be fixed before submission:

1. the encoder comparison in §5.1 compares per-task encoder AUROCs to EC-XGB's pooled AUROC;
2. the author-review agreement values for explanation and taxonomy are swapped;
3. the WP5 audit is said to cover eight false positives when it covers seven;
4. Table 10's m3 row is the tuned checkpoint but the caption does not say so, while Tables 11 and 17 use the fixed-hp checkpoint;
5. the adapted encoder's 0.694 is described as a "seen tasks" number although it is pooled over seen and unseen tasks.

None of these changes the paper's conclusions. They are reviewer-visible because each can be checked against a table or artifact.

---

## Checks completed: 15/15

| # | Check | Result |
|---:|---|---|
| 1 | Abstract numbers match body and tables | **PASS** — all abstract numbers reconcile to Tables 6, 8, 9, 11, 17, and §1/§5. |
| 2 | Conclusion numbers match abstract and body | **PASS** — 0.9854/0.9986, 99.9→61.3%, 96.8%/0.558, 0.731→0.125, 0.944/0.685, 86.7%/0.493 all match. |
| 3 | Fixed-hp vs tuned checkpoint labels correct everywhere | **FAIL** — Table 10 is the tuned B6 run but the caption omits "tuned"; Tables 11 and 17 are fixed-hp but omit "fixed-hp"; the MedHallu source-zero sensitivity is on the tuned model, not the fixed-hp table model. |
| 4 | Table 5 model comparison values match artifacts | **PASS** — all 12 rows checked against `b2`, `b6`, `review_extras`, `external_baseline`, and `neural_baseline`. |
| 5 | Table 6 component ablation values match artifacts | **PASS** — 4 variants × 7 columns match `rebuild/fixed_hp/b6_model_comparison.csv` and `b6_external_metrics.csv`. |
| 6 | Table 7 feature-group ablation values match artifacts | **PASS** — 9 groups × 4 columns match `p0/feature_group_ablation.csv`. |
| 7 | Table 8 target calibration values match artifacts | **PASS (values)** — all point values match `calibration.csv` and `b4_target_calibration.json`; the slope bracket is an envelope, not a pooled CI (low-severity caveat). |
| 8 | Table 9 conformal values match artifacts | **PASS** — all 6 rows match the disjoint/class-conditional sensitivity in `p1/conformal.json`. |
| 9 | Table 10 adaptation values match artifacts | **PASS (values)** — all 10 rows match the tuned B6 predictions; PR-AUC recomputed from `b6_predictions.parquet` + `external_features.parquet`. |
| 10 | Table 11 external validity values match artifacts | **PASS** — all 10 rows match `p1/medhallu.csv`, `p1/halubench.csv`, and `p1/task_shift.csv`. |
| 11 | Table 17 judge values match artifacts | **PASS (table)** — natural-prevalence and length-matched rows match `judge/metrics_*_full.json` and `p0/judge_length_matched.csv`; the text's F1 CI lower bound is off by 0.001 (low). |
| 12 | Table 18 length-bin values match artifacts | **PASS (values)** — all 6 bins match `b5_length_error_analysis.csv`; the paragraph around the table is about m3 while the table is B2 (low/medium attribution caveat). |
| 13 | Length/overlap attribution supported by the ablation, not overstated | **PASS with caveat** — lexical (−0.0171 F1) and length (−0.0066) are the largest groups, but "mostly a length and overlap detector" is stronger than a leave-one-group-out ablation establishes; the length-matched control (AUROC 0.956 vs 0.521) shows substantial non-length signal. |
| 14 | Judge comparison and task-shift claims scoped correctly; negative results not hidden | **FAIL** — the judge scoping is good and every negative result is reported, but the encoder comparison is per-task vs pooled and "0.694 on the seen tasks" is a pooled number. |
| 15 | No claim implies a general-purpose detector or independent human annotation from scratch | **PASS with caveat** — the abstract and conclusion explicitly say "not a general-purpose detector," and the main text says "three-author blinded review with AI pre-annotation"; the Declarations' "annotated" wording can still imply from-scratch annotation (low). |

**Tally:** 13 pass, 2 fail; 5 medium-severity issues and 10 low-severity caveats are listed below. All 15 checks were completed.

---

## Issues

| ID | Severity | Exact location / quote | Issue | Why it matters | Exact fix |
|---|---|---|---|---|---|
| I1 | **Medium** | `halurisc.tex:557` (§5.1, PDF p. 10): "near chance on each RAGTruth task (AUROC 0.458 and 0.491 vs 0.582 for EC-XGB)" | 0.458 and 0.491 are the **encoder's per-task** AUROCs on summarization and data-to-text (`docs/rebuild/P0_P1_REVIEWER_RESPONSE.md`, line 26). 0.582 is EC-XGB's **pooled** AUROC. EC-XGB's per-task AUROCs are 0.7358 and 0.7705 (Table 10). The sentence compares unlike quantities and says EC-XGB is at 0.582 "on each task." | A reviewer can see Table 10 and conclude the text is wrong. The direction (encoder is weaker) is unchanged, but the gap is understated and the per-task claim is false. | "AUROC 0.458 on summarization and 0.491 on data-to-text, against 0.736 and 0.770 for EC-XGB on those tasks; pooling gives 0.337 for the encoder and 0.582 for EC-XGB." |
| I2 | **Medium** | `halurisc.tex:854` (§5.5, PDF p. 19): "agreement was 0.94 for explanation, 0.98 for taxonomy, and 0.88 for claims" | The two values are swapped. `artifacts/results/rebuild/p0/author_review_summary.json` and `docs/rebuild/p0/author_review/AI_REVIEW_RESULTS.md` give taxonomy 0.940 and explanation 0.980; claims 0.880. | Direct misreport of the author-review result. The number audit passes because both numbers appear somewhere, but the attributes are wrong. | "agreement was 0.98 for explanation, 0.94 for taxonomy, and 0.88 for claims." |
| I3 | **Medium** | `halurisc.tex:955` (§5.6, PDF p. 21): "The WP5 audit covers all eight false positives and 12 of the 35 false negatives" | `artifacts/results/rebuild/explanations/m3_audit_summary.json` has `fp: 7`, `fn: 12`, `borderline: 3`, `other: 3`, `tp: 1`. `m3_audit_case_roles.csv` has seven `m3_fp_*` rows. The model has eight FPs, so one is not in the audit. The reviewer report (`docs/rebuild/P0_P1_REVIEWER_REPORT.md:321`) also says 7 FP. | Overstates audit coverage by one false positive; a reader checking the audit file finds 26 cases with 7 FP, not 8. | "The WP5 audit covers seven of the eight false positives and 12 of the 35 false negatives." |
| I4 | **Medium** | Captions at `halurisc.tex:741` (Table 10), `769` (Table 11), `931` (Table 17): Table 10 "B6 m0 and EC-XGB m3"; Table 11 "External validity on corpora never used..."; Table 17 "Judge stress tests..." | Table 10 is the **tuned B6** run (`artifacts/results/b6/b6_external_metrics.csv`); Tables 11 and 17 are the **fixed-hp m3** (`rebuild/fixed_hp/...`; `judge/comparison_rows.csv`). The captions do not say. The same "m3" label gives RAGTruth QA AUROC 0.5491/flag 97.4% in Table 10 and 0.5577/96.8% in Tables 8 and 17. | The reader cannot reconcile the two "EC-XGB m3" RAGTruth QA rows. This is the exact fixed-hp/tuned check, and it currently fails at the caption level. | Add "tuned" to the Table 10 caption; add "fixed-hp m3" to the Table 11 and Table 17 captions; add one sentence: "Tables 8, 11, and 17 use the fixed-hp refit; Table 10 uses the tuned B6 checkpoint." |
| I5 | **Medium** | `halurisc.tex:557` and `927` (§5.1, §5.5): "reaches 0.694 on the seen tasks, above EC-XGB's 0.582" and "reaches 0.694 on seen RAGTruth tasks" | 0.6940995 is `external.ragtruth_all_test.raw_auroc.mean` in `artifacts/results/review/neural_adapted/neural_adapted_summary.json`. `ragtruth_all_test` is 2,700 rows = QA (0.4898) + summarization + data-to-text. It is a pooled all-task mean, not a seen-task-only mean. | A pooled all-task number is labeled as a seen-task number. A reviewer can recompute the seen-task-only mean and find the claim mis-scoped. It also makes the encoder's seen-task advantage look smaller than it is. | "reaches 0.694 on the pooled RAGTruth test set" in both places, or compute and report the summarization/data-to-text mean separately. |
| I6 | **Low** | `halurisc.tex:847` (Table 15) and `812` (§5.4): "+0.0962 [0.0644, 0.1295]" | `artifacts/results/rebuild/intervals/intervals.csv`, `fixed_hp,ragtruth_all_test,m3-m0,auroc,pooled`, `point_delta = 0.0961496638`. At four decimals that is 0.0961, not 0.0962. | Small rounding error in an otherwise artifact-exact table. | Change +0.0962 to +0.0961. |
| I7 | **Low** | `halurisc.tex:927` (§5.5): "The paired interval for GPT minus EC-XGB is −0.052 [−0.127, 0.024]" | `artifacts/results/rebuild/p0/judge_length_matched.json`, `gpt_majority_vs_ec_xgb_mean.f1_ci95 = [-0.1264811065, 0.0237777064]`. At three decimals the lower bound is −0.126. | Lower bound is off by 0.001; minor, but the interval is presented as an exact artifact value. | Change −0.127 to −0.126. |
| I8 | **Low** | `halurisc.tex:681` (Table 8): slope "0.18 [0.12, 0.27]" | The bracket is the envelope of the three per-seed 95% CIs in `artifacts/results/rebuild/calibration/calibration_slope.csv`: [0.1212, 0.1962], [0.1183, 0.1948], [0.1876, 0.2651]. It is not a pooled 95% CI; no pooled interval exists in the artifact. | Looks like a 95% CI but is a min/max range. A pooled interval would be wider or differently centered. | Write "0.18 (per-seed 95% CIs 0.12–0.27)" or compute a pooled bootstrap interval and report that. |
| I9 | **Low** | `halurisc.tex:955` and caption at `966`, PDF pp. 21–22: "On the seed-42 fixed-hyperparameter run, m3 makes 43 errors... Table 18 shows why length matters" / caption "conventional B2 model" | Table 18 is B2: `artifacts/results/b5/b5_length_error_analysis.json` uses `b2_predictions.parquet`; its FP sum is 10 and FN sum 35.67. m3 seed 42 has 8 FP and 35 FN. The qualitative length pattern is similar, but the table's FP/FN rates are not m3's. | The paragraph announces m3's 43 errors, then uses a B2 table to explain them. A reader may misread Table 18 as m3's error decomposition. | Say "The same length pattern holds for m3; Table 18 reports the B2 decomposition," or add an m3 version of the table. |
| I10 | **Low** | `halurisc.tex:927` (§5.5): "GPT's F1 ranges from 0.814 to 0.854 across three prompts" | `artifacts/results/rebuild/p0/prompt_sensitivity.csv` has 9 GPT runs (3 prompts × 3 seeds). The per-prompt means are 0.815, 0.847, and 0.842; the 0.814–0.854 range includes seed variation. | The 0.040 range is prompt + seed, not prompt alone. The point (prompt choice moves F1 by about the model gap) still holds, but the attribution is loose. | "across three prompts and three seeds," or "per-prompt mean F1 ranges from 0.815 to 0.847." |
| I11 | **Low** | `halurisc.tex:989` (§5.6): "the in-domain model is mostly a length and overlap detector" | Table 7 supports lexical and length as the largest contributors (ΔF1 −0.0171 and −0.0066), but the length-matched control gives EC-XGB AUROC 0.956 against 0.521 for length-only, and the full model still separates classes without length variation. "Mostly" is stronger than a leave-one-group-out ablation establishes. | Check 13. The paper's own matched control qualifies the sentence; a reviewer may call it overstatement. | "the in-domain score is dominated by lexical overlap and length" or "lexical overlap and length are the largest contributors." |
| I12 | **Low** | `halurisc.tex:1009` (Declarations): "The three authors annotated public benchmark rows for internal verification" | The labels originated as AI pre-annotations and were reviewed and accepted by the authors (`docs/rebuild/p0/author_review/AI_REVIEW_RESULTS.md`). The main text says "three-author blinded review with AI pre-annotation." "Annotated" alone can imply from-scratch human annotation. | Check 15. The provenance docs explicitly warn against implying independent human annotation from scratch. | "The three authors reviewed and verified AI pre-annotations on public benchmark rows for internal verification." |
| I13 | **Low** | `halurisc.tex:769` (Table 11 caption): "External validity on corpora never used for training, calibration, threshold selection, or model choice" | The table includes HaluEval dialogue and summarization task-shift rows. HaluEval QA was used for training. The specific rows are held out, but the corpus is not. | Overstates the independence of the HaluEval task-shift rows. | "External validity on held-out rows and task-shift samples never used for training, calibration, threshold selection, or model choice." |
| I14 | **Low** | `halurisc.tex:797` (§5.3): "All evaluation sets are balanced 50/50 by construction" | Only the four Table 11 sets are 50/50. RAGTruth QA is 17.8%/31.1%, FaithBench is 68.1%, and RAGTruth pooled is 34.9%. The sentence is in the Table 11 paragraph but reads globally. | A reviewer may think RAGTruth and FaithBench are balanced. | "All four Table 11 evaluation sets are balanced 50/50 by construction." |
| I15 | **Low** | `halurisc.tex:797` (§5.3): "a source-zero sensitivity on MedHallu changes pair accuracy by at most 0.003 and AUROC by at most 0.002" | `artifacts/results/extras/source_ablation.json` says "hyperparameters held fixed at the **tuned** m3 values per seed"; its in-domain F1 is 0.98498, and its protocol is the tuned per-seed configuration, not the fixed-hp configuration used for Table 11. The quoted maxima are **per-seed** changes; the seed-mean changes are about 0.005 pair accuracy and 0.004 AUROC. | The checkpoint label is ambiguous, and "at most" is true only per seed; an aggregated reading is larger. The qualitative conclusion (the indicator is not the main driver) survives. | "in a tuned-m3 sensitivity, the per-seed change is at most 0.003 pair accuracy and 0.002 AUROC," or rerun the sensitivity on the fixed-hp checkpoint. |

---

## Provenance sample

**Sample size: 245 values/claims.** 202 sampled table cells across Tables 5–11, 17, and 18; 24 body numbers in §1 and §5.1–§5.6; 8 abstract numbers; 6 conclusion numbers; 5 audit claims. Of these, **236 matched** within the manuscript's stated rounding and **9 did not**. Only the mismatches are itemized below.

### Check ledger

| Check | What was checked | Result |
|---|---|---|
| 1. Abstract | 8 numbers against Tables 6, 8, 9, 10, 11, 17 and §1/§5 | Pass |
| 2. Conclusion | 6 numbers against Tables 6, 8, 9, 10, 11 | Pass |
| 3. Checkpoint labels | Captions and prose for Tables 5, 6, 7, 8, 10, 11, 17 | **Fail** — see P9 |
| 4. Table 5 | 40 sampled cells across the 12 rows against `b2`, `b6`, `review_extras`, `external_baseline`, `neural_baseline` | Pass |
| 5. Table 6 | 28 cells against `rebuild/fixed_hp/b6_model_comparison.csv` and `b6_external_metrics.csv` | Pass |
| 6. Table 7 | 36 cells against `p0/feature_group_ablation.csv` | Pass |
| 7. Table 8 | 18 cells against `calibration.csv` and `b4_target_calibration.json` | Pass (values); P6 |
| 8. Table 9 | 15 cells against `p1/conformal.json` sensitivity | Pass |
| 9. Table 10 | 21 cells recomputed from `b6_predictions.parquet` + `external_features.parquet` | Pass (values); P9 caption |
| 10. Table 11 | 18 cells against `p1/medhallu.csv`, `halubench.csv`, `task_shift.csv` | Pass |
| 11. Table 17 | 14 sampled cells (two rows × seven columns) against `judge/metrics_*_full.json` and `p0/judge_length_matched.csv`; paired CI checked | Pass (table); P5 |
| 12. Table 18 | 12 cells against `b5_length_error_analysis.csv` | Pass (values); I9 |
| 13. Length/overlap attribution | 8 claims against Table 7, length-only baseline, matched control | Pass with caveat (I11) |
| 14. Judge/task-shift scoping | 7 claims against `neural_adapted_summary.json`, `task_shift.csv`, judge reports | **Fail** — P7/P8 |
| 15. General-purpose / human annotation | 6 claims against abstract, conclusion, `AI_REVIEW_RESULTS.md` | Pass with caveat (I12) |

### Mismatches only

| # | Location | Manuscript | Artifact | Difference |
|---:|---|---|---|---|
| P1 | §5.5, `halurisc.tex:854` | explanation agreement 0.94 | `author_review_summary.json`: explanation 0.980 | Wrong attribute (taxonomy is 0.940). |
| P2 | §5.5, `halurisc.tex:854` | taxonomy agreement 0.98 | `author_review_summary.json`: taxonomy 0.940 | Wrong attribute (explanation is 0.980). |
| P3 | §5.6, `halurisc.tex:955` | "all eight false positives" | `m3_audit_summary.json`: `fp: 7` | One FP not audited. |
| P4 | Table 15, `halurisc.tex:847`; §5.4, `812` | +0.0962 | `intervals.csv`: 0.0961496638 | Rounds to +0.0961. |
| P5 | §5.5, `halurisc.tex:927` | −0.127 lower bound | `judge_length_matched.json`: −0.1264811 | Rounds to −0.126. |
| P6 | Table 8, `halurisc.tex:681` | slope [0.12, 0.27] | `calibration_slope.csv`: three per-seed CIs, no pooled interval | Bracket is an envelope, not a pooled CI. |
| P7 | §5.1, `halurisc.tex:557` | encoder 0.458/0.491 vs EC-XGB 0.582 | per-task EC-XGB 0.7358/0.7705; pooled 0.5815 | Per-task vs pooled comparison. |
| P8 | §5.1/§5.5, `557`/`927` | 0.694 "on the seen tasks" | `neural_adapted_summary.json`: 0.6940995 on `ragtruth_all_test` (QA + seen tasks) | Pooled value labeled as seen-task-only. |
| P9 | Table 10 caption, `741`; Tables 11/17 captions | "EC-XGB m3" unqualified | Table 10 tuned B6; Tables 11/17 fixed-hp | Checkpoint label missing. |

**Matched values.** The 236 matches include every headline number in the abstract, the conclusion, and Tables 5–11, 17, and 18; they are not itemized here. The nine mismatches above are the complete set found in the 245-value sample.

---

## Top 3 fixes

1. **Fix the encoder comparison and its label.** In §5.1, replace "AUROC 0.458 and 0.491 vs 0.582 for EC-XGB" with "AUROC 0.458 on summarization and 0.491 on data-to-text, against 0.736 and 0.770 for EC-XGB on those tasks; pooling gives 0.337 for the encoder and 0.582 for EC-XGB." In §5.1 and §5.5, change "0.694 on the seen tasks" to "0.694 on the pooled RAGTruth test set" or compute the seen-task-only mean. This is the clearest apples-to-oranges comparison against a table.

2. **Correct the audit provenance.** Swap "0.94 for explanation, 0.98 for taxonomy" to "0.98 for explanation, 0.94 for taxonomy" (`author_review_summary.json`), and change "all eight false positives" to "seven of the eight false positives" (`m3_audit_summary.json`). These are the two places where the paper's audit summary disagrees with its own artifact; the number audit cannot catch either because both numbers and both counts appear elsewhere.

3. **Label the checkpoints consistently.** Add "tuned" to the Table 10 caption, add "fixed-hp m3" to the Table 11 and Table 17 captions, and add one sentence in §5.3: "Tables 8, 11, and 17 use the fixed-hp refit; Table 10 uses the tuned B6 checkpoint." Then label Table 18 as the B2 decomposition (or add the m3 version) so the m3 error paragraph does not appear to be explained by a B2 table.

---

### Bottom line

The paper's central claims are supported: EC-XGB is strong in-domain, its ranking transfers to MedHallu pairs but its probability does not, target calibration fixes ECE without creating ranking skill, conformal coverage hides collapsed positive coverage, and the model fails under task shift and against LLM judges. The negative results are not hidden. The fixes above are provenance and scoping corrections, not changes to the scientific conclusion.

*No manuscript, PDF, artifact, or other existing file was modified. This report is the only file written.*
