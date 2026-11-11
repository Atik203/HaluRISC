# P0/P1 Independent Methods and Statistics Review

- **Reviewer:** independent subagent (methods/statistics)
- **Branch:** `rebuild`
- **Date:** 2026-11-11
- **Manuscript reviewed:** `Journal_Paper/halurisc.tex` (28 pages, 18 tables, 11 figures, 40 references)
- **Artifacts reviewed:** `artifacts/results/rebuild/p0/`, `artifacts/results/rebuild/p1/`, `docs/rebuild/p0/`
- **Constraints respected:** manuscript, artifacts and git state were not edited; nothing was committed.

## Bottom line

The rebuild is substantially more honest than the pre-rebuild draft: the main tables mostly trace to frozen artifacts, the grouped splits are clean, the external corpora are not used for training, and the audits pass. It is **not ready for submission** yet. There are three concrete P0 blockers in the annotation-ingest workflow that would make the three-author review fail or silently pass with one rater, and several P1 misattributions where the abstract or conclusion attaches a number to the wrong model or overstates a result. The most serious scientific issues are the P2-recall misattribution (the 5.0% figure is the conventional B2 model, not EC-XGB), the `0.44` conformal claim in the abstract, the "dialogue and summarization collapse" overclaim, and the mixed tuned/fixed-hp checkpoints in the headline numbers.

---

## P0 blockers

### P0-1. The ingest script rejects every row that an author corrects

**Location:** `src/models/p0_ingest_author_review.py:55-63` (`VERIFIED_STATES`), used by `is_verified()` at line 124 and by the completion gate at lines 313-317.

**What is wrong:** `VERIFIED_STATES` contains `author_verified`, `verified`, `adjudicated`, `final`, `complete`, `completed`, `done`, but **not `author_corrected`**. The review guide instructs authors to set `review_status = author_corrected` whenever they change an AI pre-label (`docs/rebuild/p0/P0_HUMAN_REVIEW_GUIDE.md:79,189,258,324`). `is_verified()` requires every non-null status to be in `VERIFIED_STATES`; a single corrected row therefore makes the sheet unverified, keeps `complete=False`, and makes the script return exit code 1 (`p0_ingest_author_review.py:398`). A review with any correction can never be ingested as complete.

**Evidence:**

```text
VERIFIED_STATES = {'completed','verified','adjudicated','final','author_verified','done','complete'}
'author_corrected' in set: False
is_verified(['author_verified','author_corrected']) -> False
is_verified(['author_corrected']) -> False
```

**Why it matters:** The package explicitly expects corrections (`changed = yes`). The first author who disagrees with an AI label triggers the failure. This is a hard blocker for the three-author review and for the kappa/agreement report.

**Minimal fix:** Add `"author_corrected"` to `VERIFIED_STATES` and add a unit test with a mixed `author_verified`/`author_corrected` sheet. Consider treating `author_corrected` and `adjudicated` as the only terminal states.

### P0-2. The ingest script reads the wrong blind file, so Fleiss' kappa is never computed from the three-author merge

**Location:** `src/models/p0_ingest_author_review.py:53` (`BLIND_FILE = "p0_blind_overlap.csv"`) and line 285 (`read_blind(args.docs_dir / BLIND_FILE)`). The merge instructions write a different file: `docs/rebuild/p0/P0_HUMAN_REVIEW_GUIDE.md:379` and `docs/rebuild/P0_REVIEW_QUICKSTART.md:157` both write `docs/rebuild/p0/p0_blind_overlap_merged.csv`.

**What is wrong:** The documented workflow produces `p0_blind_overlap_merged.csv` with columns `item_id,sheet,a1,a2,a3`. The ingest script reads the original blank `p0_blind_overlap.csv`, which has one blank `author_label` per item and no `a1/a2/a3`. It therefore reports one rater per sheet and a null kappa. The guide's kappa script itself correctly reads `_merged.csv` (`P0_HUMAN_REVIEW_GUIDE.md:428`), so the two halves of the workflow disagree.

**Evidence:** `author_review_summary.json` currently reports:

```json
"claims":      {"items": 20, "raters": 1, "fleiss_kappa": null}
"explanation": {"items": 20, "raters": 1, "fleiss_kappa": null}
"taxonomy":    {"items": 20, "raters": 1, "fleiss_kappa": null}
"warnings": ["claims: rating counts per item are [1], expected 3", ...]
```

`p0_blind_overlap_merged.csv` does not exist. The merged file is only created by the manual command.

**Why it matters:** The paper's required "three-author blinded review with AI pre-annotation" cannot be supported by the current ingest pipeline. The script will either fail or report `raters=1` and no kappa even after all three authors finish.

**Minimal fix:** Make the ingest script prefer `p0_blind_overlap_merged.csv` when present (or add a `--blind` argument), and update the quickstart to pass that path. Keep the raw three copies untouched. Add an integration test that runs the documented merge command and then the ingest script.

### P0-3. The completeness gate passes with one rater

**Location:** `src/models/p0_ingest_author_review.py:313-317`.

**What is wrong:** `summary["complete"]` requires all main sheets verified and `summary["blind"]` to be non-empty, but it never requires three raters or a non-null Fleiss kappa. A blind file with one rating per item is non-empty, so after the main sheets are verified the script can print `complete=True` even though no three-author agreement exists. This is exactly the state currently in `author_review_summary.json` (`raters: 1`, null kappa) and it would be accepted if the sheet statuses changed.

**Why it matters:** The paper's human-review claim is the highest-risk claim in the manuscript. A gate that accepts one rater would let a single-author pass be reported as a three-author review.

**Minimal fix:** Require every blind sheet to have `raters == 3`, every item complete, and `fleiss_kappa is not None` before `complete=True`. Emit an explicit warning when `raters < 3`.

---

## P1 major issues

### P1-1. The abstract and conclusion attribute the P2 5.0% recall to EC-XGB, but the number is from the conventional B2 model

**Location:** `Journal_Paper/halurisc.tex:273` (abstract), `:1064` (conclusion), `:719` (P2 paragraph), `:709` (Table 8 note).

**What is wrong:** The P2 threshold numbers (threshold 0.313, test recall 5.0%, FPR 5.8%, precision 15.7%, F1 0.076) come from `src/models/review_extras.py:120-160` (`x3_threshold_retune`), which reads `artifacts/results/b4/b4_predictions.parquet` — the **B4 conventional B2** predictions, not the m3 EC-XGB predictions. The script's `ece_raw_test` is 0.8185, which matches B2 raw ECE (`b4_calibration_metrics.csv`), not the m3 raw ECE of 0.731. The Table 8 note at line 709 says "The conventional row uses the B2 target-Platt checkpoint and is shown as the reference behind the P2 analysis," so the paper itself hints at the provenance, but the abstract and conclusion say "target calibration lowers ECE from 0.731 to 0.125, but a 5% false-positive budget recalls 5.0%" as if both numbers belong to the same model.

**Why it matters:** This is a direct model-number misattribution in the abstract and conclusion. A reader will conclude that EC-XGB's 5% FPR operating point recalls 5.0%; the m3 target-Platt model actually flags 0.0% at 0.5 and its `recall_at_5pct_fpr` in `calibration.csv` is 0.0396, not 0.05. The 5.0% figure is a B2 result.

**Minimal fix:** Either (a) label the P2 sentence "on the conventional B2 reference" in the abstract and conclusion, or (b) run the same P2 threshold selection on the m3 target-Platt scores and report that number. Do not mix the m3 ECE with the B2 recall in one clause.

### P1-2. The abstract's "0.44 positive coverage" is not traceable and is wrong

**Location:** `Journal_Paper/halurisc.tex:273`.

**What is wrong:** The abstract says "conformal marginal coverage hides 0.44 positive coverage." The measured positive coverage at 90% is **0.685** (Table 9, line 748; `conformal.json` disjoint mean 0.6854). The gap between marginal coverage 0.944 and positive coverage 0.685 is 0.259, not 0.44. There is no artifact with 0.44. The sentence is also grammatically odd: coverage does not "hide 0.44 positive coverage."

**Why it matters:** It is a quantitative error in the abstract, the most-read part of the paper, and it overstates the conformal failure in a way that does not match the table.

**Minimal fix:** Replace with "positive coverage of only 0.685 at 90%" (or "a 0.26 shortfall relative to marginal coverage"). The conclusion already states 0.685 correctly; make the abstract match it.

### P1-3. "Dialogue and summarization collapse to the all-positive baseline" is false for summarization

**Location:** `Journal_Paper/halurisc.tex:273` (abstract), `:1064` (conclusion), `:821` (correct text), Table 11 (`:808-812`).

**What is wrong:** On dialogue, m3 flags 99.3% and F1 0.670, which is the all-positive baseline. On summarization, m3 reaches F1 0.583, flags 37.5%, and AUROC 0.717; the all-positive F1 at 50% prevalence is 0.667, and the length-only baseline ranks summarization better (AUROC 0.788 vs 0.717). The body text at line 821 gets this right: "The dialogue result is the all-positive baseline; the summarization result is below the length-only ranking." The abstract and conclusion overgeneralize the dialogue result to both tasks.

**Why it matters:** It is an unsupported generalization about the model's failure mode. The summarization result is a partial collapse / below-length-only result, not an all-positive collapse.

**Minimal fix:** "On dialogue it collapses to the all-positive baseline; on summarization it ranks below a length-only baseline." Use the same wording in the abstract and conclusion.

### P1-4. The abstract and conclusion mix tuned and fixed-hp checkpoints without labels

**Location:** `Journal_Paper/halurisc.tex:273` (abstract), `:296` (intro), `:1064` (conclusion). Table provenance: `artifacts/results/b6/b6_model_comparison.csv` (tuned) vs `artifacts/results/rebuild/fixed_hp/fixed_hp_ablation.csv` and `medhallu.csv` (fixed-hp).

**What is wrong:** The HaluEval F1 0.9855 and AUROC 0.9984 in the abstract are the **tuned** B6 m3 checkpoint (`b6/b6_model_comparison.csv`). The RAGTruth QA AUROC 0.558, MedHallu AUROC 0.821 / ECE 0.493, and the Table 11 external rows are the **fixed-hp** m3 models. The abstract presents them as one model. The conclusion does the same: "pooled flag rate falls from 99.9% to 61.3%" is the tuned Table 10 row, while "held-out QA remains at 96.8% and AUROC 0.558" is the fixed-hp Table 17 row. The deployed checkpoint is tuned seed 42, but the external validity evidence is fixed-hp three-seed.

**Why it matters:** The headline "EC-XGB" numbers are not a single model. A reader cannot tell whether the external results apply to the deployed checkpoint. The paper's own table notes distinguish the checkpoints, but the abstract and conclusion do not.

**Minimal fix:** Add "tuned" / "fixed-hp" qualifiers to every headline number, or run the deployed tuned seed-42 checkpoint on the external corpora and report that consistently. The cleanest fix is a single sentence in the abstract: "Unless stated otherwise, HaluEval numbers are the tuned B6 m3 checkpoint and external numbers are the fixed-hp m3 three-seed mean."

### P1-5. The "deployed" P1 system flags zero rows on RAGTruth QA at 0.5, and the API's lowest risk band is above the maximum target-Platt score

**Location:** `Journal_Paper/halurisc.tex:538` ("the deployed system uses P1"), Table 8 line 704 (`m3 target Platt ... Flagged 0.0%`), `src/api/main.py:63` (`THRESHOLDS = {"low": 0.35, "medium": 0.60, "high": 1.0}`).

**What is wrong:** `calibration.csv` shows the m3 target-Platt score has `predicted_positive_rate = 0.0` at 0.5, and the maximum target-Platt score in `review_extras.json` is 0.31293. The API's "low" band starts at 0.35, so even the lowest non-low band is above every target-Platt score on RAGTruth QA. The paper says the deployed system uses P1 but never says that P1 flags nothing on this corpus. The display score is later adjusted by claim verdicts in `/verify`, but the pre-verdict score — the one covered by the calibration analysis — is below the first band.

**Why it matters:** The deployed operating point is not usable on the target domain. The paper's "deployment" claim is materially incomplete without this sentence.

**Minimal fix:** State explicitly: "At the 0.5 threshold, P1 flags 0% of RAGTruth QA; the display score is a ranking/scale aid, not a usable target flagger. The deployed card relies on the per-claim verdict adjustment." Alternatively, deploy the P2 threshold and report that.

### P1-6. The external ensemble is trained on 3,000 validation rows while EC-XGB is trained on 24,056 rows; the margin is not a fair architecture comparison

**Location:** `Journal_Paper/halurisc.tex:566` and Table 5 note `:594`; `artifacts/results/rebuild/p1/external_baseline.json` protocol (`fit_split: "halueval_val"`, `n_fit: 3000`); `artifacts/results/rebuild/p0/feature_group_ablation.json` protocol (`train_rows: 24056`).

**What is wrong:** The external reimplementation is fit on the 3,000-row HaluEval **validation** split and evaluated once on test. EC-XGB m3 is trained on 14,000 HaluEval train rows plus 10,056 RAGTruth non-QA rows. The paper reports a 0.0372 F1 / 0.0136 AUROC margin and says "the margin comes from the nonlinear learner plus the claim and source features." That causal attribution is not supported: the baseline has roughly one-eighth of the training data. The baseline is also a reduced five-feature reimplementation, as the paper says, but the training-set asymmetry is not disclosed.

**Why it matters:** This is an unfair comparison used to support a positive claim about EC-XGB's feature design. A reviewer will ask why the baseline was not trained on the training split (or a matched 3,000-row subset of it).

**Minimal fix:** Retrain the external ensemble on the 14,000-row training split, tune on validation, evaluate once on test. If that is not possible, report the 3,000-row fit explicitly and remove the causal sentence "the margin comes from the nonlinear learner plus the claim and source features." The descriptive comparison can stay.

### P1-7. The Table 17 length-only baseline is fit on the target calibration split, so "length-only ranks QA better" is not a zero-shot comparison

**Location:** `Journal_Paper/halurisc.tex:994` ("the length-only baseline at AUROC 0.712 also ranks the QA rows better than the full feature model"), Table 17 note `:1019`; `artifacts/results/rebuild/judge/comparison_rows.csv` length-only note ("fit on the 5,034 QA calibration rows; threshold_at_5pct_fpr chosen on the same split").

**What is wrong:** The length-only baseline in the judge table is a logistic regression fit on the **RAGTruth QA calibration rows** (5,034 rows), while EC-XGB's RAGTruth QA ranking is zero-shot (the classifier never saw QA rows; only the display Platt calibrator is target-fitted, and AUROC is invariant to Platt). The paper uses the 0.712 AUROC to argue that a one-feature target-fitted baseline beats the full feature model. That is not an apples-to-apples ranking comparison. The Table 11 length-only baseline is a different model fit on HaluEval train only; the paper does not distinguish them.

**Why it matters:** This is the paper's strongest negative result about its own model. It needs to be fair, or it will be read as a straw man in reverse: a target-fitted baseline beating a zero-shot model is expected.

**Minimal fix:** Add "fit on the QA calibration split" to the Table 17 note and the sentence; or refit this length-only baseline on HaluEval train and report that AUROC. Do not compare a target-fitted baseline to a zero-shot ranker without the label.

### P1-8. The neural baseline's per-task AUROC 0.455 / 0.477 and pooled 0.315 are not traceable to any artifact

**Location:** `Journal_Paper/halurisc.tex:568` and `:1064`.

**What is wrong:** `artifacts/results/review/neural_baseline/neural_baseline_summary.json` contains `ragtruth_all_test.raw_auroc.mean = 0.3373`, not 0.315, and it has no per-task summarization/data-to-text keys. `neural_baseline_predictions.parquet` has only `dataset` (qa/all/faithbench), no task column, so the 0.455/0.477 cannot be recomputed from the saved predictions. The paper says "AUROC 0.455 for summarization and 0.477 for data-to-text" and "the pooled 0.315 is lower because the two tasks have different positive rates." Neither 0.455, 0.477 nor 0.315 appears in the rebuild artifacts or the review artifacts (a repo-wide search outside `.venv` finds them only in the manuscript and its generated `.txt`).

**Why it matters:** This is a number-provenance failure in a paragraph that compares EC-XGB to a neural baseline. The pooled explanation is also statistically wrong: AUROC is prevalence-invariant within a task; pooling can lower AUROC because the two tasks have different score distributions, not because their positive rates differ (the RAGTruth task-shift report itself says this at `task_shift_REPORT.md:79`).

**Minimal fix:** Save per-task predictions and recompute the per-task and pooled AUROC from one script; or remove the 0.455/0.477/0.315 sentence and use the traceable pooled 0.337. Replace "because the two tasks have different positive rates" with "because pooling mixes between-task score shifts with within-task ranking."

### P1-9. "The Brier score follows the same pattern" is false in the multicalibration analysis

**Location:** `Journal_Paper/halurisc.tex:760`.

**What is wrong:** The multicalibration CSV shows the worst ECE bins have the **lowest** Brier scores, not the same pattern:

| bin | ECE | Brier |
|---|---:|---:|
| length <=60 | 0.275 | 0.131 |
| length 161+ | 0.082 | 0.228 |
| claim count 1 | 0.346 | 0.120 |
| claim count 6+ | 0.086 | 0.170 |

The ECE and Brier rankings are opposite for both subgroupings. Saying "The Brier score follows the same pattern" is a factual error and hides the disagreement between the two calibration summaries.

**Why it matters:** The paragraph's conclusion — "the average is not a safe summary for short answers or one-claim answers" — is supported by ECE but contradicted by Brier. A strict reviewer will see the table and question the analysis.

**Minimal fix:** "The Brier score does not follow the ECE ranking: it is lower in the short and one-claim bins, so ECE and Brier disagree about where the calibration error is concentrated." Then explain which metric is appropriate for the decision.

### P1-10. The prompt-sensitivity AUROC range is wrong (0.920–0.944 stated; 0.910–0.954 observed)

**Location:** `Journal_Paper/halurisc.tex:994`.

**What is wrong:** `prompt_sensitivity.json` `overall_range.auroc` is min 0.910, max 0.954 across the nine prompt-seed runs. The paper says AUROC ranges from 0.920 to 0.944. The F1 range 0.814–0.854 is the across-run range, so the sentence mixes an across-run F1 range with an across-prompt-mean AUROC range. The per-prompt means are F1 0.815/0.847/0.842 and AUROC 0.920/0.944/0.942; the all-run ranges are F1 0.814–0.854 and AUROC 0.910–0.954.

**Why it matters:** The prompt-sensitivity result is used to argue that prompt choice can move the judge by about as much as the model gap. The wrong AUROC range weakens or changes that argument.

**Minimal fix:** Use consistent units. Either "across the nine prompt-seed runs, F1 0.814–0.854 and AUROC 0.910–0.954" or "across prompt means, F1 0.815–0.847 and AUROC 0.920–0.944."

### P1-11. The required three-author wording is absent, and the paper still says a "second rater" is needed

**Location:** `Journal_Paper/halurisc.tex:463` ("released for three-author blinded review"), `:943` ("A second rater is needed before Cohen's kappa can be reported"), `:1024` ("input to the three-author review").

**What is wrong:** The required wording is **"three-author blinded review with AI pre-annotation."** The paper never uses it. It says "three-author blinded review" but not "with AI pre-annotation." Separately, line 943 says a second rater is needed before Cohen's kappa can be reported; that is a two-rater framing and conflicts with the three-author Fleiss' kappa design in `docs/rebuild/p0/P0_HUMAN_REVIEW_GUIDE.md:389-405`. The B5 26-case audit and the P0 250-row package are different studies; the sentence conflates them.

**Why it matters:** The human review is not complete. The paper must not imply that it is, and it must use the agreed wording. "Second rater" invites the reader to think the design is two humans, not three authors.

**Minimal fix:** Replace line 463 with the exact phrase "a three-author blinded review with AI pre-annotation" and add "the review is not complete; no human accuracy or kappa is reported." Replace line 943's "second rater" with "the three-author review is not complete, so no kappa is reported for this audit."

### P1-12. The annotation manifest identifies EC-XGB, not the AI annotator model

**Location:** `docs/rebuild/p0/annotation_manifest.json` `model` block and `prompt_definition`.

**What is wrong:** The manifest's `model` block is the **scored** EC-XGB model (`name: EC-XGB`, `variant: m3`, `seed: 42`, artifact path, SHAP settings). The `prompt_definition` says "No external LLM API prompt was used; the AI pre-annotation is the agent's reading of the source rows against the codebook." The actual AI annotator model/agent is not named. The P0 checklist requires the manifest to list "the AI model, prompt version, codebook SHA-256, and date." The codebook hash and date are present; the AI annotator is not.

**Why it matters:** The annotation provenance is incomplete. A reader cannot reproduce or audit the pre-annotation without knowing which model/agent produced it. This is a P1 annotation-integrity issue even though the hashes and counts are correct.

**Minimal fix:** Add a separate `annotator` block with the agent/model name, version, and any decoding settings; keep the EC-XGB block as `scored_model`. Update the checklist to distinguish the two.

### P1-13. The deployed display calibrator was selected on the test split; the title and abstract still claim "calibrated" without qualification

**Location:** `Journal_Paper/halurisc.tex:1057` ("The deployed display calibrator was selected in an earlier fit that compared Platt and isotonic on the test split"), title `:239`, abstract `:273`, conclusion `:1064`.

**What is wrong:** The paper discloses in the limitations that the deployed calibrator's Platt-vs-isotonic choice was made on the test split. The clean Table 8 analysis refits Platt without test selection, but it is the **fixed-hp** m3 model, not the deployed tuned checkpoint. The title "Calibrated Hallucination Risk Estimation" and the abstract's "calibrated risk score" therefore rest on a calibrator choice that is not test-clean, and the clean analysis is for a different checkpoint.

**Why it matters:** Test-set selection for the deployed calibrator is a leakage/model-selection issue. Disclosing it is good, but the paper still uses "calibrated" in the title and headline without the qualifier. A strict reviewer will treat this as a validity problem, not a footnote.

**Minimal fix:** Refit the deployed calibrator using only validation/calibration rows (choose Platt a priori, as the clean analysis does), or change the title/abstract to "domain-specific risk estimation with post-hoc calibration" and state that the deployed calibrator is not test-clean. The clean Table 8 analysis should be run on the deployed checkpoint or explicitly labeled as a fixed-hp proxy.

---

## P2 minor issues

### Numbers and statistics

- **Figure 5 mean ordering is wrong.** `Journal_Paper/halurisc.tex:637` says "the mean ordering is m1 > m3 > m2 > m0." The fixed-hp means are m3 0.985433 > m1 0.985332 > m2 0.984911 > m0 0.984881 (`artifacts/results/rebuild/fixed_hp/b6_per_seed_metrics.csv`). Fix to "m3 > m1 > m2 > m0."
- **The claim block is not "below" the seed SD.** Line 639 says the claim block costs 0.0010 F1 "which is below the seed standard deviation of the full model (0.0010)." The artifact delta is -0.0010165 and the full-model SD is 0.0009729, so the cost is slightly **above** the SD. Fix to "about the same size as" or "not distinguishable from."
- **MedHallu pair accuracy is a seed-averaged score, not a mean of per-seed pair accuracies.** Table 11 (`:794`) says "mean over seeds" and reports 0.867 [0.845, 0.887]. `medhallu.csv` `m3_seedavg` is 0.867; the per-seed pair accuracies are 0.865, 0.867, 0.850 (mean 0.8607 ± 0.0093). Either label it "seed-averaged scores" or report 0.861 ± 0.009.
- **Table 11 "three external corpora"** (`:819`) but the table has four dataset groups (MedHallu, HaluBench, dialogue, summarization). Change to "four external evaluations" or "three external corpora plus the task-shift pair."
- **Table 3 note "four external corpora"** (`:508`) but the table lists five external datasets (FaithBench, MedHallu, HaluBench, dialogue, summarization). Fix the count.
- **Table 5 caption "mean over seeds"** (`:573`) but several rows are deterministic single fits (heuristic, length-only, TF-IDF, external ensemble, majority). Say "mean over seeds where a seed applies; deterministic rows are single fits."
- **Table 6 caption n = 3,000** (`:614`) but the last two columns use the pooled RAGTruth test split (2,700 rows, note `:627`). Say "n = 3,000 for HaluEval columns; n = 2,700 for RAGTruth columns."
- **Table 10 caption "Cross-domain"** (`:766`) but summarization and data-to-text are tasks EC-XGB trained on; the note clarifies adaptation. Rename "External and adaptation performance" or add "seen-task adaptation rows are marked."
- **Table 16 "Model p50"** (`:976`) for EC-XGB m3 is the full 127.51 ms pipeline, not model-only (0.008 ms per note). Rename the column "Pipeline p50 (ms)" or split the rows.
- **Table 17 caption "two judges vs. EC-XGB"** (`:998`) but the table includes length-only and all-positive rows. Say "judges, EC-XGB, length-only and all-positive references."
- **Table 18 "Hallucinated" column** (`:1036`) is a rate, not a count; the caption says counts are seed 42 and rates are three-seed means. Rename "Hallucinated rate."
- **"two judges are 18 to 23 times slower"** (`:994`): 3050/127.5 = 23.9, so 18–24, not 18–23.
- **"Calibration does not improve on the raw model"** (`:666`) is a null claim from ECE point estimates (0.0045 vs 0.0091 vs 0.0070) with no interval. Say "the calibrated ECEs are not lower in this sample" or add a paired bootstrap.
- **Class-conditional conformal still under-covers.** Table 9 reports 0.886 marginal / 0.887 positive at 90%, and 0.936 / 0.925 at 95%. "Restores positive coverage" (`:738`) is too strong; say "raises positive coverage to 0.887/0.925, still slightly below nominal."
- **Subgroup ECEs are noisy.** The worst length bin has n=154 and the one-claim bin has n=36 (`multicalibration.csv`); the paper reports 0.275 and 0.346 without n or intervals. Add n and a bootstrap interval, or call them descriptive.
- **"The calibrator sets the level to the calibration prevalence, and the residual ECE is the prevalence gap"** (`:719`) overstates a single cause. The residual also includes slope/intercept error. Say "the residual is dominated by the prevalence gap."

### Claims and wording

- **Title "Calibrated"** (`:239`) is unqualified while target calibration is poor (MedHallu ECE 0.493, FaithBench 0.629) and the deployed calibrator is test-selected. Consider "Domain-Specific Hallucination Risk Estimation..." or add "Under Shift."
- **"The same model" in the introduction** (`:296`) for MedHallu refers to the fixed-hp m3, while the preceding HaluEval F1 is tuned. Say "the fixed-hp m3 model."
- **"The gain on seen tasks comes from the multi-source training rows, not from the claim features"** (`:610`) confounds two changes: m3 adds both the RAGTruth non-QA rows and the source indicator. The `no_source` ablation is in-domain only. Say "the m3 stack, which adds both rows and the source indicator, drives the seen-task gain; the two are not separated."
- **"The source indicator ... does not drive the result"** (`:819`) is supported only by the MedHallu source-zero sensitivity. HaluBench, RAGTruth QA and FaithBench are not tested. Say "on MedHallu, the source-zero sensitivity changes pair accuracy by at most 0.018..."
- **"Both models collapse"** (`:821`) overgeneralizes from dialogue to summarization; see P1-3.
- **"The in-domain model is a length and overlap detector"** (`:1057`) is strong; NLI is the third contributor (-0.0040 F1) and the claim block is retained for explanations. Say "dominantly a length and overlap detector."
- **MedHallu "1,000 human-annotated medical QA pairs"** (`:487`) overstates: `rebuild_medhallu.py:625` says "human-annotated ground-truth answers with LLM-written hallucinated answers." Say "1,000 medical QA pairs with human-annotated gold answers and paired LLM-written hallucinations."
- **HaluBench provenance is underdescribed.** `p1_halubench_REPORT.md` says labels are HaluBench PASS/FAIL, not original source labels, and DROP/FinanceBench were converted to QA by the HaluBench authors. The paper (`:487`, Table 11 note `:816`) does not say this. Add one sentence.
- **HaluBench/MedHallu/task-shift are balanced 50/50 by construction.** F1 and flag rate are prevalence-dependent and do not describe natural deployment. The paper states 50.0% positive but does not caveat the F1/flag-rate interpretation. Add a sentence to the Table 11 note.
- **"The four external corpora are never used for ... model choice"** (`:508`) is too broad: the source indicator is set to 1 for MedHallu and 0 for HaluBench, which is an input choice. The sensitivity check mitigates this, but the caption should say "no external row is used for fitting or threshold selection."
- **"The deployed system uses P1; P2 is the thresholded alternative"** (`:538`) should say P1 flags 0% at 0.5 on RAGTruth QA (see P1-5).
- **"Deployment uses gold context, so retrieval quality and end-to-end verdict accuracy are not evaluated"** (`:1057`) conflicts with the retrieval stress test (`:823`), which does evaluate retrieved context. Say "web-search retrieval quality and end-to-end verdict accuracy are not evaluated; the retrieval stress test uses the known source passage."
- **The judge comparison limitation** (`:1057`) mentions only the 900-row natural-prevalence set, not the 158-row length-matched set. Add the latter and its domain confound.
- **"A second rater is needed before Cohen's kappa"** (`:943`) conflicts with the three-author design; see P1-11.
- **"two audit scripts"** (`:464`) is stale: there are three number audits (`rebuild_number_audit.py`, `review_number_audit.py`, `p1_number_audit.py`) plus `float_proximity_audit.py` and pytest. Say "the audit scripts" or update the count.
- **"every aggregate table reports a mean"** (`:328`) is false; several tables have deterministic single fits. Say "every seeded aggregate table reports a mean."
- **20-case content group includes "substitution"** (`:1024`) but `docs/rebuild/wp5_error_labels.csv` has no `entity_substitution` in the 20-case draft (substitution appears in the 50-case P0 taxonomy). Remove "substitution" from the 20-case sentence or say "the 50-case sheet also includes substitution."

### Structure and presentation

- **Warning suppression.** `Journal_Paper/halurisc.tex:1087-1088` sets `\hbadness=10000` and `\sloppy` before the bibliography, and `:283` uses `\hfuzz=120pt` around `\maketitle`; the `silence` package (`:40-41`) suppresses a hyperref warning. The "0 overfull/underfull" result is therefore conditional. Remove these or report that they are suppressed.
- **Float-page typo.** `:229-230` defines `\floatpagepagefraction` and `\textpagefraction`, which are not the standard LaTeX lengths (`\floatpagefraction`, `\textfraction`). These lines are redundant with `:21` and `:20`. Remove or correct.
- **Acknowledgements** (`:1085`) thank HaluEval, RAGTruth and FaithBench but omit MedHallu and HaluBench, which are used in the paper. Add them.
- **Data availability** (`:1076`) says "Model artifacts, split indices, and the experiment manifest are included in the code distribution." In this repo `artifacts/models/*` and `artifacts/results/*` are gitignored (`git check-ignore` confirms). If the distribution is a separate zip, say so; otherwise this is inaccurate.
- **Keywords** (`:279`) list seven terms; if the venue expects 4–6, trim one (e.g., drop "SHAP" or "Question answering").
- **Table 3 "four external corpora"** and **Table 11 "three external corpora"** are inconsistent; see above.
- **Table 6 reference n** and **Table 5 caption** are minor caption-text mismatches; see above.
- **The paper has several very long paragraphs** (`:1057`, `:1064`) and many short ones. The "no many tiny paragraphs" check passes by word count (no body paragraph under 40 words), but the long conclusion paragraph is a wall of text. Consider splitting the conclusion into two paragraphs.

---

## What passes

### Number provenance

I traced more than 60 manuscript numbers against JSON/CSV artifacts. The three audits cover 512 checks total and all pass. Representative verified values:

- Table 5: heuristic 0.9352, logistic 0.9660, random forest 0.9842, NLI-only 0.6714, TF-IDF answer 0.9224, TF-IDF full 0.5999, external ensemble 0.9482, conventional XGBoost 0.9846, EC-XGB 0.9855, RoBERTa 0.9912, majority 0.0000 — all match `artifacts/results/b2/b2_model_comparison.csv`, `artifacts/results/b6/b6_model_comparison.csv`, `artifacts/results/review/neural_baseline/neural_baseline_summary.json`, and `artifacts/results/rebuild/p1/external_baseline.csv`.
- Table 6: m0 0.9849/0.9979/0.9701/0.0050, m1 0.9853/0.9976/0.9710/0.0042, m2 0.9849/0.9985/0.9701/0.0051, m3 0.9854/0.9986/0.9713/0.0078 — all match `artifacts/results/rebuild/fixed_hp/fixed_hp_ablation.csv`.
- Table 7: lexical -0.0171/-0.0040/-0.0343, length -0.0066/-0.0012/-0.0127, NLI -0.0040/-0.0004/-0.0080, claims -0.0010/-0.0003/-0.0020 — all match `artifacts/results/rebuild/p0/feature_group_ablation.csv`.
- Table 8: raw ECE 0.731/Brier 0.689, source Platt 0.765/0.742, target Platt 0.125/0.161, prior Brier 0.146 — all match `artifacts/results/rebuild/calibration/calibration.csv` and `prior_only.csv`.
- Table 9: QA 0.944/0.685/1.000/1.59 and 0.972/0.842/1.000/1.75; class-conditional 0.886/0.887/0.886/1.69 and 0.936/0.925/0.938/1.80; FaithBench 0.320/0.003 and 0.323/0.006 — all match `artifacts/results/rebuild/p1/conformal.json` disjoint mean.
- Table 10: pooled m0 0.5181/0.4753/0.3139/99.9%/0.6346 and m3 0.5608/0.5815/0.3588/61.3%/0.2782; QA m0 0.3022/0.5465/0.1911/99.9%/0.8149 and m3 0.3081/0.5491/0.1884/97.4%/0.7433 — all match `artifacts/results/b6/b6_external_metrics.csv` (tuned) and `artifacts/results/review/per_task_metrics.json`.
- Table 11: MedHallu 0.667/0.821/0.493/100%/0.867; HaluBench 0.588/0.600/0.334/50.3%; covidQA 0.833, DROP 0.513, FinanceBench 0.527; dialogue 0.670/0.763/0.488/99.3%; summarization 0.583/0.717/0.113/37.5% — all match `medhallu.csv`, `halubench.csv`, `task_shift.csv`.
- Table 12: McNemar b/c/p 10/6/0.453, 6/8/0.789, 12/8/0.502, 69/11/1.86e-10, 158/9/2.28e-30, 1006/26/5.6e-204 — all match `artifacts/results/rebuild/fixed_hp/b6_statistical_tests.json` and the B2 statistical tests.
- Table 13: m3-m0 HaluEval F1 +0.0006 [-0.0010, 0.0021], AUROC +0.0007 [0.0002, 0.0012], m2-m1 F1 -0.0004 [-0.0020, 0.0012], m3-m1 F1 +0.0001 [-0.0015, 0.0018] — all match `artifacts/results/rebuild/intervals/intervals.csv`.
- Table 14: entity 0.339/0.545, support 0.464/0.000, numeric 0.332/0.000, date 0.482/0.000, insert 0.003/0.000, shuffle 0.0001/0.000 — all match `artifacts/results/rebuild/explanations/m3_perturbation_aggregates.csv`.
- Table 15: total 127.51/234.59, claim 43.75/83.21 — match `artifacts/results/rebuild/latency/m3_latency.json`; CPU 2002.15/975.36/7564.71 — match `m3_latency_cpu.json`.
- Table 17: GPT 0.404/0.942/0.565/0.910/41.4%/2283/0.234; Qwen 0.570/0.638/0.602/0.778/19.9%/3050; EC-XGB 0.183/0.998/0.310/0.558/96.8%; length-only 0.439/0.181/0.257/0.712/7.3%; length-matched GPT 0.865/0.957, Qwen 0.821/0.863, EC-XGB 0.907/0.956 — all match `metrics_openai_full.json`, `metrics_qwen_full.json`, `fixed_hp/calibration.csv`, `judge/comparison_rows.csv`, `p0/judge_length_matched.csv`.

The audit scripts' exact summaries:

```json
rebuild_number_audit.py: {"n_checks": 116, "n_match": 116, "n_mismatch": 0}
review_number_audit.py:  {"n_checks": 298, "n_match": 298, "n_mismatch": 0}
p1_number_audit.py:      {"n_checks": 98,  "n_match": 98,  "n_mismatch": 0}
float_proximity_audit.py:{"n_floats": 29, "n_ok": 29, "n_bad": 0, "bad": []}
```

`python -m pytest tests -q`: **252 passed, 12 warnings in 67.93s** (exit 0).

### Leakage and fairness

- **HaluEval splits are grouped and disjoint.** `feature_group_ablation.json` protocol reports `item_idx_overlap_train_val = 0`, `item_idx_overlap_train_test = 0`, `item_idx_overlap_val_test = 0`; train 24,056 rows, val 3,000, test 3,000. HaluEval QA is 14,000 train + 3,000 val + 3,000 test at the item-group level.
- **RAGTruth QA test rows are not used for training.** `run_b6_modified.py:247-249` (`m3_train_rows`) uses only `task in {summarization, data_to_text}` and `split == train`; the QA task is excluded. The QA calibration block (5,034 rows) is used for the display Platt and P2 threshold, but the 900-row QA test is disjoint from it.
- **MedHallu, HaluBench, HaluEval dialogue/summarization, and FaithBench are not used for training or threshold selection.** The scripts use fixed-hp models and fixed thresholds; `medhallu.json` says `used_for_training_or_tuning: False`; `p1_halubench.py` sets `source_ragtruth = 0` and uses no HaluBench rows for fitting; `p1_task_shift.py` uses the fixed-hp models with no tuning; `p1_conformal.py` fits the quantile on the RAGTruth QA calibration split, never FaithBench.
- **Conformal disjoint split is real.** `conformal.json` reports `n_fit = 2,514`, `n_conformal = 2,520`; the 900 QA test and 750 FaithBench rows are evaluation-only. The main in-sample run is labeled a construction check, not the primary result.
- **Retrieval qrels are the known source passage, not human web-search judgments.** `retrieval_stress.json` protocol says so; the paper repeats this at line 823. Recall@1/5/10 and MRR are reported before any downstream claim; the AUROC delta CI [-0.061, 0.011] includes zero.
- **Source-indicator sensitivity is reported.** MedHallu source-zero changes pair accuracy by at most 0.018 and AUROC by at most 0.007 (`medhallu.json` `sensitivity_m3_source_0`), and the paper says so. The HaluBench source indicator is 0 by design, and the task-shift rows use 0.
- **The external baseline is labeled a reimplementation** and the paper explicitly does not claim the original authors' number (`Table 5` note, `:566`). This is honest even though the training-set asymmetry is a problem (P1-6).

### Statistics

- **Seeds, n, prevalence are stated and mostly consistent.** HaluEval 3,000 rows at 50%; RAGTruth QA 900 at 17.8%; calibration 5,034 at 31.1%; pooled RAGTruth 2,700 at 34.9%; FaithBench 750 at 68.1%; MedHallu 2,000 at 50%; HaluBench 900 at 50%; task shift 2,000 per task at 50%.
- **Bootstrap units are correct for the main claims.** HaluEval uses `item_idx` groups (1,500 groups of two prompt variants); RAGTruth QA uses `source_group_id` (150 groups of six rows); pooled RAGTruth uses 450 groups. The two-level bootstrap resamples groups within seed and then seeds; the CI is a percentile interval.
- **McNemar, Holm and Wilcoxon math is correct.** The continuity-corrected chi-square values reproduce the reported p-values; Holm adjustment over the four baseline comparisons reproduces the table; with three seeds the minimum two-sided Wilcoxon p is 0.25, and the paper says the test cannot reach significance.
- **No unsupported "significant" claim.** The paper says the XGBoost-vs-random-forest difference is not significant (p=0.502), the EC-XGB-vs-conventional difference is not significant (p=0.45/0.789), and the Wilcoxon test cannot reach significance. The significant logistic/heuristic/NLI differences are supported.
- **The length-matched judge comparison is not presented as a controlled domain comparison.** The Table 17 note says "controls length but not domain, so it is a stress test, not a second natural-prevalence estimate" (`:1019`), and the body says "when length is controlled" (`:994`). The abstract could be more explicit, but the main text is fair.

### Claims and wording

- **No claim of "general-purpose detector."** The phrase appears only in the negated conclusion ("not a general-purpose detector") and the abstract ("not a general-purpose detector").
- **No claim of "positive held-out QA."** The paper consistently reports RAGTruth QA as near chance (AUROC 0.547–0.558) and says the held-out result is not resolved.
- **No claim of human accuracy.** The paper says the AI pre-labels "are not reported as human accuracy" (`:463`) and the labels are "not independent human annotation" (`:943`, `:1024`).
- **No claim of "independent human annotation."** The phrase appears only in negations. The required positive phrase is missing (P1-11), but the paper does not overclaim human review.
- **The explanation audit is described as AI-assisted and error-enriched**, with the 26-case counts matching `m3_audit_summary.json` (7 FP, 12 FN, 3 borderline, 3 other, 1 TP) and `wp5_explanation_labels.csv` (13 implausible, 3 unsure, 10 plausible).
- **The P0 package counts match the paper.** `p0_claim_verdicts.csv` has 100 claims from 50 answers with AI counts 63 supported / 14 unsupported / 5 contradicted / 18 unsure; `p0_error_taxonomy.csv` has 50 rows (25/25 labels) with 30 artifact / 20 content; `p0_explanation_audit.csv` has 100 rows.

### Annotation package integrity

- **250 rows across the three main sheets:** 50 taxonomy + 100 explanation + 100 claims.
- **60 blind items, 20 per sheet:** `p0_blind_overlap.csv` has 60 rows, 20 claims / 20 explanation / 20 taxonomy, 60 unique `item_id`s. Seven `sample_id`s repeat in claims because two claims come from one answer; that is expected.
- **All main-sheet rows have `review_status = needs_author_verification`:** 100/100/50. `author_label`, `final_label`, `annotator_id`, `changed`, and `notes` are empty in all three sheets. The blind sheet's `author_label`, `annotator_id`, and `notes` are empty.
- **All manifest hashes match the files:** `codebook_v2.md`, `p0_error_taxonomy.csv`, `p0_explanation_audit.csv`, `p0_claim_verdicts.csv`, `p0_blind_overlap.csv`, `P0_HUMAN_REVIEW_GUIDE.md` — verified with SHA-256.
- **The kappa formula is correct.** `P0_HUMAN_REVIEW_GUIDE.md:395-402` gives the standard Fleiss' kappa formula; the Python implementation at `:413-426` uses `P_i = sum_j n_ij(n_ij-1)/(n(n-1))`, `p_j = sum_i n_ij/(Nn)`, `P_e = sum_j p_j^2`, and `kappa = (P_bar-P_e)/(1-P_e)`. This matches Fleiss (1971).
- **The merge command is correct for the merged file.** It joins the three author copies on `item_id` and produces the expected `(60, 4)` shape with `sheet,a1,a2,a3`. The problem is that the ingest script does not read that file (P0-2), not the merge command itself.

### Structure and build

- **Page count:** 28 pages (`Journal_Paper/out/halurisc.log:1372`).
- **Floats:** 18 tables, 11 figures. The task's numbering matches the `.aux`: Table 5 models, Table 6 component ablation, Table 7 feature ablation, Table 9 conformal, Table 10 cross-domain, Table 11 external, Table 17 judges.
- **Abstract:** 292 words (250–300 target). Six Results subsections: Model Comparison; EC-XGB and Ablation; Calibration, Shift, and External Validity; Statistical Analysis; Explanation, Efficiency, and Judges; Error Analysis and Discussion.
- **References:** 40 BibTeX entries, all 40 cited; no undefined citations or references. The build log has 0 overfull and 0 underfull boxes (subject to the warning-suppression caveat in P2).
- **Float proximity:** `float_proximity_audit.py` reports 29 floats, 29 within two pages, 0 bad.
- **No stale pre-rebuild numbers** (0.9886, 0.4822, 137 ms, ECE 0.0115) found in the manuscript.
- **No many tiny paragraphs:** after excluding headings and table rows, no body paragraph in Results is under 40 words; the 30 paragraphs are mostly 60–200 words. The long conclusion paragraph is a style issue, not a tiny-paragraph problem.

---

## Exact command outputs

```text
$ python src/models/rebuild_number_audit.py
{
  "n_checks": 116,
  "n_match": 116,
  "n_mismatch": 0
}
EXIT=0

$ python src/models/review_number_audit.py
{
  "n_checks": 298,
  "n_match": 298,
  "n_mismatch": 0
}
EXIT=0

$ python src/models/p1_number_audit.py
{
  "n_checks": 98,
  "n_match": 98,
  "n_mismatch": 0
}
EXIT=0

$ python src/models/float_proximity_audit.py
{
  "n_floats": 29,
  "n_ok": 29,
  "n_bad": 0,
  "bad": []
}
EXIT=0

$ python -m pytest tests -q
252 passed, 12 warnings in 67.93s (0:01:07)
EXIT=0
```

---

## What I could not verify

- I did not re-run any training, feature extraction, judge calls, or conformal fits. The audit scripts and saved artifacts are the evidence base.
- I did not independently re-label the 26-case or 250-row annotation packages; the counts and provenance were checked, not the substantive labels.
- I did not visually inspect every figure at 100% PDF zoom; the float-proximity audit and captions were checked, but not pixel-level legibility.
- The human review is incomplete. Any statement about human accuracy, kappa, or inter-annotator agreement remains unverified and must not be added to the paper until the three-author package is finished.

---

## Suggested fix order

1. Fix the three P0 annotation-ingest blockers (`author_corrected`, merged-file path, three-rater completeness gate) and add an integration test that runs the documented workflow end to end.
2. Fix the P1 model-number misattributions: P2/B2 recall, `0.44`, dialogue/summarization collapse, tuned/fixed-hp labels, and the 0.455/0.477/0.315 neural numbers.
3. Fix the P1 fairness issues: external ensemble training split, Table 17 length-only target fit, and the P1 zero-flag deployment statement.
4. Fix the P1 multicalibration Brier sentence and prompt AUROC range.
5. Add the required three-author wording and the AI-annotator manifest block.
6. Work through the P2 caption/number/limitation list before the final build.

---

## What passes (short list)

- The three number audits pass: 116/116, 298/298, 98/98; `float_proximity` 29/29; `pytest` 252 passed.
- HaluEval splits are grouped and disjoint; RAGTruth QA test rows are excluded from training; the five external corpora are not used for fitting or threshold selection.
- The conformal disjoint split is real (2,514 fit / 2,520 quantile); FaithBench is never used for fitting.
- Seeds 42/123/456, n, prevalence, bootstrap units (item groups and source groups), McNemar/Holm and Wilcoxon are correctly reported; no unsupported "significant" claim.
- The paper does not claim a general-purpose detector, human accuracy, positive held-out QA, or end-to-end retrieval; the external baseline is labeled a reimplementation.
- The annotation package has 250 rows, 60 blind items (20 per sheet), all `needs_author_verification`, blank human columns, and matching SHA-256 hashes; the Fleiss formula and merge command are correct.
- The build is 28 pages with 18 tables, 11 figures, 40 references, a 292-word abstract, six Results subsections, 0 undefined references, 0 overfull/underfull boxes, and 29/29 floats within two pages.
- Most Table 5–18 numbers trace cleanly to `artifacts/results/rebuild/p0`, `p1`, `b6`, `b2`, `calibration`, `intervals`, `judge` and `fixed_hp` artifacts.
