# Response to the independent P0/P1 review

The reviewer report is `docs/rebuild/P0_P1_REVIEWER_REPORT.md`. This file records
what was fixed after that report and what remains. The manuscript was not edited
by the reviewer; all fixes below are in the working tree.

## P0 blockers

| ID | Issue | Status | Fix |
|---|---|---|---|
| P0-1 | `author_corrected` missing from `VERIFIED_STATES` | Fixed | Added `author_corrected` to `src/models/p0_ingest_author_review.py`; added `tests/test_p0_ingest.py`. |
| P0-2 | Ingest reads blank `p0_blind_overlap.csv`, not the merged three-author file | Fixed | The ingest now prefers `p0_blind_overlap_merged.csv`, falls back with a warning, and records `blind_file`. |
| P0-3 | Completeness gate accepts one rater | Fixed | `complete=True` now requires every blind sheet to have three raters and a non-null Fleiss kappa. |

## P1 major issues

| ID | Issue | Status | Fix |
|---|---|---|---|
| P1-1 | P2 5.0% recall attributed to EC-XGB | Fixed | Abstract no longer mixes the m3 ECE with the B2 P2 recall. The conclusion labels the 5.0% as the conventional reference. The P2 paragraph says "conventional target-thresholded mode". |
| P1-2 | Abstract says 0.44 positive coverage | Fixed | Replaced with 0.69, matching the disjoint 0.685 in Table 9 and the conclusion. |
| P1-3 | Dialogue and summarization both "collapse to all-positive" | Fixed | Abstract, introduction, conclusion, and body now separate the dialogue all-positive failure from the summarization below-length-only result. |
| P1-4 | Tuned and fixed-hp checkpoints mixed without labels | Fixed | Abstract and introduction use the fixed-hp m3 0.9854/0.9986 for the external-validity headline. The conclusion states both the fixed-hp and tuned deployed values. |
| P1-5 | P1 flags 0% at 0.5; API low band is above the max target-Platt score | Fixed | Added the explicit sentence: P1 flags no RAGTruth QA rows at 0.5, the maximum target-Platt score is 0.313, and the API low band starts at 0.35, so the card relies on per-claim adjustment. |
| P1-6 | External baseline trained on 3,000 validation rows vs 24,056 for EC-XGB | Fixed | Refit the same five-feature baseline on the 14,000 training rows. Table 5 now reports F1 0.9517, AUROC 0.9850, ECE 0.0141; the paired deltas are 0.0337 F1 [0.0262, 0.0419] and 0.0135 AUROC [0.0098, 0.0177]. The causal sentence was softened to "consistent with". |
| P1-7 | Table 17 length-only baseline is target-fitted | Fixed | The text now calls it the "target-fitted length-only baseline"; the note already said it was fit on the 5,034 QA calibration rows. |
| P1-8 | Neural per-task AUROC 0.455/0.477 and pooled 0.315 untraceable | Fixed | Recomputed from `neural_baseline_predictions.parquet` joined to `unified_records.parquet`: 0.458 summarization, 0.491 data-to-text, 0.337 pooled seed mean. Text updated; the incorrect "different positive rates" explanation was replaced with the between-task score-shift explanation. |
| P1-9 | "Brier follows the same pattern" is false | Fixed | Text now says Brier does not follow the ECE ranking and the two summaries disagree. |
| P1-10 | Prompt AUROC range 0.920-0.944 vs observed 0.910-0.954 | Fixed | Text uses the all-run range 0.910-0.954. |
| P1-11 | Required three-author wording absent; "second rater" | Fixed | The exact phrase "three-author blinded review with AI pre-annotation" appears in §3.4, §5.5, and §5.6. The "second rater" sentence was replaced with the three-author final-validation statement. |
| P1-12 | Manifest names EC-XGB, not the AI annotator | Fixed | Added an `annotator` block with the agent model, date, and prompt definition; kept the EC-XGB block as the scored model with a note. |
| P1-13 | Deployed calibrator selected on test; title unqualified | Partially fixed | Title now says "Under Domain and Task Shift". The limitations already disclose the test-selected deployed calibrator and the test-clean Table 8 refit. A full test-clean refit of the deployed calibrator is left for the camera-ready. |

## P2 issues fixed

- Figure 5 mean ordering corrected to m3 > m1 > m2 > m0.
- Claim-block difference described as "about the same size" as the seed SD.
- MedHallu pair accuracy labeled as scores averaged over seeds.
- Table 11 now says "three external corpora and the task-shift pair".
- Table 3 note now says "five zero-shot corpora".
- Table 5 caption says deterministic rows are single fits.
- Table 6 caption gives n=3,000 for HaluEval and n=2,700 for RAGTruth.
- Table 10 renamed "External and adaptation performance".
- Table 16 column renamed "p50 (ms)".
- Table 17 caption lists judges, EC-XGB, length-only, and all-positive references.
- Table 18 column renamed "Hallucinated rate".
- "18 to 23 times slower" corrected to 18 to 24.
- "Calibration does not improve" softened to "the calibrated ECEs are not lower in this sample".
- Class-conditional conformal "restores" softened to "raises ... still slightly below nominal".
- Multicalibration bins now give n=154 and n=36.
- "Residual ECE is the prevalence gap" softened to "dominated by".
- "Length and overlap detector" softened to "dominantly".
- MedHallu now says human-annotated gold answers with paired LLM-written hallucinations.
- HaluBench provenance and 50/50-by-construction caveat added.
- "Four external corpora" corrected.
- "Deployment uses gold context" corrected to distinguish the retrieval stress test.
- Judge limitation now includes the length-matched set.
- "two audit scripts" corrected.
- "every aggregate table" corrected to "every seeded aggregate table".
- 20-case "substitution" removed.
- Acknowledgements include MedHallu and HaluBench.
- Data availability says "release archive".
- Conclusion split into two paragraphs.
- Non-standard float-page lengths removed.

## Remaining known issues

- The paper is 29 pages after the 8.5pt reference font and the larger UI figure; page limit is not a constraint for this draft.
- `\sloppy` and `\hbadness=10000` remain before the bibliography; the build reports 0 overfull/underfull, but the suppression is a style choice.
- Keywords remain seven; trim to six only if the venue requires it.
- The three-author review is complete (2026-11-11). AI-author agreement and
  blind kappa are reported; the main-sheet labels originated as AI
  pre-annotations and were author-verified.
