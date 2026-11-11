# Author-proxy review (draft for the corresponding author)

**Status:** AI-generated author-proxy review. This is not a human signature. The
corresponding author must read it, correct it, and sign the final version.

**Recommendation:** minor revision before submission, conditional on the
three-author review being completed and the kappa reported.

## Summary

The rebuild is a substantially stronger paper than the pre-P0/P1 draft. It no
longer rests on a single in-domain benchmark and a judge comparison at natural
prevalence. It now reports a fixed-hyperparameter feature ablation, cross-seed
SHAP stability, an independent medical corpus, a non-overlapping HaluBench
subset, a task-shift test, a retrieval stress test, split and class-conditional
conformal prediction, multicalibration, length-matched judges, prompt
sensitivity, and a refit external baseline. The tone is appropriately
self-critical: the in-domain F1 is saturated, length and lexical overlap carry
most of it, held-out RAGTruth QA is near chance, and the model is explicitly not
a general-purpose detector.

## Strengths

1. **The evidence now separates ranking from calibration.** MedHallu is the
   clearest example: pair accuracy 0.867 and AUROC 0.821, but 100% flag rate and
   ECE 0.493. This is a real result, not a benchmark score.
2. **The external validity is honest.** HaluBench is 0.600 overall, with covidQA
   0.833 but DROP 0.513 and FinanceBench 0.527. The paper does not hide the
   weak sources.
3. **The conformal analysis is methodologically careful.** The in-sample run is
   labeled a construction check, the disjoint split is the primary result, and
   positive coverage is reported beside marginal coverage.
4. **The judge comparison is no longer one-sided.** Natural QA favors the
   judges; the length-matched HaluEval set is competitive for EC-XGB. The paired
   interval is reported and the conclusion is limited.
5. **The ablation is now on the deployed fixed-hp m3 model**, not a different B2
   checkpoint, and the seed variability is reported.
6. **The reproducibility package is strong.** Five audit commands, 512 checks,
   split manifests, annotation manifest with hashes, and a runnable script for
   every P0/P1 experiment.

## Weaknesses

1. **The human review is not complete.** The paper currently reports only AI
   pre-annotation. This is the largest remaining gap. A reviewer will ask for the
   three-author labels, the 60-item overlap, and Fleiss' kappa.
2. **HaluEval remains a synthetic primary corpus.** The paper is honest about
   the length artifact, but the in-domain result is still saturated and
   benchmark-specific.
3. **There is no positive held-out QA result.** RAGTruth QA AUROC is 0.558,
   HaluBench 0.600, and the task shift collapses. The paper's contribution is
   therefore a protocol and a negative/boundary result, not a detector that
   generalizes.
4. **The deployed calibrator was selected on the test split.** The paper
   discloses this, and the clean Table 8 analysis is fixed-hp, but the title
   still contains "calibrated." The "Under Domain and Task Shift" qualifier
   helps; the camera-ready should either refit the deployed calibrator on
   validation/calibration rows or state the caveat in the abstract.
5. **The paper is long.** 29 pages, 18 tables, and 11 figures are a lot for a
   journal. The content is substantive, but a venue with a strict page limit
   will require moving some P0/P1 tables to a supplement.
6. **The in-domain model is dominantly length and overlap.** The paper says so;
   a reviewer may still ask whether the remaining feature groups are worth their
   inference cost.

## Required actions before submission

- [ ] Complete the 250-row author review and the 60-item blind overlap.
- [ ] Report Fleiss' kappa per sheet and the AI-human correction rate.
- [ ] Replace "AI pre-annotation" with the authors' final wording only after the
      authors have verified the labels.
- [ ] Refit or relabel the deployed calibrator; keep the clean Table 8 analysis
      as the primary calibration evidence.
- [ ] Decide whether the paper fits the venue page limit; if not, move the
      external-baseline, HaluBench, and task-shift tables to a supplement.
- [ ] Rebuild and rerun the five audits after the final text changes.
- [ ] Read the PDF at 100% and check the UI figure and the two architecture
      diagrams for legibility.

## Author-proxy sign-off

I would not submit this as a general-purpose hallucination detector. I would
submit it as a carefully measured, domain-specific risk model with an unusually
honest external-validity and calibration analysis. The contribution is the
protocol and the boundary result, not the in-domain F1.

**Draft prepared by:** AI author proxy, 2026-11-11.  
**Human author sign-off:** ______________________  Date: __________
