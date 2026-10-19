# Research Paper Review and Revision Checklist

**Manuscript:** *Calibrated Hallucination Risk Estimation in Large Language Models Using Evidence-Consistent XGBoost*  
**Reviewed file:** `ML_Paper.pdf` (17 pages)  
**Review date:** 2026-10-20  
**Recommendation:** **Major revision before submission**  
**Status:** open items only. The previous checklist is archived at `docs/archive/2026-09-17_Review_Checklist.md`.

> **Scope:** the reviewer evaluated the manuscript text, equations, tables, figures, and stated protocols. The source code and experiments were not independently rerun. Numbers below are manuscript claims and must be checked against the artifacts before they are treated as verified.
>
> **How to use:** check a box only when the fix is in `Journal_Paper/halurisc.tex` and the backing artifact is committed. Record the date, item, and commit hash in the done log at the end. Workstreams A–J match `roadmap.md` §18.3.

---

## Reviewer verdict in six lines

1. The length-balanced subset does not prove robustness to the answer-length shortcut.
2. Calibration and threshold-selection protocols are inconsistent across sections.
3. Calibration values are not labeled by model, split, and protocol in one place.
4. HaluEval's length–label correlation limits what F1 = 0.9855 demonstrates.
5. The LLM-judge comparison is too narrow for a general superiority claim.
6. Novelty, latency, cost, and zero-shot claims need tighter definitions.

**Already acknowledged, no action:** classifier-versus-application separation, combined RAGTruth versus held-out QA distinction, length-only baseline, label-noise and transfer caveats, and the calibration-versus-ranking note. The reviewer treats these as improvements, not open items.

**Critical path:** A1–A3, B1–B4, C1–C3, D1–D2. These close the five Priority 1 items. Do not start with the neural baseline.

---

## P1 — Critical: close before submission

### A. Calibration and threshold protocol

- [ ] **A1. Define the three protocols.** Write source-only, target-calibrated, and target-calibrated-plus-target-thresholded as named protocols in §3.3–§3.4, and use the same names in Algorithm 2, §4.4, §5.2–§5.3, the abstract, and the conclusion.
- [ ] **A2. Build one combined table.** Standard XGBoost versus EC-XGB on the same 900-row RAGTruth QA test set, with raw, source-calibrated, and target-calibrated ECE, Brier score, NLL, and predicted-positive rate.
- [ ] **A3. Label every calibration value.** Record checkpoint, calibrator, fit split, test split, and seed aggregation. Explain why 0.7433 and 0.726 differ, and why 0.126 differs from 0.1335 and 0.1316.
- [ ] **A4. State the label rules.** Confirm that no final-test label enters calibration fitting or threshold selection, and disclose that target calibration needs labeled target rows.
- [ ] **A5. Separate operating points.** Report source-only and target-adapted thresholds separately, with 95% CIs for recall and FPR. Explain why the test FPR is 5.8% when the target is 5%.
- [ ] **A6. Align the abstract and conclusion.** Remove any wording that conflicts with the protocol definitions or implies the target-tuned result is source-only.

### B. Length-shortcut control

- [ ] **B1. Report per-bin label counts.** In the 247×6 subset, give n, positives, negatives, positive rate, mean length, and class-length SMD for every bin.
- [ ] **B2. Evaluate the length-only baseline on exactly the same rows.** Add ΔF1 and ΔAUROC versus EC-XGB with group-bootstrap 95% CIs.
- [ ] **B3. Build the matched subset.** Match positives to negatives on exact answer length (widen to a one-word caliper only if needed), equal counts per bin, and report n, prevalence, and SMD.
- [ ] **B4. Rename the current test.** Call it a length-bin reweighting control, not a neutralization or proof of shortcut robustness. State what it does and does not show.
- [ ] **B5. Add the external or limitation check.** Use RAGTruth QA and FaithBench first. Add one free, license-clean dataset with a different length–label relation only if required; otherwise declare the limitation.

### C. Zero-shot and exposure

- [ ] **C1. Define zero-shot.** Specify no target examples, groups, tasks, or sources for training, calibration, or threshold selection.
- [ ] **C2. Separate classifier from display score.** The held-out QA classifier is zero-shot. The QA display score is target-calibrated and is not zero-shot.
- [ ] **C3. Label the other results.** Combined RAGTruth is seen-task adaptation. Name the calibrator used for FaithBench and define "unseen" on the relevant axis.

### D. Number and artifact audit

- [ ] **D1. Run the reproduction check.** Run `src/models/verify_artifacts.py`, save the log under `artifacts/results/review/`, and build `number_audit.csv`.
- [ ] **D2. Trace every number.** Map each value in Tables 4–14 and every headline number to an artifact file and key; mark it match, explain, or fix.
- [ ] **D3. Verify the support files.** Check split hashes, seeds, dependency pins, repository claims, and data-license notes.

---

## P2 — High: strongly recommended

### E. Neural baseline

- [ ] **E1. Train one fair neural baseline.** DeBERTa-v3-base at 512 tokens on the 16 GB GPU, or RoBERTa-base at 256 tokens on the 6 GB card, with the same grouped splits and seeds 42/123/456.
- [ ] **E2. Evaluate and calibrate it.** Report F1, AUROC, PR-AUC, MCC, ECE, Brier, and NLL in-domain and on RAGTruth and FaithBench. Fit Platt on validation only.
- [ ] **E3. Add it honestly.** Report latency and cost, add one table row, and state whether it beats EC-XGB. If it does, frame EC-XGB on calibration, cost, and interpretability.

### F. Novelty and ablations

- [ ] **F1. Reposition the contribution.** Present a reproducible method and evaluation protocol, not a fundamentally new learning algorithm.
- [ ] **F2. Ablate monotonic constraints separately.** Train the same feature set with and without the constraints, three seeds, and report F1, AUROC, ECE, and the shifted flag rate.
- [ ] **F3. Explain the claim aggregates.** State what the eight claim-level aggregates add beyond the six sentence-level NLI probabilities, with the delta and its significance.
- [ ] **F4. Compare the closest baselines.** Add a short comparison with Sundaragiri et al., Valentin et al., and SpikeScore on protocol and metrics, not only on reported metric families.

### G. LLM judge

- [ ] **G1. Document the 200 samples.** Give the source pool, sampling seed, class balance, positive prevalence, and overlap with the headline test set. Add 95% CIs.
- [ ] **G2. Add the length-only baseline on the same 200 rows.**
- [ ] **G3. Document the judge call.** Record the prompt, model identifier, temperature, decoding, and scoring rule.
- [ ] **G4. Add the local Qwen 3.5 4B judge.** Use an OpenAI-compatible endpoint, report F1 and agreement with GPT, and human-spot-check 20–40 disagreements. Label it as a second judge, not ground truth.
- [ ] **G5. Narrow the claim.** Limit it to this sample and prompt. Add a second GPT prompt only if the budget allows.

### H. Cost and latency

- [ ] **H1. Add the per-baseline table.** Inputs, model size, hardware, p50/p95 latency, and cost per 1,000 predictions for every baseline.
- [ ] **H2. Measure the full pipeline.** `/verify` end to end, with stages separated: features, NLI, model, SHAP, claim split, retrieval, judge fallback.
- [ ] **H3. Separate the two latency claims.** 61.8 ms is classifier-only. The interactive path is about one second. State whether loading, network, retrieval, and judge calls are excluded.
- [ ] **H4. State the cost assumptions.** Hardware, batch size, token counts, API prices, cache, and units. Do not call probabilities "actionable" without a decision policy.

---

## P3 — Rigor and clarity

### I. SHAP explanation reliability

- [ ] **I1. Separate stability from faithfulness.** Document the top-k neutralization procedure, baseline value, k, sample size, and the metric.
- [ ] **I2. State the limits.** Report the small perturbation groups (dates n = 13, clause shuffles n = 14) and note that a stable top feature can be a shortcut such as answer length.

### J. Tables, figures, abstract, and conclusion

- [ ] **J1. Add sample sizes and prevalence** to the key evaluation tables, and confidence intervals for the central cross-domain results.
- [ ] **J2. Label calibration plots** by model and split (raw, source, target).
- [ ] **J3. Rewrite the abstract.** HaluEval QA benchmark performance, combined versus held-out results, the EC-XGB display calibrator, target-label requirement, no length-proof claim, no general judge superiority, and "risk estimation" rather than truth.
- [ ] **J4. Rewrite the conclusion.** Keep F1 ≈ 0.30 on held-out QA, AUROC ≈ 0.58 as weak discrimination, the calibration-versus-ranking caveat, the length limitation, and the lightweight-risk-estimator framing.
- [ ] **J5. Check the venue limit and rebuild.** Confirm 17 or 16 pages, fit the new tables, and rebuild with zero warnings and all references cited.

---

## P4 — Final read

- [ ] **K1. Final language and format pass.** Long sentences, terminology, equations, feature names, figure readability, and the target venue's submission format, after the new tables are in.

---

## Done log

Check a box above only when the manuscript and artifact are committed, then add a row here.

| Date | Item | Evidence (artifact / commit) |
|---|---|---|
|  |  |  |
|  |  |  |
|  |  |  |
