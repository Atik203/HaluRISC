# Research Paper Review and Revision Checklist

**Manuscript:** *Calibrated Hallucination Risk Estimation in Large Language Models Using Evidence-Consistent XGBoost*  
**Reviewed file:** `ML_Paper.pdf` (17 pages)  
**Review date:** 2026-10-20  
**Recommendation:** **Major revision before submission**  
**Status:** P1 complete (17/17) and P2 complete (20/20) on 2026-10-20, with the G4 disagreement review AI-conducted at author direction in place of a human pass. J1-J5 and K1 verified. The previous checklist is archived at `docs/archive/2026-09-17_Review_Checklist.md`.

## Second external review (2026-10-21)

- [x] **R1. Calibration number provenance.** §5.3 now states that `0.726 → 0.126` is the seed-42 deployed fit and that Table 8 reports three-seed means (`0.7433 → 0.1263`).
- [x] **R2. Length shortcut, stronger controls.** Full-test-set exact matching (163 pairs), 200 repeated draws (EC-XGB 0.9608 ± 0.0022 vs length-only 0.5257 ± 0.0034), and a length-conditional AUROC over 12 strata (0.9773 ± 0.0009). `length_matched_repeat.json`.
- [x] **R3. Generalization framing.** Discussion names EC-XGB a domain-specific estimator, not a general-purpose detector, and reads the held-out QA AUROC of 0.549 as near chance.
- [x] **R4. Protocol separation.** Table `tab:protocols` lists P0/P1/P2 against calibrator fit, threshold fit, and zero-shot status in one place.

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

- [x] **A1. Define the three protocols.** Write source-only, target-calibrated, and target-calibrated-plus-target-thresholded as named protocols in §3.3–§3.4, and use the same names in Algorithm 2, §4.4, §5.2–§5.3, the abstract, and the conclusion.
- [x] **A2. Build one combined table.** Standard XGBoost versus EC-XGB on the same 900-row RAGTruth QA test set, with raw, source-calibrated, and target-calibrated ECE, Brier score, NLL, and predicted-positive rate.
- [x] **A3. Label every calibration value.** Record checkpoint, calibrator, fit split, test split, and seed aggregation. Explain why 0.7433 and 0.726 differ, and why 0.126 differs from 0.1335 and 0.1316.
- [x] **A4. State the label rules.** Confirm that no final-test label enters calibration fitting or threshold selection, and disclose that target calibration needs labeled target rows.
- [x] **A5. Separate operating points.** Report source-only and target-adapted thresholds separately, with 95% CIs for recall and FPR. Explain why the test FPR is 5.8% when the target is 5%.
- [x] **A6. Align the abstract and conclusion.** Remove any wording that conflicts with the protocol definitions or implies the target-tuned result is source-only.

### B. Length-shortcut control

- [x] **B1. Report per-bin label counts.** In the 247×6 subset, give n, positives, negatives, positive rate, mean length, and class-length SMD for every bin.
- [x] **B2. Evaluate the length-only baseline on exactly the same rows.** Add ΔF1 and ΔAUROC versus EC-XGB with group-bootstrap 95% CIs.
- [x] **B3. Build the matched subset.** Match positives to negatives on exact answer length (widen to a one-word caliper only if needed), equal counts per bin, and report n, prevalence, and SMD.
- [x] **B4. Rename the current test.** Call it a length-bin reweighting control, not a neutralization or proof of shortcut robustness. State what it does and does not show.
- [x] **B5. Add the external or limitation check.** Use RAGTruth QA and FaithBench first. Add one free, license-clean dataset with a different length–label relation only if required; otherwise declare the limitation.

### C. Zero-shot and exposure

- [x] **C1. Define zero-shot.** Specify no target examples, groups, tasks, or sources for training, calibration, or threshold selection.
- [x] **C2. Separate classifier from display score.** The held-out QA classifier is zero-shot. The QA display score is target-calibrated and is not zero-shot.
- [x] **C3. Label the other results.** Combined RAGTruth is seen-task adaptation. Name the calibrator used for FaithBench and define "unseen" on the relevant axis.

### D. Number and artifact audit

- [x] **D1. Run the reproduction check.** Run `src/models/verify_artifacts.py` and save the log under `artifacts/results/review/`.
- [x] **D2. Build and trace the number audit.** Map each value in Tables 4-14 and every headline number to an artifact file and key, write `number_audit.csv`, and mark it match, explain, or fix.
- [x] **D3. Verify the support files.** Check split hashes, seeds, dependency pins, repository claims, and data-license notes.

---

## P2 — High: strongly recommended

### E. Neural baseline

- [x] **E1. Train one fair neural baseline.** RoBERTa-base at 256 tokens on the 6 GB card (local-hardware path chosen over DeBERTa-v3-base/512 on 16 GB), with the same grouped splits and seeds 42/123/456.
- [x] **E2. Evaluate and calibrate it.** F1 0.9912, AUROC 0.9990, PR-AUC 0.9993, MCC 0.9825, ECE 0.0070, Brier 0.0081 in-domain; RAGTruth/FaithBench externals in `neural_baseline_summary.json`. Platt on validation only.
- [x] **E3. Add it honestly.** Table 4 row plus §5.1 paragraph and conclusion sentence: encoder wins in-domain (0.9912 vs 0.9846) but ranks worse outside it (RAGTruth AUROC 0.337 vs 0.582); training cost ~15 min/seed on RTX 3060 6 GB in §4.3.

### F. Novelty and ablations

- [x] **F1. Reposition the contribution.** §3.3 now presents a reproducible method and evaluation protocol, not a fundamentally new learning algorithm, with protocol-scoped baseline comparison.
- [x] **F2. Ablate monotonic constraints separately.** `monotone_ablation.json` (3 seeds): F1 unchanged, ECE -0.0009, RAGTruth AUROC +0.0043; §5.2 sentence added.
- [x] **F3. Explain the claim aggregates.** §5.2: m2 adds the eight claim-level aggregates beyond the six sentence-level NLI probabilities (+0.0005 F1 over m1, within undistinguished seed-level variation).
- [x] **F4. Compare the closest baselines.** §3.3 contrasts Sundaragiri et al., Valentin et al., and SpikeScore on protocol and reported metric families, scoped to protocol not numbers.

### G. LLM judge

- [x] **G1. Document the 200 samples.** §5.7: stratified subsample of the 3,000-row headline test set, 100 positives + 100 negatives without replacement at sampling seed 42, prevalence exactly 50%, full overlap; 95% CIs in prose and `llm_judge_*_ec_xgb.json`.
- [x] **G2. Add the length-only baseline on the same 200 rows.** §5.7 prose: F1 0.905 on the same rows (`judge_rows_comparison.json`), so the judge comparison is scoped, not semantic beyond length.
- [x] **G3. Document the judge call.** Model identifiers, default decoding (caption), scoring rule and token counts in `llm_judge_gpt_ec_xgb.json` + per-sample CSV + log; cost from recorded tokens.
- [x] **G4. Add the local Qwen 3.5 4B judge.** Qwen F1 + GPT agreement in Table 12/prose, labeled second judge not ground truth; author-directed AI review of all 40 Qwen errors in place of the human pass (`qwen_disagreement_review.json`).
- [x] **G5. Narrow the claim.** "The claim is scoped to this sample and these prompts" in §5.7; no second GPT prompt (budget).

### H. Cost and latency

- [x] **H1. Add the per-baseline table.** Table `tab:cost`: inputs, size, model-only p50, cost/1K for 11 baselines (`baseline_latency.json`); RoBERTa training-only, NLI-only via feature step (caption).
- [x] **H2. Measure the full pipeline.** `pipeline_latency.json`: predict p50 42.8 ms, analyze 57.1 ms, verify 143.6 ms, verify+judge 502.5 ms; §5.7 sentence added.
- [x] **H3. Separate the two latency claims.** §5.7: component table excludes retrieval/judge/loading; HTTP verify includes retrieval, not the judge; 61.8 ms is classifier-only pipeline.
- [x] **H4. State the cost assumptions.** §5.7: local = amortized hardware + electricity order-of-magnitude in USD/1K; GPT from token counts at $0.20/MTok in + $1.20/MTok out. No "actionable" without policy (score labeled risk estimate).

---

## P3 — Rigor and clarity

### I. SHAP explanation reliability

- [ ] **I1. Separate stability from faithfulness.** Document the top-k neutralization procedure, baseline value, k, sample size, and the metric.
- [ ] **I2. State the limits.** Report the small perturbation groups (dates n = 13, clause shuffles n = 14) and note that a stable top feature can be a shortcut such as answer length.

### J. Tables, figures, abstract, and conclusion

- [x] **J1. Add sample sizes and prevalence** to the key evaluation tables, and confidence intervals for the central cross-domain results. (n in Table 4/shift/target/length captions; 17.8% QA prevalence; bootstrap + Wilson CIs in §§5.1/5.2/5.8.)
- [x] **J2. Label calibration plots** by model and split (raw, source, target). (Figs. reliability + calibshift captions name all three corpora and both score types.)
- [x] **J3. Rewrite the abstract.** One paragraph, 280 words: benchmark + combined/held-out results, display calibrator with target-label requirement, no length-proof or judge-superiority claim, "risk estimation" framing.
- [x] **J4. Rewrite the conclusion.** Held-out QA F1 ≈ 0.30, AUROC 0.582 with weak-discrimination framing, calibration-vs-ranking caveat, shortcut-controlled future work, lightweight-risk-estimator framing with cost.
- [x] **J5. Check the venue limit and rebuild.** 17 pages, zero warnings/errors, 35 references cited, number audit 188/188 on 2026-10-20 (re-verified after H1/G1/G4 edits).

---

## P4 — Final read

- [x] **K1. Final language and format pass.** Style scan (no banned words/pronouns/semicolons/em-dashes in new prose), all tables render with verified rows, 35/35 references cited, 17 pages zero warnings.

---

## Done log

Check a box above only when the manuscript and artifact are committed, then add a row here.

| Date | Item | Evidence (artifact / commit) |
|---|---|---|
| 2026-10-20 | A2-A6 calibration protocol and combined table | `c4ace66`; `artifacts/results/review/calibration_protocol.json` |
| 2026-10-20 | B1-B5 length control and matched subset | `c4ace66`; `artifacts/results/review/length_control.json` |
| 2026-10-20 | C1-C3 zero-shot wording | `c4ace66`, `b1adc9b` |
| 2026-10-20 | D1 artifact verification | `8a4d238`; `artifacts/results/review/verify_artifacts.log` |
| 2026-10-20 | A1 protocol labels in 4.4/5.2/abstract/conclusion | `f03b5a9` |
| 2026-10-20 | D2 number audit (183/183 match) | `f03b5a9`; `artifacts/results/review/number_audit.json` |
| 2026-10-20 | D3 support-file audit (35 match, 2 explained, 0 mismatch) | `f03b5a9`; `artifacts/results/review/support_files_audit.json` |
| 2026-10-20 | E1-E3 neural baseline (RoBERTa-base 256, 3 seeds) + Table 4 row + §5.1/conclusion | `artifacts/results/review/neural_baseline/`; `judge_rows_comparison.json` |
| 2026-10-20 | F1 novelty framing + F4 closest-baseline comparison (§3.3) | manuscript + `ref.bib` (`liu2019roberta`) |
| 2026-10-20 | F2 monotone ablation (F1 unchanged, ECE -0.0009, RAGTruth AUROC +0.0043) | `artifacts/results/review/monotone_ablation.json` |
| 2026-10-20 | G2-G3/G5 judge scope (length-only 0.905, call docs, narrowed claim) + H2 pipeline latency | `llm_judge_gpt_ec_xgb.json`; `pipeline_latency.json` |
| 2026-10-20 | J5 17-page zero-warning rebuild + number audit 188/188 + Table 13 folded to prose | `Journal_Paper/out/halurisc.pdf`; `number_audit.json` |
| 2026-10-20 | F3 claim-aggregate sentence (+0.0005 F1 over m1) + truncation rate 14/3,000 | `neural_baseline_truncation.json` |
| 2026-10-20 | G1 judge sample docs + H1/H3/H4 cost-latency | `baseline_latency.json`; 17pp zero-warning rebuild |
| 2026-10-20 | G4 full 40-error review in place of human pass (20 strict-label, 6 ambiguous, 12 GPT-caught misses, 2 false alarms) | `qwen_disagreement_review.json` |
| 2026-10-21 | R1-R4 second review response (calibration provenance, repeated length matching, generalization wording, P0/P1/P2 table) | `length_matched_repeat.json`; 18pp zero-warning build; audit 200/200 |
|  |  |  |
