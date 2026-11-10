# HaluRISC Rebuild Plan (branch `rebuild`)

Status: active. Branch cut from `main` at `7106d72` (reframe commit `c4f48cc` plus the web agent-rules commit).
Goal: close the reviewer-visible weaknesses of the reframed paper without changing its thesis. The paper remains a protocol and negative-result study; it does not claim general hallucination detection or a positive held-out QA result.

## Target claims

The rebuilt manuscript may claim only these:

1. EC-XGB matches a saturated HaluEval benchmark; lexical overlap and length carry most of the in-domain signal.
2. Target calibration lowers ECE with confidence intervals, but a prevalence-only predictor matches it on Brier; the calibration slope quantifies overconfidence.
3. Seen-task adaptation is real on RAGTruth summarization and data-to-text.
4. Held-out RAGTruth QA remains near chance (AUROC ~0.55, mean delta +0.0026, within seed noise).
5. Claim-level features support per-claim explanations, not an aggregate transfer gain.
6. The explanation audit is a stress test; a majority of error-enriched cases are judged implausible.
7. The judge is reported at natural prevalence, and the length-only baseline is reported.
8. The feature model is cheap and API-free, but it runs three small pretrained models; encoder inference is timed.

Not claimed: general transfer, a positive QA result, retrieval quality, or independent human annotation.

## Todo list

- [ ] WP1 Fixed-hyperparameter ablation
- [ ] WP2 Calibration rebuild
- [ ] WP3 Per-seed and hierarchical intervals
- [ ] WP4 EC-XGB explanations
- [ ] WP5 Author taxonomy and audit
- [ ] WP6 Judge and encoder at natural prevalence
- [ ] WP7 External bar
- [ ] WP8 Paper integration
- [ ] WP9 Reviews and submission checks

## Artifact and branch policy

- [ ] Keep `main` frozen at the reframe commit; all rebuild work lands on `rebuild`.
- [ ] Write every new artifact under `artifacts/results/rebuild/`; never overwrite `b2`, `b4`, `b6`, or `review`.
- [ ] Freeze `artifacts/results/rebuild/rebuild_manifest.json` with config, seeds, package versions, and SHA-256 hashes.
- [ ] Merge to `main` only after the verification gates and two reviewer passes.
- [ ] No fabricated numbers; every manuscript value maps to a file in the manifest.

## Weakness closure matrix

| # | Current weakness | Fix | WP | Gate |
|---|---|---|---|---|
| 1 | Calibrator selected on test ECE; mixed Platt links | Fit Platt on validation only; one link; report slope/intercept and selection rule | WP2 | Test ECE never used for selection |
| 2 | Prior-only uses test prevalence; 50-row ECE anomaly | Report prior at both prevalences; repeated label-budget draws with mean +/- SD | WP2 | 100 draws, monotone or explained |
| 3 | Per-variant tuning confound | Fixed-hyperparameter m0-m3, 3 seeds | WP1 | Same config across variants |
| 4 | Claim features add nothing to transfer | Report m1 vs m2 explicitly; reframe contribution | WP1, WP8 | m1/m2 delta table |
| 5 | Single-seed intervals | Two-level bootstrap: groups within seed, then seeds | WP3 | 2,000 draws, per-seed deltas |
| 6 | F1 at 0.5 compares flag rates | All-positive and length-only rows; recall/FPR; common 5% FPR point | WP3, WP8 | Operating-point table |
| 7 | SHAP/audit on B2, not m3 | Rerun TreeSHAP, perturbation, and audit export on m3 | WP4 | m3 explainer + new 26 cases |
| 8 | Audit is majority-implausible; no 2x2/kappa | Add 2x2 table, Cohen's kappa, rubric, sampling frame | WP5 | Author-reviewed labels |
| 9 | Error taxonomy auto-tagged | Manual re-annotation with codebook; two annotators or author review | WP5 | Completed 20-case sheet |
| 10 | Judge balanced-only; no prompt | RAGTruth QA at 17.8%, 3 repeats, published prompt, agreement | WP6 | Natural-prevalence table |
| 11 | Encoder comparison selective; no timing | Same-recipe adapted encoder, CIs, time RoBERTa on 200 rows | WP6 | Timing + CI |
| 12 | PR-AUC/MCC missing | Add both columns to the model table; discuss the QA sign | WP8 | Table + text |
| 13 | Model map/captions ambiguous | Descriptive labels; B2/m0/m3 in every caption | WP8 | No "conventional" alone |
| 14 | Float/page/text load | Keep 16 tables/11 figures or trim; no pure float pages | WP8 | Float check |
| 15 | No external bar | 2-4 published HaluEval QA numbers with comparability notes | WP7 | Literature table |
| 16 | Deployment/retrieval claims | Limitations paragraph; no retrieval claim | WP8 | Explicit scope |

## WP1 - Fixed-hyperparameter ablation

- [ ] Add a `--fixed-params` mode to `src/models/run_b6_modified.py`.
- [ ] Use the m3 seed-42 configuration for all variants: depth 5, learning rate 0.05, 300 trees, subsample 0.7, colsample 0.7.
- [ ] Train 4 variants x 3 seeds; regenerate predictions, external metrics, and confusion matrices.
- [ ] Regenerate the component-ablation table and figure.
- [ ] Output `artifacts/results/rebuild/fixed_hp_ablation.csv` and `fixed_hp_predictions.parquet`.
- [ ] Acceptance: identical config logged for every run; m1/m2/m3 deltas reported.
- Estimate: 0.1 GPU-h, 0.3 CPU-h, 0.5 agent-day, 2 human-hours.

## WP2 - Calibration rebuild

- [ ] Add `src/models/rebuild_calibration.py`.
- [ ] Platt only, fit on the 5,034-row validation split; do not select by test ECE.
- [ ] Report calibration slope and intercept with bootstrap CIs.
- [ ] Report prior-only at 0.3107 and 0.1778.
- [ ] Repeated label-budget draws: 100 draws at 50, 100, 252, 500, 1,000, 5,034 rows.
- [ ] ECE with row- and group-bootstrap CIs; Brier skill score.
- [ ] Regenerate calibration JSON, table, reliability and operating-point figures, and the UI screenshot.
- [ ] Acceptance: test rows never used for fit or selection; slope and CIs reported.
- Estimate: 0 GPU, 1-2 CPU-h, 1 agent-day, 2 human-hours.

## WP3 - Per-seed and hierarchical intervals

- [ ] Recompute every headline delta with a two-level bootstrap, 2,000 draws.
- [ ] Report per-seed F1/AUROC/ECE for RAGTruth QA, summarization, data-to-text, FaithBench, and in-domain.
- [ ] Replace the seed-42-only cross-domain table with a three-seed mean plus per-seed range.
- [ ] Include the primary EC-XGB vs conventional comparison in the Holm family.
- [ ] Acceptance: no headline claim rests on one seed.
- Estimate: 0 GPU, 2-3 CPU-h, 1 agent-day, 2 human-hours.

## WP4 - EC-XGB explanations

- [ ] Repoint `run_b5_explanation_reliability.py` from B2 + source Platt to m3 + the display calibrator.
- [ ] Run TreeSHAP, permutation importance, top-k neutralization, and 1,000-resample stability.
- [ ] Rerun the six perturbations with full feature re-extraction on 120-200 rows.
- [ ] Export a fresh 26-case audit sheet from m3.
- [ ] Acceptance: every explanation number refers to m3.
- Estimate: 0.3-0.5 GPU-h, 0.5-1 CPU-h, 1 agent-day, 1 human-hour.

## WP5 - Author taxonomy and audit

- [ ] Replace `tag_case` heuristics with a 20-error codebook.
- [ ] Re-annotate the 26 explanation cases on EC-XGB explanations, two passes.
- [ ] Record plausible/implausible/unsure, a 2x2 table, and Cohen's kappa.
- [ ] State "author review" or "AI-assisted author review"; never "independent human annotation".
- [ ] Acceptance: completed sheets and codebook committed.
- Estimate: 0 GPU, 5-7 human-hours.

## WP6 - Judge and encoder at natural prevalence

- [ ] Publish the system and user prompt, model version, decoding, and token counts.
- [ ] Run GPT on 900 RAGTruth QA rows, 3 repeats; report F1/P/R with CIs, run-to-run agreement, cost, latency.
- [ ] Run Qwen 3.5 4B local 4-bit on the same rows.
- [ ] Add the length-only row and the EC-XGB CI.
- [ ] Time RoBERTa-base on the same 200 rows.
- [ ] Acceptance: natural-prevalence table; prompt in supplement; encoder timing reported.
- Estimate: 2-2.5 GPU-h, $0.3-1 API, 1 agent-day, 2 human-hours.

## WP7 - External bar

- [ ] Find 2-4 published HaluEval QA detector numbers.
- [ ] Record split, metric, and label protocol for each.
- [ ] If nothing is comparable, state that and cite the benchmark baselines.
- [ ] Acceptance: a comparability note, not a misleading table.
- Estimate: 0 compute, 4-6 agent-hours, 2 human-hours.

## WP8 - Paper integration

- [ ] Rewrite the abstract in the six-question order, 250-300 words, rebuilt numbers.
- [ ] Keep the title `Calibrated Hallucination Risk Estimation in Large Language Models Using Evidence-Consistent XGBoost`.
- [ ] Update the results subsections with WP1-WP6 outputs.
- [ ] Add prior-only, slope, PR-AUC, and MCC; remove duplicate tables.
- [ ] Keep 9-11 figures, each within two pages of first mention; no pure float page.
- [ ] Add the length -> overlap -> shift -> chance discussion paragraph.
- [ ] Acceptance: every number matches a frozen artifact.
- Estimate: 0 compute, 1 agent-day, 3-4 human-hours.

## WP9 - Reviews and submission checks

- [ ] Run two agent reviewers: methods/statistics and editor/domain.
- [ ] Apply only confirmed issues; rebuild and recheck.
- [ ] Run Turnitin and an AI detector; record the reports.
- [ ] Check Overleaf 2026 pdfLaTeX parity.
- [ ] Acceptance: at most minor revision from both reviewers; clean build.
- Estimate: 0 compute, 0.5 agent-day, 2 human-hours.

## Verification gates

1. [ ] Artifact gate: all rebuilt metrics in `artifacts/results/rebuild/`; manifest with hashes.
2. [ ] Audit gate: every manuscript number maps to an artifact; update the audit count.
3. [ ] Build gate: TeX Live 2026, 0 overfull, 0 underfull, 0 undefined references, 40 citations.
4. [ ] Float gate: every table/figure within two pages of first mention; no float after references; no near-empty float page.
5. [ ] Reviewer gate: two independent reviews; target at most minor revision.
6. [ ] Detector/Overleaf gate: Turnitin and AI-detector report; Overleaf 2026 parity.

## Schedule

| Day | Agent lanes | Human |
|---|---|---|
| 1 | WP1 code/run; WP2 code/run; WP3 bootstrap | Approve codebook; review WP1/WP2 outputs |
| 2 | WP4 m3 SHAP; Qwen judge overnight; WP6 GPT | Label 26 + 20 cases (5-7 h) |
| 3 | WP7 external bar; WP5 integration; WP8 paper | Check tables/claims; approve numbers |
| 4 | Build, audit, float check, two reviewers | Final read; detector/Overleaf |
| 5 (buffer) | Reviewer fixes | Sign-off |

Compute: 3.5-4.5 GPU-h, 6-9 CPU-h, $0.3-1 API, 20-25 human-hours.
Laptop: RTX 3060 6 GB, RoBERTa batch 8 fp16, Qwen 4-bit batch 1, serialize GPU jobs.

## Decision points

- After WP1: if fixed-hp ablation makes m1/m2/m3 indistinguishable, keep the negative result. Do not force a claim-feature gain.
- After WP2: if the calibration slope is severely overconfident and prior-only wins on Brier, report it. The contribution is the protocol, not the calibrator.
- After WP5: if the author audit finds more than half implausible, scope "explainable" to "diagnostic".
- After WP9: if reviewers still require major revision, submit the frozen reframe to an applied journal and keep the rebuild for a transactions submission.

## Out of scope

- A positive held-out QA result. Mean delta is +0.0026; seed 123 is -0.0673.
- Independent human annotation unless a second human is available.
- Retrieval or end-to-end evaluation without qrels and 300-500 new labels.
