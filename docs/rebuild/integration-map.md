# Rebuild to paper integration map

Status: working document. Every value below must be replaced by a frozen artifact under `artifacts/results/rebuild/` before the final build. No placeholders may survive in the submitted PDF.

## 1. New results already available (WP1, fixed hyperparameters)

Source: `artifacts/results/rebuild/fixed_hp/fixed_hp_ablation.csv` and `fixed_hp_predictions.parquet`.

HaluEval test, mean over seeds 42/123/456, n = 3,000, identical config for all variants:

| Variant | Features | F1 | AUROC | MCC | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|
| m0 | 26 | 0.98488 | 0.99789 | 0.97012 | 0.00499 | 0.01172 |
| m1 | 26 | 0.98533 | 0.99758 | 0.97100 | 0.00420 | 0.01125 |
| m2 | 34 | 0.98491 | 0.99845 | 0.97007 | 0.00508 | 0.01072 |
| m3 | 35 | 0.98543 | 0.99855 | 0.97125 | 0.00781 | 0.01067 |

External, mean over seeds, fixed threshold 0.5:

| Variant | RAGTruth QA F1 / AUROC | RAGTruth pooled F1 / AUROC | FaithBench F1 / AUROC |
|---|---|---|---|
| m0 | 0.3022 / 0.5560 | 0.5184 / 0.4937 | 0.8130 / 0.5093 |
| m1 | 0.3022 / 0.5573 | 0.5188 / 0.5567 | 0.8137 / 0.5410 |
| m2 | 0.3022 / 0.5396 | 0.5185 / 0.5461 | 0.8135 / 0.5698 |
| m3 | 0.3097 / 0.5577 | 0.5635 / 0.5898 | 0.1199 / 0.5963 |

Paper consequence: the component ablation can no longer say that per-variant tuning explains the flat result. The fixed-hp m3-minus-m1 F1 is +0.00010, and the QA AUROC gap is +0.0004. The claim-feature contribution is therefore indistinguishable from zero both in-domain and on held-out QA. This strengthens the negative result and must be stated plainly.

## 2. Table changes

### Table 6 (`tab:ec`), component ablation

- Replace the tuned-per-variant rows with the fixed-hp rows above.
- Caption: "EC-XGB component ablation on the HaluEval test set with one fixed hyperparameter configuration for every variant (mean over three seeds, n = 3,000)."
- Note: "All variants use max_depth 5, learning rate 0.05, 300 trees, subsample 0.7, colsample 0.7, selected once on m3 seed 42. No variant is tuned separately."
- Add a column `ΔF1 vs m1` or state it in the text: m2 − m1 = −0.00042, m3 − m1 = +0.00010.
- Remove the old sentence "Each variant was tuned separately". Replace with "Because the configuration is fixed, the column differences are feature effects, not tuning effects."

### Table 8 (`tab:target`), calibration

- Add calibration slope and intercept columns with 95% bootstrap CIs.
- Split the prior-only row into two: prior at 0.3107 (calibration prevalence) and prior at 0.1778 (test prevalence).
- Add a Brier skill score column against the test-prevalence prior.
- Keep raw/source-Platt/target-Platt rows. Drop isotonic if it is no longer part of the predeclared protocol; if kept, label it as a sensitivity check.
- Note: "Platt is fitted on the 5,034-row RAGTruth QA calibration split and never on test. The slope is the coefficient of the raw log-odds; a slope below 1 indicates overconfidence."

### New Table 8b (`tab:budget`), label budget

- Rows: 50, 100, 252, 500, 1,000, 5,034 calibration rows.
- Columns: ECE mean, ECE SD, ECE 2.5th, ECE 97.5th, Brier mean, Brier SD.
- Caption: "Target calibration with repeated label budgets (100 draws per budget, RAGTruth QA test n = 900)."
- This replaces the single-draw 50-row claim. If 50 rows remains unstable, say so.

### Table 9 (`tab:reliability`), reliability

- Keep if the bins are unchanged; otherwise regenerate from the rebuilt scores.
- Note that ECE is computed on the raw score, not the display score.

### Table 11 (`tab:stats`), intervals

- Replace seed-42 intervals with two-level bootstrap intervals (2,000 draws: groups within seed, then seeds).
- Add a `per-seed deltas` column or a second small table with the three seed deltas.
- Caption: "Paired two-level bootstrap deltas for EC-XGB (m3) minus the conventional model. Intervals resample groups within each seed and then seeds."

### Table 12 (`tab:perturb`), perturbation

- Replace B2 with m3.
- If the 20-row pilot is underpowered, mark the table as a pilot and move it to the supplement; do not present it as a full stability study.

### Table 13 (`tab:latency`) and Table 14 (`tab:cost`)

- RoBERTa-base encoder timing is now measured: 6.97 ms per row, batch 8, fp16, 512 tokens, 143 rows/s on the RTX 3060. Add it to Table 14 and state that the 127.5 ms m3 feature pipeline is about 18 times slower than the encoder alone; the base 26-feature pipeline is 61.8 ms.
- Correct the cost note: the feature pipeline runs three pretrained models; "API-free" does not mean model-free.

### Table 15 (`tab:judge`), judges

- Replace the 200-row balanced table with the 900-row natural-prevalence table (17.8% positive).
- Columns: model, accuracy, precision, recall, F1, AUROC, cost/1K, p50 latency.
- Add rows: GPT-5.6 Luna (3 repeats, mean and range), Qwen 3.5 4B if available, EC-XGB m3, length-only.
- Keep the balanced 200-row result only as a supplementary sensitivity check, clearly labeled.
- Note: "Natural prevalence is 17.8%; balanced-row F1 is not comparable."

### Table 16 (`tab:length`)

- Keep the B2 length table, but add one sentence that the same monotonic pattern holds for m3 if verified. Do not claim it without the artifact.

## 3. Figure changes

- Fig. 6 (`fig:perseed`): regenerate from fixed-hp per-seed metrics, not the tuned run. Caption: "Per-seed F1, AUROC and ECE for the fixed-hyperparameter B6 variants."
- Fig. 7 (`fig:reliability`): regenerate only if the rebuilt calibrators change the bin data.
- Fig. 8 (`fig:calibshift`): regenerate with the m3 display calibrator and label the panel "raw vs target-recalibrated".
- Fig. 9 (`fig:operating`): regenerate panel (b) with the 100-draw label-budget mean and 95% band.
- Fig. 10 (`fig:shapimp`): replace the B2 SHAP summary with the m3 summary from WP4.
- Fig. 11 (`fig:length`): unchanged unless m3 length diagnostics are added.
- UI screenshot: removed from the manuscript; keep any refreshed screenshot in the repository or supplementary material.

## 4. Equation additions

Add these to §5.3 and §5.4, each with one plain-language sentence:

1. Platt calibration on the raw score $s$:
   $$ \hat{p} = \sigma(a s + b), \qquad \sigma(z) = \frac{1}{1+e^{-z}}. $$
   State that $a$ is the calibration slope and $b$ the intercept, both fitted on validation rows only.

2. Brier skill score against the prior:
   $$ \mathrm{BSS} = 1 - \frac{\mathrm{BS}}{\mathrm{BS}_{\mathrm{prior}}}. $$
   State that BSS > 0 means the score beats a constant predictor at the evaluation prevalence.

3. Two-level bootstrap:
   $$ \Delta^{*} = \frac{1}{S}\sum_{s=1}^{S}\left(\theta_{\mathrm{EC}}^{*}(\mathcal{D}_{s}^{*}) - \theta_{\mathrm{conv}}^{*}(\mathcal{D}_{s}^{*})\right), $$
   where $\mathcal{D}_{s}^{*}$ resamples groups within seed $s$. State that this separates row noise from seed noise.

4. Label-budget standard deviation:
   $$ \widehat{\mathrm{SD}}_{n} = \sqrt{\frac{1}{B-1}\sum_{b=1}^{B}\left(\mathrm{ECE}_{n,b} - \overline{\mathrm{ECE}}_{n}\right)^2}. $$
   State that $B=100$ and $n$ is the number of labeled target rows.

Do not add equations that the implementation does not use.

## 5. Section text changes

### Abstract

- Keep the six-question order and 250-300 words.
- Replace the in-domain ablation sentence with the fixed-hp result: m0-m3 F1 spans 0.9849-0.9854, so the claim aggregates add no aggregate F1 gain. m2-m1 = -0.0004; m3-m1 = +0.0001 is the full multi-source stack, not the claim features.
- Replace the calibration sentence with the validation-only Platt result and the label-budget CI.
- Replace the judge sentence with the natural-prevalence result if WP6 completes; otherwise keep the balanced-row caveat.

### §5.1 Model comparison

- Keep the saturation argument.
- Add one sentence: "Under a single fixed configuration, the four variants differ by at most 0.0006 F1, so the in-domain ordering is not resolvable."

### §5.2 EC-XGB and ablation

- Rewrite around the fixed-hp table.
- State m3 − m1 F1 = +0.00010 and m3 − m1 QA AUROC = +0.0004.
- State that claim features are justified by per-claim explanations, not by aggregate gain.
- Remove the old "each variant was tuned separately" limitation because it no longer applies.

### §5.3 Calibration and transfer

- Lead with the slope: if $a < 1$, the raw score is overconfident; if $a \approx 1$, the scale is already usable.
- Report prior-only at both prevalences.
- Report BSS; if BSS ≤ 0, say the calibrated score does not beat the prior.
- Replace the single 50-row claim with the 100-draw mean and 95% band.
- Keep the P2 chance-level argument.

### §5.4 Statistical analysis

- Replace "1,000 group-level bootstrap resamples" with the two-level scheme.
- Report per-seed deltas and the pooled interval.
- Keep McNemar and Holm, but state the family explicitly.

### §5.5 Explanation, efficiency, judges

- Replace "the study predates the EC-XGB checkpoint" with the m3 SHAP results.
- Report the m3 top features and whether the B2 ranking survives.
- Replace the 200-row balanced judge table with the natural-prevalence table.
- Keep the cost caveat: three pretrained feature models, no LLM judge calls.

### §5.6 Error analysis

- State that the taxonomy is an author-verified, AI-assisted draft; never call it independent human annotation.
- Report the 20-case counts (13 content, 7 artifacts) and the kappa worksheet; if no second rater, say kappa is unavailable.
- Keep the length table and the error-enrichment caveat.

### Conclusion

- Update the headline numbers to the rebuilt values.
- Keep the negative-result framing: domain-specific risk score and adaptation check, not a general detector.

## 6. UI changes

- `web/components/risk-gauge.tsx`: "calibrated probability" -> "target-calibrated score" (done).
- `web/app/(site)/analyze/page.tsx`: explain that the score is fitted on RAGTruth QA and needs target recalibration (done).
- `web/components/assistant-ui/auto-risk-card.tsx`: "Evidence-calibrated risk probability" -> "Evidence-adjusted risk score" (done).
- `web/app/(site)/about/page.tsx` and `web/app/(site)/page.tsx`: matching wording (done).
- The UI screenshot is no longer part of the manuscript; it is kept in the repository or supplementary material.
- Update the Figure 5 caption to "Risk card with the target-calibrated display score."

## 7. Build and audit gates

- TeX Live 2026, 0 overfull, 0 underfull, 0 undefined references, 40 citations.
- Every table and figure within two pages of first mention; no pure float page.
- `src/models/review_number_audit.py` updated to the new number set; 100% match.
- Abstract 250-300 words, six-question order.
- No stale numbers from the tuned run (0.9844, 0.9852, 0.9857, 0.9855 in Table 6; 0.45 McNemar; 0.0005 claim delta) may survive.
