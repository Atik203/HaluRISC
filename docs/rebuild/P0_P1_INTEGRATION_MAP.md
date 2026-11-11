# P0/P1 paper integration map

Status: working note. Fill every placeholder from a frozen artifact before the final build.

## 1. P0 changes

### Ablation table

- Replace `tab:ablation` (B2 feature-group ablation) with the fixed-hp m3 feature-group ablation.
- Keep the same 7 groups plus two controls: no claims, no source.
- Caption: "Feature-group ablation of the fixed-hyperparameter EC-XGB m3 model on HaluEval (mean over three seeds)."
- Note: same config as Table `tab:ec`; no test tuning.
- Move the B2 table to the supplement or drop it.

### Explanation section

- Add one sentence: cross-seed SHAP rank correlation and top-10 overlap.
- If the stability is low, say so; do not claim stable attributions.
- Keep the audit paragraph, but replace the AI-only draft with the three-author blinded review counts and kappa once the authors finish.

### Judge table

- Add a compact stress-test table or two rows:
  - length-matched GPT/Qwen vs EC-XGB;
  - prompt-sensitivity F1/AUROC range.
- If the judges collapse on the length-matched set, state that the benchmark artifact affects them too.
- Keep the natural-prevalence table as the main judge table.

### Claim verdicts

- Add one paragraph in §5.5: 100-claim author audit, accuracy, kappa, and the support-first failure mode.
- Do not add a full table if the paragraph can carry the numbers.

### Abstract

- Add "three-author blinded review with AI pre-annotation" and the kappa.
- Add the length-matched judge result if it changes the conclusion.
- Keep 250-300 words and the six-question order.

## 2. P1 changes

### Task shift

- Add HaluEval dialogue and summarization as a task-shift test.
- Table or paragraph in §5.3 or §5.6: F1/AUROC/ECE for m3, m0, length-only.
- State clearly: same benchmark family, different tasks, not a second independent QA corpus.

### Retrieval stress

- Add a table in §5.6 or a short paragraph:
  - recall@1/5/10, MRR;
  - EC-XGB gold vs retrieved context: AUROC, F1, ECE;
  - if retrieval drops performance, state the deployment limitation.
- Do not claim end-to-end web retrieval quality.

### External baseline

- Add one row to the model-comparison table or a small external table:
  - lightweight ROUGE-L + SBERT + NLI ensemble;
  - F1/AUROC/ECE with CIs;
  - label as a reimplementation.
- If it is weaker than EC-XGB, say so; if stronger, say so.

### Conformal and multicalibration

- Add a short subsection paragraph in §5.3, not a new results subsection.
- Add one equation for split conformal:
  \[
  \hat{q} = \mathrm{Quantile}_{1-\alpha}\{1-\hat{p}_{true}(x_i)\},
  \]
  and the prediction set rule.
- Table: coverage at 90/95%, average set size, QA coverage, FaithBench shift coverage.
- Multicalibration: ECE/Brier by length bin and claim-count bin; report the worst subgroup.

## 3. Table and figure budget

- Current: 16 tables, 11 figures.
- Target: at most 18 tables and 12 figures.
- Combine P0 judge stress rows into one table; combine retrieval/task-shift into one external-validity table if possible.
- Do not add a figure unless it carries a result that a table cannot.

## 4. Section rules

- Keep the six results subsections.
- No 1.1.1 headings.
- Few large paragraphs; merge small paragraphs unless a float separates them.
- Every figure explained in detail; captions short.
- Body-size table text; no `\small` tables unless the venue requires it.
- New equations only if the implementation uses them.

## 5. Gates

- Every number maps to `artifacts/results/rebuild/p0` or `p1`.
- Audits updated; 0 mismatches.
- 0 overfull/underfull; 0 undefined refs; floats within two pages.
- Abstract 250-300 words; 40 references maximum.
- Two reviewer passes before submission.
