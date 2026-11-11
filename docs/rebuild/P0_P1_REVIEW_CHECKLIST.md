# P0/P1 reviewer checklist

Use this before the final build and again before submission. Every item needs a file path or command output, not a verbal claim.

## A. Annotation integrity

- [ ] All 250 rows have `review_status=needs_author_verification` or a final author status.
- [ ] The 60-row `p0_blind_overlap.csv` has no `ai_suggested` column filled.
- [ ] `annotation_manifest.json` lists the AI model, prompt version, codebook SHA-256, and date.
- [ ] Fleiss' kappa is computed from the 60 overlap rows only, not from the full set.
- [ ] AI–human agreement and correction rate are reported.
- [ ] No file says "independent human annotation" or "two human annotators."
- [ ] The paper says "three-author blinded review with AI pre-annotation."

## B. P0 experiments

- [ ] Feature ablation uses one fixed configuration for every row.
- [ ] Seeds 42/123/456 are present for every ablation row.
- [ ] No test row was used for tuning or selection.
- [ ] SHAP stability uses the same 1,000 rows for all seeds.
- [ ] CPU latency uses the same 200-row protocol as the GPU timing.
- [ ] Length-matched judge uses the 158-row matched set, not the full test set.
- [ ] Prompt sensitivity publishes all three prompts and does not select the best prompt on test.

## C. P1 experiments

- [ ] Task shift states that HaluEval dialogue/summarization are not a second independent QA corpus.
- [ ] Task-shift rows are deterministic and label-balanced.
- [ ] Retrieval qrels are the known source passage, not human web-search qrels.
- [ ] Retrieval reports recall@k before any downstream claim.
- [ ] External baseline is labeled a reimplementation; no original authors' number is claimed.
- [ ] Conformal calibration uses calibration rows only; test rows are never used to fit the quantile.
- [ ] Conformal coverage and set size are both reported; shift coverage is reported honestly.
- [ ] Multicalibration reports the worst subgroup, not only the average.

## D. Paper and audit

- [ ] Every new number maps to a file under `artifacts/results/rebuild/p0` or `p1`.
- [ ] `rebuild_number_audit.py` covers the new numbers and passes.
- [ ] The original 298-check audit still passes.
- [ ] Abstract is 250–300 words and follows the six-question order.
- [ ] No stale numbers from the pre-P0/P1 draft survive.
- [ ] Tables use body-size text; no clipped columns.
- [ ] Figures are legible at 100% PDF zoom; no overlapping labels.
- [ ] 0 overfull/underfull boxes; 0 undefined references.
- [ ] Every float is within two pages of first mention.
- [ ] 40 references maximum; every new citation is verified.

## E. Claims that must not appear

- [ ] "General-purpose hallucination detector."
- [ ] "Independent human annotation."
- [ ] "Positive held-out QA result."
- [ ] "Retrieval quality is evaluated end-to-end."
- [ ] "Calibration beats the prior on Brier."
- [ ] "The judge is worse than EC-XGB on held-out QA."
- [ ] "The model is cheaper than an encoder."

## F. Reviewer sign-off

- [ ] Methods reviewer: no leakage, no unfair comparison, correct statistics.
- [ ] Editor reviewer: contribution is clear, scope is honest, paper fits the venue.
- [ ] Author A: taxonomy and claim verdicts checked.
- [ ] Author B: explanation audit checked.
- [ ] Author C: retrieval/task-shift labels checked.
- [ ] All authors: final PDF read.
