# Rebuild status (final)

Branch: `rebuild`, cut from `main` at `7106d72`.
Paper: 25 pages, 11 figures, 16 tables, 40 references, 0 overfull/underfull, 0 undefined references, all 27 floats within two pages.
Audits: original 298/298 and rebuild 116/116.
Next: `docs/rebuild/STRENGTHENING_PLAN.md` (P0/P1/P2, proposed; not started).

## Completed

| WP | Status | Evidence |
|---|---|---|
| WP1 fixed-hp ablation | Done | `artifacts/results/rebuild/fixed_hp/` |
| WP2 calibration | Done | `artifacts/results/rebuild/calibration/` |
| WP3 two-level intervals | Done | `artifacts/results/rebuild/intervals/` |
| WP4 m3 SHAP + perturbation | Done | `artifacts/results/rebuild/explanations/` |
| WP5 taxonomy/audit draft | Done, author verification pending | `docs/rebuild/wp5_*` |
| WP6 judge + encoder | Done | `artifacts/results/rebuild/judge/` |
| WP7 external bar + length control | Done | `docs/rebuild/external_bar.md` |
| WP8 paper + UI | Done | `Journal_Paper/halurisc.tex`, `web/` |
| WP9 senior review | Done; fixes applied | review report in the session |

## Key numbers

- Fixed-hp HaluEval F1: m0 0.9849, m1 0.9853, m2 0.9849, m3 0.9854. m2-m1 = -0.0004; m3-m1 = +0.0001 (full stack).
- m0 vs m3 McNemar: b=6, c=8, p=0.789.
- RAGTruth QA raw calibration: ECE 0.731, Brier 0.689, Cox slope 0.18.
- Target Platt: ECE 0.125 [0.099, 0.151], Brier 0.161 [0.151, 0.171], BSS -0.10.
- Prior at test prevalence: Brier 0.146, lower than every calibrated row.
- Label budget, 100 draws: 50 rows 0.130 +/- 0.060 [0.020, 0.223]; 252 rows 0.124 +/- 0.029 [0.075, 0.188]; 5,034 rows 0.125.
- QA m3-m0 AUROC delta +0.0017 [-0.0270, 0.0291]; pooled F1 delta +0.0451 [0.0348, 0.0554].
- m3 SHAP: overlap 1.66, words 1.11, cosine 0.96, chars 0.82, Jaccard 0.78; top-10 77.9% of mass.
- Judge, 900 QA rows: GPT F1 0.565, AUROC 0.910, flags 41.4%, p50 2283 ms, $0.234/1K; Qwen F1 0.602, AUROC 0.778, flags 19.9%, p50 3050 ms.
- EC-XGB: F1 0.310, AUROC 0.558, flags 96.8%, p50 127.5 ms.
- Length-only: F1 0.257, AUROC 0.712, flags 7.3%. All-positive F1 0.302.
- Encoder timing: 6.97 ms/row, batch 8, fp16.
- 26-case explanation draft: 13 implausible, 3 unsure, 10 plausible.
- 20-case error draft: 13 content, 7 audit artifacts; 8 m3 false positives, 12 false negatives.

## Human verification still required

- `docs/rebuild/human_verification_checklist.md` walks through every row.
- Both WP5 CSVs have `review_status = needs_author_verification` on all 46 rows.
- No kappa is reported; a second rater is needed.
- The paper describes the audit as an AI-assisted draft, not human annotation.

## If a step must be rerun

```powershell
pwsh -File scripts/run_rebuild.ps1
```

The runner is idempotent, logs each step under `artifacts/results/rebuild/logs/`, and runs both audits at the end.

## Integrity notes

- The deployed display score is **target-calibrated** (fitted on RAGTruth QA), not source-calibrated. The paper, UI, and screenshot now say so.
- The deployed calibrator was selected in an earlier fit that compared Platt and isotonic on test; the paper states this in the limitations, and the reported calibration analysis refits Platt without test-based selection.
- The 61.8 ms base-pipeline number is reported as the base pipeline; the m3 path is 127.5 ms.
- Judge cost assumes $0.20/M uncached input, $0.02/M cached input, $1.20/M output; the full run cost $0.63.
