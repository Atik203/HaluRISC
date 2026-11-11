# Internal readiness score (not a journal score)

This is an expert readiness estimate for the `rebuild` manuscript, not a
measured metric and not a prediction of a specific reviewer's decision. It
exists so the estimate can be checked against evidence instead of trusted as a
number.

## Estimate

| Stage | Score |
|---|---:|
| Before P0/P1 | 88 / 100 |
| After P0/P1 + reviewer hardening | 93 / 100 |
| After the three-author review (completed 2026-11-11) | 94 / 100 |
| After the six-section review and revision (2026-11-11) | 95 / 100 |
| Practical ceiling for this scope | ~96 / 100 |

## Rubric

| Criterion | Weight | Before | Now | Evidence to check |
|---|---:|---:|---:|---|
| Contribution and positioning | 20 | 18 | 18 | `Journal_Paper/halurisc.tex` introduction; `docs/rebuild/P0_P1_INTEGRATION_MAP.md` |
| Technical rigor | 20 | 17 | 20 | `docs/rebuild/P0_P1_REVIEWER_REPORT.md` "what passes"; grouped splits, conformal disjoint split, source-indicator sensitivity, author-verified labels |
| Experimental breadth | 20 | 14 | 18 | `docs/rebuild/P0_P1_RESULTS_INDEX.md`; MedHallu, HaluBench, task shift, retrieval, judges, external baseline |
| Calibration and statistics | 15 | 12 | 14 | `artifacts/results/rebuild/p1/conformal.json`, `multicalibration.csv`; `p1_number_audit.py` |
| Honesty and limitations | 10 | 8 | 10 | Limitations paragraph; `P0_P1_REVIEWER_RESPONSE.md`; no human-accuracy claim |
| Writing and presentation | 10 | 8 | 10 | `Journal_Paper/out/halurisc.pdf`; 25 pages, table notes folded into prose, short captions, six section reviews applied |
| Reproducibility | 5 | 4 | 5 | `src/models/p0_*`, `p1_*`; five audit commands; `annotation_manifest.json` |

The weighted total before is 81 by strict arithmetic, which is lower than the
88 impression because the original 88 was a holistic estimate, not this rubric.
Use the rubric for relative movement, not as an absolute conversion.

## What moved the score

- **+2.0 external validity:** MedHallu pair accuracy 0.867/AUROC 0.821, HaluBench non-overlap AUROC 0.600, task-shift collapse reported honestly.
- **+1.5 statistical rigor:** split and class-conditional conformal, multicalibration, length-matched judges, prompt sensitivity, paired bootstrap intervals.
- **+1.0 reviewer hardening:** P2/B2 misattribution fixed, tuned/fixed-hp labels fixed, P1 0% operating point stated, external baseline refit on 14,000 rows.
- **+0.5 explanation/ablation:** fixed-hp m3 ablation and cross-seed SHAP stability.

## What caps the score

- **Provenance:** the main-sheet labels originated as AI pre-annotations and were
  author-verified; the paper must keep that wording rather than implying a
  from-scratch human annotation.
- **Synthetic/single primary corpus:** HaluEval remains the main training and in-domain test set.
- **No positive held-out QA result:** RAGTruth QA AUROC 0.558, HaluBench 0.600, task shift collapses.
- **Deployed calibrator caveat:** the deployed Platt-vs-isotonic choice was made on test; Table 8 is the clean fixed-hp analysis.
- **Length/presentation:** 25 pages; the statistical subsections remain dense, but table notes are folded into the prose, captions are short, and every section passed a dedicated review.

## How to verify

```powershell
cd Journal_Paper
latexmk -pdf -interaction=nonstopmode -outdir=out halurisc.tex
cd ..
& .venv\Scripts\python.exe src\models\rebuild_number_audit.py
& .venv\Scripts\python.exe src\models\review_number_audit.py
& .venv\Scripts\python.exe src\models\p1_number_audit.py
& .venv\Scripts\python.exe src\models\float_proximity_audit.py
& .venv\Scripts\python.exe -m pytest tests -q
```

Expected: 116/116, 298/298, 108/108, 29/29 floats, 255 tests passed, 25 pages.

## Author review (completed)

The three authors reviewed the package on 2026-11-11. The kappa values are:

- main-sheet AI-author agreement: taxonomy 0.940 (Cohen κ 0.923), explanation
  0.980 (κ 0.965), claims 0.880 (κ 0.788);
- blind inter-author Fleiss κ: taxonomy 0.868, explanation 0.940, claims 0.761;
- mean blind κ: 0.857.

The paper says "three-author blinded review with AI pre-annotation" and
"author-verified." If a label changes later, rerun the merge and ingest commands
in `docs/rebuild/p0/author_review/README.md`.
