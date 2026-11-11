# Author-review workspace

This folder contains the three-reviewer package for the P0 annotations. The
labels originated as three independent AI reviewer passes; the three authors
reviewed and accepted them on 2026-11-11.

## Files

| File | What it is |
|---|---|
| `blind_reviewer_a.csv` / `blind_reviewer_b.csv` / `blind_reviewer_c.csv` | The first blind pass, before score/SHAP were added to the explanation rows. Kept for provenance. |
| `blind_reviewer_a_scored.csv` / `_b_scored.csv` / `_c_scored.csv` | The final blind pass. Taxonomy and claims labels unchanged; the 20 explanation rows were re-labelled with score and signed SHAP. |
| `p0_blind_overlap_merged.csv` | The 60 items with `a1/a2/a3` labels, ready for the ingest. |
| `taxonomy_reviewer_a.csv` | Reviewer A's 50-row taxonomy review. |
| `explanation_reviewer_b.csv` | Reviewer B's 100-row explanation review. |
| `claims_reviewer_c.csv` | Reviewer C's 100-claim review. |
| `ingest/` | The reviewed sheets in the filenames the ingest script expects. |
| `AI_REVIEW_RESULTS.md` | The full results, corrections, caveats, and the author check. |
| `AUTHOR_PROXY_REVIEW.md` | The author-proxy review, approved by the three authors on 2026-11-11. |

## Results

| Sheet | AI-author agreement | Cohen's kappa | Corrected | Blind Fleiss kappa |
|---|---:|---:|---:|---:|
| Taxonomy | 94.0% | 0.923 | 6.0% | 0.868 |
| Explanation | 98.0% | 0.965 | 2.0% | 0.940 |
| Claims | 88.0% | 0.788 | 12.0% | 0.761 |

Mean blind Fleiss kappa: **0.857**.

The score/SHAP columns matter: the first blind explanation pass had no score or
SHAP and gave kappa 0.20. With the columns, kappa is 0.94. Do not report the
0.20 as a rubric failure.

## Author check (completed 2026-11-11)

The three authors checked the package and accepted the labels. The check covered:

- the 3 taxonomy corrections (`P0E07`, `P0E35`, `P0E46`) and the 7
  benchmark-positive label-noise rows (`P0E27`, `P0E28`, `P0E37`, `P0E44`,
  `P0E45`, `P0E47`, `P0E50`);
- the 12 claim corrections, including `P0C024` and `P0C046`;
- the 2 explanation boundary rows (`P0X027`, `P0X028`) and the 13 SHAP-driven
  blind explanation changes;
- the 60 blind items and the merged `a1/a2/a3` labels;
- the author-proxy review in `AUTHOR_PROXY_REVIEW.md`.

The paper wording is updated to author-verified and the kappa values are
reported. If a label is changed later, edit the relevant `_scored.csv` and rerun
the merge and ingest commands below.

## Commands

Print a full case:

```powershell
& .venv\Scripts\python.exe -c "import sys,pandas as pd; r=pd.read_parquet('data/processed/qa_clean.parquet').set_index('sample_id').loc[sys.argv[1]]; print('Q:',r['question']); print(); print('C:',r['context']); print(); print('A:',r['answer'])" q_3061_correct
```

Rerun the merge and ingest after an edit:

```powershell
& .venv\Scripts\python.exe src\models\merge_p0_author_review.py
& .venv\Scripts\python.exe src\models\p0_ingest_author_review.py --docs-dir docs/rebuild/p0/author_review/ingest
```

The generated report is `artifacts/results/rebuild/p0/author_review_report.md`.

## Subagent instructions used

Each reviewer was told to:

- read the codebook and guide,
- label the 60-item blind file first, without opening the main sheets or another
  reviewer's file,
- then label one main sheet,
- fill `author_label`, `final_label`, `annotator_id`, `changed`,
  `review_status`, and short `notes`,
- not edit the paper, the original sheets, or another reviewer's files.

The explanation reviewers were sent a second, narrow follow-up to re-label only
the 20 explanation blind rows after the score and signed SHAP columns were
added.
