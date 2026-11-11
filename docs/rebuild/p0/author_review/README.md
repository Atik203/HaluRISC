# Author-review workspace

This folder contains the three-reviewer draft for the P0 package. The reviewers
are AI subagents, not human authors. The corresponding author must spot-check
the labels and sign off before the paper describes them as a human author review.

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
| `AI_REVIEW_RESULTS.md` | The full results, corrections, caveats, and what the human authors must check. |
| `AUTHOR_PROXY_REVIEW.md` | The author-proxy review written for the corresponding author to check and sign. |

## Results

| Sheet | AI-human agreement | Cohen's kappa | Corrected | Blind Fleiss kappa |
|---|---:|---:|---:|---:|
| Taxonomy | 94.0% | 0.923 | 6.0% | 0.868 |
| Explanation | 98.0% | 0.965 | 2.0% | 0.940 |
| Claims | 88.0% | 0.788 | 12.0% | 0.761 |

Mean blind Fleiss kappa: **0.857**.

The score/SHAP columns matter: the first blind explanation pass had no score or
SHAP and gave kappa 0.20. With the columns, kappa is 0.94. Do not report the
0.20 as a rubric failure.

## What the human author must check, in this order

1. **Read the codebook and guide.** `docs/rebuild/p0/codebook_v2.md`,
   `docs/rebuild/p0/P0_HUMAN_REVIEW_GUIDE.md`.
2. **Spot-check the taxonomy corrections.** Reviewer A changed 3 of 50:
   `P0E07`, `P0E35`, `P0E46`. It also flagged 7 benchmark-positive rows as
   grounded label noise: `P0E27`, `P0E28`, `P0E37`, `P0E44`, `P0E45`, `P0E47`,
   `P0E50`.
3. **Spot-check the claim corrections.** Reviewer C changed 12 of 100. The two
   clearest are `P0C024` and `P0C046`, both `supported` because the evidence
   contains the decisive clause the AI rationale said was missing.
4. **Spot-check the explanation boundary.** Reviewer B changed 2 of 100, both to
   `unsure` because the score is within 0.05 of 0.5. The 13 SHAP-driven changes
   are in the blind set, not the main sheet.
5. **Check the 60 blind items.** The three labels and the merged file are in this
   folder. If you change a label, edit the `_scored.csv` file and rerun the
   merge and ingest.
6. **Sign the author-proxy review.** Edit `AUTHOR_PROXY_REVIEW.md`, fill the
   sign-off line, and keep it in the repo.
7. **Decide the paper wording.** If you accept the labels, change
   "AI pre-annotation" to the final author-review wording and report the kappa.
   If you change labels, rerun the commands below first.

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
