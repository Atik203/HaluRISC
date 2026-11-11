# Three-reviewer AI review results (author-verification draft)

**Integrity statement.** These are three independent AI reviewer passes, not
three human authors. The corresponding author must spot-check the labels and
sign off before the paper describes them as an author review. The current paper
wording remains "three-author blinded review with AI pre-annotation" for the
human step.

## Protocol

- Three reviewers: `ai_reviewer_a`, `ai_reviewer_b`, `ai_reviewer_c`.
- Each reviewer first labelled the 60-item blind overlap without seeing the main
  sheets or the other reviewers' files.
- After the blind file was saved, each reviewer labelled one main sheet:
  - Reviewer A: taxonomy (50 rows).
  - Reviewer B: explanation audit (100 rows).
  - Reviewer C: claim verdicts (100 claims).
- The blind explanation rows initially lacked the score and signed SHAP terms
  required by codebook §4. They were re-labelled after those columns were added.
  The pre-SHAP explanation kappa was 0.20; the scored kappa is 0.94.
- No original sheet, paper file, or frozen artifact was edited.

## Results

| Sheet | Rows | AI-human agreement | Cohen's kappa | Corrected | Blind Fleiss kappa | Blind unanimous |
|---|---:|---:|---:|---:|---:|---:|
| Taxonomy | 50 | 0.940 | 0.923 | 6.0% | 0.868 | 85% |
| Explanation | 100 | 0.980 | 0.965 | 2.0% | 0.940 | 95% |
| Claims | 100 | 0.880 | 0.788 | 12.0% | 0.761 | 80% |

Mean blind Fleiss kappa: **0.857**.

The AI reviewers agreed with the original pre-labels on 94% of taxonomy rows,
98% of explanation rows, and 88% of claim rows. The claim sheet is the least
stable: 12 of 100 verdicts changed, driven mostly by partial evidence and
off-topic evidence sentences.

## Main corrections

### Taxonomy (3 corrections)

- `P0E07`: `answer_span_copy` -> `artifact_grounded_overflag`. The long copied
  span is grounded; the artifact is primary under codebook §2.1.
- `P0E35`: `numeric_or_temporal_error` -> `wrong_granularity`. The answer gives
  stadium capacity for the asked city population.
- `P0E46`: `entity_name_distortion` -> `entity_conflation`. "Etihad Renegades"
  merges the stadium name with the team name.

Seven benchmark-positive rows were judged grounded label noise: `P0E27`,
`P0E28`, `P0E37`, `P0E44`, `P0E45`, `P0E47`, `P0E50`. All 25 benchmark-negative
rows were judged genuine negatives.

### Explanation (2 corrections + 13 SHAP-driven changes)

- After the score and SHAP terms were added, 13 of the 20 blind explanation rows
  changed. The main sheet changed only `P0X027` (0.526) and `P0X028` (0.493),
  both to `unsure` because the score is within 0.05 of 0.5 and the terms are
  mixed.
- The initial 0.20 kappa is a design artifact: the blind rows had no score or
  SHAP, so reviewers were judging answer groundedness, not explanation
  plausibility. With the missing columns, kappa is 0.94.
- The main-sheet agreement is 98%; the two corrections are boundary calls, not
  rubric failures.

### Claims (12 corrections)

The main corrections include `P0C024` (supported: the evidence says "costs at
least 20 percent more") and `P0C046` (supported: the evidence ends "allow both
employers and their employees to make a saving"). The remaining corrections are
mostly `unsupported` where the AI accepted a partial match, or `unsure` where the
evidence sentence is a fragment. The blind claim evidence is truncated relative
to the main sheet, which depresses blind agreement; this is a design caveat, not
a label error.

## Caveats the authors must address

1. **These are AI reviewers.** Do not describe the kappa as human inter-annotator
   agreement. The human authors must spot-check and sign off.
2. **Taxonomy near-threshold negatives lack a "no error" code.** The codebook has
   no category for a grounded answer below the 0.5 threshold. Reviewers used
   `artifact_grounded_overflag` for both threshold false positives and
   near-threshold negatives. Add a `no_error_near_threshold` code or state the
   caveat in the paper.
3. **Blind claim evidence is truncated.** The blind file cuts some evidence
   sentences before the decisive clause. The main sheet is not truncated. Report
   the caveat if the blind claim kappa is used.
4. **Explanation kappa depends on having score/SHAP in the blind rows.** The
   paper should say so; otherwise a reader may think the rubric is unreliable
   when the first pass was simply missing the evidence.

## What the human authors still need to do

- Spot-check the 250 rows, prioritizing the 12 claim corrections, the 7
  taxonomy label-noise rows, and the 15 explanation boundary rows.
- Read the 60 blind items and confirm or change the AI labels.
- Sign the author-proxy review in `AUTHOR_PROXY_REVIEW.md`.
- If the labels are accepted, update the paper's wording from "AI pre-annotation"
  to the final human-review wording and report the kappa.
- If the labels are changed, rerun:

```powershell
& .venv\Scripts\python.exe src\models\merge_p0_author_review.py
& .venv\Scripts\python.exe src\models\p0_ingest_author_review.py --docs-dir docs/rebuild/p0/author_review/ingest
```

The generated report is `artifacts/results/rebuild/p0/author_review_report.md`.
