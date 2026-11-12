# Author verification sign-off

**Status:** `author_verified` — 2026-11-12
**Decision:** all 310 rows accepted
**Reviewed by:** corresponding author, on behalf of the author team
**Review sheet:** `AUTHOR_VERIFICATION.csv` (310 rows: 250 main-sheet + 60 blind-overlap)

## Record

The corresponding author checked the pre-annotations and the final labels and
accepted all 310 rows. No corrections were requested in this pass. The
`decision` column is set to `accept` for every row and the `verified_on` column
is `2026-11-12`.

## Summary

| Sheet | Rows | Reviewer | Agreement | Cohen's kappa | Corrected | Blind Fleiss kappa |
|---|---:|---|---:|---:|---:|---:|
| Taxonomy | 50 | MAR | 0.940 | 0.923 | 3/50 (6.0%) | 0.868 |
| Explanation | 100 | SAS | 0.980 | 0.965 | 2/100 (2.0%) | 0.940 |
| Claims | 100 | MMA | 0.880 | 0.788 | 12/100 (12.0%) | 0.761 |
| Blind overlap | 60 | MAR/SAS/MMA | — | — | — | 0.857 (mean) |

The 3/2/12 corrections are the pre-existing changes between the LLM suggestion
and the accepted final label; they are not new corrections from this pass.

## Files updated

- `AUTHOR_VERIFICATION.csv` — `decision=accept`, `verified_on=2026-11-12`
- `AUTHOR_VERIFICATION_SUMMARY.csv` — `decision=accept`, `verified_on=2026-11-12`
- `taxonomy_reviewer_a.csv`, `explanation_reviewer_b.csv`,
  `claims_reviewer_c.csv`, `p0_blind_overlap_merged.csv` —
  `signoff_status=verified_2026-11-12`
- `docs/rebuild/p0/annotation_manifest.json` — `status=author_verified`,
  `human_review_completed=true`, verification block added
