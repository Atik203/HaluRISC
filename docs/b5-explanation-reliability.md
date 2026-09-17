# B5 — Explanation Reliability & Error Analysis (incl. Manual Review Guide)

Phase B5 defends SHAP as *evaluated evidence* rather than decoration. It runs on
the **B2 `xgboost_seed_42` model** (CPU-portable) with the **B4 Platt source
calibrator** (the predeclared deployable, roadmap B4.2).

## Contents

- [What B5 measured](#what-b5-measured)
- [Reproduce](#reproduce)
- [Manual Review Guide (B5.5)](#manual-review-guide-b55--do-this-by-hand)
- [Result (AI-assisted audit, 2026-09-17)](#result-ai-assisted-audit-2026-09-17)
- [Interpreting failure cases (B5.6)](#interpreting-failure-cases-b56)

Headline numbers and their interpretation live in
[02-experiments-and-results.md](02-experiments-and-results.md) §8–§9.

## What B5 measured

| Experiment | Roadmap | Output |
|---|---|---|
| Importance triangulation: mean-|SHAP| vs permutation importance (Kendall τ) + group ablation (neutralization proxy) | B5.1 | `b5_feature_importance.json` |
| Top-k neutralization → prediction change (k = 1,3,5,10) | B5.2 | `b5_neutralization.json` |
| Text perturbations with full feature re-extraction: numeric, date, entity (spaCy NER), support-sentence removal, irrelevant insertion, clause shuffle | B5.3 | `b5_perturbations.csv`, `b5_perturbation_aggregates.csv`, `b5_failure_cases.json` |
| Bootstrap CIs for mean-|SHAP| + top-k set stability (Jaccard) | B5.4 | `b5_stability_bootstrap.json` |
| Reviewer export (up to 10 FP / 10 FN / 20 borderline, capped by availability) | B5.5 | `b5_review_cases.csv` / `.json` |
| Expert audit of the review sheet (two AI-assisted passes, 23/26 agreement) | B5.5 | `b5_review_cases_reviewed.csv` |
| Length-binned error analysis (benchmark length confound) | B5 follow-up | `b5_length_error_analysis.csv` / `.json` |
| Failure cases (|Δscore| > 0.3 or SHAP top-1 flip) | B5.6 | `b5_failure_cases.json` |

All outputs live in `artifacts/results/b5/`. Figures (none required; tables and
CSVs carry the evidence) — the dashboard's *Explainability* tab renders them.

## Reproduce

```powershell
# Full B5 (perturbation re-extraction is the heavy step; use --device cuda on a GPU):
& .venv\Scripts\python.exe src\models\run_b5_explanation_reliability.py --device cuda --n-perturb 120

# Cheap local smoke (no heavy model loading; skips perturbations):
& .venv\Scripts\python.exe src\models\run_b5_explanation_reliability.py --n-perturb 0

# Through the B6 orchestrator (b2 → b5):
& .venv\Scripts\python.exe src\models\run_all_experiments.py --from b2 --to b5 --device cpu
```

Crash-safe: perturbation results checkpoint per sample (`b5_perturbations.csv`);
re-running resumes instead of restarting. Perturbation random seeds are fixed
(42 + sample index) → deterministic outputs.

**Known caveats to report**: entity (n=78), numeric (n=22) and date (n=13)
perturbations act only on samples that contain such tokens — smaller n than
irrelevant-insert (n=120); clause-shuffle is a limited paraphrase proxy.

## Manual Review Guide (B5.5) — do this by hand

Two independent review passes are expected. The B-run export contains **26
cases** (9 FP, 10 FN, and 7 borderline cases in the 0.35–0.65 calibrated band;
the exporter caps each pool at what is available). This is a **manual step** —
no script writes your judgment.

### 1. Open the sheet

`artifacts/results/b5/b5_review_cases.csv` (open in Excel / Google Sheets /
any CSV viewer). Columns:

| Column | Meaning |
|---|---|
| `sample_id` | row identifier (joinable to `data/processed/qa_clean.parquet`) |
| `question` / `context` / `answer` | the exact inputs the model scored |
| `label` | true label (1 = hallucinated, 0 = grounded) |
| `raw_score` | raw XGBoost probability |
| `calibrated_score` | Platt-calibrated probability (the deployed score) |
| `top5_shap_features` | top-5 features by \|SHAP\| (comma-separated) |
| `reviewer_1`, `reviewer_2`, `agreement` | **fill these** (currently empty) |

### 2. Rubric per case (proposed — adjust freely)

Read question → context → answer. Then mark **each** of the two reviewer
columns with one of:

- `plausible` — the answer looks grounded in the context, the risk score is
  sensible given the evidence, and the top-5 SHAP features point at what a
  human would flag (e.g., missing entity overlap, low NLI entailment).
- `implausible` — the explanation is inconsistent with the evidence (score too
  high/low for clearly grounded/hallucinated content, or SHAP features point
  at irrelevant signals).
- `unsure` — genuinely ambiguous; use sparingly and note why.

**`agreement` column**: `yes` if both reviewers marked the same value,
`no` otherwise (disagreements are the roadmap B5.5 deliverable — they get
reported, not "fixed").

### 3. Save and report

1. Save the filled sheet as **`artifacts/results/b5/b5_review_cases_reviewed.csv`**
   (do not overwrite the original export).
2. Tally agreement:

```powershell
& .venv\Scripts\python.exe src\models\review_tally.py
```

This prints: cases reviewed, per-reviewer distribution (plausible/implausible/
unsure), agreement rate, and the disagreement list (sample_ids). Output values
are purely descriptive — no fixed pass thresholds (B5.7).

### 4. What the tally means for the paper

- High agreement + mostly `plausible` → SHAP explanations corroborate the
  evidence; report the numbers.
- Disagreements or `implausible` clusters → report them as failure cases
  (B5.6) and discuss where SHAP is unstable or inconsistent with evidence.

## Result (AI-assisted audit, 2026-09-17)

Both passes were completed by an AI assistant at the authors' direction, and
the sheet records this in the `review_mode` column (`ai-expert`). Pass 1 judged
evidence grounding (does the score fit how well the answer is supported by the
context), pass 2 judged the SHAP feature story. Tally
(`python src/models/review_tally.py`):

- 26 cases: 9 FP, 10 FN, 7 borderline.
- Pass agreement 23/26 (88%). Disagreements: `q_4156_correct`,
  `q_1729_hallucinated`, `q_5173_hallucinated`.
- Implausible 15/26 on both passes; plausible 6 (pass 1) and 7 (pass 2);
  unsure 5 and 4.
- False positives follow the HaluEval answer-length confound
  (`b5_length_error_analysis.csv`: 67% of FPs land in the 5–8 word bin, where
  the hallucinated share is already 90%). False negatives reuse context
  entities, which keeps lexical overlap high.
- Two labelled-hallucinated cases (items 2251 and 2436) are supported by their
  context and look like benchmark label noise.

Caveat: the sheet is deliberately error-heavy, so these proportions are not
test-set estimates. Do not describe this audit as a two-human-reviewer study.

## Interpreting failure cases (B5.6)

`b5_failure_cases.json` lists perturbation evaluations where the score moved
more than 0.3 or the SHAP top-1 feature flipped. These are *candidates* for
review, not automatic claims of model failure — cross-check the perturbed text
excerpt against the original before reporting.
