# HaluRISC — Documentation Index

Guides and references for the completed HaluRISC study. All paths are
repo-root-relative and assume the `.venv` environment described in the README.

## Start here (reading order)

1. [`01-project-and-methods.md`](01-project-and-methods.md) — project overview,
   gap, datasets, features, models, training protocol, deployed system.
2. [`02-experiments-and-results.md`](02-experiments-and-results.md) — every
   experiment with numbers and artifact paths, including negative results.
3. [`03-reproduction-and-defense.md`](03-reproduction-and-defense.md) — rebuild
   and verify, demo script, numbers cheat sheet, defense Q&A, glossary.
4. [`04-interface-guide.md`](04-interface-guide.md) — what every UI element
   means (chat risk card, gauge and cutoffs, SHAP chart, dashboard tabs), written
   for a first-time viewer of the demo.

Supporting sources of truth:

- [`../blueprint.md`](../blueprint.md) — unified research design and verified results
- [`../roadmap.md`](../roadmap.md) — implementation record, phase by phase

## Deep guides

| Doc | Covers |
|---|---|
| [`b5-explanation-reliability.md`](b5-explanation-reliability.md) | Explanation-reliability experiments, outputs, and the expert audit procedure (B5.5) |
| [`b6-reproducibility.md`](b6-reproducibility.md) | Run-all orchestrator, manifest, Docker |
| [`b7-research-ui.md`](b7-research-ui.md) | Web routes, dashboard tabs, API contract, offline demo |
| [`04-interface-guide.md`](04-interface-guide.md) | Plain-language guide to the UI: risk card, gauge cutoffs, SHAP chart, dashboard tabs, troubleshooting |
| [`manifest.frozen.json`](manifest.frozen.json) | Frozen artifact hashes, seeds, hardware, environment |

## Quick reference

```powershell
# Experiment runner (dry-run first)
python src/models/run_all_experiments.py --dry-run

# Artifact gate and manifest freeze
python src/models/verify_artifacts.py
python src/models/make_manifest.py

# B5 follow-ups
python src/models/analyze_length_shortcut.py   # length-confound table
python src/models/review_tally.py              # expert-audit tally

# Human-labeling sheets (fill by hand, then report)
python src/data/build_audit_review_sheet.py    # 50-sample audit review sheet
python src/models/eval_claims.py --build --n 40 # claim-eval sheet (loads NLI)
python src/models/eval_claims.py --from-feedback data/processed/feedback_eval.csv

# Tests, API, web
python -m pytest tests -q
python -m uvicorn src.api.main:app --port 8000
cd web; pnpm install; pnpm run dev
```

## Reading paths

- **Defense prep:** 01 → 02 → 03 §9–§13 (script, cheat sheet, Q&A, do-not-say).
- **Reproduction:** 03 §1–§8 → `b6-reproducibility.md`.
- **Method deep dive:** 01 §2–§4 → `blueprint.md` §5–§8.
- **Results deep dive:** 02 → `artifacts/results/*` and `docs/manifest.frozen.json`.
- **Web/system deep dive:** 01 §5 → `b7-research-ui.md`.
- **Demo walkthrough:** 04 → `b7-research-ui.md` (Offline demo).
- **Explanation reliability:** 02 §8–§9 → `b5-explanation-reliability.md`.
