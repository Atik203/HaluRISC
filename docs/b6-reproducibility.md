# B6 — Reproducible Publication Artifact

B6 turns the Version B pipeline into a single reproducible protocol: a
config-driven orchestrator, an enriched frozen manifest, and a CPU-compatible
Docker path. No hidden local paths: a clean clone + the documented downloads
regenerate the required artifacts.

## 1. Config-driven orchestrator

```powershell
# Inspect the exact commands without executing anything:
& .venv\Scripts\python.exe src\models\run_all_experiments.py --dry-run

# Full pipeline (CPU; use --device cuda on a GPU box):
& .venv\Scripts\python.exe src\models\run_all_experiments.py --device cpu

# Subset of phases (e.g. resume after a crash) and keep-going:
& .venv\Scripts\python.exe src\models\run_all_experiments.py --from b3 --to b5 --keep-going
```

- Config: `configs/version_b.yaml` (schema `b6-run-all-v1`). Phases: HaluEval
  download/prepare → official RAGTruth + FaithBench downloads → B1 unified build
  → feature extraction → B2 → B3 → B4 → B5 → manifest → verify.
- Each phase subprocess-invokes the existing per-phase runner (`run_b2_baselines.py`,
  `run_b3_cross_domain.py`, …) — runners stay the single source of truth.
- `{device}` / `{batch_size}` placeholders resolve from config defaults or CLI
  overrides (`--device`, `--batch-size`).
- Every execution writes `artifacts/results/run_all_report.json`:
  resolved defaults, per-phase exit code / status / duration / start time, git
  commit (null outside a clone), overall status.
- `--keep-going` continues past failures (used for crash recovery); without it,
  the first failing phase aborts the run.

## 2. Manifest (`artifacts/results/manifest.json`)

Regenerate / freeze:

```powershell
& .venv\Scripts\python.exe src\models\make_manifest.py
```

Contents:

| Section | What it records |
|---|---|
| `generated_at`, `git_commit`, `source_fingerprint` | timestamp; git HEAD when inside a clone; sha256 over the notebook cell-3 embedded source hashes when run on Colab (`HALU_SOURCE_FINGERPRINT` env) |
| `model_version`, `feature_version`, `n_features`, `nli_model`, `nli_provenance` | deployed model identity |
| `seeds`, `feature_groups` | `[42, 123, 456]`; the 7 feature groups |
| `split_report` | split integrity (grouped by `item_idx`, leakage-free) |
| `dataset_sha256` | processed inputs (qa_clean, features, models, split files) |
| `raw_sha256` | every raw input file incl. download `revision.json` provenance |
| `b_artifacts_sha256` | B2–B5 result/model hashes (incl. all 7 B5 outputs) |
| `versions`, `hardware`, `env` | package versions, CPU/RAM/GPU, `HALU_*` config |
| `artifacts` | full produced-file list (**excludes** `_stages`, smoke/tmp dirs, crash logs) |

Colab note: cell 13 sets `HALU_SOURCE_FINGERPRINT` from cell-3 `HASHES` before
running `make_manifest.py`, so Colab-produced manifests still fingerprint the
exact shipped source even though `git_commit` is null there. Locally, regenerate
the manifest after the final commit to freeze the real commit hash (B8.1).

## 3. CPU-compatible Docker path (backend only)

```powershell
docker build -t halurisc-api .
docker run --rm -p 8000:8000 halurisc-api
curl http://127.0.0.1:8000/health
```

- Single-stage `Dockerfile` (`python:3.12-slim`, pinned `requirements.txt`,
  `src/` + `artifacts/`), `libgomp1` for xgboost's OpenMP runtime, one uvicorn
  worker, no `--reload`, healthcheck on `/health`.
- CUDA remains an optional local acceleration path (`HALU_API_DEVICE=cuda` on
  the host); the image itself is intentionally CPU.
- `.dockerignore` keeps the image lean and secret-free (no `.env`, no data/raw,
  no web/node_modules).

## 4. Regenerate-from-scratch checklist

1. Clone the repo (fresh).
2. Install: `python -m venv .venv` → `.venv\Scripts\pip install -r requirements.txt`
   (GPU runs use Colab instead — see README "Train on Colab").
3. Data: `python src/data/download.py` + `prepare.py` (HaluEval), and the B1
   downloads run inside the pipeline (RAGTruth/FaithBench official).
4. `python src/models/run_all_experiments.py --device cpu` (or `--from/--to`).
5. Gate: `python src/models/verify_artifacts.py` must print
   `ALL ARTIFACTS VERIFIED` (B2 boosters, VA models, B3/B4 prediction schemas,
   B4 calibrators, B5 reports).
6. Freeze: `python src/models/make_manifest.py` → commit the manifest.
7. (Optional) `docker build` the backend, or run `uvicorn src.api.main:app`.

Note: `data/raw/` is gitignored (FaithBench is CC BY-NC-SA — never committed).
`verify_artifacts.py` enforces that FaithBench text never ships in error-case
artifacts.
