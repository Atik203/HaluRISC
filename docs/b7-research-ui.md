# B7 — Research UI & Demo

The web app (Next.js 16 + assistant-ui + Tailwind v4, package manager **pnpm**)
renders the Version B evidence: a six-tab experiment dashboard over all B1–B5
artifacts, a two-answer compare analyzer, an offline presenter demo, and the
assistant-ui chat.

## Run it

```powershell
# 1. Backend (repo root, .venv) — one worker, no --reload:
& .venv\Scripts\python.exe -m uvicorn src.api.main:app --port 8000

# 2. Frontend (web/):
pnpm install
pnpm run dev        # http://localhost:3000
```

- `web/.env.local` keys: `OPENAI_API_KEY`, `OPENAI_MODEL`, `NEXT_PUBLIC_APP_URL`,
  `NEXT_PUBLIC_ML_API_URL` (server-only key never `NEXT_PUBLIC_`).
- Chat needs the OpenAI key; **Analyze and Demo work without it**.

## Routes

| Route | What it is |
|---|---|
| `/chat` | assistant-ui Thread + `/api/chat` streaming; tool runs the calibrated model; real backend health dot |
| `/analyze` | Form → `/api/ml/predict` + `/api/ml/explain`; **compare mode** = two answers vs same question/context; expandable seven-group feature inspector; SHAP labeled as raw-model attribution |
| `/dashboard/overview` | B2 baselines (grouped CV, McNemar, CIs, tuning, leakage impact) + manifest provenance |
| `/dashboard/robustness` | B3 zero-shot transfer (per-dataset metrics, Δ vs in-domain, subgroups, bootstrap CIs, label sensitivity) |
| `/dashboard/calibration` | B4 ECE/ACE/Brier/NLL per subset × method, target-calibration story (0.81 → 0.13), reliability diagrams |
| `/dashboard/explainability` | B5 Kendall τ, group SHAP vs ablation, neutralization curve, perturbation stability, bootstrap Jaccard, reviewer export |
| `/dashboard/failures` | B3 error cases (FaithBench redacted), B5 failure cases, manual review sheet |
| `/dashboard/efficiency` | per-module latency, measured cost per 1,000, LLM-judge comparison |
| `/demo` | **Offline presenter walkthrough** — server-rendered from artifacts; no API, no OpenAI key, no network |
| `/about` | methodology + versions read from `manifest.json` (nothing hardcoded) |

Dashboard rule (AGENTS.md §8.3): **no fabricated data** — every number is read
from `artifacts/results/` by the server-side data layer `web/lib/results.ts`
(NaN-tolerant JSON reader; CSV reader handles quoted fields).

## API contract (`/api/ml/*` via next.config rewrites to FastAPI)

| Endpoint | Purpose |
|---|---|
| `GET /health` | status, model/feature version, artifacts loaded, feature models ready, device |
| `GET /meta` | thresholds, warning, device, `feature_groups`, versions (frontend reads this — nothing hardcoded) |
| `POST /predict` | `{risk_score, calibrated_score, label, thresholds, latency_ms, model_version, feature_version, warning, features{...}}` |
| `POST /explain` | `{top_features[{feature,value,impact}], base_value}` |
| `POST /judge` | LLM-as-judge (needs `OPENAI_API_KEY`) |

- **Deployable model**: B2 `xgboost_seed_42` + B4 Platt source calibrator
  (the B-run deployable); falls back to the Version A bundle if B-run files are
  missing. SHAP = TreeExplainer on the **raw** B2 model — the UI labels this.
- Thresholds: UI risk bands low/medium/high at 0.30/0.70 are **UI guidance**;
  the paper threshold is fixed at 0.5 (B2/B3/B4).
- Feature vectors are LRU-cached (256 entries) — repeated predict/explain of
  the same inputs skip the NLI/embedding stage.
- One worker, no `--reload`, inference lock serializes CUDA feature extraction,
  inputs bounded (question ≤ 5,000 / context ≤ 20,000 / answer ≤ 20,000 chars).

## Figures route

`GET /api/figures/[...path]` serves PNGs from `artifacts/figures/`:
top-level (`fig_reliability.png`) and whitelisted subfolders (`b3/*.png`,
`b4/*.png`). Basename-sanitized, no-store cache.

## Offline demo

`/demo` needs nothing but the artifacts folder — ideal for a clean clone, a
talk without network, or a presenter check. It walks through: headline numbers
(B2–B4), three real review cases (grounded / hallucinated / borderline from
`b5_review_cases.json`), a real B5 perturbation failure, the calibration-shift
evidence, and transfer figures.

## Mobile & accessibility

- Hamburger nav on small screens (`aria-expanded`/`aria-controls`), skip-to-content
  link, `aria-current` on active nav, `aria-live` result regions, chart
  `role="img"` labels, `prefers-reduced-motion` support, non-color risk cues
  (text chips + icons next to colors).
- Theme: dark-first, blue-violet accents; `glass-panel` / `gradient-text`
  utilities in `app/globals.css`.

## Deployment notes

- Backend container: see `docs/b6-reproducibility.md` §3 (CPU Dockerfile).
- Web: `pnpm run build` then `pnpm start`; the dashboard is `force-dynamic`
  (reads artifacts at request time — deploy with the artifacts folder mounted
  one level above the web root, as in the repo layout).
