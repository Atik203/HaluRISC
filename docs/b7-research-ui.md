# B7 — Research UI & Demo

The web app (Next.js 16 + assistant-ui + Tailwind v4, package manager **pnpm**)
renders the project evidence: a six-tab experiment dashboard over all B1–B5
artifacts, a two-answer compare analyzer, an offline presenter demo, and the
assistant-ui chat.

## Contents

- [Run it](#run-it)
- [Routes](#routes)
- [API contract](#api-contract-apiml-via-nextconfig-rewrites-to-fastapi)
- [Figures route](#figures-route)
- [Offline demo](#offline-demo)
- [Mobile & accessibility](#mobile--accessibility)
- [Chat auto-analysis (Tier 1)](#chat-auto-analysis-b75-tier-1)
- [Claim-level verdicts (Tier 2)](#claim-level-verdicts-b75-tier-2)
- [Retrieval & citations (Tier 3)](#retrieval--citations-b75-tier-3)
- [Hybrid judge & feedback (Tier 4)](#hybrid-judge--feedback-b75-tier-4)
- [Deployment notes](#deployment-notes)

System overview: [01-project-and-methods.md](01-project-and-methods.md) §5.

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
| `POST /predict` | `{risk_score, calibrated_score, legacy_score, label, thresholds, latency_ms, model_version, feature_version, warning, features{...}}` |
| `POST /explain` | `{top_features[{feature,value,impact}], base_value}` |
| `POST /verify` | Per-claim verdicts and evidence quotes; adjusts `calibrated_score` |
| `POST /index` / `POST /retrieve` | Document indexing and passage retrieval (BM25 + FAISS + reranker) |
| `POST /judge` | LLM-as-judge for borderline claims (needs `OPENAI_API_KEY`) |
| `POST /feedback` | Stores chat verdict feedback for threshold tuning |

- **Deployed bundle**: the current `artifacts/models/` set is served, with the
  evidence-domain display calibrator (`models/b4/calibrator_display.joblib`)
  producing `calibrated_score` and the HaluEval-Platt calibrator kept as
  `legacy_score`. Older bundles are used only as a fallback when current files
  are missing. SHAP = TreeExplainer on the **raw** B2 model — the UI labels this.
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

## Chat auto-analysis (B7.5 Tier 1)

Chat works like a normal LLM conversation — **no pasting question/context/answer**.
When auto-analysis is enabled (default), every completed assistant answer gets an
automatic **risk card** below it:

- compact gauge + calibrated score + label + top-3 SHAP features
- grounding note: `Evidence: pasted context` **or** `conversation only — not
  externally grounded` (honest about the difference)
- expandable details: all feature values, model/feature version, latency
- the master toggle in the chat header disables cards entirely; the optional
  "Evidence context" panel lets you paste a document once so answers are checked
  against it (RAG-style, matching the model's training distribution)

Inputs are assembled client-side (`web/lib/analysis-input.ts`): question = last
user message; context = evidence panel or the last few conversation turns;
answer = the completed assistant message. The combined additive endpoint
`POST /api/ml/analyze` (predict + explain in one call, one feature extraction)
serves the cards; the LRU feature cache makes repeats near-instant. No LLM
tool-calling and no extra tokens are involved — analysis runs after streaming
completes, so the chat never waits for it.

### Claim-level verdicts (B7.5 Tier 2)

The card calls `POST /api/ml/verify` first (falls back to `/analyze`): the
answer is split into atomic claims (`src/claims/decompose.py` — deterministic
sentence/clause splitter) and each claim gets a 3-way NLI verdict against the
evidence sentences (`src/claims/verify.py`, cross-encoder pairs):

- `supported` (entailment ≥ 0.5 and beats contradiction)
- `contradicted` (contradiction ≥ 0.5)
- `unsupported` (no supporting evidence)

Each claim chip shows its evidence sentence; the aggregate shows counts and an
overall verdict, while the calibrated XGBoost score stays as a labelled
secondary signal (SHAP = raw-model attribution). Verdict thresholds are
documented constants, evaluated against human labels by
`src/models/eval_claims.py` (build a human-labeling CSV, then measure
flagging precision/recall — descriptive, no fixed pass thresholds).

### Retrieval & citations (B7.5 Tier 3)

Evidence can come from **web search** and **uploaded documents**, selected
per answer via `evidence_mode: auto | context | index | web` on `/verify`
(auto = index → web → pasted context):

- **Documents**: upload PDF/DOCX/TXT from the chat panel →
  `POST /api/ml/index` (chunked, embedded with the shared SBERT embedder,
  indexed with FAISS + BM25, persisted under `data/processed/retrieval_index/`).
  `GET /api/ml/index` (status), `DELETE /api/ml/index` (clear),
  `POST /api/ml/retrieve` (hybrid BM25 + dense + RRF fusion, optional
  cross-encoder rerank `cross-encoder/ms-marco-MiniLM-L-6-v2`).
- **Web search**: `BRAVE_SEARCH_API_KEY` in the root `.env` (Tavily stays as the legacy fallback), server-side only,
  never committed — `.env.example` holds the placeholder); per-claim queries,
  LRU-cached, ~1,000 credits/month on the free tier.
- **Citations**: each claim records `evidence_source` (`web:<url>` |
  `doc:<name>`) + `evidence_url`; the card renders source badges and clickable
  links. Contradicted claims also carry `evidence_quote` — the exact
  contradicting evidence sentence ("Evidence says: …"), i.e. the corrective
  answer straight from the source. Web passages are reranked with the same
  cross-encoder used for documents.
- **Abstention**: claims with no retrieved evidence above the relevance floor
  are `unsupported` with `abstained: true` and a "no evidence retrieved" note —
  the system never guesses.
- **Known limitation**: NLI verdicts can over-entail on partial matches (e.g. a
  page about Lyon containing "capital … France" can make "Lyon is the capital of
  France" look supported). The evidence sentence is always shown for human
  checking, and the claim-eval sheet (`eval_claims.py`) quantifies this.

### Hybrid judge & feedback (B7.5 Tier 4)

- **LLM-judge routing** (`judge_uncertain`, auto-on with `OPENAI_API_KEY`):
  only claims whose NLI confidence falls in the uncertain band
  (`HALU_JUDGE_CONF_LOW/HIGH`, default 0.50–0.75) or that abstained with
  evidence are sent to the LLM with their evidence passages — capped at
  `HALU_JUDGE_MAX_CLAIMS` (3) per answer. Claims are tagged `judged_by: nli|llm`
  with one-line reasoning; graceful skip without a key.
- **Feedback loop**: 👍/👎 on the aggregate and each claim verdict →
  `POST /api/ml/feedback` → `data/processed/feedback_log.jsonl` (gitignored).
  `python src/models/tune_thresholds.py` grid-searches the NLI thresholds
  against labeled rows (report-only; `--apply` writes
  `data/processed/verdict_thresholds.json`, read at runtime).
- **Evals**: `eval_claims.py --from-feedback` exports feedback claims for
  labeling; `--tier3 queries.csv` measures citation recall@k over the index.
- **Hardening**: per-IP rate limits on `/verify`, `/index`, `/judge`,
  `/feedback` (slowapi, env-tunable); `/verify` NLI runs under the inference
  lock; input bounds unchanged.

## Deployment notes

- Backend container: see `docs/b6-reproducibility.md` §3 (CPU Dockerfile).
- Web: `pnpm run build` then `pnpm start`; the dashboard is `force-dynamic`
  (reads artifacts at request time — deploy with the artifacts folder mounted
  one level above the web root, as in the repo layout).
