# HaluRISC — AI Agent Guidelines & Optimization Protocol (AGENTS.md)

This document specifies mandatory rules, design patterns, prompt caching optimizations, and documentation references for AI agents (and developers) working on the HaluRISC codebase.

---

## 0. Source of Truth (Read First)

- **`blueprint.md`** — research design, scientific claims, scope. **Version A is frozen on branch `version-A`; Version B (publication) work runs on branch `version-B` only.** Do not mix scopes or alter Version A artifacts after the gate push without a documented reason.
- **`roadmap.md`** — phased implementation plan. Treat as guidance, NOT gospel: version numbers, URLs, and library claims in it can be stale. Always prefer the **latest stable versions actually available/installed** (check `.venv` and PyPI before pinning anything).
- **`AGENTS.md`** (this file) — binding agent rules. On conflict with `roadmap.md`, this file and `blueprint.md` win.

---

## 1. LLM Prompt Caching Maximization Protocol

To minimize API latency and reduce LLM token costs by up to **90–98%** (e.g., DeepSeek v4 Flash cache hit at $0.0028/M tokens vs $0.14/M; GPT 5.6 Luna cached input at $0.02/M vs $0.20/M), all LLM integration code MUST adhere to the following rules:

### 1.1 Static System Prompt Prefixing
- **Rule:** Place all static instructions, system definitions, persona rules, and tool schemas at the **very beginning** of the prompt or message array.
- **Rule:** Never insert dynamic variables (timestamps, user IDs, query-specific data) into the middle of the system prompt.
- **Minimum Prefix Threshold:** Keep the static system prefix above 1,024 tokens to guarantee prefix cache hits on DeepSeek v4 Flash and OpenAI models.

```typescript
// ✅ CORRECT: Static prefix remains 100% identical across requests -> High Cache Hit Rate
const SYSTEM_PROMPT = `
You are HaluRISC, an AI hallucination risk analyst... [Static Instructions > 1024 tokens]
Tool Definitions: [Static Tool Schemas]
Rules: [Static Rules]
`;

const messages = [
  { role: "system", content: SYSTEM_PROMPT },
  ...userConversationHistory, // Dynamic user inputs placed strictly AFTER static system prompt
];

// ❌ INCORRECT: Dynamic variables inside system prompt destroy context caching
const BAD_SYSTEM_PROMPT = `You are HaluRISC. Time: ${new Date().toISOString()}. User: ${userId}...`;
```

### 1.2 Tool Definition Standardization
- Standardize tool definitions across calls using `zod` schemas.
- Do not dynamically generate tool schemas per request; export a static `tools` object.

### 1.3 Target API Benchmarks & Caching Multipliers
- **DeepSeek-V4-Flash-0731:** $0.14/M input (cache miss) vs **$0.0028/M input (cache hit — 50x discount)**.
- **GPT-5.6-Luna:** $0.20/M input (cache miss) vs **$0.02/M input (cache hit — 10x discount)**.

---

## 2. assistant-ui Documentation & Integration Map

When building or modifying the frontend chat interface, refer to these canonical documentation sources and architectural rules:

### 2.1 Official Documentation Reference
- **Primary Docs:** [https://www.assistant-ui.com/docs](https://www.assistant-ui.com/docs)
- **Component Registry:** `https://r.assistant-ui.com/styles/default/{name}.json`
- **CLI Scaffolding:** `npx assistant-ui@latest create web`
- **Component Installation:** `npx shadcn@latest add @assistant-ui/thread`

### 2.2 Core Runtime Stack
- **Next.js Integration Adapter:** `@assistant-ui/react-ai-sdk` via `useChatRuntime` hook.
- **Streaming Provider:** Vercel AI SDK (`ai` package) using `streamText()` and `toDataStreamResponse()`.
- **API Route Location:** `app/api/chat/route.ts` (Next.js App Router).

### 2.3 Generative UI & Toolkits (Modern API)
- Use `defineToolkit` with the `"use generative"` directive in `app/toolkit.tsx`.
- **Do NOT use deprecated APIs** (`makeAssistantToolUI`, `makeAssistantTool`).
- Wrap Next.js config with `withAui()` plugin from `@assistant-ui/next`.
- Toolkit `execute: externalTool()` runs server-side in `/api/chat`; `render` functions are client components and MUST receive a `result` matching the tool output schema.

### 2.4 Thread UI Rule
- The chat page MUST use the assistant-ui `Thread` primitives (`ThreadPrimitive.*` from `@assistant-ui/react`) wired to `/api/chat` via `useChatRuntime` + `AssistantRuntimeProvider`. Do not hand-roll a custom chat message loop that bypasses the streaming route.
- Assistant-UI source lives in the repo (open-code philosophy): custom styled components go in `components/assistant-ui/`.

---

## 3. Open-Code Philosophy & LLM Assisting

assistant-ui follows the **shadcn/ui "Open Code" philosophy**:

1. **Source in Your Repo:** `assistant-ui` components live directly inside your `components/assistant-ui/` directory. They are NOT hidden inside `node_modules`.
2. **LLM Assistance Advantage:** Because the component source code is in your codebase, LLMs (Claude, Cursor, Copilot) can read, understand, and directly refactor the components, Tailwind classes, and Radix primitives without guessing abstract library APIs.
3. **Customization Rule:** Modify files in `components/assistant-ui/` directly to match the HaluRISC dark theme, blue-violet accent gradients, and typography without fear of breaking updates.

---

## 4. Codebase Architecture & Boundary Rules

- **Next.js (`web/`):** Serves as the Backend-for-Frontend (BFF) on port 3000. Handles UI rendering, static pages, OpenAI API key security, and streaming (`/api/chat`).
- **FastAPI (`src/api/main.py`):** Pure Python ML inference server on `http://127.0.0.1:8000`. Serves XGBoost predictions, feature extraction, SHAP explanations, and LLM-as-judge runs (`/predict`, `/explain`, `/judge`, `/health`).
- **Boundary Rule:** the API LOADS artifacts (`artifacts/models/*`) and feature models at startup; it NEVER trains. All training happens offline via scripts in `src/models/`.
- **Next.js Proxy:** All requests from frontend to FastAPI MUST route through `next.config.ts` rewrites (`/api/ml/:path*` → `http://127.0.0.1:8000/:path*`) to prevent CORS issues.
- **Python ML Pipeline:** Python code MUST run in `.venv` (Python 3.12). NumPy is at 2.4.6 (latest stable, 2026-era stack) — do NOT downgrade; `numpy<2` notes in older docs are stale. Torch is the CUDA 12.8 build (`2.11.0+cu128`) for the RTX 3060 (6 GB VRAM); keep it GPU-capable.

### Repo layout

```
HaluRISC/
├── blueprint.md          # research source of truth (Version A active)
├── roadmap.md            # implementation plan (guidance only)
├── requirements.txt      # EXACT pins only (==)
├── src/
│   ├── data/             # download.py (HaluEval, RAGTruth), prepare.py (splits)
│   ├── features/         # extract_features.py (core) + entity/nli/semantic modules
│   ├── models/           # train_baselines.py, train_pipeline.py (full protocol)
│   ├── explain/          # shap_analysis.py
│   └── api/              # FastAPI app (main.py)
├── web/                  # Next.js + assistant-ui frontend
├── data/raw/             # gitignored raw datasets
├── data/processed/       # parquet (cleaned data, feature matrices)
├── artifacts/
│   ├── models/           # joblib model, scaler, params.json (gitignored)
│   ├── figures/          # all plots (PNG/PDF)
│   └── results/          # metric tables (CSV/JSON)
└── tests/                # pytest suites
```

---

## 5. Environment & Secrets Configuration (.env Placement)

- **Template Reference File:** Root [**.env.example**](file:///d:/ML/HaluRISC/.env.example) contains all environment variable keys and descriptions.
- **Next.js Frontend Environment:**
  - **Location:** `web/.env.local`
  - **Keys:** `OPENAI_API_KEY`, `OPENAI_MODEL`, `NEXT_PUBLIC_APP_URL`, `NEXT_PUBLIC_ML_API_URL`
  - **Security Rule:** Server-only variables (`OPENAI_API_KEY`) must NEVER start with `NEXT_PUBLIC_`. They are strictly accessed in `app/api/chat/route.ts` (server side).
- **FastAPI Python Backend Environment:**
  - **Location:** Root `.env` or system environment variables loaded via `python-dotenv`.
  - **Keys:** `FASTAPI_HOST`, `FASTAPI_PORT`, `FASTAPI_DEBUG`, `OPENAI_API_KEY` (for `/judge`), `OPENAI_MODEL`, `DEEPSEEK_API_KEY` (optional fallback judge), `HALU_API_DEVICE` (`cuda`|`cpu`, default `cpu`; `cuda` auto-falls back to CPU if torch has no CUDA, models load fp16 on CUDA), `HALU_API_PRELOAD` (default `1`; `0` skips the startup preload of heavy spaCy/NLI/SBERT models), `HALU_XGB_DEVICE` (`cuda`|`cpu`|`auto`; set `cpu` in Colab so saved XGBoost models are portable across platforms — CUDA-trained boosters do not unserialize cross-platform).
- **Git Security Rule:** Neither `.env` nor `.env.local` are ever committed to Git (`.gitignore` protects both).

### 5.1 Dependency Pinning Rule
- `requirements.txt` MUST contain exact pins (`==`), never `>=`/`~=` (blueprint A18).
- When adding a package, install the **latest stable version** that resolves against the installed stack, then pin the resolved version.
- Installed versions in `.venv` take precedence over stale roadmap pins (e.g., scikit-learn 1.9, xgboost 3.4, pandas 3.0 are correct as installed — do NOT downgrade to older roadmap values).

---

## 6. Mandatory ML Experiment Protocol (Version A)

Grading-critical rules (blueprint A9–A10, roadmap Phases 4–5). All training scripts MUST follow these:

- **Splits:** 70/15/15 stratified by label; split indices saved to `artifacts/split_indices.json` + `.npy`. Never re-split using only a seed.
- **Cross-validation:** 5-fold stratified CV on train for tuning and model comparison.
- **Hyperparameter tuning:** randomized search (30–50 iterations, 5-fold CV) for XGBoost over `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`. Report best params.
- **Seeds:** repeat every experiment with seeds **42, 123, 456** and report mean ± std. Single-seed results are not acceptable.
- **Calibration:** fit the calibrator (Platt/sigmoid default) on the **validation** split only — never the test split. Compare Platt vs isotonic on test.
- **Metrics:** Precision, Recall, F1, AUROC, PR-AUC, MCC + ECE, Brier score, reliability diagram.
- **Statistics:** McNemar's test (XGBoost vs best baseline), bootstrap 95% CIs (1,000 resamples) for F1/AUROC, Wilcoxon signed-rank across seeds.
- **Ablations:** remove each feature group one at a time (7 groups × 3 seeds), report F1/AUROC deltas.
- **Artifacts:** every run MUST save model `.joblib`, scaler, params JSON, and result tables to `artifacts/`.
- **External bar:** cite published HaluEval QA detection numbers; run zero-shot evaluation on the RAGTruth QA holdout.
- **Heuristic baseline:** `1 - overlap_answer_context` (overlap threshold tuned on validation).

---

## 7. Verification Commands (run after relevant changes)

Python (from repo root, use the venv interpreter explicitly):

```powershell
& .venv\Scripts\python.exe -m pytest tests -v            # tests
& .venv\Scripts\python.exe -m uvicorn src.api.main:app --port 8000   # API (no --reload; see §5 for HALU_API_DEVICE/HALU_API_PRELOAD)
& .venv\Scripts\python.exe src\data\download.py          # dataset acquisition
& .venv\Scripts\python.exe src\data\prepare.py           # splits
& .venv\Scripts\python.exe src\features\extract_features.py   # feature matrix
& .venv\Scripts\python.exe src\models\train_pipeline.py  # full experiment protocol
```

Web (from `web/`, package manager is **pnpm** — do not use npm; Next.js 16 + Tailwind v4 + assistant-ui 0.15 + AI SDK v7):

```powershell
pnpm install    # install deps (pnpm-workspace.yaml holds settings)
pnpm run dev    # dev server on http://localhost:3000
pnpm run build  # production build (must pass before finishing web work)
pnpm run lint   # eslint (flat config, eslint 9)
```

### 7.1 Colab Training Workflow (optional, for heavy compute)

- `colab/HaluRISC_Training.ipynb` runs the full pipeline (features → tuning → calibration → SHAP → RAGTruth) on a Colab GPU and saves a `halurisc_artifacts_<date>.zip` to Google Drive.
- Upload `colab/halurisc_src.zip` (regenerated with `Compress-Archive` from `src/`, `requirements.txt`, `colab/`) when the notebook asks.
- After training, download the Drive zip and unzip **at the repo root** so `artifacts/*` and `data/processed/features_full.parquet` land in place. The API and web dashboard then load the real artifacts.
- Keep `src/` scripts Colab-compatible: paths must be repo-root-relative, no hardcoded absolute Windows paths, no reliance on the local `.venv` at import time.

---

## 8. Web Frontend Rules

1. **Chat page** MUST use assistant-ui `Thread` primitives + `/api/chat` (Vercel AI SDK streaming). Never rebuild a custom chat loop.
2. **`next.config.ts`** MUST wrap config with `withAui()` from `@assistant-ui/next` and keep the `/api/ml/:path*` → FastAPI rewrite.
3. **No fabricated data:** dashboard/experiment numbers MUST come from `artifacts/results/*` (read via fs in a server component or generated JSON). Never hardcode fake metrics or model rows.
4. **API contract:** frontend consumes `POST /api/ml/predict` → `{risk_score, calibrated_score, label, thresholds, latency_ms, model_version, feature_version, warning, features}` and `POST /api/ml/explain` → `{top_features[], base_value}`. Keep field names stable.
5. **Theme:** dark theme, blue-violet accent gradients, `glass-panel`/`gradient-text` utility classes defined in `app/globals.css`.
