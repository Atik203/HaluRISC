# HaluRISC — Detailed Implementation Roadmap

**Goal:** Finish and freeze a defensible Version A course project first, then extend it on `version-B` into a publication study with leakage-controlled cross-domain evaluation, calibration under shift, explanation reliability, and a research-focused web artifact. Heavy experiments run in Colab Pro; the local RTX 3060 Laptop GPU (6 GB VRAM) and 32 GB RAM support inference, profiling, UI development, and demos.

**Convention:** items marked `[verified 2026]` were checked against current web/PyPI info in July 2026.

> **IMPLEMENTATION STATUS (updated 2026-08-06):** ✅ DONE | 🔶 PARTIAL | ⬜ TODO
>
> Version A integrity repair is complete: leakage-free group split, corrected artifacts (XGBoost F1 0.9842 / AUROC 0.9982), per-seed metrics, manifest, LLM-judge, API + pytest verified, and the Colab zip now produces portable models (HALU_XGB_DEVICE=cpu). Version A is frozen on `version-A`; Version B work runs on `version-B` and B1 (unified dataset layer) is complete. Remaining: B2 corrected baselines and shortcut controls next.

### Operating rule

1. Fix Version A first.
2. Run heavy training and evaluation in Colab Pro.
3. Download the corrected artifacts to the repository root.
4. Verify the API and UI locally on the RTX 3060 laptop.
5. Push the final Version A correction to `version-A`.
6. Only then implement Version B on `version-B`.

---

## 1. Tech Stack at a Glance

| Layer | Tool | Version (2026) | Why / Source |
|---|---|---|---|
| Language | Python | 3.12 or 3.13 | All libs support it; XGBoost 3.3.0 supports 3.12–3.14, scikit-learn 1.7 supports 3.10–3.13 `[verified 2026]` |
| Tabular ML | scikit-learn | 1.7.x `[verified]` | LR, RF, CV, tuning, calibration, metrics |
| Gradient boosting | XGBoost | 3.3.0 `[verified]` (PyPI, Jun 2026) | Final model; `XGBClassifier` scikit-learn API |
| NLI model | HuggingFace `cross-encoder/nli-deberta-v3-base` | primary `[verified]` | SNLI+MultiNLI, outputs contradiction/entailment/neutral |
| NLI fallback | `cross-encoder/nli-MiniLM2-L6-H768` | if DeBERTa too slow | ~1/5 size of DeBERTa, ~90%+ of its accuracy |
| Deep learning | PyTorch | 2.11.0+cu128 (CUDA 12.8 wheel via pytorch index, pinned in requirements.txt) | NLI CrossEncoder + SBERT embeddings; fp16 on GPU for the 6 GB RTX 3060 |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | latest 4.x | Semantic similarity features |
| NER | spaCy `en_core_web_sm` | spaCy 3.9 | Entity overlap features |
| Explainability | SHAP | 0.46+ | `TreeExplainer` for XGBoost |
| Statistics | scipy + statsmodels | latest | Bootstrap CIs, McNemar's test |
| Backend API | FastAPI + uvicorn | ≥0.130 (needs Python ≥3.10) `[verified]` | Pydantic v2 validation, ~50x faster validation than v1 |
| Frontend | Next.js (App Router) + React 19 + TypeScript | Next.js latest, React 19 `[verified]` | SSR, API routes, BFF layer for OpenAI + FastAPI proxy |
| Chat UI | assistant-ui | `@assistant-ui/react` + `@assistant-ui/react-ai-sdk` | Production AI chat interface, streaming, Generative UI |
| AI SDK | Vercel AI SDK | `ai` + `@ai-sdk/openai` | Streaming, tool calling, SSE plumbing for chat |
| LLM API | GPT 5.6 Luna | `gpt-5.6-luna` (OpenAI, existing credits) | Conversational explanations + LLM-as-judge baseline |
| Styling | Tailwind CSS v4 | v4 stable `[verified]` | Utility-first, shadcn/ui official support |
| UI components | shadcn/ui | latest (Next.js install path) `[verified]` | Professional components, judge-friendly polish |
| Charts | Recharts + lucide-react | latest | Calibration curves, bars; custom SVG for risk gauge |
| Data | pandas + numpy + pyarrow | latest | Feature matrix to Parquet, fast I/O |
| Artifacts | joblib | latest | Save model, scaler, split indices |

---

## 2. Project Folder Structure

> **Note:** this is the project layout currently used by the implementation. Large raw/processed datasets, model files, figures, and result artifacts remain gitignored and must be regenerated or restored from the Colab export.

```
HaluRISC/
├── blueprint.md            # research blueprint (source of truth)
├── proposal.md             # supervisor proposal (Markdown copy)
├── roadmap.md              # this file
├── report/                 # LaTeX proposal
│   ├── proposal.tex        # main proposal source
│   ├── uiu.png             # university logo
│   └── out/                # compiled PDF output
├── data/
│   ├── raw/halueval/       # qa_data.json (downloaded)
│   ├── raw/ragtruth/       # RAGTruth processed parquet/json
│   └── processed/          # cleaned parquet, feature matrix, splits
├── src/
│   ├── data/               # download + prep scripts
│   ├── features/           # extract_features.py + per-group modules
│   ├── models/             # train.py, tune.py, calibrate.py
│   ├── evaluate/           # metrics.py, stats.py, ablation.py
│   ├── explain/            # shap_analysis.py
│   └── api/                # FastAPI app (main.py, schemas.py)
├── web/                    # React dashboard (Vite)
├── notebooks/              # optional exploration
├── artifacts/
│   ├── models/             # joblib model, scaler, params.json
│   ├── figures/            # all plots (PNG)
│   └── results/            # metric tables (CSV/JSON)
└── requirements.txt        # pinned versions
```

---

## 3. Phase 0 — Environment Setup (Day 1) — ✅ DONE (venv Python 3.12, pinned requirements.txt, pnpm + Next 16 + Tailwind v4 + AI SDK v7)

1. Install Python 3.13 from [python.org](https://www.python.org/downloads/) (check: `python --version`).
2. Create a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1        # Windows
   ```
3. Install core packages:
   ```powershell
   pip install pandas numpy scikit-learn==1.7.* xgboost==3.3.0 scipy statsmodels
   pip install shap joblib pyarrow
   pip install spacy && python -m spacy download en_core_web_sm
   pip install sentence-transformers
   pip install fastapi "uvicorn[standard]" pydantic httpx
   ```
4. Git + `.gitignore`: ignore `.venv/`, `data/raw/` (large), `node_modules/`, `artifacts/`, `__pycache__/`.

---

## 4. Phase 1 — Data Acquisition (Week 1) — 🔶 IMPLEMENTED, MANUAL AUDIT PENDING

### 4.1 HaluEval (primary, train/tune/test)

- **Where:** GitHub `RUCAIBox/HaluEval` `[verified]`
  - Repo: `https://github.com/RUCAIBox/HaluEval`
  - Direct raw file: `https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json`
- **What is inside `qa_data.json`:** 10,000 QA entries; each has `question`, `knowledge` (facts = context), `answer` (correct), `hallucinated_answer`, and `hallucination_label` (binary).
- **How to build the binary dataset:** for each entry, use `knowledge` as **context**, `answer` as the response, and `hallucination_label` as the label. If you want both variants per question (correct + hallucinated answer), create two rows per question and label accordingly — decide once and document it.
  - **⚠ Decide the row layout up front** — it changes dataset size and NLI cost: one row per question = **10,000 rows / 20K NLI pairs**; two rows per question = **20,000 rows / 40K NLI pairs** (~2–4 hrs CPU instead of 1–2). The proposal quotes "~10,000 samples", so the **two-rows layout means updating that number to ~20,000 everywhere** (proposal §Dataset, paper dataset table).
- **License:** the official repository currently states an MIT License. Retain attribution, citation, and the exact source revision in the manifest; do not assume that local cached data should be committed.
- **Manual audit (mandatory):** randomly sample 50 entries, read each, and record whether the label looks right. Keep the audit notes file — it goes in the paper's dataset-limitations section.

### 4.2 RAGTruth (external validation)

- **Where:** official GitHub `ParticleMedia/RAGTruth` or ready-made HuggingFace mirror `wandb/RAGTruth-processed` `[verified]`.
- **What it is:** ~18,000 naturally generated RAG responses (GPT-3.5/4, LLaMA-2, Mistral) with **human word-level hallucination spans** (Niu et al., ACL 2024).
- **How to use here:** download the QA portion; derive a binary label (`has_hallucination = any annotated span`). Keep ~1,000–2,000 samples untouched — this is the zero-shot validation set for the final model.
- **Paper framing:** results on HaluEval = main results; RAGTruth = "does the model generalize to natural responses?" — exactly what the proposal promises.

---

## 5. Phase 2 — Preprocessing & Splits (Week 1–2) — ✅ CORRECTED GROUP SPLIT DONE (leakage-free)

Rules (encode in `src/data/prepare.py`, cache to `data/processed/qa_clean.parquet`):

- Normalize whitespace; tokenize with simple split (or spaCy).
- Keep original casing for NER/NLI; lowercase only for lexical features.
- Drop invalid rows (empty answer, corrupted text).
- Empty/missing context → lexical overlap = 0, NLI = neutral (0.33/0.33/0.33), document the rule.
- Check class balance. If not ~50/50, keep it (real-world) and set `scale_pos_weight` for XGBoost.
- **Split:** 70% train / 15% validation / 15% test, stratified at the original-question group level using `item_idx`. Both the correct and hallucinated answer for a question must remain in one partition.
- **Validation:** assert that `item_idx` has exactly one split value; save group counts, label counts, and a leakage report.
- **Save split indices** with `np.save("artifacts/split_indices.npy")` — reproducibility requires exact indices, group identifiers, and a split hash, not just a seed.
- **Required repair:** the existing row-level split places 4,682 source questions across multiple partitions. Rebuild it before any final metric or paper claim.

---

## 6. Phase 3 — Feature Extraction (Week 2–3) — 🔶 IMPLEMENTED, RERUN AFTER SPLIT REPAIR

One module per group in `src/features/`. Output: a single DataFrame, cached to `data/processed/features_qa.parquet` (extract once, reuse everywhere).

| # | Group | Features (exact names) | Library / How |
|---|---|---|---|
| 1 | Length/style | `n_chars`, `n_words`, `n_sentences`, `avg_word_len` | `re`, `nltk`-free simple splitter |
| 2 | Lexical overlap | `overlap_answer_context`, `overlap_answer_question`, `jaccard_ans_ctx`, `jaccard_ans_q` | token-set intersection ratios |
| 3 | Entity overlap | `n_entities_answer`, `n_entities_context`, `entity_overlap_ratio`, `novel_entity_ratio` | spaCy `en_core_web_sm` NER |
| 4 | NLI consistency | `nli_ctx_entails_ans`, `nli_ctx_contradicts_ans`, `nli_ctx_neutral_ans` + same 3 reversed | `sentence_transformers.CrossEncoder("cross-encoder/nli-deberta-v3-base")` |
| 5 | Numeric | `n_numbers_answer`, `n_numbers_context`, `number_overlap_ratio`, `novel_numbers` | regex for numbers/dates/percentages |
| 6 | Hedging | `hedge_count`, `hedge_density` | lexicon regex: maybe, might, likely, possibly, probably, could, seems, I think, uncertain… |
| 7 | Semantic similarity | `cosine_ctx_ans`, `cosine_q_ans` | `sentence-transformers/all-MiniLM-L6-v2` |

**NLI details (the most important group):**

- Run **both directions**: (context, answer) and (answer, context) → 6 probability features per sample.
- Batch inference: `model.predict(pairs, batch_size=64)`, `show_progress_bar=True`.
- 10K samples × 2 directions = 20K pairs; DeBERTa on CPU ≈ 1–2 hours **once**. Cache the outputs.
- Fallback if download/speed is a problem: `cross-encoder/nli-MiniLM2-L6-H768` (much smaller, ~15ms/pair).

**Time each group** (lexical vs NLI vs semantic) — the latency breakdown feeds the paper's efficiency section.

---

## 7. Phase 4 — Modeling (Week 4) — ✅ CORRECTED RERUN DONE

`src/models/`:

1. **Heuristic baseline** (no training): risk = `nli_ctx_contradicts_ans` probability, threshold 0.5. Gives the "minimum bar". *If NLI is dropped (see Phase 3 fallback), switch the heuristic to `1 - overlap_answer_context` so the baseline still works without NLI.*
2. **Logistic Regression**: `Pipeline(StandardScaler, LogisticRegression(max_iter=2000))` — features must be scaled.
3. **Random Forest**: `RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight='balanced_subsample')`.
4. **XGBoost (final)**: `XGBClassifier(objective="binary:logistic", eval_metric="logloss", scale_pos_weight=<from class ratio>, early_stopping_rounds=30)`.

**Tuning (mandatory, ~30 min compute):** `RandomizedSearchCV` over 30–50 iterations, 5-fold CV on train:

- `max_depth`: [3, 4, 5, 6, 7]
- `learning_rate`: [0.01, 0.05, 0.1, 0.2]
- `n_estimators`: [100, 200, 300, 500]
- `subsample`: [0.7, 0.8, 0.9, 1.0]
- `colsample_bytree`: [0.7, 0.9, 1.0]  *(optional extra — the proposal/blueprint grid lists `scale_pos_weight` as the 5th parameter; both are fine, keep them consistent across docs)*

**Seeds (mandatory):** repeat every experiment with seeds 42, 123, 456 → report mean ± std.

**Artifacts:** `joblib.dump` model + scaler + feature names + best params (JSON) into `artifacts/models/`.

---

## 8. Phase 5 — Calibration & Evaluation (Week 4–5) — ✅ CORRECTED EVIDENCE PRODUCED (manual error-case review pending)

- **Calibration:** `CalibratedClassifierCV(estimator=best_xgb, method="sigmoid", cv="prefit")` fit on the **validation** set. Compare Platt (sigmoid) vs isotonic on the test set.
- **Metrics:** Precision, Recall, F1, AUROC, PR-AUC, MCC (classification); **ECE**, **Brier score**, reliability diagram (calibration).
- **ECE:** implement simple 10-bin equal-width ECE yourself (10 lines) or use `netcal`; plot reliability diagram.
- **Statistical tests (mandatory):**
  - McNemar's test — XGBoost vs best baseline on the same test predictions (`statsmodels.stats.contingency_tables.mcnemar`).
  - Bootstrap 95% CI (1,000 resamples) for F1 and AUROC.
  - Paired Wilcoxon across the 3 seeds.
- **Ablation:** remove each feature group one at a time (7 runs × 3 seeds), report F1/AUROC deltas table.
- **External comparison:**
  - Run the final calibrated model **zero-shot** on the RAGTruth QA holdout.
  - Cite published HaluEval QA detection numbers (SelfCheckGPT, IEEE TAI) as the external bar.
- Efficiency/cost: measure feature, model, SHAP, and total latency at p50/p95; record RAM/VRAM and artifact size. Compare against **GPT 5.6 Luna as LLM-as-judge** only with explicit sample size, token assumptions, and measured/estimated labels. Do not hardcode a 100x claim before the corrected measurements.
- Add answer-only, context-only, lexical/TF-IDF, and overlap controls to identify benchmark shortcuts.

---

## 9. Phase 6 — Explainability (Week 5) — 🔶 SHAP IMPLEMENTED, RELIABILITY PENDING

- `shap.TreeExplainer(xgb)` → `shap_values` on a test subsample (500–1,000 rows for speed).
- **Global:** SHAP summary plot (beeswarm) + mean-|SHAP| bar chart.
- **Local:** waterfall plots for 3 hand-picked cases: clear hallucination, clearly correct, borderline.
- Save figures to `artifacts/figures/` **and** raw SHAP values to JSON — the dashboard will re-render them.
- Required before publication: compare SHAP rankings with feature/group ablation and permutation importance; run controlled perturbation stability tests; report confidence intervals and failure cases. Do not use arbitrary FAC/PSI pass thresholds.

---

## 10. Phase 7 — Backend API (Week 6) — 🔶 IMPLEMENTED, LIVE INTEGRATION VERIFICATION PENDING

`src/api/` — FastAPI + uvicorn, Pydantic v2 `[verified]`.

**Load everything once at startup** (lifespan context manager): XGBoost model, scaler (if LR needed), NLI CrossEncoder, embedding model, spaCy NER, SHAP explainer. Inference must stay under ~200ms.

**FastAPI Endpoints (ML inference only):**

```
POST /predict   {question, context?, answer, domain:"qa"} 
                → {risk_score, label, calibrated_score, latency_ms, features}
POST /explain   {question, context?, answer}
                → {shap_values[], feature_names[], base_value, top_features[]}
POST /judge     {question, context, answer}
                → {judgment, confidence, reasoning, model:"gpt-5.6-luna"}
GET  /health    → {status:"ok", model:"xgb-calibrated", version:"A.1.0"}
```

**Next.js API Route (BFF — NOT FastAPI):**

```
POST /api/chat  {messages[]}  ← Vercel AI SDK streamText() + tool calling
                → SSE stream: GPT 5.6 Luna response with embedded tool results
                   (internally calls FastAPI /predict + /explain for real ML data)
```

- FastAPI: validate inputs (answer required, max lengths, meaningful error messages).
- FastAPI: CORS allows `http://localhost:3000` in dev.
- FastAPI: LRU cache on NLI/embedding calls to keep demo snappy.
- FastAPI: `& .venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000` (no `--reload` — uvicorn's file watcher restarts the server whenever repo files change).
- FastAPI: GPU/stability config via root `.env` — `HALU_API_DEVICE=cuda|cpu` (default `cpu`; `cuda` auto-falls back to CPU if torch has no CUDA; models load in fp16 on CUDA to fit 6 GB VRAM) and `HALU_API_PRELOAD=0` to skip the startup preload of heavy models (spaCy/NLI/SBERT).
- FastAPI: before publication, add a combined or cached analysis path so `/predict` and `/explain` do not repeat expensive NER/NLI/SBERT extraction for the same input.
- Next.js: `npm run dev` on port 3000; `next.config.ts` rewrites `/api/ml/*` → FastAPI.
- OpenAI key stored in `web/.env.local` — never exposed to browser.

---

## 11. Phase 8 — Frontend Dashboard (Week 7) — 🔶 IMPLEMENTED, DEMO HARDENING AND BUILD VERIFICATION PENDING

**Framework: Next.js + assistant-ui** (the official recommended stack for AI chat UIs).

Scaffold (`[verified 2026]` flow):

```powershell
# Step 1: Scaffold Next.js + assistant-ui in one command
npx assistant-ui@latest create web
cd web

# Step 2: Install additional dependencies
npm install recharts lucide-react zod
npm install ai @ai-sdk/openai               # Vercel AI SDK + OpenAI provider

# Step 3: Add shadcn/ui components
npx shadcn@latest add card button input textarea label tabs badge progress separator tooltip skeleton sonner

# Step 4: Add OPENAI_API_KEY to .env.local
# OPENAI_API_KEY=sk-...
```

**Key config files:**

```typescript
// next.config.ts — proxy /api/ml/* to FastAPI, enable Generative UI plugin
import { withAui } from "@assistant-ui/next";
const nextConfig = {
  async rewrites() {
    return [{ source: "/api/ml/:path*", destination: "http://127.0.0.1:8000/:path*" }];
  },
};
export default withAui(nextConfig);
```

**4 Pages (priority order):**

1. **💬 Chat Mode (THE SHOW-STOPPER — Page 1):** Full-screen `assistant-ui` Thread component powered by GPT 5.6 Luna + Generative UI.
   - Streaming responses with animated typing indicator.
   - `defineToolkit` with `"use generative"` directive: risk gauge + SHAP chart **rendered inside chat messages** as interactive React components.
   - Suggestion pills: *"Check this answer for hallucination"*, *"Compare two answers"*, *"Why is this answer risky?"*
   - Luna extracts Q/C/A, calls `/predict` + `/explain` via tool, explains XGBoost decisions in natural language.
   - **WOW factor:** watching risk gauge animate to 92% and SHAP chart appear inside a streaming AI response.

2. **📊 Analyze Mode (paper demonstration — Page 2):** Classic form (question, context, answer) → `POST /api/ml/predict` + `/api/ml/explain`.
   - **Risk gauge:** custom **SVG semicircular arc gauge** (animated needle + color zones green/yellow/red).
   - Calibrated score readout, verdict badge ("Low / Medium / High risk"), latency line.
   - SHAP top-5 feature contribution bars with plain-language labels.
   - 4 pre-baked example buttons (clearly hallucinated, correct, borderline, entity-mismatch).

3. **📈 Dashboard (experiment gallery — Page 3):** Model comparison table, calibration curve, ablation results, SHAP summary plot image, ROC/PR curves, **LLM vs XGBoost cost/accuracy comparison card**.

4. **ℹ️ About / Method (Page 4):** Animated pipeline diagram, feature group cards, team info.

**UI/UX Design:** Dark theme default; blue-violet accent gradient; glassmorphism cards; micro-animations on all interactions; Geist font (from assistant-ui scaffold); loading skeletons; error toasts (`sonner`). Risk gauge is the centerpiece visual on both Chat and Analyze pages.

Before Version A is final, verify that the Generative UI toolkit is actually registered and rendered, remove hardcoded research claims and stale package versions, and pass `pnpm run lint` and `pnpm run build`. Version B should add mobile layout, keyboard focus, accessible chart alternatives, `aria-live` result states, reduced-motion support, and an offline/precomputed demo path.

**Serving:**
```powershell
# Development (2 terminals)
& .venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000  # FastAPI (no --reload)
npm run dev  # Next.js on http://localhost:3000

# Production
npm run build  # builds Next.js static + server
# Run both as separate processes or use Docker multi-stage build
```

---

## 12. Phase 9 — Integration, Demo & Delivery (Week 8) — 🔶 PARTIAL (app runs end-to-end; pending: demo rehearsal script, backup video, optional Docker)

- Build frontend, serve from FastAPI, run one-command demo.
- Pre-bake 4 demo examples: grounded, unsupported entity/date/number, borderline, and empty/weak evidence — plus live input.
- Rehearse a 5-minute script: 1 min problem → 1 min approach → 1 min live demo → 1 min results → 1 min "why it matters".
- Optional: two-stage Dockerfile (node build → python runtime) for portability.
- Final checks: corrected grouped split, audit file, `requirements.txt` pinned, README with setup/run commands, split/group hashes, tests, API smoke test, `pnpm run lint`, `pnpm run build`, and clean-clone reproduction.

---

## 13. Phase 10 — Reproducibility & Paper Mapping — 🔶 PARTIAL

| Paper section | Artifact |
|---|---|
| Dataset | `data/processed/*.parquet` + completed manual audit + dataset/license manifest |
| Features | `src/features/` + feature table CSV |
| Experiments | `artifacts/results/*.csv` (all 3 seeds, corrected grouped split, shortcut controls) |
| Calibration | ECE/Brier + reliability diagram PNG |
| Statistics | McNemar output + bootstrap CIs |
| Efficiency | latency breakdown table + cost estimate |
| Explainability | SHAP figures + JSON |
| Reproducibility | pinned requirements, saved group-aware splits, hashes, saved models, run manifest |

**Before submission — verify every citation (Week 8):**
- Click through each DOI / URL in `report/proposal.tex` and the paper — the 2026 entries (`ijert`, `ieeeTai`, `multimedia`, `spikescore`) in particular must resolve. A broken DOI in a project about hallucination is a credibility hit.
- Confirm the two live claims behind the roadmap `[verified]` markers: XGBoost `3.3.0` and the `cross-encoder/nli-deberta-v3-base` model card still match before you install (PyPI/HF links in §16).

**Roadmap completion checklist before Version A is final:**
- [ ] All corrected `data/processed/*.parquet`, `artifacts/models/*`, and `artifacts/results/*` regenerated (gitignored by design; reproducible via Colab)
- [x] `requirements.txt` fully pinned (with the PyTorch CUDA index directive)
- [ ] Group-aware split indices, source-group report, and split hash saved
- [ ] Completed 50-sample manual audit saved
- [ ] Corrected three-seed metrics, shortcut controls, and calibration artifacts saved
- [ ] Reviewed error-analysis cases and explanation-faithfulness notes saved
- [x] `report/out/paper.pdf` currently compiles from source
- [ ] Every citation, license, and numerical claim verified against corrected outputs
- [ ] README setup/run commands, clean clone, API smoke test, `pnpm run lint`, and `pnpm run build` verified

---

## 14. Version B Publication Roadmap — LOCKED UNTIL VERSION A PASSES

Version B is implemented only on `version-B` after the corrected Version A state is committed and pushed to `version-A`. Heavy experiments use Colab Pro. The local RTX 3060 Laptop GPU (6 GB VRAM) and 32 GB RAM are for inference, profiling, UI development, screenshots, and the live demo.

### B0 — Version A integrity gate

Environment: Colab Pro for the rerun; local machine for verification.

1. Change `src/data/prepare.py` to split original questions by `item_idx`, not individual rows.
2. Add a hard assertion that every `item_idx` has one and only one split.
3. Manually review 50 unique source questions and save `data/processed/audit_50_samples.json`.
4. Regenerate `qa_clean.parquet`, `features_full.parquet`, split files, models, calibrators, figures, and results.
5. Add answer-only, context-only, lexical/TF-IDF, and overlap shortcut controls.
6. Record the actual NLI checkpoint, feature version, split hash, Git commit, hardware, CUDA mode, RAM, and VRAM in a manifest.
7. Manually review the sampled FP/FN cases and correct the heuristic categories.
8. Run Python tests, a live artifact-backed API smoke test, `pnpm run lint`, `pnpm run build`, and a clean-clone test.
9. Rewrite Version A paper claims from the corrected artifacts.
10. Push the final correction to `version-A`.

Exit condition: no cross-split source groups, completed audit, corrected artifacts, verified paper claims, and all required local checks passing.

### B1 — Unified data schema — ✅ DONE (2026-08-06)

Status: canonical schema (`src/data/schema.py`), label mappings
(`src/data/mappings.py`), dataset registry/license manifest
(`src/data/registry.py`), official RAGTruth + FaithBench downloaders, and
`src/data/prepare_unified.py` are implemented; 32 new unit tests pass; the
local run produces 38,540 canonical rows (HaluEval 20,000 / RAGTruth 17,790 /
FaithBench 750) with byte-identical deterministic output. Mapping report:
`artifacts/results/dataset_mapping_report.json`; license manifest:
`artifacts/results/dataset_license_manifest.json`. RAGTruth counts match the
ACL 2024 paper exactly (17,790 responses, 2,965 sources, 7,664 hallucinated,
14,289 spans).

Environment: Colab Pro for downloads and preprocessing; local machine for schema tests.

1. Preserve `source_dataset`, `source_group_id`, `task`, `domain`, `question`, `context`, `answer`, `label`, `span_annotations`, `generator_model`, and official split metadata.
2. Use HaluEval QA as the training and in-domain benchmark.
3. Use RAGTruth QA as the primary external benchmark; group by `source_id` because one source can produce multiple responses.
4. Use FaithBench as a difficult summarization stress test with a documented label mapping.
5. Keep HalluLens and TRIVIA+ optional; they are not Version B blockers.
6. Do not bundle CC BY-NC-SA FaithBench or primarily CC BY-NC HalluLens data in the repository. Ship download instructions, citations, hashes, and license notes instead.
7. Save a dataset mapping report with counts, missing fields, label distributions, and excluded records.

Exit condition: every dataset has a documented schema, label definition, source-group rule, license record, and reproducible preprocessing script.

### B2 — Corrected baseline and artifact controls — ✅ DONE (2026-08-06, local GPU run)

Status: `src/models/run_b2_baselines.py` implements all nine baselines
(majority, overlap heuristic, TF-IDF all/answer/context, NLI-only, LR, RF,
tuned XGBoost) with grouped 5-fold CV tuning keyed by `item_idx`,
train-only TF-IDF fitting, validation-only thresholds (0.5 models / tuned
overlap), seeds 42/123/456, per-seed predictions and confusion matrices,
McNemar/bootstrap/Wilcoxon, and the leakage-removal impact report. 7 new
unit tests; full local run ~7 min on RTX 3060. Artifacts:
`artifacts/results/b2/` + `artifacts/models/b2/`. Key result: XGBoost
F1 0.9857 / AUROC 0.9980; answer-only TF-IDF control reaches 0.9224
(answer-style shortcut), context-only has zero signal; leakage removal
costs only Δ−0.003 F1 vs the historical leaky 0.9886.

Environment: Colab Pro.

1. Train the existing 26-feature pipeline on grouped HaluEval data.
2. Run majority, overlap, lexical/TF-IDF, answer-only, context-only, NLI-only, Logistic Regression, Random Forest, and XGBoost baselines.
3. Keep CatBoost optional; do not add transformer fine-tuning or hidden-state probing.
4. Repeat stochastic experiments with seeds 42, 123, and 456.
5. Save per-seed metrics, per-example predictions, confusion matrices, and model artifacts.
6. Report whether performance falls after leakage removal; treat a large drop as an important result, not a failure.

Exit condition: corrected in-domain results and shortcut controls are available before any external-dataset claim is written.

### B3 — Cross-domain robustness — 🔶 IMPLEMENTED, EXECUTION MOVED TO COLAB

Status: `src/models/run_b3_cross_domain.py` is implemented and unit-tested
(10 tests): predeclared subsets (RAGTruth QA official test primary, all
tasks/splits secondary, FaithBench locked), 26-feature extraction on external
data (cached, keyed by unified-parquet hash), zero-shot B2 XGBoost with fixed
threshold 0.5, source-group bootstrap CIs (1000), subgroup metrics with
minimum-size rules (100 rows / 20 groups), span-type and context/answer-length
subgroups, FaithBench label sensitivity, transfer-failure analysis, error-case
sampling, and 4 figures. A local run was attempted on the RTX 3060 laptop but
the external feature extraction (~18.5K rows) overheated the GPU (~100 °C) and
was abandoned. **Decision: all heavy training/extraction now runs in Colab.**
Colab notebook cells 7d (B1 unified build), 7e (B3 run), 7f (B3 display) were
added; `colab/halurisc_src.zip` regenerated with the B3 runner and tests.
B2 must run before B3 in the same Colab session so the CPU-portable boosters
(HALU_XGB_DEVICE=cpu) exist.

Environment: Colab Pro (T4/L4).

1. Evaluate the HaluEval-trained model zero-shot on RAGTruth QA.
2. Report performance by RAGTruth task type, source group, label type, and context length when available.
3. Evaluate FaithBench with source text as context and summary as answer.
4. Report both aggregate and subgroup metrics with bootstrap confidence intervals.
5. Keep zero-shot, lightly adapted, and retrained experiments as separate labels.
6. Never tune thresholds or preprocessing on an external test set.

Exit condition: one table and one figure clearly show in-domain versus out-of-domain performance and the sources of transfer failure.

### B4 — Calibration under distribution shift — 🔶 IMPLEMENTED, RUNS AFTER B3 IN COLAB

Status: `src/models/run_b4_calibration_shift.py` is implemented and
unit-tested (10 tests). Source calibration (Platt/isotonic fit on HaluEval
validation only, applied unchanged to HaluEval test / RAGTruth QA test /
other tasks / FaithBench) and target calibration (RAGTruth QA official train
-> official test with disjoint source groups, overlap removed and reported).
Metrics: ECE, adaptive ECE, Brier, NLL, calibration slope/intercept,
reliability curves, F1/AUROC at fixed 0.5. Subgroup calibration with minimum
sizes (100 rows / 20 groups) for task/split/domain/model/quality/context/
answer bins. Selection rule predeclared: Platt is the deployable default;
isotonic reported for comparison. Outputs:
`artifacts/results/b4/` + `artifacts/figures/b4/` + `artifacts/models/b4/`
(pure-sklearn calibrators — portable, no CUDA booster serialization; B4 adds
no heavy feature extraction, runs on CPU). Notebook cells 7g/7h added;
`colab/halurisc_src.zip` regenerated. **Run order in Colab: 7b (B2) -> 7d (B1)
-> 7e (B3) -> 7g (B4) -> 7i (artifact verification) -> 13 (manifest) -> 15
(package).** Local execution is possible only after B3 artifacts are
downloaded from the Colab run.

Pre-Colab audit fixes (2026-08-06): (1) `b3_external_features.parquet` cache
and `b3_error_cases.json` no longer carry raw external text — FaithBench
(CC BY-NC-SA) text never leaves the Colab VM; (2) saved B4 target calibrators
are now fit on the exact filtered calibration frame (overlap-removed) that
produced the metrics; (3) single-class subgroups are flagged descriptive-only
instead of crashing slope/intercept fitting; (4) `src/models/verify_artifacts.py`
(cell 7i) loads every B2/B3/B4 artifact and predicts before packaging,
preventing the historical post-download loading error; (5)
`colab/build_src_zip.py` rebuilds the source zip without `__pycache__`/`.pyc`;
(6) `make_manifest.py` now hashes B2/B3/B4 artifacts and unified records.
(7) **Drive caching of deterministic heavy artifacts** (new cells 5b/6b/7d.5,
`colab/drive_cache.py`): `features_full.parquet` and the B3 external-feature
cache are stored in `Drive/halurisc_cache/` and restored on later sessions
after verification — HaluEval features are checked against the freshly built
`qa_clean.parquet` (rows/sample_ids/labels/splits/leakage), the B3 cache only
when its unified-parquet hash matches; mismatches automatically fall back to
full extraction. Cell 6 (5–10 min) and B3 extraction (~15–40 min) are skipped
on repeat runs.
(8) **Full checkpoint/resume (L4-ready)**: Version A cell 7 (when selected)
now checkpoints root artifacts under `version_a/`; every B heavy phase also
checkpoints to `Drive/halurisc_cache/` immediately after finishing — B2
(7b.5), B3 features (7e), B3 results+figures (7e), B4 artifacts (7g).
Restore cells `7.0` / `7b.0` / `7d.5` / `7d.6` / `7g.0` verify stored run-config hashes
(features/unified/B2-model/b3-predictions) before restoring, so a crashed
runtime resumes from Drive without redoing completed phases. B4 writes
`b4_crash.log` with a full traceback on failure, its per-sample export is
vectorized (~5 s end-to-end locally, adversarial-tested), and the
CRASH-RECOVERY markdown cell documents the resume sequence. Colab runtime
recommendation: **L4 (22.5 GB VRAM / 54 GB RAM)** to avoid free-T4 crashes.
The old cells 8–12 remain optional Version A analyses and are outside the
B2–B4 resume path; skip them for the publication run unless those legacy
outputs are specifically required.
(9) **Credit-safe B3 extraction**: the external-feature extractor now runs in
chunks (2000 rows) and saves a PARTIAL cache + meta (`complete: false`) after
every chunk; if the runtime dies mid-extraction (the ~40 min GPU step), the
next run RESUMES from the partial cache instead of re-extracting from scratch.
`--skip-features` is accepted only for a complete hash-matched cache. Cell 7d.5
restores partial caches too (B3_CACHE_OK only when complete) and the B3 failure
branch checkpoints the partial cache to Drive. Regression tests cover crash →
resume → no-redundant-work paths (101 tests total).
(10) **Self-contained notebook (no zip upload)**: `colab/build_self_contained.py`
embeds all 29 runtime files (src/, colab/drive_cache.py, requirements-colab)
as base64 in cell 3; running it writes + hash-verifies the source tree in
Colab. The user uploads ONLY the single `.ipynb`; per-cell patches are done by
editing that file's EMBEDDED entry and rerunning cell 3.
(11) **Local-first restore**: restore cells (7.0/7b.0/7d.5/7d.6/7g.0) check
local VM artifacts first (kernel restarts keep VM files alive), then Drive —
B2/B3 results survive Colab quota kills without any Drive dependency; the B3
runner also self-heals from a complete local feature cache. Cell 7e uploads
the feature cache to Drive first (most valuable).

Environment: Colab Pro.

1. Compare raw XGBoost, Platt, and isotonic probabilities.
2. Define calibrator fitting and selection rules before reading the locked test results.
3. Report ECE, adaptive ECE if available, Brier, NLL, reliability diagrams, and calibration slope/intercept.
4. Report subgroup calibration for dataset, task, context length, answer length, and generator model where available.
5. Use minimum subgroup sizes and pooled fallback behavior; do not fit unstable tiny cluster calibrators.
6. Compare source calibration applied directly to external data with a separately labeled target-calibration experiment.

Exit condition: calibration results explain not only which method has the lowest ECE, but where calibration fails under shift.

### B5 — Explanation reliability and error analysis

Environment: Colab Pro for experiments; manual review by the team.

1. Compare mean absolute SHAP ranking with permutation importance and feature/group ablation impact.
2. Run top-feature deletion and neutralization tests and measure prediction change.
3. Run controlled perturbations: entity replacement, numeric/date replacement, support-sentence removal, irrelevant-sentence insertion, and limited paraphrases.
4. Measure top-k feature stability and score stability with confidence intervals.
5. Have two reviewers assess explanation plausibility on 30–50 cases and record disagreements.
6. Report failure cases where SHAP is unstable or inconsistent with the evidence.
7. Do not use fixed FAC/PSI pass thresholds without empirical justification.

Exit condition: the paper can defend SHAP as evaluated evidence rather than decorative visualization.

### B6 — Reproducible publication artifact

Environment: local machine plus Colab export.

1. Add `run_all_experiments.py` or an equivalent config-driven runner.
2. Save `artifact_manifest.json` with dataset hashes, split hash, code commit, package versions, model checkpoints, seeds, device, RAM, VRAM, and outputs.
3. Regenerate `colab/halurisc_src.zip` after every source change used by Colab.
4. Update the existing `colab/HaluRISC_Training.ipynb` cells rather than creating a second competing notebook.
5. The first notebook update will replace the data-preparation/training cells after the grouped-split repair; later cells will call corrected evaluation scripts and export the manifest.
6. Do not put raw datasets, restricted benchmark files, API keys, or model secrets in the repository.
7. Provide a CPU-compatible Docker path and retain CUDA as an optional local acceleration path.

Exit condition: a clean clone plus documented downloads can regenerate the required artifacts without hidden local paths.

### B7 — Research UI and demo

Environment: local RTX 3060 laptop, 32 GB RAM.

1. Keep the four routes: Chat, Analyze, Dashboard, and About/Method.
2. Add Analyze compare mode for two answers against the same evidence.
3. Show calibrated score, thresholds, warning, dataset provenance, model version, feature version, latency, device, and expandable seven-group features.
4. Add an offline/precomputed demo path so the method can be shown without an OpenAI key or network access.
5. Upgrade Dashboard into URL-addressable tabs: Overview, Robustness, Calibration, Explainability, Failures, and Efficiency.
6. Render only generated artifact data; remove hardcoded savings, metrics, versions, and fake borderline scores.
7. Label SHAP as raw-model feature attribution when the displayed score is calibrated.
8. Add manually reviewed failure-case browsing.
9. Add mobile navigation, keyboard focus, chart text alternatives, `aria-live` result states, reduced-motion handling, and non-color risk cues.
10. Avoid fake claim-level evidence highlighting because the current model produces feature-level signals, not token-level proof.
11. Keep one Uvicorn worker, no `--reload`, CUDA fp16, inference locking, bounded input lengths, and feature-result caching.

Exit condition: a presenter can explain the method, run a live example, show a real failure, open robustness/calibration evidence, and recover if the network or LLM API is unavailable.

### B8 — Manuscript and delivery

Environment: local machine and manual team work.

1. Freeze the artifact manifest before writing the Version B results section.
2. Write the paper around leakage control, cross-domain robustness, calibration under shift, explanation reliability, and deployment cost.
3. Include dataset licenses, limitations, source-group rules, and negative results.
4. Do not claim SOTA, universal truth detection, guaranteed Q2 acceptance, or an unverified first contribution.
5. Prepare a 5-minute demo: problem, grounded example, unsupported example, explanation, calibration/shift result, failure case, and efficiency.
6. Record a backup video and capture screenshots from the final commit.
7. Select the journal only after corrected results are available; recheck scope, quartile, APC, and author guidelines at submission time.

Exit condition: manuscript numbers, dashboard numbers, manifest, screenshots, and demo video all come from the same frozen commit and artifact bundle.

### Colab notebook update note

The existing `colab/HaluRISC_Training.ipynb` remains the notebook to use. It should not be replaced now. After the Version A code repair, update the relevant cells in this order: source upload, dependency install, grouped data preparation, feature extraction, training/evaluation, SHAP/error/latency evaluation, manifest creation, and artifact export. The exact cells and source files to change will be reported after the grouped-split implementation is complete.

---

## 15. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| NLI model download fails / too slow on CPU | Fallback `nli-MiniLM2-L6-H768`; cache all NLI outputs |
| Feature extraction too slow | Batch everything, extract once to Parquet |
| Class imbalance | `scale_pos_weight`; report PR-AUC alongside AUROC |
| Overfitting | 5-fold CV + early stopping + 3 seeds |
| HaluEval synthetic bias or benchmark shortcuts | Use grouped splits, shortcut controls, manual audit, and RAGTruth transfer evaluation |
| Dashboard scope creep | Static gallery first, live page second; polish last 2 days |
| Statistical tests confusing | Use `scipy`/`statsmodels` one-liners; put interpretations in paper |
| External dataset licensing | Do not bundle restricted files; ship download scripts, hashes, citations, and license notes |
| Laptop VRAM pressure | Run heavy experiments in Colab Pro; use one local API worker, fp16, caching, and no reload |

---

## 16. Resource Links

- HaluEval repo: `https://github.com/RUCAIBox/HaluEval` (raw: `.../main/data/qa_data.json`)
- RAGTruth official: `https://github.com/ParticleMedia/RAGTruth` · HF mirror: `https://huggingface.co/datasets/wandb/RAGTruth-processed`
- NLI: `https://huggingface.co/cross-encoder/nli-deberta-v3-base` · fallback `.../cross-encoder/nli-MiniLM2-L6-H768`
- Embeddings: `https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2`
- XGBoost: `https://pypi.org/project/xgboost/` (3.3.0, Jun 2026) · scikit-learn: `https://scikit-learn.org`
- FastAPI: `https://fastapi.tiangolo.com` · Pydantic v2: `https://docs.pydantic.dev`
- shadcn/ui Vite install: `https://ui.shadcn.com/docs/installation/vite` · Tailwind v4: `https://tailwindcss.com`
- spaCy: `https://spacy.io` · SHAP: `https://shap.readthedocs.io`
- Reference paper (verified): RAGTruth — Niu et al., ACL 2024, DOI 10.18653/v1/2024.acl-long.585
- HaluEval ACL paper: `https://aclanthology.org/2023.emnlp-main.397/`
- FaithBench ACL paper: `https://aclanthology.org/2025.naacl-short.38/`
- HalluLens ACL paper: `https://aclanthology.org/2025.acl-long.1176/`
- Cost-Effective Hallucination Detection: `https://arxiv.org/abs/2407.21424`
- PARALLAX benchmark-artifact preprint: `https://arxiv.org/html/2605.17028v1`
- TRIVIA+ benchmark-design preprint: `https://arxiv.org/abs/2605.11330v1`

