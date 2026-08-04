# HaluRISC — Detailed Implementation Roadmap

**Goal:** Version A course project — HaluEval QA hallucination-risk classifier with calibrated scores, SHAP explanations, a GPT 5.6 Luna-powered conversational AI interface (assistant-ui), and a show-winning web dashboard, in ~8 weeks. CPU-only ML + OpenAI API (existing credits) for chat and LLM-as-judge baseline.

**Convention:** items marked `[verified 2026]` were checked against current web/PyPI info in July 2026.

---

## 1. Tech Stack at a Glance

| Layer | Tool | Version (2026) | Why / Source |
|---|---|---|---|
| Language | Python | 3.12 or 3.13 | All libs support it; XGBoost 3.3.0 supports 3.12–3.14, scikit-learn 1.7 supports 3.10–3.13 `[verified 2026]` |
| Tabular ML | scikit-learn | 1.7.x `[verified]` | LR, RF, CV, tuning, calibration, metrics |
| Gradient boosting | XGBoost | 3.3.0 `[verified]` (PyPI, Jun 2026) | Final model; `XGBClassifier` scikit-learn API |
| NLI model | HuggingFace `cross-encoder/nli-deberta-v3-base` | primary `[verified]` | SNLI+MultiNLI, outputs contradiction/entailment/neutral |
| NLI fallback | `cross-encoder/nli-MiniLM2-L6` | if DeBERTa too slow | ~1/5 size of DeBERTa, ~90%+ of its accuracy |
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

> **Note:** this is the **target** layout to build toward, not the current state. As of now the repo contains only `blueprint.md`, `proposal.md`, `roadmap.md`, and `report/` (LaTeX proposal + PDF). The `src/`, `data/`, `web/`, and `artifacts/` trees are created during the phases below.

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

## 3. Phase 0 — Environment Setup (Day 1)

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

## 4. Phase 1 — Data Acquisition (Week 1)

### 4.1 HaluEval (primary, train/tune/test)

- **Where:** GitHub `RUCAIBox/HaluEval` `[verified]`
  - Repo: `https://github.com/RUCAIBox/HaluEval`
  - Direct raw file: `https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json`
- **What is inside `qa_data.json`:** 10,000 QA entries; each has `question`, `knowledge` (facts = context), `answer` (correct), `hallucinated_answer`, and `hallucination_label` (binary).
- **How to build the binary dataset:** for each entry, use `knowledge` as **context**, `answer` as the response, and `hallucination_label` as the label. If you want both variants per question (correct + hallucinated answer), create two rows per question and label accordingly — decide once and document it.
  - **⚠ Decide the row layout up front** — it changes dataset size and NLI cost: one row per question = **10,000 rows / 20K NLI pairs**; two rows per question = **20,000 rows / 40K NLI pairs** (~2–4 hrs CPU instead of 1–2). The proposal quotes "~10,000 samples", so the **two-rows layout means updating that number to ~20,000 everywhere** (proposal §Dataset, paper dataset table).
- **License:** the repo has **no LICENSE file** — use locally only, do not redistribute, and note this in the paper's dataset section.
- **Manual audit (mandatory):** randomly sample 50 entries, read each, and record whether the label looks right. Keep the audit notes file — it goes in the paper's dataset-limitations section.

### 4.2 RAGTruth (external validation)

- **Where:** official GitHub `ParticleMedia/RAGTruth` or ready-made HuggingFace mirror `wandb/RAGTruth-processed` `[verified]`.
- **What it is:** ~18,000 naturally generated RAG responses (GPT-3.5/4, LLaMA-2, Mistral) with **human word-level hallucination spans** (Niu et al., ACL 2024).
- **How to use here:** download the QA portion; derive a binary label (`has_hallucination = any annotated span`). Keep ~1,000–2,000 samples untouched — this is the zero-shot validation set for the final model.
- **Paper framing:** results on HaluEval = main results; RAGTruth = "does the model generalize to natural responses?" — exactly what the proposal promises.

---

## 5. Phase 2 — Preprocessing & Splits (Week 1–2)

Rules (encode in `src/data/prepare.py`, cache to `data/processed/qa_clean.parquet`):

- Normalize whitespace; tokenize with simple split (or spaCy).
- Keep original casing for NER/NLI; lowercase only for lexical features.
- Drop invalid rows (empty answer, corrupted text).
- Empty/missing context → lexical overlap = 0, NLI = neutral (0.33/0.33/0.33), document the rule.
- Check class balance. If not ~50/50, keep it (real-world) and set `scale_pos_weight` for XGBoost.
- **Split:** 70% train / 15% validation / 15% test, stratified by label.
- **Save split indices** with `np.save("artifacts/split_indices.npy")` — reproducibility requires exact splits, not just a seed.

---

## 6. Phase 3 — Feature Extraction (Week 2–3)

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
- Fallback if download/speed is a problem: `cross-encoder/nli-MiniLM2-L6` (much smaller, ~15ms/pair).

**Time each group** (lexical vs NLI vs semantic) — the latency breakdown feeds the paper's efficiency section.

---

## 7. Phase 4 — Modeling (Week 4)

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

## 8. Phase 5 — Calibration & Evaluation (Week 4–5)

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
- Efficiency/cost: measure per-sample latency (features + predict + SHAP) and compare against **GPT 5.6 Luna as LLM-as-judge** baseline. Luna pricing: $0.20/M input + $1.20/M output — estimated **~$1.10 total for 10,000 samples**. This is the paper's cost-argument table: HaluRISC at ~$0.001/prediction vs GPT judge at ~$0.10/prediction — **100x cheaper**.

---

## 9. Phase 6 — Explainability (Week 5)

- `shap.TreeExplainer(xgb)` → `shap_values` on a test subsample (500–1,000 rows for speed).
- **Global:** SHAP summary plot (beeswarm) + mean-|SHAP| bar chart.
- **Local:** waterfall plots for 3 hand-picked cases: clear hallucination, clearly correct, borderline.
- Save figures to `artifacts/figures/` **and** raw SHAP values to JSON — the dashboard will re-render them.
- Optional but strong: explanation reliability — remove top-SHAP feature and measure predicted-probability change vs SHAP magnitude (feature-ablation correlation).

---

## 10. Phase 7 — Backend API (Week 6)

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
- FastAPI: `uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload`
- Next.js: `npm run dev` on port 3000; `next.config.ts` rewrites `/api/ml/*` → FastAPI.
- OpenAI key stored in `web/.env.local` — never exposed to browser.

---

## 11. Phase 8 — Frontend Dashboard (Week 7) — show-winning, judge-facing

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

**Serving:**
```powershell
# Development (2 terminals)
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload  # FastAPI
npm run dev  # Next.js on http://localhost:3000

# Production
npm run build  # builds Next.js static + server
# Run both as separate processes or use Docker multi-stage build
```

---

## 12. Phase 9 — Integration, Demo & Delivery (Week 8)

- Build frontend, serve from FastAPI, run one-command demo.
- Pre-bake 3 demo examples: clearly hallucinated, clearly correct, borderline — plus live input.
- Rehearse a 5-minute script: 1 min problem → 1 min approach → 1 min live demo → 1 min results → 1 min "why it matters".
- Optional: two-stage Dockerfile (node build → python runtime) for portability.
- Final checks: `requirements.txt` pinned, README with setup/run commands, split indices + seeds saved.

---

## 13. Phase 10 — Reproducibility & Paper Mapping

| Paper section | Artifact |
|---|---|
| Dataset | `data/processed/*.parquet` + manual audit notes (the 50-sample audit from Phase 1) |
| Features | `src/features/` + feature table CSV |
| Experiments | `artifacts/results/*.csv` (all 3 seeds) |
| Calibration | ECE/Brier + reliability diagram PNG |
| Statistics | McNemar output + bootstrap CIs |
| Efficiency | latency breakdown table + cost estimate |
| Explainability | SHAP figures + JSON |
| Reproducibility | pinned requirements, saved splits, saved models |

**Before submission — verify every citation (Week 8):**
- Click through each DOI / URL in `report/proposal.tex` and the paper — the 2026 entries (`ijert`, `ieeeTai`, `multimedia`, `spikescore`) in particular must resolve. A broken DOI in a project about hallucination is a credibility hit.
- Confirm the two live claims behind the roadmap `[verified]` markers: XGBoost `3.3.0` and the `cross-encoder/nli-deberta-v3-base` model card still match before you install (PyPI/HF links in §15).

**Roadmap completion checklist (Week 8, before submission):**
- [ ] All `data/processed/*.parquet`, `artifacts/models/*`, `artifacts/results/*` produced and committed
- [ ] `requirements.txt` fully pinned (no `>=`, no un-pinned top-level packages)
- [ ] Split indices + seeds saved (not just a fixed seed)
- [ ] `report/out/proposal.pdf` and paper PDF built from source
- [ ] Every citation's DOI/URL verified live
- [ ] README setup/run commands verified by a clean clone test (or documented as pending if skipped)

---

## 14. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| NLI model download fails / too slow on CPU | Fallback `nli-MiniLM2-L6`; cache all NLI outputs |
| Feature extraction too slow | Batch everything, extract once to Parquet |
| Class imbalance | `scale_pos_weight`; report PR-AUC alongside AUROC |
| Overfitting | 5-fold CV + early stopping + 3 seeds |
| HaluEval has no license / synthetic bias | Use locally, document; RAGTruth covers generalization |
| Dashboard scope creep | Static gallery first, live page second; polish last 2 days |
| Statistical tests confusing | Use `scipy`/`statsmodels` one-liners; put interpretations in paper |

---

## 15. Resource Links

- HaluEval repo: `https://github.com/RUCAIBox/HaluEval` (raw: `.../main/data/qa_data.json`)
- RAGTruth official: `https://github.com/ParticleMedia/RAGTruth` · HF mirror: `https://huggingface.co/datasets/wandb/RAGTruth-processed`
- NLI: `https://huggingface.co/cross-encoder/nli-deberta-v3-base` · fallback `.../cross-encoder/nli-MiniLM2-L6`
- Embeddings: `https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2`
- XGBoost: `https://pypi.org/project/xgboost/` (3.3.0, Jun 2026) · scikit-learn: `https://scikit-learn.org`
- FastAPI: `https://fastapi.tiangolo.com` · Pydantic v2: `https://docs.pydantic.dev`
- shadcn/ui Vite install: `https://ui.shadcn.com/docs/installation/vite` · Tailwind v4: `https://tailwindcss.com`
- spaCy: `https://spacy.io` · SHAP: `https://shap.readthedocs.io`
- Reference paper (verified): RAGTruth — Niu et al., ACL 2024, DOI 10.18653/v1/2024.acl-long.585
